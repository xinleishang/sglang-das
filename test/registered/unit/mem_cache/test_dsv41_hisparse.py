"""CPU regression tests for C1/C2 HiSparse ownership and PD addressing."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.hisparse import _paged_layout_flags
from sglang.srt.managers.hisparse_v41_coordinator import (
    DeepSeekV41HiSparseCoordinator,
    _LogicalPageHostMirror,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV41HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    HiSparseLowRatioDevicePool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestV41HiSparse(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                page_size=256,
                enable_hisparse=True,
                hisparse_config='{"host_to_device_ratio": 2}',
            ),
            role="tokenizer",
        )

    def make_pool(self):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=4,
            swa_size=1024,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=4,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[1, 2, 1, 2],
            kv_source_layers=[0, 1],
            full_size=4096,
            enable_hisparse=True,
        )

    def test_only_main_kv_shrinks_and_source_sharing_is_preserved(self):
        pool = self.make_pool()
        for ratio in (1, 2):
            self.assertIsInstance(pool.kv_pools[ratio], HiSparseLowRatioDevicePool)
            self.assertEqual(pool.kv_pools[ratio].size, 2048 // ratio)
            self.assertEqual(pool.kv_pools[ratio].page_size, 256 // ratio)
            self.assertEqual(pool.kv_pools[ratio].compress_ratio, ratio)
            self.assertGreaterEqual(pool.index_pools[ratio].size, 4096 // ratio)
            self.assertEqual(pool.kv_pools[ratio].layer_num, 1)
        self.assertEqual(pool.layer_mapping[2].compress_layer_id, 0)
        self.assertEqual(pool.layer_mapping[3].compress_layer_id, 0)

    def test_logical_free_does_not_double_free_hot_buffers(self):
        pool = self.make_pool()
        logical = PagedTokenToKVPoolAllocator(
            4096, 256, pool.dtype, "cpu", pool, need_sort=False
        )
        allocator = DeepSeekV41HiSparseTokenToKVPoolAllocator(logical)
        logical_slots = logical.alloc(256)
        for ratio, child in allocator.ratio_allocators.items():
            slots = child.hisparse_attn_allocator.alloc(256 // ratio)
            child.full_to_hisparse_device_index_mapping[
                logical_slots[ratio - 1 :: ratio] // ratio
            ] = slots
        before = {
            r: a.hisparse_attn_allocator.available_size()
            for r, a in allocator.ratio_allocators.items()
        }
        allocator.free(logical_slots)
        self.assertEqual(logical.available_size(), 4096)
        self.assertEqual(
            before,
            {
                r: a.hisparse_attn_allocator.available_size()
                for r, a in allocator.ratio_allocators.items()
            },
        )

    def test_admission_is_limited_by_both_hot_pools(self):
        allocator = object.__new__(DeepSeekV41HiSparseTokenToKVPoolAllocator)
        allocator.ratio_allocators = {
            1: SimpleNamespace(
                hisparse_page_size=256,
                hisparse_attn_allocator=SimpleNamespace(available_size=lambda: 1536),
            ),
            2: SimpleNamespace(
                hisparse_page_size=128,
                hisparse_attn_allocator=SimpleNamespace(available_size=lambda: 640),
            ),
        }
        self.assertEqual(allocator.available_request_buffers(512), 1)

    def test_host_mapping_uses_logical_pages_even_when_noncontiguous(self):
        logical = torch.cat((torch.arange(768, 1024), torch.arange(256, 512)))
        for ratio in (1, 2):
            mirror = object.__new__(_LogicalPageHostMirror)
            mirror.ratio = ratio
            mirror.req_to_token_pool = SimpleNamespace(req_to_token=logical[None, :])
            table = torch.full((1, 512), -1, dtype=torch.int64)
            allocated = torch.zeros(1, dtype=torch.int64)
            for start, length in ((0, 256 // ratio - 1), (256 // ratio - 1, 3)):
                got = mirror.alloc_paged_token_slots(table, allocated, 0, start, length)
                want = (
                    logical[
                        start * ratio + ratio - 1 : (start + length) * ratio : ratio
                    ]
                    // ratio
                )
                torch.testing.assert_close(got, want)
            self.assertEqual(int(allocated[0]), 256 // ratio + 2)

    def test_pd_replaces_kv_pointers_without_reordering_indexer(self):
        c1, index, c2, scales = [
            torch.empty(2, 32, dtype=torch.uint8) for _ in range(4)
        ]
        h1, h2 = [torch.empty(9, 32, dtype=torch.uint8) for _ in range(2)]
        coordinator = object.__new__(DeepSeekV41HiSparseCoordinator)
        coordinator.pools = {
            r: SimpleNamespace(
                mem_pool_device=SimpleNamespace(kv_buffer=[gpu]),
                mem_pool_host=SimpleNamespace(kv_buffer=[host]),
            )
            for r, gpu, host in ((1, c1, h1), (2, c2, h2))
        }
        coordinator.kvcache = SimpleNamespace(
            get_contiguous_buf_infos=lambda: (
                [x.data_ptr() for x in (c1, index, c2, scales)],
                [64] * 4,
                [32] * 4,
            )
        )
        ptrs, sizes, strides, kinds = coordinator.get_transfer_buf_infos()
        self.assertEqual(
            ptrs, [h1.data_ptr(), index.data_ptr(), h2.data_ptr(), scales.data_ptr()]
        )
        self.assertEqual(sizes, [288, 64, 288, 64])
        self.assertEqual(strides, [32] * 4)
        self.assertEqual(kinds, ["DRAM", "VRAM", "DRAM", "VRAM"])

    def test_shared_layer_reads_its_physical_source_cache(self):
        coordinator = object.__new__(DeepSeekV41HiSparseCoordinator)
        coordinator.reset_swap_cache()
        child = MagicMock()
        coordinator.pools = {2: child}
        coordinator.kvcache = SimpleNamespace(
            layer_mapping={7: SimpleNamespace(compress_ratio=2, compress_layer_id=1)}
        )
        reqs, lens, topk = (
            torch.tensor([0]),
            torch.tensor([513]),
            torch.tensor([[0, 255]]),
        )
        coordinator.swap_in_selected_pages(reqs, lens, topk, 7)
        args = child.swap_in_selected_pages.call_args.args
        torch.testing.assert_close(args[1], torch.tensor([256]))
        self.assertIs(args[2], topk)
        self.assertEqual(args[3], 1)

    def test_long_pd_tail_is_loaded_before_first_c2_nonwriting_step(self):
        coordinator = object.__new__(DeepSeekV41HiSparseCoordinator)
        child = MagicMock()
        child.host_token_len.return_value = 5
        child.device_buffer_size = 4
        child.mem_pool_device.layer_num = 2
        child.req_to_host_pool = torch.arange(8)[None, :]
        child.req_to_device_buffer = torch.arange(10, 16)[None, :]
        coordinator.pools = {2: child}
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0, kv_allocated_len=10))
        coordinator.admit_request_direct(req)
        self.assertEqual(child.mem_pool_host.load_to_device_per_layer.call_count, 2)
        args = child.mem_pool_host.load_to_device_per_layer.call_args.args
        torch.testing.assert_close(args[1], torch.tensor([4]))
        torch.testing.assert_close(args[2], torch.tensor([14]))

    def test_layout_flags_cover_v4_and_v41_pages(self):
        self.assertEqual(_paged_layout_flags(None), [])
        for layout in (KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4):
            for page in (64, 128, 256):
                flags = _paged_layout_flags(
                    (
                        page,
                        layout.data_bytes,
                        layout.scale_bytes,
                        layout.page_bytes(page),
                    )
                )
                self.assertIn(f"-DSGL_HISPARSE_PAGE_SIZE={page}", flags)
                self.assertIn(
                    f"-DSGL_HISPARSE_PAGE_BYTES={layout.page_bytes(page)}", flags
                )
        with self.assertRaises(ValueError):
            _paged_layout_flags((128, 256, 32, 128))

    def test_fused_writer_preserves_padding_and_incomplete_c2_state(self):
        for ratio in (1, 2):
            pool = object.__new__(HiSparseLowRatioDevicePool)
            pool.compress_ratio = ratio
            pool.full_to_hisparse_device_index_mapping = torch.arange(2048) + 100
            raw = torch.tensor([0, 256, 257, 512, 513])
            pos = torch.tensor([0, 0, 1, 2, 3])
            actual = pool.remap_fused_write_loc(raw, pos)
            want = raw.clone()
            for i in range(raw.numel()):
                if raw[i] > 0 and (pos[i] + 1) % ratio == 0:
                    want[i] = (raw[i] // ratio + 100) * ratio + ratio - 1
            torch.testing.assert_close(actual, want)

    def test_capacity_estimate_shrinks_main_kv_but_not_indexer(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        config = object.__new__(DSV4PoolConfigurator)
        config.kv_bytes = 584
        config.low_ratio_index_bytes = 132
        for shrink in (1, 2, 4):
            config.c4_shrink_factor = shrink
            for ratio in (1, 2):
                self.assertEqual(
                    config._compressed_bytes_per_full_token(ratio),
                    (584 / shrink + 132) / ratio,
                )

    def test_v41_config_validation_and_unsupported_combinations(self):
        from sglang.srt.arg_groups import hisparse_hook
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            enable_encoder_swa_bounded_replay=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.DISABLED),
                prefill=SimpleNamespace(backend=Backend.DISABLED),
            ),
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=False,
            disaggregation_decode_enable_radix_cache=False,
            kv_cache_dtype="fp8_e4m3",
            hisparse_config='{"device_buffer_size": 4096, "host_to_device_ratio": 2}',
            page_size=256,
        )
        with (
            patch.object(hisparse_hook, "resolving_view", return_value=args),
            patch.object(
                hisparse_hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(index_topk=2048), compress_ratios=[1, 2]
                ),
            ),
            patch.object(
                hisparse_hook,
                "get_platform",
                return_value=SimpleNamespace(is_cuda=True, is_hip=False),
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            hisparse_hook.validate_deepseek_v41_hisparse(args)
            with patch.object(args.cuda_graph_config.decode, "backend", Backend.FULL):
                hisparse_hook.validate_deepseek_v41_hisparse(args)
            for phase, backend in (
                ("decode", Backend.BREAKABLE),
                ("decode", Backend.TC_PIECEWISE),
                ("prefill", Backend.FULL),
            ):
                with (
                    self.subTest(phase=phase, backend=backend),
                    patch.object(
                        getattr(args.cuda_graph_config, phase), "backend", backend
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError, f"cuda-graph-backend-{phase}"
                    ):
                        hisparse_hook.validate_deepseek_v41_hisparse(args)
            for attr, value in (
                ("disaggregation_mode", "prefill"),
                ("disaggregation_transfer_backend", "nixl"),
                ("pp_size", 2),
                ("attn_cp_size", 8),
                ("dcp_size", 2),
                ("enable_prefill_cp", True),
                ("enable_encoder_swa_bounded_replay", True),
                ("enable_hierarchical_cache", True),
                ("enable_unified_cache_external_linker", True),
                ("disaggregation_decode_enable_radix_cache", True),
                ("kv_cache_dtype", "bfloat16"),
                ("hisparse_config", '{"host_to_device_ratio": 0.5}'),
                ("hisparse_config", '{"device_buffer_size": 4097}'),
            ):
                with (
                    self.subTest(attr=attr, value=value),
                    patch.object(args, attr, value),
                ):
                    with self.assertRaises(ValueError):
                        hisparse_hook.validate_deepseek_v41_hisparse(args)


class TestV41SwapReuse(unittest.TestCase):
    def setUp(self):
        self.coord = object.__new__(DeepSeekV41HiSparseCoordinator)
        self.coord.reset_swap_cache()
        self.coord.kvcache = SimpleNamespace(
            layer_mapping={
                layer: SimpleNamespace(
                    compress_ratio=2 if layer < 20 else 1,
                    compress_layer_id=(layer - 2) // 6 if layer < 20 else 0,
                )
                for layer in range(2, 40)
            }
        )
        self.coord.pools = {ratio: MagicMock() for ratio in (1, 2)}
        for child in self.coord.pools.values():
            child.swap_in_selected_pages.side_effect = (
                lambda reqs, lens, topk, source: topk.clone()
            )
            child.top_k_device_locs_buffer = torch.zeros((2, 4), dtype=torch.int32)
        self.reqs = torch.tensor([0, 1])
        self.lens = torch.tensor([16384, 32768])
        self.topk = {ratio: torch.tensor([[0, 1], [2, 3]]) for ratio in (1, 2)}

    def swap(self, layer=20, *, reuse_topk=True):
        ratio = self.coord.kvcache.layer_mapping[layer].compress_ratio
        return self.coord.swap_in_selected_pages(
            self.reqs, self.lens, self.topk[ratio], layer, reuse_topk=reuse_topk
        )

    def test_model_groups_reduce_38_swaps_to_8(self):
        index_sources = {2, 8, 14, 20, 24, 28, 32, 36}
        previous = None
        for layer in range(2, 40):
            if layer in index_sources:
                ratio = self.coord.kvcache.layer_mapping[layer].compress_ratio
                self.coord.reset_swap_cache(ratio)
                self.topk[ratio].add_(1)
            result = self.swap(layer)
            if layer not in index_sources:
                self.assertIs(result, previous)
            previous = result
        self.assertEqual(self.coord.pools[2].swap_in_selected_pages.call_count, 3)
        self.assertEqual(self.coord.pools[1].swap_in_selected_pages.call_count, 5)

    def test_indexer_rewrite_invalidates_same_tensor_and_kv_source(self):
        first = self.swap(20)
        self.assertIs(self.swap(21), first)
        self.coord.reset_swap_cache(1)
        self.topk[1].add_(100)
        second = self.swap(24)
        self.assertIsNot(first, second)
        torch.testing.assert_close(second, first + 100)
        self.assertIs(self.swap(25), second)

    def test_different_physical_sources_cannot_share_output_table(self):
        first = self.swap(2)
        self.assertIs(self.swap(3), first)
        second = self.swap(8)
        self.assertIsNot(first, second)
        self.assertIs(self.swap(9), second)
        # C2's one output table was overwritten by source 1.
        self.assertIsNot(self.swap(2), first)
        calls = self.coord.pools[2].swap_in_selected_pages.call_args_list
        self.assertEqual([call.args[3] for call in calls], [0, 1, 0])

    def test_ratios_have_independent_results_and_invalidation(self):
        c1, c2 = self.swap(20), self.swap(2)
        self.assertIs(self.swap(21), c1)
        self.assertIs(self.swap(3), c2)
        self.coord.reset_swap_cache(1)
        self.assertIs(self.swap(4), c2)
        self.assertIsNot(self.swap(22), c1)

    def test_new_batch_or_topk_object_requires_fresh_swap(self):
        previous = self.swap()
        self.reqs = self.reqs.flip(0)
        current = self.swap()
        self.assertIsNot(current, previous)
        previous, self.lens = current, self.lens + 1
        current = self.swap()
        self.assertIsNot(current, previous)
        previous, self.topk[1] = current, self.topk[1].clone()
        self.assertIsNot(self.swap(), previous)
        self.assertEqual(self.coord.pools[1].swap_in_selected_pages.call_count, 4)

    def test_forward_reset_handles_in_place_replay_inputs(self):
        previous = self.swap()
        self.reqs.copy_(self.reqs.flip(0))
        self.lens.add_(1)
        self.topk[1].add_(20)
        self.coord.reset_swap_cache()
        current = self.swap()
        self.assertIsNot(current, previous)
        torch.testing.assert_close(current, previous + 20)

    def test_uncached_call_invalidates_previously_cached_output(self):
        first = self.swap()
        direct = self.swap(reuse_topk=False)
        self.assertIsNot(direct, first)
        second = self.swap()
        self.assertIsNot(second, first)
        self.assertIs(self.swap(21), second)
        self.assertEqual(self.coord.pools[1].swap_in_selected_pages.call_count, 3)

    def test_request_lifecycle_and_graph_resize_clear_reuse(self):
        operations = (
            lambda: self.coord.request_finished(SimpleNamespace()),
            lambda: self.coord.map_last_loc_to_buffer(),
            lambda: self.coord.init_cuda_graph_state(4),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                previous = self.swap()
                operation()
                self.assertIsNot(self.swap(), previous)


if __name__ == "__main__":
    unittest.main()
