"""Opt-in small GPU tests; no model weights or running server required.

Run with SGLANG_TEST_V41_HISPARSE_GPU=1 and one visible CUDA/HIP device.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
from sglang.srt.managers.hisparse_v41_coordinator import DeepSeekV41HiSparseCoordinator
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV41HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def record(buffer, slot, page, layout):
    row, offset = divmod(int(slot), page)
    return torch.cat(
        (
            buffer[row, offset * layout.data_bytes : (offset + 1) * layout.data_bytes],
            buffer[
                row,
                page * layout.data_bytes
                + offset * layout.scale_bytes : page * layout.data_bytes
                + (offset + 1) * layout.scale_bytes,
            ],
        )
    )


def fill_record(buffer, slot, page, layout, value):
    row, offset = divmod(int(slot), page)
    buffer[row, offset * layout.data_bytes : (offset + 1) * layout.data_bytes] = value
    buffer[
        row,
        page * layout.data_bytes
        + offset * layout.scale_bytes : page * layout.data_bytes
        + (offset + 1) * layout.scale_bytes,
    ] = (
        value + 1
    )


def records(buffer, slots, page, layout):
    rows, offsets = slots // page, slots % page
    value_offsets = offsets[:, None] * layout.data_bytes + torch.arange(
        layout.data_bytes
    )
    scale_offsets = (
        page * layout.data_bytes
        + offsets[:, None] * layout.scale_bytes
        + torch.arange(layout.scale_bytes)
    )
    return torch.cat(
        (buffer[rows[:, None], value_offsets], buffer[rows[:, None], scale_offsets]),
        dim=1,
    )


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_V41_HISPARSE_GPU") == "1", "opt-in GPU test"
)
class TestV41HiSparseGPU(unittest.TestCase):
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

    def test_attention_metadata_prepares_raw_topk_before_indexer_skip(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend

        core = SimpleNamespace(low_ratios=(1, 2), page_table=torch.zeros(1, 4))
        pages = {ratio: torch.zeros((1, 128), dtype=torch.int32) for ratio in (1, 2)}
        core.sparse_page_indices = lambda ratio: pages[ratio]
        core.sparse_raw_indices = lambda ratio: getattr(
            core, f"c{ratio}_sparse_raw_indices", None
        )
        core.sparse_topk_lengths = lambda ratio: torch.tensor([128])
        core.page_size = 256
        instance = object.__new__(backend.DeepseekV4AttnBackend)
        instance.hisparse_coordinator = SimpleNamespace(is_dsv41=True)
        instance.forward_metadata = backend.DSV4Metadata(core, None)
        with (
            patch(
                "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
                return_value=True,
            ),
            patch.object(backend, "fill_all_compressed_indices") as fill,
        ):
            instance.init_forward_metadata_in_graph(SimpleNamespace(out_cache_loc=None))
        self.assertEqual(fill.call_count, 2)
        for call in fill.call_args_list:
            self.assertIsNotNone(call.kwargs["raw_indices"])
            self.assertTrue(torch.all(call.kwargs["raw_indices"] == -1))

    def test_idle_attention_dp_rank_does_not_attempt_swap(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        instance = object.__new__(DeepseekV4AttnBackend)
        instance.hisparse_coordinator = SimpleNamespace(is_dsv41=True)
        instance.mtp_enabled = False
        instance.trtllm_attn = False
        q = torch.empty((0, 2, 512))
        result = instance._forward_attention(
            q,
            None,
            None,
            SimpleNamespace(v_head_dim=512),
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
            1,
        )
        self.assertEqual(result.shape, q.shape)

    def test_paged_round_trip_all_layouts(self):
        for layout in (KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4):
            for page in (64, 128, 256):
                with self.subTest(layout=layout, page=page):
                    params = (
                        page,
                        layout.data_bytes,
                        layout.scale_bytes,
                        layout.page_bytes(page),
                    )
                    gpu = torch.zeros((4, params[-1]), dtype=torch.uint8, device="cuda")
                    host = DeepSeekV4PagedHostPool(
                        "roundtrip", [gpu], params[-1], 4, page
                    )
                    try:
                        source = torch.randint(
                            0, 255, host.kv_buffer[0].shape, dtype=torch.uint8
                        )
                        host.kv_buffer[0].copy_(source)
                        src = torch.tensor(
                            [0, page - 1, page, 2 * page + 3], device="cuda"
                        )
                        dst = torch.tensor(
                            [2 * page, page + 2, 3 * page - 1, 1], device="cuda"
                        )
                        transfer_cache_dsv4_mla(
                            host.data_ptrs,
                            host.device_ptrs,
                            src,
                            dst,
                            paged_layout=params,
                        )
                        torch.cuda.synchronize()
                        snapshot = gpu.cpu()
                        for a, b in zip(src.tolist(), dst.tolist()):
                            torch.testing.assert_close(
                                record(source, a, page, layout),
                                record(snapshot, b, page, layout),
                            )
                        host.kv_buffer[0].zero_()
                        transfer_cache_dsv4_mla(
                            host.device_ptrs,
                            host.data_ptrs,
                            dst,
                            src,
                            paged_layout=params,
                        )
                        torch.cuda.synchronize()
                        for a in src.tolist():
                            torch.testing.assert_close(
                                record(source, a, page, layout),
                                record(host.kv_buffer[0], a, page, layout),
                            )
                    finally:
                        torch.cuda.synchronize()
                        host.destroy()

    def test_pd_admission_swap_decode_backup_and_release(self):
        self._exercise_lifecycle(top_k=128, hot_size=256)

    def test_default_2048_topk_4096_hot_working_set(self):
        self._exercise_lifecycle(top_k=2048, hot_size=4096)

    def test_cuda_graph_c1_c2_swap_padding_and_slot_reuse(self):
        self._exercise_lifecycle(top_k=512, hot_size=1024, cuda_graph=True)

    def _exercise_lifecycle(self, top_k, hot_size, cuda_graph=False):
        full_size = max(4096, hot_size * (12 if cuda_graph else 8))
        allocated_size = hot_size * 6
        with tempfile.TemporaryDirectory() as tmp:
            torch.distributed.init_process_group(
                "gloo", init_method=f"file://{tmp}/rendezvous", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)
            pool = DeepSeekV4TokenToKVPool(
                max_num_reqs=2,
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
                device="cuda",
                enable_memory_saver=False,
                compression_ratios=[1, 2, 1, 2],
                kv_source_layers=[0, 1],
                full_size=full_size,
                enable_hisparse=True,
            )
            logical = PagedTokenToKVPoolAllocator(
                full_size, 256, pool.dtype, "cuda", pool, need_sort=False
            )
            allocator = DeepSeekV41HiSparseTokenToKVPoolAllocator(logical)
            req_pool = SimpleNamespace(
                req_to_token=torch.zeros(
                    (2, allocated_size), dtype=torch.int64, device="cuda"
                ),
                max_context_len=allocated_size,
            )
            coord = DeepSeekV41HiSparseCoordinator(
                req_pool, allocator, top_k, hot_size, "cuda", None
            )
            self.addCleanup(coord.destroy)
            if cuda_graph:
                # Capture before admission, with more padded rows than request
                # slots. Even invalid request IDs must not be dereferenced.
                coord.init_cuda_graph_state(4)
                ptrs = [
                    c.top_k_device_locs_buffer.data_ptr() for c in coord.pools.values()
                ]
                coord.init_cuda_graph_state(2)
                self.assertEqual(
                    ptrs,
                    [c.top_k_device_locs_buffer.data_ptr() for c in coord.pools.values()],
                )
                graph_reqs = torch.full((4,), 2**30, device="cuda", dtype=torch.int64)
                graph_lens = torch.ones(4, device="cuda", dtype=torch.int64)
                graph_topk = torch.zeros((4, top_k), device="cuda", dtype=torch.int32)
                graphs = {}
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    for layer in range(4):
                        coord.swap_in_selected_pages(
                            graph_reqs, graph_lens, graph_topk, layer
                        )
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()
                for layer in range(4):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        output = coord.swap_in_selected_pages(
                            graph_reqs, graph_lens, graph_topk, layer
                        )
                    graphs[layer] = (graph, output)
                    graph.replay()
                    self.assertTrue(torch.all(output == -1))

            peer = None
            swap_count = 0

            def swap(req_indices, lens, topk, layer):
                nonlocal swap_count
                if not cuda_graph:
                    return coord.swap_in_selected_pages(req_indices, lens, topk, layer)
                swap_count += 1
                real_bs = 2 if peer is not None and swap_count % 2 == 0 else 1
                graph_reqs[0].copy_(req_indices[0])
                graph_lens[0].copy_(lens[0])
                graph_topk[0].copy_(topk[0])
                ratio = pool.layer_mapping[layer].compress_ratio
                child = coord.pools[ratio]
                if real_bs == 2:
                    graph_reqs[1] = peer.kv.req_pool_idx
                    graph_lens[1] = peer.kv.kv_allocated_len
                    peer_n = peer.kv.kv_allocated_len // ratio
                    graph_topk[1] = torch.arange(top_k, device="cuda")
                    graph_topk[1, peer_n:] = -1
                graph, output = graphs[layer]
                coord.num_real_reqs.fill_(real_bs)
                graph.replay()
                result = output[:1].clone()
                self.assertTrue(torch.all(output[real_bs:] == -1))
                if real_bs == 2:
                    torch.testing.assert_close(
                        records(
                            child.mem_pool_device.kv_buffer[0].cpu(),
                            output[1, :peer_n].cpu().long(),
                            child.page_size,
                            child.mem_pool_device.kv_layout,
                        ),
                        records(
                            child.mem_pool_host.kv_buffer[0],
                            child.req_to_host_pool[peer.kv.req_pool_idx, :peer_n].cpu(),
                            child.page_size,
                            child.mem_pool_device.kv_layout,
                        ),
                    )
                coord.num_real_reqs.fill_(1)
                eager = coord.swap_in_selected_pages(req_indices, lens, topk, layer)
                torch.testing.assert_close(result, eager)
                # An idle DP rank must invalidate even a previously active row.
                coord.num_real_reqs.zero_()
                graph_reqs.fill_(2**30)
                graph.replay()
                self.assertTrue(torch.all(output == -1))
                coord.num_real_reqs.fill_(1)
                return result

            coord.num_real_reqs.fill_(1)
            before = {
                r: a.hisparse_attn_allocator.available_size()
                for r, a in allocator.ratio_allocators.items()
            }
            for iteration, length in enumerate((1, hot_size, hot_size * 5)):
                req_idx = iteration % 2
                req_indices = torch.tensor([req_idx], dtype=torch.int64, device="cuda")
                slots = logical.alloc(allocated_size)
                req_pool.req_to_token[req_idx, :allocated_size] = slots
                req = SimpleNamespace(
                    rid="test",
                    extend_end=length,
                    kv=SimpleNamespace(req_pool_idx=req_idx, kv_allocated_len=length),
                )
                for child in coord.pools.values():
                    for host in child.mem_pool_host.kv_buffer:
                        host.copy_(torch.randint(0, 255, host.shape, dtype=torch.uint8))
                coord.admit_request_direct(req)
                if cuda_graph:
                    peer_slots = logical.alloc(256)
                    req_pool.req_to_token[1 - req_idx, :256] = peer_slots
                    peer = SimpleNamespace(
                        rid="peer",
                        extend_end=255,
                        kv=SimpleNamespace(req_pool_idx=1 - req_idx, kv_allocated_len=255),
                    )
                    coord.admit_request_direct(peer)
                for layer in range(4):
                    ratio = pool.layer_mapping[layer].compress_ratio
                    child = coord.pools[ratio]
                    n = length // ratio
                    for start in (
                        0,
                        max(0, n - top_k),
                        min(top_k, max(0, n - top_k)),
                        min(2 * top_k, max(0, n - top_k)),
                        0,
                    ):
                        topk = torch.arange(
                            start, start + top_k, dtype=torch.int32, device="cuda"
                        )[None, :]
                        topk[topk >= n] = -1
                        out = swap(
                            req_indices,
                            torch.tensor([length], device="cuda"),
                            topk,
                            layer,
                        ).clone()
                        torch.cuda.synchronize()
                        data = child.mem_pool_device.kv_buffer[0].cpu()
                        positions, locations = topk[0].cpu().long(), out[0].cpu().long()
                        valid = positions >= 0
                        self.assertTrue(torch.all(locations[~valid] == -1))
                        host_locs = child.req_to_host_pool[req_idx].cpu()[
                            positions[valid]
                        ]
                        torch.testing.assert_close(
                            records(
                                data,
                                locations[valid],
                                child.page_size,
                                child.mem_pool_device.kv_layout,
                            ),
                            records(
                                child.mem_pool_host.kv_buffer[0],
                                host_locs,
                                child.page_size,
                                child.mem_pool_device.kv_layout,
                            ),
                        )
                for step in (1, 2, 3):
                    seq = length + step
                    req.kv.kv_allocated_len = seq
                    lens = torch.tensor([seq], device="cuda")
                    coord.map_last_loc_to_buffer(
                        lens,
                        slots[seq - 1 : seq],
                        req_indices,
                        lens.cpu(),
                        req_indices.cpu(),
                    )
                    coord.wait_for_pending_backup()
                    torch.cuda.synchronize()
                    for ratio, child in coord.pools.items():
                        if step > 1 and (seq - 1) % ratio == 0:
                            loc = slots[seq - 2].item() // ratio
                            got = record(
                                child.mem_pool_host.kv_buffer[0],
                                loc,
                                child.page_size,
                                child.mem_pool_device.kv_layout,
                            )
                            self.assertTrue(
                                torch.all(
                                    got[: child.mem_pool_device.kv_layout.data_bytes]
                                    == 20 + step - 1
                                )
                            )
                        if seq % ratio == 0:
                            loc = child.mem_pool_device.full_to_hisparse_device_index_mapping[
                                slots[seq - 1] // ratio
                            ].item()
                            self.assertGreater(loc, 0)
                            fill_record(
                                child.mem_pool_device.kv_buffer[0],
                                loc,
                                child.page_size,
                                child.mem_pool_device.kv_layout,
                                20 + step,
                            )
                    torch.cuda.synchronize()
                    if cuda_graph:
                        # Replay with growing C1/C2 lengths, including odd C2
                        # steps whose latest compressed record is unchanged.
                        for layer in range(4):
                            ratio = pool.layer_mapping[layer].compress_ratio
                            child = coord.pools[ratio]
                            n = seq // ratio
                            start = max(0, n - top_k)
                            topk = torch.arange(
                                start, start + top_k, device="cuda", dtype=torch.int32
                            )[None, :]
                            topk[topk >= n] = -1
                            out = swap(req_indices, lens, topk, layer)
                            newest = record(
                                child.mem_pool_device.kv_buffer[0].cpu(),
                                out[0, min(n, top_k) - 1].item(),
                                child.page_size,
                                child.mem_pool_device.kv_layout,
                            )
                            written_step = (seq // ratio) * ratio - length
                            if written_step > 0:
                                self.assertTrue(
                                    torch.all(
                                        newest[
                                            : child.mem_pool_device.kv_layout.data_bytes
                                        ] == 20 + written_step
                                    ),
                                    (length, seq, layer, ratio),
                                )
                coord.request_finished(req)
                coord.request_finished(req)
                allocator.free(slots)
                if peer is not None:
                    coord.request_finished(peer)
                    allocator.free(peer_slots)
                    peer = None
                self.assertEqual(logical.available_size(), full_size)
                for ratio, child in allocator.ratio_allocators.items():
                    self.assertEqual(
                        child.hisparse_attn_allocator.available_size(), before[ratio]
                    )
                    self.assertEqual(
                        child.full_to_hisparse_device_index_mapping[:-1]
                        .count_nonzero()
                        .item(),
                        0,
                    )
                    self.assertEqual(
                        child.full_to_hisparse_device_index_mapping[-1].item(), -1
                    )


if __name__ == "__main__":
    unittest.main()
