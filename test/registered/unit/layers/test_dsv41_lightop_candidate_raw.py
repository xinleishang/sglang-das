import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_torch
from sglang.srt.layers.attention.dsv4 import candidate_indexer_lightop as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def reference_block_table(logits, lens, pages, page_size, topk_blocks, block_size):
    blocks = torch.full(
        (logits.shape[0], topk_blocks),
        torch.iinfo(torch.int32).max,
        dtype=torch.int32,
        device=logits.device,
    )
    valid_lens = torch.zeros_like(lens)
    for row, length in enumerate(lens.tolist()):
        count = (length + block_size - 1) // block_size
        if not count:
            continue
        scores = (
            torch.nn.functional.pad(
                logits[row, :length], (0, count * block_size - length), value=-torch.inf
            )
            .view(count, block_size)
            .amax(-1)
        )
        scores[-1] = torch.inf
        kept = scores.topk(min(count, topk_blocks)).indices.sort().values
        starts = kept * block_size
        slots = pages[row, starts // page_size] * page_size + starts % page_size
        blocks[row, : kept.numel()] = (slots // block_size).to(torch.int32)
        valid_lens[row] = (
            (kept.numel() - 1) * block_size + (length - 1) % block_size + 1
        )
    return blocks, valid_lens


class TestLightopCandidateRaw(unittest.TestCase):
    def _exercise(self, page_size, *, device="cpu", native=False, topk=16):
        topk_blocks = 2048 if native else 4
        width = 32768 if native else 256
        full_lengths = torch.tensor([0, 1, 9, 63, 253, width - 3], device=device)
        for ratio in (1, 2):
            with self.subTest(ratio=ratio, page_size=page_size, device=device):
                lens = (full_lengths // ratio).to(torch.int32)
                rows, page_count = lens.numel(), width // page_size
                # Reverse the allocation order and leave gaps between physical
                # pages: neither compact offsets nor pool slots are logical IDs.
                pages = (
                    torch.arange(page_count - 1, -1, -1, device=device) * 2 + 3
                ).to(torch.int32).repeat(rows, 1)
                metadata = SimpleNamespace(
                    page_table=pages, compressed_page_size=page_size
                )
                source_logits = (
                    torch.arange(width, device=device).float().repeat(rows, 1)
                )
                indexer = object.__new__(mod.LightopCandidateIndexer)
                indexer.topk_blocks, indexer.block_size = topk_blocks, 8

                def block_table(*args):
                    return reference_block_table(*args)

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            mod.lightop_indexer,
                            "schedule_rows_supported",
                            return_value=False,
                        )
                    )
                    if not native:
                        stack.enter_context(
                            patch.object(
                                mod.lightop_indexer, "candidate_block_table", block_table
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                mod, "topk_transform_paged", topk_transform_paged_torch
                            )
                        )
                    original = indexer._block_table(source_logits, lens, metadata)
                    table = indexer._block_table(
                        source_logits, lens, metadata, need_raw_indices=True
                    )
                    valid_blocks = (
                        torch.arange(topk_blocks, device=device)[None, :] * 8
                        < table.valid_lens[:, None]
                    )
                    torch.testing.assert_close(table.valid_lens, original.valid_lens)
                    torch.testing.assert_close(
                        table.phys_blocks[valid_blocks],
                        original.phys_blocks[valid_blocks],
                    )
                    self.assertTrue(
                        torch.all(table.logical_blocks[~valid_blocks] == -1)
                    )
                    self.assertIsNone(original.logical_blocks)

                    # Mock only scoring. Native GPU tests still execute the real
                    # LightOp block selector and top-k with both output tensors.
                    columns = torch.arange(topk_blocks * 8, device=device)[None, :]
                    # Exactly k preferred values define an unambiguous set,
                    # independent of the kernel's ordering of equal scores.
                    sparse_logits = (
                        (columns >= (table.valid_lens[:, None] - topk))
                        & (columns < table.valid_lens[:, None])
                    ).float()
                    inputs = SimpleNamespace(
                        q_fp4=None, q_sf=None, k_cache=None, weights=None
                    )
                    physical = torch.full(
                        (rows, topk), -7, device=device, dtype=torch.int32
                    )
                    logical = torch.full_like(physical, -7)
                    without_raw = torch.full_like(physical, -7)
                    stack.enter_context(
                        patch.object(
                            mod.lightop_indexer,
                            "paged_mqa_logits_fp4",
                            return_value=sparse_logits,
                        )
                    )
                    indexer.select_decode(original, inputs, without_raw)
                    indexer.select_decode(table, inputs, physical, logical)
                    torch.testing.assert_close(
                        physical.sort(dim=1).values, without_raw.sort(dim=1).values
                    )
                    torch.testing.assert_close(physical == -1, logical == -1)
                    for row, length in enumerate(lens.tolist()):
                        raw = logical[row, logical[row] >= 0].long()
                        self.assertEqual(
                            raw.numel(), min(topk, int(table.valid_lens[row]))
                        )
                        self.assertTrue(torch.all(raw < length))
                        expected = (
                            pages[row].long()[raw // page_size] * page_size
                            + raw % page_size
                        )
                        torch.testing.assert_close(
                            physical[row, physical[row] >= 0].long(), expected
                        )
                    # The long row actually drops blocks, so returning compact
                    # sparse-row offsets instead of logical positions must fail.
                    if int(lens[-1]) > topk_blocks * 8:
                        self.assertGreater(int(logical[-1].max()), topk_blocks * 8)

    def test_noncontiguous_pages_and_c1_c2_lengths(self):
        for page_size in (16, 64):
            self._exercise(page_size)

    def test_raw_requires_logical_publication(self):
        table = mod.LightopBlockTable(
            phys_blocks=torch.zeros(1, 4, dtype=torch.int32),
            valid_lens=torch.ones(1, dtype=torch.int32),
            schedule=None,
        )
        indexer = object.__new__(mod.LightopCandidateIndexer)
        with self.assertRaisesRegex(AssertionError, "published with logical indices"):
            indexer.select_decode(table, None, torch.empty(1, 16), torch.empty(1, 16))

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_LIGHTOP_CANDIDATE_GPU") == "1",
        "explicit HCU GPU test",
    )
    def test_native_candidate_and_topk_mapping(self):
        self._exercise(64, device="cuda", native=True, topk=512)
        self._exercise(64, device="cuda", native=True, topk=1024)

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_LIGHTOP_CANDIDATE_GPU") == "1",
        "explicit HCU GPU test",
    )
    def test_native_publish_and_select(self):
        self._exercise_native_publish_and_select()

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_LIGHTOP_CANDIDATE_GPU") == "1",
        "explicit HCU GPU test",
    )
    def test_native_publish_and_select_cuda_graph(self):
        self._exercise_native_publish_and_select(cuda_graph=True)

    def _exercise_native_publish_and_select(self, cuda_graph=False):
        device, rows, width, page_size = "cuda", 4, 32768, 64
        page_count = width // page_size
        pages = (
            torch.arange(page_count - 1, -1, -1, device=device) * 2 + 3
        ).to(torch.int32).repeat(rows, 1)
        cache = torch.full(
            (page_count * 2 + 4, page_size, 1, 68),
            0x11,
            device=device,
            dtype=torch.uint8,
        )
        cache[..., 64:] = 127
        q = torch.full((rows, 1, 64, 64), 0x11, device=device, dtype=torch.int8)
        scales = torch.full(
            (rows, 1, 64), 0x7F7F7F7F, device=device, dtype=torch.int32
        )
        weights = torch.full((rows, 64), 1 / 64, device=device)
        indexer = mod.LightopCandidateIndexer(topk_blocks=2048, block_size=8)
        for ratio in (1, 2):
            lens = torch.tensor([0, 1, 9, width - 3], device=device, dtype=torch.int32)
            lens = lens // ratio
            metadata = SimpleNamespace(
                compressed_seq_lens=lens,
                page_table=pages,
                compressed_page_size=page_size,
                deep_gemm_metadata=mod.lightop_indexer.get_paged_mqa_logits_schedule(lens),
                max_compressed_seq_len=width,
                rows_per_request=1,
            )
            inputs = mod.IndexerInputs(q, scales, cache, weights, metadata)
            physical = torch.full((rows, 512), -7, device=device, dtype=torch.int32)
            raw = torch.full_like(physical, -7)

            def run():
                metadata.deep_gemm_metadata = (
                    mod.lightop_indexer.get_paged_mqa_logits_schedule(lens)
                )
                table = indexer.publish_decode(inputs, physical, raw)
                published = physical.clone(), raw.clone()
                indexer.select_decode(table, inputs, physical, raw)
                return (published, (physical, raw))

            if cuda_graph:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = run()
            for lengths in (
                [0, 1, 9, width - 3],
                [1025, 513, 0, 16387],
                [7, 0, 512, width - 1],
            ):
                lens.copy_(
                    torch.tensor(lengths, device=device, dtype=torch.int32) // ratio
                )
                if cuda_graph:
                    graph.replay()
                else:
                    outputs = run()
                for phase, (phys_result, raw_result) in enumerate(outputs):
                    with self.subTest(ratio=ratio, phase=phase, lengths=lengths):
                        torch.testing.assert_close(phys_result == -1, raw_result == -1)
                        self._assert_raw_mapping(
                            phys_result, raw_result, lens, pages, page_size
                        )

    def _assert_raw_mapping(self, physical, raw, lens, pages, page_size):
        for row, length in enumerate(lens.tolist()):
            logical = raw[row, raw[row] >= 0].long()
            self.assertEqual(logical.numel(), min(512, length))
            self.assertTrue(torch.all(logical < length))
            expected = (
                pages[row].long()[logical // page_size] * page_size
                + logical % page_size
            )
            torch.testing.assert_close(
                physical[row, physical[row] >= 0].long(), expected
            )


if __name__ == "__main__":
    unittest.main()
