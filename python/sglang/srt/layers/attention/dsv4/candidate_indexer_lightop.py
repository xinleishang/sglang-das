"""The two-level low-ratio decode indexer on LightOp (HCU).

Same protocol and numerics contract as ``DeepGemmCandidateIndexer``:

* publish (candidate-source layer): dense paged logits, the layer's own plain
  top-k, and, on a side stream, the candidate block table -- the
  ``topk_blocks`` best blocks of 8 by block maximum, the newest block always
  kept, ascending, as pool slots / 8.
* select (consumer layers): logits of the published blocks only, then the
  top-k over the sparse row.

The sparse logits need no separate kernel: ``phys_blocks`` is a block table at
block size 8, so the dense LightOp kernel scores column ``j`` of the sparse row
at pool slot ``phys_blocks[b, j // 8] * 8 + j % 8``, and the plain paged top-k
maps its picks back through the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4 import lightop_indexer
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMetadata,
    IndexerInputs,
    get_candidate_stream,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    deep_gemm_fp4_paged_mqa_logits,
)


@dataclass
class LightopBlockTable(CandidateMetadata):
    # [rows, topk_blocks] int32: the kept blocks ascending, as pool slots / 8;
    # valid for the first min(topk_blocks, ceil(seq_len / 8)) entries of a row
    phys_blocks: torch.Tensor
    # [rows] int32: length of each sparse logits row (newest block may be partial)
    valid_lens: torch.Tensor
    # split-KV schedule of the sparse rows, None past the planner's row limit
    schedule: Optional[torch.Tensor]
    # Same kept blocks in request-local compressed coordinates, for HiSparse.
    logical_blocks: Optional[torch.Tensor] = None


def _int32_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    seq_lens = seq_lens.reshape(-1)
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    return seq_lens.contiguous()


class LightopCandidateIndexer:
    def __init__(self, topk_blocks: int, block_size: int):
        if block_size not in lightop_indexer.SUPPORTED_CANDIDATE_BLOCK_SIZES:
            raise ValueError(
                f"LightOp candidate block sizes are "
                f"{lightop_indexer.SUPPORTED_CANDIDATE_BLOCK_SIZES}, got {block_size}"
            )
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.alt_stream = get_candidate_stream()

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> LightopBlockTable:
        metadata = inputs.metadata
        seq_lens = _int32_lens(metadata.compressed_seq_lens)
        logits = deep_gemm_fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
            table_block_size=metadata.compressed_page_size,
            rows_per_request=metadata.rows_per_request,
        )
        main_stream = torch.cuda.current_stream()
        # The block-table chain reads logits and nothing else, so it forks off
        # the dense logits launch and runs beside the layer's own top-k.
        producer_ready = torch.cuda.Event()
        producer_ready.record(main_stream)
        logits.record_stream(self.alt_stream)
        self._dense_topk(logits, seq_lens, metadata, page_indices, raw_indices)
        with torch.cuda.stream(self.alt_stream):
            self.alt_stream.wait_event(producer_ready)
            table = self._block_table(
                logits, seq_lens, metadata, need_raw_indices=raw_indices is not None
            )
            # select_decode reads these on the main stream
            for t in (
                table.phys_blocks,
                table.valid_lens,
                table.schedule,
                table.logical_blocks,
            ):
                if t is not None:
                    t.record_stream(main_stream)
            join = torch.cuda.Event()
            join.record(self.alt_stream)
        # Close the fork here; one left open to the first consumer spans the
        # layers in between, which a replayed graph may put on the side stream.
        main_stream.wait_event(join)
        return table

    @staticmethod
    def _dense_topk(logits, seq_lens, metadata, page_indices, raw_indices) -> None:
        topk_transform_paged(
            logits,
            seq_lens,
            metadata.page_table,
            page_indices,
            metadata.compressed_page_size,
            raw_indices,
        )

    def _block_table(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        metadata,
        need_raw_indices: bool = False,
    ) -> LightopBlockTable:
        page_table = metadata.page_table
        if need_raw_indices:
            # Select once in logical coordinates, then map the same blocks to
            # pool slots. Compact sparse-row offsets are not HiSparse KV indices.
            assert metadata.compressed_page_size % self.block_size == 0
            page_table = torch.arange(
                page_table.shape[1], dtype=page_table.dtype, device=page_table.device
            ).repeat(page_table.shape[0], 1)
        phys_blocks, valid_lens = lightop_indexer.candidate_block_table(
            logits,
            seq_lens,
            page_table,
            metadata.compressed_page_size,
            self.topk_blocks,
            self.block_size,
        )
        logical_blocks = None
        if need_raw_indices:
            kept = (
                torch.arange(self.topk_blocks, device=logits.device)[None, :]
                * self.block_size
                < valid_lens[:, None]
            )
            logical_blocks = torch.where(kept, phys_blocks, -1)
            starts = logical_blocks.clamp_min(0).long() * self.block_size
            page_size = metadata.compressed_page_size
            slots = (
                metadata.page_table.gather(1, starts // page_size) * page_size
                + starts % page_size
            )
            phys_blocks = torch.where(kept, slots // self.block_size, -1).to(torch.int32)
        schedule = None
        if lightop_indexer.schedule_rows_supported(valid_lens.numel()):
            # One route per row: select_decode's TB=8 kernel never pairs
            # verify rows. The planner reads valid_lens on device, so a
            # captured graph replans on every replay.
            schedule = lightop_indexer.get_paged_mqa_logits_schedule(valid_lens)
        return LightopBlockTable(
            phys_blocks=phys_blocks,
            valid_lens=valid_lens,
            schedule=schedule,
            logical_blocks=logical_blocks,
        )

    def select_decode(
        self,
        candidate_metadata: LightopBlockTable,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        table = candidate_metadata
        assert isinstance(table, LightopBlockTable), type(table)
        if raw_indices is not None:
            assert table.logical_blocks is not None, (
                "HiSparse requires candidate blocks published with logical indices"
            )
        logits = lightop_indexer.paged_mqa_logits_fp4(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            table.valid_lens,
            table.phys_blocks,
            table.schedule,
            self.topk_blocks * self.block_size,
            table_block_size=self.block_size,
        )
        topk_transform_paged(
            logits,
            table.valid_lens,
            table.phys_blocks,
            page_indices,
            self.block_size,
            raw_indices,
        )
        if raw_indices is not None:
            compact = raw_indices.long()
            blocks = table.logical_blocks.gather(
                1, compact.clamp_min(0) // self.block_size
            )
            logical = blocks * self.block_size + compact % self.block_size
            raw_indices.copy_(torch.where(compact >= 0, logical, -1))
