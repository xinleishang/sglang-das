"""Decode-only C1/C2 HiSparse, using V4's per-pool cache management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    HiSparseTokenStats,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

logger = logging.getLogger(__name__)


@dataclass
class _SwapResult:
    source_layer: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    topk: torch.Tensor
    locations: torch.Tensor
    positions: torch.Tensor | None = None


class _LogicalPageHostMirror(DeepSeekV4PagedHostPool):
    """Host page IDs equal FULL logical page IDs, for every C1/C2 family.

    The logical allocator owns these pages. This avoids independent host page
    allocations changing the C1/C2/indexer destinations in the PD wire protocol.
    """

    def __init__(self, allocator, req_to_token_pool):
        pool = allocator.hisparse_kvcache
        self.ratio = allocator.compress_ratio
        self.logical_allocator = allocator.logical_attn_allocator
        self.req_to_token_pool = req_to_token_pool
        self.hisparse_layout = (
            pool.page_size,
            pool.kv_layout.data_bytes,
            pool.kv_layout.scale_bytes,
            pool.bytes_per_page_padded,
        )
        super().__init__(
            pool_name=f"dsv41_hisparse_c{self.ratio}",
            device_buffers=pool.kv_buffer,
            item_bytes=pool.bytes_per_page_padded,
            # Logical page zero is reserved, but still has a physical host row.
            num_host_pages=allocator.size_full // allocator.page_size + 1,
            slot_page_size=pool.page_size,
        )

    def alloc_paged_token_slots(
        self, req_to_host, allocated_lens, req_idx, start_pos, num_tokens
    ):
        end = start_pos + num_tokens
        full = self.req_to_token_pool.req_to_token[
            req_idx,
            start_pos * self.ratio + self.ratio - 1 : end * self.ratio : self.ratio,
        ]
        slots = full.to(torch.int64) // self.ratio
        if slots.numel() != num_tokens:
            raise ValueError("V4.1 HiSparse host mapping exceeds request capacity")
        req_to_host[req_idx, start_pos:end] = slots
        allocated_lens[req_idx] = max(int(allocated_lens[req_idx]), end)
        return slots

    def free(self, indices):
        # The request's logical FULL pages are freed by the normal allocator.
        pass

    def available_size(self):
        return self.logical_allocator.full_available_size() // self.ratio

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend="kernel"
    ):
        transfer_cache_dsv4_mla(
            self.data_ptrs[layer_id : layer_id + 1],
            device_pool.data_ptrs[layer_id : layer_id + 1],
            host_indices,
            device_indices,
            paged_layout=self.hisparse_layout,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        transfer_cache_dsv4_mla(
            device_pool.data_ptrs,
            self.data_ptrs,
            device_indices,
            host_indices,
            paged_layout=self.hisparse_layout,
        )


class DeepSeekV41HiSparseCoordinator:
    """Coordinate independent C1/C2 caches with shared logical FULL ownership."""

    is_dsv41 = True

    def __init__(
        self,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        top_k,
        device_buffer_size,
        device,
        tp_group,
        host_to_device_ratio=2,
        swap_in_block_size=960,
        shared_index_layers=None,
        num_draft_tokens=0,
    ):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.kvcache = token_to_kv_pool_allocator.get_kvcache()
        self.device_buffer_size = device_buffer_size
        self.ack_staging_queue = []
        self.spec_pools = {}
        self.num_draft_tokens = num_draft_tokens
        self.reset_swap_cache()
        if device_buffer_size < top_k or device_buffer_size >= 32768:
            raise ValueError(
                "V4.1 HiSparse requires top_k <= device_buffer_size < 32768"
            )
        if device_buffer_size % token_to_kv_pool_allocator.page_size:
            raise ValueError(
                "V4.1 HiSparse device_buffer_size must align to the FULL page size"
            )
        if num_draft_tokens:
            token_to_kv_pool_allocator.spec_scratch_size = (
                (num_draft_tokens * top_k + token_to_kv_pool_allocator.page_size - 1)
                // token_to_kv_pool_allocator.page_size
                * token_to_kv_pool_allocator.page_size
            )
        if token_to_kv_pool_allocator.available_request_buffers(device_buffer_size) < 1:
            raise ValueError(
                "V4.1 HiSparse GPU pools cannot hold one request's hot buffer plus "
                "its reserved page and verify scratch. Reduce device_buffer_size or host_to_device_ratio."
            )
        self.pools = {}
        try:
            for ratio, allocator in token_to_kv_pool_allocator.ratio_allocators.items():
                host = _LogicalPageHostMirror(allocator, req_to_token_pool)
                try:
                    child = HiSparseCoordinator(
                        req_to_token_pool,
                        allocator,
                        top_k,
                        device_buffer_size,
                        device,
                        tp_group,
                        host_to_device_ratio,
                        swap_in_block_size,
                        # KV-source sharing is not the DSA shared-index prefetch protocol.
                        shared_index_layers=None,
                        host_pool=host,
                    )
                except Exception:
                    host.destroy()
                    raise
                self.pools[ratio] = child
                if num_draft_tokens:
                    from sglang.srt.managers.hisparse_v41_spec import V41SpeculativePool

                    self.spec_pools[ratio] = V41SpeculativePool(child, num_draft_tokens)
        except Exception:
            self.destroy()
            raise
        self.num_real_reqs = next(iter(self.pools.values())).num_real_reqs
        for child in self.pools.values():
            child.num_real_reqs = self.num_real_reqs
        if self.spec_pools:
            metadata_bytes = sum(
                tensor.numel() * tensor.element_size()
                for pool in self.spec_pools.values()
                for state in pool.states
                for tensor in state
            )
            logger.info(
                "V4.1 HiSparse DSpark: verify_width=%d, top_k=%d, "
                "scratch_slots_per_request=%s, metadata=%.1f MiB, available_requests=%d",
                num_draft_tokens,
                top_k,
                {r: p.capacity for r, p in self.spec_pools.items()},
                metadata_bytes / 2**20,
                token_to_kv_pool_allocator.available_request_buffers(
                    device_buffer_size
                ),
            )

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self.reset_swap_cache()
        for pool in self.spec_pools.values():
            pool.resize_output(max_bs)
        # Output rows follow the padded graph batch, not the request-slot count.
        # Request-owned hot/host tables remain unchanged; num_real_reqs masks
        # padding before the swap kernel accesses any of those tables.
        for child in self.pools.values():
            output = child.top_k_device_locs_buffer
            if output.shape[0] < max_bs:
                child.top_k_device_locs_buffer = output.new_full(
                    (max_bs, output.shape[1]), -1
                )

    def get_transfer_buf_infos(self):
        """Preserve PD region order/strides; replace only main-KV addresses."""
        replacements = {}
        for child in self.pools.values():
            for gpu, host in zip(
                child.mem_pool_device.kv_buffer, child.mem_pool_host.kv_buffer
            ):
                replacements[gpu.data_ptr()] = host
        ptrs, sizes, strides = self.kvcache.get_contiguous_buf_infos()
        kinds = []
        for i, ptr in enumerate(ptrs):
            host = replacements.get(ptr)
            if host is None:
                kinds.append("VRAM")
            else:
                if strides[i] != host.shape[1] * host.element_size():
                    raise ValueError(
                        "V4.1 HiSparse PD host/device page stride mismatch"
                    )
                ptrs[i], sizes[i] = host.data_ptr(), host.nbytes
                kinds.append("DRAM")
        return ptrs, sizes, strides, kinds

    def admit_request_direct(self, req):
        self.reset_swap_cache()
        try:
            for child in self.pools.values():
                n = child.host_token_len(req.kv.kv_allocated_len)
                child.mem_pool_host.alloc_paged_token_slots(
                    child.req_to_host_pool,
                    child.req_to_host_pool_allocated_len,
                    req.kv.req_pool_idx,
                    0,
                    n,
                )
                child.admit_request_direct(req)
                if child.compress_ratio in self.spec_pools:
                    self.spec_pools[child.compress_ratio].admit(req)
                # On an odd C2 step no new compressed record is written. The
                # swap kernel nevertheless reads its newest record from the
                # reserved slot, which must already contain the PD tail.
                if n > child.device_buffer_size:
                    host_locs = child.req_to_host_pool[req.kv.req_pool_idx, n - 1 : n]
                    device_locs = child.req_to_device_buffer[
                        req.kv.req_pool_idx,
                        child.device_buffer_size : child.device_buffer_size + 1,
                    ]
                    for layer in range(child.mem_pool_device.layer_num):
                        child.mem_pool_host.load_to_device_per_layer(
                            child.mem_pool_device, host_locs, device_locs, layer
                        )
        except Exception:
            self.request_finished(req)
            raise

    def map_last_loc_to_buffer(self, *args, **kwargs):
        if self.num_draft_tokens:
            raise RuntimeError(
                "DSpark HiSparse must use prepare_spec_verify, not single-token mapping"
            )
        self.reset_swap_cache()
        for child in self.pools.values():
            child.map_last_loc_to_buffer(*args, **kwargs)

    def reset_swap_cache(self, compress_ratio: int | None = None) -> None:
        """Invalidate at forward boundaries and before an indexer rewrites Top-K."""
        if compress_ratio is None:
            self._swap_cache: dict[int, _SwapResult] = {}
        else:
            self._swap_cache.pop(compress_ratio, None)

    def swap_in_selected_pages(
        self,
        req_pool_indices,
        seq_lens,
        topk,
        layer_id,
        *,
        reuse_topk: bool = False,
        positions=None,
    ):
        """Reuse only within a forward whose indexer updates invalidate the cache.

        During graph capture this omits redundant swaps; every replay still
        executes the captured producer swap before its consumers' attention.
        """
        item = self.kvcache.layer_mapping[layer_id]
        child = self.pools[item.compress_ratio]
        if topk is None:
            raise RuntimeError("V4.1 HiSparse requires logical top-k indices")
        cached = self._swap_cache.get(item.compress_ratio)
        if (
            reuse_topk
            and cached is not None
            and cached.source_layer == item.compress_layer_id
            and cached.req_pool_indices is req_pool_indices
            and cached.seq_lens is seq_lens
            and cached.topk is topk
            and cached.positions is positions
        ):
            return cached.locations

        # Each ratio has one output table, even when it has multiple KV sources.
        # Any swap into it invalidates the previous source's cached result.
        self.reset_swap_cache(item.compress_ratio)
        if positions is not None:
            locations = self.spec_pools[item.compress_ratio].swap(
                req_pool_indices, positions, topk, item.compress_layer_id
            )
        else:
            locations = child.swap_in_selected_pages(
                req_pool_indices,
                seq_lens // item.compress_ratio,
                topk,
                item.compress_layer_id,
            )
        if reuse_topk:
            self._swap_cache[item.compress_ratio] = _SwapResult(
                item.compress_layer_id,
                req_pool_indices,
                seq_lens,
                topk,
                locations,
                positions,
            )
        return locations

    def prepare_spec_verify(self, reqs, prefix_lens, verify_locs):
        if not self.spec_pools:
            raise RuntimeError("HiSparse coordinator was not initialized for DSpark")
        if verify_locs.numel() != reqs.numel() * self.num_draft_tokens:
            raise ValueError("HiSparse DSpark verify window has an unexpected width")
        self.reset_swap_cache()
        self.wait_for_pending_backup()
        for pool in self.spec_pools.values():
            pool.prepare(reqs, prefix_lens, verify_locs)

    def commit_spec_verify(self, reqs, prefix_lens, verify_locs, commit_lens):
        # Same producer stream as target verify; finish copies before publication
        # allows another step or request reclamation to reuse the write page.
        for pool in self.spec_pools.values():
            pool.commit(reqs, prefix_lens, verify_locs, commit_lens)
        self.reset_swap_cache()

    def set_decode_producer_stream(self, stream):
        for child in self.pools.values():
            child.set_decode_producer_stream(stream)

    def wait_for_pending_backup(self):
        for child in self.pools.values():
            child.wait_for_pending_backup()

    def request_finished(self, req):
        self.reset_swap_cache()
        for child in self.pools.values():
            child.request_finished(req)
            if child.compress_ratio in self.spec_pools:
                self.spec_pools[child.compress_ratio].release(req)

    def retract_req(self, req):
        self.request_finished(req)

    def abort_staging_request(self, req):
        self.request_finished(req)

    def admit_request_into_staging(self, req):
        raise RuntimeError("V4.1 HiSparse supports PD direct-to-host decode only")

    def has_ongoing_staging(self):
        return False

    def collect_ready_reqs(self):
        return []

    def get_token_stats(self):
        device_capacity = device_tokens = 0
        for child in self.pools.values():
            allocator = child.token_to_kv_pool_allocator.hisparse_attn_allocator
            device_capacity += allocator.size
            device_tokens += allocator.size - allocator.available_size()
        allocator = self.token_to_kv_pool_allocator
        host_tokens = allocator.size_full - allocator.full_available_size()
        return HiSparseTokenStats(
            device_tokens,
            device_tokens / device_capacity if device_capacity else 0,
            host_tokens,
            host_tokens / allocator.size_full if allocator.size_full else 0,
        )

    def destroy(self):
        for child in self.pools.values():
            child.destroy()
