"""Fixed-width DSpark transactions for the paged C1/C2 HiSparse pools."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.hisparse import (
    HiSparseSpecState,
    load_cache_to_device_buffer_spec_mla,
)


@triton.jit
def _prepare_verify(
    REQS,
    PREFIX,
    LOCS,
    ORIGINAL_LOCS,
    TOKENS,
    HOST,
    MAPPING,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    HOT: tl.constexpr,
    PAGE: tl.constexpr,
    BUFFER_STRIDE: tl.constexpr,
    TOKEN_LAYER_STRIDE: tl.constexpr,
    HOST_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid, layer = tl.program_id(0), tl.program_id(1)
    rid = tl.load(REQS + bid)
    prefix = tl.load(PREFIX + bid)
    offsets = tl.arange(0, BLOCK)
    base = prefix // RATIO
    count = (prefix + WIDTH) // RATIO - base
    # One reserved page owns this round's writes; old labels must not survive.
    tl.store(
        TOKENS + layer * TOKEN_LAYER_STRIDE + rid * BUFFER_STRIDE + HOT + offsets,
        tl.where(offsets < count, base + offsets, -1),
        offsets < PAGE,
    )
    if layer == 0:
        pos = prefix + offsets
        complete = (offsets < WIDTH) & ((pos + 1) % RATIO == 0)
        logical = tl.load(LOCS + bid * WIDTH + offsets, complete, other=0) // RATIO
        compressed = (pos + 1) // RATIO - 1
        dst = tl.load(
            ORIGINAL_LOCS + rid * BUFFER_STRIDE + HOT + compressed - base,
            complete,
            other=0,
        )
        tl.store(MAPPING + logical, dst, complete & (logical > 0))
        tl.store(HOST + rid * HOST_STRIDE + compressed, logical, complete)


@triton.jit
def _commit_indices(
    REQS,
    PREFIX,
    LOCS,
    ACCEPT,
    ORIGINAL_LOCS,
    HOST,
    MAPPING,
    SRC,
    DST,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    HOT: tl.constexpr,
    BUFFER_STRIDE: tl.constexpr,
    HOST_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    rid = tl.load(REQS + bid)
    prefix = tl.load(PREFIX + bid)
    accepted = tl.load(ACCEPT + bid)
    step = tl.arange(0, BLOCK)
    pos = prefix + step
    complete = (step < WIDTH) & ((pos + 1) % RATIO == 0)
    logical = tl.load(LOCS + bid * WIDTH + step, complete, other=0) // RATIO
    compressed = (pos + 1) // RATIO - 1
    source = tl.load(
        ORIGINAL_LOCS + rid * BUFFER_STRIDE + HOT + compressed - prefix // RATIO,
        complete,
        other=0,
    )
    keep = complete & (step < accepted) & (logical > 0)
    tl.store(SRC + bid * WIDTH + step, tl.where(keep, source, -1), step < WIDTH)
    tl.store(DST + bid * WIDTH + step, tl.where(keep, logical, -1), step < WIDTH)
    # Accepted records are now backed by host memory, not permanently resident
    # in this round's reserved write page. Next round may reuse every write slot.
    tl.store(MAPPING + logical, 0, complete & (logical > 0))
    tl.store(HOST + rid * HOST_STRIDE + compressed, -1, complete & ~keep)


class V41SpeculativePool:
    """One ratio's cache state; scratch ownership is independent of slot rotation."""

    def __init__(self, child, width: int):
        self.child = child
        self.width = width
        self.top_k = child.top_k
        self.occurrences = width * self.top_k
        self.page = child.mem_pool_device.page_size
        if not 2 <= width <= min(16, self.page):
            raise ValueError("V4.1 HiSparse DSpark requires verify width in [2, 16]")
        if self.top_k % 64 or self.top_k < 64 or self.occurrences > 8192:
            raise ValueError(
                "V4.1 HiSparse DSpark requires top_k % 64 == 0 and width * top_k <= 8192"
            )
        self.capacity = triton.cdiv(self.occurrences, self.page) * self.page
        layers, slots, _ = child.req_device_buffer_tokens.shape
        if slots > child.device_buffer_size:
            raise ValueError(
                "HiSparse DSpark request-slot count exceeds CLOCK control capacity"
            )
        device = child.req_to_device_buffer.device
        self.original_scratch = {}
        self.states = []
        for _ in range(layers):
            self.states.append(
                HiSparseSpecState(
                    torch.full(
                        (slots, 2, triton.next_power_of_2(child.device_buffer_size)),
                        -1,
                        dtype=torch.int64,
                        device=device,
                    ),
                    torch.zeros(
                        (slots + 1, child.device_buffer_size),
                        dtype=torch.int32,
                        device=device,
                    ),
                    torch.full(
                        (slots, self.capacity), -1, dtype=torch.int32, device=device
                    ),
                    torch.zeros(
                        (slots + 1, max(4 * slots, 5 * self.occurrences)),
                        dtype=torch.int32,
                        device=device,
                    ),
                )
            )
        self.output = torch.full(
            (slots, width, self.top_k), -1, dtype=torch.int32, device=device
        )

    def resize_output(self, max_bs):
        if self.output.shape[0] < max_bs:
            self.output = self.output.new_full((max_bs, self.width, self.top_k), -1)

    def admit(self, req):
        rid = req.kv.req_pool_idx
        if rid in self.original_scratch:
            raise RuntimeError("HiSparse DSpark request already owns scratch pages")
        allocator = self.child.token_to_kv_pool_allocator.hisparse_attn_allocator
        locs = allocator.alloc(self.capacity)
        if locs is None:
            raise RuntimeError(
                "HiSparse DSpark cannot allocate per-request verify scratch"
            )
        self.original_scratch[rid] = locs
        for state in self.states:
            state.scratch_locs[rid].copy_(locs)
        # Short prefills initialize unused hot labels optimistically for ordinary
        # decode. DSpark writes into its reserved page instead of those hot slots.
        n = self.child.host_token_len(req.extend_end)
        labels = self.child.req_device_buffer_tokens[
            :, rid, : self.child.device_buffer_size
        ]
        labels.masked_fill_(labels >= n, -1)
        self.child.req_device_buffer_tokens[
            :, rid, self.child.device_buffer_size :
        ].fill_(-1)

    def release(self, req):
        rid = req.kv.req_pool_idx
        locs = self.original_scratch.pop(rid, None)
        if locs is not None:
            self.child.token_to_kv_pool_allocator.hisparse_attn_allocator.free(locs)
        slots = self.child.req_to_device_buffer.shape[0]
        for state in self.states:
            state.cache_index[rid].fill_(-1)
            state.cache_policy[0, rid] = 0
            state.cache_policy[rid + 1].zero_()
            state.scratch_locs[rid].fill_(-1)
            state.scratch_state[0, rid : 4 * slots : slots] = 0
            state.scratch_state[rid + 1].zero_()

    def prepare(self, reqs, prefix, locs):
        child = self.child
        _prepare_verify[(reqs.numel(), len(self.states))](
            reqs,
            prefix,
            locs,
            child.req_to_device_buffer,
            child.req_device_buffer_tokens,
            child.req_to_host_pool,
            child.mem_pool_device.full_to_hisparse_device_index_mapping,
            self.width,
            child.compress_ratio,
            child.device_buffer_size,
            self.page,
            child.padded_buffer_size,
            child.req_device_buffer_tokens.stride(0),
            child.req_to_host_pool.stride(0),
            triton.next_power_of_2(self.page),
        )

    def commit(self, reqs, prefix, locs, accepted):
        child = self.child
        src = torch.empty(
            reqs.numel() * self.width, dtype=torch.int64, device=reqs.device
        )
        dst = torch.empty_like(src)
        _commit_indices[(reqs.numel(),)](
            reqs,
            prefix,
            locs,
            accepted,
            child.req_to_device_buffer,
            child.req_to_host_pool,
            child.mem_pool_device.full_to_hisparse_device_index_mapping,
            src,
            dst,
            self.width,
            child.compress_ratio,
            child.device_buffer_size,
            child.padded_buffer_size,
            child.req_to_host_pool.stride(0),
            triton.next_power_of_2(self.width),
        )
        child.mem_pool_host.backup_from_device_all_layer(
            child.mem_pool_device, dst, src
        )

    def swap(self, reqs, positions, topk, layer):
        child = self.child
        bs = reqs.numel()
        if topk.shape != (bs * self.width, self.top_k):
            raise ValueError(
                "HiSparse DSpark requires a fixed-width, request-major Top-K table"
            )
        output = self.output[:bs]
        load_cache_to_device_buffer_spec_mla(
            top_k_tokens=topk.view(bs, self.width, self.top_k),
            device_buffer_tokens=child.req_device_buffer_tokens[layer],
            host_cache_locs=child.req_to_host_pool,
            device_buffer_locs=child.req_device_buffer_token_locs[layer],
            host_cache=child.mem_pool_host.kv_buffer[layer],
            device_buffer=child.mem_pool_device.kv_buffer[layer],
            top_k_device_locs=output,
            req_pool_indices=reqs,
            seq_lens=((positions + 1) // child.compress_ratio).to(torch.int32),
            state=self.states[layer],
            num_real_reqs=child.num_real_reqs,
            paged_layout=child.mem_pool_host.hisparse_layout,
        )
        return output.view(bs * self.width, self.top_k)
