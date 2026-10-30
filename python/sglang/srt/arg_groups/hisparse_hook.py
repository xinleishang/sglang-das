from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "triton", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if get_platform().is_hip:
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if get_platform().is_hip:
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse_feature_combos(server_args: ServerArgs) -> None:

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm is not None:
        hf_config = model_config_of(server_args).hf_config
        if not (
            cfg.speculative_algorithm == "DSPARK"
            and getattr(hf_config, "model_type", None) == "deepseek_v41"
        ):
            raise ValueError(
                "--enable-hisparse supports speculative decoding only with DeepSeek-V4.1 DSpark."
            )
    if cfg.enable_mixed_chunk:
        raise ValueError("--enable-hisparse does not support --enable-mixed-chunk yet.")


def validate_deepseek_v41_hisparse(server_args: ServerArgs) -> None:
    """Keep the initial C1/C2 implementation on its direct-to-host decode path."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config
    from sglang.srt.model_executor.cuda_graph_config import Backend

    cfg = resolving_view(server_args)
    restrictions = (
        (cfg.disaggregation_mode != "decode", "requires PD decode mode"),
        (
            cfg.disaggregation_transfer_backend != "mooncake",
            "requires Mooncake PD transfer",
        ),
        (cfg.pp_size != 1, "requires PP=1"),
        (cfg.attn_cp_size != 1 or cfg.dcp_size != 1, "requires decode CP=1 and DCP=1"),
        (cfg.enable_prefill_cp, "does not support CP on the decode instance"),
        (cfg.enable_encoder_swa_bounded_replay, "does not support encoder SWA replay"),
        (
            cfg.cuda_graph_config.decode.backend
            not in (Backend.DISABLED, Backend.FULL),
            "requires --cuda-graph-backend-decode full or disabled",
        ),
        (
            cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            "requires --cuda-graph-backend-prefill disabled",
        ),
        (
            cfg.enable_hierarchical_cache or cfg.enable_unified_cache_external_linker,
            "cannot share the decode instance with HiCache or an external linker",
        ),
        (
            cfg.disaggregation_decode_enable_radix_cache,
            "does not support decode-side radix caching",
        ),
        (
            cfg.kv_cache_dtype not in ("auto", "fp8_e4m3"),
            "requires quantized paged KV (--kv-cache-dtype fp8_e4m3)",
        ),
        (is_unified_kv_triton(), "does not support the unified KV layout"),
        (
            not (get_platform().is_cuda or get_platform().is_hip),
            "requires CUDA or HIP/HCU",
        ),
    )
    for invalid, message in restrictions:
        if invalid:
            raise ValueError(f"DeepSeek-V4.1 HiSparse {message}.")
    # Validation runs before the resolved memory configuration is published.
    config = parse_hisparse_config(cfg.hisparse_config or "{}")
    ratio = config.host_to_device_ratio
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio < 1
    ):
        raise ValueError("V4.1 HiSparse host_to_device_ratio must be finite and >= 1")
    buffer_size = config.device_buffer_size
    model_config = model_config_of(server_args)
    if any(ratio not in (0, 1, 2) for ratio in model_config.compress_ratios):
        raise ValueError("V4.1 HiSparse currently supports C1/C2 compressed KV only")
    model_topk = model_config.hf_config.index_topk
    if cfg.speculative_algorithm == "DSPARK":
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        width = cfg.speculative_num_draft_tokens
        if (
            cfg.speculative_adaptive
            or read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
        ):
            raise ValueError(
                "V4.1 HiSparse DSpark requires fixed-width static verify, without adaptive speculation"
            )
        if (
            width is None
            or not 2 <= width <= 16
            or width * model_topk > 8192
            or model_topk < 64
            or model_topk % 64
        ):
            raise ValueError(
                "V4.1 HiSparse DSpark requires verify width 2-16, width * index_topk <= 8192, and index_topk % 64 == 0"
            )
        if not get_platform().is_hcu:
            raise ValueError(
                "V4.1 HiSparse DSpark currently requires the HCU indexer path with logical Top-K outputs"
            )
    if (
        not isinstance(buffer_size, int)
        or isinstance(buffer_size, bool)
        or not model_topk <= buffer_size < 32768
        or (cfg.page_size is not None and buffer_size % cfg.page_size)
    ):
        raise ValueError(
            "V4.1 HiSparse device_buffer_size must be an integer aligned to page_size, "
            "at least the model index_topk, and less than 32768."
        )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""

    cfg = resolving_view(server_args)
    if not cfg.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
        is_minimax_sparse,
    )

    hf_config = model_config_of(server_args).hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_m3_hisparse = is_minimax_sparse(hf_config)
    is_hip = get_platform().is_hip
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse or is_m3_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5), DeepSeek V4, and MiniMax M3 now. "
    )

    assert cfg.disable_radix_cache, (
        "Hierarchical sparse attention currently requires --disable-radix-cache."
    )
    validate_hisparse_feature_combos(server_args)
    if getattr(hf_config, "model_type", None) == "deepseek_v41":
        validate_deepseek_v41_hisparse(server_args)
        return

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    # MiniMax M3 uses its own Triton sparse kernels.
    if is_m3_hisparse:
        return

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)
