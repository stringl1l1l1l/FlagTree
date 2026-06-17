"""TLE-backed kernels used by the Qwen3 megakernel tutorial."""

from .attention import attention_decode, attention_ws, flash_attn_varlen_func
from .embedding import embedding
from .linear import (
    linear,
    linear_backend_name,
    linear_cluster_slicedk,
    linear_cluster_slicedk_tma,
    linear_gemv,
    linear_hopper_tma,
    linear_hopper_tma_splitk,
    linear_slicedk,
    linear_splitk,
    linear_streamk,
    linear_tilewise,
    linear_tilewise_tma,
    lm_head,
    qkv_linear,
    silu_and_mul,
    silu_and_mul_out,
)
from .linear_rmsnorm import (
    linear_rmsnorm_mega_scheduler,
    linear_rmsnorm_mega_task_grid,
    linear_rmsnorm_reference,
    linear_rmsnorm_triton_baseline,
    validate_linear_rmsnorm_mega_scheduler,
    validate_linear_rmsnorm_mega_task_grid,
)
from .norm import fused_add_rms_norm, rms_norm
from .rotary_cache import apply_rotary_pos_emb, head_rmsnorm_rope, store_cache

__all__ = [
    "apply_rotary_pos_emb",
    "attention_decode",
    "attention_ws",
    "embedding",
    "flash_attn_varlen_func",
    "fused_add_rms_norm",
    "head_rmsnorm_rope",
    "linear",
    "linear_backend_name",
    "linear_cluster_slicedk",
    "linear_cluster_slicedk_tma",
    "linear_gemv",
    "linear_hopper_tma",
    "linear_hopper_tma_splitk",
    "linear_slicedk",
    "linear_splitk",
    "linear_streamk",
    "linear_tilewise",
    "linear_tilewise_tma",
    "lm_head",
    "linear_rmsnorm_mega_scheduler",
    "linear_rmsnorm_mega_task_grid",
    "linear_rmsnorm_reference",
    "linear_rmsnorm_triton_baseline",
    "qkv_linear",
    "rms_norm",
    "silu_and_mul",
    "silu_and_mul_out",
    "store_cache",
    "validate_linear_rmsnorm_mega_scheduler",
    "validate_linear_rmsnorm_mega_task_grid",
]
