"""Qwen3 head RMSNorm, RoPE, and KV cache update kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import next_power_of_2, require_cuda_contiguous


_HEAD_RMSNORM_ROPE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
]

_STORE_CACHE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
]


@triton.jit
def _head_rmsnorm_rope_kernel(
    x,
    weight,
    cos,
    sin,
    out,
    TOKENS,
    X_STRIDE_TOKEN,
    X_STRIDE_HEAD,
    Q_LEN,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    q_pos = token - (token // Q_LEN) * Q_LEN
    cols = tl.arange(0, BLOCK_D)
    mask = cols < HEAD_DIM
    base = token * X_STRIDE_TOKEN + head * X_STRIDE_HEAD
    out_base = (token * NUM_HEADS + head) * HEAD_DIM
    vals = tl.load(x + base + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(vals * vals, axis=0) / HEAD_DIM
    scale = tl.rsqrt(variance + EPS)
    w = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    vals = vals * scale * w

    half_cols = tl.arange(0, BLOCK_HALF)
    half_mask = half_cols < (HEAD_DIM // 2)
    x0 = tl.load(x + base + half_cols, mask=half_mask, other=0.0).to(tl.float32) * scale * tl.load(
        weight + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x + base + half_cols + HEAD_DIM // 2, mask=half_mask, other=0.0).to(tl.float32) * scale * tl.load(
        weight + half_cols + HEAD_DIM // 2, mask=half_mask, other=0.0).to(tl.float32)
    c = tl.load(cos + (START_POS + q_pos) * (HEAD_DIM // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    s = tl.load(sin + (START_POS + q_pos) * (HEAD_DIM // 2) + half_cols, mask=half_mask, other=0.0).to(tl.float32)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    tl.store(out + out_base + half_cols, y0.to(out.dtype.element_ty), mask=half_mask)
    tl.store(out + out_base + half_cols + HEAD_DIM // 2, y1.to(out.dtype.element_ty), mask=half_mask)


@triton.jit
def _store_cache_kernel(
    src,
    cache,
    TOKENS,
    SRC_STRIDE_TOKEN,
    SRC_STRIDE_HEAD,
    Q_LEN,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < HEAD_DIM
    batch = token // Q_LEN
    q_pos = token - batch * Q_LEN
    cache_pos = START_POS + q_pos
    vals = tl.load(src + token * SRC_STRIDE_TOKEN + head * SRC_STRIDE_HEAD + cols, mask=mask, other=0.0)
    cache_base = ((batch * MAX_SEQ_LEN + cache_pos) * NUM_HEADS + head) * HEAD_DIM
    tl.store(cache + cache_base + cols, vals, mask=mask)


_head_rmsnorm_rope_kernel_autotuned = triton.autotune(
    configs=_HEAD_RMSNORM_ROPE_AUTOTUNE_CONFIGS,
    key=["TOKENS", "Q_LEN", "NUM_HEADS", "HEAD_DIM"],
    cache_results=True,
)(_head_rmsnorm_rope_kernel)

_store_cache_kernel_autotuned = triton.autotune(
    configs=_STORE_CACHE_AUTOTUNE_CONFIGS,
    key=["TOKENS", "Q_LEN", "NUM_HEADS", "HEAD_DIM"],
    cache_results=True,
)(_store_cache_kernel)


def head_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    eps: float,
) -> torch.Tensor:
    """Apply per-head RMSNorm and Qwen-style rotary embedding to ``[tokens, heads, dim]``."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.stride(-1) != 1:
        raise ValueError(f"x last dimension must be contiguous, got stride={x.stride()}")
    require_cuda_contiguous("weight", weight)
    require_cuda_contiguous("cos", cos)
    require_cuda_contiguous("sin", sin)
    if x.dim() != 3:
        raise ValueError(f"x must be [tokens, heads, head_dim], got {tuple(x.shape)}")
    tokens, heads, head_dim = x.shape
    if weight.numel() != head_dim:
        raise ValueError(f"weight size {weight.numel()} does not match head_dim {head_dim}")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    block_d = next_power_of_2(head_dim)
    block_half = next_power_of_2(head_dim // 2)
    _head_rmsnorm_rope_kernel_autotuned[(tokens, heads)](
        x,
        weight,
        cos,
        sin,
        out,
        TOKENS=tokens,
        X_STRIDE_TOKEN=x.stride(0),
        X_STRIDE_HEAD=x.stride(1),
        Q_LEN=q_len,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        START_POS=start_pos,
        EPS=eps,
        BLOCK_D=block_d,
        BLOCK_HALF=block_half,
    )
    return out


def store_cache(src: torch.Tensor, cache: torch.Tensor, *, q_len: int, start_pos: int) -> None:
    """Store ``src[tokens, heads, dim]`` into ``cache[batch, max_seq, heads, dim]``."""
    if not src.is_cuda:
        raise ValueError("src must be a CUDA tensor")
    if src.stride(-1) != 1:
        raise ValueError(f"src last dimension must be contiguous, got stride={src.stride()}")
    require_cuda_contiguous("cache", cache)
    if src.dim() != 3 or cache.dim() != 4:
        raise ValueError(f"expected src [tokens, heads, dim] and cache [batch, max, heads, dim]")
    tokens, heads, head_dim = src.shape
    batch, max_seq_len, cache_heads, cache_dim = cache.shape
    if cache_heads != heads or cache_dim != head_dim:
        raise ValueError(f"cache shape {tuple(cache.shape)} does not match src shape {tuple(src.shape)}")
    if tokens != batch * q_len:
        raise ValueError(f"tokens={tokens} does not equal batch={batch} * q_len={q_len}")
    if start_pos + q_len > max_seq_len:
        raise ValueError(f"cache overflow: start_pos={start_pos} q_len={q_len} max_seq_len={max_seq_len}")
    block_d = next_power_of_2(head_dim)
    _store_cache_kernel_autotuned[(tokens, heads)](
        src,
        cache,
        TOKENS=tokens,
        SRC_STRIDE_TOKEN=src.stride(0),
        SRC_STRIDE_HEAD=src.stride(1),
        Q_LEN=q_len,
        MAX_SEQ_LEN=max_seq_len,
        NUM_HEADS=heads,
        HEAD_DIM=head_dim,
        START_POS=start_pos,
        BLOCK_D=block_d,
    )
