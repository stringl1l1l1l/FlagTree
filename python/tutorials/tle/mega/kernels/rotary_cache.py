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


@triton.jit
def _apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    rotary_interleaved: tl.constexpr,
    max_position_embeddings: tl.constexpr,
):
    s_id = tl.program_id(0)
    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_id = tl.load(pos_ptr + s_id * p_stride_s)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s
    tl.device_assert(pos_id < max_position_embeddings, "position id out of bound")

    ordered_block = tl.arange(0, padded_head_dim)
    mask = ordered_block < head_dim
    if rotary_interleaved:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + head_dim // 2) % head_dim
        sin_cos_block = ordered_block % (head_dim // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < head_dim // 2, sin, -sin)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s
    for off_h in tl.static_range(0, num_q_heads):
        ordered_cols = off_h * q_stride_h + ordered_block * q_stride_d
        rotated_cols = off_h * q_stride_h + rotated_block * q_stride_d
        output_offs = off_h * oq_stride_h + ordered_block * oq_stride_d
        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        tl.store(oq_ptr + output_offs, q * cos + rotated_q * sin, mask=mask)

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s
    for off_h in tl.static_range(0, num_k_heads):
        ordered_cols = off_h * k_stride_h + ordered_block * k_stride_d
        rotated_cols = off_h * k_stride_h + rotated_block * k_stride_d
        output_offs = off_h * ok_stride_h + ordered_block * ok_stride_d
        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        tl.store(ok_ptr + output_offs, k * cos + rotated_k * sin, mask=mask)


@triton.jit
def _apply_rotary_pos_emb_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    rotary_interleaved: tl.constexpr,
    max_position_embeddings: tl.constexpr,
):
    s_id = tl.program_id(0)
    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_id = tl.load(pos_ptr + s_id * p_stride_s)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s
    tl.device_assert(pos_id < max_position_embeddings, "position id out of bound")

    ordered_block = tl.arange(0, padded_head_dim)
    mask = ordered_block < head_dim
    if rotary_interleaved:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + head_dim // 2) % head_dim
        sin_cos_block = ordered_block % (head_dim // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < head_dim // 2, sin, -sin)

    q_ptr += s_id * q_stride_s
    for off_h in tl.static_range(0, num_q_heads):
        ordered_cols = off_h * q_stride_h + ordered_block * q_stride_d
        rotated_cols = off_h * q_stride_h + rotated_block * q_stride_d
        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        tl.store(q_ptr + ordered_cols, q * cos + rotated_q * sin, mask=mask)

    k_ptr += s_id * k_stride_s
    for off_h in tl.static_range(0, num_k_heads):
        ordered_cols = off_h * k_stride_h + ordered_block * k_stride_d
        rotated_cols = off_h * k_stride_h + rotated_block * k_stride_d
        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        tl.store(k_ptr + ordered_cols, k * cos + rotated_k * sin, mask=mask)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape[-1] != q.shape[-1]:
        raise ValueError(f"q and k must have the same last dimension, got {q.shape} and {k.shape}")
    if cos.shape[-1] != sin.shape[-1] or cos.shape[-1] * 2 != q.shape[-1]:
        raise ValueError(f"cos/sin shape {cos.shape}/{sin.shape} is incompatible with q shape {q.shape}")
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("cos and sin must be contiguous at the last dimension")
    if q.shape[:-2] != k.shape[:-2]:
        raise ValueError(f"q and k must have the same token shape, got {q.shape[:-2]} and {k.shape[:-2]}")
    q_shape = q.shape
    k_shape = k.shape
    if position_ids is not None:
        if position_ids.shape != q.shape[:-2]:
            raise ValueError(f"position_ids must have shape {q.shape[:-2]}, got {position_ids.shape}")
        position_ids = position_ids.view(-1)
        seq_len = 0
    else:
        if len(q.shape) != 4:
            raise ValueError(f"q must have 4 dimensions if position_ids is not provided, got {q.shape}")
        seq_len = q.shape[-3]
    q_view = q.view(-1, q.shape[-2], q.shape[-1])
    k_view = k.view(-1, k.shape[-2], k.shape[-1])
    head_dim = q_view.shape[-1]
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)
    if inplace:
        _apply_rotary_pos_emb_inplace_kernel[(q_view.shape[0], )](
            q_view,
            k_view,
            cos,
            sin,
            position_ids,
            q_view.stride(0),
            q_view.stride(1),
            q_view.stride(2),
            k_view.stride(0),
            k_view.stride(1),
            k_view.stride(2),
            position_ids.stride(0) if position_ids is not None else 0,
            cos.stride(0),
            sin.stride(0),
            seq_len,
            q_view.shape[-2],
            k_view.shape[-2],
            head_dim,
            padded_head_dim,
            rotary_interleaved,
            cos.shape[0],
        )
        return q_view.view(q_shape), k_view.view(k_shape)

    q_embed = torch.empty_like(q_view)
    k_embed = torch.empty_like(k_view)
    _apply_rotary_pos_emb_kernel[(q_view.shape[0], )](
        q_embed,
        k_embed,
        q_view,
        k_view,
        cos,
        sin,
        position_ids,
        q_view.stride(0),
        q_view.stride(1),
        q_view.stride(2),
        k_view.stride(0),
        k_view.stride(1),
        k_view.stride(2),
        q_embed.stride(0),
        q_embed.stride(1),
        q_embed.stride(2),
        k_embed.stride(0),
        k_embed.stride(1),
        k_embed.stride(2),
        position_ids.stride(0) if position_ids is not None else 0,
        cos.stride(0),
        sin.stride(0),
        seq_len,
        q_view.shape[-2],
        k_view.shape[-2],
        head_dim,
        padded_head_dim,
        rotary_interleaved,
        cos.shape[0],
    )
    return q_embed.view(q_shape), k_embed.view(k_shape)
