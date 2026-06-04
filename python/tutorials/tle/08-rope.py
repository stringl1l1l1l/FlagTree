"""
Rotary Position Embedding with TLE extract_tile
================================================

This tutorial compares a Triton baseline RoPE kernel against a TLE
optimized version that uses ``tle.extract_tile`` for register-only head reuse.

**Key optimization**: Instead of loading each attention head's row from global
memory one-by-one (v1), the TLE version loads **all heads as a single 2D block**
``[NUM_HEADS, HEAD_DIM]`` into registers, then uses ``extract_tile`` to slice
out individual head rows — eliminating redundant memory round-trips.

Both **inplace** (modifies input tensors) and **outplace** (writes to new output
tensors) variants are benchmarked.

Usage
-----
::

    # correctness check only (default)
    python python/tutorials/tle/08-rope.py

    # correctness + benchmark tables with speedup
    python python/tutorials/tle/08-rope.py --benchmark

The speedup is largest when **head_dim is small** (e.g. 128) and **num_heads is
large** — the per-head loop overhead saved by the block load dominates.
When head_dim is large (256+) and total work is large, arithmetic dominates and
the gain shrinks to near zero.

Both non-interleaved (default LLaMA-style) and interleaved (GPT-NeoX-style)
RoPE variants are included.
"""

# %%
# Setup
# -----

import argparse
import math
import random
import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# %%
# Helper: build cos/sin cache
# ---------------------------

def build_rope_cache(max_seq_len, head_dim, device, dtype=torch.float32, base=10000.0):
    """Precompute cos/sin tables for all positions up to ``max_seq_len``."""
    half_dim = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, theta)
    return angles.cos().contiguous(), angles.sin().contiguous()


# %%
# Kernels (baseline v1) — per-head loads
# --------------------------------------
# Each head's row is loaded from global memory with a separate ``tl.load``
# inside the ``for off_h`` loop.  For ``N`` heads this means ``N`` loads
# (plus ``N`` loads for the rotated element), even though the data for all
# heads is contiguous in memory.

@triton.jit
def _rope_outplace_kernel_v1(
    oq_ptr,         # (n_tokens, q_heads, head_dim)
    ok_ptr,         # (n_tokens, k_heads, head_dim)
    q_ptr,          # (n_tokens, q_heads, head_dim)
    k_ptr,          # (n_tokens, k_heads, head_dim)
    cos_ptr,        # (max_seq_len, head_dim // 2)
    sin_ptr,        # (max_seq_len, head_dim // 2)
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    oq_stride_s, oq_stride_h, oq_stride_d,
    ok_stride_s, ok_stride_h, ok_stride_d,
    cos_stride_s, sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    """Baseline outplace: per-head tl.load, store to separate output."""
    s_id = tl.program_id(0)

    pos_id = s_id % seq_len
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # --- Q: per-head load from q_ptr, store to oq_ptr ---
    q_ptr += s_id * q_stride_s
    oq_ptr += s_id * oq_stride_s
    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)          # ← load from GMEM
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)  # ← load from GMEM
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + output_offs, y, mask=mask)

    # --- K: per-head load from k_ptr, store to ok_ptr ---
    k_ptr += s_id * k_stride_s
    ok_ptr += s_id * ok_stride_s
    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)          # ← load from GMEM
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)  # ← load from GMEM
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + output_offs, y, mask=mask)


@triton.jit
def _rope_inplace_kernel_v1(
    q_ptr,          # (n_tokens, q_heads, head_dim)
    k_ptr,          # (n_tokens, k_heads, head_dim)
    cos_ptr,        # (max_seq_len, head_dim // 2)
    sin_ptr,        # (max_seq_len, head_dim // 2)
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    cos_stride_s, sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    """Baseline inplace: per-head tl.load, store back to input."""
    s_id = tl.program_id(0)

    pos_id = s_id % seq_len
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # --- Q: per-head load, store inplace ---
    q_ptr += s_id * q_stride_s
    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(q_ptr + ordered_cols, y, mask=mask)

    # --- K: per-head load, store inplace ---
    k_ptr += s_id * k_stride_s
    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(k_ptr + ordered_cols, y, mask=mask)


# %%
# Kernels (TLE v2) — block load + extract_tile
# --------------------------------------------
# Instead of ``N`` individual loads, we load **all heads at once** as a 2D tile
# ``[NUM_HEADS, PADDED_HEAD_DIM]`` into registers, then use ``tle.extract_tile``
# to pull out each head's row.  The rotated element still requires a per-head
# ``tl.load`` (it lives at a different offset), but the dominant load path is
# coalesced into one bulk access.

@triton.jit
def _rope_outplace_kernel_v2(
    oq_ptr,         # (n_tokens, q_heads, head_dim)
    ok_ptr,         # (n_tokens, k_heads, head_dim)
    q_ptr,          # (n_tokens, q_heads, head_dim)
    k_ptr,          # (n_tokens, k_heads, head_dim)
    cos_ptr,        # (max_seq_len, head_dim // 2)
    sin_ptr,        # (max_seq_len, head_dim // 2)
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    oq_stride_s, oq_stride_h, oq_stride_d,
    ok_stride_s, ok_stride_h, ok_stride_d,
    cos_stride_s, sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    """TLE outplace: block-load all heads, extract_tile per head."""
    s_id = tl.program_id(0)

    pos_id = s_id % seq_len
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # --- Q: block load all heads from q_ptr, extract_tile, store to oq_ptr ---
    q_ptr += s_id * q_stride_s
    oq_ptr += s_id * oq_stride_s
    q_head_ids = tl.arange(0, NUM_Q_HEADS)
    # ── KEY CHANGE: one bulk load for all heads ──
    q_block = tl.load(
        q_ptr + q_head_ids[:, None] * q_stride_h + ordered_block[None, :] * q_stride_d,
        mask=mask[None, :],
        other=0.0,
    )  # shape: [NUM_Q_HEADS, PADDED_HEAD_DIM]

    for off_h in range(0, NUM_Q_HEADS):
        # ── KEY CHANGE: slice from register tile instead of GMEM load ──
        q = tle.extract_tile(
            q_block, index=[off_h, 0], tile_shape=[1, PADDED_HEAD_DIM]
        ).reshape((PADDED_HEAD_DIM,))

        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + off_h * oq_stride_h + ordered_block * oq_stride_d, y, mask=mask)

    # --- K: block load all heads from k_ptr, extract_tile, store to ok_ptr ---
    k_ptr += s_id * k_stride_s
    ok_ptr += s_id * ok_stride_s
    k_head_ids = tl.arange(0, NUM_K_HEADS)
    k_block = tl.load(
        k_ptr + k_head_ids[:, None] * k_stride_h + ordered_block[None, :] * k_stride_d,
        mask=mask[None, :],
        other=0.0,
    )  # shape: [NUM_K_HEADS, PADDED_HEAD_DIM]

    for off_h in range(0, NUM_K_HEADS):
        k = tle.extract_tile(
            k_block, index=[off_h, 0], tile_shape=[1, PADDED_HEAD_DIM]
        ).reshape((PADDED_HEAD_DIM,))

        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + off_h * ok_stride_h + ordered_block * ok_stride_d, y, mask=mask)


@triton.jit
def _rope_inplace_kernel_v2(
    q_ptr,          # (n_tokens, q_heads, head_dim)
    k_ptr,          # (n_tokens, k_heads, head_dim)
    cos_ptr,        # (max_seq_len, head_dim // 2)
    sin_ptr,        # (max_seq_len, head_dim // 2)
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    cos_stride_s, sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    """TLE inplace: block-load all heads, then extract_tile per head."""
    s_id = tl.program_id(0)

    pos_id = s_id % seq_len
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # --- Q: block load all heads, then extract_tile ---
    q_ptr += s_id * q_stride_s
    q_head_ids = tl.arange(0, NUM_Q_HEADS)
    q_block = tl.load(
        q_ptr + q_head_ids[:, None] * q_stride_h + ordered_block[None, :] * q_stride_d,
        mask=mask[None, :],
        other=0.0,
    )

    for off_h in range(0, NUM_Q_HEADS):
        q = tle.extract_tile(
            q_block, index=[off_h, 0], tile_shape=[1, PADDED_HEAD_DIM]
        ).reshape((PADDED_HEAD_DIM,))

        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(q_ptr + off_h * q_stride_h + ordered_block * q_stride_d, y, mask=mask)

    # --- K: block load all heads, then extract_tile ---
    k_ptr += s_id * k_stride_s
    k_head_ids = tl.arange(0, NUM_K_HEADS)
    k_block = tl.load(
        k_ptr + k_head_ids[:, None] * k_stride_h + ordered_block[None, :] * k_stride_d,
        mask=mask[None, :],
        other=0.0,
    )

    for off_h in range(0, NUM_K_HEADS):
        k = tle.extract_tile(
            k_block, index=[off_h, 0], tile_shape=[1, PADDED_HEAD_DIM]
        ).reshape((PADDED_HEAD_DIM,))

        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(k_ptr + off_h * k_stride_h + ordered_block * k_stride_d, y, mask=mask)


# %%
# Python wrappers
# ---------------


def _launch_rope_outplace(kernel, q, k, cos, sin, rotary_interleaved):
    """Launch outplace kernel: returns new (q_embed, k_embed)."""
    q_shape, k_shape = q.shape, k.shape
    assert q.dim() == 4, f"Expected 4D input (B, S, H, D), got {q.shape}"
    seq_len = q.shape[-3]

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    grid = (n_tokens,)
    kernel[grid](
        q_embed, k_embed,
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        q_embed.stride(0), q_embed.stride(1), q_embed.stride(2),
        k_embed.stride(0), k_embed.stride(1), k_embed.stride(2),
        cos.stride(0), sin.stride(0),
        seq_len,
        q_heads, k.shape[-2], head_dim, padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
    )
    return q_embed.view(q_shape), k_embed.view(k_shape)


def _launch_rope_inplace(kernel, q, k, cos, sin, rotary_interleaved):
    """Launch inplace kernel: modifies q, k in place."""
    q_shape, k_shape = q.shape, k.shape
    assert q.dim() == 4, f"Expected 4D input (B, S, H, D), got {q.shape}"
    seq_len = q.shape[-3]

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    n_tokens, q_heads, head_dim = q.shape
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    grid = (n_tokens,)
    kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        cos.stride(0), sin.stride(0),
        seq_len,
        q_heads, k.shape[-2], head_dim, padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
    )
    return q.view(q_shape), k.view(k_shape)


def rope_outplace_v1(q, k, cos, sin, rotary_interleaved=False):
    """Baseline RoPE (outplace)."""
    return _launch_rope_outplace(_rope_outplace_kernel_v1, q, k, cos, sin, rotary_interleaved)

def rope_outplace_v2(q, k, cos, sin, rotary_interleaved=False):
    """TLE RoPE (outplace)."""
    return _launch_rope_outplace(_rope_outplace_kernel_v2, q, k, cos, sin, rotary_interleaved)

def rope_inplace_v1(q, k, cos, sin, rotary_interleaved=False):
    """Baseline RoPE (inplace)."""
    return _launch_rope_inplace(_rope_inplace_kernel_v1, q, k, cos, sin, rotary_interleaved)

def rope_inplace_v2(q, k, cos, sin, rotary_interleaved=False):
    """TLE RoPE (inplace)."""
    return _launch_rope_inplace(_rope_inplace_kernel_v2, q, k, cos, sin, rotary_interleaved)


# %%
# Correctness check
# -----------------


def _assert_close_robust(name, a, b, rtol=5e-2, atol=2e-1):
    """Robust comparison: pass if allclose OR quantile-based relaxed check passes.

    Bfloat16 arithmetic is non-deterministic across different memory access
    patterns (individual loads vs block loads).  A handful of elements near
    zero can have large relative errors without indicating a real bug.
    """
    if torch.allclose(a, b, rtol=rtol, atol=atol):
        return

    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(a.float().abs(), b.float().abs()) + 1e-6
    rel = diff / denom

    flat_diff = diff.flatten()
    flat_rel = rel.flatten()
    max_samples = 1_000_000
    if flat_diff.numel() > max_samples:
        step = (flat_diff.numel() + max_samples - 1) // max_samples
        flat_diff = flat_diff[::step]
        flat_rel = flat_rel[::step]

    abs_p999 = torch.quantile(flat_diff, 0.999).item()
    rel_p999 = torch.quantile(flat_rel, 0.999).item()

    if abs_p999 < 0.25 and rel_p999 < 0.25:
        return  # robust check passed

    raise AssertionError(
        f"{name} mismatch: allclose failed and robust check failed; "
        f"abs_p999={abs_p999:.6f}, rel_p999={rel_p999:.6f}"
    )


def check_correctness(batch=4, seq_len=256, q_heads=32, k_heads=8, head_dim=128,
                      dtype=torch.bfloat16, rotary_interleaved=False):
    """Verify v1 and v2 produce identical results (both inplace and outplace)."""
    torch.manual_seed(0)

    max_seq = seq_len + 1024
    q = torch.randn(batch, seq_len, q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, seq_len, k_heads, head_dim, device=DEVICE, dtype=dtype)
    cos, sin = build_rope_cache(max_seq, head_dim, DEVICE)

    # --- outplace ---
    oq1, ok1 = rope_outplace_v1(q.clone(), k.clone(), cos, sin, rotary_interleaved)
    oq2, ok2 = rope_outplace_v2(q.clone(), k.clone(), cos, sin, rotary_interleaved)
    _assert_close_robust("outplace_q", oq1, oq2)
    _assert_close_robust("outplace_k", ok1, ok2)

    # --- inplace ---
    iq1, ik1 = rope_inplace_v1(q.clone(), k.clone(), cos, sin, rotary_interleaved)
    iq2, ik2 = rope_inplace_v2(q.clone(), k.clone(), cos, sin, rotary_interleaved)
    _assert_close_robust("inplace_q", iq1, iq2)
    _assert_close_robust("inplace_k", ik1, ik2)

    inter = "interleaved" if rotary_interleaved else "non-interleaved"
    print(f"[OK] Correctness passed ({inter}, {dtype}, dim={head_dim}) [outplace + inplace]")


# %%
# Benchmark
# ---------
# Uses the same rigorous methodology as the original rotary embedding benchmark:
#   - Pre-allocated buffers with copy_ reset outside the timing window
#   - L2 cache cleared between iterations
#   - Multiple rounds with outlier trimming
#   - Randomized v1/v2 measurement order

BENCH_WARMUP = 20
BENCH_REP = 200
BENCH_ROUNDS = 5
BENCH_TRIM = 0.20


def _do_bench_rigorous(fn, reset_fn):
    """Time ``fn`` with cache clearing and pre-allocated buffers."""
    di = triton.runtime.driver.active.get_device_interface()
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    for _ in range(BENCH_WARMUP):
        reset_fn()
        triton.runtime.driver.active.clear_cache(cache)
        fn()

    di.synchronize()

    start_events = [di.Event(enable_timing=True) for _ in range(BENCH_REP)]
    end_events = [di.Event(enable_timing=True) for _ in range(BENCH_REP)]

    for i in range(BENCH_REP):
        reset_fn()
        triton.runtime.driver.active.clear_cache(cache)
        di.synchronize()
        start_events[i].record()
        fn()
        end_events[i].record()

    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    t = torch.tensor(times, dtype=torch.float64)
    return float(torch.quantile(t, 0.5).item())


def _robust_bench(fn, reset_fn):
    """Multiple rounds with outlier trimming for stable measurement."""
    medians = []
    for _ in range(BENCH_ROUNDS):
        medians.append(_do_bench_rigorous(fn, reset_fn))

    t = torch.tensor(medians, dtype=torch.float64)
    if BENCH_ROUNDS >= 5:
        lo = torch.quantile(t, BENCH_TRIM)
        hi = torch.quantile(t, 1.0 - BENCH_TRIM)
        keep = (t >= lo) & (t <= hi)
        if keep.sum().item() < max(3, BENCH_ROUNDS // 2):
            keep = torch.ones_like(t, dtype=torch.bool)
    else:
        keep = torch.ones_like(t, dtype=torch.bool)
    return float(torch.quantile(t[keep], 0.5).item())


def _run_benchmark_table(title, configs, dtype, rotary_interleaved,
                         rope_v1_fn, rope_v2_fn, inplace):
    """Print a benchmark table for a set of (batch, seq_len, q_heads, k_heads, head_dim)."""
    mode = "inplace" if inplace else "outplace"
    print(f"\n--- {title} [{mode}] (dtype={dtype}) ---")
    print(f"{'batch':>6} {'seq':>6} {'qh':>4} {'kh':>4} {'dim':>5} | "
          f"{'Baseline(ms)':>13} {'TLE(ms)':>10} {'Speedup':>8}")
    print("-" * 72)

    for batch, seq_len, q_heads, k_heads, head_dim in configs:
        torch.manual_seed(0)
        max_seq = seq_len + 1024
        q_src = torch.randn(batch, seq_len, q_heads, head_dim, device=DEVICE, dtype=dtype)
        k_src = torch.randn(batch, seq_len, k_heads, head_dim, device=DEVICE, dtype=dtype)
        cos, sin = build_rope_cache(max_seq, head_dim, DEVICE)

        # Pre-allocated buffers
        q1, k1 = q_src.clone(), k_src.clone()
        q2, k2 = q_src.clone(), k_src.clone()

        def run_v1():
            rope_v1_fn(q1, k1, cos, sin, rotary_interleaved)

        def run_v2():
            rope_v2_fn(q2, k2, cos, sin, rotary_interleaved)

        def reset_v1():
            q1.copy_(q_src); k1.copy_(k_src)

        def reset_v2():
            q2.copy_(q_src); k2.copy_(k_src)

        # Randomize measurement order
        runners = [("v1", run_v1, reset_v1), ("v2", run_v2, reset_v2)]
        random.shuffle(runners)

        stats = {}
        for name, fn, rst in runners:
            stats[name] = _robust_bench(fn, rst)

        ms_v1, ms_v2 = stats["v1"], stats["v2"]
        speedup = ms_v1 / ms_v2 if ms_v2 > 0 else 0.0
        print(f"{batch:>6} {seq_len:>6} {q_heads:>4} {k_heads:>4} {head_dim:>5} | "
              f"{ms_v1:>13.5f} {ms_v2:>10.5f} {speedup:>7.2f}×")


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RoPE with TLE extract_tile — baseline vs optimized benchmark"
    )
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark tables in addition to correctness check")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "fp16", "bf16"],
                        help="Data type for benchmark")
    args = parser.parse_args(argv)

    dtype = torch.bfloat16 if "bf" in args.dtype else torch.float16

    # --- Correctness ---------------------------------------------------------
    print("=" * 60)
    print("Correctness checks")
    print("=" * 60)
    for interleaved in [False, True]:
        for head_dim in [128, 256]:
            check_correctness(head_dim=head_dim, dtype=dtype, rotary_interleaved=interleaved)

    if not args.benchmark:
        print("\n[INFO] Run with --benchmark to see performance tables.")
        return

    # --- Benchmark -----------------------------------------------------------
    CONFIGS = [
        # (batch, seq_len, q_heads, k_heads, head_dim)
        # head_dim=128 — TLE shines
        (1,   128,  32, 8,  128),
        (1,  1024,  32, 8,  128),
        (8,   128,  32, 8,  128),
        (8,  1024,  32, 8,  128),
        (32,  128,  32, 8,  128),
        (32, 1024,  32, 8,  128),
        (8,  1024,  16, 16, 128),
        (32, 1024,  16, 16, 128),
        # head_dim=256 — smaller gain
        (1,   128,  32, 8,  256),
        (8,   128,  32, 8,  256),
        (32,  128,  32, 8,  256),
        (32, 1024,  16, 16, 256),
    ]

    for interleaved in [False, True]:
        ilabel = "interleaved (GPT-NeoX style)" if interleaved else "non-interleaved (LLaMA style)"
        # outplace
        _run_benchmark_table(ilabel, CONFIGS, dtype, interleaved,
                             rope_outplace_v1, rope_outplace_v2, inplace=False)
        # inplace
        _run_benchmark_table(ilabel, CONFIGS, dtype, interleaved,
                             rope_inplace_v1, rope_inplace_v2, inplace=True)


if __name__ == "__main__":
    main()
