"""Stride-aware RMSNorm kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import NORM_LOOP_CONFIGS, norm_2d_shape, prev_multiple_of, row_stride


@triton.jit(do_not_specialize=["eps"])
def _rms_norm_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    w_stride_c,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(in_ptr + pid * x_stride_r + offsets * x_stride_c, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offsets * w_stride_c, mask=mask, other=0.0)
    y = (x * inv_rms * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + pid * y_stride_r + offsets * y_stride_c, y, mask=mask)


@triton.autotune(configs=NORM_LOOP_CONFIGS, key=["n_cols"])
@triton.jit(do_not_specialize=["eps"])
def _rms_norm_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    w_stride_c,
    n_cols: tl.constexpr,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros((TILE_N, ), dtype=tl.float32)
    num_steps = tl.cdiv(n_cols, TILE_N)

    for step in range(0, num_steps - 1):
        offsets = step * TILE_N + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * x_stride_r + offsets * x_stride_c).to(tl.float32)
        acc += x * x

    offsets = (num_steps - 1) * TILE_N + tl.arange(0, TILE_N)
    mask = offsets < n_cols
    x = tl.load(in_ptr + pid * x_stride_r + offsets * x_stride_c, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    inv_rms = tl.rsqrt(tl.sum(acc) / n_cols + eps)
    prev_multiple = prev_multiple_of(n_cols, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = offsets < n_cols
        x = tl.load(
            in_ptr + pid * x_stride_r + offsets * x_stride_c,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + offsets * w_stride_c, mask=mask, other=0.0)
        y = (x * inv_rms * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + pid * y_stride_r + offsets * y_stride_c, y, mask=mask)

    for start_n in range(TILE_N, n_cols, TILE_N):
        offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * x_stride_r + offsets * x_stride_c, eviction_policy="evict_first").to(tl.float32)
        w = tl.load(w_ptr + offsets * w_stride_c)
        y = (x * inv_rms * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + pid * y_stride_r + offsets * y_stride_c, y)


def rms_norm(
    x: torch.Tensor,
    normalized_shape: int | tuple[int, ...] | list[int],
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    rows, n_cols = norm_2d_shape(x, normalized_shape)
    if weight.shape[-1] != n_cols:
        raise ValueError(f"weight last dimension {weight.shape[-1]} does not match normalized size {n_cols}")
    y = torch.empty_like(x)
    if n_cols <= 4096:
        _rms_norm_kernel[(rows, )](
            y,
            x,
            weight,
            row_stride(y),
            y.stride(-1),
            row_stride(x),
            x.stride(-1),
            weight.stride(-1),
            n_cols,
            eps,
            triton.next_power_of_2(n_cols),
        )
    else:
        _rms_norm_loop_kernel[(rows, )](
            y,
            x,
            weight,
            row_stride(y),
            y.stride(-1),
            row_stride(x),
            x.stride(-1),
            weight.stride(-1),
            n_cols,
            eps,
        )
    return y


@triton.jit(do_not_specialize=["eps"])
def _fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    w_ptr,
    in_stride_r,
    in_stride_c,
    r_stride_r,
    r_stride_c,
    w_stride_c,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(input_ptr + pid * in_stride_r + offsets * in_stride_c, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + pid * r_stride_r + offsets * r_stride_c, mask=mask, other=0.0).to(tl.float32)
    x += r
    tl.store(residual_ptr + pid * r_stride_r + offsets * r_stride_c, x, mask=mask)
    var = tl.sum(x * x / n_cols, axis=0)
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offsets * w_stride_c, mask=mask, other=0.0)
    y = (x * inv_rms * w).to(input_ptr.dtype.element_ty)
    tl.store(input_ptr + pid * in_stride_r + offsets * in_stride_c, y, mask=mask)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: int | tuple[int, ...] | list[int],
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, n_cols = norm_2d_shape(input, normalized_shape)
    if residual.shape != input.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} does not match input shape {tuple(input.shape)}")
    if weight.shape[-1] != n_cols:
        raise ValueError(f"weight last dimension {weight.shape[-1]} does not match normalized size {n_cols}")
    _fused_add_rms_norm_kernel[(rows, )](
        input,
        residual,
        weight,
        row_stride(input),
        input.stride(-1),
        row_stride(residual),
        residual.stride(-1),
        weight.stride(-1),
        n_cols,
        eps,
        triton.next_power_of_2(n_cols),
    )
    return input, residual
