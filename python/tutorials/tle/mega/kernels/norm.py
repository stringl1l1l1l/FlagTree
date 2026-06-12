"""RMS inv-scale kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from ._common import next_power_of_2, require_cuda_contiguous


_RMS_INV_SCALE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]


@triton.jit
def _rms_inv_scale_kernel(
    x,
    inv_scale_out,
    ROWS,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    vals = tl.load(x + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    smem = tle.gpu.alloc([1, BLOCK_H], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    stage = tl.zeros([BLOCK_H], dtype=tl.int32)
    ptrs = tle.gpu.local_ptr(smem, (stage, cols))
    tl.store(ptrs, vals, mask=mask)
    vals = tl.load(ptrs, mask=mask, other=0.0)

    variance = tl.sum(vals * vals, axis=0) / HIDDEN
    tl.store(inv_scale_out + row, tl.rsqrt(variance + EPS))


@triton.jit
def _add_rms_inv_scale_kernel(
    x,
    residual,
    residual_out,
    inv_scale_out,
    ROWS,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    vals = tl.load(x + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    residual_vals = tl.load(residual + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    summed = vals + residual_vals

    smem = tle.gpu.alloc([1, BLOCK_H], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    stage = tl.zeros([BLOCK_H], dtype=tl.int32)
    ptrs = tle.gpu.local_ptr(smem, (stage, cols))
    tl.store(ptrs, summed, mask=mask)
    summed = tl.load(ptrs, mask=mask, other=0.0)

    variance = tl.sum(summed * summed, axis=0) / HIDDEN
    tl.store(residual_out + row * HIDDEN + cols, summed.to(residual_out.dtype.element_ty), mask=mask)
    tl.store(inv_scale_out + row, tl.rsqrt(variance + EPS))


_rms_inv_scale_kernel_autotuned = triton.autotune(
    configs=_RMS_INV_SCALE_AUTOTUNE_CONFIGS,
    key=["ROWS", "HIDDEN"],
    cache_results=True,
)(_rms_inv_scale_kernel)

_add_rms_inv_scale_kernel_autotuned = triton.autotune(
    configs=_RMS_INV_SCALE_AUTOTUNE_CONFIGS,
    key=["ROWS", "HIDDEN"],
    cache_results=True,
)(_add_rms_inv_scale_kernel)


def rms_inv_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return per-row ``rsqrt(mean(x * x) + eps)`` as ``float32``."""
    require_cuda_contiguous("x", x)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")
    rows, hidden = x.shape
    inv_scale = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    _rms_inv_scale_kernel_autotuned[(rows, )](
        x,
        inv_scale,
        ROWS=rows,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK_H=next_power_of_2(hidden),
    )
    return inv_scale


def add_rms_inv_scale(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add ``x`` to ``residual`` and return ``(sum, inv_rms(sum))``.

    The scale is kept as a compact per-row tensor so the following projection
    can consume the unnormalized hidden state directly.
    """
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("residual", residual)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} does not match x shape {tuple(x.shape)}")
    rows, hidden = x.shape
    residual_out = torch.empty_like(x)
    inv_scale = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    _add_rms_inv_scale_kernel_autotuned[(rows, )](
        x,
        residual,
        residual_out,
        inv_scale,
        ROWS=rows,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK_H=next_power_of_2(hidden),
    )
    return residual_out, inv_scale
