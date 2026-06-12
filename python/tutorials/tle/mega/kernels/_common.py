"""Shared helpers for tutorial kernels."""

from __future__ import annotations

import torch
import triton


def next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def default_block_m(rows: int) -> int:
    # Keep the M dimension tensor-core friendly even for decode where rows=1.
    return 16


def cuda_events_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def cdiv(a: int, b: int) -> int:
    return triton.cdiv(a, b)
