"""Shared helpers for tutorial kernels."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


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


NORM_LOOP_CONFIGS = [
    triton.Config({"TILE_N": tile_n}, num_warps=warps)
    for tile_n in (1024, 2048, 4096, 8192)
    for warps in (4, 8, 16)
]


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


def pointwise_num_warps(tile_size: int) -> int:
    if tile_size < 2048:
        return 4
    if tile_size < 4096:
        return 8
    return 16


def pointwise_1d_launch(total: int) -> tuple[int, int, int, int]:
    tile_size = min(512, triton.next_power_of_2(total))
    num_tiles = triton.cdiv(total, tile_size)
    num_ctas = min(65536, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    return tile_size, num_ctas, tiles_per_cta, pointwise_num_warps(tile_size)


def mha_config(q_len: int, num_heads: int, num_sms: int) -> tuple[int, int, int, int]:
    total_rows = q_len * num_heads
    avg_rows_per_sm = total_rows / num_sms
    avg_rows_per_batch = q_len
    avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
    if avg_rows_per_cta > 64:
        return 128, 32, 4, 1
    if avg_rows_per_cta > 32:
        return 64, 64, 4, 1
    if avg_rows_per_cta > 16:
        return 32, 64, 4, 1
    return 16, 64, 4, 1


def as_normalized_shape_tuple(normalized_shape: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (normalized_shape, )
    return tuple(int(dim) for dim in normalized_shape)


def norm_2d_shape(x: torch.Tensor, normalized_shape: int | tuple[int, ...] | list[int]) -> tuple[int, int]:
    shape = as_normalized_shape_tuple(normalized_shape)
    if len(shape) != 1:
        raise NotImplementedError("RMSNorm kernels currently support one normalized dimension")
    if x.shape[-1] != shape[0]:
        raise ValueError(f"normalized_shape {shape} does not match input last dimension {x.shape[-1]}")
    return math.prod(x.shape[:-1]), shape[0]


def row_stride(tensor: torch.Tensor) -> int:
    return tensor.stride(-2) if tensor.ndim >= 2 else 0
