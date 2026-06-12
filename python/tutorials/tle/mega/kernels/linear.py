"""BF16 linear projection kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from ._common import cdiv, default_block_m, require_cuda_contiguous


def _linear_prune_configs(configs, named_args, **kwargs):
    m = int(kwargs["M"])
    n = int(kwargs["N"])
    k = int(kwargs["K"])
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        if m < 64 and block_m > 32:
            continue
        if n <= 64 and block_n > 64:
            continue
        if k < 128 and block_k > 64:
            continue
        pruned.append(config)
    return pruned or configs[:1]


def _gate_up_silu_prune_configs(configs, named_args, **kwargs):
    linear_kwargs = dict(kwargs)
    linear_kwargs["N"] = int(kwargs["INTERMEDIATE"])
    return _linear_prune_configs(configs, named_args, **linear_kwargs)


_LINEAR_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]


def _linear_tma_pre_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    nargs["x_desc"].block_shape = [block_m, block_k]
    nargs["weight_desc"].block_shape = [block_n, block_k]
    nargs["out_desc"].block_shape = [block_m, block_n]


_LINEAR_TMA_PIPE_WS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "PIPE_CAPACITY": 2}, num_warps=4,
                  num_stages=1, pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "PIPE_CAPACITY": 2}, num_warps=4,
                  num_stages=1, pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "PIPE_CAPACITY": 2}, num_warps=4,
                  num_stages=1, pre_hook=_linear_tma_pre_hook),
]


_LINEAR_TMA_DIRECT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
]


_GATE_UP_SILU_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]


_GATE_UP_SILU_TMA_PIPE_WS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "PIPE_CAPACITY": 2}, num_warps=4,
                  num_stages=1, pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "PIPE_CAPACITY": 2}, num_warps=4,
                  num_stages=1, pre_hook=_linear_tma_pre_hook),
]


_GATE_UP_SILU_TMA_DIRECT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3,
                  pre_hook=_linear_tma_pre_hook),
]


def _is_tma_compatible_2d(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 2 or tensor.stride(-1) != 1:
        return False
    elem_bytes = tensor.element_size()
    return all((stride * elem_bytes) % 16 == 0 for stride in tensor.stride()[:-1])


def _supports_host_tma(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> bool:
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 9:
        return False
    return _is_tma_compatible_2d(x) and _is_tma_compatible_2d(weight) and _is_tma_compatible_2d(out)


def _use_tma_pipe_ws(m: int, n: int, k: int) -> bool:
    return m >= 64 and n >= 64 and k >= 64


@triton.jit
def _linear_kernel(
    x,
    weight,
    bias,
    residual,
    input_scale,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    a_cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    b_rows = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
    b_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
    a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (a_rows, a_cols))
    b_smem_ptrs = tle.gpu.local_ptr(b_smem, (b_rows, b_cols))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        a_ptrs = x + offs_m[:, None] * K + k_idx[None, :]
        b_ptrs = weight + offs_n[None, :] * K + k_idx[:, None]
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_idx[:, None] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        tl.store(a_smem_ptrs, a_tile, mask=a_mask)
        tl.store(b_smem_ptrs, b_tile, mask=b_mask)
        a_tile = tl.load(a_smem_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_smem_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a_tile, b_tile, acc, out_dtype=tl.float32)

    if HAS_INPUT_SCALE:
        scale = tl.load(input_scale + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        acc *= scale[:, None]

    if HAS_BIAS:
        bias_vals = tl.load(bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    if HAS_RESIDUAL:
        residual_ptrs = residual + offs_m[:, None] * N + offs_n[None, :]
        residual_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        acc += tl.load(residual_ptrs, mask=residual_mask, other=0.0).to(tl.float32)

    out_ptrs = out + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out.dtype.element_ty), mask=out_mask)


@triton.jit
def _linear_tma_direct_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias,
    residual,
    input_scale,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        a_tile = x_desc.load([offs_m, k0])
        b_tile = weight_desc.load([offs_n, k0])
        acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)

    if HAS_INPUT_SCALE:
        offs = offs_m + tl.arange(0, BLOCK_M)
        scale = tl.load(input_scale + offs, mask=offs < M, other=0.0).to(tl.float32)
        acc *= scale[:, None]

    if HAS_BIAS:
        offs = offs_n + tl.arange(0, BLOCK_N)
        bias_vals = tl.load(bias + offs, mask=offs < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    if HAS_RESIDUAL:
        offs_m_vec = offs_m + tl.arange(0, BLOCK_M)
        offs_n_vec = offs_n + tl.arange(0, BLOCK_N)
        residual_ptrs = residual + offs_m_vec[:, None] * N + offs_n_vec[None, :]
        residual_mask = (offs_m_vec[:, None] < M) & (offs_n_vec[None, :] < N)
        acc += tl.load(residual_ptrs, mask=residual_mask, other=0.0).to(tl.float32)

    out_desc.store([offs_m, offs_n], acc.to(out_desc.dtype))


@triton.jit
def _linear_tma_pipe_consumer(
    ab_reader,
    out_desc,
    bias,
    residual,
    input_scale,
    pid_m,
    pid_n,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in tl.range(k_tiles):
        wait = ab_reader.wait(kt)
        slot = wait.slot
        a_tile = tl.load(tle.gpu.local_ptr(slot.a))
        b_tile = tl.load(tle.gpu.local_ptr(slot.b))
        acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32)
        ab_reader.release(kt)

    if HAS_INPUT_SCALE:
        offs = offs_m + tl.arange(0, BLOCK_M)
        scale = tl.load(input_scale + offs, mask=offs < M, other=0.0).to(tl.float32)
        acc *= scale[:, None]

    if HAS_BIAS:
        offs = offs_n + tl.arange(0, BLOCK_N)
        bias_vals = tl.load(bias + offs, mask=offs < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    if HAS_RESIDUAL:
        offs_m_vec = offs_m + tl.arange(0, BLOCK_M)
        offs_n_vec = offs_n + tl.arange(0, BLOCK_N)
        residual_ptrs = residual + offs_m_vec[:, None] * N + offs_n_vec[None, :]
        residual_mask = (offs_m_vec[:, None] < M) & (offs_n_vec[None, :] < N)
        acc += tl.load(residual_ptrs, mask=residual_mask, other=0.0).to(tl.float32)

    out_desc.store([offs_m, offs_n], acc.to(out_desc.dtype))


@triton.jit
def _linear_tma_pipe_producer(
    ab_writer,
    x_desc,
    weight_desc,
    pid_m,
    pid_n,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)

    for kt in tl.range(k_tiles):
        slot = ab_writer.acquire(kt)
        k0 = kt * BLOCK_K
        tle.gpu.copy(x_desc, slot.a, [BLOCK_M, BLOCK_K], [offs_m, k0])
        tle.gpu.copy(weight_desc, slot.b, [BLOCK_N, BLOCK_K], [offs_n, k0])
        ab_writer.commit(kt)


@triton.jit
def _linear_tma_pipe_ws_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias,
    residual,
    input_scale,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_M, BLOCK_K], dtype=x_desc.dtype, layout=None, scope=tle.gpu.smem)
    b_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_N, BLOCK_K], dtype=weight_desc.dtype, layout=None,
                           scope=tle.gpu.smem)
    ab_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="linear_ab", a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _linear_tma_pipe_consumer,
                (
                    ab_pipe.reader(),
                    out_desc,
                    bias,
                    residual,
                    input_scale,
                    pid_m,
                    pid_n,
                    M,
                    N,
                    K,
                    HAS_BIAS,
                    HAS_RESIDUAL,
                    HAS_INPUT_SCALE,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
            (
                _linear_tma_pipe_producer,
                (
                    ab_pipe.writer(),
                    x_desc,
                    weight_desc,
                    pid_m,
                    pid_n,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        [1],
        [48],
    )


@triton.jit
def _gate_up_silu_kernel(
    x,
    weight,
    bias,
    input_scale,
    out,
    M,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    a_cols = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    b_rows = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
    b_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
    a_smem = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    gate_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                              nv_mma_shared_layout=False)
    up_smem = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=False)
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (a_rows, a_cols))
    gate_smem_ptrs = tle.gpu.local_ptr(gate_smem, (b_rows, b_cols))
    up_smem_ptrs = tle.gpu.local_ptr(up_smem, (b_rows, b_cols))

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        a_ptrs = x + offs_m[:, None] * K + k_idx[None, :]
        gate_ptrs = weight + offs_n[None, :] * K + k_idx[:, None]
        up_ptrs = weight + (INTERMEDIATE + offs_n[None, :]) * K + k_idx[:, None]
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (offs_n[None, :] < INTERMEDIATE) & (k_idx[:, None] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        gate_tile = tl.load(gate_ptrs, mask=b_mask, other=0.0)
        up_tile = tl.load(up_ptrs, mask=b_mask, other=0.0)
        tl.store(a_smem_ptrs, a_tile, mask=a_mask)
        tl.store(gate_smem_ptrs, gate_tile, mask=b_mask)
        tl.store(up_smem_ptrs, up_tile, mask=b_mask)
        a_tile = tl.load(a_smem_ptrs, mask=a_mask, other=0.0)
        gate_tile = tl.load(gate_smem_ptrs, mask=b_mask, other=0.0)
        up_tile = tl.load(up_smem_ptrs, mask=b_mask, other=0.0)
        gate_acc = tl.dot(a_tile, gate_tile, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a_tile, up_tile, up_acc, out_dtype=tl.float32)

    if HAS_INPUT_SCALE:
        scale = tl.load(input_scale + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        gate_acc *= scale[:, None]
        up_acc *= scale[:, None]

    if HAS_BIAS:
        gate_bias = tl.load(bias + offs_n, mask=offs_n < INTERMEDIATE, other=0.0).to(tl.float32)
        up_bias = tl.load(bias + INTERMEDIATE + offs_n, mask=offs_n < INTERMEDIATE, other=0.0).to(tl.float32)
        gate_acc += gate_bias[None, :]
        up_acc += up_bias[None, :]

    out_vals = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    out_ptrs = out + offs_m[:, None] * INTERMEDIATE + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < INTERMEDIATE)
    tl.store(out_ptrs, out_vals.to(out.dtype.element_ty), mask=out_mask)


@triton.jit
def _gate_up_silu_tma_direct_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias,
    input_scale,
    M,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        a_tile = x_desc.load([offs_m, k0])
        gate_tile = weight_desc.load([offs_n, k0])
        up_tile = weight_desc.load([offs_n + INTERMEDIATE, k0])
        gate_acc = tl.dot(a_tile, gate_tile.T, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a_tile, up_tile.T, up_acc, out_dtype=tl.float32)

    if HAS_INPUT_SCALE:
        offs = offs_m + tl.arange(0, BLOCK_M)
        scale = tl.load(input_scale + offs, mask=offs < M, other=0.0).to(tl.float32)
        gate_acc *= scale[:, None]
        up_acc *= scale[:, None]

    if HAS_BIAS:
        offs = offs_n + tl.arange(0, BLOCK_N)
        gate_bias = tl.load(bias + offs, mask=offs < INTERMEDIATE, other=0.0).to(tl.float32)
        up_bias = tl.load(bias + INTERMEDIATE + offs, mask=offs < INTERMEDIATE, other=0.0).to(tl.float32)
        gate_acc += gate_bias[None, :]
        up_acc += up_bias[None, :]

    out_vals = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    out_desc.store([offs_m, offs_n], out_vals.to(out_desc.dtype))


@triton.jit
def _gate_up_silu_tma_pipe_consumer(
    ab_reader,
    out_desc,
    bias,
    input_scale,
    pid_m,
    pid_n,
    M,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in tl.range(k_tiles):
        wait = ab_reader.wait(kt)
        slot = wait.slot
        a_tile = tl.load(tle.gpu.local_ptr(slot.a))
        gate_tile = tl.load(tle.gpu.local_ptr(slot.gate))
        up_tile = tl.load(tle.gpu.local_ptr(slot.up))
        gate_acc = tl.dot(a_tile, gate_tile.T, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a_tile, up_tile.T, up_acc, out_dtype=tl.float32)
        ab_reader.release(kt)

    if HAS_INPUT_SCALE:
        offs = offs_m + tl.arange(0, BLOCK_M)
        scale = tl.load(input_scale + offs, mask=offs < M, other=0.0).to(tl.float32)
        gate_acc *= scale[:, None]
        up_acc *= scale[:, None]

    if HAS_BIAS:
        offs = offs_n + tl.arange(0, BLOCK_N)
        gate_bias = tl.load(bias + offs, mask=offs < INTERMEDIATE, other=0.0).to(tl.float32)
        up_bias = tl.load(bias + INTERMEDIATE + offs, mask=offs < INTERMEDIATE, other=0.0).to(tl.float32)
        gate_acc += gate_bias[None, :]
        up_acc += up_bias[None, :]

    out_vals = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    out_desc.store([offs_m, offs_n], out_vals.to(out_desc.dtype))


@triton.jit
def _gate_up_silu_tma_pipe_producer(
    ab_writer,
    x_desc,
    weight_desc,
    pid_m,
    pid_n,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_K)

    for kt in tl.range(k_tiles):
        slot = ab_writer.acquire(kt)
        k0 = kt * BLOCK_K
        tle.gpu.copy(x_desc, slot.a, [BLOCK_M, BLOCK_K], [offs_m, k0])
        tle.gpu.copy(weight_desc, slot.gate, [BLOCK_N, BLOCK_K], [offs_n, k0])
        tle.gpu.copy(weight_desc, slot.up, [BLOCK_N, BLOCK_K], [offs_n + INTERMEDIATE, k0])
        ab_writer.commit(kt)


@triton.jit
def _gate_up_silu_tma_pipe_ws_kernel(
    x_desc,
    weight_desc,
    out_desc,
    bias,
    input_scale,
    M,
    INTERMEDIATE: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INPUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_M, BLOCK_K], dtype=x_desc.dtype, layout=None, scope=tle.gpu.smem)
    gate_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_N, BLOCK_K], dtype=weight_desc.dtype, layout=None,
                              scope=tle.gpu.smem)
    up_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_N, BLOCK_K], dtype=weight_desc.dtype, layout=None,
                            scope=tle.gpu.smem)
    ab_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="gate_up_ab", a=a_smem, gate=gate_smem, up=up_smem)
    tle.gpu.warp_specialize(
        [
            (
                _gate_up_silu_tma_pipe_consumer,
                (
                    ab_pipe.reader(),
                    out_desc,
                    bias,
                    input_scale,
                    pid_m,
                    pid_n,
                    M,
                    INTERMEDIATE,
                    K,
                    HAS_BIAS,
                    HAS_INPUT_SCALE,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
            (
                _gate_up_silu_tma_pipe_producer,
                (
                    ab_pipe.writer(),
                    x_desc,
                    weight_desc,
                    pid_m,
                    pid_n,
                    INTERMEDIATE,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
            ),
        ],
        [1],
        [48],
    )


_linear_kernel_autotuned = triton.autotune(
    configs=_LINEAR_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS", "HAS_RESIDUAL", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _linear_prune_configs},
    cache_results=True,
)(_linear_kernel)


_linear_tma_direct_kernel_autotuned = triton.autotune(
    configs=_LINEAR_TMA_DIRECT_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS", "HAS_RESIDUAL", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _linear_prune_configs},
    cache_results=True,
)(_linear_tma_direct_kernel)


_linear_tma_pipe_ws_kernel_autotuned = triton.autotune(
    configs=_LINEAR_TMA_PIPE_WS_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "HAS_BIAS", "HAS_RESIDUAL", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _linear_prune_configs},
    cache_results=True,
)(_linear_tma_pipe_ws_kernel)


_gate_up_silu_kernel_autotuned = triton.autotune(
    configs=_GATE_UP_SILU_AUTOTUNE_CONFIGS,
    key=["M", "INTERMEDIATE", "K", "HAS_BIAS", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _gate_up_silu_prune_configs},
    cache_results=True,
)(_gate_up_silu_kernel)


_gate_up_silu_tma_direct_kernel_autotuned = triton.autotune(
    configs=_GATE_UP_SILU_TMA_DIRECT_AUTOTUNE_CONFIGS,
    key=["M", "INTERMEDIATE", "K", "HAS_BIAS", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _gate_up_silu_prune_configs},
    cache_results=True,
)(_gate_up_silu_tma_direct_kernel)


_gate_up_silu_tma_pipe_ws_kernel_autotuned = triton.autotune(
    configs=_GATE_UP_SILU_TMA_PIPE_WS_AUTOTUNE_CONFIGS,
    key=["M", "INTERMEDIATE", "K", "HAS_BIAS", "HAS_INPUT_SCALE"],
    prune_configs_by={"early_config_prune": _gate_up_silu_prune_configs},
    cache_results=True,
)(_gate_up_silu_tma_pipe_ws_kernel)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    residual: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` for contiguous BF16 tensors."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("TLE linear tutorial kernel currently expects bfloat16 inputs and weights")
    m, k = x.shape
    n, wk = weight.shape
    if k != wk:
        raise ValueError(f"linear dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != n:
            raise ValueError(f"bias size {bias.numel()} does not match output size {n}")
    if residual is not None:
        require_cuda_contiguous("residual", residual)
        if residual.shape != (m, n):
            raise ValueError(f"residual shape {tuple(residual.shape)} does not match output shape {(m, n)}")
        if residual.dtype != x.dtype:
            raise ValueError(f"residual dtype {residual.dtype} does not match input dtype {x.dtype}")
    if input_scale is not None:
        require_cuda_contiguous("input_scale", input_scale)
        if input_scale.dim() != 1 or input_scale.numel() != m:
            raise ValueError(f"input_scale shape {tuple(input_scale.shape)} does not match rows={m}")
        if input_scale.dtype != torch.float32:
            raise ValueError(f"input_scale must be float32, got {input_scale.dtype}")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    residual_arg = residual if residual is not None else x
    input_scale_arg = input_scale if input_scale is not None else x
    if block_m is not None or block_n is not None or block_k is not None:
        bm = default_block_m(m) if block_m is None else block_m
        bn = 64 if block_n is None else block_n
        bk = 64 if block_k is None else block_k
        grid = (cdiv(m, bm), cdiv(n, bn))
        _linear_kernel[grid](
            x,
            weight,
            bias if bias is not None else x,
            residual_arg,
            input_scale_arg,
            out,
            m,
            n,
            k,
            bias is not None,
            residual is not None,
            input_scale is not None,
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=3,
        )
    elif _supports_host_tma(x, weight, out):
        from triton.tools.tensor_descriptor import TensorDescriptor

        dummy_block = [1, 1]
        x_desc = TensorDescriptor.from_tensor(x, dummy_block)
        weight_desc = TensorDescriptor.from_tensor(weight, dummy_block)
        out_desc = TensorDescriptor.from_tensor(out, dummy_block)
        grid = lambda meta: (cdiv(m, meta["BLOCK_M"]), cdiv(n, meta["BLOCK_N"]))
        if _use_tma_pipe_ws(m, n, k):
            _linear_tma_pipe_ws_kernel_autotuned[grid](
                x_desc,
                weight_desc,
                out_desc,
                bias if bias is not None else x,
                residual_arg,
                input_scale_arg,
                M=m,
                N=n,
                K=k,
                HAS_BIAS=bias is not None,
                HAS_RESIDUAL=residual is not None,
                HAS_INPUT_SCALE=input_scale is not None,
            )
        else:
            _linear_tma_direct_kernel_autotuned[grid](
                x_desc,
                weight_desc,
                out_desc,
                bias if bias is not None else x,
                residual_arg,
                input_scale_arg,
                M=m,
                N=n,
                K=k,
                HAS_BIAS=bias is not None,
                HAS_RESIDUAL=residual is not None,
                HAS_INPUT_SCALE=input_scale is not None,
            )
    else:
        grid = lambda meta: (cdiv(m, meta["BLOCK_M"]), cdiv(n, meta["BLOCK_N"]))
        _linear_kernel_autotuned[grid](
            x,
            weight,
            bias if bias is not None else x,
            residual_arg,
            input_scale_arg,
            out,
            M=m,
            N=n,
            K=k,
            HAS_BIAS=bias is not None,
            HAS_RESIDUAL=residual is not None,
            HAS_INPUT_SCALE=input_scale is not None,
        )
    return out


def linear_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias + residual`` in the linear epilogue."""
    return linear(x, weight, bias, residual=residual)


def qkv_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    q_dim: int,
    kv_dim: int,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute packed QKV projection and return split Q, K, V views."""
    qkv = linear(x, weight, bias, input_scale=input_scale)
    q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    return q, k, v


def gate_up_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    intermediate_size: int,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``silu(gate(x)) * up(x)`` from a packed ``[gate, up]`` weight."""
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("weight", weight)
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f"x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("TLE fused gate/up tutorial kernel currently expects bfloat16 inputs and weights")
    m, k = x.shape
    rows, wk = weight.shape
    if k != wk:
        raise ValueError(f"gate_up_silu dimension mismatch: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    if rows != 2 * intermediate_size:
        raise ValueError(f"packed gate/up weight rows {rows} do not match 2 * intermediate_size={intermediate_size}")
    if bias is not None:
        require_cuda_contiguous("bias", bias)
        if bias.numel() != rows:
            raise ValueError(f"bias size {bias.numel()} does not match packed gate/up size {rows}")
    if input_scale is not None:
        require_cuda_contiguous("input_scale", input_scale)
        if input_scale.dim() != 1 or input_scale.numel() != m:
            raise ValueError(f"input_scale shape {tuple(input_scale.shape)} does not match rows={m}")
        if input_scale.dtype != torch.float32:
            raise ValueError(f"input_scale must be float32, got {input_scale.dtype}")

    out = torch.empty((m, intermediate_size), device=x.device, dtype=x.dtype)
    input_scale_arg = input_scale if input_scale is not None else x
    if _supports_host_tma(x, weight, out):
        from triton.tools.tensor_descriptor import TensorDescriptor

        dummy_block = [1, 1]
        x_desc = TensorDescriptor.from_tensor(x, dummy_block)
        weight_desc = TensorDescriptor.from_tensor(weight, dummy_block)
        out_desc = TensorDescriptor.from_tensor(out, dummy_block)
        grid = lambda meta: (cdiv(m, meta["BLOCK_M"]), cdiv(intermediate_size, meta["BLOCK_N"]))
        if _use_tma_pipe_ws(m, intermediate_size, k):
            _gate_up_silu_tma_pipe_ws_kernel_autotuned[grid](
                x_desc,
                weight_desc,
                out_desc,
                bias if bias is not None else x,
                input_scale_arg,
                M=m,
                INTERMEDIATE=intermediate_size,
                K=k,
                HAS_BIAS=bias is not None,
                HAS_INPUT_SCALE=input_scale is not None,
            )
        else:
            _gate_up_silu_tma_direct_kernel_autotuned[grid](
                x_desc,
                weight_desc,
                out_desc,
                bias if bias is not None else x,
                input_scale_arg,
                M=m,
                INTERMEDIATE=intermediate_size,
                K=k,
                HAS_BIAS=bias is not None,
                HAS_INPUT_SCALE=input_scale is not None,
            )
    else:
        grid = lambda meta: (cdiv(m, meta["BLOCK_M"]), cdiv(intermediate_size, meta["BLOCK_N"]))
        _gate_up_silu_kernel_autotuned[grid](
            x,
            weight,
            bias if bias is not None else x,
            input_scale_arg,
            out,
            M=m,
            INTERMEDIATE=intermediate_size,
            K=k,
            HAS_BIAS=bias is not None,
            HAS_INPUT_SCALE=input_scale is not None,
        )
    return out
