"""Token embedding kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from ._common import next_power_of_2, require_cuda_contiguous


_EMBEDDING_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]


def _supports_embedding_host_tma(input_ids: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> bool:
    if not weight.is_cuda or torch.cuda.get_device_capability(weight.device)[0] < 9:
        return False
    if input_ids.numel() == 0 or weight.dim() != 2 or output.dim() != 2:
        return False
    elem_bytes = weight.element_size()
    return (weight.is_contiguous() and output.is_contiguous() and (weight.stride(0) * elem_bytes) % 16 == 0
            and (output.stride(0) * output.element_size()) % 16 == 0)


@triton.jit
def _embedding_kernel(
    input_ids,
    weight,
    output,
    NUM_TOKENS,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    token_id = tl.load(input_ids + token_idx).to(tl.int64)
    vals = tl.load(weight + token_id * HIDDEN + cols, mask=mask, other=0.0)

    smem = tle.gpu.alloc([1, BLOCK_H], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                         nv_mma_shared_layout=False)
    stage = tl.zeros([BLOCK_H], dtype=tl.int32)
    ptrs = tle.gpu.local_ptr(smem, (stage, cols))
    tl.store(ptrs, vals, mask=mask)
    vals = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(output + token_idx * HIDDEN + cols, vals, mask=mask)


@triton.jit
def _embedding_tma_kernel(
    input_ids,
    weight_desc,
    output_desc,
    NUM_TOKENS,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_id = tl.load(input_ids + token_idx).to(tl.int32)
    vals = weight_desc.load([token_id, 0])
    output_desc.store([token_idx, 0], vals)


_embedding_kernel_autotuned = triton.autotune(
    configs=_EMBEDDING_AUTOTUNE_CONFIGS,
    key=["NUM_TOKENS", "HIDDEN"],
    cache_results=True,
)(_embedding_kernel)

_embedding_tma_kernel_autotuned = triton.autotune(
    configs=_EMBEDDING_AUTOTUNE_CONFIGS,
    key=["NUM_TOKENS", "HIDDEN"],
    cache_results=True,
)(_embedding_tma_kernel)


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return embedding(input_ids) as a flattened ``[num_tokens, hidden]`` tensor."""
    require_cuda_contiguous("input_ids", input_ids)
    require_cuda_contiguous("weight", weight)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.to(torch.long)
    hidden = weight.shape[1]
    output = torch.empty((input_ids.numel(), hidden), device=weight.device, dtype=weight.dtype)
    block_h = next_power_of_2(hidden)
    if _supports_embedding_host_tma(input_ids, weight, output):
        from triton.tools.tensor_descriptor import TensorDescriptor

        weight_desc = TensorDescriptor(weight, shape=list(weight.shape), strides=list(weight.stride()),
                                       block_shape=[1, block_h])
        output_desc = TensorDescriptor(output, shape=list(output.shape), strides=list(output.stride()),
                                       block_shape=[1, block_h])
        _embedding_tma_kernel_autotuned[(input_ids.numel(), )](
            input_ids,
            weight_desc,
            output_desc,
            NUM_TOKENS=input_ids.numel(),
            HIDDEN=hidden,
            BLOCK_H=block_h,
        )
    else:
        _embedding_kernel_autotuned[(input_ids.numel(), )](
            input_ids,
            weight,
            output,
            NUM_TOKENS=input_ids.numel(),
            HIDDEN=hidden,
            BLOCK_H=block_h,
        )
    return output
