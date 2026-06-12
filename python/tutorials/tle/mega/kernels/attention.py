"""Reference TLE attention kernels for prefill and decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from ._common import next_power_of_2, require_cuda_contiguous


def _attention_prune_configs(configs, named_args, **kwargs):
    kv_len = int(kwargs["KV_LEN"])
    pruned = []
    for config in configs:
        block_n = config.kwargs["BLOCK_N"]
        if kv_len <= 64 and block_n > 64:
            continue
        pruned.append(config)
    return pruned or configs[:1]


_ATTENTION_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
]


@triton.jit
def _attention_kernel(
    q,
    k_cache,
    v_cache,
    out,
    TOKENS,
    Q_LEN,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = token // Q_LEN
    q_pos = token - batch * Q_LEN
    q_global_pos = START_POS + q_pos
    group = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // group
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    q_vals = tl.load(q + (token * NUM_Q_HEADS + q_head) * HEAD_DIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    k_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    v_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    k_smem_ptrs = tle.gpu.local_ptr(k_smem, (rows, cols))
    v_smem_ptrs = tle.gpu.local_ptr(v_smem, (rows, cols))

    for n0 in tl.range(0, KV_LEN, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        key_mask = (offs_n < KV_LEN) & (offs_n <= q_global_pos)
        cache_offsets = ((batch * MAX_SEQ_LEN + offs_n[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[
            None, :]
        k_vals = tl.load(k_cache + cache_offsets, mask=key_mask[:, None] & mask_d[None, :], other=0.0)
        v_vals = tl.load(v_cache + cache_offsets, mask=key_mask[:, None] & mask_d[None, :], other=0.0)
        tl.store(k_smem_ptrs, k_vals, mask=key_mask[:, None] & mask_d[None, :])
        tl.store(v_smem_ptrs, v_vals, mask=key_mask[:, None] & mask_d[None, :])
        k_vals = tl.load(k_smem_ptrs, mask=key_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v_vals = tl.load(v_smem_ptrs, mask=key_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.sum(k_vals * q_vals[None, :], axis=1) * SM_SCALE
        scores = tl.where(key_mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out_vals = acc / l_i
    tl.store(out + (token * NUM_Q_HEADS + q_head) * HEAD_DIM + offs_d, out_vals.to(out.dtype.element_ty), mask=mask_d)


_attention_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_AUTOTUNE_CONFIGS,
    key=["Q_LEN", "KV_LEN", "NUM_Q_HEADS", "NUM_KV_HEADS", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _attention_prune_configs},
    cache_results=True,
)(_attention_kernel)


def attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_n: int | None = None,
) -> torch.Tensor:
    """Causal GQA attention over ``k_cache``/``v_cache`` for prefill or decode."""
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [tokens, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if tokens != batch * q_len:
        raise ValueError(f"tokens={tokens} does not equal batch={batch} * q_len={q_len}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    out = torch.empty_like(q)
    block_d = next_power_of_2(head_dim)
    if block_n is not None:
        _attention_kernel[(tokens, num_q_heads)](
            q,
            k_cache,
            v_cache,
            out,
            tokens,
            q_len,
            max_seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            kv_len,
            sm_scale,
            block_n,
            block_d,
            num_warps=4,
        )
    else:
        _attention_kernel_autotuned[(tokens, num_q_heads)](
            q,
            k_cache,
            v_cache,
            out,
            TOKENS=tokens,
            Q_LEN=q_len,
            MAX_SEQ_LEN=max_seq_len,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            START_POS=start_pos,
            KV_LEN=kv_len,
            SM_SCALE=sm_scale,
            BLOCK_D=block_d,
        )
    return out
