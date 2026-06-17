"""Reference TLE attention kernels for prefill and decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from ._common import cdiv, mha_config, next_power_of_2, require_cuda_contiguous


def _attention_prune_configs(configs, named_args, **kwargs):
    kv_len = int(kwargs["KV_LEN"])
    pruned = []
    for config in configs:
        block_n = config.kwargs["BLOCK_N"]
        if kv_len <= 64 and block_n > 64:
            continue
        pruned.append(config)
    return pruned or configs[:1]


_ATTENTION_WS_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "PIPE_CAPACITY": 2}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPE_CAPACITY": 2}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_CAPACITY": 2}, num_warps=4, num_stages=1),
]


def _attention_decode_tma_pre_hook(nargs):
    if "q_desc" not in nargs:
        return
    block_h = nargs["BLOCK_H"]
    valid_block_h = nargs["VALID_BLOCK_H"]
    block_n = nargs["BLOCK_N"]
    half_block_d = nargs["BLOCK_D"] // 2
    nargs["q_desc"].block_shape = [1, block_h, half_block_d]
    nargs["k_desc"].block_shape = [1, 1, block_n, half_block_d]
    nargs["v_desc"].block_shape = [1, 1, block_n, half_block_d]
    nargs["out_desc"].block_shape = [1, valid_block_h, half_block_d]


_ATTENTION_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 64, "PIPELINE_STAGES": 2}, num_warps=4, num_stages=2,
                  pre_hook=_attention_decode_tma_pre_hook),
    triton.Config({"BLOCK_N": 128, "PIPELINE_STAGES": 2}, num_warps=4, num_stages=2,
                  pre_hook=_attention_decode_tma_pre_hook),
]


@triton.jit
def _attention_ws_producer(
    kv_writer,
    k_cache,
    v_cache,
    batch,
    q_head,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_LEN,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = q_head // group
    offs_d = tl.arange(0, BLOCK_D)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    mask_d = offs_d < HEAD_DIM

    for n0 in tl.range(0, KV_LEN, BLOCK_N):
        tile = n0 // BLOCK_N
        slot = kv_writer.acquire(tile)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        load_mask = (offs_n[:, None] < KV_LEN) & mask_d[None, :]
        cache_offsets = ((batch * MAX_SEQ_LEN + offs_n[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[
            None, :]
        k_vals = tl.load(k_cache + cache_offsets, mask=load_mask, other=0.0)
        v_vals = tl.load(v_cache + cache_offsets, mask=load_mask, other=0.0)
        tl.store(tle.gpu.local_ptr(slot.k, (rows, cols)), k_vals, mask=load_mask)
        tl.store(tle.gpu.local_ptr(slot.v, (rows, cols)), v_vals, mask=load_mask)
        kv_writer.commit(tile)


@triton.jit
def _attention_ws_consumer(
    kv_reader,
    q,
    q_smem,
    out,
    batch,
    q_head,
    m_block,
    Q_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    START_POS,
    KV_LEN,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    ROWS_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    offs_m = m_block * BLOCK_M + ROW_OFFSET + tl.arange(0, ROWS_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_rows = tl.broadcast_to(tl.arange(0, ROWS_M)[:, None], (ROWS_M, BLOCK_D))
    q_cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (ROWS_M, BLOCK_D))
    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    kv_cols = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    q_mask = offs_m < Q_LEN
    mask_d = offs_d < HEAD_DIM

    token_offsets = batch * Q_LEN + offs_m
    q_ptrs = q + (token_offsets[:, None] * NUM_Q_HEADS + q_head) * HEAD_DIM + offs_d[None, :]
    q_vals = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)
    q_smem_ptrs = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    tl.store(q_smem_ptrs, q_vals, mask=q_mask[:, None] & mask_d[None, :])
    q_vals = tl.load(q_smem_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0)

    acc = tl.zeros([ROWS_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([ROWS_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([ROWS_M], dtype=tl.float32)

    for n0 in tl.range(0, KV_LEN, BLOCK_N):
        tile = n0 // BLOCK_N
        wait = kv_reader.wait(tile)
        slot = wait.slot
        offs_n = n0 + tl.arange(0, BLOCK_N)
        key_mask = offs_n < KV_LEN
        tile_mask = key_mask[:, None] & mask_d[None, :]
        k_vals = tl.load(tle.gpu.local_ptr(slot.k, (kv_rows, kv_cols)), mask=tile_mask, other=0.0)
        v_vals = tl.load(tle.gpu.local_ptr(slot.v, (kv_rows, kv_cols)), mask=tile_mask, other=0.0)

        scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32) * SM_SCALE
        causal_mask = offs_n[None, :] <= (START_POS + offs_m[:, None])
        scores = tl.where(q_mask[:, None] & key_mask[None, :] & causal_mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new = tl.where(q_mask, m_new, 0.0)
        alpha = tl.where(q_mask, tl.exp(m_i - m_new), 0.0)
        p = tl.where(q_mask[:, None], tl.exp(scores - m_new[:, None]), 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        kv_reader.release(tile)

    out_vals = acc / l_i[:, None]
    out_ptrs = out + (token_offsets[:, None] * NUM_Q_HEADS + q_head) * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, out_vals.to(out.dtype.element_ty), mask=q_mask[:, None] & mask_d[None, :])


@triton.jit
def _attention_ws_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    m_block = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    HALF_M: tl.constexpr = BLOCK_M // 2
    q_smem_lo = tle.gpu.alloc([HALF_M, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                              nv_mma_shared_layout=False)
    q_smem_hi = tle.gpu.alloc([HALF_M, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                              nv_mma_shared_layout=False)
    k_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    v_smem = tle.gpu.alloc([PIPE_CAPACITY, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    kv_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="attention_kv", readers=("lo", "hi"), k=k_smem,
                       v=v_smem)
    tle.gpu.warp_specialize(
        [
            (
                _attention_ws_consumer,
                (
                    kv_pipe.reader(name="lo"),
                    q,
                    q_smem_lo,
                    out,
                    batch,
                    q_head,
                    m_block,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BLOCK_M,
                    0,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                ),
            ),
            (
                _attention_ws_consumer,
                (
                    kv_pipe.reader(name="hi"),
                    q,
                    q_smem_hi,
                    out,
                    batch,
                    q_head,
                    m_block,
                    Q_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    START_POS,
                    KV_LEN,
                    SM_SCALE,
                    BLOCK_M,
                    HALF_M,
                    HALF_M,
                    BLOCK_N,
                    BLOCK_D,
                ),
            ),
            (
                _attention_ws_producer,
                (
                    kv_pipe.writer(),
                    k_cache,
                    v_cache,
                    batch,
                    q_head,
                    MAX_SEQ_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    KV_LEN,
                    BLOCK_N,
                    BLOCK_D,
                ),
            ),
        ],
        [4, 1],
        [240, 48],
    )


_attention_ws_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_WS_AUTOTUNE_CONFIGS,
    key=["Q_LEN", "KV_LEN", "NUM_Q_HEADS", "NUM_KV_HEADS", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _attention_prune_configs},
    cache_results=True,
)(_attention_ws_kernel)


@triton.jit
def _attention_decode_no_split_producer(
    q_writer,
    k_writer,
    v_writer,
    q_desc,
    k_desc,
    v_desc,
    REAL_KV_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    HALF_D: tl.constexpr = BLOCK_D // 2
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    head_start = hid * VALID_BLOCK_H

    q_slot = q_writer.acquire(0)
    q_lo = q_slot.q.subslice([0, 0], [BLOCK_H, HALF_D])
    q_hi = q_slot.q.subslice([0, HALF_D], [BLOCK_H, HALF_D])
    tle.gpu.copy(q_desc, q_lo, [1, BLOCK_H, HALF_D], [bid, head_start, 0])
    tle.gpu.copy(q_desc, q_hi, [1, BLOCK_H, HALF_D], [bid, head_start, HALF_D])
    q_writer.commit(0)

    for n0 in tl.range(0, REAL_KV_LEN, BLOCK_N):
        tile = n0 // BLOCK_N
        k_slot = k_writer.acquire(tile)
        k_lo = k_slot.k.subslice([0, 0, 0, 0], [1, 1, BLOCK_N, HALF_D])
        k_hi = k_slot.k.subslice([0, 0, 0, HALF_D], [1, 1, BLOCK_N, HALF_D])
        tle.gpu.copy(k_desc, k_lo, [1, 1, BLOCK_N, HALF_D], [bid, kv_head, n0, 0])
        tle.gpu.copy(k_desc, k_hi, [1, 1, BLOCK_N, HALF_D], [bid, kv_head, n0, HALF_D])
        k_writer.commit(tile)
        v_slot = v_writer.acquire(tile)
        v_lo = v_slot.v.subslice([0, 0, 0, 0], [1, 1, BLOCK_N, HALF_D])
        v_hi = v_slot.v.subslice([0, 0, 0, HALF_D], [1, 1, BLOCK_N, HALF_D])
        tle.gpu.copy(v_desc, v_lo, [1, 1, BLOCK_N, HALF_D], [bid, kv_head, n0, 0])
        tle.gpu.copy(v_desc, v_hi, [1, 1, BLOCK_N, HALF_D], [bid, kv_head, n0, HALF_D])
        v_writer.commit(tile)


@triton.jit
def _attention_decode_no_split_consumer(
    q_reader,
    k_reader,
    v_reader,
    out_desc,
    o_smem,
    REAL_KV_LEN,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    HALF_D: tl.constexpr = BLOCK_D // 2
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    head_start = hid * VALID_BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    rows_h = tl.broadcast_to(tl.arange(0, BLOCK_H)[:, None], (BLOCK_H, BLOCK_D))
    rows_n = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols_h = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_H, BLOCK_D))
    cols_n = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    head_mask = offs_h < NUM_Q_HEADS
    store_rows = tl.arange(0, BLOCK_H) < VALID_BLOCK_H
    dim_mask = offs_d < HEAD_DIM
    zero = tl.full((), 0, tl.int32)

    q_wait = q_reader.wait(0)
    q_smem = q_wait.slot.q
    q_smem_ptrs = tle.gpu.local_ptr(q_smem, (rows_h, cols_h))
    q_vals = tl.load(q_smem_ptrs)

    acc_o = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    scores_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    logsum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for n0 in tl.range(0, REAL_KV_LEN, BLOCK_N, num_stages=PIPELINE_STAGES):
        n_idx = n0 + offs_n
        kv_mask = n_idx < REAL_KV_LEN
        tile = n0 // BLOCK_N
        k_wait = k_reader.wait(tile)
        k_smem = k_wait.slot.k.slot(zero).slot(zero)
        k_smem_ptrs = tle.gpu.local_ptr(k_smem, (rows_n, cols_n))
        k_vals = tl.load(k_smem_ptrs)

        scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))
        k_reader.release(tile)
        scores_max_prev = scores_max
        scores_max = tl.maximum(scores_max_prev, tl.max(scores, axis=1))
        scores_scale = tl.exp2((scores_max_prev - scores_max) * LOG2E_SCALE)
        probs = tl.exp2((scores - scores_max[:, None]) * LOG2E_SCALE)
        probs = tl.where(kv_mask[None, :], probs, 0.0)
        v_wait = v_reader.wait(tile)
        v_smem = v_wait.slot.v.slot(zero).slot(zero)
        v_smem_ptrs = tle.gpu.local_ptr(v_smem, (rows_n, cols_n))
        v_vals = tl.load(v_smem_ptrs)
        acc_o = acc_o * scores_scale[:, None] + tl.dot(probs.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
        logsum = logsum * scores_scale + tl.sum(probs, axis=1)
        v_reader.release(tile)

    out_vals = acc_o / logsum[:, None]
    o_smem_ptrs = tle.gpu.local_ptr(o_smem, (rows_h, cols_h))
    tl.store(
        o_smem_ptrs,
        out_vals.to(q_vals.dtype),
        mask=store_rows[:, None] & head_mask[:, None] & dim_mask[None, :],
    )
    o_lo = o_smem.subslice([0, 0], [VALID_BLOCK_H, HALF_D])
    o_hi = o_smem.subslice([0, HALF_D], [VALID_BLOCK_H, HALF_D])
    tle.gpu.copy(o_lo, out_desc, [1, VALID_BLOCK_H, HALF_D], [bid, head_start, 0])
    tle.gpu.copy(o_hi, out_desc, [1, VALID_BLOCK_H, HALF_D], [bid, head_start, HALF_D])
    q_reader.release(0)


@triton.jit
def _attention_decode_no_split_kernel(
    q_desc,
    k_desc,
    v_desc,
    out_desc,
    REAL_KV_LEN,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    q_smem = tle.gpu.alloc([1, BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([PIPELINE_STAGES, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([PIPELINE_STAGES, 1, 1, BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=True)
    o_smem = tle.gpu.alloc([BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    q_pipe = tle.pipe(capacity=1, scope="cta", name="decode_q", one_shot=True, q=q_smem)
    k_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="decode_k", k=k_smem)
    v_pipe = tle.pipe(capacity=PIPELINE_STAGES, scope="cta", name="decode_v", v=v_smem)
    tle.gpu.warp_specialize(
        [
            (
                _attention_decode_no_split_consumer,
                (
                    q_pipe.reader(),
                    k_pipe.reader(),
                    v_pipe.reader(),
                    out_desc,
                    o_smem,
                    REAL_KV_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    LOG2E_SCALE,
                    VALID_BLOCK_H,
                    BLOCK_H,
                    BLOCK_N,
                    BLOCK_D,
                    PIPELINE_STAGES,
                ),
            ),
            (
                _attention_decode_no_split_producer,
                (
                    q_pipe.writer(),
                    k_pipe.writer(),
                    v_pipe.writer(),
                    q_desc,
                    k_desc,
                    v_desc,
                    REAL_KV_LEN,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    VALID_BLOCK_H,
                    BLOCK_H,
                    BLOCK_N,
                    BLOCK_D,
                ),
            ),
        ],
        [4],
        [24],
    )


@triton.jit
def _attention_decode_split_kernel(
    q,
    k_cache,
    v_cache,
    partial_out,
    partial_lse,
    MAX_SEQ_LEN: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2E_SCALE: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    SID_OFFSET: tl.constexpr,
    VALID_BLOCK_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2) + SID_OFFSET
    kv_group_num: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head = hid // (kv_group_num // VALID_BLOCK_H)
    split_start = sid * SPLIT_BASE

    head_start = hid * VALID_BLOCK_H
    offs_h = head_start + tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    rows_h = tl.broadcast_to(tl.arange(0, BLOCK_H)[:, None], (BLOCK_H, BLOCK_D))
    rows_n = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, BLOCK_D))
    cols_h = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_H, BLOCK_D))
    cols_n = tl.broadcast_to(tl.arange(0, BLOCK_D)[None, :], (BLOCK_N, BLOCK_D))
    head_mask = offs_h < NUM_Q_HEADS
    store_rows = tl.arange(0, BLOCK_H) < VALID_BLOCK_H
    dim_mask = offs_d < HEAD_DIM

    q_smem = tle.gpu.alloc([BLOCK_H, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    k_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    v_smem = tle.gpu.alloc([BLOCK_N, BLOCK_D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)

    q_ptrs = q + (bid * NUM_Q_HEADS + offs_h[:, None]) * HEAD_DIM + offs_d[None, :]
    q_mask = head_mask[:, None] & dim_mask[None, :]
    q_vals = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q_smem_ptrs = tle.gpu.local_ptr(q_smem, (rows_h, cols_h))
    tl.store(q_smem_ptrs, q_vals)
    q_vals = tl.load(q_smem_ptrs)

    acc_o = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    scores_max = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    logsum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for local_n0 in tl.range(0, SPLIT_LEN, BLOCK_N, num_stages=PIPELINE_STAGES):
        n_idx = split_start + local_n0 + offs_n
        local_mask = (local_n0 + offs_n) < SPLIT_LEN
        kv_ptrs = ((bid * MAX_SEQ_LEN + n_idx[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[None, :]
        tile_mask = local_mask[:, None] & dim_mask[None, :]
        k_vals = tl.load(k_cache + kv_ptrs, mask=tile_mask, other=0.0)
        k_smem_ptrs = tle.gpu.local_ptr(k_smem, (rows_n, cols_n))
        tl.store(k_smem_ptrs, k_vals)
        k_vals = tl.load(k_smem_ptrs)

        scores = tl.dot(q_vals, tl.trans(k_vals), out_dtype=tl.float32)
        scores = tl.where(head_mask[:, None] & local_mask[None, :], scores, -float("inf"))
        scores_max_prev = scores_max
        scores_max = tl.maximum(scores_max_prev, tl.max(scores, axis=1))
        scores_scale = tl.exp2((scores_max_prev - scores_max) * LOG2E_SCALE)
        probs = tl.exp2((scores - scores_max[:, None]) * LOG2E_SCALE)
        probs = tl.where(head_mask[:, None] & local_mask[None, :], probs, 0.0)
        v_idx = split_start + local_n0 + offs_n
        v_local_mask = (local_n0 + offs_n) < SPLIT_LEN
        v_ptrs = ((bid * MAX_SEQ_LEN + v_idx[:, None]) * NUM_KV_HEADS + kv_head) * HEAD_DIM + offs_d[None, :]
        v_tile_mask = v_local_mask[:, None] & dim_mask[None, :]
        v_vals = tl.load(v_cache + v_ptrs, mask=v_tile_mask, other=0.0)
        v_smem_ptrs = tle.gpu.local_ptr(v_smem, (rows_n, cols_n))
        tl.store(v_smem_ptrs, v_vals, mask=v_tile_mask)
        v_vals = tl.load(v_smem_ptrs, mask=v_tile_mask, other=0.0)
        acc_o = acc_o * scores_scale[:, None] + tl.dot(probs.to(tl.bfloat16), v_vals, out_dtype=tl.float32)
        logsum = logsum * scores_scale + tl.sum(probs, axis=1)

    out_vals = acc_o / logsum[:, None]
    partial_ptrs = partial_out + (((bid * NUM_Q_HEADS + offs_h[:, None]) * NUM_SPLIT + sid) * HEAD_DIM + offs_d[
        None, :])
    tl.store(partial_ptrs, out_vals.to(partial_out.dtype.element_ty), mask=store_rows[:, None] & q_mask)
    lse = tl.log2(logsum) + scores_max * LOG2E_SCALE
    tl.store(partial_lse + (bid * NUM_Q_HEADS + offs_h) * NUM_SPLIT + sid, lse, mask=store_rows & head_mask)


@triton.jit
def _attention_decode_combine_kernel(
    partial_out,
    partial_lse,
    out,
    NUM_SPLIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    bid = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    lse_offsets = (bid * tl.num_programs(0) + head) * NUM_SPLIT + tl.arange(0, NUM_SPLIT)
    lse_vals = tl.load(partial_lse + lse_offsets).to(tl.float32)
    lse_max = tl.max(lse_vals, axis=0)
    weights_unnorm = tl.exp2(lse_vals - lse_max)
    lse_logsum = tl.log2(tl.sum(weights_unnorm, axis=0)) + lse_max
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for sid in tl.static_range(0, NUM_SPLIT):
        weight = tl.exp2(tl.load(partial_lse + (bid * tl.num_programs(0) + head) * NUM_SPLIT + sid).to(tl.float32) -
                         lse_logsum)
        vals = tl.load(partial_out + (((bid * tl.num_programs(0) + head) * NUM_SPLIT + sid) * HEAD_DIM + offs_d),
                       mask=mask_d, other=0.0).to(tl.float32)
        acc += vals * weight
    tl.store(out + (bid * tl.num_programs(0) + head) * HEAD_DIM + offs_d, acc.to(out.dtype.element_ty), mask=mask_d)


_attention_decode_no_split_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_DECODE_AUTOTUNE_CONFIGS,
    key=["NUM_Q_HEADS", "NUM_KV_HEADS", "HEAD_DIM", "VALID_BLOCK_H", "BLOCK_H"],
    cache_results=True,
)(_attention_decode_no_split_kernel)


_attention_decode_split_kernel_autotuned = triton.autotune(
    configs=_ATTENTION_DECODE_AUTOTUNE_CONFIGS,
    key=[
        "NUM_Q_HEADS",
        "NUM_KV_HEADS",
        "HEAD_DIM",
        "NUM_SPLIT",
        "SPLIT_BASE",
        "SPLIT_LEN",
        "SID_OFFSET",
        "VALID_BLOCK_H",
        "BLOCK_H",
    ],
    cache_results=True,
)(_attention_decode_split_kernel)


def attention_ws(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_n: int | None = None,
    block_m: int = 128,
) -> torch.Tensor:
    """Causal block-M GQA attention with TLE pipe and dot-based QK/PV stages."""
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [tokens, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("attention_ws currently expects bfloat16 q/k/v tensors")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if tokens != batch * q_len:
        raise ValueError(f"tokens={tokens} does not equal batch={batch} * q_len={q_len}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires q heads divisible by kv heads, got {num_q_heads} and {num_kv_heads}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    out = torch.empty_like(q)
    block_d = next_power_of_2(head_dim)
    if block_n is not None:
        _attention_ws_kernel[(cdiv(q_len, block_m), num_q_heads, batch)](
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
            block_m,
            block_n,
            block_d,
            2,
            num_warps=4,
            num_stages=1,
        )
    else:
        grid = lambda meta: (cdiv(q_len, meta["BLOCK_M"]), num_q_heads, batch)
        _attention_ws_kernel_autotuned[grid](
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


def attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    q_len: int,
    start_pos: int,
    kv_len: int,
    sm_scale: float,
    block_h: int = 64,
    block_n: int | None = None,
    num_split: int = 16,
) -> torch.Tensor:
    """TileOps-style GQA decode attention for a single query token.

    The short-context path computes one KV head group per CTA. The long-context
    path splits KV into ``num_split`` chunks, writes partial outputs and LSEs,
    then combines them with the same log-sum-exp weighting used by TileOps.
    """
    require_cuda_contiguous("q", q)
    require_cuda_contiguous("k_cache", k_cache)
    require_cuda_contiguous("v_cache", v_cache)
    if q_len != 1:
        raise ValueError(f"attention_decode expects q_len=1, got {q_len}")
    if q.dim() != 3 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("expected q [batch, q_heads, dim] and cache [batch, max, kv_heads, dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise ValueError("attention_decode currently expects bfloat16 q/k/v tensors")
    tokens, num_q_heads, head_dim = q.shape
    batch, max_seq_len, num_kv_heads, cache_dim = k_cache.shape
    if tokens != batch:
        raise ValueError(f"decode q tokens={tokens} must equal batch={batch}")
    if cache_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("KV cache shapes do not match query head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"GQA requires q heads divisible by kv heads, got {num_q_heads} and {num_kv_heads}")
    if start_pos + q_len > kv_len:
        raise ValueError(f"kv_len={kv_len} must include current query range start_pos={start_pos} q_len={q_len}")
    if kv_len <= 0:
        raise ValueError("attention_decode requires kv_len > 0")

    q_per_kv = num_q_heads // num_kv_heads
    compute_block_h = block_h
    valid_block_h = min(compute_block_h, q_per_kv)
    if q_per_kv % valid_block_h != 0:
        valid_block_h = q_per_kv
        compute_block_h = max(compute_block_h, valid_block_h)
    block_d = next_power_of_2(head_dim)
    out = torch.empty_like(q)
    log2e_scale = sm_scale * 1.4426950408889634
    selected_block_n = 128 if block_n is None else block_n
    if selected_block_n < 16:
        raise ValueError(f"attention_decode block_n must be >= 16 for tl.dot, got {selected_block_n}")
    split_threshold = num_split * selected_block_n

    if kv_len < split_threshold:
        grid = (batch, cdiv(num_q_heads, valid_block_h))
        half_block_d = block_d // 2
        q_desc = TensorDescriptor(
            q,
            shape=[batch, num_q_heads, head_dim],
            strides=[num_q_heads * head_dim, head_dim, 1],
            block_shape=[1, compute_block_h, half_block_d],
        )
        k_desc = TensorDescriptor(
            k_cache,
            shape=[batch, num_kv_heads, max_seq_len, head_dim],
            strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
            block_shape=[1, 1, selected_block_n, half_block_d],
        )
        v_desc = TensorDescriptor(
            v_cache,
            shape=[batch, num_kv_heads, max_seq_len, head_dim],
            strides=[max_seq_len * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1],
            block_shape=[1, 1, selected_block_n, half_block_d],
        )
        out_desc = TensorDescriptor(
            out,
            shape=[batch, num_q_heads, head_dim],
            strides=[num_q_heads * head_dim, head_dim, 1],
            block_shape=[1, valid_block_h, half_block_d],
        )
        if block_n is not None:
            _attention_decode_no_split_kernel[grid](
                q_desc,
                k_desc,
                v_desc,
                out_desc,
                kv_len,
                max_seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                log2e_scale,
                valid_block_h,
                compute_block_h,
                selected_block_n,
                block_d,
                2,
                num_warps=4,
                num_stages=2,
            )
        else:
            _attention_decode_no_split_kernel_autotuned[grid](
                q_desc,
                k_desc,
                v_desc,
                out_desc,
                REAL_KV_LEN=kv_len,
                MAX_SEQ_LEN=max_seq_len,
                NUM_Q_HEADS=num_q_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=head_dim,
                LOG2E_SCALE=log2e_scale,
                VALID_BLOCK_H=valid_block_h,
                BLOCK_H=compute_block_h,
                BLOCK_D=block_d,
            )
        return out

    partial_out = torch.empty((batch, num_q_heads, num_split, head_dim), device=q.device, dtype=q.dtype)
    partial_lse = torch.empty((batch, num_q_heads, num_split), device=q.device, dtype=torch.float32)
    split_base = (kv_len // (num_split * selected_block_n)) * selected_block_n
    last_split_len = kv_len - (num_split - 1) * split_base

    def _launch_split(split_count: int, sid_offset: int, split_len: int) -> None:
        if split_count <= 0:
            return
        split_grid = (batch, cdiv(num_q_heads, valid_block_h), split_count)
        if block_n is not None:
            _attention_decode_split_kernel[split_grid](
                q,
                k_cache,
                v_cache,
                partial_out,
                partial_lse,
                max_seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                log2e_scale,
                num_split,
                split_base,
                split_len,
                sid_offset,
                valid_block_h,
                compute_block_h,
                selected_block_n,
                block_d,
                2,
                num_warps=4,
                num_stages=2,
            )
        else:
            _attention_decode_split_kernel_autotuned[split_grid](
                q,
                k_cache,
                v_cache,
                partial_out,
                partial_lse,
                MAX_SEQ_LEN=max_seq_len,
                NUM_Q_HEADS=num_q_heads,
                NUM_KV_HEADS=num_kv_heads,
                HEAD_DIM=head_dim,
                LOG2E_SCALE=log2e_scale,
                NUM_SPLIT=num_split,
                SPLIT_BASE=split_base,
                SPLIT_LEN=split_len,
                SID_OFFSET=sid_offset,
                VALID_BLOCK_H=valid_block_h,
                BLOCK_H=compute_block_h,
                BLOCK_D=block_d,
            )

    if last_split_len == split_base:
        _launch_split(num_split, 0, split_base)
    else:
        _launch_split(num_split - 1, 0, split_base)
        _launch_split(1, num_split - 1, last_split_len)
    _attention_decode_combine_kernel[(num_q_heads, batch)](
        partial_out,
        partial_lse,
        out,
        NUM_SPLIT=num_split,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _flash_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_head_stride,
    q_dim_stride,
    k_dim_stride,
    v_dim_stride,
    o_dim_stride,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    total_q: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    is_causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    q_head = tl.program_id(2)
    group_size: tl.constexpr = num_q_heads // num_kv_heads
    kv_head = q_head // group_size
    offs_d = tl.arange(0, block_d)

    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
    k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
    q_len = q_eos - q_bos
    k_len = k_eos - k_bos

    offs_m = m_block * block_m + tl.arange(0, block_m)
    if m_block * block_m > q_len:
        return

    q_offsets = (
        (q_bos + offs_m[:, None]) * q_row_stride
        + q_head * q_head_stride
        + offs_d[None, :] * q_dim_stride
    )
    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

    acc = tl.zeros((block_m, block_d), dtype=tl.float32)
    row_max = tl.full((block_m, ), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((block_m, ), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    n_block_max = tl.cdiv(k_len, block_n)
    if is_causal:
        n_block_max = min(n_block_max, tl.cdiv((m_block + 1) * block_m + k_len - q_len, block_n))

    for n_block in tl.range(n_block_max - 1, -1, step=-1):
        offs_n = n_block * block_n + tl.arange(0, block_n)
        k_offsets = (
            (k_bos + offs_n[None, :]) * k_row_stride
            + kv_head * k_head_stride
            + offs_d[:, None] * k_dim_stride
        )
        v_offsets = (
            (k_bos + offs_n[:, None]) * v_row_stride
            + kv_head * v_head_stride
            + offs_d[None, :] * v_dim_stride
        )
        kv_mask = offs_n < k_len
        k = tl.load(k_ptr + k_offsets, mask=(offs_d[:, None] < head_dim) & kv_mask[None, :], other=0.0)
        scores = tl.dot(q, k, out_dtype=tl.float32)
        causal_limit = offs_m + k_len - q_len
        if is_causal:
            keep = kv_mask[None, :] & (offs_n[None, :] <= causal_limit[:, None]) & (offs_m[:, None] < q_len)
        else:
            keep = kv_mask[None, :] & (offs_m[:, None] < q_len)
        scores = tl.where(keep, scores * sm_scale * log2e, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        p = tl.exp2(scores - new_max[:, None])
        alpha = tl.exp2(row_max - new_max)
        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr + v_offsets, mask=kv_mask[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        row_max = new_max

    inv_sum = tl.where(row_sum == 0.0, 0.0, 1.0 / row_sum)
    out = (acc * inv_sum[:, None]).to(out_ptr.dtype.element_ty)
    out_offsets = (
        (q_bos + offs_m[:, None]) * o_row_stride
        + q_head * o_head_stride
        + offs_d[None, :] * o_dim_stride
    )
    tl.store(out_ptr + out_offsets, out, mask=q_mask)
    lse = tl.where(row_sum == 0.0, float("inf"), row_max / log2e + tl.log(row_sum))
    tl.store(softmax_lse_ptr + q_head * total_q + q_bos + offs_m, lse, mask=offs_m < q_len)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    q_v: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] | None = None,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
    out: torch.Tensor | None = None,
    scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: torch.Tensor | None = None,
    fa_version: int = 2,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if fa_version != 2:
        raise RuntimeError("Only FA2-compatible local varlen attention is implemented.")
    if cu_seqlens_k is None or seqused_k is not None:
        raise NotImplementedError("local varlen attention currently implements non-paged cu_seqlens_k inputs")
    if q_v is not None or dropout_p != 0.0 or return_attn_probs:
        raise NotImplementedError("dropout, q_v, and returned attention probabilities are not implemented")
    if window_size not in (None, (-1, -1)):
        raise NotImplementedError("sliding-window attention is not implemented")
    if softcap != 0.0 or alibi_slopes is not None:
        raise NotImplementedError("softcap and alibi are not implemented")
    if block_table is not None or num_splits > 0 or cp_world_size != 1 or cp_rank != 0:
        raise NotImplementedError("paged/split/context-parallel attention is not implemented")
    if any(x is not None for x in (scheduler_metadata, q_descale, k_descale, v_descale, s_aux, cp_tot_seqused_k)):
        raise NotImplementedError("FA3 auxiliary arguments are not implemented")
    del deterministic
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    max_seqlen_q = int(max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k)
    q_len = int(max_seqlen_q)
    batch_size = cu_seqlens_q.numel() - 1
    total_q = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    if out is None:
        out = torch.empty_like(q)
    softmax_lse = torch.empty((num_q_heads, total_q), device=q.device, dtype=torch.float32)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    block_m, block_n, num_warps, num_stages = mha_config(total_q // batch_size, num_q_heads, num_sms)
    block_d = triton.next_power_of_2(head_dim)
    _flash_varlen_fwd_kernel[(triton.cdiv(q_len, block_m), batch_size, num_q_heads)](
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        out.stride(-3),
        q.stride(-2),
        k.stride(-2),
        v.stride(-2),
        out.stride(-2),
        q.stride(-1),
        k.stride(-1),
        v.stride(-1),
        out.stride(-1),
        max_seqlen_q,
        max_seqlen_k,
        total_q,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        causal,
        block_m,
        block_n,
        block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (out, softmax_lse) if return_softmax_lse else out
