"""Compare local Qwen3 operators with vLLM operators."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))


def _install_tle_mega_graph_import_stub() -> None:
    """Keep this bench runnable while the host-side mega API is under construction."""
    python_root = Path(__file__).resolve().parents[4]
    graph_py = python_root / "triton" / "experimental" / "tle" / "mega" / "graph.py"
    module_name = "triton.experimental.tle.mega.graph"
    if graph_py.exists() or module_name in sys.modules:
        return

    @dataclass
    class FieldSpec:
        name: str
        value: Any = None

    @dataclass
    class GridSpec:
        name: str
        fields: list[Any] = field(default_factory=list)

    @dataclass
    class TaskMapSpec:
        grid: Any
        mapping: Any = None

    @dataclass
    class TaskSpec:
        name: str
        maps: list[Any] = field(default_factory=list)

    @dataclass
    class AffineMapSpec:
        expr: Any

    @dataclass
    class MegaGraph:
        grids: list[Any] = field(default_factory=list)
        tasks: list[Any] = field(default_factory=list)

        def add_grid(self, grid: Any) -> Any:
            self.grids.append(grid)
            return grid

        def add_task(self, task: Any) -> Any:
            self.tasks.append(task)
            return task

        def compile(self) -> None:
            raise NotImplementedError("TLE mega_graph scheduler codegen is not implemented yet")

    def affine_map(expr: Any) -> AffineMapSpec:
        return AffineMapSpec(expr)

    def mega_graph() -> MegaGraph:
        return MegaGraph()

    module = types.ModuleType(module_name)
    module.ALL = object()
    module.AffineMapSpec = AffineMapSpec
    module.FieldSpec = FieldSpec
    module.GridSpec = GridSpec
    module.MegaGraph = MegaGraph
    module.TaskMapSpec = TaskMapSpec
    module.TaskSpec = TaskSpec
    module.affine_map = affine_map
    module.mega_graph = mega_graph
    sys.modules[module_name] = module


_install_tle_mega_graph_import_stub()

attention_kernels = importlib.import_module("kernels.attention")
linear_kernels = importlib.import_module("kernels.linear")
norm_kernels = importlib.import_module("kernels.norm")
rotary_cache = importlib.import_module("kernels.rotary_cache")


QWEN3_32B_DEFAULTS = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1.0e-6,
    "rope_theta": 1_000_000.0,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Qwen3 ops vs vLLM ops for Qwen3-32B")
    parser.add_argument("--model-path", default="/data/dataset/Qwen3-32B")
    parser.add_argument("--output-dir", default="build/mega/qwen3-32b")
    parser.add_argument("--prefill-len", type=int, action="append", default=[], help="Prefill token count.")
    parser.add_argument("--decode-kv-len", type=int, action="append", default=[], help="Decode KV length.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-load-config", action="store_true")
    return parser


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) / max(iters, 1))


def _maybe_load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(QWEN3_32B_DEFAULTS)
    if args.no_load_config:
        cfg["source"] = "built-in defaults"
        return cfg
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
        for key in QWEN3_32B_DEFAULTS:
            value = getattr(hf_cfg, key, None)
            if value is not None:
                cfg[key] = value
        cfg["source"] = f"transformers AutoConfig from {args.model_path}"
    except Exception as exc:  # noqa: BLE001 - recorded in benchmark metadata.
        cfg["source"] = f"built-in defaults; AutoConfig failed: {type(exc).__name__}: {exc}"
    return cfg


def _rope_cache(max_seq_len: int, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device="cuda", dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _vllm_cos_sin_cache(max_seq_len: int, head_dim: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    cos, sin = _rope_cache(max_seq_len, head_dim, theta, dtype)
    return torch.cat((cos, sin), dim=-1).contiguous()


def _torch_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight.t())


def _make_row(
    *,
    kind: str,
    scenario: str,
    op: str,
    tokens: int,
    shape: str,
    ours_kernel: str,
    vllm_kernel: str,
    ours_ms: float | None,
    vllm_ms: float | None,
    note: str,
) -> dict[str, Any]:
    speedup = ""
    if ours_ms and vllm_ms:
        speedup = vllm_ms / ours_ms
    return {
        "kind": kind,
        "scenario": scenario,
        "op": op,
        "tokens": tokens,
        "shape": shape,
        "ours_kernel": ours_kernel,
        "vllm_kernel": vllm_kernel,
        "ours_ms": ours_ms if ours_ms is not None else "",
        "vllm_ms": vllm_ms if vllm_ms is not None else "",
        "speedup_ours_vs_vllm": speedup,
        "compare_note": note,
    }


def _scenario_name(kind: str, value: int) -> str:
    return f"prefill{value}" if kind == "prefill" else f"decode_kv{value}"


def _bench_rms(rows: list[dict[str, Any]], *, tokens: int, cfg: dict[str, Any], warmup: int, iters: int) -> None:
    import vllm._custom_ops as vllm_ops

    hidden = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((hidden,), device="cuda", dtype=torch.bfloat16)
    vllm_out = torch.empty_like(x)

    ours_ms = _time_cuda(lambda: norm_kernels.rms_norm(x, [hidden], weight, eps), warmup=warmup, iters=iters)
    vllm_ms = _time_cuda(lambda: vllm_ops.rms_norm(vllm_out, x, weight, eps), warmup=warmup, iters=iters)
    rows.append(
        _make_row(
            kind="token",
            scenario=f"tokens{tokens}",
            op="rms_norm",
            tokens=tokens,
            shape=f"[{tokens}, {hidden}]",
            ours_kernel="kernels.norm.rms_norm",
            vllm_kernel="vllm._custom_ops.rms_norm",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses the rms_norm interface without wrapper-side contiguous copies",
        ))


def _bench_fused_add_rms(
    rows: list[dict[str, Any]], *, tokens: int, cfg: dict[str, Any], warmup: int, iters: int
) -> None:
    import vllm._custom_ops as vllm_ops

    hidden = int(cfg["hidden_size"])
    eps = float(cfg["rms_norm_eps"])
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones((hidden,), device="cuda", dtype=torch.bfloat16)
    ours_x = x.clone()
    ours_residual = residual.clone()
    vllm_x = x.clone()
    vllm_residual = residual.clone()

    ours_ms = _time_cuda(lambda: norm_kernels.fused_add_rms_norm(ours_x, ours_residual, [hidden], weight, eps),
                         warmup=warmup, iters=iters)
    vllm_ms = _time_cuda(lambda: vllm_ops.fused_add_rms_norm(vllm_x, vllm_residual, weight, eps),
                         warmup=warmup, iters=iters)
    rows.append(
        _make_row(
            kind="token",
            scenario=f"tokens{tokens}",
            op="fused_add_rms_norm",
            tokens=tokens,
            shape=f"[{tokens}, {hidden}]",
            ours_kernel="kernels.norm.fused_add_rms_norm",
            vllm_kernel="vllm._custom_ops.fused_add_rms_norm",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses the fused_add_rms_norm interface without wrapper-side contiguous copies",
        ))


def _bench_silu_and_mul(
    rows: list[dict[str, Any]], *, scenario: str, tokens: int, cfg: dict[str, Any], warmup: int, iters: int
) -> None:
    import vllm._custom_ops  # noqa: F401 - registers torch.ops._C.silu_and_mul

    intermediate = int(cfg["intermediate_size"])
    x = torch.randn((tokens, 2 * intermediate), device="cuda", dtype=torch.bfloat16)
    x_a = x[:, :intermediate]
    x_b = x[:, intermediate:]
    ours_out = torch.empty((tokens, intermediate), device="cuda", dtype=torch.bfloat16)
    vllm_out = torch.empty_like(ours_out)

    ours_ms = _time_cuda(lambda: linear_kernels.silu_and_mul_out(x_a, x_b, ours_out), warmup=warmup, iters=iters)
    vllm_ms = _time_cuda(lambda: torch.ops._C.silu_and_mul(vllm_out, x), warmup=warmup, iters=iters)
    rows.append(
        _make_row(
            kind="activation",
            scenario=scenario,
            op="silu_and_mul",
            tokens=tokens,
            shape=f"[{tokens}, {2 * intermediate}] -> [{tokens}, {intermediate}]",
            ours_kernel="kernels.linear.silu_and_mul_out",
            vllm_kernel="torch.ops._C.silu_and_mul",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses silu_and_mul_out(A, B, out) on sliced views",
        ))


def _bench_rotary_embedding(
    rows: list[dict[str, Any]], *, tokens: int, cfg: dict[str, Any], warmup: int, iters: int
) -> None:
    import vllm._custom_ops as vllm_ops

    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    theta = float(cfg["rope_theta"])
    max_seq_len = max(tokens, 1)
    q = torch.randn((1, tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, tokens, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    ours_q = q.clone()
    ours_k = k.clone()
    vllm_q = q.view(tokens, heads * head_dim).clone()
    vllm_k = k.view(tokens, kv_heads * head_dim).clone()
    cos, sin = _rope_cache(max_seq_len, head_dim, theta, torch.bfloat16)
    cos_sin_cache = _vllm_cos_sin_cache(max_seq_len, head_dim, theta, torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.long)

    ours_ms = _time_cuda(
        lambda: rotary_cache.apply_rotary_pos_emb(
            ours_q, ours_k, cos, sin, position_ids=positions.view(1, tokens), rotary_interleaved=False, inplace=True
        ),
        warmup=warmup,
        iters=iters,
    )
    vllm_ms = _time_cuda(
        lambda: vllm_ops.rotary_embedding(positions, vllm_q, vllm_k, head_dim, cos_sin_cache, True),
        warmup=warmup,
        iters=iters,
    )
    rows.append(
        _make_row(
            kind="token",
            scenario=f"tokens{tokens}",
            op="rotary_embedding",
            tokens=tokens,
            shape=f"q=[{tokens}, {heads * head_dim}], k=[{tokens}, {kv_heads * head_dim}], head_dim={head_dim}",
            ours_kernel="kernels.rotary_cache.apply_rotary_pos_emb",
            vllm_kernel="vllm._custom_ops.rotary_embedding",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses the RoPE interface with explicit inplace=True",
        ))


def _bench_attention_prefill(
    rows: list[dict[str, Any]], *, seq_len: int, cfg: dict[str, Any], warmup: int, iters: int
) -> None:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn((seq_len, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((seq_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

    ours_ms = _time_cuda(
        lambda: attention_kernels.flash_attn_varlen_func(
            q, k, v, seq_len, cu, seq_len, cu_seqlens_k=cu, dropout_p=0.0,
            softmax_scale=scale, causal=True,
        ),
        warmup=warmup,
        iters=iters,
    )
    vllm_ms = _time_cuda(
        lambda: flash_attn_varlen_func(
            q, k, v, seq_len, cu, seq_len, cu_seqlens_k=cu, dropout_p=0.0,
            softmax_scale=scale, causal=True, fa_version=3,
        ),
        warmup=warmup,
        iters=iters,
    )
    rows.append(
        _make_row(
            kind="attention",
            scenario=_scenario_name("prefill", seq_len),
            op="attention.prefill",
            tokens=seq_len,
            shape=f"q=[{seq_len}, {heads}, {head_dim}], kv=[{seq_len}, {kv_heads}, {head_dim}]",
            ours_kernel="kernels.attention.flash_attn_varlen_func",
            vllm_kernel="vllm_flash_attn.flash_attn_varlen_func(fa_version=3)",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses varlen interface and non-paged launch heuristic",
        ))


def _bench_attention_decode(
    rows: list[dict[str, Any]], *, kv_len: int, cfg: dict[str, Any], warmup: int, iters: int
) -> None:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn((1, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((kv_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)

    ours_ms = _time_cuda(
        lambda: attention_kernels.flash_attn_varlen_func(
            q, k, v, 1, cu_q, kv_len, cu_seqlens_k=cu_k, dropout_p=0.0,
            softmax_scale=scale, causal=True,
        ),
        warmup=warmup,
        iters=iters,
    )
    vllm_ms = _time_cuda(
        lambda: flash_attn_varlen_func(
            q, k, v, 1, cu_q, kv_len, cu_seqlens_k=cu_k, dropout_p=0.0,
            softmax_scale=scale, causal=True, fa_version=3,
        ),
        warmup=warmup,
        iters=iters,
    )
    rows.append(
        _make_row(
            kind="attention",
            scenario=_scenario_name("decode", kv_len),
            op="attention.decode",
            tokens=1,
            shape=f"q=[1, {heads}, {head_dim}], kv=[{kv_len}, {kv_heads}, {head_dim}]",
            ours_kernel="kernels.attention.flash_attn_varlen_func",
            vllm_kernel="vllm_flash_attn.flash_attn_varlen_func(fa_version=3)",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note="ours uses non-paged cu_seqlens_k varlen path; paged attention intentionally excluded",
        ))


def _bench_linear(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    op: str,
    tokens: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
) -> None:
    x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    if op == "linear.lm_head":
        backend = linear_kernels.linear_backend_name(tokens, n, k)
        ours_kernel = f"kernels.linear.lm_head({backend})"
        note = f"runtime lm_head route uses {backend}"
        ours_ms = _time_cuda(lambda: linear_kernels.lm_head(x, weight, None), warmup=warmup, iters=iters)
    else:
        backend = linear_kernels.linear_backend_name(tokens, n, k)
        ours_kernel = f"kernels.linear.linear({backend})"
        note = f"runtime dense-linear route uses {backend}"
        ours_ms = _time_cuda(lambda: linear_kernels.linear(x, weight, None), warmup=warmup, iters=iters)
    vllm_ms = _time_cuda(lambda: _torch_linear(x, weight), warmup=warmup, iters=iters)
    rows.append(
        _make_row(
            kind="linear",
            scenario=scenario,
            op=op,
            tokens=tokens,
            shape=f"M={tokens}, N={n}, K={k}",
            ours_kernel=ours_kernel,
            vllm_kernel="torch.matmul/cuBLAS",
            ours_ms=ours_ms,
            vllm_ms=vllm_ms,
            note=note,
        ))

    def append_candidate(name: str, kernel_name: str, fn: Callable[[], torch.Tensor], note: str) -> None:
        try:
            candidate_ms: float | None = _time_cuda(fn, warmup=warmup, iters=iters)
        except Exception as exc:  # noqa: BLE001 - recorded in benchmark metadata.
            candidate_ms = None
            note = f"skipped: {type(exc).__name__}: {exc}"
        rows.append(
            _make_row(
                kind="linear",
                scenario=scenario,
                op=f"{op}.{name}",
                tokens=tokens,
                shape=f"M={tokens}, N={n}, K={k}",
                ours_kernel=kernel_name,
                vllm_kernel="torch.matmul/cuBLAS",
                ours_ms=candidate_ms,
                vllm_ms=vllm_ms,
                note=note,
            ))

    append_candidate(
        "hopper_tma",
        "kernels.linear.linear_hopper_tma",
        lambda: linear_kernels.linear_hopper_tma(x, weight, None),
        "candidate copied from FlagGems Hopper mm.py general host-TMA GEMM",
    )
    append_candidate(
        "hopper_tma_splitk",
        "kernels.linear.linear_hopper_tma_splitk",
        lambda: linear_kernels.linear_hopper_tma_splitk(x, weight, None),
        "candidate using host-TMA GEMM per K split with FP32 partial buffer and explicit reduction",
    )
    if tokens == 1:
        append_candidate(
            "gemv",
            "kernels.linear.linear_gemv",
            lambda: linear_kernels.linear_gemv(x, weight, None),
            "candidate adapted from FlagGems mv.py for single-token decode matrix-vector projection",
        )
    append_candidate(
        "slicedk",
        "kernels.linear.linear_slicedk",
        lambda: linear_kernels.linear_slicedk(x, weight, None),
        "candidate using a single CTA-local sliced-K reduction with no global partial buffer",
    )
    append_candidate(
        "cluster_slicedk",
        "kernels.linear.linear_cluster_slicedk",
        lambda: linear_kernels.linear_cluster_slicedk(x, weight, None),
        "candidate using a 2-CTA cluster split over K with rank0 DSMEM remote accumulator reduction",
    )
    append_candidate(
        "cluster_slicedk_tma",
        "kernels.linear.linear_cluster_slicedk_tma",
        lambda: linear_kernels.linear_cluster_slicedk_tma(x, weight, None),
        "autotuned candidate using host-TMA A/B/C descriptors plus a 2-CTA cluster K split and DSMEM remote accumulator reduction",
    )
    if op != "linear.gate_up_proj":
        append_candidate(
            "splitk",
            "kernels.linear.linear_splitk",
            lambda: linear_kernels.linear_splitk(x, weight, None),
            "candidate copied from FlagGems Hopper mm.py split-K GEMM",
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind", "scenario", "op", "tokens", "shape", "ours_kernel", "vllm_kernel",
        "ours_ms", "vllm_ms", "speedup_ours_vs_vllm", "compare_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.3f}"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# Local Qwen3 Ops vs vLLM Qwen3-32B Benchmark",
        "",
        f"- Model path: `{summary['model_path']}`",
        f"- Config source: {summary['config']['source']}",
        f"- Device: `{summary['device']}`",
        f"- Warmup: `{summary['warmup']}`",
        f"- Iters: `{summary['iters']}`",
        f"- Torch version: `{summary['versions'].get('torch', 'unknown')}`",
        f"- vLLM version: `{summary['versions'].get('vllm', 'unknown')}`",
        "- Ours source: local kernels in `python/tutorials/tle/mega/kernels/{norm.py,linear.py,"
        "rotary_cache.py,attention.py}`.",
        "- Imported tuning policy: RMSNorm loop uses `TILE_N={1024,2048,4096,8192}` x "
        "`num_warps={4,8,16}`; SiluAndMul uses pointwise `max_tile_size=512` and NVIDIA warp "
        "heuristic; attention uses non-paged `mha_block_*` heuristic.",
        "- Interface policy: local ops expose vLLM-compatible signatures where applicable and do not "
        "force wrapper-side `.contiguous()` copies.",
        "",
        "`speedup` is `vLLM ms / ours ms`; values above 1.0 mean our local kernel is faster.",
        "",
        "## Latency",
        "",
        "| kind | scenario | op | ours ms | vLLM ms | speedup | note |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {kind} | {scenario} | `{op}` | {ours_ms} | {vllm_ms} | {speedup} | {note} |".format(
                kind=row["kind"],
                scenario=row["scenario"],
                op=row["op"],
                ours_ms=_fmt(row["ours_ms"]),
                vllm_ms=_fmt(row["vllm_ms"]),
                speedup=_fmt(row["speedup_ours_vs_vllm"]),
                note=row["compare_note"],
            ))
    lines.extend([
        "",
        "## Shapes And Kernels",
        "",
        "| scenario | op | shape | ours kernel | vLLM kernel |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['scenario']} | `{row['op']}` | `{row['shape']}` | "
            f"`{row['ours_kernel']}` | `{row['vllm_kernel']}` |")
    lines.extend([
        "",
        "## Coverage Notes",
        "",
        "- vLLM-aligned rows use vLLM custom ops for RMSNorm, fused add RMSNorm, SiluAndMul, RoPE, and FA3 varlen attention.",
        "- RMSNorm, fused add RMSNorm, SiluAndMul, and RoPE use explicit tensor strides rather than wrapper-side contiguous copies.",
        "- Attention exposes `flash_attn_varlen_func` and uses stride-aware non-paged `cu_seqlens_k` varlen inputs plus the `mha_block_*` launch heuristic.",
        "- Paged attention, dropout, alibi, softcap, sliding-window, split-KV, and FA3 auxiliary arguments are intentionally excluded from this bench-local kernel.",
        "- Linear rows are context-only: no portable public vLLM bf16 dense-linear op is available.",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _unique_path(output_dir: Path, stem: str, suffix: str, run_id: str) -> Path:
    path = output_dir / f"{stem}-{run_id}.{suffix}"
    index = 1
    while path.exists():
        path = output_dir / f"{stem}-{run_id}-{index}.{suffix}"
        index += 1
    return path


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    cfg = _maybe_load_config(args)
    prefill_lens = args.prefill_len or [1, 128, 512, 1024]
    decode_kv_lens = args.decode_kv_len or [1, 128, 512, 1024]

    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    rows: list[dict[str, Any]] = []

    for tokens in sorted(set(prefill_lens + [1])):
        _bench_rms(rows, tokens=tokens, cfg=cfg, warmup=args.warmup, iters=args.iters)
        _bench_fused_add_rms(rows, tokens=tokens, cfg=cfg, warmup=args.warmup, iters=args.iters)
        _bench_rotary_embedding(rows, tokens=tokens, cfg=cfg, warmup=args.warmup, iters=args.iters)

    for tokens in prefill_lens:
        scenario = _scenario_name("prefill", tokens)
        _bench_linear(rows, scenario=scenario, op="linear.qkv_proj", tokens=tokens, n=qkv_dim, k=hidden,
                      warmup=args.warmup, iters=args.iters)
        _bench_linear(rows, scenario=scenario, op="linear.o_proj", tokens=tokens, n=hidden, k=q_dim,
                      warmup=args.warmup, iters=args.iters)
        _bench_linear(rows, scenario=scenario, op="linear.gate_up_proj", tokens=tokens, n=2 * intermediate, k=hidden,
                      warmup=args.warmup, iters=args.iters)
        _bench_silu_and_mul(rows, scenario=scenario, tokens=tokens, cfg=cfg, warmup=args.warmup, iters=args.iters)
        _bench_linear(rows, scenario=scenario, op="linear.down_proj", tokens=tokens, n=hidden, k=intermediate,
                      warmup=args.warmup, iters=args.iters)
        if tokens > 1:
            _bench_attention_prefill(rows, seq_len=tokens, cfg=cfg, warmup=args.warmup, iters=args.iters)

    _bench_linear(rows, scenario="lm_head", op="linear.lm_head", tokens=1, n=int(cfg["vocab_size"]), k=hidden,
                  warmup=args.warmup, iters=args.iters)
    for kv_len in decode_kv_lens:
        _bench_attention_decode(rows, kv_len=kv_len, cfg=cfg, warmup=args.warmup, iters=args.iters)

    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        vllm_version = "unknown"
    summary = {
        "model_path": args.model_path,
        "config": cfg,
        "device": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iters": args.iters,
        "versions": {"torch": torch.__version__, "vllm": vllm_version},
        "rows": rows,
    }
    csv_path = _unique_path(output_dir, "vllm-op-compare", "csv", run_id)
    json_path = _unique_path(output_dir, "vllm-op-compare", "json", run_id)
    report_path = _unique_path(output_dir, "vllm-op-compare", "md", run_id)
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, summary)
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
