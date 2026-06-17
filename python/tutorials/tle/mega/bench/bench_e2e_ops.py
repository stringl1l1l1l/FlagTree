"""Qwen3 TLE end-to-end and per-op latency benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels import attention_decode, attention_ws, embedding, fused_add_rms_norm  # noqa: E402
from kernels import head_rmsnorm_rope  # noqa: E402
from kernels import linear, lm_head, qkv_linear, rms_norm, silu_and_mul_out, store_cache  # noqa: E402
from models import Qwen3TLEEngine  # noqa: E402


class OpTimer:
    def __init__(self) -> None:
        self.enabled = False
        self.scenario = ""
        self.phase = ""
        self.step: int | None = None
        self.entries: list[dict] = []

    @contextmanager
    def scope(self, *, scenario: str, phase: str, step: int | None = None):
        old = (self.scenario, self.phase, self.step)
        self.scenario = scenario
        self.phase = phase
        self.step = step
        try:
            yield
        finally:
            self.scenario, self.phase, self.step = old

    def record(self, op: str, layer: int | None, fn):
        if not self.enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        self.entries.append({
            "scenario": self.scenario,
            "phase": self.phase,
            "step": self.step,
            "layer": layer,
            "op": op,
            "ms": float(start.elapsed_time(end)),
        })
        return result


class TimedQwen3TLEEngine(Qwen3TLEEngine):
    def __init__(self, *args, timer: OpTimer | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timer = timer or OpTimer()

    def _linear_timed(self, op: str, layer: int | None, x: torch.Tensor, weights):
        return self.timer.record(
            op,
            layer,
            lambda: linear(x.contiguous(), weights.weight, weights.bias),
        )

    def _lm_head_timed(self, x: torch.Tensor, weights):
        return self.timer.record(
            "linear.lm_head",
            None,
            lambda: lm_head(x.contiguous(), weights.weight, weights.bias),
        )

    def _rms_norm_timed(self, op: str, layer: int | None, x: torch.Tensor, weight: torch.Tensor):
        return self.timer.record(
            op,
            layer,
            lambda: rms_norm(x.contiguous(), (self.config.hidden_size, ), weight, self.config.rms_norm_eps),
        )

    def _fused_add_rms_norm_timed(
        self,
        op: str,
        layer: int | None,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ):
        return self.timer.record(
            op,
            layer,
            lambda: fused_add_rms_norm(x.contiguous(), residual.contiguous(), (self.config.hidden_size, ), weight,
                                       self.config.rms_norm_eps),
        )

    def _qkv_linear_timed(self, layer: int, x: torch.Tensor, weights):
        q_dim = self.config.num_attention_heads * self.config.head_dim
        kv_dim = self.config.num_key_value_heads * self.config.head_dim
        return self.timer.record(
            "linear.qkv_proj",
            layer,
            lambda: qkv_linear(x.contiguous(), weights.weight, weights.bias, q_dim=q_dim, kv_dim=kv_dim),
        )

    def _mlp_activation_timed(self, layer: int, x: torch.Tensor, weights):
        packed = self._linear_timed("linear.gate_up_proj", layer, x, weights)
        gate, up = packed.split((self.config.intermediate_size, self.config.intermediate_size), dim=-1)
        out = torch.empty_like(gate)
        return self.timer.record("silu_and_mul", layer, lambda: silu_and_mul_out(gate, up, out))

    def _attention_timed(
        self,
        layer: int,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        q_len: int,
        start_pos: int,
        kv_len: int,
        sm_scale: float,
    ):
        if q_len == 1:
            fn = attention_decode
            op = "attention.decode"
        else:
            fn = attention_ws
            op = "attention.ws"
        return self.timer.record(
            op,
            layer,
            lambda: fn(q, k_cache, v_cache, q_len=q_len, start_pos=start_pos, kv_len=kv_len, sm_scale=sm_scale),
        )

    def _run_layer(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        layer_idx: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.cache is not None
        layer = self.weights.layers[layer_idx]
        batch_size, q_len, hidden = x.shape
        tokens = batch_size * q_len
        q_dim = self.config.num_attention_heads * self.config.head_dim
        x_flat = x.reshape(tokens, hidden).contiguous()

        if residual is None:
            residual_flat = x_flat
            hidden_flat = self._rms_norm_timed("rms_norm.input", layer_idx, x_flat, layer.input_norm_weight)
        else:
            residual_flat = residual.reshape(tokens, hidden).contiguous()
            hidden_flat, residual_flat = self._fused_add_rms_norm_timed(
                "fused_add_rms_norm.input",
                layer_idx,
                x_flat,
                residual_flat,
                layer.input_norm_weight,
            )

        q_flat, k_flat, v_flat = self._qkv_linear_timed(layer_idx, hidden_flat, layer.qkv_proj)
        q = q_flat.view(tokens, self.config.num_attention_heads, self.config.head_dim)
        k = k_flat.view(tokens, self.config.num_key_value_heads, self.config.head_dim)
        v = v_flat.view(tokens, self.config.num_key_value_heads, self.config.head_dim)

        q = self.timer.record(
            "head_rmsnorm_rope.q",
            layer_idx,
            lambda: head_rmsnorm_rope(q, layer.q_norm_weight, self.cos, self.sin, q_len=q_len,
                                      start_pos=start_pos, eps=layer.q_norm_eps),
        )
        k = self.timer.record(
            "head_rmsnorm_rope.k",
            layer_idx,
            lambda: head_rmsnorm_rope(k, layer.k_norm_weight, self.cos, self.sin, q_len=q_len,
                                      start_pos=start_pos, eps=layer.k_norm_eps),
        )
        self.timer.record("store_cache.k", layer_idx,
                          lambda: store_cache(k, self.cache.k[layer_idx], q_len=q_len, start_pos=start_pos))
        self.timer.record("store_cache.v", layer_idx,
                          lambda: store_cache(v, self.cache.v[layer_idx], q_len=q_len, start_pos=start_pos))

        attn = self._attention_timed(
            layer_idx,
            q,
            self.cache.k[layer_idx],
            self.cache.v[layer_idx],
            q_len=q_len,
            start_pos=start_pos,
            kv_len=start_pos + q_len,
            sm_scale=1.0 / math.sqrt(self.config.head_dim),
        )
        attn_flat = attn.reshape(tokens, q_dim).contiguous()
        attn_out = self._linear_timed("linear.o_proj", layer_idx, attn_flat, layer.o_proj)

        hidden_flat, residual_flat = self._fused_add_rms_norm_timed(
            "fused_add_rms_norm.post_attention",
            layer_idx,
            attn_out,
            residual_flat,
            layer.post_norm_weight,
        )
        hidden_act = self._mlp_activation_timed(layer_idx, hidden_flat, layer.gate_up_proj)
        out = self._linear_timed("linear.down_proj", layer_idx, hidden_act, layer.down_proj)
        return out.view(batch_size, q_len, hidden), residual_flat.view(batch_size, q_len, hidden)

    @torch.inference_mode()
    def forward_chunk(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        if self.cache is None:
            self.reset_cache(batch_size=input_ids.shape[0])
        assert self.cache is not None
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(device=self.device, dtype=torch.long).contiguous()
        batch_size, q_len = input_ids.shape
        if batch_size != self.cache.batch_size:
            raise ValueError(f"cache batch={self.cache.batch_size} does not match input batch={batch_size}")
        if start_pos + q_len > self.cache.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + q_len} exceeds cache max_seq_len={self.cache.max_seq_len}")

        x = self.timer.record("embedding", None, lambda: embedding(input_ids.reshape(-1), self.weights.embed_tokens))
        x = x.view(batch_size, q_len, self.config.hidden_size)
        residual = None
        for layer_idx in range(self.config.num_hidden_layers):
            x, residual = self._run_layer(x, residual, layer_idx, start_pos)

        last_hidden = x[:, -1, :].contiguous().view(batch_size, self.config.hidden_size)
        if residual is None:
            last_hidden = self._rms_norm_timed("rms_norm.final", None, last_hidden, self.weights.final_norm_weight)
        else:
            last_residual = residual[:, -1, :].contiguous().view(batch_size, self.config.hidden_size)
            last_hidden, _ = self._fused_add_rms_norm_timed(
                "fused_add_rms_norm.final",
                None,
                last_hidden,
                last_residual,
                self.weights.final_norm_weight,
            )
        return self._lm_head_timed(last_hidden, self.weights.lm_head)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 TLE end-to-end and per-op benchmark")
    parser.add_argument("--model-path", default="/data/dataset/Qwen3-32B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scenario", action="append", default=[],
                        help="Scenario as prompt_len:decode_steps. Can be repeated.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="build/mega/qwen3-32b")
    parser.add_argument("--attention-backend", choices=("ws", ), default="ws")
    return parser


def _parse_scenarios(values: list[str]) -> list[tuple[int, int]]:
    if not values:
        values = ["1:16", "128:4"]
    scenarios = []
    for value in values:
        try:
            prompt_len, decode_steps = value.split(":", 1)
            scenarios.append((int(prompt_len), int(decode_steps)))
        except ValueError as exc:
            raise ValueError(f"invalid --scenario {value!r}; expected prompt_len:decode_steps") from exc
    return scenarios


def _random_input_ids(engine: Qwen3TLEEngine, batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.randint(0, int(engine.config.vocab_size), (batch_size, seq_len), device=engine.device,
                         dtype=torch.long)


def _time_cuda(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def _run_sequence(engine: Qwen3TLEEngine, input_ids: torch.Tensor, *, max_seq_len: int, decode_steps: int,
                  timer: OpTimer | None = None, scenario: str = "") -> None:
    batch_size = input_ids.shape[0]
    engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
    if timer is None:
        logits = engine.prefill(input_ids)
    else:
        with timer.scope(scenario=scenario, phase="prefill"):
            logits = engine.prefill(input_ids)
    token = _random_input_ids(engine, batch_size, 1)
    for step in range(decode_steps):
        if timer is None:
            logits = engine.decode(token)
            token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)
        else:
            with timer.scope(scenario=scenario, phase="decode", step=step):
                logits = engine.decode(token)
                token = timer.record("torch.argmax", None,
                                     lambda: torch.argmax(logits, dim=-1, keepdim=True).to(torch.long))


def _bench_e2e(engine: Qwen3TLEEngine, input_ids: torch.Tensor, *, max_seq_len: int, decode_steps: int, warmup: int,
               iters: int) -> tuple[float, float]:
    batch_size = input_ids.shape[0]
    for _ in range(warmup):
        _run_sequence(engine, input_ids, max_seq_len=max_seq_len, decode_steps=decode_steps)

    prefill_ms = _time_cuda(
        lambda: (engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len), engine.prefill(input_ids)),
        iters,
    )

    elapsed_decode_ms = 0.0
    for _ in range(iters):
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)
        token = _random_input_ids(engine, batch_size, 1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(decode_steps):
            logits = engine.decode(token)
            token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)
        end.record()
        torch.cuda.synchronize()
        elapsed_decode_ms += float(start.elapsed_time(end))

    return prefill_ms, elapsed_decode_ms / max(iters * decode_steps, 1)


def _aggregate_ops(entries: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for entry in entries:
        key = (entry["scenario"], entry["phase"], entry["op"])
        buckets[key].append(float(entry["ms"]))
    rows = []
    for (scenario, phase, op), values in sorted(buckets.items()):
        total = sum(values)
        rows.append({
            "scenario": scenario,
            "phase": phase,
            "op": op,
            "count": len(values),
            "total_ms": total,
            "avg_ms": total / len(values),
            "min_ms": min(values),
            "max_ms": max(values),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _write_report(path: Path, summary: dict) -> None:
    scenarios = summary["scenarios"]
    op_rows = summary["op_aggregate"]
    by_scope: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in op_rows:
        by_scope[(row["scenario"], row["phase"])].append(row)

    lines = [
        "# Qwen3 TLE Mega Benchmark Report",
        "",
        f"- Model: `{summary['model_path']}`",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Attention backend: `{summary['attention_backend']}`",
        f"- Warmup: `{summary['warmup']}`",
        f"- Iters: `{summary['iters']}`",
        "",
        "Operator timings are CUDA event measurements around the current high-level TLE kernel calls. "
        "End-to-end latency is measured separately.",
        "",
        "Operator boundaries follow vLLM-style responsibilities: RMSNorm is a full normalization op, "
        "`fused_add_rms_norm` owns residual add plus normalization, packed `gate_up_proj` is a linear op, "
        "and `silu_and_mul` is measured as the separate MLP activation op.",
        "",
        "Attention backend `ws` uses the dot-based TLE pipe prefill kernel for `q_len > 1`. Decode uses the "
        "TileOps-style split/no-split GQA algorithm. On sm90+ tensors that satisfy host descriptor alignment, "
        "decode stages K/V tiles through a `tle.pipe` producer that issues explicit `tle.gpu.copy` TMA operations "
        "from host `TensorDescriptor` objects, while a warp-specialized consumer runs QK/PV. Q and output/partial "
        "output tiles also use host descriptor TMA. K/V descriptors use logical `[batch, kv_head, seq, dim]` "
        "shape/stride so GQA cache loads do not flatten across adjacent KV heads.",
        "",
        "## End-to-End Latency",
        "",
        "| scenario | batch | prefill ms | decode ms/token | prefill tok/s | decode tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in scenarios:
        lines.append(
            "| {scenario} | {batch_size} | {prefill_ms} | {decode_ms_per_token} | "
            "{prefill_tokens_per_s} | {decode_tokens_per_s} |".format(
                scenario=record["scenario"],
                batch_size=record["batch_size"],
                prefill_ms=_fmt(record["prefill_ms"]),
                decode_ms_per_token=_fmt(record["decode_ms_per_token"]),
                prefill_tokens_per_s=_fmt(record["prefill_tokens_per_s"]),
                decode_tokens_per_s=_fmt(record["decode_tokens_per_s"]),
            ))

    lines.extend(["", "## Top Operators", ""])
    for record in scenarios:
        scenario = record["scenario"]
        for phase in ("prefill", "decode"):
            rows = sorted(by_scope.get((scenario, phase), []), key=lambda item: item["total_ms"], reverse=True)
            if not rows:
                continue
            lines.extend([
                f"### {scenario} / {phase}",
                "",
                "| op | count | total ms | avg ms | min ms | max ms |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for row in rows[:12]:
                lines.append(
                    "| {op} | {count} | {total_ms} | {avg_ms} | {min_ms} | {max_ms} |".format(
                        op=row["op"],
                        count=row["count"],
                        total_ms=_fmt(row["total_ms"]),
                        avg_ms=_fmt(row["avg_ms"]),
                        min_ms=_fmt(row["min_ms"]),
                        max_ms=_fmt(row["max_ms"]),
                    ))
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _unique_report_path(output_dir: Path, run_id: str) -> Path:
    path = output_dir / f"report-{run_id}.md"
    suffix = 1
    while path.exists():
        path = output_dir / f"report-{run_id}-{suffix}.md"
        suffix += 1
    return path


def main() -> None:
    args = _build_parser().parse_args()
    scenarios = _parse_scenarios(args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    timer = OpTimer()
    engine = TimedQwen3TLEEngine.from_pretrained(
        args.model_path,
        device=args.device,
        dtype="bf16",
        max_seq_len=args.max_seq_len or max(prompt + decode for prompt, decode in scenarios),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attention_backend=args.attention_backend,
    )
    engine.timer = timer

    records = []
    for prompt_len, decode_steps in scenarios:
        scenario_name = f"prefill{prompt_len}_decode{decode_steps}"
        max_seq_len = args.max_seq_len or (prompt_len + decode_steps)
        input_ids = _random_input_ids(engine, args.batch_size, prompt_len)

        prefill_ms, decode_ms = _bench_e2e(
            engine,
            input_ids,
            max_seq_len=max_seq_len,
            decode_steps=decode_steps,
            warmup=args.warmup,
            iters=args.iters,
        )

        timer.enabled = True
        _run_sequence(engine, input_ids, max_seq_len=max_seq_len, decode_steps=decode_steps, timer=timer,
                      scenario=scenario_name)
        timer.enabled = False

        record = {
            "timestamp": int(time.time()),
            "scenario": scenario_name,
            "model_path": args.model_path,
            "backend": "tle",
            "attention_backend": args.attention_backend,
            "batch_size": args.batch_size,
            "prompt_len": prompt_len,
            "decode_steps": decode_steps,
            "max_seq_len": max_seq_len,
            "prefill_ms": prefill_ms,
            "decode_ms_per_token": decode_ms,
            "prefill_tokens_per_s": args.batch_size * prompt_len * 1000.0 / prefill_ms,
            "decode_tokens_per_s": args.batch_size * 1000.0 / decode_ms,
            "num_layers": engine.config.num_hidden_layers,
            "hidden_size": engine.config.hidden_size,
            "num_attention_heads": engine.config.num_attention_heads,
            "num_key_value_heads": engine.config.num_key_value_heads,
            "head_dim": engine.config.head_dim,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    aggregate_rows = _aggregate_ops(timer.entries)
    summary_path = output_dir / "summary.json"
    e2e_csv_path = output_dir / "e2e_latency.csv"
    op_csv_path = output_dir / "op_latency_aggregate.csv"
    detail_path = output_dir / "op_latency_detail.jsonl"
    report_path = _unique_report_path(output_dir, run_id)

    summary = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "attention_backend": args.attention_backend,
        "report_path": str(report_path),
        "run_id": run_id,
        "warmup": args.warmup,
        "iters": args.iters,
        "scenarios": records,
        "op_aggregate": aggregate_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(e2e_csv_path, records)
    _write_csv(op_csv_path, aggregate_rows)
    _write_report(report_path, summary)
    with detail_path.open("w", encoding="utf-8") as f:
        for entry in timer.entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    print(f"wrote {summary_path}")
    print(f"wrote {e2e_csv_path}")
    print(f"wrote {op_csv_path}")
    print(f"wrote {report_path}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
