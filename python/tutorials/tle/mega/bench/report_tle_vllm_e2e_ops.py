"""Build a combined TLE/vLLM Qwen3-32B e2e and per-op report."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("build/mega/qwen3-32b")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate combined TLE/vLLM Qwen3-32B benchmark report")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tle-summary", default="", help="TLE summary.json from bench_e2e_ops.py")
    parser.add_argument("--vllm-e2e", default="", help="vLLM serving JSON from bench_serving_engines.py")
    parser.add_argument("--vllm-op", default="", help="vLLM op compare JSON from bench_vllm_op_compare.py")
    parser.add_argument("--top-n", type=int, default=999, help="Maximum TLE op rows per scenario/phase")
    return parser


def _latest(pattern: str, output_dir: Path) -> Path:
    matches = sorted(output_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no files matching {output_dir / pattern}")
    return matches[-1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def _speedup(numerator: float | None, denominator: float | None) -> float | None:
    if not numerator or not denominator:
        return None
    return numerator / denominator


def _scenario_key(record: dict[str, Any]) -> str:
    if record.get("scenario"):
        return str(record["scenario"])
    return f"prefill{record['prompt_len']}_decode{record['decode_steps']}"


def _unique_path(output_dir: Path, stem: str, suffix: str, run_id: str) -> Path:
    path = output_dir / f"{stem}-{run_id}.{suffix}"
    index = 1
    while path.exists():
        path = output_dir / f"{stem}-{run_id}-{index}.{suffix}"
        index += 1
    return path


def _build_summary(
    *,
    tle_summary_path: Path,
    vllm_e2e_path: Path,
    vllm_op_path: Path,
    top_n: int,
) -> dict[str, Any]:
    tle_summary = _load_json(tle_summary_path)
    vllm_e2e = _load_json(vllm_e2e_path)
    vllm_op = _load_json(vllm_op_path)

    tle_by_scenario = {_scenario_key(row): row for row in tle_summary["scenarios"]}
    vllm_by_scenario = {_scenario_key(row): row for row in vllm_e2e["results"]}

    def _scenario_sort_key(name: str) -> tuple[int, int, str]:
        row = tle_by_scenario[name]
        return int(row["prompt_len"]), int(row["decode_steps"]), name

    e2e_rows = []
    for scenario in sorted(tle_by_scenario, key=_scenario_sort_key):
        tle = tle_by_scenario[scenario]
        vllm = vllm_by_scenario.get(scenario)
        decode_steps = int(tle["decode_steps"])
        tle_e2e_ms = float(tle["prefill_ms"]) + float(tle["decode_ms_per_token"]) * decode_steps
        vllm_e2e_ms = float(vllm["e2e_ms"]) if vllm else None
        e2e_rows.append({
            "scenario": scenario,
            "batch_size": int(tle["batch_size"]),
            "prompt_len": int(tle["prompt_len"]),
            "decode_steps": decode_steps,
            "tle_prefill_ms": float(tle["prefill_ms"]),
            "tle_decode_ms_per_token": float(tle["decode_ms_per_token"]),
            "tle_estimated_request_ms": tle_e2e_ms,
            "vllm_request_ms": vllm_e2e_ms,
            "vllm_speedup_vs_tle": _speedup(tle_e2e_ms, vllm_e2e_ms),
            "tle_prefill_tokens_per_s": float(tle["prefill_tokens_per_s"]),
            "tle_decode_tokens_per_s": float(tle["decode_tokens_per_s"]),
            "vllm_generated_tokens_per_s_including_prefill": (
                float(vllm["decode_tokens_per_s_including_prefill"]) if vllm else None
            ),
        })

    tle_ops_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tle_summary["op_aggregate"]:
        tle_ops_by_scope[(row["scenario"], row["phase"])].append(row)
    for rows in tle_ops_by_scope.values():
        rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)

    def _scope_sort_key(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> tuple[int, int, str, str]:
        (scenario, phase), _rows = item
        row = tle_by_scenario.get(scenario, {})
        return int(row.get("prompt_len", 0)), int(row.get("decode_steps", 0)), scenario, phase

    vllm_ops = []
    for row in vllm_op["rows"]:
        ours_ms = row.get("ours_ms")
        vllm_ms = row.get("vllm_ms")
        vllm_ops.append({
            **row,
            "speedup_vllm_vs_tle_op": _speedup(float(ours_ms), float(vllm_ms)) if ours_ms != "" and vllm_ms != "" else None,
        })

    vllm_lookup = {(row["scenario"], row["op"]): row for row in vllm_ops}

    def _lookup_vllm(parts: list[tuple[str, str]]) -> tuple[float | None, str, str, str]:
        rows = []
        missing = []
        for scenario, op in parts:
            row = vllm_lookup.get((scenario, op))
            if row is None:
                missing.append(f"{scenario}:{op}")
            else:
                rows.append(row)
        if missing:
            return None, " + ".join(op for _scenario, op in parts), " + ".join(missing), "missing vLLM op row"
        return (
            sum(float(row["vllm_ms"]) for row in rows),
            " + ".join(row["op"] for row in rows),
            " + ".join(row["scenario"] for row in rows),
            "",
        )

    def _match_vllm_op(tle_op: str, scenario: str, phase: str) -> dict[str, Any] | None:
        meta = tle_by_scenario[scenario]
        prompt_len = int(meta["prompt_len"])
        decode_steps = int(meta["decode_steps"])
        token_scenario = f"prefill{prompt_len if phase == 'prefill' else 1}"
        note = ""
        if tle_op == "linear.qkv_proj":
            parts = [(token_scenario, "linear.qkv_proj")]
            group = "linear.qkv_proj"
        elif tle_op == "linear.o_proj":
            parts = [(token_scenario, "linear.o_proj")]
            group = "linear.o_proj"
        elif tle_op == "linear.gate_up_proj":
            parts = [(token_scenario, "linear.gate_up_proj")]
            group = "linear.gate_up_proj"
        elif tle_op == "silu_and_mul":
            parts = [(token_scenario, "silu_and_mul")]
            group = "silu_and_mul"
        elif tle_op == "linear.down_proj":
            parts = [(token_scenario, "linear.down_proj")]
            group = "linear.down_proj"
        elif tle_op == "linear.lm_head":
            parts = [("lm_head", "linear.lm_head")]
            group = "linear.lm_head"
        elif tle_op == "attention.ws":
            parts = [(f"prefill{prompt_len}", "attention.prefill")]
            group = "attention.prefill"
        elif tle_op == "attention.decode":
            kv_len = 1 if phase == "prefill" else prompt_len + decode_steps
            parts = [(f"decode_kv{kv_len}", "attention.decode")]
            group = "attention.decode"
            note = "Decode e2e average spans growing KV length; vLLM row uses max scenario KV length."
        elif tle_op.startswith("rms_norm."):
            tokens = prompt_len if phase == "prefill" and tle_op == "rms_norm.input" else 1
            parts = [(f"tokens{tokens}", "rms_norm")]
            group = tle_op
        elif tle_op.startswith("fused_add_rms_norm."):
            tokens = prompt_len if phase == "prefill" and tle_op != "fused_add_rms_norm.final" else 1
            parts = [(f"tokens{tokens}", "fused_add_rms_norm")]
            group = tle_op
        else:
            return None

        vllm_ms, vllm_op, vllm_scenario, lookup_note = _lookup_vllm(parts)
        if lookup_note:
            note = f"{note} {lookup_note}".strip()
        return {
            "group": group,
            "tle_op": tle_op,
            "vllm_op": vllm_op,
            "vllm_scenario": vllm_scenario,
            "vllm_ms": vllm_ms,
            "note": note,
        }

    matched_ops_by_scope: dict[str, list[dict[str, Any]]] = {}
    for (scenario, phase), rows in sorted(tle_ops_by_scope.items(), key=_scope_sort_key):
        matched_rows = []
        for row in rows:
            match = _match_vllm_op(row["op"], scenario, phase)
            if match is None:
                continue
            tle_avg_ms = float(row["avg_ms"])
            vllm_ms = match["vllm_ms"]
            matched_rows.append({
                **match,
                "scenario": scenario,
                "phase": phase,
                "tle_count": int(row["count"]),
                "tle_total_ms": float(row["total_ms"]),
                "tle_avg_ms": tle_avg_ms,
                "speedup_vllm_vs_tle": _speedup(tle_avg_ms, vllm_ms),
            })
        matched_ops_by_scope[f"{scenario}/{phase}"] = sorted(
            matched_rows,
            key=lambda item: item["tle_total_ms"],
            reverse=True,
        )

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "tle_summary": str(tle_summary_path),
            "vllm_e2e": str(vllm_e2e_path),
            "vllm_op": str(vllm_op_path),
        },
        "model_path": tle_summary.get("model_path") or vllm_e2e.get("metadata", {}).get("model"),
        "device": vllm_op.get("device", "unknown"),
        "top_n": top_n,
        "methodology": {
            "tle_e2e": {
                "warmup": tle_summary.get("warmup"),
                "iters": tle_summary.get("iters"),
                "attention_backend": tle_summary.get("attention_backend"),
                "note": "TLE request latency is estimated as prefill_ms + decode_ms_per_token * decode_steps.",
            },
            "vllm_e2e": vllm_e2e.get("metadata", {}),
            "vllm_op": {
                "warmup": vllm_op.get("warmup"),
                "iters": vllm_op.get("iters"),
                "versions": vllm_op.get("versions", {}),
                "note": "vLLM per-op rows are isolated same-shape microbenchmarks, not an end-to-end vLLM trace.",
            },
        },
        "e2e_rows": e2e_rows,
        "matched_ops_by_scope": matched_ops_by_scope,
        "tle_ops_by_scope": {
            f"{scenario}/{phase}": rows[:top_n]
            for (scenario, phase), rows in sorted(tle_ops_by_scope.items(), key=_scope_sort_key)
        },
        "vllm_op_rows": vllm_ops,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-32B TLE vs vLLM E2E And Operator Report",
        "",
        f"- Created: `{summary['created_utc']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Device: `{summary['device']}`",
        f"- TLE source: `{summary['sources']['tle_summary']}`",
        f"- vLLM e2e source: `{summary['sources']['vllm_e2e']}`",
        f"- vLLM op source: `{summary['sources']['vllm_op']}`",
        "",
        "## Methodology",
        "",
        f"- TLE e2e: warmup `{summary['methodology']['tle_e2e']['warmup']}`, "
        f"iters `{summary['methodology']['tle_e2e']['iters']}`, "
        f"attention backend `{summary['methodology']['tle_e2e']['attention_backend']}`.",
        f"- vLLM e2e: warmup `{summary['methodology']['vllm_e2e'].get('warmup')}`, "
        f"iters `{summary['methodology']['vllm_e2e'].get('iters')}`, "
        f"dtype `{summary['methodology']['vllm_e2e'].get('dtype')}`.",
        f"- vLLM op microbench: warmup `{summary['methodology']['vllm_op']['warmup']}`, "
        f"iters `{summary['methodology']['vllm_op']['iters']}`.",
        "- TLE e2e per-op timings are CUDA events around high-level TLE kernel calls inside the TLE engine.",
        "- vLLM operator rows are isolated same-shape microbenchmarks against vLLM custom ops or cuBLAS, not a vLLM serving trace.",
        "",
        "## E2E Latency",
        "",
        "`vLLM speedup` is `TLE estimated request ms / vLLM request ms`; values above 1.0 mean vLLM is faster.",
        "",
        "| scenario | batch | prompt | decode | TLE prefill ms | TLE decode ms/token | TLE request ms | vLLM request ms | vLLM speedup | TLE prefill tok/s | TLE decode tok/s | vLLM gen tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["e2e_rows"]:
        lines.append(
            "| {scenario} | {batch_size} | {prompt_len} | {decode_steps} | {tle_prefill_ms} | "
            "{tle_decode_ms_per_token} | {tle_estimated_request_ms} | {vllm_request_ms} | "
            "{vllm_speedup_vs_tle} | {tle_prefill_tokens_per_s} | {tle_decode_tokens_per_s} | "
            "{vllm_generated_tokens_per_s_including_prefill} |".format(
                scenario=row["scenario"],
                batch_size=row["batch_size"],
                prompt_len=row["prompt_len"],
                decode_steps=row["decode_steps"],
                tle_prefill_ms=_fmt(row["tle_prefill_ms"]),
                tle_decode_ms_per_token=_fmt(row["tle_decode_ms_per_token"]),
                tle_estimated_request_ms=_fmt(row["tle_estimated_request_ms"]),
                vllm_request_ms=_fmt(row["vllm_request_ms"]),
                vllm_speedup_vs_tle=_fmt(row["vllm_speedup_vs_tle"], 2),
                tle_prefill_tokens_per_s=_fmt(row["tle_prefill_tokens_per_s"]),
                tle_decode_tokens_per_s=_fmt(row["tle_decode_tokens_per_s"]),
                vllm_generated_tokens_per_s_including_prefill=_fmt(row["vllm_generated_tokens_per_s_including_prefill"]),
            )
        )

    lines.extend([
        "",
        "## Scenario-Level Matched Operators",
        "",
        "`vLLM op speedup` is `TLE avg ms/call / vLLM matched op ms`; values above 1.0 mean the vLLM op is faster. "
        "Matched rows use the same vLLM-style operator responsibility where available.",
        "",
    ])
    for scope, rows in summary["matched_ops_by_scope"].items():
        if not rows:
            continue
        scenario, phase = scope.split("/", 1)
        lines.extend([
            f"### {scenario} / {phase}",
            "",
            "| op group | TLE op | TLE count | TLE total ms | TLE avg ms/call | vLLM matched op(s) | vLLM op scenario | vLLM ms/call | vLLM op speedup | note |",
            "|---|---|---:|---:|---:|---|---|---:|---:|---|",
        ])
        for row in rows:
            lines.append(
                "| {group} | `{tle_op}` | {tle_count} | {tle_total_ms} | {tle_avg_ms} | `{vllm_op}` | `{vllm_scenario}` | {vllm_ms} | {speedup} | {note} |".format(
                    group=row["group"],
                    tle_op=row["tle_op"],
                    tle_count=row["tle_count"],
                    tle_total_ms=_fmt(row["tle_total_ms"]),
                    tle_avg_ms=_fmt(row["tle_avg_ms"]),
                    vllm_op=row["vllm_op"],
                    vllm_scenario=row["vllm_scenario"],
                    vllm_ms=_fmt(row["vllm_ms"]),
                    speedup=_fmt(row["speedup_vllm_vs_tle"], 2),
                    note=row["note"],
                )
            )
        lines.append("")

    lines.extend(["", "## TLE E2E Operators", ""])
    for scope, rows in summary["tle_ops_by_scope"].items():
        scenario, phase = scope.split("/", 1)
        lines.extend([
            f"### {scenario} / {phase}",
            "",
            "| op | count | total ms | avg ms | min ms | max ms |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                "| {op} | {count} | {total_ms} | {avg_ms} | {min_ms} | {max_ms} |".format(
                    op=row["op"],
                    count=row["count"],
                    total_ms=_fmt(row["total_ms"]),
                    avg_ms=_fmt(row["avg_ms"]),
                    min_ms=_fmt(row["min_ms"]),
                    max_ms=_fmt(row["max_ms"]),
                )
            )
        lines.append("")

    lines.extend([
        "## TLE Kernel vs vLLM Op Microbenchmarks",
        "",
        "`vLLM op speedup` is `TLE/local-op ms / vLLM-op ms`; values above 1.0 mean the vLLM op is faster.",
        "",
        "| kind | scenario | op | shape | TLE/local ms | vLLM op ms | vLLM op speedup | TLE/local kernel | vLLM kernel |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ])
    for row in summary["vllm_op_rows"]:
        lines.append(
            "| {kind} | {scenario} | `{op}` | `{shape}` | {ours_ms} | {vllm_ms} | {speedup} | `{ours_kernel}` | `{vllm_kernel}` |".format(
                kind=row["kind"],
                scenario=row["scenario"],
                op=row["op"],
                shape=row["shape"],
                ours_ms=_fmt(row["ours_ms"]),
                vllm_ms=_fmt(row["vllm_ms"]),
                speedup=_fmt(row["speedup_vllm_vs_tle_op"], 2),
                ours_kernel=row["ours_kernel"],
                vllm_kernel=row["vllm_kernel"],
            )
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    tle_summary_path = Path(args.tle_summary) if args.tle_summary else output_dir / "summary.json"
    vllm_e2e_path = Path(args.vllm_e2e) if args.vllm_e2e else _latest("vllm-qwen3-32b-*.json", output_dir)
    vllm_op_path = Path(args.vllm_op) if args.vllm_op else _latest("vllm-op-compare-*.json", output_dir)

    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    summary = _build_summary(
        tle_summary_path=tle_summary_path,
        vllm_e2e_path=vllm_e2e_path,
        vllm_op_path=vllm_op_path,
        top_n=args.top_n,
    )

    md_path = _unique_path(output_dir, "tle-vllm-e2e-op-report", "md", run_id)
    json_path = _unique_path(output_dir, "tle-vllm-e2e-op-report", "json", run_id)
    _write_markdown(md_path, summary)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
