"""Benchmark Qwen3 TLE prefill and decode.

Example:
  python python/tutorials/tle/mega/bench/bench_prefill_decode.py \
      --model-path Qwen/Qwen3-32B --prompt-len 1024 --decode-steps 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels._common import cuda_events_time_ms  # noqa: E402
from models import Qwen3TLEEngine  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 TLE prefill/decode benchmark")
    parser.add_argument("--model-path", default="Qwen/Qwen3-32B", help="HF model id or local model directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    return parser


def _random_input_ids(engine: Qwen3TLEEngine, batch_size: int, seq_len: int) -> torch.Tensor:
    low = 0
    high = int(engine.config.vocab_size)
    return torch.randint(low, high, (batch_size, seq_len), device=engine.device, dtype=torch.long)


def bench_prefill(
    engine: Qwen3TLEEngine,
    input_ids: torch.Tensor,
    *,
    max_seq_len: int,
    warmup: int,
    iters: int,
) -> float:
    batch_size = input_ids.shape[0]

    def run():
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)

    return cuda_events_time_ms(run, warmup=warmup, iters=iters)


def bench_decode(
    engine: Qwen3TLEEngine,
    input_ids: torch.Tensor,
    *,
    max_seq_len: int,
    decode_steps: int,
    warmup: int,
    iters: int,
) -> float:
    batch_size = input_ids.shape[0]

    def run_warmup_sequence():
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)
        token = _random_input_ids(engine, batch_size, 1)
        for _ in range(decode_steps):
            logits = engine.decode(token)
            token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)

    for _ in range(warmup):
        run_warmup_sequence()
    torch.cuda.synchronize()

    elapsed = 0.0
    for _ in range(iters):
        engine.reset_cache(batch_size=batch_size, max_seq_len=max_seq_len)
        engine.prefill(input_ids)
        token = _random_input_ids(engine, batch_size, 1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _step in range(decode_steps):
            logits = engine.decode(token)
            token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.long)
        end.record()
        torch.cuda.synchronize()
        elapsed += start.elapsed_time(end)
    return elapsed / max(iters * decode_steps, 1)


def _write_jsonl(results_dir: Path, record: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "qwen3_32b_tle_prefill_decode.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def main() -> None:
    args = _build_parser().parse_args()
    max_seq_len = args.max_seq_len or (args.prompt_len + args.decode_steps)
    engine = Qwen3TLEEngine.from_pretrained(
        args.model_path,
        device=args.device,
        dtype="bf16",
        max_seq_len=max_seq_len,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    input_ids = _random_input_ids(engine, args.batch_size, args.prompt_len)

    prefill_ms = bench_prefill(engine, input_ids, max_seq_len=max_seq_len, warmup=args.warmup, iters=args.iters)
    decode_ms = bench_decode(
        engine,
        input_ids,
        max_seq_len=max_seq_len,
        decode_steps=args.decode_steps,
        warmup=args.warmup,
        iters=args.iters,
    )
    prefill_tokens_per_s = args.batch_size * args.prompt_len * 1000.0 / prefill_ms
    decode_tokens_per_s = args.batch_size * 1000.0 / decode_ms
    record = {
        "timestamp": int(time.time()),
        "model_path": args.model_path,
        "backend": "tle",
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "max_seq_len": max_seq_len,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms,
        "prefill_tokens_per_s": prefill_tokens_per_s,
        "decode_tokens_per_s": decode_tokens_per_s,
        "num_layers": engine.config.num_hidden_layers,
        "hidden_size": engine.config.hidden_size,
        "num_attention_heads": engine.config.num_attention_heads,
        "num_key_value_heads": engine.config.num_key_value_heads,
        "head_dim": engine.config.head_dim,
    }
    result_path = _write_jsonl(Path(args.results_dir), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()

