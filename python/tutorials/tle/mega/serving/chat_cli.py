"""Interactive Qwen3 chat CLI backed by the tutorial TLE engine.

Example:
  python python/tutorials/tle/mega/serving/chat_cli.py --model-path Qwen/Qwen3-32B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from models import Qwen3TLEEngine  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 TLE chat CLI")
    parser.add_argument("--model-path", default="Qwen/Qwen3-32B", help="HF model id or local model directory")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0")
    parser.add_argument("--max-seq-len", type=int, default=None, help="KV cache length")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _encode_chat(tokenizer, messages, device: torch.device) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"
        input_ids = tokenizer(text, return_tensors="pt").input_ids
    return input_ids.to(device=device, dtype=torch.long)


def main() -> None:
    args = _build_parser().parse_args()
    engine = Qwen3TLEEngine.from_pretrained(
        args.model_path,
        device=args.device,
        dtype="bf16",
        max_seq_len=args.max_seq_len,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Qwen3 TLE chat. Press Ctrl-D or enter /exit to quit.", flush=True)
    while True:
        try:
            prompt = input("\nuser> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": prompt})
        input_ids = _encode_chat(engine.tokenizer, messages, engine.device)
        print("assistant> ", end="", flush=True)
        token_ids: list[int] = []
        printed = ""
        for token_id in engine.generate_ids(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            token_ids.append(token_id)
            text = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
            delta = text[len(printed):]
            printed = text
            print(delta, end="", flush=True)
        print()
        messages.append({"role": "assistant", "content": printed})


if __name__ == "__main__":
    main()
