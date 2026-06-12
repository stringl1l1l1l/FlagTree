"""Qwen3 config normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen3TLEConfig:
    model_name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    eos_token_id: int | list[int] | None
    pad_token_id: int | None

    @classmethod
    def from_hf_config(cls, hf_config: Any, model_name: str) -> "Qwen3TLEConfig":
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling not in (None, {}):
            raise ValueError(
                "This tutorial engine currently supports plain Qwen rotary embeddings only; "
                f"got rope_scaling={rope_scaling!r}")
        num_attention_heads = int(hf_config.num_attention_heads)
        head_dim = int(getattr(hf_config, "head_dim", hf_config.hidden_size // num_attention_heads))
        return cls(
            model_name=model_name,
            vocab_size=int(hf_config.vocab_size),
            hidden_size=int(hf_config.hidden_size),
            intermediate_size=int(hf_config.intermediate_size),
            num_hidden_layers=int(hf_config.num_hidden_layers),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=int(hf_config.num_key_value_heads),
            head_dim=head_dim,
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
            rope_theta=float(getattr(hf_config, "rope_theta", 1000000.0)),
            max_position_embeddings=int(getattr(hf_config, "max_position_embeddings", 40960)),
            eos_token_id=getattr(hf_config, "eos_token_id", None),
            pad_token_id=getattr(hf_config, "pad_token_id", None),
        )

