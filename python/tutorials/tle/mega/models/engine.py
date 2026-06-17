"""TLE-first Qwen3 decode engine for tutorial serving and benchmarking."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from ..kernels import attention_decode, attention_ws, embedding, fused_add_rms_norm
    from ..kernels import head_rmsnorm_rope
    from ..kernels import linear, lm_head, qkv_linear, rms_norm, silu_and_mul_out, store_cache
except ImportError:  # pragma: no cover - supports direct script execution from mega/
    from kernels import attention_decode, attention_ws, embedding, fused_add_rms_norm
    from kernels import head_rmsnorm_rope
    from kernels import linear, lm_head, qkv_linear, rms_norm, silu_and_mul_out, store_cache

from .config import Qwen3TLEConfig
from .weights import LinearWeights, Qwen3Weights, extract_qwen3_weights


@dataclass
class KVCache:
    k: list[torch.Tensor]
    v: list[torch.Tensor]
    batch_size: int
    max_seq_len: int
    past_len: int = 0


def _parse_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError("Only bf16/bfloat16 is supported by the tutorial TLE kernels") from exc


def _eos_ids(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(v) for v in value}


class Qwen3TLEEngine:
    """Qwen3 inference engine whose transformer compute path uses tutorial TLE kernels."""

    def __init__(
        self,
        *,
        config: Qwen3TLEConfig,
        weights: Qwen3Weights,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
        attention_backend: str = "ws",
    ) -> None:
        if dtype is not torch.bfloat16:
            raise ValueError("Qwen3TLEEngine currently supports bf16 only")
        if attention_backend != "ws":
            raise ValueError(f"attention_backend must be 'ws', got {attention_backend!r}")
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.attention_backend = attention_backend
        self.cos, self.sin = self._build_rotary_cache(max_seq_len)
        self.cache: KVCache | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "Qwen/Qwen3-32B",
        *,
        device: str = "cuda",
        dtype: str | torch.dtype = "bf16",
        max_seq_len: int | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        attention_backend: str = "ws",
    ) -> "Qwen3TLEEngine":
        torch_dtype = _parse_dtype(dtype)
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError("The TLE Qwen3 tutorial engine requires a CUDA device")
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                               local_files_only=local_files_only)
        config = Qwen3TLEConfig.from_hf_config(hf_config, model_path)
        max_len = int(max_seq_len or config.max_position_embeddings)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                                  local_files_only=local_files_only)
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()
        weights = extract_qwen3_weights(model, device=torch_device, dtype=torch_dtype, default_eps=config.rms_norm_eps)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return cls(
            config=config,
            weights=weights,
            tokenizer=tokenizer,
            device=torch_device,
            dtype=torch_dtype,
            max_seq_len=max_len,
            attention_backend=attention_backend,
        )

    def _build_rotary_cache(self, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.config.head_dim // 2
        inv_freq = 1.0 / (self.config.rope_theta**(
            torch.arange(0, self.config.head_dim, 2, device=self.device, dtype=torch.float32) / self.config.head_dim))
        positions = torch.arange(max_seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq[:half_dim])
        return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()

    def reset_cache(self, batch_size: int = 1, max_seq_len: int | None = None) -> None:
        max_len = int(max_seq_len or self.max_seq_len)
        if max_len > self.max_seq_len:
            self.max_seq_len = max_len
            self.cos, self.sin = self._build_rotary_cache(max_len)
        if (self.cache is None or self.cache.batch_size != batch_size or self.cache.max_seq_len != max_len):
            k_cache = []
            v_cache = []
            shape = (batch_size, max_len, self.config.num_key_value_heads, self.config.head_dim)
            for _ in range(self.config.num_hidden_layers):
                k_cache.append(torch.empty(shape, device=self.device, dtype=self.dtype))
                v_cache.append(torch.empty(shape, device=self.device, dtype=self.dtype))
            self.cache = KVCache(k=k_cache, v=v_cache, batch_size=batch_size, max_seq_len=max_len)
        self.cache.past_len = 0

    def _linear(self, x: torch.Tensor, weights: LinearWeights) -> torch.Tensor:
        return linear(x.contiguous(), weights.weight, weights.bias)

    def _lm_head(self, x: torch.Tensor, weights: LinearWeights) -> torch.Tensor:
        return lm_head(x.contiguous(), weights.weight, weights.bias)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rms_norm(x.contiguous(), (self.config.hidden_size, ), weight, self.config.rms_norm_eps)

    def _fused_add_rms_norm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_add_rms_norm(x.contiguous(), residual.contiguous(), (self.config.hidden_size, ), weight,
                                  self.config.rms_norm_eps)

    def _qkv_linear(self, x: torch.Tensor, weights: LinearWeights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.config.num_attention_heads * self.config.head_dim
        kv_dim = self.config.num_key_value_heads * self.config.head_dim
        return qkv_linear(x.contiguous(), weights.weight, weights.bias, q_dim=q_dim, kv_dim=kv_dim)

    def _mlp_activation(self, x: torch.Tensor, weights: LinearWeights) -> torch.Tensor:
        packed = self._linear(x, weights)
        gate, up = packed.split((self.config.intermediate_size, self.config.intermediate_size), dim=-1)
        out = torch.empty_like(gate)
        return silu_and_mul_out(gate, up, out)

    def _attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        q_len: int,
        start_pos: int,
        kv_len: int,
        sm_scale: float,
    ) -> torch.Tensor:
        if q_len == 1:
            fn = attention_decode
        else:
            fn = attention_ws
        return fn(q, k_cache, v_cache, q_len=q_len, start_pos=start_pos, kv_len=kv_len, sm_scale=sm_scale)

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
            hidden_flat = self._rms_norm(x_flat, layer.input_norm_weight)
        else:
            residual_flat = residual.reshape(tokens, hidden).contiguous()
            hidden_flat, residual_flat = self._fused_add_rms_norm(x_flat, residual_flat, layer.input_norm_weight)

        q_flat, k_flat, v_flat = self._qkv_linear(hidden_flat, layer.qkv_proj)
        q = q_flat.view(tokens, self.config.num_attention_heads, self.config.head_dim)
        k = k_flat.view(tokens, self.config.num_key_value_heads, self.config.head_dim)
        v = v_flat.view(tokens, self.config.num_key_value_heads, self.config.head_dim)

        q = head_rmsnorm_rope(q, layer.q_norm_weight, self.cos, self.sin, q_len=q_len, start_pos=start_pos,
                              eps=layer.q_norm_eps)
        k = head_rmsnorm_rope(k, layer.k_norm_weight, self.cos, self.sin, q_len=q_len, start_pos=start_pos,
                              eps=layer.k_norm_eps)
        store_cache(k, self.cache.k[layer_idx], q_len=q_len, start_pos=start_pos)
        store_cache(v, self.cache.v[layer_idx], q_len=q_len, start_pos=start_pos)

        attn = self._attention(
            q,
            self.cache.k[layer_idx],
            self.cache.v[layer_idx],
            q_len=q_len,
            start_pos=start_pos,
            kv_len=start_pos + q_len,
            sm_scale=1.0 / math.sqrt(self.config.head_dim),
        )
        attn_flat = attn.reshape(tokens, q_dim).contiguous()
        attn_out = self._linear(attn_flat, layer.o_proj)

        hidden_flat, residual_flat = self._fused_add_rms_norm(attn_out, residual_flat, layer.post_norm_weight)
        hidden_act = self._mlp_activation(hidden_flat, layer.gate_up_proj)
        out = self._linear(hidden_act, layer.down_proj)
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

        x = embedding(input_ids.reshape(-1), self.weights.embed_tokens)
        x = x.view(batch_size, q_len, self.config.hidden_size)
        residual = None
        for layer_idx in range(self.config.num_hidden_layers):
            x, residual = self._run_layer(x, residual, layer_idx, start_pos)

        last_hidden = x[:, -1, :].contiguous().view(batch_size, self.config.hidden_size)
        if residual is None:
            last_hidden = self._rms_norm(last_hidden, self.weights.final_norm_weight)
        else:
            last_residual = residual[:, -1, :].contiguous().view(batch_size, self.config.hidden_size)
            last_hidden, _ = self._fused_add_rms_norm(last_hidden, last_residual, self.weights.final_norm_weight)
        return self._lm_head(last_hidden, self.weights.lm_head)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.cache is None or self.cache.batch_size != input_ids.shape[0]:
            self.reset_cache(batch_size=input_ids.shape[0])
        assert self.cache is not None
        logits = self.forward_chunk(input_ids, start_pos=0)
        self.cache.past_len = int(input_ids.shape[1])
        return logits

    @torch.inference_mode()
    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("decode called before prefill/reset_cache")
        if input_ids.shape[1] != 1:
            raise ValueError("decode expects one token per batch")
        start_pos = self.cache.past_len
        logits = self.forward_chunk(input_ids, start_pos=start_pos)
        self.cache.past_len += 1
        return logits

    def sample_next_token(self, logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= top_p
            filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(filtered, num_samples=1)
            return torch.gather(sorted_idx, dim=-1, index=sampled)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.8,
        stop_token_ids: Iterable[int] | None = None,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("interactive generation currently supports batch_size=1")
        self.reset_cache(batch_size=1, max_seq_len=max(self.max_seq_len, input_ids.shape[1] + max_new_tokens))
        logits = self.prefill(input_ids)
        stop_ids = set(stop_token_ids or ()) | _eos_ids(self.config.eos_token_id)
        next_id = None
        for _ in range(max_new_tokens):
            next_id = self.sample_next_token(logits, temperature=temperature, top_p=top_p)
            token = int(next_id.item())
            yield token
            if token in stop_ids:
                break
            logits = self.decode(next_id.to(device=self.device, dtype=torch.long))
