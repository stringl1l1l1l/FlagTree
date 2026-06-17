"""HF Qwen3 weight extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor | None = None


@dataclass(frozen=True)
class Qwen3LayerWeights:
    input_norm_weight: torch.Tensor
    post_norm_weight: torch.Tensor
    qkv_proj: LinearWeights
    o_proj: LinearWeights
    gate_up_proj: LinearWeights
    down_proj: LinearWeights
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    q_norm_eps: float
    k_norm_eps: float


@dataclass(frozen=True)
class Qwen3Weights:
    embed_tokens: torch.Tensor
    layers: tuple[Qwen3LayerWeights, ...]
    final_norm_weight: torch.Tensor
    lm_head: LinearWeights


def _tensor(param: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return param.detach().to(device=device, dtype=dtype).contiguous()


def _linear(module, *, device: torch.device, dtype: torch.dtype) -> LinearWeights:
    weight = _tensor(module.weight, device=device, dtype=dtype)
    bias = None if module.bias is None else _tensor(module.bias, device=device, dtype=dtype)
    return LinearWeights(weight=weight, bias=bias)


def _packed_linear(
    modules,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> LinearWeights:
    weights = [_tensor(module.weight, device=device, dtype=dtype) for module in modules]
    weight = torch.cat(weights, dim=0).contiguous()
    biases = [getattr(module, "bias", None) for module in modules]
    if all(bias is None for bias in biases):
        bias = None
    elif any(bias is None for bias in biases):
        raise ValueError("Cannot pack linear layers when only some projections have bias")
    else:
        bias = torch.cat([_tensor(bias, device=device, dtype=dtype) for bias in biases], dim=0).contiguous()
    return LinearWeights(weight=weight, bias=bias)


def _norm_eps(module, default: float) -> float:
    return float(getattr(module, "variance_epsilon", getattr(module, "eps", default)))


def extract_qwen3_weights(model, *, device: torch.device, dtype: torch.dtype, default_eps: float) -> Qwen3Weights:
    """Extract the subset of HF Qwen3 weights used by the TLE engine."""
    if not hasattr(model, "model"):
        raise ValueError("Expected a Qwen-style AutoModelForCausalLM with a .model attribute")
    base = model.model
    layers: list[Qwen3LayerWeights] = []
    for layer in base.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        if not hasattr(attn, "q_norm") or not hasattr(attn, "k_norm"):
            raise ValueError("Qwen3 attention q_norm/k_norm modules are required")
        input_norm_weight = _tensor(layer.input_layernorm.weight, device=device, dtype=dtype)
        post_norm_weight = _tensor(layer.post_attention_layernorm.weight, device=device, dtype=dtype)
        layers.append(
            Qwen3LayerWeights(
                input_norm_weight=input_norm_weight,
                post_norm_weight=post_norm_weight,
                qkv_proj=_packed_linear((attn.q_proj, attn.k_proj, attn.v_proj), device=device, dtype=dtype),
                o_proj=_linear(attn.o_proj, device=device, dtype=dtype),
                gate_up_proj=_packed_linear((mlp.gate_proj, mlp.up_proj), device=device, dtype=dtype),
                down_proj=_linear(mlp.down_proj, device=device, dtype=dtype),
                q_norm_weight=_tensor(attn.q_norm.weight, device=device, dtype=dtype),
                k_norm_weight=_tensor(attn.k_norm.weight, device=device, dtype=dtype),
                q_norm_eps=_norm_eps(attn.q_norm, default_eps),
                k_norm_eps=_norm_eps(attn.k_norm, default_eps),
            ))

    lm_head_weight = model.lm_head.weight if hasattr(model, "lm_head") else base.embed_tokens.weight
    lm_head_bias = getattr(getattr(model, "lm_head", None), "bias", None)
    final_norm_weight = _tensor(base.norm.weight, device=device, dtype=dtype)
    return Qwen3Weights(
        embed_tokens=_tensor(base.embed_tokens.weight, device=device, dtype=dtype),
        layers=tuple(layers),
        final_norm_weight=final_norm_weight,
        lm_head=LinearWeights(
            weight=_tensor(lm_head_weight, device=device, dtype=dtype),
            bias=None if lm_head_bias is None else _tensor(lm_head_bias, device=device, dtype=dtype),
        ),
    )
