from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AttentionForward:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    valid_scores: torch.Tensor
    attn_probs: torch.Tensor
    core_attn_output: torch.Tensor
    projected_attn_output: torch.Tensor


@dataclass(frozen=True)
class BlockForward:
    hidden_in: torch.Tensor
    ln1_out: torch.Tensor
    attention: AttentionForward
    residual_after_attn: torch.Tensor
    ln2_out: torch.Tensor
    mlp_fc: torch.Tensor
    mlp_act: torch.Tensor
    mlp_proj: torch.Tensor
    mlp_out: torch.Tensor
    hidden_out: torch.Tensor


def _split_qkv(attn_module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = attn_module.c_attn(hidden_states)
    split_size = qkv.shape[-1] // 3
    query, key, value = qkv.split(split_size, dim=2)
    num_heads = attn_module.num_heads
    head_dim = split_size // num_heads

    def reshape(x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    return reshape(query), reshape(key), reshape(value)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(batch, seq, heads * head_dim)


def manual_attention_forward(attn_module: Any, hidden_states: torch.Tensor, *, layer_idx: int) -> AttentionForward:
    query, key, value = _split_qkv(attn_module, hidden_states)
    _, _, _, head_dim = value.shape

    raw_scores = torch.matmul(query, key.transpose(-1, -2))
    if getattr(attn_module, "scale_attn_weights", True):
        raw_scores = raw_scores / math.sqrt(head_dim)
    if getattr(attn_module, "scale_attn_by_inverse_layer_idx", False):
        raw_scores = raw_scores / float(layer_idx + 1)

    q_len = query.shape[-2]
    k_len = key.shape[-2]
    if hasattr(attn_module, "bias"):
        causal_mask = attn_module.bias[:, :, k_len - q_len : k_len, :k_len].to(torch.bool)
    else:
        causal_mask = torch.tril(
            torch.ones((q_len, k_len), device=raw_scores.device, dtype=torch.bool)
        ).view(1, 1, q_len, k_len)
    valid_scores = torch.where(causal_mask, raw_scores, torch.zeros_like(raw_scores))
    mask_value = torch.finfo(raw_scores.dtype).min
    attn_scores = torch.where(causal_mask, raw_scores, torch.full_like(raw_scores, mask_value))

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = attn_module.attn_dropout(attn_probs)
    core_attn_output = torch.matmul(attn_probs, value)
    merged = _merge_heads(core_attn_output)
    attn_output = attn_module.c_proj(merged)
    attn_output = attn_module.resid_dropout(attn_output)

    return AttentionForward(
        query=query,
        key=key,
        value=value,
        valid_scores=valid_scores,
        attn_probs=attn_probs,
        core_attn_output=core_attn_output,
        projected_attn_output=attn_output,
    )


def manual_block_forward(block: Any, hidden_states: torch.Tensor, *, layer_idx: int) -> BlockForward:
    hidden_in = hidden_states
    ln1_out = block.ln_1(hidden_in)
    attention = manual_attention_forward(block.attn, ln1_out, layer_idx=layer_idx)
    residual_after_attn = hidden_in + attention.projected_attn_output
    ln2_out = block.ln_2(residual_after_attn)
    mlp_fc = block.mlp.c_fc(ln2_out)
    mlp_act = block.mlp.act(mlp_fc)
    mlp_proj = block.mlp.c_proj(mlp_act)
    mlp_out = block.mlp.dropout(mlp_proj)
    hidden_out = residual_after_attn + mlp_out
    return BlockForward(
        hidden_in=hidden_in,
        ln1_out=ln1_out,
        attention=attention,
        residual_after_attn=residual_after_attn,
        ln2_out=ln2_out,
        mlp_fc=mlp_fc,
        mlp_act=mlp_act,
        mlp_proj=mlp_proj,
        mlp_out=mlp_out,
        hidden_out=hidden_out,
    )


def embed_inputs(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    transformer = model.transformer
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
    return transformer.drop(hidden_states)


def manual_forward_with_prefixes(model: Any, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    hidden_states = embed_inputs(model, input_ids)
    prefixes: list[torch.Tensor] = []
    for layer_idx, block in enumerate(model.transformer.h):
        prefixes.append(hidden_states.detach())
        hidden_states = manual_block_forward(block, hidden_states, layer_idx=layer_idx).hidden_out
    final_hidden = model.transformer.ln_f(hidden_states)
    return final_hidden, prefixes


def manual_continue_from_hidden(model: Any, hidden_states: torch.Tensor, *, start_layer: int) -> torch.Tensor:
    current = hidden_states
    for layer_idx in range(start_layer, len(model.transformer.h)):
        current = manual_block_forward(model.transformer.h[layer_idx], current, layer_idx=layer_idx).hidden_out
    return model.transformer.ln_f(current)


def manual_patched_forward(ref_model: Any, tgt_model: Any, target_prefix: torch.Tensor, *, patch_layer: int) -> torch.Tensor:
    ref_dtype = next(ref_model.parameters()).dtype
    tgt_dtype = next(tgt_model.parameters()).dtype
    ref_hidden = target_prefix.detach().to(device=target_prefix.device, dtype=ref_dtype)
    patched_hidden = manual_block_forward(
        ref_model.transformer.h[patch_layer],
        ref_hidden,
        layer_idx=patch_layer,
    ).hidden_out
    patched_hidden = patched_hidden.to(device=target_prefix.device, dtype=tgt_dtype)
    return manual_continue_from_hidden(tgt_model, patched_hidden, start_layer=patch_layer + 1)
