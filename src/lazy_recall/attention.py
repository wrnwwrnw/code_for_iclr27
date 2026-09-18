from types import MethodType

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ._core.cache import FlatCache

_deterministic = None


def flash_attn_deterministic_status():
    return _deterministic


def packed_attention(queries, keys, values, metadata, length, kv_heads):
    global _deterministic
    from flash_attn import flash_attn_varlen_func

    batch, query_heads, _, dimension = queries.shape
    groups = query_heads // kv_heads
    queries = queries.reshape(batch, kv_heads, groups, length, dimension)
    queries = queries.transpose(2, 3).reshape(-1, groups, dimension)
    cumulative_queries = metadata.decoding_cu_seqlens_q
    if length != 1:
        cumulative_queries = cumulative_queries * length
    needs_backward = torch.is_grad_enabled() and queries.requires_grad
    extra = {"deterministic": True} if needs_backward else {}
    result = flash_attn_varlen_func(
        queries, keys.reshape(-1, 1, dimension), values.reshape(-1, 1, dimension),
        cumulative_queries, metadata.cu_seqlens_k, length, metadata.max_seqlen_k,
        causal=True, dropout_p=0.0, softmax_scale=dimension ** -0.5, **extra,
    )
    if needs_backward:
        _deterministic = True
    return result.view(batch, kv_heads, length, groups, dimension).transpose(1, 2).reshape(batch, length, -1)


def varlen_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                   past_key_value=None, cache_position=None, **kwargs):
    cache = past_key_value if past_key_value is not None else kwargs.get("past_key_values")
    if not isinstance(cache, FlatCache):
        raise TypeError("LazyRecall attention requires its own cache")
    batch, length, _ = hidden_states.shape
    if batch != 1 or (length != 1 and len(cache) > self.layer_idx):
        raise ValueError("Only unpadded batch=1 prefill followed by single-token decode is supported")
    if attention_mask is not None:
        raise ValueError("Pass unpadded input IDs without an attention_mask")
    if cache._audit_insert and not hidden_states.requires_grad:
        hidden_states = hidden_states.detach().requires_grad_(True)
    query_heads = self.config.num_attention_heads
    kv_heads = self.config.num_key_value_heads
    dimension = self.head_dim
    queries = self.q_proj(hidden_states).view(batch, length, query_heads, dimension).transpose(1, 2)
    keys = self.k_proj(hidden_states).view(batch, length, kv_heads, dimension).transpose(1, 2)
    values = self.v_proj(hidden_states).view(batch, length, kv_heads, dimension).transpose(1, 2)
    if position_embeddings is None:
        raise ValueError("Transformers must supply absolute-position RoPE embeddings")
    queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
    if length == 1:
        self._last_query_states = queries
    keys, values = cache.update(keys, values, self.layer_idx, {"cache_position": cache_position})
    metadata = cache.metadata_list[self.layer_idx]
    result = packed_attention(queries, keys, values, metadata, length, kv_heads)
    return self.o_proj(result), None


class AttentionPatch:
    def __init__(self, model):
        if model.config.model_type not in ("llama", "mistral"):
            raise ValueError("This release supports Llama and full-attention Mistral architectures")
        if getattr(model.config, "sliding_window", None) is not None:
            raise ValueError("Sliding-window models are unsupported; do not silently disable their window")
        if getattr(model, "_lazy_recall_in_use", False):
            raise RuntimeError("A model instance may have only one active LazyRecall request")
        attentions = [block.self_attn for block in model.model.layers]
        if not attentions:
            raise ValueError("Model has no attention layers")
        for attention in attentions:
            if not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "layer_idx")):
                raise ValueError("Unsupported attention module layout")
        self.model, self.originals = model, []
        model._lazy_recall_in_use = True
        for attention in attentions:
            self.originals.append((attention, attention.forward))
            attention.forward = MethodType(varlen_forward, attention)

    def close(self):
        for attention, forward in self.originals:
            attention.forward = forward
            if hasattr(attention, "_last_query_states"):
                del attention._last_query_states
        self.originals.clear()
        self.model._lazy_recall_in_use = False
