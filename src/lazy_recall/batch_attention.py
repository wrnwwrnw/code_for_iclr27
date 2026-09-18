"""One model forward over independent request/head varlen segments."""

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from . import attention as single_attention
from ._core.cache import FlatCache
from .cuda_ops import extension


class BatchCache(FlatCache):
    def __init__(self):
        super().__init__()
        self.requests = []

    def select(self, requests):
        self.requests = list(requests)

    def get_seq_length(self, layer_idx=0, cache_position=None):
        return max((request.cache._seen_tokens for request in self.requests), default=0)

    def get_mask_sizes(self, query_length, layer_idx):
        length = query_length if isinstance(query_length, int) else query_length.shape[0]
        return self.get_seq_length() + length, 0


def append_reference(requests, keys, values, layer_index):
    key_parts, value_parts, lengths = [], [], []
    for row, request in enumerate(requests):
        cache = request.cache
        packed_keys, packed_values = cache.update(keys[row:row + 1], values[row:row + 1], layer_index)
        key_parts.append(packed_keys)
        value_parts.append(packed_values)
        lengths.extend(cache.metadata_list[layer_index].head_lens_cpu)
    return torch.cat(key_parts), torch.cat(value_parts), lengths


def append_packed(requests, keys, values, layer_index):
    if (not keys.is_cuda or torch.is_grad_enabled() or keys.shape[2] != 1
            or any(len(request.cache) <= layer_index for request in requests)):
        return append_reference(requests, keys, values, layer_index)
    kernels = extension()
    if not hasattr(kernels, "batch_append"):
        raise RuntimeError("Recompile LazyRecall: the installed CUDA extension lacks batch_append")
    caches = [request.cache for request in requests]
    lengths = [list(cache.metadata_list[layer_index].head_lens_cpu) for cache in caches]
    logical = [cache._seen_tokens if layer_index == 0 else cache._seen_tokens - 1 for cache in caches]
    packed_keys, packed_values, positions = kernels.batch_append(
        [cache.key_cache[layer_index] for cache in caches], [cache.value_cache[layer_index] for cache in caches],
        [cache.position_cache[layer_index] for cache in caches], keys.contiguous(), values.contiguous(), lengths, logical)
    start = 0
    for cache, head_lengths in zip(caches, lengths):
        end = start + sum(head_lengths) + len(head_lengths)
        cache.key_cache[layer_index] = packed_keys[start:end]
        cache.value_cache[layer_index] = packed_values[start:end]
        cache.position_cache[layer_index] = positions[start:end]
        cache.metadata_list[layer_index].append(1)
        if layer_index == 0:
            cache._seen_tokens += 1
        start = end
    return packed_keys, packed_values, [length + 1 for head_lengths in lengths for length in head_lengths]


def batch_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                  past_key_value=None, cache_position=None, **kwargs):
    cache = past_key_value if past_key_value is not None else kwargs.get("past_key_values")
    if not isinstance(cache, BatchCache):
        raise TypeError("Batch attention requires BatchCache")
    requests = cache.requests
    batch, length, _ = hidden_states.shape
    if batch != len(requests) or not requests:
        raise ValueError("Active request rows differ from the model input")
    if attention_mask is not None or (length != 1 and batch != 1):
        raise ValueError("Use independent unpadded prefills, then batched one-token decode")
    if cache._audit_insert and not hidden_states.requires_grad:
        hidden_states = hidden_states.detach().requires_grad_(True)
    query_heads = self.config.num_attention_heads
    kv_heads = self.config.num_key_value_heads
    dimension = self.head_dim
    queries = self.q_proj(hidden_states).view(batch, length, query_heads, dimension).transpose(1, 2)
    keys = self.k_proj(hidden_states).view(batch, length, kv_heads, dimension).transpose(1, 2)
    values = self.v_proj(hidden_states).view(batch, length, kv_heads, dimension).transpose(1, 2)
    if position_embeddings is None:
        raise ValueError("Absolute per-request RoPE embeddings are required")
    queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
    if length == 1:
        self._last_query_states = queries
    packed_keys, packed_values, lengths = append_packed(requests, keys, values, self.layer_idx)
    cumulative = torch.tensor([0] + lengths, device=keys.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata = SimpleNamespace(cu_seqlens_k=cumulative, max_seqlen_k=max(lengths),
                               decoding_cu_seqlens_q=torch.arange(batch * kv_heads + 1, device=keys.device,
                                                                 dtype=torch.int32))
    result = single_attention.packed_attention(queries, packed_keys, packed_values, metadata, length, kv_heads)
    return self.o_proj(result), None


class BatchAttentionPatch:
    def __init__(self, model):
        if model.config.model_type not in ("llama", "mistral"):
            raise ValueError("Batched inference supports Llama and full-attention Mistral; Qwen3.5 remains batch=1")
        if getattr(model.config, "sliding_window", None) is not None:
            raise ValueError("Sliding-window Mistral is not supported")
        if getattr(model, "_lazy_recall_in_use", False):
            raise RuntimeError("A model instance may have only one active LazyRecall engine")
        self.model, self.originals = model, []
        for layer in model.model.layers:
            attention = layer.self_attn
            if not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "layer_idx")):
                raise ValueError("Unsupported batched attention layout")
        model._lazy_recall_in_use = True
        for layer in model.model.layers:
            attention = layer.self_attn
            self.originals.append((attention, attention.forward))
            attention.forward = MethodType(batch_forward, attention)

    @contextmanager
    def request_queries(self, row):
        saved = []
        try:
            for layer in self.model.model.layers:
                attention = layer.self_attn
                query = attention._last_query_states
                saved.append((attention, query))
                attention._last_query_states = query[row:row + 1].detach()
            yield
        finally:
            for attention, query in saved:
                attention._last_query_states = query

    def close(self):
        for attention, original in self.originals:
            attention.forward = original
            if hasattr(attention, "_last_query_states"):
                del attention._last_query_states
        self.originals.clear()
        self.model._lazy_recall_in_use = False
