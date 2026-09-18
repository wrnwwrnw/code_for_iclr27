"""Dense text Qwen3.5: full-attention KV management and native DeltaNet state."""

from contextlib import contextmanager
from copy import copy
from types import MethodType, SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from ..attention import packed_attention
from ..cache import LazyRecallCache
from . import text_config


def full_layer_indices(config):
    config = text_config(config)
    kinds = getattr(config, "layer_types", ())
    if len(kinds) != config.num_hidden_layers or set(kinds) - {"linear_attention", "full_attention"}:
        raise ValueError("Qwen3.5 requires explicit full/linear layer types")
    indices = tuple(index for index, kind in enumerate(kinds) if kind == "full_attention")
    if not indices or config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Invalid full-attention/GQA layout")
    return indices


class FullAttentionView:
    attention_output_gate = True

    def __init__(self, model):
        config = text_config(model.config)
        self.indices = full_layer_indices(config)
        if not hasattr(model.model, "layers"):
            raise ValueError("Load Qwen3_5ForCausalLM, not a multimodal conditional-generation wrapper")
        self._model = model
        self.model = SimpleNamespace(layers=[model.model.layers[index] for index in self.indices])
        self.config = SimpleNamespace(
            num_hidden_layers=len(self.indices), num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
            hidden_size=config.hidden_size,
        )

    def get_output_embeddings(self):
        return self._model.get_output_embeddings()


class HybridCache(DynamicCache):
    def __init__(self, config, full_cache):
        super().__init__(config=config)
        self.indices = full_layer_indices(config)
        self.slots = {layer: slot for slot, layer in enumerate(self.indices)}
        self.full_cache = full_cache
        self.audit_commits = 0

    def get_seq_length(self, layer_idx=0, cache_position=None):
        return int(self.full_cache._seen_tokens)

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length() + int(query_length), 0

    def get_max_cache_shape(self, layer_idx=0):
        return -1

    def linear_state_bytes(self):
        return sum(tensor.numel() * tensor.element_size()
                   for layer in self.layers if isinstance(layer, LinearAttentionLayer)
                   for name in ("conv_states", "recurrent_states")
                   for tensor in getattr(layer, name).values() if tensor is not None)


class AuditLinearLayer(LinearAttentionLayer):
    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        self.recurrent_states[state_idx] = recurrent_states.detach().clone()
        return self.recurrent_states[state_idx]


def gated_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                  past_key_values=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value")
    if not isinstance(cache, HybridCache):
        raise TypeError("Qwen3.5 LazyRecall attention requires HybridCache")
    batch, length, _ = hidden_states.shape
    slot = cache.slots[self.layer_idx]
    if batch != 1 or (length != 1 and len(cache.full_cache) > slot) or attention_mask is not None:
        raise ValueError("Only unpadded batch=1 prefill and single-token decode are supported")
    if position_embeddings is None:
        raise ValueError("Native Qwen3.5 partial/interleaved RoPE embeddings are required")
    if cache.full_cache._audit_insert and not hidden_states.requires_grad:
        hidden_states = hidden_states.detach().requires_grad_(True)
    dimension = self.head_dim
    queries, gate = self.q_proj(hidden_states).view(batch, length, -1, 2 * dimension).chunk(2, dim=-1)
    queries = self.q_norm(queries).transpose(1, 2)
    keys = self.k_norm(self.k_proj(hidden_states).view(batch, length, -1, dimension)).transpose(1, 2)
    values = self.v_proj(hidden_states).view(batch, length, -1, dimension).transpose(1, 2)
    queries, keys = apply_rotary_pos_emb(queries, keys, *position_embeddings)
    gate = gate.reshape(batch, length, -1).sigmoid()
    if length == 1:
        self._last_query_states = queries
        self._last_output_gate = gate.detach()
    keys, values = cache.full_cache.update(keys, values, slot)
    result = packed_attention(queries, keys, values, cache.full_cache.metadata_list[slot],
                              length, self.config.num_key_value_heads)
    return self.o_proj(result * gate), None


class Qwen35Backend:
    def __init__(self, model, transport=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        if not isinstance(model, Qwen3_5ForCausalLM):
            raise ValueError("Only the dense Qwen3_5ForCausalLM text decoder is supported")
        if getattr(model, "_lazy_recall_in_use", False):
            raise RuntimeError("A model instance may have only one active LazyRecall request")
        self.view = FullAttentionView(model)
        self.model = model
        self.cache = LazyRecallCache(transport)
        self.model_cache = HybridCache(text_config(model.config), self.cache)
        self.originals = []
        for block in self.view.model.layers:
            attention = block.self_attn
            for name in ("q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj"):
                if not hasattr(attention, name):
                    raise ValueError(f"Missing Qwen attention component: {name}")
        model._lazy_recall_in_use = True
        for block in self.view.model.layers:
            attention = block.self_attn
            self.originals.append((attention, attention.forward))
            attention.forward = MethodType(gated_forward, attention)

    @contextmanager
    def audit_context(self):
        """Differentiate one cached step, then commit its recurrent state exactly once."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_causal_conv1d_update, torch_recurrent_gated_delta_rule,
        )

        states, functions = [], []
        committed = False
        try:
            for index, layer in enumerate(self.model_cache.layers):
                if index in self.model_cache.slots:
                    continue
                if type(layer) is not LinearAttentionLayer or not all(layer.is_recurrent_states_initialized.values()):
                    raise RuntimeError("Audit requires initialized native DeltaNet state")
                proxy = copy(layer)
                proxy.__class__ = AuditLinearLayer
                for name in ("conv_states", "recurrent_states"):
                    setattr(proxy, name, {slot: tensor.detach().clone()
                                          for slot, tensor in getattr(layer, name).items()})
                states.append((index, layer, proxy))
                self.model_cache.layers[index] = proxy
                module = self.model.model.layers[index].linear_attn
                functions.append((module, module.causal_conv1d_update, module.recurrent_gated_delta_rule))
                module.causal_conv1d_update = torch_causal_conv1d_update
                module.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            yield
            committed = True
        finally:
            for module, conv, recurrent in functions:
                module.causal_conv1d_update, module.recurrent_gated_delta_rule = conv, recurrent
            for index, original, proxy in states:
                self.model_cache.layers[index] = original
            if committed:
                with torch.no_grad():
                    for _, original, proxy in states:
                        for name in ("conv_states", "recurrent_states"):
                            for slot, destination in getattr(original, name).items():
                                destination.copy_(getattr(proxy, name)[slot])
                self.model_cache.audit_commits += 1

    def stats(self):
        return {
            "architecture": "qwen3_5", "full_attention_layer_indices": list(self.view.indices),
            "linear_attention_layers": len(self.model_cache.layers) - len(self.view.indices),
            "linear_state_compressed": False, "linear_state_bytes": self.model_cache.linear_state_bytes(),
            "audit_linear_state_commits": self.model_cache.audit_commits,
            "audit_delta_rule": "differentiable_torch_reference",
        }

    def close(self):
        for attention, forward in self.originals:
            attention.forward = forward
            for name in ("_last_query_states", "_last_output_gate"):
                if hasattr(attention, name):
                    delattr(attention, name)
        self.originals.clear()
        self.model._lazy_recall_in_use = False
