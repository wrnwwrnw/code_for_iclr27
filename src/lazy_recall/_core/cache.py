"""Flattened, variable-length per-head KV cache for LazyRecall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import Cache

from .cuda_ops import append_flattened_cache, append_flattened_cache_pytorch
from .cuda_ops import compact_flattened_kv


@dataclass
class CacheMetadata:
    decoding_cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    head_lens: torch.Tensor
    head_lens_cpu: List[int]
    max_seqlen_k: int
    cu_offset: torch.Tensor
    batch_size: int
    num_key_value_heads: int
    prompt_lens: Optional[torch.Tensor] = None
    prompt_lens_cpu: Optional[List[int]] = None

    @classmethod
    def from_states(cls, states: torch.Tensor) -> "CacheMetadata":
        batch_size, num_heads, sequence_length, _ = states.shape
        total_heads = batch_size * num_heads
        device = states.device
        head_lens = torch.full(
            (total_heads,),
            sequence_length,
            dtype=torch.int32,
            device=device,
        )
        cu_seqlens_k = torch.zeros(
            total_heads + 1, dtype=torch.int32, device=device
        )
        cu_seqlens_k[1:] = torch.cumsum(head_lens, dim=0)
        decoding_cu_seqlens_q = torch.arange(
            total_heads + 1, dtype=torch.int32, device=device
        )
        return cls(
            decoding_cu_seqlens_q=decoding_cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            head_lens=head_lens,
            head_lens_cpu=[sequence_length] * total_heads,
            max_seqlen_k=sequence_length,
            cu_offset=torch.arange(
                total_heads + 1, dtype=torch.int32, device=device
            ),
            batch_size=batch_size,
            num_key_value_heads=num_heads,
        )

    def freeze_prompt(self) -> None:
        if self.prompt_lens is None:
            self.prompt_lens = self.head_lens.clone()
            self.prompt_lens_cpu = list(self.head_lens_cpu)

    def append(self, sequence_length: int) -> None:
        self.head_lens.add_(sequence_length)
        self.cu_seqlens_k.add_(self.cu_offset, alpha=sequence_length)
        self.head_lens_cpu = [
            length + sequence_length for length in self.head_lens_cpu
        ]
        self.max_seqlen_k += sequence_length

    def replace_head_lens(self, head_lens: Sequence[int]) -> None:
        self.head_lens_cpu = [int(length) for length in head_lens]
        self.head_lens = torch.tensor(
            self.head_lens_cpu,
            dtype=torch.int32,
            device=self.head_lens.device,
        )
        self.cu_seqlens_k = torch.zeros(
            self.head_lens.numel() + 1,
            dtype=torch.int32,
            device=self.head_lens.device,
        )
        self.cu_seqlens_k[1:] = torch.cumsum(self.head_lens, dim=0)
        self.max_seqlen_k = max(self.head_lens_cpu, default=0)

    def replace_prompt_lens(self, prompt_lens: Sequence[int]) -> None:
        self.prompt_lens_cpu = [int(length) for length in prompt_lens]
        self.prompt_lens = torch.tensor(
            self.prompt_lens_cpu,
            dtype=torch.int32,
            device=self.head_lens.device,
        )


@dataclass(frozen=True)
class RemovedKV:
    layer: int
    head: int
    key: torch.Tensor
    value: torch.Tensor


class FlatCache(Cache):
    """Cache layout ``[head0 tokens | head1 tokens | ...]`` per layer."""

    is_compileable = False
    is_sliding: List[bool] = []

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.metadata_list: List[CacheMetadata] = []
        self._seen_tokens = 0
        self._audit_insert = False

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        for layer_index in range(len(self)):
            yield self.key_cache[layer_index], self.value_cache[layer_index]

    def __getitem__(self, layer_index: int):
        if layer_index >= len(self):
            raise KeyError(
                f"Cache has {len(self)} layers, requested layer {layer_index}"
            )
        return self.key_cache[layer_index], self.value_cache[layer_index]

    def get_seq_length(
        self, layer_idx: Optional[int] = 0, cache_position=None
    ) -> int:
        del cache_position
        if layer_idx is None or layer_idx >= len(self.key_cache):
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return -1

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx):
        del layer_idx
        length = query_length if isinstance(query_length, int) else query_length.shape[0]
        return self._seen_tokens + length, 0

    def to_legacy_cache(self):
        raise NotImplementedError("FlatCache has unequal per-head lengths")

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        del past_key_values
        raise NotImplementedError("FlatCache cannot import a dense cache")

    def set_audit_insert(self, enabled: bool) -> None:
        self._audit_insert = bool(enabled)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs
        if len(self.key_cache) <= layer_idx:
            batch_size, num_heads, sequence_length, head_dim = key_states.shape
            self.key_cache.append(
                key_states.reshape(
                    batch_size * num_heads * sequence_length, head_dim
                )
            )
            self.value_cache.append(
                value_states.reshape(
                    batch_size * num_heads * sequence_length, head_dim
                )
            )
            self.metadata_list.append(CacheMetadata.from_states(key_states))
            if layer_idx == 0:
                self._seen_tokens = sequence_length
        else:
            metadata = self.metadata_list[layer_idx]
            if self._audit_insert:
                self.key_cache[layer_idx] = append_flattened_cache_pytorch(
                    self.key_cache[layer_idx],
                    key_states,
                    metadata.head_lens_cpu,
                )
                self.value_cache[layer_idx] = append_flattened_cache_pytorch(
                    self.value_cache[layer_idx],
                    value_states,
                    metadata.head_lens_cpu,
                )
            else:
                self.key_cache[layer_idx] = append_flattened_cache(
                    self.key_cache[layer_idx],
                    key_states,
                    metadata.head_lens,
                    metadata.cu_seqlens_k,
                )
                self.value_cache[layer_idx] = append_flattened_cache(
                    self.value_cache[layer_idx],
                    value_states,
                    metadata.head_lens,
                    metadata.cu_seqlens_k,
                )
            metadata.append(int(key_states.shape[2]))
            if layer_idx == 0:
                self._seen_tokens += int(key_states.shape[2])
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def detach_(self) -> None:
        self.key_cache = [tensor.detach() for tensor in self.key_cache]
        self.value_cache = [tensor.detach() for tensor in self.value_cache]

    def freeze_prompt(self) -> None:
        for metadata in self.metadata_list:
            metadata.freeze_prompt()

    def total_slots(self) -> int:
        return sum(sum(meta.head_lens_cpu) for meta in self.metadata_list)

    def prompt_slots(self) -> int:
        return sum(
            sum(meta.prompt_lens_cpu)
            for meta in self.metadata_list
            if meta.prompt_lens_cpu is not None
        )

    def generation_slots(self) -> int:
        return self.total_slots() - self.prompt_slots()
