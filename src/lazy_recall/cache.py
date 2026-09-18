"""LazyRecall cache with causal absolute-position metadata."""

from __future__ import annotations

from bisect import bisect_left
from typing import Dict, List, Sequence, Tuple

import torch
import numpy as np

from ._core.cache import FlatCache, RemovedKV
from ._core.cuda_ops import append_flattened_cache, compact_flattened_kv
from .cuda_ops import extension


class LazyRecallCache(FlatCache):
    """Track the original position of every surviving physical KV slot."""

    def __init__(self, transport=None) -> None:
        super().__init__()
        self.position_cache: List[torch.Tensor] = []
        self._append_positions: torch.Tensor | None = None
        self.transport = transport
        self.recalled_identities = set()
        self._recalled_locations = {}
        self.recalled_visible_forwards = 0
        self.recalled_resident_peak = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        batch_size, num_heads, sequence_length, _ = key_states.shape
        if batch_size != 1:
            raise ValueError("LazyRecall supports batch_size=1 only")
        if (len(self.key_cache) > layer_idx and key_states.is_cuda
                and not self._audit_insert and not torch.is_grad_enabled()):
            metadata = self.metadata_list[layer_idx]
            if num_heads != metadata.num_key_value_heads:
                raise ValueError("KV head count changed during append")
            logical_start = self._seen_tokens if layer_idx == 0 else self._seen_tokens - sequence_length
            keys, values, positions, cumulative, lengths = extension().append(
                self.key_cache[layer_idx].contiguous(), self.value_cache[layer_idx].contiguous(),
                self.position_cache[layer_idx].contiguous(), key_states.contiguous(),
                value_states.contiguous(), metadata.cu_seqlens_k,
                logical_start, metadata.max_seqlen_k,
            )
            self.key_cache[layer_idx], self.value_cache[layer_idx] = keys, values
            self.position_cache[layer_idx] = positions
            metadata.cu_seqlens_k, metadata.head_lens = cumulative, lengths
            metadata.head_lens_cpu = [length + sequence_length for length in metadata.head_lens_cpu]
            metadata.max_seqlen_k += sequence_length
            if layer_idx == 0:
                self._seen_tokens += sequence_length
            return keys, values
        if layer_idx == 0:
            start = int(self._seen_tokens)
            self._append_positions = torch.arange(
                start,
                start + sequence_length,
                dtype=torch.float32,
                device=key_states.device,
            )
        if self._append_positions is None or (
            self._append_positions.numel() != sequence_length
        ):
            raise RuntimeError("Layer updates disagree on logical token positions")

        is_new_layer = len(self.key_cache) <= layer_idx
        position_states = self._append_positions.reshape(
            1, 1, sequence_length, 1
        ).expand(batch_size, num_heads, sequence_length, 1).contiguous()
        if not is_new_layer:
            metadata = self.metadata_list[layer_idx]
            appended_positions = append_flattened_cache(
                self.position_cache[layer_idx],
                position_states,
                metadata.head_lens,
                metadata.cu_seqlens_k,
            )
        output = super().update(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )
        if is_new_layer:
            if layer_idx != len(self.position_cache):
                raise RuntimeError("Cache layers were initialized out of order")
            self.position_cache.append(position_states.reshape(-1, 1))
        else:
            self.position_cache[layer_idx] = appended_positions
        return output

    def remove_positions(
        self,
        removals: Dict[Tuple[int, int], Sequence[int]],
        capture_removed: bool = False,
    ) -> List[RemovedKV]:
        by_layer: Dict[int, Dict[int, List[int]]] = {}
        for (layer, head), raw_positions in removals.items():
            positions = sorted(set(int(position) for position in raw_positions))
            if not positions:
                continue
            head_length = self.metadata_list[layer].head_lens_cpu[head]
            if positions[0] < 0 or positions[-1] >= head_length:
                raise IndexError(
                    f"Invalid LazyRecall removal in layer={layer}, head={head}: "
                    f"length={head_length}, positions={positions[:4]}"
                )
            by_layer.setdefault(layer, {})[head] = positions

        records: List[RemovedKV] = []
        if self.transport is not None:
            self.transport.record_evicted(self, removals)
        for layer, head_map in by_layer.items():
            metadata = self.metadata_list[layer]
            keys = self.key_cache[layer]
            values = self.value_cache[layer]
            bounds = [0]
            for length in metadata.head_lens_cpu:
                bounds.append(bounds[-1] + int(length))
            position_rows = self.position_cache[layer]
            if position_rows.shape[0] != keys.shape[0]:
                raise RuntimeError("Position and KV rows differ before compaction")
            absolute_positions = []
            removed_heads = []
            new_head_lens = list(metadata.head_lens_cpu)
            new_prompt_lens = (
                None if metadata.prompt_lens_cpu is None
                else list(metadata.prompt_lens_cpu)
            )
            prompt_lens_changed = False
            for head, positions in head_map.items():
                absolute_positions.extend(bounds[head] + position for position in positions)
                removed_heads.extend([head] * len(positions))
                new_head_lens[head] -= len(positions)
                locations = self._recalled_locations.pop((layer, head), {})
                shifted = {}
                removed_set = set(positions)
                for relative, logical in locations.items():
                    if relative in removed_set:
                        self.recalled_identities.discard((layer, head, logical))
                    else:
                        shifted[relative - bisect_left(positions, relative)] = logical
                if shifted:
                    self._recalled_locations[(layer, head)] = shifted
                if new_prompt_lens is not None:
                    removed_prompt = sum(position < new_prompt_lens[head] for position in positions)
                    new_prompt_lens[head] -= removed_prompt
                    prompt_lens_changed = prompt_lens_changed or removed_prompt > 0
            absolute = torch.tensor(absolute_positions, dtype=torch.long, device=keys.device)
            if capture_removed:
                selected_keys = keys.index_select(0, absolute).detach().cpu()
                selected_values = values.index_select(0, absolute).detach().cpu()
                records.extend(
                    RemovedKV(layer, head, key, value)
                    for head, key, value in zip(removed_heads, selected_keys, selected_values)
                )
            if keys.is_cuda:
                ordered = torch.tensor(sorted(absolute_positions), dtype=torch.int64, device=keys.device)
                compacted_keys, compacted_values, compacted_positions = extension().compact(
                    keys.contiguous(), values.contiguous(), position_rows.contiguous(), ordered,
                )
            else:
                keep_mask = torch.ones(position_rows.shape[0], dtype=torch.bool, device=keys.device)
                keep_mask[absolute] = False
                keep_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
                compacted_keys, compacted_values = compact_flattened_kv(keys, values, keep_indices)
                compacted_positions = position_rows.index_select(0, keep_indices)
            self.key_cache[layer] = compacted_keys
            self.value_cache[layer] = compacted_values
            self.position_cache[layer] = compacted_positions
            metadata.head_lens_cpu = new_head_lens
            metadata.head_lens = torch.tensor(new_head_lens, dtype=torch.int32, device=keys.device)
            metadata.cu_seqlens_k = torch.tensor(
                np.concatenate(([0], np.cumsum(new_head_lens))), dtype=torch.int32, device=keys.device,
            )
            metadata.max_seqlen_k = max(new_head_lens, default=0)
            if new_prompt_lens is not None and prompt_lens_changed:
                metadata.replace_prompt_lens(new_prompt_lens)
        self.assert_position_alignment()
        return records

    def layer_positions_tensor(
        self,
        layer: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.position_cache[layer].reshape(-1).to(device=device)

    def active_position_sets(self, metadata):
        result = {}
        for layer in np.unique(metadata[:, 0]):
            layer = int(layer)
            positions = self.position_cache[layer].reshape(-1).detach().cpu().numpy().astype(np.int64)
            lengths = self.metadata_list[layer].head_lens_cpu
            bounds = np.concatenate(([0], np.cumsum(lengths)))
            for head in np.unique(metadata[metadata[:, 0] == layer, 1]):
                head = int(head)
                result[(layer, head)] = set(positions[bounds[head]:bounds[head + 1]].tolist())
        return result

    @torch.no_grad()
    def insert_payload(self, payload, metadata):
        from .recall.planning import merge_plan

        metadata = np.asarray(metadata, dtype=np.int64).reshape(-1, 3)
        if payload.shape[:2] != (len(metadata), 2):
            raise ValueError("Recall payload/identity shape mismatch")
        before, logical = self.total_slots(), self._seen_tokens
        for layer in np.unique(metadata[:, 0]):
            layer = int(layer)
            rows = np.flatnonzero(metadata[:, 0] == layer)
            device = self.key_cache[layer].device
            indices = torch.tensor(rows, device=device, dtype=torch.long)
            recalled = payload.index_select(0, indices).contiguous()
            recalled_meta = metadata[rows]
            previous = self.position_cache[layer].reshape(-1).detach().cpu().numpy()
            mapping, lengths = merge_plan(previous, self.metadata_list[layer].head_lens_cpu, recalled_meta)
            old_bounds = np.concatenate(([0], np.cumsum(self.metadata_list[layer].head_lens_cpu)))
            new_bounds = np.concatenate(([0], np.cumsum(lengths)))
            existing_mask = mapping >= 0
            existing_locations = np.empty(len(previous), dtype=np.int64)
            existing_locations[mapping[existing_mask]] = np.flatnonzero(existing_mask)
            added_locations = np.empty(len(recalled_meta), dtype=np.int64)
            added_locations[-mapping[~existing_mask] - 1] = np.flatnonzero(~existing_mask)
            for head in range(len(lengths)):
                old_locations = self._recalled_locations.pop((layer, head), {})
                updated = {int(existing_locations[old_bounds[head] + relative] - new_bounds[head]): logical
                           for relative, logical in old_locations.items()}
                for added in np.flatnonzero(recalled_meta[:, 1] == head):
                    updated[int(added_locations[added] - new_bounds[head])] = int(recalled_meta[added, 2])
                if updated:
                    self._recalled_locations[(layer, head)] = updated
            source_map = torch.tensor(mapping, device=device, dtype=torch.long)
            positions = torch.tensor(recalled_meta[:, 2], device=device, dtype=torch.float32)
            if device.type == "cuda":
                keys, values, positions = extension().merge(
                    self.key_cache[layer].contiguous(), self.value_cache[layer].contiguous(),
                    self.position_cache[layer].contiguous(), recalled, positions, source_map,
                )
            else:
                old_count = len(previous)
                gather = torch.where(source_map >= 0, source_map, old_count - source_map - 1)
                keys = torch.cat((self.key_cache[layer], recalled[:, 0])).index_select(0, gather)
                values = torch.cat((self.value_cache[layer], recalled[:, 1])).index_select(0, gather)
                positions = torch.cat((self.position_cache[layer].reshape(-1), positions)).index_select(0, gather).view(-1, 1)
            self.key_cache[layer], self.value_cache[layer], self.position_cache[layer] = keys, values, positions
            self.metadata_list[layer].replace_head_lens(lengths)
        if self.total_slots() != before + len(metadata) or self._seen_tokens != logical:
            raise AssertionError("Recall changed logical time or inserted an incorrect slot count")
        self.recalled_identities.update(map(tuple, metadata.tolist()))
        self.recalled_resident_peak = max(self.recalled_resident_peak, len(self.recalled_identities))
        self.assert_position_alignment()
        return len(metadata)

    def assert_position_alignment(self) -> None:
        if len(self.position_cache) != len(self.metadata_list):
            raise RuntimeError("Position/cache layer counts differ")
        for layer, metadata in enumerate(self.metadata_list):
            observed = int(self.position_cache[layer].shape[0])
            expected = sum(metadata.head_lens_cpu)
            if observed != expected:
                raise RuntimeError(
                    f"Position/KV lengths differ in layer {layer}: "
                    f"{observed} != {expected}"
                )

