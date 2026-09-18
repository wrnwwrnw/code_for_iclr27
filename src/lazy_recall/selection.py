"""Compact rank-slot queues, preserving the original torch.topk tie behavior."""

from collections.abc import Sequence

import numpy as np


class ArrayReferences(Sequence):
    def __init__(self, scores, segments, positions, coordinates):
        self.scores = np.asarray(scores)
        self.segments = np.asarray(segments, dtype=np.int64)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.coordinates = np.asarray(coordinates, dtype=np.int64).reshape(-1, 2)
        if len(self.scores) != len(self.segments) or len(self.scores) != len(self.positions):
            raise ValueError("Queue column lengths differ")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ArrayReferences(self.scores[index], self.segments[index],
                                   self.positions[index], self.coordinates)
        from ._core.scoring import KVReference

        layer, head = self.coordinates[self.segments[index]]
        return KVReference(float(self.scores[index]), int(layer), int(head), int(self.positions[index]))

    def groups(self):
        if not len(self):
            return {}
        order = np.argsort(self.segments, kind="stable")
        sorted_segments = self.segments[order]
        boundaries = np.flatnonzero(sorted_segments[1:] != sorted_segments[:-1]) + 1
        grouped = {}
        for indexes in np.split(order, boundaries):
            layer, head = self.coordinates[self.segments[indexes[0]]]
            grouped[(int(layer), int(head))] = np.sort(self.positions[indexes])
        return grouped


def transplant(segment_ids, local_positions):
    """Place each head's sorted local selections in its unchanged global slots."""
    order = np.argsort(segment_ids, kind="stable")
    result = np.empty(len(segment_ids), dtype=np.int64)
    if len(local_positions) != len(result):
        raise ValueError("Refinement did not fill every global slot")
    result[order] = local_positions
    return result


def refine_arrays(allocation, refinement, count):
    import torch

    if allocation.segments != refinement.segments or allocation.cumulative_ends != refinement.cumulative_ends:
        raise ValueError("Allocation and refinement layouts differ")
    coordinates = [(segment.layer, segment.head) for segment in allocation.segments]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Each physical head must have exactly one score segment")
    count = min(max(0, int(count)), allocation.scores.numel())
    if not count:
        return ArrayReferences([], [], [], coordinates), 0, 0
    ends = np.asarray(allocation.cumulative_ends, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    if ends[-1] != allocation.scores.numel() or any(
        end - start != segment.count for start, end, segment in zip(starts, ends, allocation.segments)
    ):
        raise ValueError("Invalid segment bounds")
    values, indices = torch.topk(allocation.scores, count, largest=False, sorted=True)
    indices_cpu = indices.detach().cpu().numpy()
    values_cpu = values.detach().cpu().numpy()
    segment_ids = np.searchsorted(ends, indices_cpu, side="right")
    quotas = np.bincount(segment_ids, minlength=len(ends))
    selected_parts, selected_segments = [], []
    for segment_index, (start, end, quota) in enumerate(zip(starts, ends, quotas)):
        if quota:
            selected_parts.append(torch.topk(refinement.scores[int(start):int(end)],
                                             int(quota), largest=False, sorted=True).indices)
            selected_segments.extend([segment_index] * int(quota))
    selected_local = torch.cat(selected_parts).detach().cpu().numpy()
    selected_segments = np.asarray(selected_segments, dtype=np.int64)
    position_starts = np.asarray([segment.position_start for segment in allocation.segments], dtype=np.int64)
    selected_positions = selected_local + position_starts[selected_segments]
    positions = transplant(segment_ids, selected_positions)
    refined_global = selected_local + starts[selected_segments]
    changed = ~np.isin(refined_global, indices_cpu, assume_unique=True)
    changed_heads = np.unique(selected_segments[changed]).size
    return ArrayReferences(values_cpu, segment_ids, positions, coordinates), int(changed.sum()), int(changed_heads)

