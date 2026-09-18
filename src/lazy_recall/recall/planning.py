import numpy as np


def select_candidates(metadata, active_positions, prompt_length, logical_position,
                      max_per_head, admitted_versions=None, query_step=0):
    metadata = np.asarray(metadata, dtype=np.int64).reshape(-1, 3)
    selected, seen, counts = [], set(), {}
    admitted_versions = {} if admitted_versions is None else admitted_versions
    for index, (layer, head, position) in enumerate(metadata):
        identity = int(layer), int(head), int(position)
        segment = identity[:2]
        if identity in seen or not prompt_length <= int(position) < logical_position:
            continue
        if int(position) in active_positions.get(segment, ()):
            continue
        if admitted_versions.get(identity, -1) >= query_step:
            continue
        if counts.get(segment, 0) >= max_per_head:
            continue
        selected.append(index)
        seen.add(identity)
        counts[segment] = counts.get(segment, 0) + 1
    return np.asarray(selected, dtype=np.int64)


def admission_budget(active, wall, slots_per_step, queue_length, requested):
    if min(active, wall, slots_per_step, queue_length, requested) < 0 or slots_per_step == 0:
        raise ValueError("Invalid recall budget")
    room = max(0, wall - active - slots_per_step)
    admitted = min(requested, room + queue_length)
    return admitted, max(0, admitted - room)


def merge_plan(old_positions, head_lengths, recalled_metadata):
    old_positions = np.asarray(old_positions, dtype=np.int64).reshape(-1)
    recalled_metadata = np.asarray(recalled_metadata, dtype=np.int64).reshape(-1, 3)
    if sum(head_lengths) != len(old_positions):
        raise ValueError("Position lengths do not match metadata")
    mapping, lengths = [], []
    start = 0
    for head, length in enumerate(head_lengths):
        existing = old_positions[start:start + length]
        chosen = np.flatnonzero(recalled_metadata[:, 1] == head)
        added = recalled_metadata[chosen, 2]
        combined = np.concatenate((existing, added))
        if len(np.unique(combined)) != len(combined):
            raise ValueError("Cannot merge duplicate physical KV identities")
        sources = np.concatenate((np.arange(start, start + length), -chosen - 1))
        mapping.extend(sources[np.argsort(combined, kind="stable")])
        lengths.append(len(combined))
        start += length
    return np.asarray(mapping, dtype=np.int64), lengths

