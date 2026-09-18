"""Recent RoPE-applied query states for LazyRecall temporal scoring."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Sequence, Tuple

import torch


class QueryHistory:
    def __init__(self, num_layers: int, window_size: int) -> None:
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("temporal window must be positive")
        self._queries: List[Deque[torch.Tensor]] = [
            deque(maxlen=self.window_size) for _ in range(int(num_layers))
        ]
        self._positions: Deque[int] = deque(maxlen=self.window_size)

    def append(
        self,
        queries: Sequence[torch.Tensor],
        absolute_position: int,
    ) -> None:
        if len(queries) != len(self._queries):
            raise RuntimeError("Query-history layer count mismatch")
        position = int(absolute_position)
        if self._positions and position <= self._positions[-1]:
            raise RuntimeError("Query positions must be strictly increasing")
        self._positions.append(position)
        for layer_index, query in enumerate(queries):
            if query.shape[0] != 1 or query.shape[2] != 1:
                raise RuntimeError(
                    f"Expected one decode query at layer {layer_index}, got "
                    f"{tuple(query.shape)}"
                )
            self._queries[layer_index].append(query.detach())

    def capture_model(self, model, absolute_position: int) -> None:
        queries = []
        for layer_index, layer in enumerate(model.model.layers):
            query = getattr(layer.self_attn, "_last_query_states", None)
            if query is None:
                raise RuntimeError(
                    f"Layer {layer_index} did not expose _last_query_states"
                )
            queries.append(query.detach())
        self.append(queries, absolute_position)

    def layer(self, layer_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history = self._queries[layer_index]
        if not history:
            raise RuntimeError(f"No query history for layer {layer_index}")
        queries = torch.cat(list(history), dim=2)
        positions = torch.tensor(
            list(self._positions),
            dtype=torch.long,
            device=queries.device,
        )
        if queries.shape[2] != positions.numel():
            raise RuntimeError("Query and position history lengths differ")
        return queries, positions

    @property
    def size(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> Tuple[int, ...]:
        return tuple(self._positions)

    @property
    def retained_bytes(self) -> int:
        total = 0
        for history in self._queries:
            total += sum(query.numel() * query.element_size() for query in history)
        return total

