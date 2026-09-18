"""Temporal-v1 scoring with allocation-preserving Pool7 refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .cuda_ops import extension
from .selection import ArrayReferences, refine_arrays

from ._core.scoring import (
    AuditResult,
    KVReference,
    ScoreSegment,
    ScoreTable,
    select_bottom_references,
    synchronize,
)

from .cache import LazyRecallCache
from .history import QueryHistory


SCORE_NAME = "allocation_preserving_pool7_temporal_v1"


@dataclass(frozen=True)
class LazyRecallScoreDiagnostics:
    query_count: int
    candidate_count: int
    query_chunk_size: int


@dataclass(frozen=True)
class LazyRecallScoreResult:
    table: ScoreTable
    diagnostics: LazyRecallScoreDiagnostics


@dataclass(frozen=True)
class AllocationPreservingResult:
    references: ArrayReferences
    identity_change_count: int
    changed_head_count: int
    selection_ms: float

    @property
    def identity_change_fraction(self) -> float:
        return self.identity_change_count / max(1, len(self.references))


def _softplus64(value: torch.Tensor) -> torch.Tensor:
    return F.softplus(value.to(torch.float64))


def _risk_from_margin_drop(
    base_margin: torch.Tensor,
    margin_drop: torch.Tensor,
) -> torch.Tensor:
    margin = base_margin.to(device=margin_drop.device, dtype=torch.float64)
    compared = margin - margin_drop.to(torch.float64)
    probability = torch.sigmoid(margin)
    return (
        probability * (margin - compared)
        - _softplus64(margin)
        + _softplus64(compared)
    ).clamp_min(0.0)


def temporal_head_scores_reference(
    queries: torch.Tensor,
    query_positions: torch.Tensor,
    keys: torch.Tensor,
    key_positions: torch.Tensor,
    values: torch.Tensor,
    candidate_start: int,
    candidate_end: int,
    margin_gradient: torch.Tensor,
    base_margin: torch.Tensor,
    denominator_epsilon: float,
    query_chunk_size: int,
) -> torch.Tensor:
    """Return max historical v1 risk for one physical KV head."""
    if queries.ndim != 3:
        raise ValueError("queries must have shape [window, mapped_heads, dim]")
    if query_positions.numel() != queries.shape[0]:
        raise ValueError("query position count does not match query window")
    if keys.shape != values.shape:
        raise ValueError("key/value shapes differ")
    if key_positions.numel() != keys.shape[0]:
        raise ValueError("key position count does not match key rows")
    if not 0 <= candidate_start < candidate_end <= keys.shape[0]:
        raise ValueError("invalid candidate interval")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    gradient = margin_gradient.float()
    all_value_gradient = torch.matmul(
        gradient,
        values.float().transpose(0, 1),
    )
    value_gradient = all_value_gradient[:, candidate_start:candidate_end]
    candidate_count = candidate_end - candidate_start
    minimum_drop = torch.full(
        (candidate_count,),
        torch.inf,
        dtype=torch.float32,
        device=keys.device,
    )
    maximum_drop = torch.full_like(minimum_drop, -torch.inf)
    key_transpose = keys.transpose(0, 1)

    for start in range(0, queries.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, queries.shape[0])
        query_chunk = queries[start:end].to(keys.dtype)
        logits = torch.matmul(query_chunk, key_transpose) / math.sqrt(keys.shape[-1])
        causal = key_positions.reshape(1, 1, -1) <= query_positions[
            start:end
        ].reshape(-1, 1, 1)
        logits = logits.masked_fill(~causal, -torch.inf)
        probabilities = F.softmax(logits.float(), dim=-1)
        output_gradient = torch.matmul(
            probabilities.transpose(0, 1),
            all_value_gradient.unsqueeze(-1),
        ).squeeze(-1).transpose(0, 1)
        candidate_probabilities = probabilities[
            :, :, candidate_start:candidate_end
        ]
        ratio = candidate_probabilities / (
            1.0 - candidate_probabilities
        ).clamp_min(denominator_epsilon)
        margin_drop = (
            ratio
            * (
                value_gradient.reshape(1, *value_gradient.shape)
                - output_gradient.unsqueeze(-1)
            )
        ).sum(dim=1)
        minimum_drop = torch.minimum(minimum_drop, margin_drop.amin(dim=0))
        maximum_drop = torch.maximum(maximum_drop, margin_drop.amax(dim=0))

    lower_risk = _risk_from_margin_drop(base_margin, minimum_drop)
    upper_risk = _risk_from_margin_drop(base_margin, maximum_drop)
    return torch.maximum(lower_risk, upper_risk).to(torch.float32)


@torch.no_grad()
def temporal_head_scores(
    queries, query_positions, keys, key_positions, values,
    candidate_start, candidate_end, margin_gradient, base_margin,
    denominator_epsilon, query_chunk_size,
):
    if not keys.is_cuda:
        return temporal_head_scores_reference(
            queries, query_positions, keys, key_positions, values,
            candidate_start, candidate_end, margin_gradient, base_margin,
            denominator_epsilon, query_chunk_size,
        )
    if queries.ndim != 3 or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("Invalid temporal query dimensions")
    if keys.shape != values.shape or key_positions.numel() != keys.shape[0]:
        raise ValueError("KV and position shapes differ")
    if query_positions.numel() != queries.shape[0] or queries.shape[0] == 0:
        raise ValueError("Query positions do not match the nonempty history")
    if not 0 <= candidate_start < candidate_end <= keys.shape[0]:
        raise ValueError("Invalid candidate interval")
    if query_chunk_size <= 0 or denominator_epsilon <= 0:
        raise ValueError("Chunk size and denominator epsilon must be positive")
    kernels = extension()
    projected_values = torch.matmul(margin_gradient.float(), values.float().transpose(0, 1)).contiguous()
    key_positions = key_positions.to(dtype=torch.float32).contiguous()
    query_positions = query_positions.to(dtype=torch.int64).contiguous()
    margin = base_margin.to(device=keys.device, dtype=torch.float32).reshape(1).contiguous()
    bytes_per_query = queries.shape[1] * keys.shape[0] * (keys.element_size() + 4)
    chunk_size = min(query_chunk_size, max(1, (64 * 1024**2) // bytes_per_query))
    result = None
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        logits = torch.matmul(queries[start:end].to(keys.dtype), keys.transpose(0, 1))
        probabilities, output_projection = kernels.temporal_probabilities(
            logits.contiguous(), projected_values, query_positions[start:end],
            key_positions, 1.0 / math.sqrt(keys.shape[-1]),
        )
        scores = kernels.temporal_risk(
            probabilities, output_projection, projected_values, margin,
            candidate_start, candidate_end, denominator_epsilon,
        )
        result = scores if result is None else torch.maximum(result, scores)
        del logits, probabilities, output_projection
    return result


@torch.no_grad()
def compute_score_table(
    model,
    cache: LazyRecallCache,
    audit: AuditResult,
    history: QueryHistory,
    protect_prompt: bool,
    protect_recent: int,
    denominator_epsilon: float = 1e-6,
    query_chunk_size: int = 250,
) -> LazyRecallScoreResult:
    """Score all eligible physical KVs in one shared global risk space."""
    cache.assert_position_alignment()
    score_parts: List[torch.Tensor] = []
    segments: List[ScoreSegment] = []
    cumulative_ends: List[int] = []
    running_count = 0
    synchronize()
    started = time.perf_counter()

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        query_history, query_positions = history.layer(layer_index)
        num_query_heads = int(attention.config.num_attention_heads)
        num_kv_heads = int(attention.config.num_key_value_heads)
        groups = num_query_heads // num_kv_heads
        head_dim = int(attention.head_dim)
        if query_history.shape[1] != num_query_heads:
            raise RuntimeError("Historical query-head count changed")

        post_gradient = audit.attention_gradients[layer_index][0, -1].float()
        pre_gradient = torch.matmul(
            post_gradient,
            attention.o_proj.weight.detach().float(),
        ).view(num_query_heads, head_dim)
        if getattr(model, "attention_output_gate", False):
            gate = getattr(attention, "_last_output_gate", None)
            if gate is None or gate.shape != (1, 1, num_query_heads * head_dim):
                raise RuntimeError("Missing current Qwen3.5 attention output gate")
            pre_gradient = pre_gradient * gate[0, -1].float().view(num_query_heads, head_dim)
        keys_flat = cache.key_cache[layer_index].detach()
        values_flat = cache.value_cache[layer_index].detach()
        positions_flat = cache.layer_positions_tensor(layer_index, keys_flat.device)
        metadata = cache.metadata_list[layer_index]
        bounds = [0]
        for length in metadata.head_lens_cpu:
            bounds.append(bounds[-1] + int(length))
        prompt_lengths = metadata.prompt_lens_cpu

        for kv_head in range(num_kv_heads):
            head_start = bounds[kv_head]
            head_end = bounds[kv_head + 1]
            head_length = head_end - head_start
            candidate_start = 0
            if protect_prompt and prompt_lengths is not None:
                candidate_start = int(prompt_lengths[kv_head])
            candidate_end = max(candidate_start, head_length - protect_recent)
            if candidate_end <= candidate_start:
                continue

            query_start = kv_head * groups
            query_end = query_start + groups
            mapped_queries = query_history[0, query_start:query_end].permute(
                1, 0, 2
            )
            scores = temporal_head_scores(
                queries=mapped_queries,
                query_positions=query_positions,
                keys=keys_flat[head_start:head_end],
                key_positions=positions_flat[head_start:head_end],
                values=values_flat[head_start:head_end],
                candidate_start=candidate_start,
                candidate_end=candidate_end,
                margin_gradient=pre_gradient[query_start:query_end],
                base_margin=audit.margin,
                denominator_epsilon=denominator_epsilon,
                query_chunk_size=query_chunk_size,
            )
            count = candidate_end - candidate_start
            score_parts.append(scores)
            segments.append(
                ScoreSegment(
                    layer=layer_index,
                    head=kv_head,
                    position_start=candidate_start,
                    count=count,
                )
            )
            running_count += count
            cumulative_ends.append(running_count)

    if not score_parts:
        raise RuntimeError("No eligible KV remains after prompt/recent protection")
    scores = torch.cat(score_parts, dim=0)
    if not torch.isfinite(scores).all():
        raise RuntimeError("LazyRecall produced non-finite scores; refusing to construct a deletion queue")
    synchronize()
    table = ScoreTable(
        scores=scores,
        segments=segments,
        cumulative_ends=cumulative_ends,
        score_ms=(time.perf_counter() - started) * 1000.0,
    )
    return LazyRecallScoreResult(
        table=table,
        diagnostics=LazyRecallScoreDiagnostics(
            query_count=history.size,
            candidate_count=int(scores.numel()),
            query_chunk_size=int(query_chunk_size),
        ),
    )


def _segment_offsets(table: ScoreTable):
    start = 0
    for end, segment in zip(table.cumulative_ends, table.segments):
        if end - start != segment.count:
            raise RuntimeError("Score-table cumulative offsets are inconsistent")
        yield start, end, segment
        start = end
    if start != int(table.scores.numel()):
        raise RuntimeError("Score-table segments do not cover every score")


def pool_score_table(table: ScoreTable, kernel_size: int = 7) -> ScoreTable:
    """Protect local neighborhoods with per-physical-head max pooling."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("pool kernel must be a positive odd integer")
    synchronize()
    started = time.perf_counter()
    if table.scores.is_cuda:
        offsets = list(_segment_offsets(table))
        bounds = torch.tensor([0] + table.cumulative_ends, dtype=torch.int64, device=table.scores.device)
        scores = extension().pool(table.scores.contiguous(), bounds,
                                  max(segment.count for _, _, segment in offsets), kernel_size)
        synchronize()
        return ScoreTable(scores, list(table.segments), list(table.cumulative_ends),
                          table.score_ms + (time.perf_counter() - started) * 1000.0)
    parts = []
    for start, end, _ in _segment_offsets(table):
        parts.append(
            F.max_pool1d(
                table.scores[start:end].reshape(1, 1, -1),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ).reshape(-1)
        )
    synchronize()
    transform_ms = (time.perf_counter() - started) * 1000.0
    return ScoreTable(
        scores=torch.cat(parts, dim=0),
        segments=list(table.segments),
        cumulative_ends=list(table.cumulative_ends),
        score_ms=float(table.score_ms + transform_ms),
    )


def allocation_preserving_references(allocation_table, refinement_table, count):
    synchronize()
    started = time.perf_counter()
    references, changed, changed_heads = refine_arrays(allocation_table, refinement_table, count)
    return AllocationPreservingResult(references, changed, changed_heads,
                                      (time.perf_counter() - started) * 1000.0)
