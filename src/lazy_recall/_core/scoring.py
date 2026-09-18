"""Frozen LazyRecall decision-risk scoring and shared VJP audit."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import time
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..attention import flash_attn_deterministic_status
from .cache import FlatCache
from .cuda_ops import decision_risk_scores


SCORE_NAME = "global_decision_risk_v1"


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def one_vs_rest_margin(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    flat_logits = logits.reshape(-1)
    competitors = flat_logits.clone()
    competitors[target_id] = -torch.inf
    return flat_logits[target_id] - torch.logsumexp(competitors, dim=0)


def last_token_logits_float32(
    model,
    final_hidden_state: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise RuntimeError("Model has no linear output embedding layer")
    hidden = final_hidden_state.detach().float()
    weight = output_embeddings.weight
    bias = getattr(output_embeddings, "bias", None)
    chunks = []
    with torch.no_grad():
        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            chunk_bias = None if bias is None else bias[start:end].float()
            chunks.append(F.linear(hidden, weight[start:end].float(), chunk_bias))
    return torch.cat(chunks, dim=-1)


def one_vs_rest_hidden_gradient(
    model,
    logits: torch.Tensor,
    target_id: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    weight = model.get_output_embeddings().weight
    competitor_logits = logits.detach().float().clone()
    competitor_logits[:, target_id] = -torch.inf
    competitor_probabilities = F.softmax(competitor_logits, dim=-1)
    gradient = weight[target_id].float().unsqueeze(0).expand(
        logits.shape[0], -1
    ).clone()
    with torch.no_grad():
        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            gradient.sub_(
                torch.matmul(
                    competitor_probabilities[:, start:end],
                    weight[start:end].float(),
                )
            )
    return gradient


@dataclass
class AuditResult:
    margin: torch.Tensor
    target_id: int
    attention_gradients: List[torch.Tensor]
    attention_queries: List[torch.Tensor]
    forward_ms: float
    vjp_ms: float


class DecisionAudit:
    """Capture one real decode forward and execute one shared margin VJP."""

    def __init__(self, model, cache: FlatCache) -> None:
        self.model = model
        self.cache = cache
        self.layers = list(model.model.layers)
        self._captured_outputs: List[Optional[torch.Tensor]] = [
            None for _ in self.layers
        ]
        self._hooks = []
        self._started = 0.0
        self._finished = False

    def __enter__(self) -> "DecisionAudit":
        self.cache.set_audit_insert(True)
        for layer_index, layer in enumerate(self.layers):
            def capture(module, inputs, output, index=layer_index):
                del module, inputs
                self._captured_outputs[index] = (
                    output[0] if isinstance(output, tuple) else output
                )

            self._hooks.append(layer.self_attn.register_forward_hook(capture))
        synchronize()
        self._started = time.perf_counter()
        return self

    def finish(self, outputs) -> AuditResult:
        synchronize()
        forward_ms = (time.perf_counter() - self._started) * 1000.0
        if outputs.hidden_states is None:
            raise RuntimeError("Audit forward requires output_hidden_states=True")
        if any(output is None for output in self._captured_outputs):
            missing = [
                index
                for index, output in enumerate(self._captured_outputs)
                if output is None
            ]
            raise RuntimeError(f"Attention hooks did not fire for layers {missing}")
        if any(not output.requires_grad for output in self._captured_outputs):
            missing = [
                index
                for index, output in enumerate(self._captured_outputs)
                if not output.requires_grad
            ]
            raise RuntimeError(
                f"Attention outputs lack gradients in layers {missing}"
            )
        deterministic = flash_attn_deterministic_status()
        if deterministic is not True:
            raise RuntimeError(
                "FlashAttention deterministic varlen backward is required for "
                "LazyRecall scoring"
            )

        final_hidden = outputs.hidden_states[-1][:, -1]
        logits = last_token_logits_float32(self.model, final_hidden)
        target_id = int(logits.argmax(dim=-1).item())
        margin = one_vs_rest_margin(logits, target_id)
        hidden_gradient = one_vs_rest_hidden_gradient(
            self.model, logits, target_id
        )

        synchronize()
        started = time.perf_counter()
        gradients = torch.autograd.grad(
            final_hidden.float(),
            tuple(self._captured_outputs),
            grad_outputs=hidden_gradient,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        synchronize()
        vjp_ms = (time.perf_counter() - started) * 1000.0

        queries = []
        for layer_index, layer in enumerate(self.layers):
            query = getattr(layer.self_attn, "_last_query_states", None)
            if query is None:
                raise RuntimeError(
                    f"Layer {layer_index} did not expose _last_query_states"
                )
            queries.append(query.detach())
            layer.self_attn._last_query_states = query.detach()

        self.cache.detach_()
        self._finished = True
        return AuditResult(
            margin=margin.detach(),
            target_id=target_id,
            attention_gradients=[gradient.detach() for gradient in gradients],
            attention_queries=queries,
            forward_ms=forward_ms,
            vjp_ms=vjp_ms,
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._captured_outputs.clear()
        self.cache.set_audit_insert(False)
        if not self._finished:
            self.cache.detach_()


@dataclass(frozen=True)
class ScoreSegment:
    layer: int
    head: int
    position_start: int
    count: int


@dataclass
class ScoreTable:
    scores: torch.Tensor
    segments: List[ScoreSegment]
    cumulative_ends: List[int]
    score_ms: float


@dataclass(order=True)
class KVReference:
    score: float
    layer: int
    head: int
    position: int


def compute_score_table(
    model,
    cache: FlatCache,
    audit: AuditResult,
    protect_prompt: bool,
    protect_recent: int,
    denominator_epsilon: float = 1e-6,
    fallback_chunk_size: int = 4096,
) -> ScoreTable:
    """Score every eligible physical KV without per-head normalization."""
    score_parts = []
    segments: List[ScoreSegment] = []
    cumulative_ends = []
    running_count = 0
    base_margin_value = float(audit.margin.item())
    synchronize()
    started = time.perf_counter()

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        query_states = audit.attention_queries[layer_index]
        if query_states.shape[0] != 1 or query_states.shape[2] != 1:
            raise ValueError(
                "LazyRecall scoring supports batch=1 and q_len=1 only; "
                f"got {tuple(query_states.shape)}"
            )
        num_query_heads = int(attention.config.num_attention_heads)
        num_kv_heads = int(attention.config.num_key_value_heads)
        groups = num_query_heads // num_kv_heads
        head_dim = int(attention.head_dim)

        query = query_states[0, :, 0]
        post_projection_gradient = audit.attention_gradients[layer_index][
            0, -1
        ].float()
        pre_projection_gradient = torch.matmul(
            post_projection_gradient,
            attention.o_proj.weight.detach().float(),
        ).view(num_query_heads, head_dim)

        keys = cache.key_cache[layer_index].detach()
        values = cache.value_cache[layer_index].detach()
        metadata = cache.metadata_list[layer_index]
        cumulative_lengths = [0]
        for length in metadata.head_lens_cpu:
            cumulative_lengths.append(cumulative_lengths[-1] + length)
        prompt_lengths = (
            metadata.prompt_lens_cpu
            if metadata.prompt_lens_cpu is not None
            else None
        )
        for kv_head in range(num_kv_heads):
            head_start = int(cumulative_lengths[kv_head])
            head_end = int(cumulative_lengths[kv_head + 1])
            head_length = head_end - head_start
            candidate_start = 0
            if protect_prompt and prompt_lengths is not None:
                candidate_start = int(prompt_lengths[kv_head])
            candidate_end = max(candidate_start, head_length - protect_recent)
            if candidate_end <= candidate_start:
                continue

            head_keys = keys[head_start:head_end]
            head_values = values[head_start:head_end]
            query_start = kv_head * groups
            query_end = query_start + groups
            mapped_query = query[query_start:query_end].to(head_keys.dtype)
            raw_attention = torch.matmul(
                mapped_query, head_keys.transpose(0, 1)
            ) / math.sqrt(head_dim)
            probabilities = F.softmax(raw_attention.float(), dim=-1)
            head_outputs = torch.matmul(probabilities, head_values.float())
            candidate_probabilities = probabilities[
                :, candidate_start:candidate_end
            ]
            candidate_values = head_values[candidate_start:candidate_end]
            candidate_scores = decision_risk_scores(
                probabilities=candidate_probabilities,
                values=candidate_values,
                head_outputs=head_outputs,
                margin_gradients=pre_projection_gradient[
                    query_start:query_end
                ],
                base_margin=base_margin_value,
                denominator_epsilon=denominator_epsilon,
                chunk_size=fallback_chunk_size,
            )
            count = candidate_end - candidate_start
            score_parts.append(candidate_scores)
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
        raise RuntimeError(
            "No eligible KV remains after prompt/recent-token protection"
        )
    scores = torch.cat(score_parts, dim=0)
    synchronize()
    return ScoreTable(
        scores=scores,
        segments=segments,
        cumulative_ends=cumulative_ends,
        score_ms=(time.perf_counter() - started) * 1000.0,
    )


def select_bottom_references(
    table: ScoreTable,
    count: int,
) -> List[KVReference]:
    if count <= 0:
        return []
    count = min(count, int(table.scores.numel()))
    values, indices = torch.topk(
        table.scores,
        k=count,
        largest=False,
        sorted=True,
    )
    indices_cpu = indices.detach().cpu().tolist()
    values_cpu = values.detach().cpu().tolist()
    references = []
    for score, global_index in zip(values_cpu, indices_cpu):
        segment_index = bisect_right(table.cumulative_ends, global_index)
        segment = table.segments[segment_index]
        previous_end = (
            0 if segment_index == 0 else table.cumulative_ends[segment_index - 1]
        )
        references.append(
            KVReference(
                score=float(score),
                layer=segment.layer,
                head=segment.head,
                position=segment.position_start + global_index - previous_end,
            )
        )
    return references

