"""Optional CUDA kernels with numerically matched PyTorch fallbacks."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F


try:
    from lazy_recall import _legacy_C as _extension
except ImportError:
    _extension = None


def extension_available() -> bool:
    return _extension is not None


def append_flattened_cache(
    cache: torch.Tensor,
    states: torch.Tensor,
    head_lens: torch.Tensor,
    cu_head_lens: torch.Tensor,
) -> torch.Tensor:
    """Insert each head's new states at the end of its flattened segment."""
    if _extension is not None and cache.is_cuda:
        return _extension.append_flattened_cache(
            cache.contiguous(),
            states.contiguous(),
            head_lens.to(dtype=torch.int32).contiguous(),
            cu_head_lens.to(dtype=torch.int32).contiguous(),
        )
    return append_flattened_cache_pytorch(cache, states, head_lens)


def append_flattened_cache_pytorch(
    cache: torch.Tensor,
    states: torch.Tensor,
    head_lens: Union[torch.Tensor, Sequence[int]],
) -> torch.Tensor:
    """Differentiable insertion path used by VJP audit steps and CPU fallback."""
    batch_size, num_heads, new_length, head_dim = states.shape
    parts = []
    source_offset = 0
    for flat_head in range(batch_size * num_heads):
        old_length_value = head_lens[flat_head]
        old_length = (
            int(old_length_value.item())
            if isinstance(old_length_value, torch.Tensor)
            else int(old_length_value)
        )
        if old_length:
            parts.append(cache[source_offset : source_offset + old_length])
        batch_index = flat_head // num_heads
        head_index = flat_head % num_heads
        parts.append(states[batch_index, head_index])
        source_offset += old_length
    if not parts:
        return cache.new_empty((0, head_dim))
    return torch.cat(parts, dim=0).reshape(-1, head_dim)


def compact_flattened_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    keep_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused K/V gather for the eight-step physical eviction."""
    indices = keep_indices.to(device=keys.device, dtype=torch.long).contiguous()
    if _extension is not None and keys.is_cuda:
        return _extension.compact_flattened_kv(
            keys.contiguous(), values.contiguous(), indices
        )
    return keys.index_select(0, indices), values.index_select(0, indices)


def _softplus64(value: torch.Tensor) -> torch.Tensor:
    return F.softplus(value.to(torch.float64))


def decision_risk_scores(
    probabilities: torch.Tensor,
    values: torch.Tensor,
    head_outputs: torch.Tensor,
    margin_gradients: torch.Tensor,
    base_margin: Union[torch.Tensor, float],
    denominator_epsilon: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute frozen local-LOO plus Bernoulli decision-KL scores.

    ``probabilities`` has shape ``[mapped_query_heads, candidates]``.  Values
    are the candidate V rows, while ``head_outputs`` and ``margin_gradients``
    have shape ``[mapped_query_heads, head_dim]``.
    """
    if probabilities.numel() == 0:
        return probabilities.new_empty((probabilities.shape[-1],))
    margin_value = (
        float(base_margin.item())
        if isinstance(base_margin, torch.Tensor)
        else float(base_margin)
    )
    if _extension is not None and probabilities.is_cuda:
        return _extension.decision_risk_scores(
            probabilities.float().contiguous(),
            values.contiguous(),
            head_outputs.float().contiguous(),
            margin_gradients.float().contiguous(),
            margin_value,
            float(denominator_epsilon),
        )
    return decision_risk_scores_pytorch(
        probabilities=probabilities,
        values=values,
        head_outputs=head_outputs,
        margin_gradients=margin_gradients,
        base_margin=margin_value,
        denominator_epsilon=denominator_epsilon,
        chunk_size=chunk_size,
    )


def decision_risk_scores_pytorch(
    probabilities: torch.Tensor,
    values: torch.Tensor,
    head_outputs: torch.Tensor,
    margin_gradients: torch.Tensor,
    base_margin: Union[torch.Tensor, float],
    denominator_epsilon: float,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Reference implementation used to validate the fused CUDA scorer."""
    margin_value = (
        float(base_margin.item())
        if isinstance(base_margin, torch.Tensor)
        else float(base_margin)
    )

    reference_margin = torch.tensor(
        margin_value,
        dtype=torch.float64,
        device=probabilities.device,
    )
    reference_probability = torch.sigmoid(reference_margin)
    output = []
    for start in range(0, probabilities.shape[1], chunk_size):
        end = min(start + chunk_size, probabilities.shape[1])
        alpha = probabilities[:, start:end].float()
        denominator = (1.0 - alpha).clamp_min(denominator_epsilon)
        local_delta = (
            alpha[:, :, None]
            / denominator[:, :, None]
            * (
                values[None, start:end].float()
                - head_outputs[:, None].float()
            )
        )
        margin_drop = (
            local_delta * margin_gradients[:, None].float()
        ).sum(dim=(0, 2))
        compared_margin = reference_margin - margin_drop.to(torch.float64)
        kl = (
            reference_probability * (reference_margin - compared_margin)
            - _softplus64(reference_margin)
            + _softplus64(compared_margin)
        ).clamp_min(0.0)
        output.append(kl.to(torch.float32))
    return torch.cat(output, dim=0)

