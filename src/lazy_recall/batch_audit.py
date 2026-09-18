"""A single VJP with one unchanged margin objective per audited request."""

import time

import torch

from .attention import flash_attn_deterministic_status
from ._core.scoring import AuditResult, last_token_logits_float32, synchronize


def row_margins(logits):
    targets = logits.argmax(dim=-1)
    competitors = logits.scatter(1, targets[:, None], -torch.inf)
    margins = logits.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(competitors, dim=-1)
    return margins, targets, competitors


def row_hidden_gradients(model, logits, targets, competitors, due, chunk_size=4096):
    weight = model.get_output_embeddings().weight
    probabilities = competitors.softmax(-1)
    gradient = weight.index_select(0, targets).float()
    for start in range(0, weight.shape[0], chunk_size):
        gradient.sub_(probabilities[:, start:start + chunk_size] @ weight[start:start + chunk_size].float())
    return gradient * torch.tensor(due, device=logits.device, dtype=gradient.dtype)[:, None]


class BatchAudit:
    def __init__(self, model, cache, due):
        self.model, self.cache, self.due = model, cache, list(due)
        self.captured, self.hooks = [None] * len(model.model.layers), []

    def __enter__(self):
        self.cache.set_audit_insert(True)
        for request in self.cache.requests:
            request.cache.set_audit_insert(True)
        for index, layer in enumerate(self.model.model.layers):
            def capture(module, inputs, output, slot=index):
                self.captured[slot] = output[0] if isinstance(output, tuple) else output
            self.hooks.append(layer.self_attn.register_forward_hook(capture))
        synchronize()
        self.started = time.perf_counter()
        return self

    @torch.enable_grad()
    def finish(self, outputs):
        synchronize()
        forward_ms = (time.perf_counter() - self.started) * 1000
        if flash_attn_deterministic_status() is not True:
            raise RuntimeError("Deterministic FlashAttention backward is required")
        if outputs.hidden_states is None or any(output is None or not output.requires_grad for output in self.captured):
            raise RuntimeError("Missing differentiable attention outputs")
        final_hidden = outputs.hidden_states[-1][:, -1]
        logits = last_token_logits_float32(self.model, final_hidden)
        margins, targets, competitors = row_margins(logits)
        with torch.no_grad():
            hidden_gradient = row_hidden_gradients(self.model, logits, targets, competitors, self.due)
        synchronize()
        started = time.perf_counter()
        gradients = torch.autograd.grad(final_hidden.float(), tuple(self.captured), grad_outputs=hidden_gradient,
                                        retain_graph=False, create_graph=False, allow_unused=False)
        synchronize()
        vjp_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - self.started) * 1000
        queries = [layer.self_attn._last_query_states.detach() for layer in self.model.model.layers]
        audits = []
        for row, enabled in enumerate(self.due):
            if not enabled:
                audits.append(None)
                continue
            result = AuditResult(margins[row].detach(), int(targets[row].item()),
                                 [gradient[row:row + 1].detach() for gradient in gradients],
                                 [query[row:row + 1] for query in queries], forward_ms, vjp_ms)
            result.complete_audit_ms = total_ms
            result.head_and_other_ms = max(0.0, total_ms - forward_ms - vjp_ms)
            audits.append(result)
        return audits

    def __exit__(self, kind, value, traceback):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.captured.clear()
        self.cache.set_audit_insert(False)
        for request in self.cache.requests:
            request.cache.detach_()
            request.cache.set_audit_insert(False)
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "_last_query_states"):
                layer.self_attn._last_query_states = layer.self_attn._last_query_states.detach()
