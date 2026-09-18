"""Time the unchanged VJP, including its FP32 vocabulary-head preparation."""

import time
import torch

from ._core.scoring import DecisionAudit as BaseDecisionAudit, synchronize


class DecisionAudit(BaseDecisionAudit):
    @torch.enable_grad()
    def finish(self, outputs):
        result = super().finish(outputs)
        synchronize()
        result.complete_audit_ms = (time.perf_counter() - self._started) * 1000.0
        result.head_and_other_ms = max(0.0, result.complete_audit_ms - result.forward_ms - result.vjp_ms)
        return result

