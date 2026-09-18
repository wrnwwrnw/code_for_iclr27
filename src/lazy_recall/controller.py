from collections import deque
from dataclasses import asdict
import time

import numpy as np

from .recall.planning import admission_budget, select_candidates
from .scoring import SCORE_NAME, compute_score_table, pool_score_table, allocation_preserving_references
from .selection import ArrayReferences


class LazyController:
    def __init__(self, model_config, config, history):
        self.config, self.history = config, history
        self.slots_per_step = model_config.num_hidden_layers * model_config.num_key_value_heads
        self.wall = config.budget * self.slots_per_step
        self.staging = config.evict_interval * self.slots_per_step
        self.retained = self.wall - self.staging
        self.active = self.generated_tokens = self.peak = 0
        self.steps_since_audit = self.steps_since_evict = 0
        self.queue = ArrayReferences([], [], [], [])
        self.removed_at_audit = {}
        self.audits = self.eviction_batches = self.evicted = self.early_audits = 0
        self.recalled = self.recall_swaps = self.recall_events = 0
        self.admitted_versions = {}
        self.audit_history = deque(maxlen=128)
        self.total_audit_ms = 0.0
        self.last_admitted = 0

    def on_prefill(self, cache):
        cache.freeze_prompt()
        self.active = cache.generation_slots()
        self.check_budget(cache)

    def check_budget(self, cache):
        actual = cache.generation_slots()
        if actual != self.active or not 0 <= actual <= self.wall:
            raise AssertionError(f"Physical generation budget mismatch: tracked={self.active}, actual={actual}, cap={self.wall}")
        self.peak = max(self.peak, actual)

    def needs_audit(self):
        if not self.audits:
            return self.active + self.slots_per_step >= self.wall
        if self.steps_since_audit + 1 >= self.config.audit_period:
            return True
        if self.steps_since_evict + 1 >= self.config.evict_interval and len(self.queue) < self.staging:
            self.early_audits += 1
            return True
        return False

    def remove(self, cache, removals):
        count = sum(len(values) for values in removals.values())
        cache.remove_positions(removals)
        self.active -= count
        self.evicted += count

    def evict_prefix(self, cache, count):
        if not count:
            return
        if count > len(self.queue):
            raise RuntimeError("Discard queue underflow")
        removals, updates = {}, {}
        for segment, positions in self.queue[:count].groups().items():
            removed = np.asarray(self.removed_at_audit.get(segment, ()), dtype=np.int64)
            removals[segment] = (positions - np.searchsorted(removed, positions)).tolist()
            updates[segment] = np.union1d(removed, positions)
        self.remove(cache, removals)
        self.removed_at_audit.update(updates)
        self.queue = self.queue[count:]
        self.eviction_batches += 1

    def prepare_audit(self, cache):
        transport = cache.transport
        self.last_admitted = 0
        if transport is None:
            return
        if self.config.recall.wait_at_audit:
            transport.wait_pending()
        ready = transport.ready_result()
        if ready is None:
            return
        slot, query_step, payload, device_metadata, metadata = ready
        del device_metadata
        try:
            active = cache.active_position_sets(metadata)
            prompt_length = cache.metadata_list[0].prompt_lens_cpu[0]
            selected = select_candidates(
                metadata, active, prompt_length, cache._seen_tokens,
                self.config.recall.max_admit_per_head, self.admitted_versions, query_step,
            )
            count, swap = admission_budget(self.active, self.wall, self.slots_per_step,
                                            len(self.queue), len(selected))
            selected = selected[:count]
            if not count:
                return
            import torch

            chosen_metadata = metadata[selected].copy()
            chosen_payload = payload.index_select(
                0, torch.tensor(selected, device=payload.device, dtype=torch.long)).contiguous()
            transport.remove_admitted(chosen_metadata)
            self.evict_prefix(cache, swap)
            inserted = cache.insert_payload(chosen_payload, chosen_metadata)
            self.active += inserted
            self.recalled += inserted
            self.recall_swaps += swap
            self.recall_events += 1
            self.last_admitted = inserted
            self.admitted_versions.update((tuple(identity), query_step) for identity in chosen_metadata.tolist())
            transport.counters["spliced_vectors"] += inserted
            transport.counters["splices"] += 1
            transport.counters["query_age_tokens_max"] = max(
                transport.counters["query_age_tokens_max"], self.generated_tokens - query_step)
            self.queue = ArrayReferences([], [], [], [])
            self.removed_at_audit = {}
            self.check_budget(cache)
            if self.active + self.slots_per_step > self.wall:
                raise AssertionError("Recall did not leave room for the audit token")
        finally:
            transport.release_result(slot)

    def on_decode(self, model, cache, audit=None):
        self.generated_tokens += 1
        self.active += self.slots_per_step
        self.check_budget(cache)
        if audit is not None:
            self.run_audit(model, cache, audit)
        elif self.audits:
            self.steps_since_audit += 1
            self.steps_since_evict += 1
            if self.steps_since_evict >= self.config.evict_interval:
                self.evict_prefix(cache, self.staging)
                self.steps_since_evict = 0
        elif self.active >= self.wall:
            raise AssertionError("Reached the budget without an audit")
        self.check_budget(cache)
        if cache.transport is not None:
            cache.transport.step(model, cache, self.generated_tokens)

    def run_audit(self, model, cache, audit):
        started = time.perf_counter()
        temporal = compute_score_table(
            model, cache, audit, self.history, True, self.config.protect_recent,
            self.config.denominator_epsilon, self.config.query_chunk_size,
        )
        table = temporal.table
        pooled = pool_score_table(table, self.config.pool_kernel)
        immediate = max(0, self.active - self.retained)
        selection = allocation_preserving_references(
            table, pooled, immediate + self.config.audit_period * self.slots_per_step)
        references = selection.references
        if len(references) < immediate:
            raise RuntimeError("Not enough unprotected candidates to enforce the budget")
        removals = {key: values.tolist() for key, values in references[:immediate].groups().items()}
        if removals:
            self.remove(cache, removals)
        self.removed_at_audit = removals
        self.queue = references[immediate:]
        self.steps_since_audit = self.steps_since_evict = 0
        self.audits += 1
        total_ms = ((time.perf_counter() - started) * 1000 + audit.forward_ms
                    + audit.vjp_ms + getattr(audit, "head_and_other_ms", 0))
        self.total_audit_ms += total_ms
        self.audit_history.append(dict(
            audit=self.audits, step=self.generated_tokens, margin=float(audit.margin.item()),
            eligible=table.scores.numel(), immediate=immediate, queued=len(self.queue),
            recalled=self.last_admitted, forward_ms=audit.forward_ms, vjp_ms=audit.vjp_ms,
            temporal_ms=table.score_ms, selection_ms=selection.selection_ms, total_ms=total_ms,
        ))
        if self.config.log_audits:
            print(f"[LazyRecall] audit={self.audits} step={self.generated_tokens} "
                  f"recalled={self.last_admitted} queued={len(self.queue)} "
                  f"active={self.active}/{self.wall} total={total_ms:.1f}ms", flush=True)

    def stats(self, cache):
        recall = dict(enabled=cache.transport is not None, admitted=self.recalled,
                      swapped_out=self.recall_swaps, admission_events=self.recall_events,
                      recalled_visible_forwards=cache.recalled_visible_forwards,
                      recalled_resident=len(cache.recalled_identities),
                      recalled_resident_peak=cache.recalled_resident_peak)
        if cache.transport is not None:
            recall.update(cache.transport.stats)
        return dict(score=SCORE_NAME, budget=self.config.budget, audit_period=self.config.audit_period,
                    temporal_window=self.config.temporal_window, pool_kernel=self.config.pool_kernel,
                    evict_interval=self.config.evict_interval, prompt_protected=True,
                    prompt_charged_to_budget=False, active_generation_slots=self.active,
                    peak_generation_slots=self.peak, generation_slot_cap=self.wall,
                    persistent_budget_respected=self.peak <= self.wall,
                    physical_evictions=self.evicted, eviction_batches=self.eviction_batches,
                    audits=self.audits, early_audits=self.early_audits, queue_slots=len(self.queue),
                    mean_audit_ms=self.total_audit_ms / max(1, self.audits),
                    recent_audit_history=list(self.audit_history), cpu_recall=recall)

