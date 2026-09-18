"""Registered shared-memory DMA rings and a single CPU-owned FAISS index."""

import multiprocessing as mp
from multiprocessing import shared_memory
import os
import queue
import time
from collections import deque

import numpy as np
import torch

from .cpu_worker import run_worker


class SharedArray:
    def __init__(self, shape, dtype):
        from lazy_recall import _host
        self.bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.handle = shared_memory.SharedMemory(create=True, size=self.bytes)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self.handle.buf)
        self.tensor = torch.from_numpy(self.array)
        self.registered = False
        try:
            _host.register_host(self.tensor.data_ptr(), self.bytes)
            self.registered = True
        except BaseException:
            self.tensor = self.array = None
            self.handle.close()
            self.handle.unlink()
            raise

    @property
    def descriptor(self):
        return dict(name=self.handle.name, shape=list(self.array.shape), dtype=self.array.dtype.str)

    def close(self):
        from lazy_recall import _host
        if self.registered:
            _host.unregister_host(self.tensor.data_ptr())
            self.registered = False
        self.tensor = self.array = None
        self.handle.close()
        self.handle.unlink()


class RecallTransport:
    def __init__(self, layers, heads, query_heads, dimension, search_interval=32,
                 topk=4, archive_buffers=16, cpu_threads=2, evict_interval=8,
                 archive_capacity_per_head=32768, timeout_seconds=60.0):
        if query_heads % heads or min(layers, heads, dimension, search_interval, topk, archive_buffers, cpu_threads) <= 0:
            raise ValueError("Invalid recall configuration")
        self.layers, self.heads, self.groups, self.dimension = layers, heads, query_heads // heads, dimension
        self.search_interval = search_interval
        self.capacity = max(evict_interval, topk) * layers * heads
        self.result_capacity = topk * layers * heads
        self.timeout = timeout_seconds
        self.slots, self.process, self.commands, self.replies = {}, None, None, None
        self.closed = False
        self.copy_stream = torch.cuda.Stream()
        self.archive_states = ["free"] * archive_buffers
        self.query_states = ["free"] * 2
        self.archive_jobs, self.query_jobs, self.events = {}, {}, deque(maxlen=32)
        self.counters = dict(archived_vectors=0, archive_batches=0, indexed_vectors=0,
                             submitted_queries=0, searches_completed=0, skipped_busy_queries=0,
                             retrieved_vectors=0, spliced_vectors=0, splices=0,
                             d2h_bytes=0, h2d_bytes=0, archive_overflows=0,
                             cpu_peak_rss_kib=0, query_age_tokens_max=0,
                             archive_backpressure=0, capacity_drops=0, superseded_results=0)
        specs = dict(archive_kv=((archive_buffers, self.capacity, 2, dimension), np.uint16),
                     archive_meta=((archive_buffers, self.capacity, 3), np.int64),
                     queries=((2, layers, heads, dimension), np.float32),
                     result_kv=((2, self.result_capacity, 2, dimension), np.uint16),
                     result_meta=((2, self.result_capacity, 3), np.int64))
        self.config = dict(layers=layers, heads=heads, dimension=dimension, topk=topk,
                           cpu_threads=cpu_threads, archive_capacity_per_head=archive_capacity_per_head)
        try:
            for name, (shape, dtype) in specs.items():
                self.slots[name] = SharedArray(shape, dtype)
            self.archive_gpu = torch.empty(specs["archive_kv"][0], device="cuda", dtype=torch.bfloat16)
            self.archive_meta_gpu = torch.empty(specs["archive_meta"][0], device="cuda", dtype=torch.int64)
            self.query_gpu = torch.empty(specs["queries"][0], device="cuda", dtype=torch.float32)
            self.result_gpu = torch.empty(specs["result_kv"][0], device="cuda", dtype=torch.bfloat16)
            self.result_meta_gpu = torch.empty(specs["result_meta"][0], device="cuda", dtype=torch.int64)
            context = mp.get_context("spawn")
            self.commands = context.Queue(maxsize=archive_buffers + 6)
            self.replies = context.Queue()
            descriptors = {name: slot.descriptor for name, slot in self.slots.items()}
            self.process = context.Process(target=run_worker,
                                           args=(self.config, descriptors, self.commands, self.replies),
                                           daemon=True, name="LazyRecall-CPU-FAISS")
            old_environment = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
            try:
                for name in old_environment:
                    os.environ[name] = str(cpu_threads)
                self.process.start()
            finally:
                for name, value in old_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            message = self.replies.get(timeout=self.timeout)
            if message["kind"] != "ready":
                raise RuntimeError(f"CPU worker failed: {message}")
            self.worker_info = message
        except BaseException:
            self.close(abort=True)
            raise

    def _send(self, message):
        try:
            self.commands.put(message, timeout=self.timeout)
        except queue.Full as error:
            raise RuntimeError("Recall command queue backpressure exceeded timeout") from error

    def _copy_out(self, gpu_pairs):
        packed = torch.cuda.Event()
        packed.record(torch.cuda.current_stream())
        done = torch.cuda.Event()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(packed)
            for destination, source in gpu_pairs:
                destination.copy_(source, non_blocking=True)
                self.counters["d2h_bytes"] += source.numel() * source.element_size()
            done.record(self.copy_stream)
        return done

    @torch.no_grad()
    def record_evicted(self, cache, removals):
        self.pump()
        count = sum(len(set(positions)) for positions in removals.values())
        if count == 0:
            return
        if count > self.capacity:
            self.counters["archive_overflows"] += 1
            raise RuntimeError("Eviction batch exceeds configured archive staging capacity")
        started = time.perf_counter()
        if "free" not in self.archive_states:
            self.counters["archive_backpressure"] += 1
        while "free" not in self.archive_states:
            if time.perf_counter() - started > self.timeout:
                raise TimeoutError("CPU archive backpressure exceeded timeout")
            self.pump()
            time.sleep(0.001)
        slot = self.archive_states.index("free")
        cursor = 0
        for layer in sorted({layer for layer, _ in removals}):
            bounds = np.concatenate(([0], np.cumsum(cache.metadata_list[layer].head_lens_cpu)))
            indices, head_ids = [], []
            for (record_layer, head), positions in sorted(removals.items()):
                if record_layer == layer:
                    selected = sorted(set(int(position) for position in positions))
                    indices.extend(int(bounds[head]) + position for position in selected)
                    head_ids.extend([head] * len(selected))
            if not indices:
                continue
            absolute = torch.tensor(indices, device="cuda", dtype=torch.int64)
            end = cursor + len(indices)
            from ..cuda_ops import extension
            extension().pack(
                cache.key_cache[layer].contiguous(), cache.value_cache[layer].contiguous(),
                cache.position_cache[layer].contiguous(), absolute,
                torch.tensor(head_ids, device="cuda", dtype=torch.int64), layer,
                self.archive_gpu[slot, cursor:end], self.archive_meta_gpu[slot, cursor:end],
            )
            cursor = end
        if cursor != count:
            raise AssertionError("Archive packing count mismatch")
        done = self._copy_out([
            (self.slots["archive_kv"].tensor[slot, :count].view(torch.bfloat16), self.archive_gpu[slot, :count]),
            (self.slots["archive_meta"].tensor[slot, :count], self.archive_meta_gpu[slot, :count]),
        ])
        self.archive_states[slot] = "copying"
        self.archive_jobs[slot] = dict(event=done, count=count)
        self.counters["archived_vectors"] += count
        self.counters["archive_batches"] += 1

    @torch.no_grad()
    def pump(self):
        if self.closed:
            return
        if not self.process.is_alive():
            raise RuntimeError(f"CPU recall worker exited unexpectedly: {self.process.exitcode}")
        for slot, job in list(self.archive_jobs.items()):
            if self.archive_states[slot] == "copying" and job["event"].query():
                self._send(dict(kind="archive", slot=slot, count=job["count"]))
                self.archive_states[slot] = "worker"
        for slot, job in list(self.query_jobs.items()):
            if self.query_states[slot] == "copying_query" and job["event"].query():
                self._send(dict(kind="search", slot=slot, step=job["step"]))
                self.query_states[slot] = "worker"
        while True:
            try:
                message = self.replies.get_nowait()
            except queue.Empty:
                break
            kind, slot = message["kind"], message.get("slot")
            if kind == "error":
                raise RuntimeError(message["traceback"])
            if kind not in ("archived", "searched", "flushed"):
                raise RuntimeError(f"Unexpected CPU reply: {message}")
            self.events.append(dict(kind=kind, start_ns=message["start_ns"], end_ns=message["end_ns"]))
            self.counters["indexed_vectors"] = message["indexed"]
            self.counters["capacity_drops"] = message.get("capacity_drops", 0)
            self.counters["cpu_peak_rss_kib"] = max(self.counters["cpu_peak_rss_kib"], message["cpu_peak_rss_kib"])
            if kind == "flushed":
                self._flushed = True
            elif kind == "archived":
                del self.archive_jobs[slot]
                self.archive_states[slot] = "free"
            else:
                self.counters["searches_completed"] += 1
                self.counters["retrieved_vectors"] += message["count"]
                job = self.query_jobs[slot]
                job.update(count=message["count"], indexed=message["indexed"])
                if not message["count"]:
                    del self.query_jobs[slot]
                    self.query_states[slot] = "free"
                    continue
                count = job["count"]
                done = torch.cuda.Event()
                with torch.cuda.stream(self.copy_stream):
                    self.result_gpu[slot, :count].copy_(self.slots["result_kv"].tensor[slot, :count].view(torch.bfloat16), non_blocking=True)
                    self.result_meta_gpu[slot, :count].copy_(self.slots["result_meta"].tensor[slot, :count], non_blocking=True)
                    done.record(self.copy_stream)
                self.counters["h2d_bytes"] += count * (2 * self.dimension * 2 + 3 * 8)
                job["event"] = done
                self.query_states[slot] = "copying_result"

    def release_result(self, slot):
        complete = torch.cuda.Event()
        complete.record(torch.cuda.current_stream())
        self.query_jobs[slot]["event"] = complete
        self.query_states[slot] = "retiring"

    def retire_finished(self):
        for slot, job in list(self.query_jobs.items()):
            if self.query_states[slot] == "retiring" and job["event"].query():
                del self.query_jobs[slot]
                self.query_states[slot] = "free"

    def ready_result(self):
        self.pump()
        self.retire_finished()
        candidates = [(job["step"], slot) for slot, job in self.query_jobs.items()
                      if self.query_states[slot] == "copying_result" and job["event"].query()]
        if not candidates:
            return None
        _, chosen = max(candidates)
        for _, slot in candidates:
            if slot != chosen:
                self.release_result(slot)
                self.counters["superseded_results"] += 1
        job = self.query_jobs[chosen]
        return (chosen, job["step"], self.result_gpu[chosen, :job["count"]],
                self.result_meta_gpu[chosen, :job["count"]],
                self.slots["result_meta"].array[chosen, :job["count"]].copy())

    def remove_admitted(self, metadata):
        self._send(dict(kind="remove", metadata=np.asarray(metadata, dtype=np.int64).tolist()))

    def wait_pending(self):
        started = time.perf_counter()
        while self.archive_jobs or any(
            self.query_states[slot] in ("copying_query", "worker")
            or (self.query_states[slot] == "copying_result" and not job["event"].query())
            for slot, job in self.query_jobs.items()
        ):
            if time.perf_counter() - started > self.timeout:
                raise TimeoutError("Recall diagnostic wait exceeded timeout")
            self.pump()
            self.retire_finished()
            time.sleep(0.001)

    @torch.no_grad()
    def step(self, model, cache, step):
        self.pump()
        self.retire_finished()
        if step % self.search_interval or not self.counters["archived_vectors"]:
            return
        if "free" not in self.query_states:
            ready = [(job["step"], slot) for slot, job in self.query_jobs.items()
                     if self.query_states[slot] == "copying_result" and job["event"].query()]
            if ready:
                _, oldest = min(ready)
                del self.query_jobs[oldest]
                self.query_states[oldest] = "free"
                self.counters["superseded_results"] += 1
        if "free" not in self.query_states:
            self.counters["skipped_busy_queries"] += 1
            return
        slot = self.query_states.index("free")
        for layer, block in enumerate(model.model.layers):
            query = block.self_attn._last_query_states.detach()[0, :, 0]
            self.query_gpu[slot, layer].copy_(
                query.view(self.heads, self.groups, self.dimension).float().mean(dim=1))
        done = self._copy_out([(self.slots["queries"].tensor[slot], self.query_gpu[slot])])
        self.query_jobs[slot] = dict(event=done, step=step)
        self.query_states[slot] = "copying_query"
        self.counters["submitted_queries"] += 1

    def finish(self):
        self.wait_pending()
        for slot in list(self.query_jobs):
            if self.query_states[slot] == "copying_result":
                self.release_result(slot)
        torch.cuda.current_stream().synchronize()
        self.retire_finished()
        self._flushed = False
        self._send(dict(kind="flush"))
        started = time.perf_counter()
        while not self._flushed:
            if time.perf_counter() - started > self.timeout:
                raise TimeoutError("CPU archive flush exceeded timeout")
            self.pump()
            time.sleep(0.001)

    @property
    def stats(self):
        if hasattr(self, "_closed_stats"):
            return dict(self._closed_stats)
        return dict(self.counters, worker=self.worker_info,
                    registered_shared_bytes=sum(slot.bytes for slot in self.slots.values()),
                    gpu_staging_bytes=sum(tensor.numel() * tensor.element_size() for tensor in
                                          (self.archive_gpu, self.archive_meta_gpu, self.query_gpu, self.result_gpu, self.result_meta_gpu)))

    def close(self, abort=False):
        if self.closed:
            return
        if hasattr(self, "worker_info"):
            self._closed_stats = self.stats
        self.closed = True
        if self.process is not None and self.process.pid is not None:
            if abort:
                if self.process.is_alive():
                    self.process.terminate()
            elif self.process.is_alive():
                try:
                    self.commands.put(dict(kind="stop"), timeout=5)
                except queue.Full:
                    self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        torch.cuda.synchronize()
        for slot in self.slots.values():
            slot.close()
        self.slots.clear()
        for name in ("archive_gpu", "archive_meta_gpu", "query_gpu", "result_gpu", "result_meta_gpu"):
            if hasattr(self, name):
                delattr(self, name)
        for channel in (self.commands, self.replies):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()

