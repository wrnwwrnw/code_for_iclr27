"""Multiplex existing exact-recall transport semantics into one CPU worker."""

from collections import deque
import multiprocessing as mp
import os
import queue
import time

import numpy as np
import torch

from .batch_worker import run_batch_worker
from .transport import RecallTransport, SharedArray


class ReplyInbox:
    def __init__(self, pool, request_id):
        self.pool, self.request_id = pool, request_id

    def get_nowait(self):
        self.pool.route()
        return self.pool.inboxes[self.request_id].get_nowait()


class BatchRecallPool:
    def __init__(self, request_count, recall):
        context = mp.get_context("spawn")
        self.timeout = recall.timeout_seconds
        self.commands = context.Queue(maxsize=request_count * (recall.archive_buffers + 6))
        self.replies = context.Queue()
        self.inboxes, self.endpoints = {}, {}
        self.closed = False
        self.process = context.Process(target=run_batch_worker, args=(self.commands, self.replies, recall.cpu_threads),
                                       daemon=True, name="LazyRecall-Batch-FAISS")
        environment = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        try:
            for name in environment:
                os.environ[name] = str(recall.cpu_threads)
            self.process.start()
            ready = self.replies.get(timeout=self.timeout)
            if ready["kind"] != "ready":
                raise RuntimeError(f"Batch CPU worker failed: {ready}")
            self.worker_info = ready
        except BaseException:
            self.close(abort=True)
            raise
        finally:
            for name, value in environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def send(self, message):
        if self.closed or not self.process.is_alive():
            raise RuntimeError("Batch CPU recall worker is not alive")
        try:
            self.commands.put(message, timeout=self.timeout)
        except queue.Full as error:
            raise TimeoutError("Batch recall command queue is full") from error

    def route(self):
        while True:
            try:
                message = self.replies.get_nowait()
            except queue.Empty:
                break
            if message["kind"] == "error":
                raise RuntimeError(message["traceback"])
            request_id = message["request_id"]
            if request_id not in self.inboxes:
                raise RuntimeError("Reply for an unregistered request; refusing cross-request delivery")
            self.inboxes[request_id].put(message)

    def wait_control(self, request_id, expected):
        started = time.perf_counter()
        while True:
            self.route()
            try:
                message = self.inboxes[request_id].get_nowait()
                if message["kind"] != expected:
                    raise RuntimeError(f"Unexpected batch recall lifecycle reply: {message}")
                return
            except queue.Empty:
                if not self.process.is_alive() or time.perf_counter() - started > self.timeout:
                    raise TimeoutError(f"Batch recall {expected} timed out")
                time.sleep(0.001)

    def create(self, request_id, shape, config):
        if request_id in self.inboxes:
            raise ValueError("Duplicate batch recall request ID")
        self.inboxes[request_id] = queue.Queue()
        endpoint = BatchRecallTransport(self, request_id, shape, config)
        self.endpoints[request_id] = endpoint
        return endpoint

    def close(self, abort=False):
        if self.closed:
            return
        try:
            if not abort:
                for endpoint in list(self.endpoints.values()):
                    if not endpoint.closed:
                        endpoint.finish()
                        endpoint.close()
                if self.process.pid is not None and self.process.is_alive():
                    self.send(dict(kind="stop"))
        finally:
            self.closed = True
            if self.process.pid is not None:
                if abort and self.process.is_alive():
                    self.process.terminate()
                self.process.join(timeout=10)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=5)
            for endpoint in list(self.endpoints.values()):
                endpoint.release_buffers()
            for channel in (self.commands, self.replies):
                channel.cancel_join_thread()
                channel.close()


class BatchRecallTransport(RecallTransport):
    def __init__(self, pool, request_id, shape, config):
        recall = config.recall
        self.pool, self.request_id = pool, request_id
        self.layers, self.heads = shape.num_hidden_layers, shape.num_key_value_heads
        self.groups = shape.num_attention_heads // self.heads
        self.dimension = getattr(shape, "head_dim", shape.hidden_size // shape.num_attention_heads)
        self.search_interval, self.timeout = recall.search_interval, recall.timeout_seconds
        self.capacity = max(config.evict_interval, recall.topk_per_head) * self.layers * self.heads
        self.result_capacity = recall.topk_per_head * self.layers * self.heads
        self.closed, self.slots = False, {}
        self.process = pool.process
        self.replies = ReplyInbox(pool, request_id)
        self.worker_info = dict(pool.worker_info, request_id=request_id, shared_worker=True)
        self.copy_stream = torch.cuda.Stream()
        self.archive_states = ["free"] * recall.archive_buffers
        self.query_states = ["free"] * 2
        self.archive_jobs, self.query_jobs, self.events = {}, {}, deque(maxlen=32)
        self.counters = dict(archived_vectors=0, archive_batches=0, indexed_vectors=0,
                             submitted_queries=0, searches_completed=0, skipped_busy_queries=0,
                             retrieved_vectors=0, spliced_vectors=0, splices=0,
                             d2h_bytes=0, h2d_bytes=0, archive_overflows=0,
                             cpu_peak_rss_kib=0, query_age_tokens_max=0,
                             archive_backpressure=0, capacity_drops=0, superseded_results=0)
        specs = dict(archive_kv=((recall.archive_buffers, self.capacity, 2, self.dimension), np.uint16),
                     archive_meta=((recall.archive_buffers, self.capacity, 3), np.int64),
                     queries=((2, self.layers, self.heads, self.dimension), np.float32),
                     result_kv=((2, self.result_capacity, 2, self.dimension), np.uint16),
                     result_meta=((2, self.result_capacity, 3), np.int64))
        self.config = dict(layers=self.layers, heads=self.heads, dimension=self.dimension,
                           topk=recall.topk_per_head, cpu_threads=recall.cpu_threads,
                           archive_capacity_per_head=recall.archive_capacity_per_head)
        try:
            for name, (shape, dtype) in specs.items():
                self.slots[name] = SharedArray(shape, dtype)
            for name, spec, dtype in (
                ("archive_gpu", "archive_kv", torch.bfloat16), ("archive_meta_gpu", "archive_meta", torch.int64),
                ("query_gpu", "queries", torch.float32), ("result_gpu", "result_kv", torch.bfloat16),
                ("result_meta_gpu", "result_meta", torch.int64),
            ):
                setattr(self, name, torch.empty(specs[spec][0], device="cuda", dtype=dtype))
            self._send(dict(kind="register", config=self.config,
                            descriptors={name: slot.descriptor for name, slot in self.slots.items()}))
            pool.wait_control(request_id, "registered")
        except BaseException:
            pool.close(abort=True)
            self.release_buffers()
            raise

    def _send(self, message):
        self.pool.send(dict(message, request_id=self.request_id))

    def close(self, abort=False):
        if self.closed:
            return
        if abort:
            self.pool.close(abort=True)
            return
        self._closed_stats = self.stats
        self._send(dict(kind="unregister"))
        self.pool.wait_control(self.request_id, "unregistered")
        self.release_buffers()

    def release_buffers(self):
        if self.closed:
            return
        torch.cuda.synchronize()
        self.closed = True
        for slot in self.slots.values():
            slot.close()
        self.slots.clear()
        for name in ("archive_gpu", "archive_meta_gpu", "query_gpu", "result_gpu", "result_meta_gpu"):
            if hasattr(self, name):
                delattr(self, name)
