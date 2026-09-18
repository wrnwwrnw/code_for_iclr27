"""CPU-owned bounded exact MIPS archive. No torch or CUDA imports."""

import os
import resource
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np


def bf16_to_float32(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


class HeadArchive:
    def __init__(self, dimension, capacity, faiss):
        self.dimension, self.capacity = dimension, capacity
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.payload = np.empty((0, 2, dimension), dtype=np.uint16)
        self.positions = np.empty(0, dtype=np.int64)
        self.lookup = {}
        self.free = []
        self.used = 0
        self.cursor = 0
        self.dropped = 0

    def _grow(self):
        length = min(self.capacity, max(128, len(self.payload) * 2))
        payload = np.empty((length, 2, self.dimension), dtype=np.uint16)
        positions = np.full(length, -1, dtype=np.int64)
        payload[:self.used] = self.payload[:self.used]
        positions[:self.used] = self.positions[:self.used]
        self.payload, self.positions = payload, positions

    def remove(self, positions):
        selected = np.array([int(position) for position in positions if int(position) in self.lookup],
                            dtype=np.int64)
        if not len(selected):
            return
        self.index.remove_ids(selected)
        for position in selected:
            slot = self.lookup.pop(int(position))
            self.positions[slot] = -1
            self.free.append(slot)

    def add(self, positions, payload):
        if len(np.unique(positions)) != len(positions):
            raise ValueError("Duplicate identities in one archive batch")
        self.remove(positions)
        slots, kept = [], []
        for row, position in enumerate(positions):
            if self.free:
                slot = self.free.pop()
            elif self.used < self.capacity:
                if self.used == len(self.payload):
                    self._grow()
                slot = self.used
                self.used += 1
            else:
                slot = self.cursor
                self.cursor = (self.cursor + 1) % self.capacity
                previous = int(self.positions[slot])
                if previous >= 0:
                    self.index.remove_ids(np.array([previous], dtype=np.int64))
                    self.lookup.pop(previous)
                    self.dropped += 1
                if slot in slots:
                    earlier = slots.index(slot)
                    slots.pop(earlier)
                    kept.pop(earlier)
            self.payload[slot] = payload[row]
            self.positions[slot] = int(position)
            self.lookup[int(position)] = slot
            slots.append(slot)
            kept.append(row)
        if kept:
            self.index.add_with_ids(
                np.ascontiguousarray(bf16_to_float32(payload[kept, 0])),
                np.ascontiguousarray(positions[kept], dtype=np.int64),
            )

    def search(self, query, topk):
        if not self.lookup:
            return np.empty(0, np.int64), np.empty((0, 2, self.dimension), np.uint16)
        _, identifiers = self.index.search(np.ascontiguousarray(query.reshape(1, -1)), min(topk, len(self.lookup)))
        positions = identifiers[0]
        positions = positions[positions >= 0]
        slots = [self.lookup[int(position)] for position in positions]
        return positions, self.payload[slots].copy()


class ArchiveIndex:
    def __init__(self, layers, heads, dimension, threads, capacity):
        import faiss
        faiss.omp_set_num_threads(threads)
        self.faiss, self.layers, self.heads, self.dimension = faiss, layers, heads, dimension
        self.indices = [HeadArchive(dimension, capacity, faiss) for _ in range(layers * heads)]

    @property
    def size(self):
        return sum(len(index.lookup) for index in self.indices)

    @property
    def dropped(self):
        return sum(index.dropped for index in self.indices)

    def add(self, metadata, payload):
        if metadata.shape != (len(payload), 3) or payload.shape[1:] != (2, self.dimension):
            raise ValueError("Archive payload shape mismatch")
        if (metadata < 0).any() or (metadata[:, 0] >= self.layers).any() or (metadata[:, 1] >= self.heads).any():
            raise ValueError("Invalid archive identity")
        segments = metadata[:, 0] * self.heads + metadata[:, 1]
        for segment in np.unique(segments):
            selected = segments == segment
            self.indices[int(segment)].add(metadata[selected, 2], payload[selected])

    def remove(self, metadata):
        metadata = np.asarray(metadata, dtype=np.int64).reshape(-1, 3)
        segments = metadata[:, 0] * self.heads + metadata[:, 1]
        for segment in np.unique(segments):
            self.indices[int(segment)].remove(metadata[segments == segment, 2])

    def search(self, queries, topk):
        metadata, payload = [], []
        for segment, index in enumerate(self.indices):
            positions, vectors = index.search(queries.reshape(-1, self.dimension)[segment], topk)
            for position, vector in zip(positions, vectors):
                metadata.append((segment // self.heads, segment % self.heads, int(position)))
                payload.append(vector)
        return (np.asarray(metadata, dtype=np.int64).reshape(-1, 3),
                np.asarray(payload, dtype=np.uint16).reshape(-1, 2, self.dimension))


def run_worker(config, descriptors, commands, replies):
    handles, views = [], {}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for name, descriptor in descriptors.items():
            handle = shared_memory.SharedMemory(name=descriptor["name"])
            handles.append(handle)
            views[name] = np.ndarray(descriptor["shape"], np.dtype(descriptor["dtype"]), buffer=handle.buf)
        index = ArchiveIndex(config["layers"], config["heads"], config["dimension"],
                             config["cpu_threads"], config["archive_capacity_per_head"])
        replies.put(dict(kind="ready", pid=os.getpid(), faiss_version=index.faiss.__version__,
                         torch_imported="torch" in sys.modules))
        while True:
            message = commands.get()
            kind, slot = message["kind"], message.get("slot")
            started = time.perf_counter_ns()
            if kind == "stop":
                return
            if kind == "archive":
                count = message["count"]
                index.add(views["archive_meta"][slot, :count], views["archive_kv"][slot, :count])
                reply = dict(kind="archived", slot=slot, count=count)
            elif kind == "remove":
                index.remove(message["metadata"])
                continue
            elif kind == "search":
                metadata, payload = index.search(views["queries"][slot], config["topk"])
                count = len(metadata)
                views["result_meta"][slot, :count] = metadata
                views["result_kv"][slot, :count] = payload
                reply = dict(kind="searched", slot=slot, count=count, query_step=message["step"])
            elif kind == "flush":
                reply = dict(kind="flushed")
            else:
                raise ValueError(f"Unknown worker command: {kind}")
            reply.update(indexed=index.size, capacity_drops=index.dropped, start_ns=started,
                         end_ns=time.perf_counter_ns(),
                         cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                         (1024 if sys.platform == "darwin" else 1))
            replies.put(reply)
    except BaseException:
        replies.put(dict(kind="error", traceback=traceback.format_exc()))
    finally:
        views.clear()
        for handle in handles:
            handle.close()

