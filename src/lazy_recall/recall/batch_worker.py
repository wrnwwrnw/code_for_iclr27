"""One CPU worker; disjoint FAISS namespaces and DMA views per request."""

from multiprocessing import shared_memory
import os
import resource
import sys
import time
import traceback

import numpy as np

from .cpu_worker import ArchiveIndex


class RequestArchive:
    def __init__(self, config, descriptors):
        self.config = config
        self.handles, self.views = [], {}
        try:
            for name, descriptor in descriptors.items():
                handle = shared_memory.SharedMemory(name=descriptor["name"])
                self.handles.append(handle)
                self.views[name] = np.ndarray(descriptor["shape"], dtype=np.dtype(descriptor["dtype"]), buffer=handle.buf)
            self.index = ArchiveIndex(config["layers"], config["heads"], config["dimension"],
                                      config["cpu_threads"], config["archive_capacity_per_head"])
        except BaseException:
            self.close()
            raise

    def execute(self, message):
        kind, slot = message["kind"], message.get("slot")
        if kind == "archive":
            count = message["count"]
            self.index.add(self.views["archive_meta"][slot, :count], self.views["archive_kv"][slot, :count])
            return dict(kind="archived", slot=slot, count=count)
        if kind == "remove":
            self.index.remove(message["metadata"])
            return None
        if kind == "search":
            metadata, payload = self.index.search(self.views["queries"][slot], self.config["topk"])
            count = len(metadata)
            self.views["result_meta"][slot, :count] = metadata
            self.views["result_kv"][slot, :count] = payload
            return dict(kind="searched", slot=slot, count=count, query_step=message["step"])
        if kind == "flush":
            return dict(kind="flushed")
        raise ValueError(f"Unknown request command: {kind}")

    def close(self):
        self.views.clear()
        for handle in self.handles:
            handle.close()
        self.handles.clear()
        self.index = None


def run_batch_worker(commands, replies, cpu_threads):
    archives, seen = {}, set()
    request_id = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import faiss
        faiss.omp_set_num_threads(cpu_threads)
        replies.put(dict(kind="ready", request_id=None, pid=os.getpid(), torch_imported="torch" in sys.modules,
                         faiss_version=faiss.__version__, cpu_threads=cpu_threads))
        while True:
            message = commands.get()
            request_id = message.get("request_id")
            kind = message["kind"]
            if kind == "stop":
                break
            if type(request_id) is not int or request_id < 0:
                raise ValueError("Every batch recall message needs a nonnegative request ID")
            started = time.perf_counter_ns()
            if kind == "register":
                if request_id in seen:
                    raise ValueError("Request IDs cannot be reused within a recall worker lifetime")
                config = dict(message["config"], cpu_threads=cpu_threads)
                archives[request_id] = RequestArchive(config, message["descriptors"])
                seen.add(request_id)
                replies.put(dict(kind="registered", request_id=request_id))
                continue
            if request_id not in archives:
                raise ValueError("Unknown or already retired request")
            archive = archives[request_id]
            if kind == "unregister":
                archive.close()
                del archives[request_id]
                replies.put(dict(kind="unregistered", request_id=request_id))
                continue
            reply = archive.execute(message)
            if reply is not None:
                reply.update(request_id=request_id, indexed=archive.index.size, capacity_drops=archive.index.dropped,
                             start_ns=started, end_ns=time.perf_counter_ns(),
                             cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                             (1024 if sys.platform == "darwin" else 1))
                replies.put(reply)
    except BaseException:
        replies.put(dict(kind="error", request_id=request_id, traceback=traceback.format_exc()))
    finally:
        for archive in archives.values():
            archive.close()
