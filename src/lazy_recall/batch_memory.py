"""Conservative admission estimate, not a guarantee against runtime OOM."""

from dataclasses import asdict, dataclass
import math
import warnings


@dataclass(frozen=True)
class BatchCapacity:
    requested_batch_size: int
    safe_batch_size: int
    free_bytes: int
    total_bytes: int
    model_bytes: int
    spendable_bytes: int
    shared_workspace_bytes: int
    per_request_bytes: int
    components: dict
    cpu_archive_bound_bytes_per_request: int
    model_already_loaded: bool

    def to_dict(self):
        return asdict(self)

    def require(self):
        if self.requested_batch_size > self.safe_batch_size:
            message = (f"Requested batch_size={self.requested_batch_size} exceeds the conservative GPU limit "
                       f"{self.safe_batch_size}; free={self.free_bytes/2**30:.2f} GiB, "
                       f"estimated/request={self.per_request_bytes/2**30:.2f} GiB. "
                       "Inference refused; reduce batch size, prompt length or KV budget.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            raise MemoryError(message)
        return self


def shape_values(model_config):
    from .models import batch_rope_type, text_config
    shape = text_config(model_config)
    kind = shape.model_type
    if kind not in ("llama", "mistral"):
        raise ValueError("Memory admission for batch>1 supports Llama/full-attention Mistral only")
    if getattr(shape, "sliding_window", None) is not None:
        raise ValueError("Batch mode requires full attention, not sliding-window attention")
    if batch_rope_type(shape) not in ("default", "llama3", "linear", "yarn"):
        raise ValueError("Batch mode rejects length-dependent dynamic RoPE; use batch=1")
    return (shape.num_hidden_layers, shape.hidden_size, shape.intermediate_size,
            shape.num_attention_heads, shape.num_key_value_heads,
            getattr(shape, "head_dim", shape.hidden_size // shape.num_attention_heads), shape.vocab_size)


def estimate_model_bytes(model_config):
    layers, hidden, intermediate, query_heads, kv_heads, dimension, vocabulary = shape_values(model_config)
    embeddings = vocabulary * hidden * (1 if model_config.tie_word_embeddings else 2)
    attention = 2 * hidden * query_heads * dimension + 2 * hidden * kv_heads * dimension
    biases = (query_heads * dimension + 2 * kv_heads * dimension + hidden
              if getattr(model_config, "attention_bias", False) else 0)
    biases += 2 * intermediate + hidden if getattr(model_config, "mlp_bias", False) else 0
    return 2 * (embeddings + layers * (attention + 3 * hidden * intermediate + 2 * hidden + biases) + hidden)


def estimate_capacity(model_config, config, prompt_lengths, free_bytes, total_bytes, model_bytes=None, loaded=False):
    config.validate()
    if not prompt_lengths or any(type(length) is not int or length < 1 for length in prompt_lengths):
        raise ValueError("Capacity checks require nonempty, actual tokenized prompt lengths")
    layers, hidden, intermediate, query_heads, kv_heads, dimension, vocabulary = shape_values(model_config)
    if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
        raise ValueError("Invalid GPU memory readings")
    prompt = max(prompt_lengths)
    if prompt + config.max_new_tokens > model_config.max_position_embeddings or prompt + config.max_new_tokens >= 2**24:
        raise ValueError("Prompt + generation exceeds the supported context/position range")
    generation = min(config.budget, config.max_new_tokens)
    kv = (prompt + generation) * layers * kv_heads * dimension * 4
    history = min(config.temporal_window, config.max_new_tokens) * layers * query_heads * dimension * 2
    metadata = (prompt + generation) * layers * kv_heads * 12
    token_state = (prompt + config.max_new_tokens) * 16 + vocabulary * 16
    backward = layers * (8 * hidden + 6 * intermediate + 4 * (query_heads + 2 * kv_heads) * dimension) * 4
    recall = config.recall
    if recall.enabled:
        archive_rows = recall.archive_buffers * max(config.evict_interval, recall.topk_per_head) * layers * kv_heads
        result_rows = 2 * recall.topk_per_head * layers * kv_heads
        staging = (archive_rows + result_rows) * (4 * dimension + 24) + 2 * layers * kv_heads * dimension * 4
        archive_bound = min(recall.archive_capacity_per_head, config.max_new_tokens) * layers * kv_heads * (8 * dimension + 96)
    else:
        staging = archive_bound = 0
    components = dict(kv=kv, append_and_audit_kv_reserve=kv, history=history, position_and_queue_metadata=metadata,
                      token_and_logit_state=token_state, backward_activations=backward, recall_gpu_staging=staging)
    per_request = sum(components.values())
    prefill = prompt * (8 * hidden + 4 * intermediate + 4 * query_heads * dimension) * 2
    head_and_scoring = 4 * hidden * min(4096, vocabulary) + 4 * hidden * hidden + 128 * 2**20
    shared = max(prefill, head_and_scoring)
    model_bytes = estimate_model_bytes(model_config) if model_bytes is None else int(model_bytes)
    spendable = max(0, math.floor(free_bytes * config.memory_safety_fraction) - config.memory_reserve_mib * 2**20)
    remaining = max(0, spendable - shared - (0 if loaded else model_bytes))
    cap = remaining // max(1, per_request)
    return BatchCapacity(config.batch_size, cap, int(free_bytes), int(total_bytes), model_bytes,
                         spendable, shared, per_request, components, archive_bound, loaded)


def model_storage_bytes(model):
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def check_batch_capacity(model, config, prompt_lengths):
    import torch
    device = model.get_input_embeddings().weight.device
    free, total = torch.cuda.mem_get_info(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return estimate_capacity(model.config, config, prompt_lengths, min(total, free + reusable), total,
                             model_storage_bytes(model), loaded=True).require()


def preflight_capacity(model_config, config, prompt_lengths):
    import torch
    free, total = torch.cuda.mem_get_info(torch.device(config.device))
    return estimate_capacity(model_config, config, prompt_lengths, free, total).require()
