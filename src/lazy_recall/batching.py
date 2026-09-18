"""Static request batches with shared weights, independent policies and real recall."""

from dataclasses import dataclass
import time

import torch

from .batch_attention import BatchAttentionPatch, BatchCache
from .batch_audit import BatchAudit
from .batch_memory import check_batch_capacity
from .cache import LazyRecallCache
from .config import GenerationConfig
from .controller import LazyController
from .generation import GenerationResult, generate, seed_everything
from .history import QueryHistory
from .models import batch_rope_type


@dataclass
class RequestState:
    request_id: int
    prompt_length: int
    cache: LazyRecallCache
    history: QueryHistory
    controller: LazyController
    processed: int = 0
    finished: bool = False
    final_stats: dict | None = None


def validate_batch_model(model, config):
    config.validate()
    device = model.get_input_embeddings().weight.device
    if device.type != "cuda" or next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("Batch inference requires an on-device BF16 model")
    if torch.is_inference_mode_enabled() or model.training:
        raise ValueError("Use model.eval() without torch.inference_mode(); audits need autograd")
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("Model sharding and offloading are unsupported")
    if model.config.model_type not in ("llama", "mistral") or getattr(model.config, "sliding_window", None) is not None:
        raise ValueError("Batch>1 supports Llama/full-attention Mistral only; use batch=1 for Qwen3.5")
    if batch_rope_type(model.config) not in ("default", "llama3", "linear", "yarn"):
        raise ValueError("Batch mode rejects length-dependent dynamic RoPE; use batch=1")
    if model.config._attn_implementation != "flash_attention_2":
        raise ValueError("Load the model with flash_attention_2")
    torch.cuda.set_device(device)


class BatchEngine:
    def __init__(self, model, config, prompt_lengths):
        validate_batch_model(model, config)
        if not prompt_lengths or len(prompt_lengths) > config.batch_size:
            raise ValueError("Prompt count must lie within configured batch_size")
        self.capacity = check_batch_capacity(model, config, prompt_lengths)
        self.model, self.config = model, config
        self.device = model.get_input_embeddings().weight.device
        self.requests, self.patch, self.pool = [], None, None
        self.cache = BatchCache()
        self.closed = False
        self.batch_audits = 0
        try:
            self.patch = BatchAttentionPatch(model)
            if config.recall.enabled:
                from .recall.batch_transport import BatchRecallPool
                self.pool = BatchRecallPool(len(prompt_lengths), config.recall)
            for request_id, length in enumerate(prompt_lengths):
                transport = self.pool.create(request_id, model.config, config) if self.pool is not None else None
                cache = LazyRecallCache(transport)
                history = QueryHistory(model.config.num_hidden_layers, config.temporal_window)
                controller = LazyController(model.config, config, history)
                self.requests.append(RequestState(request_id, length, cache, history, controller))
        except BaseException:
            self.close(abort=True)
            raise

    def prefill(self, inputs):
        if len(inputs) != len(self.requests) or any(len(request.cache) for request in self.requests):
            raise ValueError("Prefill requires one fresh unpadded prompt per request")
        logits = []
        for request, tokens in zip(self.requests, inputs):
            if tokens.shape != (1, request.prompt_length):
                raise ValueError("Prompt shape differs from its tokenized length")
            self.cache.select([request])
            positions = torch.arange(request.prompt_length, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, past_key_values=self.cache, use_cache=True,
                                     position_ids=positions[None], cache_position=positions, logits_to_keep=1)
            logits.append(outputs.logits[:, -1].detach())
            request.controller.on_prefill(request.cache)
        return torch.cat(logits)

    def step(self, tokens, request_ids):
        if self.closed or torch.is_inference_mode_enabled():
            raise ValueError("Engine closed or autograd disabled by inference_mode")
        if len(set(request_ids)) != len(request_ids) or any(type(index) is not int or not 0 <= index < len(self.requests)
                                                           for index in request_ids):
            raise ValueError("Request IDs must be distinct live batch IDs")
        requests = [self.requests[index] for index in request_ids]
        if not requests or tokens.shape != (len(requests), 1):
            raise ValueError("Exactly one input token is required per active request")
        if any(request.finished or request.processed >= self.config.max_new_tokens - 1 for request in requests):
            raise ValueError("Cannot append to a finished or over-length request")
        due = [request.controller.needs_audit() for request in requests]
        for request, audit_due in zip(requests, due):
            if audit_due:
                request.controller.prepare_audit(request.cache)
            if request.cache.recalled_identities:
                request.cache.recalled_visible_forwards += 1
        self.cache.select(requests)
        positions = torch.tensor([[request.prompt_length + request.processed] for request in requests],
                                 device=self.device, dtype=torch.long)
        arguments = dict(input_ids=tokens, past_key_values=self.cache, use_cache=True,
                         position_ids=positions, cache_position=positions[0], logits_to_keep=1)
        if any(due):
            with BatchAudit(self.model, self.cache, due) as session:
                with torch.enable_grad():
                    outputs = self.model(**arguments, output_hidden_states=True)
                    audits = session.finish(outputs)
            self.batch_audits += 1
        else:
            with torch.no_grad():
                outputs = self.model(**arguments)
            audits = [None] * len(requests)
        logits = outputs.logits[:, -1].detach()
        del outputs
        for row, (request, audit) in enumerate(zip(requests, audits)):
            position = request.prompt_length + request.processed
            with self.patch.request_queries(row):
                if audit is None:
                    request.history.capture_model(self.model, position)
                else:
                    request.history.append(audit.attention_queries, position)
                request.controller.on_decode(self.model, request.cache, audit)
            request.processed += 1
        return logits

    def request_stats(self, request_id):
        request = self.requests[request_id]
        if request.final_stats is not None:
            return request.final_stats
        stats = request.controller.stats(request.cache)
        stats.update(request_id=request_id, prompt_tokens=request.prompt_length,
                     processed_generation_tokens=request.processed,
                     architecture=self.model.config.model_type, linear_state_bytes=0,
                     kv_bytes=sum(tensor.numel() * tensor.element_size()
                                  for tensor in request.cache.key_cache + request.cache.value_cache),
                     query_history_bytes=request.history.retained_bytes)
        return stats

    def retire(self, request_id):
        request = self.requests[request_id]
        if request.finished:
            return
        if request.cache.transport is not None:
            request.cache.transport.finish()
            request.cache.transport.close()
        request.final_stats = self.request_stats(request_id)
        request.finished = True
        request.cache.key_cache.clear()
        request.cache.value_cache.clear()
        request.cache.position_cache.clear()
        request.history = None
        request.controller.history = None

    def close(self, abort=False):
        if self.closed:
            return
        try:
            if not abort:
                for request in self.requests:
                    self.retire(request.request_id)
            if self.pool is not None:
                self.pool.close(abort=abort)
        finally:
            if self.pool is not None and not self.pool.closed:
                self.pool.close(abort=True)
            if self.patch is not None:
                self.patch.close()
            self.cache.select([])
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(abort=kind is not None)


@dataclass
class BatchGenerationResult:
    results: list
    batch_size: int
    elapsed_seconds: float
    generated_tokens: int
    tokens_per_second: float
    capacity: dict | None


def tokenize_prompts(tokenizer, prompts, config):
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a nonempty list of strings")
    if config.chat_template:
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template")
        return [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                             add_generation_prompt=True) for prompt in prompts]
    return [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]


def generate_batch(model, tokenizer, prompts, config=None):
    config = (GenerationConfig() if config is None else config).validate()
    if isinstance(prompts, str) or not prompts:
        raise ValueError("Pass a nonempty list of prompts")
    if config.batch_size == 1:
        started = time.perf_counter()
        results = [generate(model, tokenizer, prompt, config) for prompt in prompts]
        elapsed = time.perf_counter() - started
        count = sum(result.generated_tokens for result in results)
        return BatchGenerationResult(results, 1, elapsed, count, count / elapsed, None)
    identifiers = tokenize_prompts(tokenizer, prompts, config)
    capacity = check_batch_capacity(model, config, [len(prompt) for prompt in identifiers])
    all_results = []
    started = time.perf_counter()
    for offset in range(0, len(prompts), config.batch_size):
        batch = identifiers[offset:offset + config.batch_size]
        results = _generate_chunk(model, tokenizer, batch, config)
        for row, result in enumerate(results):
            result.stats["input_index"] = offset + row
        all_results.extend(results)
    elapsed = time.perf_counter() - started
    count = sum(result.generated_tokens for result in all_results)
    return BatchGenerationResult(all_results, config.batch_size, elapsed, count, count / elapsed, capacity.to_dict())


def _generate_chunk(model, tokenizer, identifiers, config):
    from transformers import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper
    seed_everything(config.seed)
    device = model.get_input_embeddings().weight.device
    inputs = [torch.tensor([prompt], dtype=torch.long, device=device) for prompt in identifiers]
    outputs = [torch.empty((1, len(prompt) + config.max_new_tokens), device=device, dtype=torch.long) for prompt in identifiers]
    for output, prompt in zip(outputs, inputs):
        output[:, :prompt.shape[1]] = prompt
    processors = []
    if config.repetition_penalty != 1:
        processors.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))
    if config.temperature > 0:
        processors.append(TemperatureLogitsWarper(config.temperature))
        if config.top_p < 1:
            processors.append(TopPLogitsWarper(config.top_p))
    generators = [torch.Generator(device=device).manual_seed(config.seed) for _ in identifiers]
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    eos_tensor = torch.tensor(eos, device=device, dtype=torch.long)
    counts = [0] * len(identifiers)
    active = list(range(len(identifiers)))
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    engine = BatchEngine(model, config, [len(prompt) for prompt in identifiers])
    try:
        logits = engine.prefill(inputs)
        while active:
            if not processors and config.temperature == 0:
                chosen = logits.float().argmax(-1, keepdim=True)
            else:
                samples = []
                for row, request_id in enumerate(active):
                    scores = logits[row:row + 1].float()
                    position = len(identifiers[request_id]) + counts[request_id]
                    for processor in processors:
                        scores = processor(outputs[request_id][:, :position], scores)
                    token = (scores.argmax(-1, keepdim=True) if config.temperature == 0 else
                             torch.multinomial(scores.softmax(-1), 1, generator=generators[request_id]))
                    samples.append(token)
                chosen = torch.cat(samples)
            ended = ((chosen == eos_tensor.reshape(1, -1)).any(-1).tolist()
                     if config.stop_on_eos and eos else [False] * len(active))
            remaining, next_rows = [], []
            for row, request_id in enumerate(active):
                token = chosen[row:row + 1]
                position = len(identifiers[request_id]) + counts[request_id]
                outputs[request_id][:, position:position + 1] = token
                counts[request_id] += 1
                done = counts[request_id] >= config.max_new_tokens or ended[row]
                if done:
                    engine.retire(request_id)
                else:
                    remaining.append(request_id)
                    next_rows.append(row)
            all_continuing = len(remaining) == len(active)
            active = remaining
            if active:
                next_tokens = chosen if all_continuing else chosen.index_select(
                    0, torch.tensor(next_rows, device=device, dtype=torch.long))
                logits = engine.step(next_tokens, active)
        engine.close()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        results = []
        for request_id, prompt in enumerate(identifiers):
            generated = outputs[request_id][0, len(prompt):len(prompt) + counts[request_id]].tolist()
            stats = engine.request_stats(request_id)
            stats["elapsed_scope"] = "whole_static_batch_including_recall_drain"
            results.append(GenerationResult(tokenizer.decode(generated, skip_special_tokens=True), generated,
                                            len(prompt), len(generated), elapsed,
                                            stats, config.to_dict()))
        return results
    except BaseException:
        engine.close(abort=True)
        raise
