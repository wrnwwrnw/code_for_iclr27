from dataclasses import dataclass
import random
import time

import numpy as np
import torch

from .audit import DecisionAudit
from .config import GenerationConfig
from .controller import LazyController
from .cuda_ops import extension
from .history import QueryHistory
from .models import architecture, check_runtime, create_backend, management_config, text_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config.validate()
    if not config.model:
        raise ValueError("Set --model to a local model directory")
    model_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
    check_runtime(model_config)
    if config.batch_size > 1 and architecture(model_config) not in ("llama", "mistral"):
        raise ValueError("Batch>1 currently supports Llama/full-attention Mistral; use batch=1 for Qwen3.5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(config.device))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA device is required")
    extension()
    model_class = AutoModelForCausalLM
    extra = {}
    if architecture(model_config) == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        model_class = Qwen3_5ForCausalLM
        extra["config"] = text_config(model_config)
    model = model_class.from_pretrained(
        config.model, dtype=torch.bfloat16, device_map={"": config.device},
        attn_implementation="flash_attention_2", local_files_only=True, **extra,
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
    return model, tokenizer


class LazyRecallEngine:
    def __init__(self, model, config, prompt_length):
        self.model, self.config = model, config.validate()
        self.device = model.get_input_embeddings().weight.device
        if self.device.type != "cuda" or next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError("Use an on-device BF16 model")
        if any(parameter.device != self.device for parameter in model.parameters()):
            raise ValueError("Model sharding and CPU offloading are unsupported")
        if text_config(model.config)._attn_implementation != "flash_attention_2":
            raise ValueError("Load the model with attn_implementation='flash_attention_2'")
        if torch.is_inference_mode_enabled():
            raise ValueError("Do not wrap LazyRecall in torch.inference_mode(); audits require autograd")
        if type(prompt_length) is not int or prompt_length < 1:
            raise ValueError("A nonempty tokenized prompt is required")
        torch.cuda.set_device(self.device)
        if model.training:
            raise ValueError("Call model.eval() before inference")
        total = prompt_length + config.max_new_tokens
        if total > text_config(model.config).max_position_embeddings or total >= 2 ** 24:
            raise ValueError("Prompt + generation exceeds supported absolute-position range")
        self.prompt_length, self.processed = prompt_length, 0
        self.backend = self.transport = None
        self.closed = False
        self.positions = torch.arange(total, device=self.device, dtype=torch.long)
        try:
            shape = management_config(model)
            if config.recall.enabled:
                from .recall.transport import RecallTransport
                recall = config.recall
                self.transport = RecallTransport(
                    shape.num_hidden_layers, shape.num_key_value_heads,
                    shape.num_attention_heads,
                    getattr(shape, "head_dim", shape.hidden_size // shape.num_attention_heads),
                    search_interval=recall.search_interval, topk=recall.topk_per_head,
                    archive_buffers=recall.archive_buffers, cpu_threads=recall.cpu_threads,
                    evict_interval=config.evict_interval,
                    archive_capacity_per_head=recall.archive_capacity_per_head,
                    timeout_seconds=recall.timeout_seconds,
                )
            self.backend = create_backend(model, self.transport)
            self.cache, self.model_cache = self.backend.cache, self.backend.model_cache
            self.view = self.backend.view
            self.history = QueryHistory(shape.num_hidden_layers, config.temporal_window)
            self.controller = LazyController(shape, config, self.history)
        except BaseException:
            self.close(abort=True)
            raise

    def prefill(self, input_ids):
        if input_ids.shape != (1, self.prompt_length) or len(self.cache):
            raise ValueError("Prefill must initialize one fresh, unpadded request")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, past_key_values=self.model_cache, use_cache=True,
                                 position_ids=self.positions[:self.prompt_length].unsqueeze(0),
                                 cache_position=self.positions[:self.prompt_length], logits_to_keep=1)
        self.controller.on_prefill(self.cache)
        return outputs.logits[:, -1].detach()

    def step(self, token):
        if torch.is_inference_mode_enabled():
            raise ValueError("Audits require autograd; use no_grad rather than inference_mode")
        if token.shape != (1, 1) or self.processed >= self.config.max_new_tokens - 1:
            raise ValueError("Decode expects one token within the configured request length")
        due = self.controller.needs_audit()
        if due:
            self.controller.prepare_audit(self.cache)
        if self.cache.recalled_identities:
            self.cache.recalled_visible_forwards += 1
        position = self.prompt_length + self.processed
        kwargs = dict(input_ids=token, past_key_values=self.model_cache, use_cache=True,
                      position_ids=self.positions[position:position + 1].unsqueeze(0),
                      cache_position=self.positions[position:position + 1], logits_to_keep=1)
        audit = None
        if due:
            with self.backend.audit_context(), DecisionAudit(self.view, self.cache) as session:
                with torch.enable_grad():
                    outputs = self.model(**kwargs, output_hidden_states=True)
                    audit = session.finish(outputs)
            self.history.append(audit.attention_queries, position)
        else:
            with torch.no_grad():
                outputs = self.model(**kwargs)
            self.history.capture_model(self.view, position)
        logits = outputs.logits[:, -1].detach()
        del outputs
        self.controller.on_decode(self.view, self.cache, audit)
        self.processed += 1
        return logits

    def stats(self):
        stats = self.controller.stats(self.cache)
        stats.update(self.backend.stats())
        stats.update(prompt_tokens=self.prompt_length, processed_generation_tokens=self.processed,
                     kv_bytes=sum(tensor.numel() * tensor.element_size()
                                  for tensor in self.cache.key_cache + self.cache.value_cache),
                     query_history_bytes=self.history.retained_bytes)
        return stats

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.transport is not None:
                try:
                    if not abort:
                        self.transport.finish()
                finally:
                    self.transport.close(abort=abort)
        finally:
            if self.backend is not None:
                self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(abort=kind is not None)


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    stats: dict
    config: dict


def generate(model, tokenizer, prompt, config=None):
    from transformers import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper

    config = GenerationConfig() if config is None else config
    config.validate()
    if config.batch_size != 1:
        raise ValueError("Use generate_batch(model, tokenizer, prompts, config) for batch_size>1")
    seed_everything(config.seed)
    if config.chat_template:
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template; use chat_template=false")
        identifiers = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    else:
        identifiers = tokenizer.encode(prompt, add_special_tokens=True)
    device = model.get_input_embeddings().weight.device
    prompt_ids = torch.tensor([identifiers], dtype=torch.long, device=device)
    output = torch.empty((1, len(identifiers) + config.max_new_tokens), dtype=torch.long, device=device)
    output[:, :len(identifiers)] = prompt_ids
    processors = []
    if config.repetition_penalty != 1:
        processors.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))
    if config.temperature > 0:
        processors.append(TemperatureLogitsWarper(config.temperature))
        if config.top_p < 1:
            processors.append(TopPLogitsWarper(config.top_p))
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    eos_tensor = torch.tensor(eos, device=device, dtype=torch.long)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    engine = LazyRecallEngine(model, config, len(identifiers))
    try:
        logits = engine.prefill(prompt_ids)
        produced = 0
        for index in range(config.max_new_tokens):
            scores = logits.float()
            previous = output[:, :len(identifiers) + index]
            for processor in processors:
                scores = processor(previous, scores)
            token = (scores.argmax(-1, keepdim=True) if config.temperature == 0
                     else torch.multinomial(scores.softmax(-1), 1))
            output[:, len(identifiers) + index:len(identifiers) + index + 1] = token
            produced += 1
            if config.stop_on_eos and bool((token == eos_tensor).any()):
                break
            if produced < config.max_new_tokens:
                logits = engine.step(token)
        engine.close()
        stats = engine.stats()
    except BaseException:
        engine.close(abort=True)
        raise
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated = output[0, len(identifiers):len(identifiers) + produced].tolist()
    return GenerationResult(tokenizer.decode(generated, skip_special_tokens=True), generated,
                            len(identifiers), produced, elapsed, stats, config.to_dict())
