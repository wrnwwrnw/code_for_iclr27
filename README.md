# LazyRecall

> Version 0.3.1 uses one runtime for all supported architectures and supports
> static multi-request decoding for Llama/full-attention Mistral.

Training-free, global KV-cache management with temporal decision-risk scoring,
lazy eviction, and optional CPU retrieval with real GPU reinsertion.

## Supported models

| Decoder | Management | Batch support |
|---|---|---|
| Llama, including Llama-based distilled checkpoints | All full-attention layers | Single/static multi-request |
| Mistral with full attention | All full-attention layers | Single/static multi-request |
| Dense Qwen3.5 text decoder | Full-attention layers only; native DeltaNet states remain | Single only |

Architecture is read from the checkpoint configuration, not inferred from its
directory name. Sliding-window Mistral, Qwen3.5 MoE, multimodal inputs, padded
batches, beam search, model sharding and CPU weight offload are not supported.
The implementation targets one BF16 CUDA GPU. Static batches accept different
unpadded prompt lengths and EOS times; new requests enter only after the current
chunk finishes. Multi-request mode rejects length-dependent dynamic RoPE.
It provides its own autoregressive loop; replacing Hugging Face `generate()`
globally is not required.

## Installation and CUDA compilation

All supported architectures use **one environment**: Python >=3.10,
PyTorch 2.11.0, Transformers 5.14.1 and FlashAttention 2.8.3.
[environment.txt](environment.txt) is a pip-compatible dependency file.
Install a CUDA-enabled PyTorch 2.11.0 build compatible with your driver first;
the local CUDA toolkit must also be compatible with that build.

```bash
python -m pip install setuptools wheel ninja
MAX_JOBS=4 python -m pip install --no-build-isolation -r environment.txt
MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps --no-cache-dir .
```

The last command compiles `lazy_recall._C` (including the new `batch_append`
kernel), `lazy_recall._legacy_C` and
`lazy_recall._host`. `--no-build-isolation` is necessary because the extension
build imports the installed PyTorch. An NVIDIA GPU, NVCC, a C++17 compiler and
FlashAttention with deterministic backward are required for generation.
Optionally set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability.

Recompile after changing PyTorch/CUDA, using a clean source tree without previous
build artifacts. Do not copy compiled extensions between incompatible environments
or machines. The same installation serves Llama, Mistral and Qwen3.5; change the
model path, not the environment. Optional `causal-conv1d` and
`flash-linear-attention` packages accelerate native Qwen3.5 decoding; audits retain
the differentiable Torch recurrence.

```bash
python -c "from lazy_recall import _C, _legacy_C, _host; print(_C.__file__)"
python -c "from lazy_recall import _C; assert hasattr(_C, 'batch_append')"
python run.py --help
```

`LAZY_RECALL_BUILD_CUDA=0` is only for CPU mathematical/development checks; it
does **not** provide a CPU generation fallback.

## Generate

All model files are loaded locally; download the desired checkpoint separately.
The same entry point supports all architectures listed above.

```bash
python run.py --model /path/to/model --config configs/default.json \
  --prompt "Explain a long proof step by step." \
  --budget 2000 --max-new-tokens 8000 --audit-period 250 --recall

python run.py --model /path/to/model --prompt-file input.txt \
  --temperature 0 --no-recall --output outputs/greedy.json
```

For Qwen3.5, pass the local checkpoint to `--model` in the same environment;
no architecture flag is required. Only its text decoder is loaded.
The installed `lazy-recall` command is equivalent to `python run.py`.

```python
from lazy_recall import GenerationConfig, RecallConfig, generate, load_model

config = GenerationConfig(
    model="/path/to/model", budget=2000, audit_period=250,
    temporal_window=250, recall=RecallConfig(enabled=True),
)
model, tokenizer = load_model(config)
result = generate(model, tokenizer, "Continue this explanation.", config)
print(result.text)
print(result.stats)
```

Use a main guard when embedding generation in a script: the recall index uses
a spawned CPU worker. Do not wrap generation in `torch.inference_mode()`:
occasional audits need autograd, while ordinary decoding runs without gradients.
Each model instance supports one active engine, either single-request or batched.
`LazyRecallEngine` is also a context manager for explicit single-request steps.

### Multiple requests

Supply a JSON array of strings, not a padded input tensor:

```bash
python run.py --model /path/to/model --prompts-file configs/prompts.example.json \
  --batch-size 2 --budget 2000 --max-new-tokens 8000 --audit-period 150 \
  --recall --output outputs/batch.json
```

```python
from lazy_recall import GenerationConfig, generate_batch, load_model

def main():
    config = GenerationConfig(model="/path/to/model", batch_size=2)
    model, tokenizer = load_model(config)
    result = generate_batch(model, tokenizer, ["Explain a proof.", "Write a long story."], config)
    for request in result.results:
        print(request.text)

if __name__ == "__main__":
    main()
```

At `batch_size=1`, `generate_batch` calls the original `generate` once per prompt:
the attention, audit, scoring, sampling and recall algorithms remain unchanged.
The cache mask-size interface accepts the query-length API used by Transformers
5.14.1. Updating PyTorch/Transformers does not promise bitwise outputs or identical
timing across runtime versions. At larger values, prefills are sequential and unpadded, but decoding
uses **one model forward for all active requests**. Weights, QKV/MLP projections
and VJP execution are shared, not replicated model processes. `budget` applies
**per request**, not across the batch. Global scoring remains within each request.
EOS removes only that request; output order follows input order. Each request has
its own seeded sampling generator. More than `batch_size` prompts run in static chunks.

FlashAttention segments use `(request, KV head)` boundaries and native GQA.
Queries, RoPE positions, protection masks, queues and admission versions are
request-local. A single VJP uses the **sum of independent margin objectives**,
with zero output gradient for rows not due for an audit; the result is not divided
by batch size. Shared forward/VJP timing fields in per-request audit logs refer to
the same batch operation and must not be summed.

One CPU worker owns separate exact FAISS indices for each request. Messages and
DMA slots carry request identity; search never retrieves globally and filters
other requests afterward. CPU thread count is shared, while archive capacity and
buffers remain per request. IDs are not reused during a worker's lifetime.

Ordinary no-grad decode fuses per-request append and cross-request packing in one
CUDA copy kernel per layer. It copies K/V/positions without rounding or scoring
arithmetic. Audits retain the differentiable append path. No scoring precision,
Temporal window, Pool7, budget, retrieval top-k or eviction interval is reduced.
Finished-request buffers are released; shared history storage may remain until
the surviving request's historical window expires.
Greedy selection without logits processors runs as a row-wise batch argmax, and
EOS flags transfer to the host together rather than synchronizing once per request.

Batch GEMM/FlashAttention can have different floating-point reduction orders from
separate single requests. Async recall also changes CPU response arrival times.
Therefore batch>1 is **not promised bitwise-equivalent** to separate runs, and no
quality equivalence or speedup is assumed without GPU tests. Deterministic recall
waits are a diagnostic option, not the production throughput setting.

`BatchGenerationResult.tokens_per_second` reports total generated tokens divided
by batch-job wall time, including prefills/cleanup. Each inner result's
`elapsed_seconds` is the whole static-chunk wall time, not its individual TTFT.
Increasing aggregate throughput does not imply lower per-request latency.

### Conservative GPU admission limit

For batch>1 the CLI tokenizes real prompts and estimates a limit **before loading
weights**; the engine checks again using actual GPU residency before creating
request buffers. Oversized requests warn and raise `MemoryError`, without running
prefill, silently reducing batch, or falling back to serial inference.
The Python API checks at engine creation after model loading.

The estimate counts BF16 model parameters, actual maximum prompt length, bounded
generation K/V, query history, append/backward copies, activations, metadata,
scoring workspace and recall GPU buffers. CPU archive bounds are reported
separately. By default it uses 85% of currently available memory and additionally
reserves 1024 MiB. `--memory-safety-fraction` and `--memory-reserve-mib` configure
this policy. Both a larger KV budget and longer prompts reduce the admitted batch.

This is a conservative allocation estimate, **not a proven hardware maximum or an
OOM guarantee**: other processes, allocator fragmentation, libraries and changed
model implementations can consume memory afterward. Use an otherwise idle GPU
and validate measured peaks before increasing batch size. The default batch=1
path deliberately keeps its previous admission and execution behavior.

The implementation follows the independent-sequence/GQA contract of
[FlashAttention 2.8.3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py),
and borrows logical/physical request isolation from
[vLLM's cache design](https://docs.vllm.ai/en/latest/design/paged_attention/).
It does not implement vLLM PagedAttention, continuous batching or its scheduler.

## Configuration

`configs/default.json` contains all public options. CLI values override JSON.
`python run.py --print-config` prints the resolved configuration without loading
a model. The model path is deliberately unset.

| Option | Default | Meaning |
|---|---:|---|
| `batch_size` | 1 | Maximum concurrent requests; 1 uses the original engine |
| `memory_safety_fraction` | 0.85 | Available GPU memory fraction for batch admission |
| `memory_reserve_mib` | 1024 | Additional GPU admission reserve |
| `budget` | 2000 | Generation-token-equivalent GPU capacity |
| `audit_period` | 250 | Maximum steps between complete re-ranking audits |
| `temporal_window` | 250 | Retained historical queries for temporal scoring |
| `evict_interval` | 8 | Batch size of physical lazy eviction |
| `protect_recent` | 32 | Recent generation positions excluded from selection |
| `pool_kernel` | 7 | Per-head score pooling width |
| `query_chunk_size` | 250 | Temporary scoring workspace control |
| `recall.enabled` | true | CPU archive, search and actual attention-visible reinsertion |
| `recall.search_interval` | 32 | Nonblocking CPU search cadence |
| `recall.topk_per_head` | 4 | Retrieved candidates per full-attention KV head |
| `recall.max_admit_per_head` | 4 | Maximum admitted candidates per head at an audit |
| `recall.archive_capacity_per_head` | 32768 | Bounded CPU archive capacity |
| `recall.archive_buffers` | 16 | Bounded DMA staging ring |
| `recall.cpu_threads` | 2 | FAISS worker thread limit |
| `recall.wait_at_audit` | false | Diagnostic wait; leave off for production timing |

Prompt KV is protected and **not charged to the generation budget**. Physical
generation capacity is `budget × managed_layers × KV_heads`; an eight-step
staging reserve is inside that cap. Query history, metadata, CPU archives,
DMA buffers and temporary scoring tensors are additional memory. For Qwen3.5,
managed layers exclude DeltaNet, whose recurrent/convolution states are retained
and reported separately.

## Method and implementation

An audit uses one shared margin VJP to map local leave-one-out attention changes
to a common decision-risk scale. Temporal-v1 aggregates historical-query risks;
the unpooled scores determine global head allocations, and Pool7 refines
positions within those allocations. Scores and the queue are reused between
audits. Ordinary eviction releases only the capacity needed for new generation
states, rather than clearing the entire queue up front.

Recall archives evicted K/V on the CPU, searches per-head keys with FAISS inner
product search, and transfers a bounded candidate set back asynchronously.
At an audit, valid candidates are merged into the **actual** attention cache,
with their original RoPE positions. Admission reserves room inside the same
budget, invalidates the old queue, and rebuilds it after the recalled states
participate in the forward pass. This is not an IO-only recall simulation.

The CUDA path includes fused append, K/V/position compaction, causal temporal
score reductions, Pool7, archive packing and a two-source chronological recall
merge. Value projections are reused rather than constructing a full
query-by-token-by-value tensor. CPU array-based queue bookkeeping avoids
per-slot GPU synchronization. No fast-math compiler option is used.
These are implementation optimizations, not a guarantee of speedup on every GPU.

### Qwen3.5 details

The adapter retains native Q/K normalization, partial/interleaved RoPE, output
gating, and GatedDeltaNet transitions. The gate is also included in the local
margin-gradient mapping. Only the full-attention layer view reaches the shared
scorer, queue and recall transport.

Some fused recurrent DeltaNet kernels do not implement backward. During an
audit only, the adapter uses the pinned Transformers differentiable Torch
recurrent rule and convolution update, with private recurrent state; after VJP,
it commits that single transition once. Normal decoding retains the native
fast kernels. This is not a second generated step or detached downstream
gradient. Different kernel reduction orders may cause floating-point differences;
target-GPU validation is required. See the
[pinned Transformers source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [FLA recurrent implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py).

## Layout

```text
run.py                    Reader-facing CLI
configs/default.json      Public runtime configuration
src/lazy_recall/
  models/                 Llama/Mistral dispatch and Qwen3.5 hybrid adapter
  generation.py           Request lifecycle and token generation
  batching.py             Static batch scheduler and independent request state
  batch_attention.py      Request/head varlen routing and fused packed append
  batch_audit.py          Independent-margin shared VJP
  batch_memory.py         Conservative GPU admission checks
  scoring.py              Temporal decision-risk and allocation-preserving Pool7
  controller.py           Budget, audit and eviction/admission coordination
  cache.py                Ragged physical KV and original-position metadata
  recall/                 Bounded CPU index, DMA transport and admission planning
csrc/                     CUDA scoring/cache/recall operations
environment.txt           Unified pip-compatible runtime dependencies
```

Private development diagnostics are not required for generation or included in
the distribution. No pretrained weights, datasets, credentials, experiment
results or machine-specific paths are required in the public source.
