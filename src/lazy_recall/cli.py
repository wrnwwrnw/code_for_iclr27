import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import GenerationConfig, RecallConfig
from .storage import atomic_json


def parse_args():
    parser = argparse.ArgumentParser(description="LazyRecall single-request or static-batch generation")
    parser.add_argument("--config", type=Path)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument("--prompts-file", type=Path, help="UTF-8 JSON array of prompt strings")
    parser.add_argument("--output", type=Path, default=Path("outputs/generation.json"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--device")
    for name in ("batch_size", "memory_reserve_mib", "budget", "max_new_tokens", "audit_period", "temporal_window", "evict_interval",
                 "protect_recent", "pool_kernel", "query_chunk_size", "seed"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("memory_safety_fraction", "temperature", "top_p", "repetition_penalty", "denominator_epsilon"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    for name in ("stop_on_eos", "chat_template", "log_audits"):
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recall", action=argparse.BooleanOptionalAction, default=None)
    for name in ("search_interval", "topk_per_head", "max_admit_per_head", "archive_capacity_per_head",
                 "archive_buffers", "cpu_threads"):
        parser.add_argument("--recall-" + name.replace("_", "-"), type=int)
    parser.add_argument("--recall-timeout-seconds", type=float)
    parser.add_argument("--recall-wait-at-audit", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    values = GenerationConfig.from_dict(values).to_dict()
    for name in values:
        if name != "recall" and getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    if args.recall is not None:
        values["recall"]["enabled"] = args.recall
    for name in values["recall"]:
        if name != "enabled" and getattr(args, "recall_" + name, None) is not None:
            values["recall"][name] = getattr(args, "recall_" + name)
    config = GenerationConfig.from_dict(values)
    if not args.print_config and args.prompt is None and args.prompt_file is None and args.prompts_file is None:
        parser.error("Provide --prompt, --prompt-file or --prompts-file")
    if not args.print_config and config.batch_size > 1 and args.prompts_file is None:
        parser.error("batch-size>1 requires --prompts-file; prompts are not silently duplicated")
    if not args.print_config and not config.model:
        parser.error("Provide --model or set model in the JSON configuration")
    return args, config


def main():
    args, config = parse_args()
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return
    from .generation import generate, load_model
    import signal

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    if args.prompts_file is not None:
        prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("--prompts-file must contain a nonempty JSON array of strings")
    else:
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
        prompts = [prompt]
    if config.batch_size > 1:
        from transformers import AutoConfig, AutoTokenizer
        from .batching import tokenize_prompts
        from .batch_memory import preflight_capacity
        model_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        identifiers = tokenize_prompts(tokenizer, prompts, config)
        capacity = preflight_capacity(model_config, config, [len(prompt) for prompt in identifiers])
        print("[capacity] " + json.dumps(capacity.to_dict()), flush=True)
    print("[LazyRecall] configuration: " + json.dumps(config.to_dict()), flush=True)
    model, tokenizer = load_model(config)
    if args.prompts_file is not None:
        from .batching import generate_batch
        result = generate_batch(model, tokenizer, prompts, config)
    else:
        result = generate(model, tokenizer, prompts[0], config)
    atomic_json(args.output, asdict(result))
    if args.prompts_file is not None:
        for index, row in enumerate(result.results):
            print(f"[request {index}] {row.text}")
    else:
        print(result.text)
    print(f"[saved] {args.output.resolve()}")


if __name__ == "__main__":
    main()
