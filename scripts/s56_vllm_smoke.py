#!/usr/bin/env python3
"""S56: throughput smoke for the post-trained 0.8B and 2B chat checkpoints.

This measures **tokens/s only**. It produces no accuracy, no score and no
comparison with the transformer harness. The vLLM image exists because the
anchor/eval stack pins torch 2.15.0+cu132 while vLLM 0.34.0 pins torch 2.14.0
(see `docs/S56-VLLM.md`), so the throughput engine is a separate, labelled
instrument.

It refuses to report a vLLM number when vLLM is absent. `--allow-transformers`
runs the same prompts on the pinned transformers stack and labels the engine
`transformers`; that number is a fallback and is never described as vLLM.

Usage (queued; GPU must be free):
    SELVEDGE_MEM=12g SELVEDGE_CPUS=4 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \\
        scripts/court --gpu vllm -- python3 scripts/s56_vllm_smoke.py

Plan only (CPU, no imports of torch):
    SELVEDGE_MEM=2g scripts/court python -- python3 \\
        scripts/s56_vllm_smoke.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GIB = float(1 << 30)

DEFAULT_MODELS = (
    ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
)
PROMPTS = (
    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total does it take?",
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes "
    "muffins with four. She sells the rest for $2. How much does she make?",
    "What is the sum of the first 20 positive integers?",
    "Solve for x: 3x + 7 = 22.",
    "A train travels 60 miles per hour for 2.5 hours. How far does it go?",
    "Compute 17 * 23 - 19 * 11.",
    "If f(x) = 2x^2 - 3x + 1, what is f(4)?",
    "A rectangle has perimeter 36 and width 7. What is its area?",
)
MAX_NEW_TOKENS = 256


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pip_freeze() -> tuple[str, str]:
    try:
        frozen = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return "", "unavailable"
    return frozen, sha256_text(frozen)


def cgroup_memory() -> dict:
    values = {}
    for name in ("memory.max", "memory.peak", "memory.swap.peak"):
        path = f"/sys/fs/cgroup/{name}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                values[name.replace(".", "_")] = handle.read().strip()
    return values


def build_prompts(tokenizer) -> list[str]:
    rendered = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt +
                     "\n\nSolve the problem step by step. Put the final answer "
                     "in \\boxed{}."}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        rendered.append(text)
    return rendered


def run_vllm(model_dir: str, vram_fraction: float) -> dict:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    prompts = build_prompts(tokenizer)
    llm = LLM(model=model_dir, dtype="bfloat16", trust_remote_code=False,
              gpu_memory_utilization=vram_fraction, max_model_len=2048)
    params = SamplingParams(temperature=0.0, top_p=1.0,
                            max_tokens=MAX_NEW_TOKENS)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    outputs = llm.generate(prompts, params)
    wall = time.time() - started
    tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return {
        "engine": "vllm",
        "prompts": len(prompts),
        "generated_tokens": tokens,
        "wall_seconds": wall,
        "tokens_per_second": tokens / wall if wall > 0 else None,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
        "first_completion": outputs[0].outputs[0].text[:400],
    }


def run_transformers(args, model_dir: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    prompts = build_prompts(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=False)
    model.to("cuda")
    model.eval()
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    prompt_length = int(input_ids.shape[1])
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)
    wall = time.time() - started
    generated = output[:, prompt_length:]
    tokens = int((generated != (tokenizer.pad_token_id or -1)).sum().item())
    return {
        "engine": "transformers",
        "prompts": len(prompts),
        "generated_tokens": tokens,
        "wall_seconds": wall,
        "tokens_per_second": tokens / wall if wall > 0 else None,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
        "first_completion": tokenizer.decode(generated[0],
                                             skip_special_tokens=True)[:400],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", action="append", default=None,
                        help="repeatable; default: the two post-trained dirs")
    parser.add_argument("--vram-limit-gib", type=float, default=12.5)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--allow-transformers", action="store_true",
                        help="run the fallback engine and label it as such")
    parser.add_argument("--out", default="results/stage2/s56-vllm-smoke.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cache = os.environ.get("HF_HOME", "/hf-cache")
    model_dirs = args.model_dir or [
        os.path.join(cache, "models", model_id, rev)
        for model_id, rev in DEFAULT_MODELS]
    if args.dry_run:
        print("s56-vllm-smoke: plan (no model loaded, no CUDA)")
        for path in model_dirs:
            print(f"s56-vllm-smoke: model {path}")
        print(f"s56-vllm-smoke: prompts {len(PROMPTS)} "
              f"sha256 {sha256_text(json.dumps(PROMPTS))[:16]}…")
        print(f"s56-vllm-smoke: greedy, cap {args.max_new_tokens}, "
              "vram limit " f"{args.vram_limit_gib} GiB")
        print("s56-vllm-smoke: requires the vllm court image "
              "(scripts/build-courts.sh vllm) and a free GPU")
        print("s56-vllm-smoke: command: SELVEDGE_MEM=12g SELVEDGE_CPUS=4 "
              "SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g scripts/court --gpu vllm -- "
              "python3 scripts/s56_vllm_smoke.py")
        return 0

    try:
        import vllm  # noqa: F401
        have_vllm = True
    except Exception as error:  # pragma: no cover - environment dependent
        have_vllm = False
        vllm_error = repr(error)
    if not have_vllm and not args.allow_transformers:
        raise SystemExit(
            "s56-vllm-smoke: vLLM is not importable in this image "
            f"({vllm_error}); refusing to report a throughput number. "
            "Build the vllm court image, or pass --allow-transformers for a "
            "clearly labelled fallback.")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("s56-vllm-smoke: CUDA required; run with --gpu vllm")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s56-vllm-smoke: vram limit {args.vram_limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(
        min(0.99, args.vram_limit_gib * GIB / total_bytes))

    frozen, freeze_sha = pip_freeze()
    results = []
    for path in model_dirs:
        if not os.path.isdir(path):
            raise SystemExit(f"s56-vllm-smoke: missing model dir {path}")
        started = time.time()
        if have_vllm:
            measured = run_vllm(path, args.gpu_memory_fraction)
        else:
            measured = run_transformers(args, path)
        measured.update({
            "model_dir": path,
            "model_id": os.path.basename(os.path.dirname(path)),
            "revision": os.path.basename(path),
            "wall_seconds_total": round(time.time() - started, 1),
        })
        results.append(measured)
        print(f"s56-vllm-smoke: {measured['engine']} "
              f"{measured['model_id']}@{measured['revision'][:12]} "
              f"{measured['tokens_per_second']:.1f} tok/s "
              f"({measured['generated_tokens']} tokens, "
              f"{measured['wall_seconds']:.2f} s)", flush=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema": "selvedge/s56-vllm-smoke/v1",
        "kind": "throughput-smoke",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "engine": {
            "name": "vllm" if have_vllm else "transformers",
            "vllm_version": getattr(sys.modules.get("vllm"), "__version__", None)
            if have_vllm else None,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "image_ref": os.environ.get("SELVEDGE_IMAGE_REF", "unknown"),
            "pip_freeze_sha256": freeze_sha,
            "pip_freeze": frozen.splitlines(),
        },
        "prompts": {"count": len(PROMPTS),
                    "sha256": sha256_text(json.dumps(PROMPTS)),
                    "template": "the stage-1 user instruction, thinking disabled"},
        "protocol": {"decoding": "greedy", "temperature": 0.0, "top_p": 1.0,
                     "max_new_tokens": args.max_new_tokens,
                     "vram_limit_gib": args.vram_limit_gib,
                     "gpu_memory_fraction": args.gpu_memory_fraction},
        "models": results,
        "cgroup_memory": cgroup_memory(),
        "note": ("throughput smoke only. It is not an accuracy score, not a "
                 "comparison with the transformer eval, and it does not change "
                 "any scoring instrument."),
        "command": sys.argv,
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s56-vllm-smoke: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
