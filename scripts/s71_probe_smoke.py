#!/usr/bin/env python3
"""S71 smoke: measure the calibration scoring workload's peak memory and rate.

The S69 calibration scorer runs 64 greedy batches of 32 sequences with a
2048-token cap per checkpoint. This script runs a small sample of exactly that
workload and reports the container's cgroup peak and the GPU peak, so the real
runs can be sized from a measured peak plus headroom as AGENTS.md requires.

It also runs one batch with ``min_new_tokens`` forced to the cap, because a
batch of sequences that all loop to 2048 holds the largest KV cache the real
runs can produce; a smoke that only measured naturally terminating sequences
would understate the limit the preference checkpoints need.

Usage
-----
    SELVEDGE_MEM=14g scripts/court --gpu eval -- python3 \
      scripts/s71_probe_smoke.py \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --checkpoint results/stage2/s67-soup/interp-0.375-arm.pt \
        --batch-size 32 --batches 2 --out results/stage2/s71-overshoot/smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from s54_post_trained_eval import (  # noqa: E402
    cgroup_memory,
    load_tokenizer,
    render_prompt,
    revision,
    sha256_file,
    stamp,
    utcnow,
    write_json,
)


def load_items(limit: int) -> list[dict]:
    registration = json.load(open(
        os.path.join(ROOT, "docs/s66-calibration-splits-v1.json"),
        encoding="utf-8"))
    path = os.path.join(ROOT, registration["splits"]["math_cal"]["path"])
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM
    from s60_interpolate import _load_into, torch_load

    if not torch.cuda.is_available():
        raise SystemExit("s71-smoke: CUDA required; run through scripts/court "
                         "--gpu eval")
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa").to(device)
    model.eval()
    checkpoint = os.path.join(ROOT, args.checkpoint)
    payload = torch_load(checkpoint)
    if payload.get("arm") != "full":
        raise SystemExit(f"s71-smoke: {args.checkpoint} is not a full arm")
    _load_into(model, payload["arm_state"])
    model.eval()
    tokenizer = load_tokenizer(args.model_dir)

    items = load_items(args.batch_size * (args.batches + 1))
    results = []
    for index in range(args.batches):
        chunk = items[index * args.batch_size:(index + 1) * args.batch_size]
        prompts = [render_prompt(tokenizer, item["question"], False)
                   for item in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_width = int(input_ids.shape[1])
        started = time.time()
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        wall = time.time() - started
        generated = output[:, prompt_width:]
        pad = tokenizer.pad_token_id
        lengths = []
        for row in generated:
            ids = row.tolist()
            while ids and ids[-1] == pad:
                ids.pop()
            lengths.append(len(ids))
        results.append({
            "kind": "natural",
            "batch": index,
            "wall_seconds": round(wall, 2),
            "generated_tokens": int(sum(lengths)),
            "max_sequence_tokens": int(max(lengths)),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "cgroup": cgroup_memory(),
        })
        print(f"s71-smoke: natural batch {index} wall={wall:.1f}s "
              f"tokens={sum(lengths)} peak_vram_gib="
              f"{torch.cuda.max_memory_allocated() / (1 << 30):.2f}", flush=True)

    # Worst case: every sequence in the batch is held to the cap, which is what
    # the preference checkpoints do (all 2,000 sequences hit 2048).
    chunk = items[args.batches * args.batch_size:
                  (args.batches + 1) * args.batch_size]
    prompts = [render_prompt(tokenizer, item["question"], False)
               for item in chunk]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.shape[1])
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            num_return_sequences=1, min_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    wall = time.time() - started
    results.append({
        "kind": "forced_cap",
        "batch": args.batches,
        "wall_seconds": round(wall, 2),
        "generated_tokens": int(args.batch_size * args.max_new_tokens),
        "max_sequence_tokens": args.max_new_tokens,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1 << 30),
        "cgroup": cgroup_memory(),
    })
    print(f"s71-smoke: forced-cap batch wall={wall:.1f}s "
          f"peak_vram_gib={torch.cuda.max_memory_allocated() / (1 << 30):.2f}",
          flush=True)

    peak_bytes = 0
    for row in results:
        value = row["cgroup"].get("memory_peak")
        if value and value.isdigit():
            peak_bytes = max(peak_bytes, int(value))
    payload = {
        "schema": "selvedge/s71-probe-smoke/v1",
        "kind": "s71-overshoot-smoke",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "instrument": {
            "tasks": "MATH-cal prefix",
            "decoding": "greedy non-thinking",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "attention": "sdpa",
            "test_split_read": False,
        },
        "cgroup_peak_bytes": peak_bytes or None,
        "cgroup_peak_gib": (peak_bytes / (1 << 30)) if peak_bytes else None,
        "results": results,
        "label": "diagnostic",
        "test_read": False,
    }
    write_json(os.path.join(ROOT, args.out), payload)
    print(f"s71-smoke: peak cgroup {payload['cgroup_peak_gib']} GiB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
