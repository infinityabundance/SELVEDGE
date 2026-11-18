#!/usr/bin/env python3
"""S55 stage 2a: verified teacher solutions from the post-trained 2B.

The S53 stage-2 registration asks for on-policy distillation with per-token
reverse KL, and the S54 amendment for first-error corrective supervision on
failed rollouts. Neither is implemented in `oracle/frontier/train.py`: the
trainer has a completion-masked cross-entropy path (`--objective sft`) and a
token-stream KL path, and no per-token first-error suffix mask. This script
builds the **Stage 2a** corpus instead, registered in
`docs/ws19-post-trained-route-stage2a-amendment-s55-v1.json`:

* prompts are the GSM8K-train and MATH-train items with an extractable
  reference answer (`frontier.rft.load_items`);
* the prompt is the **same** chat-template prompt stage 1 scores with, so the
  teacher's solutions are in the distribution the student is asked for
  (`enable_thinking=False`, "Put the final answer in \\boxed{}");
* the post-trained 2B `15852e8c...` samples `K` completions per prompt at
  temperature 0.8 / top-p 0.95, EOS-only stopping, `--max-new-tokens` cap;
* every completion is verified with `frontier.rft.verify` against the train
  label; only `verdict == correct` solutions are kept, exact-duplicate token
  sequences are removed, and truncated (no-EOS) completions are counted
  separately;
* the accepted rows carry `prompt_ids` / `solution_ids` taken from the
  **generated token ids themselves** (not a decode/re-tokenise round trip), so
  `train.py --sft-examples` supervises exactly the tokens the teacher produced.

Nothing is typed by hand: every acceptance rate, token count and throughput in
the receipt is computed from the generations in the `accepted.jsonl` beside it.

Usage
-----
    SELVEDGE_MEM=.. scripts/court --gpu oracle -- python3 \
        scripts/s55_stage2_data.py \
        --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B/15852e8c... \
        --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc0636... \
        --out-dir results/artifacts/stage2/s55-stage2a --label smoke \
        --datasets gsm8k-train --limit-gsm8k 8 --k-gsm8k 2

The container limit is chosen by the launcher from this workload's own measured
peak; the script only refuses a CUDA free-memory check against
`--vram-limit-gib`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import rft  # noqa: E402

SCHEMA = "selvedge/s55-stage2a-teacher-data/v1"
#: The stage-1 prompt template. Training prompts must match the scored prompt,
#: otherwise the solutions are answers to a different question.
USER_TEMPLATE = ("{question}\n\nSolve the problem step by step. "
                 "Put the final answer in \\boxed{{}}.")
GIB = float(1 << 30)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": USER_TEMPLATE.format(question=question)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


def _questions(dataset: str, items: list[dict]) -> list[str]:
    """The bare problem text each prompt template wraps.

    `frontier.rft.load_items` builds `prompt` by wrapping the raw problem text;
    the wrapper is stripped by length rather than by splitting, because GSM8K
    questions contain newlines.
    """
    if dataset == "gsm8k-train":
        prefix, suffix = "Question: ", "\nAnswer:"
    elif dataset == "math-train":
        prefix, suffix = "Problem:\n", "\n\nSolution:"
    else:
        raise SystemExit(f"s55-data: unknown dataset {dataset!r}")
    questions = []
    for item in items:
        prompt = item["prompt"]
        if not prompt.startswith(prefix) or not prompt.endswith(suffix):
            raise SystemExit(
                f"s55-data: {item['id']} prompt does not carry the expected "
                f"wrapper {prefix!r} .. {suffix!r}")
        questions.append(prompt[len(prefix):len(prompt) - len(suffix)])
    return questions


def generate(model, tokenizer, prompts: list[str], *, samples: int,
             max_new_tokens: int, temperature: float, top_p: float,
             device: str) -> tuple[list[list[int]], dict]:
    """Sample `samples` completions per prompt; return raw generated token ids."""
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.shape[1])
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            num_return_sequences=samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    wall = time.time() - started
    generated = output[:, prompt_width:]
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    rows: list[list[int]] = []
    truncated = 0
    tokens = 0
    for row in generated:
        ids = row.tolist()
        while ids and ids[-1] in (pad, eos):
            ids.pop()
        if len(ids) >= max_new_tokens:
            truncated += 1
        tokens += len(ids)
        rows.append(ids)
    return rows, {
        "wall_seconds": wall,
        "generated_tokens": tokens,
        "truncated": truncated,
        "prompt_tokens": int(attention_mask.sum().item()),
        "tokens_per_second": (tokens / wall) if wall > 0 else None,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True,
                        help="the student's own tokenizer and chat template")
    parser.add_argument("--datasets", default="gsm8k-train,math-train")
    parser.add_argument("--k-gsm8k", type=int, default=2)
    parser.add_argument("--k-math", type=int, default=1)
    parser.add_argument("--limit-gsm8k", type=int, default=None)
    parser.add_argument("--limit-math", type=int, default=None)
    parser.add_argument("--select-gsm8k", type=int, default=None,
                        help="seeded stratified selection count (replaces a "
                             "prefix limit)")
    parser.add_argument("--select-math", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens-gsm8k", type=int, default=768)
    parser.add_argument("--max-new-tokens-math", type=int, default=1024)
    parser.add_argument("--batch-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5500)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1,
                        help="split the prompt order into this many shards")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=14.0)
    parser.add_argument("--receipt-only", action="store_true",
                        help="rebuild the receipt from an existing jsonl")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing accepted jsonl; the "
                             "default fails closed so a partial corpus is never "
                             "clobbered by a re-run")
    args = parser.parse_args(argv)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    accepted_path = os.path.join(out_dir, f"{args.label}-accepted.jsonl")
    receipt_path = os.path.join(out_dir, f"{args.label}-receipt.json")

    # The writer below opens the accepted file with mode "w". A re-run of the
    # same label would therefore destroy a partial corpus (S58 resumed exactly
    # such a partial: 2,997 accepted rows). Fail closed instead, and require an
    # explicit --overwrite or a new --out-dir/--label. Accepted corpora are
    # combined by `scripts/s58_merge_corpus.py`, never by rewriting them.
    if (not args.receipt_only and os.path.exists(accepted_path)
            and not args.overwrite):
        raise SystemExit(
            f"s55-data: {os.path.relpath(accepted_path, ROOT)} already exists; "
            "this script writes with mode 'w' and would clobber it. Pass "
            "--overwrite deliberately, or write to a new --out-dir/--label "
            "and merge with scripts/s58_merge_corpus.py")

    import torch

    if not args.receipt_only:
        if not torch.cuda.is_available():
            raise SystemExit("s55-data: CUDA required; run through scripts/court --gpu")
        free_bytes, _total = torch.cuda.mem_get_info()
        if args.vram_limit_gib * GIB > free_bytes:
            raise SystemExit(
                f"s55-data: vram limit {args.vram_limit_gib} GiB exceeds free "
                f"{free_bytes / GIB:.2f} GiB")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
        tokenizer = load_tokenizer(args.tokenizer_dir)
        vocab_size = len(tokenizer)
        eos_id = int(tokenizer.eos_token_id)
        print(f"s55-data: loading teacher {args.teacher_dir}", flush=True)
        from frontier import arms
        model = arms.load_model(args.teacher_dir, dtype=torch.bfloat16,
                                device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        datasets = [name for name in args.datasets.split(",") if name]
        plan = {
            "gsm8k-train": (args.k_gsm8k, args.limit_gsm8k,
                            args.max_new_tokens_gsm8k, args.select_gsm8k),
            "math-train": (args.k_math, args.limit_math,
                           args.max_new_tokens_math, args.select_math),
        }
        for name in datasets:
            if name not in plan:
                raise SystemExit(f"s55-data: unknown dataset {name!r}")

        # One deterministic prompt plan over every requested dataset, sharded
        # across processes if asked (the shards are disjoint prompt slices).
        tasks: list[dict] = []
        selection_report: dict[str, dict] = {}
        for name in datasets:
            samples, limit, max_new, select = plan[name]
            items = rft.load_items(name, limit=None)
            if select is not None:
                items, report = rft.stratified_select(items, select, args.seed)
                selection_report[name] = report
            elif limit is not None:
                items = items[:limit]
            questions = _questions(name, items)
            for index, (item, question) in enumerate(zip(items, questions)):
                if index % args.shards != args.shard_index:
                    continue
                tasks.append({
                    "id": item["id"], "dataset": item["dataset"],
                    "target": item["target"], "question": question,
                    "samples": samples, "max_new_tokens": max_new,
                    "difficulty": item.get("difficulty"),
                    "subject": item.get("subject")})
        if not tasks:
            raise SystemExit("s55-data: no prompts selected")
        print(f"s55-data: {len(tasks)} prompts, "
              f"{sum(t['samples'] for t in tasks)} completions planned",
              flush=True)

        accepted_handle = open(accepted_path, "w", encoding="utf-8")
        per_dataset: dict[str, dict] = {}
        global_seen: set[str] = set()
        totals = {"prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
                  "duplicates": 0, "truncated": 0, "generated_tokens": 0,
                  "wall_seconds": 0.0}
        started_all = time.time()

        # Group by (dataset, max_new_tokens) so one batch never mixes two caps.
        groups: dict[tuple[str, int], list[dict]] = {}
        for task in tasks:
            groups.setdefault((task["dataset"], task["max_new_tokens"]),
                              []).append(task)

        for (dataset, max_new), group in sorted(groups.items()):
            stats = per_dataset.setdefault(dataset, {
                "prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
                "duplicates": 0, "truncated": 0, "generated_tokens": 0,
                "wall_seconds": 0.0,
                "verdicts": {"correct": 0, "wrong": 0, "ambiguous": 0,
                             "unparseable": 0},
                "max_new_tokens": max_new})
            for start in range(0, len(group), args.batch_prompts):
                batch = group[start:start + args.batch_prompts]
                rendered = [render_prompt(tokenizer, task["question"])
                            for task in batch]
                # The tokenizer is a pure function of the rendered string, so
                # these ids are exactly the ids `generate` consumed (unpadded).
                prompt_ids_batch = [
                    tokenizer(text, add_special_tokens=False)["input_ids"]
                    for text in rendered]
                samples = batch[0]["samples"]
                ids_batch, gen_stats = generate(
                    model, tokenizer, rendered, samples=samples,
                    max_new_tokens=max_new,
                    temperature=args.temperature, top_p=args.top_p,
                    device=device)
                stats["wall_seconds"] += gen_stats["wall_seconds"]
                stats["generated_tokens"] += gen_stats["generated_tokens"]
                stats["truncated"] += gen_stats["truncated"]
                totals["wall_seconds"] += gen_stats["wall_seconds"]
                totals["generated_tokens"] += gen_stats["generated_tokens"]
                totals["truncated"] += gen_stats["truncated"]
                stats["prompts"] += len(batch)
                totals["prompts"] += len(batch)
                # `num_return_sequences` repeats each prompt `samples` times in
                # row, so completion j of prompt p is at p * samples + j.
                for prompt_index, task in enumerate(batch):
                    prompt_ids = prompt_ids_batch[prompt_index]
                    for sample_index in range(samples):
                        solution_ids = ids_batch[prompt_index * samples + sample_index]
                        stats["sampled"] += 1
                        totals["sampled"] += 1
                        if len(solution_ids) >= max_new:
                            # Truncated: no EOS, so it is not a complete answer.
                            stats["verdicts"]["unparseable"] += 1
                            continue
                        text = tokenizer.decode(solution_ids,
                                                skip_special_tokens=True)
                        result = rft.verify(text, task["target"], task["dataset"])
                        stats["verdicts"][result["verdict"]] += 1
                        if result["verdict"] != "correct":
                            continue
                        stats["correct"] += 1
                        totals["correct"] += 1
                        key = hashlib.sha256(
                            ",".join(str(i) for i in solution_ids).encode()
                        ).hexdigest()
                        if key in global_seen:
                            stats["duplicates"] += 1
                            totals["duplicates"] += 1
                            continue
                        global_seen.add(key)
                        row = {
                            "id": task["id"], "dataset": task["dataset"],
                            "sample_index": sample_index,
                            "difficulty": task["difficulty"],
                            "subject": task["subject"],
                            "prompt": rendered[prompt_index],
                            "solution": text,
                            "target": task["target"],
                            "extracted": result["extracted"],
                            "verification_method": result["method"],
                            "prompt_ids": prompt_ids,
                            "solution_ids": solution_ids,
                            "max_new_tokens": max_new,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "teacher_revision": os.path.basename(
                                args.teacher_dir.rstrip("/")),
                        }
                        accepted_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        accepted_handle.flush()
                        stats["kept"] += 1
                        totals["kept"] += 1
                if (start // args.batch_prompts) % 10 == 0:
                    print(f"s55-data: {dataset} prompts "
                          f"{start + len(batch)}/{len(group)} "
                          f"sampled={stats['sampled']} kept={stats['kept']} "
                          f"tok/s="
                          f"{(stats['generated_tokens'] / max(stats['wall_seconds'], 1e-9)):.0f}",
                          flush=True)
        accepted_handle.close()
        elapsed = time.time() - started_all
        peak_alloc = torch.cuda.max_memory_allocated() / GIB
        peak_reserved = torch.cuda.max_memory_reserved() / GIB
        generation = {
            "elapsed_seconds": elapsed,
            "sampling_wall_seconds": totals["wall_seconds"],
            "generated_tokens": totals["generated_tokens"],
            "tokens_per_second": (totals["generated_tokens"]
                                  / max(totals["wall_seconds"], 1e-9)),
            "peak_allocated_gib": peak_alloc,
            "peak_reserved_gib": peak_reserved,
        }
        del model
        torch.cuda.empty_cache()
    else:
        generation = {"rebuilt_from": accepted_path}

    # -- receipt from the jsonl on disk --------------------------------------
    rows = []
    with open(accepted_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("s55-data: no accepted rows; nothing to receipt")
    supervised = sum(len(row["solution_ids"]) for row in rows)
    prompt_tokens = sum(len(row["prompt_ids"]) for row in rows)
    by_dataset: dict[str, dict] = {}
    for row in rows:
        entry = by_dataset.setdefault(row["dataset"], {
            "examples": 0, "prompt_tokens": 0, "solution_tokens": 0,
            "prompts": set()})
        entry["examples"] += 1
        entry["prompt_tokens"] += len(row["prompt_ids"])
        entry["solution_tokens"] += len(row["solution_ids"])
        entry["prompts"].add(row["id"])
    for entry in by_dataset.values():
        entry["distinct_prompts"] = len(entry["prompts"])
        del entry["prompts"]

    contamination = {"status": "not_run"}
    if not args.receipt_only and args.datasets:
        contamination = rft.contamination_check(
            [{"id": row["id"], "prompt": row["prompt"],
              "solution": row["solution"]} for row in rows])

    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "kind": "stage2a-teacher-solutions",
        "stage": "2a teacher-solution SFT (on-policy path not implemented)",
        "teacher_dir": args.teacher_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "prompt_template": USER_TEMPLATE,
        "prompt_protocol": ("the student's own chat template, "
                            "add_generation_prompt=True, "
                            "enable_thinking=False"),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "do_sample": True, "stop": "EOS only",
                     "k_gsm8k": args.k_gsm8k, "k_math": args.k_math,
                     "max_new_tokens_gsm8k": args.max_new_tokens_gsm8k,
                     "max_new_tokens_math": args.max_new_tokens_math,
                     "seed": args.seed, "batch_prompts": args.batch_prompts,
                     "shards": args.shards, "shard_index": args.shard_index},
        "verification": {
            "extractor": "frontier.rft.verify against the train label",
            "kept": "verdict == correct only; ambiguous, unparseable, wrong and "
                    "truncated completions are never a target",
            "deduplication": "exact generated-token-sequence SHA-256, global"},
        "accepted": {
            "path": os.path.relpath(accepted_path, ROOT),
            "sha256": sha256_file(accepted_path),
            "write_mode": ("rebuild" if args.receipt_only else
                           "truncate (mode 'w')"),
            "rows": len(rows),
            "supervised_solution_tokens": supervised,
            "prompt_tokens": prompt_tokens,
            "by_dataset": by_dataset,
        },
        "generation": generation,
        "contamination": contamination,
        "selection": selection_report if not args.receipt_only else {},
        "command": sys.argv,
    }
    if not args.receipt_only:
        payload["acceptance"] = {"totals": totals, "by_dataset": per_dataset}
        payload["acceptance"]["verified_rate"] = (
            totals["correct"] / totals["sampled"] if totals["sampled"] else 0.0)
        payload["acceptance"]["kept_rate"] = (
            totals["kept"] / totals["sampled"] if totals["sampled"] else 0.0)
    write_json(receipt_path, payload)
    print(f"s55-data: {len(rows)} examples, {supervised} supervised tokens "
          f"-> {accepted_path}")
    print(f"s55-data: receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
