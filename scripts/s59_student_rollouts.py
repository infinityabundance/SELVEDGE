#!/usr/bin/env python3
"""S59 stage 2b: student-policy rejection-sampling rollouts.

The stage-2b registration (`docs/ws19-post-trained-route-stage2b-preregistration-v1.json`)
samples the **post-trained 0.8B's own** trajectories on the GSM8K-train and
MATH-train prompts, verifies every completion with the exact extractor used by
stage 2a (`frontier.rft.verify`, against the train label), keeps only the
verified-correct ones, and deduplicates exact token sequences within a prompt.
This is rejection-sampling fine-tuning (RFT) data: supervision is the student's
own text, not a teacher's.

Two decode rules are registered and implemented here:

* **cap 2048**, EOS-only stopping, temperature 0.8 / top-p 0.95, K=4 per prompt;
* **loop cut**: a sequence stops at the first token of a repeated block when its
  trailing 64 tokens are four copies of a 16-token block, or its trailing 144
  tokens are three copies of a 48-token block. The cut is per sequence — the
  criterion returns one BoolTensor element per row, so the row that loops
  stops while the other rows in the batch keep decoding — and the retained
  prefix is still verified, so a trajectory that boxes a correct answer before
  looping is kept.

**What the stop does and does not save.** A per-row stop ends that row's token
emission; it does not reduce the batch's forward cost, because a batched
`generate` keeps running every row until the slowest row stops. Any
saved-token or throughput number from this script is therefore a **textual
replay** count (tokens that would have been emitted), not compute saved. The
receipt carries `compute_saved: false` beside every such count.

The prompts are the seeded stratified selection of stage 2a's prompt universe
minus every id in the frozen stage-2a hold-out (`--exclude-ids`), so the
hold-out stays a clean cross-arm CE instrument.

Nothing in the receipt is typed by hand: acceptance, truncation, loop-cut,
duplication and the length profile are all computed from the rows written beside
it.

Usage
-----
    SELVEDGE_MEM=12g scripts/court --gpu oracle -- python3 \
        scripts/s59_student_rollouts.py \
        --policy-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc0636... \
        --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc0636... \
        --select-gsm8k 60 --select-math 60 --k-gsm8k 4 --k-math 4 \
        --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
        --out-dir results/artifacts/stage2/s59-rft --label pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import rft  # noqa: E402
# Shared with stage 2a on purpose: the prompt template, the tokenizer loading and
# the receipt helpers must not drift between the two routes.
from s55_stage2_data import (  # noqa: E402
    _questions,
    load_tokenizer,
    render_prompt,
    revision,
    sha256_file,
    stamp,
    utcnow,
    write_json,
)

SCHEMA = "selvedge/s59-student-policy-rollouts/v1"
GIB = float(1 << 30)


def loop_cut_index(ids: list[int]) -> int | None:
    """The registered loop cut, on a generated-token list.

    Rule A: the trailing 64 tokens are 4 consecutive copies of a 16-token block.
    Rule B: the trailing 144 tokens are 3 consecutive copies of a 48-token block.

    Returns the index of the first token of the first repeated copy (the cut
    position), or None.
    """
    length = len(ids)
    if length >= 144:
        tail = ids[-48:]
        if ids[-144:-96] == tail and ids[-96:-48] == tail and ids[-48:] == tail:
            return length - 144
    if length >= 64:
        tail = ids[-16:]
        if (ids[-64:-48] == tail and ids[-48:-32] == tail
                and ids[-32:-16] == tail and ids[-16:] == tail):
            return length - 64
    return None


class LoopCutStopping:
    """Per-sequence loop cut for `model.generate`.

    The criterion receives the padded `input_ids` of the whole batch, so it can
    track every row independently: a row is finished when its last token is EOS
    or its loop-cut has been recorded, and generation stops when every row is
    finished. Rows that are already finished are padded by the generation loop,
    so their last token is the pad id.
    """

    def __init__(self, prompt_width: int, eos: int, pad: int):
        self.prompt_width = prompt_width
        self.eos = eos
        self.pad = pad
        self.finished: set[int] = set()
        self.looped: set[int] = set()
        self.cut_at: dict[int, int] = {}
        self.eos_seen: set[int] = set()

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ARG002
        """Return one BoolTensor element per row: True stops that row.

        Transformers' `StoppingCriteriaList` ORs the criteria into a per-row
        BoolTensor. Returning a single bool for the whole batch (as the first
        implementation did) only stops generation once every row has finished,
        which is exactly the behaviour the loop cut exists to avoid: a looping
        row keeps consuming decode steps until its siblings terminate.
        """
        import torch

        rows, width = input_ids.shape
        for row in range(rows):
            if row in self.finished:
                continue
            last = int(input_ids[row, width - 1])
            if last == self.pad:
                # The generation loop pads a sequence only after it finished.
                self.finished.add(row)
                continue
            if last == self.eos:
                self.finished.add(row)
                self.eos_seen.add(row)
                continue
            # `loop_cut_index` only inspects the trailing 144 tokens. Slicing
            # that tail keeps the per-step cost bounded by the rule, not by the
            # generated length, and `cut_at` is translated back to an index into
            # the full generated sequence.
            tail_start = max(self.prompt_width, width - 144)
            tail = input_ids[row, tail_start:width].tolist()
            cut = loop_cut_index(tail)
            if cut is not None:
                self.looped.add(row)
                self.cut_at[row] = (tail_start - self.prompt_width) + cut
                self.finished.add(row)
        return torch.tensor([row in self.finished for row in range(rows)],
                            dtype=torch.bool, device=input_ids.device)

    def emitted_eos(self, row: int) -> bool:
        return row in self.eos_seen


def generate(model, tokenizer, prompts: list[str], *, samples: int,
             max_new_tokens: int, temperature: float, top_p: float,
             device: str) -> tuple[list[list[int]], dict]:
    """Sample `samples` completions per prompt with the registered loop cut."""
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_width = int(input_ids.shape[1])
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    criterion = LoopCutStopping(prompt_width, eos, pad)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            num_return_sequences=samples,
            pad_token_id=pad, eos_token_id=eos,
            stopping_criteria=[criterion])
    wall = time.time() - started
    generated = output[:, prompt_width:]
    rows: list[list[int]] = []
    truncated = 0
    loop_cut = 0
    tokens = 0
    details: list[dict] = []
    for index, row in enumerate(generated):
        ids = row.tolist()
        raw_length = len(ids)
        while ids and ids[-1] == pad:
            ids.pop()
        terminated = criterion.emitted_eos(index)
        if index in criterion.looped:
            cut = criterion.cut_at[index]
            ids = ids[:cut]
            loop_cut += 1
        elif not terminated and raw_length >= max_new_tokens:
            truncated += 1
        tokens += len(ids)
        rows.append(ids)
        details.append({"loop_cut": index in criterion.looped,
                        "cut_at": criterion.cut_at.get(index),
                        "truncated": (index not in criterion.looped
                                      and not terminated
                                      and raw_length >= max_new_tokens)})
    return rows, {
        "wall_seconds": wall,
        "generated_tokens": tokens,
        "truncated": truncated,
        "loop_cut": loop_cut,
        "details": details,
        "prompt_tokens": int(attention_mask.sum().item()),
        "tokens_per_second": (tokens / wall) if wall > 0 else None,
        "compute_saved": False,
        "token_savings_basis": (
            "textual replay only: the per-row criterion ends that row's token "
            "emission, but a batched generate still runs every row until the "
            "slowest row stops, so the batch forward cost is not reduced"),
        "stopped_rows": len(criterion.finished),
    }


def cgroup_memory() -> dict:
    values = {}
    for name in ("memory.max", "memory.peak", "memory.swap.peak", "memory.events"):
        path = f"/sys/fs/cgroup/{name}"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                values[name.replace(".", "_")] = handle.read().strip()
    return values


def load_excluded(path: str | None) -> set[str]:
    if not path:
        return set()
    ids = set()
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("id"):
                    ids.add(str(row["id"]))
    return ids


def length_profile(lengths: list[int]) -> dict:
    if not lengths:
        return {"rows": 0}
    ordered = sorted(lengths)

    def quantile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "rows": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def ledger_entry(task: dict) -> dict:
    """One problem's outcome counters, before any sample is inspected."""
    return {
        "id": str(task["id"]),
        "dataset": task["dataset"],
        "attempts": 0,
        "verified_correct": 0,
        "wrong": 0,
        "ambiguous": 0,
        "unparseable": 0,
        "truncated": 0,
        "loop_cut": 0,
        "duplicates": 0,
        "accepted_rows": 0,
        "accepted_solution_tokens": 0,
        "k": int(task["samples"]),
        "max_new_tokens": int(task["max_new_tokens"]),
    }


def finalise_ledger_entry(entry: dict) -> dict:
    """Attach the pass rate and the p(1-p) weight computed from attempts.

    `p` is `verified_correct / attempts`, where `attempts` counts every sampled
    completion including truncations and duplicates. It is never recomputed from
    the deduplicated accepted rows: deduplication removes correct trajectories,
    so a p built from accepted rows would understate the pass rate exactly where
    the p(1-p) weight matters most.
    """
    attempts = int(entry["attempts"])
    correct = int(entry["verified_correct"])
    p = (correct / attempts) if attempts else 0.0
    entry["p"] = p
    entry["weight_p1mp"] = p * (1.0 - p)
    entry["p_basis"] = "verified_correct / attempts (attempts include truncations "
    "and duplicates; accepted rows are never the denominator)"
    return entry


def summarise_ledger(entries: list[dict]) -> dict:
    """Receipt summary of the per-problem ledger."""
    if not entries:
        return {"prompts": 0, "p_basis": "no prompts"}
    attempts = sum(int(entry["attempts"]) for entry in entries)
    correct = sum(int(entry["verified_correct"]) for entry in entries)
    successes = [int(entry["verified_correct"]) for entry in entries]
    p_values = [float(entry["p"]) for entry in entries]
    histogram: dict[str, int] = {}
    for entry in entries:
        numerator = int(entry["verified_correct"])
        denominator = max(1, int(entry["attempts"]))
        key = f"{numerator}/{denominator}"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "prompts": len(entries),
        "attempts": attempts,
        "verified_correct": correct,
        "verified_rate_from_attempts": (correct / attempts) if attempts else 0.0,
        "mean_p": (sum(p_values) / len(p_values)) if p_values else 0.0,
        "sum_weight_p1mp": sum(entry["weight_p1mp"] for entry in entries),
        "zero_p_problems": sum(1 for value in p_values if value == 0.0),
        "one_p_problems": sum(1 for value in p_values if value == 1.0),
        "truncated": sum(int(entry["truncated"]) for entry in entries),
        "failures_wrong": sum(int(entry["wrong"]) for entry in entries),
        "ambiguous": sum(int(entry["ambiguous"]) for entry in entries),
        "unparseable": sum(int(entry["unparseable"]) for entry in entries),
        "duplicates": sum(int(entry["duplicates"]) for entry in entries),
        "loop_cut": sum(int(entry["loop_cut"]) for entry in entries),
        "accepted_rows": sum(int(entry["accepted_rows"]) for entry in entries),
        "p_histogram": dict(sorted(histogram.items())),
        "p_basis": "verified_correct / attempts, attempts = every sampled "
                    "completion before deduplication",
        "weight_basis": "p(1-p)",
        "max_attempts": max((int(entry["attempts"]) for entry in entries),
                            default=0),
        "successes_per_problem": successes,
    }


def read_ledger(path: str) -> list[dict]:
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def write_ledger_entry(handle, entry: dict) -> dict:
    """Finalise and append one problem's ledger row, then flush.

    This is called for every prompt, accepted rows or not: the ledger is the
    only place a problem's failures and truncations survive when the accepted
    file discards their text.
    """
    finalise_ledger_entry(entry)
    handle.write(json.dumps(entry, sort_keys=True) + "\n")
    handle.flush()
    return entry


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--datasets", default="gsm8k-train,math-train")
    parser.add_argument("--k-gsm8k", type=int, default=4)
    parser.add_argument("--k-math", type=int, default=4)
    parser.add_argument("--select-gsm8k", type=int, default=1500)
    parser.add_argument("--select-math", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens-gsm8k", type=int, default=2048)
    parser.add_argument("--max-new-tokens-math", type=int, default=2048)
    parser.add_argument("--batch-prompts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5900)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--exclude-ids", default=None,
                        help="jsonl whose `id` field is removed from the prompt "
                             "pool (the frozen stage-2a hold-out)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    parser.add_argument("--receipt-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    accepted_path = os.path.join(out_dir, f"{args.label}-accepted.jsonl")
    receipt_path = os.path.join(out_dir, f"{args.label}-receipt.json")
    ledger_path = os.path.join(out_dir, f"{args.label}-ledger.jsonl")

    if (not args.receipt_only and os.path.exists(accepted_path)
            and not args.overwrite):
        raise SystemExit(
            f"s59-rollouts: {os.path.relpath(accepted_path, ROOT)} exists; this "
            "writer truncates. Pass --overwrite deliberately or use a new "
            "--out-dir/--label and merge by token dedup")

    excluded = load_excluded(args.exclude_ids)

    import torch

    if not args.receipt_only:
        if not torch.cuda.is_available():
            raise SystemExit("s59-rollouts: CUDA required; run through scripts/court --gpu")
        free_bytes, _total = torch.cuda.mem_get_info()
        if args.vram_limit_gib * GIB > free_bytes:
            raise SystemExit(
                f"s59-rollouts: vram limit {args.vram_limit_gib} GiB exceeds "
                f"free {free_bytes / GIB:.2f} GiB")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda"
        tokenizer = load_tokenizer(args.tokenizer_dir)
        print(f"s59-rollouts: loading policy {args.policy_dir}", flush=True)
        from frontier import arms
        model = arms.load_model(args.policy_dir, dtype=torch.bfloat16,
                                device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        datasets = [name for name in args.datasets.split(",") if name]
        plan = {
            "gsm8k-train": (args.k_gsm8k, args.select_gsm8k,
                            args.max_new_tokens_gsm8k),
            "math-train": (args.k_math, args.select_math,
                           args.max_new_tokens_math),
        }
        tasks: list[dict] = []
        selection_report: dict[str, dict] = {}
        excluded_counts: dict[str, int] = {}
        for name in datasets:
            if name not in plan:
                raise SystemExit(f"s59-rollouts: unknown dataset {name!r}")
            samples, select, max_new = plan[name]
            items = rft.load_items(name, limit=None)
            pool = len(items)
            items = [item for item in items if item["id"] not in excluded]
            excluded_counts[name] = pool - len(items)
            items, report = rft.stratified_select(items, select, args.seed)
            report["pool_before_holdout_exclusion"] = pool
            report["excluded_holdout_prompts"] = excluded_counts[name]
            selection_report[name] = report
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
            raise SystemExit("s59-rollouts: no prompts selected")
        print(f"s59-rollouts: {len(tasks)} prompts, "
              f"{sum(task['samples'] for task in tasks)} rollouts planned",
              flush=True)

        totals = {"prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
                  "duplicates": 0, "truncated": 0, "loop_cut": 0,
                  "generated_tokens": 0, "wall_seconds": 0.0}
        per_dataset: dict[str, dict] = {}
        started_all = time.time()
        handle = open(accepted_path, "w", encoding="utf-8")
        ledger_handle = open(ledger_path, "w", encoding="utf-8")
        ledger_entries: list[dict] = []
        for start in range(0, len(tasks), args.batch_prompts):
            batch = tasks[start:start + args.batch_prompts]
            rendered = [render_prompt(tokenizer, task["question"])
                        for task in batch]
            prompt_ids_batch = [
                tokenizer(text, add_special_tokens=False)["input_ids"]
                for text in rendered]
            samples = batch[0]["samples"]
            if any(task["samples"] != samples for task in batch):
                raise SystemExit("s59-rollouts: one batch must share K")
            max_new = max(task["max_new_tokens"] for task in batch)
            ids_batch, gen_stats = generate(
                model, tokenizer, rendered, samples=samples,
                max_new_tokens=max_new,
                temperature=args.temperature, top_p=args.top_p,
                device=device)
            totals["wall_seconds"] += gen_stats["wall_seconds"]
            totals["generated_tokens"] += gen_stats["generated_tokens"]
            totals["truncated"] += gen_stats["truncated"]
            totals["loop_cut"] += gen_stats["loop_cut"]
            for prompt_index, task in enumerate(batch):
                stats = per_dataset.setdefault(task["dataset"], {
                    "prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
                    "duplicates": 0, "truncated": 0, "loop_cut": 0,
                    "generated_tokens": 0, "wall_seconds": 0.0,
                    "verdicts": {"correct": 0, "wrong": 0, "ambiguous": 0,
                                 "unparseable": 0, "truncated": 0},
                    "solution_lengths": [],
                })
                stats["prompts"] += 1
                totals["prompts"] += 1
                entry = ledger_entry(task)
                seen: set[str] = set()
                for sample_index in range(samples):
                    row_index = prompt_index * samples + sample_index
                    solution_ids = ids_batch[row_index]
                    detail = gen_stats["details"][row_index]
                    entry["attempts"] += 1
                    stats["sampled"] += 1
                    totals["sampled"] += 1
                    stats["truncated"] += 1 if detail["truncated"] else 0
                    stats["loop_cut"] += 1 if detail["loop_cut"] else 0
                    entry["loop_cut"] += 1 if detail["loop_cut"] else 0
                    stats["generated_tokens"] += len(solution_ids)
                    if detail["truncated"]:
                        entry["truncated"] += 1
                        stats["verdicts"]["truncated"] += 1
                        continue
                    text = tokenizer.decode(solution_ids,
                                            skip_special_tokens=True)
                    result = rft.verify(text, task["target"], task["dataset"])
                    stats["verdicts"][result["verdict"]] += 1
                    if result["verdict"] == "correct":
                        entry["verified_correct"] += 1
                    else:
                        entry[result["verdict"]] += 1
                    if result["verdict"] != "correct":
                        continue
                    stats["correct"] += 1
                    totals["correct"] += 1
                    key = hashlib.sha256(
                        ",".join(str(i) for i in solution_ids).encode()
                    ).hexdigest()
                    if key in seen:
                        stats["duplicates"] += 1
                        totals["duplicates"] += 1
                        entry["duplicates"] += 1
                        continue
                    seen.add(key)
                    stats["solution_lengths"].append(len(solution_ids))
                    handle.write(json.dumps({
                        "id": task["id"], "dataset": task["dataset"],
                        "sample_index": sample_index,
                        "difficulty": task["difficulty"],
                        "subject": task["subject"],
                        "prompt": rendered[prompt_index],
                        "solution": text,
                        "target": task["target"],
                        "extracted": result["extracted"],
                        "verification_method": result["method"],
                        "prompt_ids": prompt_ids_batch[prompt_index],
                        "solution_ids": solution_ids,
                        "max_new_tokens": task["max_new_tokens"],
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "loop_cut": bool(detail["loop_cut"]),
                        "loop_cut_at": detail["cut_at"],
                        "policy_revision": os.path.basename(
                            args.policy_dir.rstrip("/")),
                    }, sort_keys=True) + "\n")
                    handle.flush()
                    stats["kept"] += 1
                    totals["kept"] += 1
                    entry["accepted_rows"] += 1
                    entry["accepted_solution_tokens"] += len(solution_ids)
                write_ledger_entry(ledger_handle, entry)
                ledger_entries.append(entry)
            if (start // args.batch_prompts) % 5 == 0:
                print(f"s59-rollouts: prompts {totals['prompts']}/{len(tasks)} "
                      f"sampled={totals['sampled']} kept={totals['kept']} "
                      f"loop={totals['loop_cut']} trunc={totals['truncated']} "
                      f"tok/s={(totals['generated_tokens'] / max(totals['wall_seconds'], 1e-9)):.0f}",
                      flush=True)
        handle.close()
        ledger_handle.close()
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
            "cgroup_memory": cgroup_memory(),
        }
        del model
        torch.cuda.empty_cache()
    else:
        generation = {"rebuilt_from": accepted_path}

    # -- receipt from the jsonl on disk --------------------------------------
    if args.receipt_only:
        ledger_entries = read_ledger(ledger_path)
    rows = []
    with open(accepted_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("s59-rollouts: no accepted rows; nothing to receipt")
    supervised = sum(len(row["solution_ids"]) for row in rows)
    prompt_tokens = sum(len(row["prompt_ids"]) for row in rows)
    by_dataset: dict[str, dict] = {}
    for row in rows:
        entry = by_dataset.setdefault(row["dataset"], {
            "rows": 0, "prompt_tokens": 0, "solution_tokens": 0,
            "prompts": set(), "lengths": [], "loop_cut_rows": 0})
        entry["rows"] += 1
        entry["prompt_tokens"] += len(row["prompt_ids"])
        entry["solution_tokens"] += len(row["solution_ids"])
        entry["prompts"].add(row["id"])
        entry["lengths"].append(len(row["solution_ids"]))
        entry["loop_cut_rows"] += 1 if row.get("loop_cut") else 0
    for entry in by_dataset.values():
        entry["distinct_prompts"] = len(entry["prompts"])
        entry["solution_length"] = length_profile(entry.pop("lengths"))
        del entry["prompts"]

    contamination = {"status": "not_run"}
    if not args.receipt_only:
        contamination = rft.contamination_check(
            [{"id": row["id"], "prompt": row["prompt"],
              "solution": row["solution"]} for row in rows])

    payload = {
        "schema": SCHEMA,
        "kind": "student-policy-rollouts",
        "stage": "2b student-policy rejection-sampling self-distillation",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": "docs/ws19-post-trained-route-stage2b-preregistration-v1.json",
        "policy_dir": args.policy_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "prompt_protocol": ("the policy's own chat template, "
                            "add_generation_prompt=True, enable_thinking=False"),
        "sampling": {
            "temperature": args.temperature, "top_p": args.top_p,
            "do_sample": True, "stop": "EOS only",
            "k_gsm8k": args.k_gsm8k, "k_math": args.k_math,
            "max_new_tokens_gsm8k": args.max_new_tokens_gsm8k,
            "max_new_tokens_math": args.max_new_tokens_math,
            "seed": args.seed, "batch_prompts": args.batch_prompts,
            "shards": args.shards, "shard_index": args.shard_index,
        },
        "loop_cut": {
            "rule_a": "trailing 64 tokens = 4 copies of a 16-token block",
            "rule_b": "trailing 144 tokens = 3 copies of a 48-token block",
            "action": "stop the sequence at the first token of the first "
                      "repeated copy; retained prefix is verified",
            "per_sequence": True,
            "criterion_output": "one BoolTensor element per row; the criterion "
                                "is a transformers StoppingCriteriaList member",
            "compute_saved": False,
            "saved_token_claim": (
                "textual replay only. A batched generate advances every row "
                "until the slowest row stops, so the tokens cut from one row "
                "are not GPU compute saved. Nothing in this receipt claims "
                "throughput or compute savings from the cut."),
        },
        "verification": {
            "extractor": "frontier.rft.verify against the train label",
            "kept": "verdict == correct only; wrong, ambiguous, unparseable and "
                    "truncated completions never become a target",
            "deduplication": "within a prompt, exact generated-token-sequence "
                             "SHA-256; across prompts duplicates are counted "
                             "but not removed",
        },
        "holdout_exclusion": {
            "path": args.exclude_ids,
            "ids": len(excluded),
            "excluded_from_pool": excluded_counts if not args.receipt_only else {},
        },
        "selection": selection_report if not args.receipt_only else {},
        "ledger": {
            "path": os.path.relpath(ledger_path, ROOT),
            "sha256": (sha256_file(ledger_path)
                       if os.path.exists(ledger_path) else None),
            "rows": len(ledger_entries),
            "summary": summarise_ledger(ledger_entries),
            "written_when_accepted_text_discarded": True,
            "why": "the accepted file holds only deduplicated verified-correct "
                   "trajectories; the ledger holds every problem's attempts, "
                   "verified successes, failures, truncations and duplicates, "
                   "so p is computed from attempts and survives the discard",
        },
        "accepted": {
            "path": os.path.relpath(accepted_path, ROOT),
            "sha256": sha256_file(accepted_path),
            "rows": len(rows),
            "supervised_solution_tokens": supervised,
            "prompt_tokens": prompt_tokens,
            "by_dataset": by_dataset,
        },
        "generation": generation,
        "contamination": contamination,
        "command": sys.argv,
    }
    if not args.receipt_only:
        payload["acceptance"] = {"totals": totals, "by_dataset": per_dataset}
        payload["acceptance"]["verified_rate"] = (
            totals["correct"] / totals["sampled"] if totals["sampled"] else 0.0)
        payload["acceptance"]["kept_rate"] = (
            totals["kept"] / totals["sampled"] if totals["sampled"] else 0.0)
        payload["acceptance"]["truncation_rate"] = (
            totals["truncated"] / totals["sampled"] if totals["sampled"] else 0.0)
        payload["acceptance"]["loop_cut_rate"] = (
            totals["loop_cut"] / totals["sampled"] if totals["sampled"] else 0.0)
    write_json(receipt_path, payload)
    print(f"s59-rollouts: {len(rows)} accepted rows, {supervised} supervised "
          f"tokens -> {accepted_path}")
    print(f"s59-rollouts: receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
