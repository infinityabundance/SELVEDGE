#!/usr/bin/env python3
"""S58: combine accepted stage-2a teacher-solution files into one corpus.

`scripts/s55_stage2_data.py` opens its accepted JSONL with mode ``"w"``. A
re-run of the same ``--out-dir``/``--label`` therefore truncates whatever is
there, which is how the 2,997-row partial corpus could have been lost when the
host restarted. S58 finishes the corpus by writing the remaining MATH generation
to a **new** out-dir and merging here, so no existing artifact is rewritten.

Merge rules (deterministic, fail-closed, all counted):

1. sources are read in the order given on the command line;
2. an exact duplicate generated-token solution (SHA-256 over the token ids) is
   dropped and counted;
3. a second accepted solution for an already-seen ``(dataset, id)`` is dropped
   and counted separately (one solution per prompt at k=1), so one prompt is
   never weighted twice;
4. rows are written in a stable order ``(dataset, id, sample_index)``;
5. the receipt carries the per-source and per-dataset counts, the exact
   acceptance arithmetic when the source receipt is supplied, the truncation
   rate, the supervised/prompt token totals, the duplicates removed, and an
   independent 5-word-shingle contamination screen against every cached eval
   set (`frontier.rft.contamination_check`).

Nothing here is typed by hand: every count is recomputed from the files.

Usage
-----
    SELVEDGE_MEM=4g scripts/court python -- python3 scripts/s58_merge_corpus.py \
        --source results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl \
        --source results/artifacts/stage2/s58-stage2a-math-full/stage2a-math-full-accepted.jsonl \
        --label stage2a-accepted-combined \
        --out-dir results/artifacts/stage2/s58-combined \
        --gzip results/stage2/data/stage2a-accepted-combined.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import rft  # noqa: E402

SCHEMA = "selvedge/s58-stage2a-merge/v1"


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


def solution_key(row: dict) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in row["solution_ids"]).encode()
    ).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="accepted jsonl; repeatable, merged in order")
    parser.add_argument("--source-receipt", action="append", default=[],
                        help="generation receipt for a source, repeatable")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gzip", default=None,
                        help="also write a .gz copy (for the tracked corpus)")
    parser.add_argument("--skip-contamination", action="store_true")
    args = parser.parse_args(argv)

    sources = []
    seen_solutions: dict[str, str] = {}
    seen_prompts: dict[tuple[str, str], str] = {}
    merged: list[dict] = []
    duplicates_solution = 0
    duplicates_prompt = 0
    duplicate_examples: list[dict] = []

    for raw in args.source:
        path = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
        rows = 0
        kept = 0
        by_dataset: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                key = solution_key(row)
                prompt_key = (row["dataset"], row["id"])
                if key in seen_solutions:
                    duplicates_solution += 1
                    if len(duplicate_examples) < 10:
                        duplicate_examples.append({
                            "reason": "solution_tokens", "id": row["id"],
                            "dataset": row["dataset"],
                            "same_as": seen_solutions[key]})
                    continue
                if prompt_key in seen_prompts:
                    duplicates_prompt += 1
                    if len(duplicate_examples) < 10:
                        duplicate_examples.append({
                            "reason": "prompt_already_has_a_solution",
                            "id": row["id"], "dataset": row["dataset"],
                            "same_as": seen_prompts[prompt_key]})
                    continue
                seen_solutions[key] = f"{row['dataset']}:{row['id']}"
                seen_prompts[prompt_key] = f"{row['dataset']}:{row['id']}"
                merged.append(row)
                kept += 1
                by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        sources.append({
            "path": os.path.relpath(path, ROOT),
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
            "rows": rows,
            "kept": kept,
            "by_dataset": dict(sorted(by_dataset.items())),
        })

    if not merged:
        raise SystemExit("s58-merge: no rows survived the merge")

    merged.sort(key=lambda row: (row["dataset"], row["id"],
                                 int(row.get("sample_index", 0))))

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}.jsonl")
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # -- accounting -----------------------------------------------------------
    per_dataset: dict[str, dict] = {}
    for row in merged:
        entry = per_dataset.setdefault(row["dataset"], {
            "rows": 0, "prompts": set(), "solution_tokens": 0,
            "prompt_tokens": 0, "lengths": [], "caps": set()})
        entry["rows"] += 1
        entry["prompts"].add(row["id"])
        entry["solution_tokens"] += len(row["solution_ids"])
        entry["prompt_tokens"] += len(row["prompt_ids"])
        entry["lengths"].append(len(row["solution_ids"]))
        entry["caps"].add(row.get("max_new_tokens"))

    def quantile(values: list[int], fraction: float) -> float:
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    receipt_by_dataset: dict[str, dict] = {}
    for raw in args.source_receipt:
        path = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
        payload = json.load(open(path, encoding="utf-8"))
        for name, entry in payload.get("acceptance", {}).get(
                "by_dataset", {}).items():
            receipt_by_dataset.setdefault(name, []).append({
                "receipt": os.path.relpath(path, ROOT),
                **entry,
            })

    datasets = {}
    for name, entry in sorted(per_dataset.items()):
        accepted = {
            "rows": entry["rows"],
            "distinct_prompts": len(entry["prompts"]),
            "prompt_tokens": entry["prompt_tokens"],
            "solution_tokens": entry["solution_tokens"],
            "solution_length": {
                "mean": sum(entry["lengths"]) / len(entry["lengths"]),
                "p50": quantile(entry["lengths"], 0.50),
                "p90": quantile(entry["lengths"], 0.90),
                "p99": quantile(entry["lengths"], 0.99),
                "max": max(entry["lengths"]),
                "cap": max((cap for cap in entry["caps"] if cap is not None),
                           default=None),
            },
        }
        generation = receipt_by_dataset.get(name, [])
        sampled = sum(item.get("sampled") or 0 for item in generation)
        correct = sum(item.get("correct") or 0 for item in generation)
        truncated = sum(item.get("truncated") or 0 for item in generation)
        prompts_sampled = sum(item.get("prompts") or 0 for item in generation)
        accepted["generation_receipts"] = generation
        accepted["sampling"] = {
            "prompts_sampled": prompts_sampled,
            "completions_sampled": sampled,
            "verified_correct": correct,
            "truncated_without_eos": truncated,
            "acceptance_kept_over_sampled": (
                entry["rows"] / sampled if sampled else None),
            "correct_over_sampled": (correct / sampled if sampled else None),
            "truncation_over_sampled": (truncated / sampled if sampled else None),
        }
        datasets[name] = accepted

    contamination = {"status": "skipped"}
    if not args.skip_contamination:
        contamination = rft.contamination_check(
            [{"id": f"{row['dataset']}:{row['id']}", "prompt": row["prompt"],
              "solution": row["solution"]} for row in merged])

    payload = {
        "schema": SCHEMA,
        "kind": "stage2a-merged-teacher-corpus",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "label": args.label,
        "merge_rules": {
            "order": "sources in command-line order",
            "exact_duplicate": "SHA-256 over the generated solution token ids",
            "one_solution_per_prompt": True,
            "sort": "(dataset, id, sample_index)",
        },
        "sources": sources,
        "merged": {
            "path": os.path.relpath(out_path, ROOT),
            "sha256": sha256_file(out_path),
            "bytes": os.path.getsize(out_path),
            "rows": len(merged),
            "distinct_prompts": len({(row["dataset"], row["id"])
                                     for row in merged}),
            "prompt_tokens": sum(len(row["prompt_ids"]) for row in merged),
            "solution_tokens": sum(len(row["solution_ids"]) for row in merged),
            "supervised_positions_with_eos": sum(
                len(row["solution_ids"]) + 1 for row in merged),
        },
        "duplicates_removed": {
            "solution_tokens": duplicates_solution,
            "prompt_already_has_a_solution": duplicates_prompt,
            "examples": duplicate_examples,
        },
        "by_dataset": datasets,
        "contamination": contamination,
        "command": sys.argv,
    }

    gzip_path = None
    if args.gzip:
        gzip_path = args.gzip if os.path.isabs(args.gzip) else os.path.join(
            ROOT, args.gzip)
        os.makedirs(os.path.dirname(gzip_path), exist_ok=True)
        tmp = gzip_path + ".tmp"
        with open(out_path, "rb") as raw_handle, \
                gzip.GzipFile(tmp, "wb", mtime=0) as gz_handle:
            shutil.copyfileobj(raw_handle, gz_handle)
        os.replace(tmp, gzip_path)
        payload["merged"]["gzip_path"] = os.path.relpath(gzip_path, ROOT)
        payload["merged"]["gzip_sha256"] = sha256_file(gzip_path)

    receipt_path = os.path.join(out_dir, f"{args.label}-receipt.json")
    write_json(receipt_path, payload)

    print(f"s58-merge: {len(merged)} rows from {len(sources)} sources; "
          f"dropped {duplicates_solution} exact duplicates and "
          f"{duplicates_prompt} second-solutions-per-prompt")
    for name, entry in datasets.items():
        sampling = entry["sampling"]
        print(f"s58-merge: {name} rows={entry['rows']} "
              f"prompts={entry['distinct_prompts']} "
              f"kept/sampled={sampling['acceptance_kept_over_sampled']} "
              f"truncated/sampled={sampling['truncation_over_sampled']} "
              f"solution_tokens={entry['solution_tokens']}")
    print(f"s58-merge: contamination max_jaccard="
          f"{contamination.get('max_jaccard')} "
          f"fail_closed={contamination.get('fail_closed')}")
    print(f"s58-merge: {os.path.relpath(out_path, ROOT)}")
    print(f"s58-merge: receipt {os.path.relpath(receipt_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
