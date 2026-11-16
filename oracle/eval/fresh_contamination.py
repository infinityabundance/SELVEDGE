#!/usr/bin/env python3
"""P8.1 — the contamination receipt for the fresh task set.

The fresh set must be disjoint from every T2/T1 evaluation item. This is checked
two ways, because the two catch different leaks:

* **text, 8-word shingles** over normalised (lower-cased, whitespace-collapsed)
  words against HumanEval's prompts, canonical solutions and tests, every GSM8K
  question and answer, and every TruthfulQA question and choice;
* **tokens, 13-token shingles** between the tokenised fresh prompts and the
  pinned held-out windows the KL metric uses — the same tokenizer, so the
  comparison is in the model's own units.

`--require-disjoint` (the default) exits non-zero on any overlap, so a run that
inherits a contaminated set cannot be registered. The receipt always records the
counts, so the margin is visible even when it passes.

Deterministic and CPU-only.

    scripts/court oracle -- python3 oracle/eval/fresh_contamination.py \
        --out results/p8/contamination-$(date -u +%Y%m%dT%H%M%SZ).json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORACLE, "train"))

import fresh_tasks  # noqa: E402

SCHEMA = "selvedge/p8-contamination/v1"
TEXT_WIDTH = 8
TOKEN_WIDTH = 13


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# evaluation sources


def humaneval_texts(dataset_root: str) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "humaneval-test", "test-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["task_id", "prompt", "canonical_solution",
                                        "test"]).to_pylist()
    return [(row["task_id"], "\n".join([row["prompt"], row["canonical_solution"],
                                        row["test"]])) for row in rows]


def gsm8k_texts(dataset_root: str) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "gsm8k-test", "test-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    return [(f"gsm8k-{index:04d}", row["question"] + "\n" + row["answer"])
            for index, row in enumerate(rows)]


def truthfulqa_texts(dataset_root: str) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "truthfulqa-mc-validation",
                        "validation-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["question", "mc1_targets"]).to_pylist()
    return [(f"truthfulqa-{index:04d}",
             row["question"] + "\n" + "\n".join(row["mc1_targets"]["choices"]))
            for index, row in enumerate(rows)]


def source_hashes(dataset_root: str) -> dict:
    hashes = {}
    for name, relative in (
            ("humaneval", "humaneval-test/test-00000-of-00001.parquet"),
            ("gsm8k", "gsm8k-test/test-00000-of-00001.parquet"),
            ("truthfulqa", "truthfulqa-mc-validation/validation-00000-of-00001.parquet")):
        path = os.path.join(dataset_root, relative)
        hashes[name] = {"path": path, "sha256": sha256_file(path)}
    return hashes


# ---------------------------------------------------------------------------


def overlap_report(item_sets: dict[str, set[str]], source: set[str]) -> dict:
    offenders = {item: len(shingles & source) for item, shingles in item_sets.items()
                 if shingles & source}
    return {"offending_items": len(offenders),
            "offenders": dict(sorted(offenders.items())[:20]),
            "shared_shingles": sum(offenders.values())}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--require-disjoint", action="store_true", default=True)
    parser.add_argument("--allow-overlap", dest="require_disjoint",
                        action="store_false")
    args = parser.parse_args(argv)

    item_words = {item["id"]: fresh_tasks.word_shingles(fresh_tasks.item_text(item),
                                                        TEXT_WIDTH)
                  for item in fresh_tasks.TASKS}

    sources = {
        "humaneval": humaneval_texts(args.dataset_root),
        "gsm8k": gsm8k_texts(args.dataset_root),
        "truthfulqa": truthfulqa_texts(args.dataset_root),
    }
    text_checks = {}
    for name, rows in sources.items():
        shingles: set[str] = set()
        for _identifier, text in rows:
            shingles |= fresh_tasks.word_shingles(text, TEXT_WIDTH)
        report = overlap_report(item_words, shingles)
        report["source_items"] = len(rows)
        report["source_shingles"] = len(shingles)
        text_checks[name] = report

    # token-level check against the pinned held-out windows
    import tasks as eval_tasks
    import harness

    split, split_sha = eval_tasks.load_split(args.split_dir, args.split_receipts)
    windows = eval_tasks.heldout_windows(split)
    heldout_tokens = 0
    heldout_shingles: set[str] = set()
    for domain, items in windows.items():
        for item in items:
            heldout_tokens += len(item["tokens"])
            heldout_shingles |= fresh_tasks.token_shingles(item["tokens"], TOKEN_WIDTH)

    tokenizer = harness.load_tokenizer(args.hf_root)
    token_sets = {}
    item_tokens = {}
    for item in fresh_tasks.TASKS:
        ids = tokenizer(fresh_tasks.item_text(item), add_special_tokens=False).input_ids
        item_tokens[item["id"]] = len(ids)
        token_sets[item["id"]] = fresh_tasks.token_shingles(list(ids), TOKEN_WIDTH)
    token_report = overlap_report(token_sets, heldout_shingles)
    token_report["source_windows"] = sum(len(items) for items in windows.values())
    token_report["source_tokens"] = heldout_tokens
    token_report["source_shingles"] = len(heldout_shingles)

    total_offenders = (sum(check["offending_items"] for check in text_checks.values())
                       + token_report["offending_items"])
    verdict = "disjoint" if total_offenders == 0 else "overlap"

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "tasks": fresh_tasks.manifest(),
        "method": {
            "text": f"{TEXT_WIDTH}-word shingles over lower-cased, whitespace-collapsed text",
            "tokens": f"{TOKEN_WIDTH}-token shingles over the registered Qwen3-0.6B "
                      f"tokenizer",
            "comparison": "set intersection per item against the union of the source",
        },
        "sources": {
            "dataset_files": source_hashes(args.dataset_root),
            "split_sha256": split_sha,
            "split_seed": split["seed"],
        },
        "checks": {"text": text_checks, "tokens": token_report},
        "item_tokens": item_tokens,
        "verdict": verdict,
        "command": " ".join(sys.argv),
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({"verdict": verdict, "out": args.out,
                      "offending_items": total_offenders,
                      "text": {name: check["offending_items"]
                               for name, check in text_checks.items()},
                      "tokens": token_report["offending_items"]}, indent=2,
                     sort_keys=True))
    if args.require_disjoint and verdict != "disjoint":
        print("fresh_contamination: the fresh set overlaps an evaluation source; "
              "refusing.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
