#!/usr/bin/env python3
"""Contamination receipt for the fresh capability suite (P12.9).

Every item in `fixtures/capability/` is hashed and shingled and compared against
every registered evaluation and training source, in two units:

* **8-word shingles** (lower-cased, whitespace-collapsed) against HumanEval's
  prompts, canonical solutions and tests, every GSM8K question and answer, every
  TruthfulQA question and choice, a deterministic bounded sample of the T1/T2
  training corpus (fineweb-edu, codeparrot, hendrycks-math), and the T2
  evaluation prompts;
* **13-token shingles** with the registered Qwen3-0.6B tokenizer against the
  pinned held-out windows the KL metric uses.

The corpus sample is bounded and its coverage is recorded, so the receipt states
exactly how much of the 754 MB training cache was compared. A full-corpus pass is
not claimed: the sample rule is deterministic (first N documents in file order),
the counts are in the receipt, and the docs say so.

`--require-disjoint` (default) exits non-zero on any overlap.

    scripts/court oracle -- python3 scripts/capability_contamination.py \
        --out results/capability/contamination-$(date -u +%Y%m%dT%H%M%SZ).json
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ORACLE = os.path.join(REPO, "oracle")
sys.path.insert(0, os.path.join(ORACLE, "eval"))
sys.path.insert(0, ORACLE)

import fresh_contamination as fc  # noqa: E402
from capability import suite  # noqa: E402

SCHEMA = "selvedge/capability-contamination/v1"
TEXT_WIDTH = 8
TOKEN_WIDTH = 13
DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
CORPUS_ROOT = os.path.join(REPO, "results", "artifacts", "train-corpus")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def word_shingles(text: str, width: int) -> set:
    words = " ".join(text.lower().split()).split()
    return {" ".join(words[index:index + width])
            for index in range(0, max(0, len(words) - width + 1))}


def token_shingles(tokens: list, width: int) -> set:
    return {tuple(tokens[index:index + width])
            for index in range(0, max(0, len(tokens) - width + 1))}


def item_text(task) -> str:
    """Everything an arm could see or the checker could use."""
    parts = [task.metadata.get("prompt_intro", ""),
             task.metadata.get("observed_failure", "")]
    for relative in task.initial_files:
        parts.append(task.read_initial(relative))
    for relative in task.hidden_files:
        parts.append(task.read_hidden(relative))
    for relative in task.reference_files:
        parts.append(task.read_reference(relative))
    return "\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# bounded corpus sample


def _fineweb(path: str, limit: int) -> list:
    import pyarrow.parquet as pq

    documents = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32, columns=["text"]):
        for value in batch.column("text").to_pylist():
            documents.append(value)
            if len(documents) >= limit:
                return documents
    return documents


def _hendrycks(path: str, limit: int) -> list:
    import pyarrow.parquet as pq

    documents = []
    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    field = "problem" if "problem" in columns else columns[0]
    for batch in parquet.iter_batches(batch_size=32, columns=[field]):
        for value in batch.column(field).to_pylist():
            documents.append(value if isinstance(value, str) else str(value))
            if len(documents) >= limit:
                return documents
    return documents


def _codeparrot(path: str, limit: int) -> list:
    documents = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                payload = {"content": line}
            documents.append(str(payload.get("content", "")))
            if len(documents) >= limit:
                break
    return documents


def corpus_sample(limit_per_source: int) -> dict:
    sources = {
        "fineweb-edu-sample-10bt": ("fineweb-edu-sample-10bt", _fineweb),
        "codeparrot-clean-0001": ("codeparrot-clean-0001", _codeparrot),
        "hendrycks-math-train": ("hendrycks-math-train", _hendrycks),
    }
    report = {}
    for name, (directory, loader) in sources.items():
        root = os.path.join(CORPUS_ROOT, directory)
        if not os.path.isdir(root):
            report[name] = {"available": False,
                            "reason": f"{os.path.relpath(root, REPO)} is absent"}
            continue
        files = sorted(
            os.path.join(base, filename)
            for base, _dirs, names in os.walk(root) for filename in names
            if filename.endswith((".parquet", ".json.gz")))
        documents: list = []
        per_file = max(1, limit_per_source // max(1, len(files)))
        for path in files:
            if len(documents) >= limit_per_source:
                break
            remaining = limit_per_source - len(documents)
            documents.extend(loader(path, min(per_file, remaining)))
        report[name] = {
            "available": True,
            "files_total": len(files),
            "files_sampled": len(files),
            "documents_sampled": len(documents),
            "characters_sampled": sum(len(document) for document in documents),
            "rule": "first documents in sorted file order, capped per source",
        }
        report[name]["shingles"] = set()
        for document in documents:
            report[name]["shingles"] |= word_shingles(document, TEXT_WIDTH)
        report[name]["shingles"] = report[name]["shingles"]
    return report


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--corpus-docs", type=int, default=1500)
    parser.add_argument("--allow-missing-corpus", action="store_true")
    parser.add_argument("--require-disjoint", action="store_true", default=True)
    parser.add_argument("--allow-overlap", dest="require_disjoint",
                        action="store_false")
    args = parser.parse_args(argv)

    manifest_status, manifest_message = suite.check_manifest(args.suite)
    if not manifest_status:
        print(f"capability_contamination: {manifest_message}", file=sys.stderr)
        return 4
    manifest = suite.build_manifest(args.suite)
    collections = suite.load_suite(args.suite)
    tasks = collections["suite"] + collections["trio"]

    item_words = {task.id: word_shingles(item_text(task), TEXT_WIDTH)
                  for task in tasks}
    prompt_words = {task.id: word_shingles(
        task.metadata.get("prompt_intro", ""), TEXT_WIDTH) for task in tasks}

    sources = {
        "humaneval": fc.humaneval_texts(args.dataset_root),
        "gsm8k": fc.gsm8k_texts(args.dataset_root),
        "truthfulqa": fc.truthfulqa_texts(args.dataset_root),
    }
    text_checks = {}
    for name, rows in sources.items():
        shingles: set = set()
        for _identifier, text in rows:
            shingles |= word_shingles(text, TEXT_WIDTH)
        report = fc.overlap_report(item_words, shingles)
        report["source_items"] = len(rows)
        report["source_shingles"] = len(shingles)
        text_checks[name] = report

    # The T2 evaluation prompts are the pinned windows; their text is compared
    # here and their tokens below.
    import harness  # noqa: E402
    import tasks as eval_tasks  # noqa: E402

    split, split_sha = eval_tasks.load_split(args.split_dir, args.split_receipts)
    windows = eval_tasks.heldout_windows(split)
    window_text = "\n".join(
        " ".join(str(token) for token in item["tokens"])
        for items in windows.values() for item in items)
    window_shingles = word_shingles(window_text, TEXT_WIDTH)
    report = fc.overlap_report(prompt_words, window_shingles)
    report["source_windows"] = sum(len(items) for items in windows.values())
    report["source_shingles"] = len(window_shingles)
    report["method"] = "window token ids joined as text; a structural near-miss check"
    text_checks["t2_eval_windows"] = report

    corpus = corpus_sample(args.corpus_docs)
    corpus_checks = {}
    missing = []
    for name, entry in corpus.items():
        if not entry["available"]:
            missing.append(name)
            corpus_checks[name] = entry
            continue
        shingles = entry.pop("shingles")
        report = fc.overlap_report(item_words, shingles)
        report.update({
            "files_total": entry["files_total"],
            "files_sampled": entry["files_sampled"],
            "documents_sampled": entry["documents_sampled"],
            "characters_sampled": entry["characters_sampled"],
            "sample_rule": entry["rule"],
            "source_shingles": len(shingles),
        })
        corpus_checks[name] = report
    if missing and not args.allow_missing_corpus:
        print(f"capability_contamination: corpus sources absent: {missing}", file=sys.stderr)
        return 5

    # token-level comparison with the pinned held-out windows
    heldout_shingles: set = set()
    heldout_tokens = 0
    for items in windows.values():
        for item in items:
            heldout_tokens += len(item["tokens"])
            heldout_shingles |= token_shingles(item["tokens"], TOKEN_WIDTH)
    tokenizer = harness.load_tokenizer(args.hf_root)
    token_sets = {}
    item_tokens = {}
    for task in tasks:
        ids = tokenizer(item_text(task), add_special_tokens=False).input_ids
        item_tokens[task.id] = len(ids)
        token_sets[task.id] = token_shingles(list(ids), TOKEN_WIDTH)
    token_report = fc.overlap_report(token_sets, heldout_shingles)
    token_report["source_windows"] = sum(len(items) for items in windows.values())
    token_report["source_tokens"] = heldout_tokens
    token_report["source_shingles"] = len(heldout_shingles)

    total_offenders = (sum(check["offending_items"] for check in text_checks.values())
                       + sum(check.get("offending_items", 0)
                             for check in corpus_checks.values()
                             if "offending_items" in check)
                       + token_report["offending_items"])
    verdict = "disjoint" if total_offenders == 0 else "overlap"

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "suite": {
            "root": "fixtures/capability",
            "suite_sha256": manifest["suite_sha256"],
            "counts": manifest["counts"],
            "items_checked": len(tasks),
            "item_tree_sha256": {task.id: task.tree_sha256 for task in tasks},
        },
        "method": {
            "text": f"{TEXT_WIDTH}-word shingles over lower-cased, "
                    f"whitespace-collapsed text",
            "tokens": f"{TOKEN_WIDTH}-token shingles over the registered "
                      f"Qwen3-0.6B tokenizer",
            "comparison": "set intersection per item against the union of the source",
            "corpus": "bounded deterministic prefix sample; coverage recorded",
        },
        "sources": {
            "dataset_files": fc.source_hashes(args.dataset_root),
            "split_sha256": split_sha,
            "split_seed": split["seed"],
            "corpus_root": "results/artifacts/train-corpus",
        },
        "checks": {
            "text": text_checks,
            "corpus": corpus_checks,
            "tokens": token_report,
        },
        "item_tokens": item_tokens,
        "derived": {
            "offending_items": total_offenders,
            "corpus_documents_sampled": sum(
                entry.get("documents_sampled", 0) for entry in corpus_checks.values()),
            "corpus_characters_sampled": sum(
                entry.get("characters_sampled", 0) for entry in corpus_checks.values()),
            "humaneval_items": text_checks["humaneval"]["source_items"],
            "gsm8k_items": text_checks["gsm8k"]["source_items"],
            "truthfulqa_items": text_checks["truthfulqa"]["source_items"],
            "t2_windows": text_checks["t2_eval_windows"]["source_windows"],
            "t2_tokens": token_report["source_tokens"],
            "suite_sha256": manifest["suite_sha256"],
        },
        "verdict": verdict,
    }
    fc.write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({
        "verdict": verdict,
        "suite_sha256": manifest["suite_sha256"],
        "out": args.out,
        "offending_items": total_offenders,
        "text": {name: check["offending_items"] for name, check in text_checks.items()},
        "corpus": {name: check.get("offending_items") for name, check in corpus_checks.items()},
        "tokens": token_report["offending_items"],
    }, indent=2, sort_keys=True))
    if args.require_disjoint and verdict != "disjoint":
        print("capability_contamination: the suite overlaps an evaluation source; "
              "refusing.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
