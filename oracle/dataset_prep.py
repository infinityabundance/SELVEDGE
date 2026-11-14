#!/usr/bin/env python3
"""Build the pinned, hash-split, near-duplicate-checked experiment inputs.

This script is data preparation, not measurement. It:

  1. verifies the dataset cache against `data/datasets.lock.json`;
  2. parses wikitext-2 validation into articles (the document unit);
  3. ranks articles by the SHA-256 of their normalized text and assigns the 16
     lowest-hash articles to calibration and the next 40 that pass the
     near-duplicate filter to held-out;
  4. runs a word 5-shingle Jaccard check between calibration and every
     held-out candidate, discarding candidates at or above 0.2;
  5. tokenizes with the pinned Qwen3 tokenizer into non-overlapping 512-token
     windows (calibration <= 4 per article, held-out <= 2);
  6. selects the 40 code prompts and 40 math questions by the pre-registered
     deterministic rules;
  7. writes the derived token-id file under the gitignored
     `results/artifacts/split/` and a tracked receipt under `results/splits/`.

The receipt carries every document's SHA-256, token count and window count, the
derived file's SHA-256, the dataset-lock hash, the tokenizer identity and the
near-duplicate statistics. It does not contain corpus text.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

SCHEMA_RECEIPT = "selvedge/split/v1"
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
HEADER_RE = re.compile(r"^ = [^=].* = \s*$")
SHINGLE = 5
NEAR_DUP_THRESHOLD = 0.2
WINDOW = 512
MIN_TRAILING = 64
CALIBRATION_ARTICLES = 16
HELDOUT_ARTICLES = 40
CALIBRATION_WINDOWS = 4
HELDOUT_WINDOWS = 2
CODE_ITEMS = 40
MATH_ITEMS = 40
SEED = 20261114


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_cache(lock: dict) -> None:
    for item in lock["datasets"]:
        path = os.path.join(lock["cache_relative_root"], item["cache_relative_path"])
        if not os.path.exists(path):
            raise SystemExit(f"dataset cache file is missing: {path}")
        if os.path.getsize(path) != item["bytes"]:
            raise SystemExit(f"dataset cache file size changed: {path}")
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"dataset cache file hash changed: {path}")


def normalize(text: str) -> str:
    return " ".join(NORMALIZE_RE.sub(" ", text.lower()).split())


def shingles(text: str) -> set[str]:
    words = normalize(text).split()
    if len(words) < SHINGLE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def parse_articles(rows: list[str]) -> list[dict]:
    articles: list[dict] = []
    current = None
    for line in rows:
        if HEADER_RE.match(line.rstrip("\n")):
            current = {"title": line.strip(), "lines": []}
            articles.append(current)
        elif current is not None:
            current["lines"].append(line)
    documents = []
    for index, article in enumerate(articles):
        text = article["title"] + "\n" + "".join(article["lines"])
        documents.append(
            {
                "doc_id": f"wikitext-article-{index:03d}",
                "title": article["title"][:200],
                "text": text,
                "sha256": sha256_text(normalize(text)),
            }
        )
    return documents


def windows_for(token_ids: list[int], limit: int) -> list[list[int]]:
    windows = []
    for start in range(0, len(token_ids), WINDOW):
        chunk = token_ids[start : start + WINDOW]
        if len(chunk) == WINDOW or (start == 0 and chunk) or len(chunk) >= MIN_TRAILING:
            if len(chunk) >= 2:
                windows.append(chunk)
        if len(windows) >= limit:
            break
    return windows


def tokenize(tokenizer, text: str) -> list[int]:
    return list(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-lock", default="data/datasets.lock.json")
    parser.add_argument("--model-lock", default="models/MODELS.lock.json")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out", default="results/artifacts/split/m1-split.json")
    parser.add_argument("--receipt-dir", default="results/splits")
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    lock = load_json(args.dataset_lock)
    if lock.get("schema") != "selvedge/datasets-lock/v1":
        raise SystemExit(f"{args.dataset_lock}: unexpected schema")
    verify_cache(lock)
    datasets = {item["id"]: item for item in lock["datasets"]}

    model_lock = load_json(args.model_lock)
    model_entry = next(
        (item for item in model_lock["models"] if item["id"] == args.model_id), None
    )
    if model_entry is None:
        raise SystemExit(f"{args.model_lock}: no entry for {args.model_id}")
    tokenizer_dir = os.path.join(
        "/hf-cache", model_entry["cache_relative_dir"]
    ) if os.path.isdir("/hf-cache") else os.path.join(
        os.path.expanduser("~/.cache/selvedge/hf"), model_entry["cache_relative_dir"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=False, local_files_only=True
    )

    # ---- wikitext -------------------------------------------------------
    wikitext = datasets["wikitext-2-raw-v1-validation"]
    wikitext_path = os.path.join(lock["cache_relative_root"], wikitext["cache_relative_path"])
    rows = [row["text"] for row in pq.read_table(wikitext_path).to_pylist()]
    articles = parse_articles(rows)
    ranked = sorted(articles, key=lambda item: item["sha256"])
    if len(ranked) != len({item["sha256"] for item in ranked}):
        raise SystemExit("duplicate article hash in wikitext: the split would be ambiguous")

    calibration_docs = ranked[:CALIBRATION_ARTICLES]
    candidates = ranked[CALIBRATION_ARTICLES:]
    calibration_shingles = [(doc["doc_id"], shingles(doc["text"])) for doc in calibration_docs]

    pair_count = 0
    near_pairs = {"ge_0.5": 0, "ge_0.2_lt_0.5": 0}
    max_jaccard = 0.0
    candidate_scores = []
    for doc in candidates:
        doc_shingles = shingles(doc["text"])
        worst = 0.0
        for _cal_id, cal_shingles_ in calibration_shingles:
            pair_count += 1
            score = jaccard(doc_shingles, cal_shingles_)
            worst = max(worst, score)
        max_jaccard = max(max_jaccard, worst)
        candidate_scores.append((doc, worst))
        if worst >= 0.5:
            near_pairs["ge_0.5"] += 1
        elif worst >= NEAR_DUP_THRESHOLD:
            near_pairs["ge_0.2_lt_0.5"] += 1

    heldout_docs = [
        doc for doc, worst in candidate_scores if worst < NEAR_DUP_THRESHOLD
    ][:HELDOUT_ARTICLES]
    worst_by_id = {doc["doc_id"]: worst for doc, worst in candidate_scores}
    heldout_worst = [worst_by_id[doc["doc_id"]] for doc in heldout_docs]
    discarded = [
        {"doc_id": doc["doc_id"], "sha256": doc["sha256"], "max_jaccard": round(worst, 6)}
        for doc, worst in candidate_scores
        if worst >= NEAR_DUP_THRESHOLD
    ]
    if len(heldout_docs) < 30:
        raise SystemExit(
            f"only {len(heldout_docs)} held-out articles survived the near-duplicate "
            "filter; the pre-registered minimum is 30"
        )

    def materialize(docs: list[dict], assignment: str, limit: int) -> list[dict]:
        out = []
        for doc in docs:
            ids = tokenize(tokenizer, doc["text"])
            windows = windows_for(ids, limit)
            out.append(
                {
                    "dataset": "wikitext-2-raw-v1-validation",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "sha256": doc["sha256"],
                    "assignment": assignment,
                    "tokens": len(ids),
                    "window_count": len(windows),
                    "windows": windows,
                }
            )
        return out

    calibration = materialize(calibration_docs, "calibration", CALIBRATION_WINDOWS)
    heldout = materialize(heldout_docs, "heldout", HELDOUT_WINDOWS)

    # ---- code -----------------------------------------------------------
    humaneval = datasets["humaneval-test"]
    humaneval_path = os.path.join(lock["cache_relative_root"], humaneval["cache_relative_path"])
    code_rows = pq.read_table(humaneval_path).to_pylist()
    code_total = len(code_rows)

    def task_number(row: dict) -> int:
        return int(str(row["task_id"]).split("/")[-1])

    code_rows = sorted(code_rows, key=task_number)[:CODE_ITEMS]
    code = []
    for row in code_rows:
        ids = tokenize(tokenizer, row["prompt"])[:WINDOW]
        code.append(
            {
                "dataset": "humaneval-test",
                "doc_id": str(row["task_id"]),
                "sha256": sha256_text(normalize(row["prompt"])),
                "assignment": "heldout",
                "tokens": len(ids),
                "window_count": 1,
                "windows": [ids],
            }
        )

    # ---- math -----------------------------------------------------------
    gsm8k = datasets["gsm8k-test"]
    gsm8k_path = os.path.join(lock["cache_relative_root"], gsm8k["cache_relative_path"])
    math_rows = pq.read_table(gsm8k_path).to_pylist()
    for row in math_rows:
        row["_hash"] = sha256_text(normalize(row["question"]))
    math_rows = sorted(math_rows, key=lambda row: row["_hash"])[:MATH_ITEMS]
    math = []
    for row in math_rows:
        ids = tokenize(tokenizer, row["question"])[:WINDOW]
        math.append(
            {
                "dataset": "gsm8k-test",
                "doc_id": row["_hash"][:16],
                "sha256": row["_hash"],
                "assignment": "heldout",
                "tokens": len(ids),
                "window_count": 1,
                "windows": [ids],
            }
        )

    derived = {
        "schema": "selvedge/split-inputs/v1",
        "created_utc": utc_now(),
        "seed": SEED,
        "dataset_lock_sha256": sha256_file(args.dataset_lock),
        "tokenizer": {
            "model_id": args.model_id,
            "revision": model_entry["revision"],
            "tokenizer_json_sha256": next(
                (
                    item["sha256"]
                    for item in model_entry["tokenizer"]["files"]
                    if item["path"] == "tokenizer.json"
                ),
                None,
            ),
        },
        "windows": {
            "calibration": calibration,
            "heldout_general": heldout,
            "heldout_code": code,
            "heldout_math": math,
        },
    }
    write_json_atomic(args.out, derived)

    def summarise(docs: list[dict]) -> dict:
        return {
            "documents": len(docs),
            "windows": sum(doc["window_count"] for doc in docs),
            "tokens": sum(
                sum(len(window) for window in doc["windows"]) for doc in docs
            ),
        }

    receipt = {
        "schema": SCHEMA_RECEIPT,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "seed": SEED,
        "data": {
            "dataset_lock": args.dataset_lock,
            "dataset_lock_sha256": sha256_file(args.dataset_lock),
            "datasets": [
                {
                    "id": item["id"],
                    "repo_id": item["repo_id"],
                    "revision": item["revision"],
                    "file": item["file"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "license": item["license"],
                }
                for item in lock["datasets"]
            ],
        },
        "split_rule": {
            "unit": "article (level-1 wikitext header plus following lines until the next header)",
            "rank": "SHA-256 of normalized text, ascending hex",
            "calibration_documents": CALIBRATION_ARTICLES,
            "heldout_candidates": len(candidates),
            "heldout_documents": len(heldout_docs),
            "near_duplicate": {
                "shingle_words": SHINGLE,
                "threshold": NEAR_DUP_THRESHOLD,
                "calibration_heldout_pairs_scored": pair_count,
                "max_jaccard": round(max_jaccard, 6),
                "pairs_ge_0.5": near_pairs["ge_0.5"],
                "pairs_ge_0.2_lt_0.5": near_pairs["ge_0.2_lt_0.5"],
                "discarded": discarded,
                "max_jaccard_after_discard": round(max(heldout_worst, default=0.0), 6),
            },
            "code_rule": f"{code_total} prompts, first {CODE_ITEMS} by numeric task_id",
            "math_rule": f"40 lowest SHA-256 of normalized question text",
            "window_tokens": WINDOW,
            "min_trailing_tokens": MIN_TRAILING,
            "calibration_windows_per_document": CALIBRATION_WINDOWS,
            "heldout_windows_per_document": HELDOUT_WINDOWS,
        },
        "counts": {
            "calibration": summarise(calibration),
            "heldout_general": summarise(heldout),
            "heldout_code": summarise(code),
            "heldout_math": summarise(math),
        },
        "documents": [
            {
                "dataset": doc["dataset"],
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "sha256": doc["sha256"],
                "assignment": doc["assignment"],
                "tokens": doc["tokens"],
                "window_count": doc["window_count"],
            }
            for doc in calibration + heldout + code + math
        ],
        "derived": {
            "path": args.out,
            "bytes": os.path.getsize(args.out),
            "sha256": sha256_file(args.out),
        },
        "tokenizer": derived["tokenizer"],
        "notes": [
            "no corpus text is written outside the gitignored derived file",
            "the near-duplicate check compares calibration against every held-out candidate",
            "code and math inputs are held out from every selector and greedy step",
            "pretraining contamination cannot be tested locally; see docs/EXPERIMENT-CONTRACT.md section 8",
        ],
    }
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, f"split-{receipt['stamp']}.json")
    write_json_atomic(receipt_path, receipt)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        for name, summary in receipt["counts"].items():
            print(
                f"split: {name}: {summary['documents']} documents, "
                f"{summary['windows']} windows, {summary['tokens']} tokens"
            )
        print(
            f"split: near-duplicate max jaccard {max_jaccard:.4f}, "
            f"discarded {len(discarded)}, pairs scored {pair_count}"
        )
        print(f"split: derived {args.out} sha256={receipt['derived']['sha256']}")
        print(f"split: receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
