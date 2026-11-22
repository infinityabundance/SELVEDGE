#!/usr/bin/env python3
"""S72 Stage 0: freeze the teacher-qualification sample before any scoring.

Draws a seeded 250-problem MATH-train sample and a seeded 250-problem
GSM8K-train sample, then writes the sample file, its sha256 and a receipt. The
sample is frozen *before* a teacher is scored; the receipt's hash goes into
`docs/s72-teacher-qualification-preregistration-v1.json`.

Exclusions (all fail closed on a missing file):

* the S58 holdout and the S66 1,000-item calibration splits, i.e. every id a
  calibration or interpolation decision has ever seen;
* any train problem whose question text is an exact normalised match for a
  question in the MATH test corpus or the GSM8K test split, so the sample
  cannot overlap a test instrument even through a duplicated row.

Selection rule: restrict to the eligible pool, sort by id, seed-shuffle with a
named `random.Random`, take the first N, sort by id again. Deterministic under
the recorded seed and pool; the same rule as the S66 split builder.

Usage (CPU only, in the oracle court so the test arrows resolve):

    SELVEDGE_MEM=4g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
      scripts/court oracle -- python3 scripts/s72_stage0_sample.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.environ.get("HF_HOME", "/hf-cache")
sys.path.insert(0, os.path.join(ROOT, "oracle"))

_spec = importlib.util.spec_from_file_location(
    "s72_rft", os.path.join(ROOT, "oracle", "frontier", "rft.py"))
rft = importlib.util.module_from_spec(_spec)
sys.modules["s72_rft"] = rft
_spec.loader.exec_module(rft)

SYSTEM_PROMPT = None
USER_TEMPLATE = ("{question}\n\nSolve the problem step by step. "
                 "Put the final answer in \\boxed{{}}.")

GSM8K_TEST = os.path.join(
    CACHE, "datasets", "openai___gsm8k", "main", "0.0.0",
    "740312add88f781978c0658806c59bc2815b9866", "gsm8k-test.arrow")
MATH500_TEST = os.path.join(
    CACHE, "datasets", "HuggingFaceH4___math-500", "default", "0.0.0",
    "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "math-500-test.arrow")
MATH_TEST_CORPUS = os.path.join(ROOT, "results", "artifacts", "math-test",
                                "math-test-5000.jsonl")

EXCLUSION_FILES = [
    "results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl",
    "results/stage2/s66-calibration/gsm8k-cal-1000.jsonl",
    "results/stage2/s66-calibration/math-cal-1000.jsonl",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def now_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_questions() -> tuple[set[str], dict]:
    import pyarrow.ipc as ipc

    math_questions = {}
    if os.path.isfile(MATH_TEST_CORPUS):
        for row in read_jsonl(MATH_TEST_CORPUS):
            math_questions[normalise(row["question"])] = row.get("problem_id")
    math500 = set()
    if os.path.isfile(MATH500_TEST):
        for row in ipc.open_stream(MATH500_TEST).read_all().to_pylist():
            math500.add(normalise(row["problem"]))
    gsm8k = set()
    if os.path.isfile(GSM8K_TEST):
        for row in ipc.open_stream(GSM8K_TEST).read_all().to_pylist():
            gsm8k.add(normalise(row["question"]))
    evidence = {
        "math_test_corpus": {"path": os.path.relpath(MATH_TEST_CORPUS, ROOT),
                             "exists": os.path.isfile(MATH_TEST_CORPUS),
                             "questions": len(math_questions)},
        "math500": {"path": MATH500_TEST, "exists": os.path.isfile(MATH500_TEST),
                    "questions": len(math500)},
        "gsm8k_test": {"path": GSM8K_TEST, "exists": os.path.isfile(GSM8K_TEST),
                       "questions": len(gsm8k)},
    }
    return {"math": set(math_questions) | math500, "gsm8k": gsm8k}, evidence


def exclusion_ids() -> tuple[dict, list]:
    ids: dict[str, set[str]] = {"math": set(), "gsm8k": set()}
    record = []
    for rel in EXCLUSION_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            raise SystemExit(f"s72-stage0: missing exclusion file {rel}")
        rows = read_jsonl(path)
        for row in rows:
            identifier = row.get("id") or row.get("problem_id")
            if not identifier:
                raise SystemExit(f"s72-stage0: {rel} row has no id/problem_id")
            dataset = (row.get("dataset")
                       or ("gsm8k" if identifier.startswith("gsm8k") else "math"))
            ids.setdefault(dataset, set()).add(identifier)
        record.append({"path": rel, "sha256": sha256_file(path), "rows": len(rows)})
    return ids, record


def question_of(item: dict) -> str:
    """Recover the raw question from the loader's own prompt template."""
    prompt = item["prompt"]
    if prompt.startswith("Problem:\n"):
        return prompt[len("Problem:\n"):].split("\n\nSolution:", 1)[0]
    if prompt.startswith("Question: "):
        return prompt[len("Question: "):].rsplit("\nAnswer:", 1)[0]
    raise SystemExit(f"s72-stage0: unknown prompt template for {item['id']!r}")


def eligible(items: list[dict], dataset: str, excluded_ids: set[str],
             excluded_questions: set[str]) -> tuple[list[dict], int, int]:
    by_id = 0
    by_question = 0
    pool = []
    for item in items:
        if item["id"] in excluded_ids:
            by_id += 1
            continue
        if normalise(question_of(item)) in excluded_questions:
            by_question += 1
            continue
        pool.append(item)
    return pool, by_id, by_question


def draw(pool: list[dict], count: int, seed: int) -> list[dict]:
    ordered = sorted(pool, key=lambda item: item["id"])
    random.Random(seed).shuffle(ordered)
    chosen = ordered[:count]
    if len(chosen) != count:
        raise SystemExit(f"s72-stage0: pool has {len(chosen)} < {count}")
    return sorted(chosen, key=lambda item: item["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-count", type=int, default=250)
    parser.add_argument("--gsm8k-count", type=int, default=250)
    parser.add_argument("--seed-math", type=int, default=7200)
    parser.add_argument("--seed-gsm8k", type=int, default=7201)
    parser.add_argument("--out", default="results/artifacts/stage2/s72-stage0/sample-250x2.jsonl")
    parser.add_argument("--receipt", default="results/stage2/s72-recon/stage0-sample.json")
    args = parser.parse_args()

    questions, question_evidence = test_questions()
    excluded_ids, exclusion_record = exclusion_ids()

    math_items = rft.load_items("math-train")
    gsm8k_items = rft.load_items("gsm8k-train")

    math_pool, math_by_id, math_by_question = eligible(
        math_items, "math", excluded_ids.get("math", set()), questions["math"])
    gsm_pool, gsm_by_id, gsm_by_question = eligible(
        gsm8k_items, "gsm8k", excluded_ids.get("gsm8k", set()), questions["gsm8k"])

    math_sample = draw(math_pool, args.math_count, args.seed_math)
    gsm_sample = draw(gsm_pool, args.gsm8k_count, args.seed_gsm8k)

    corpus_path = os.path.join(ROOT, "results", "artifacts", "stage2",
                               "s58-combined", "stage2a-accepted-combined.jsonl")
    corpus_ids = {row["id"] for row in read_jsonl(corpus_path)} \
        if os.path.isfile(corpus_path) else set()
    math_in_corpus = sum(1 for item in math_sample if item["id"] in corpus_ids)
    gsm_in_corpus = sum(1 for item in gsm_sample if item["id"] in corpus_ids)

    rows = []
    for item in math_sample + gsm_sample:
        question = question_of(item)
        rows.append({
            "id": item["id"],
            "dataset": item["dataset"],
            "subject": item.get("subject"),
            "level": item.get("difficulty"),
            "question": question,
            "gold": item["target"],
            "prompt": USER_TEMPLATE.format(question=question),
            "system": SYSTEM_PROMPT,
        })
    rows.sort(key=lambda row: row["id"])

    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    sample_hash = sha256_bytes(payload)
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(payload)

    def profile(sample: list[dict]) -> dict:
        return {
            "items": len(sample),
            "by_level": dict(sorted(collections.Counter(
                str(item.get("difficulty")) for item in sample).items())),
            "by_subject": dict(sorted(collections.Counter(
                str(item.get("subject")) for item in sample).items())),
        }

    receipt = {
        "schema": "selvedge/s72-stage0-sample/v1",
        "stamp": now_stamp(),
        "created_utc": now_iso(),
        "kind": "stage0-qualification-sample",
        "protocol": {
            "system": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "decoding": "greedy (temperature 0, top_p 1)",
            "mode": "non-thinking",
            "max_new_tokens": 4096,
            "stop": "EOS only",
            "primary_basis": "flexible",
            "also_reported": ["strict", "exact"],
        },
        "sample": {
            "path": args.out,
            "sha256": sample_hash,
            "bytes": len(payload),
            "rows": len(rows),
            "math_rows": len(math_sample),
            "gsm8k_rows": len(gsm_sample),
        },
        "seed": {"math": args.seed_math, "gsm8k": args.seed_gsm8k,
                 "rule": "sort pool by id, random.Random(seed).shuffle, take N, sort by id"},
        "exclusions": {
            "files": exclusion_record,
            "test_overlap_check": question_evidence,
            "math_pool": {
                "train_items": len(math_items),
                "removed_by_id": math_by_id,
                "removed_by_test_question": math_by_question,
                "eligible": len(math_pool),
            },
            "gsm8k_pool": {
                "train_items": len(gsm8k_items),
                "removed_by_id": gsm_by_id,
                "removed_by_test_question": gsm_by_question,
                "eligible": len(gsm_pool),
            },
        },
        "math_sample_profile": profile(math_sample),
        "gsm8k_sample_profile": profile(gsm_sample),
        "diagnostic_training_overlap": {
            "note": "not an exclusion: the S58 accepted corpus is training data, "
                    "not a calibration or test set. Reported so a reader knows how "
                    "much of the qualification sample the 0.8B student has already "
                    "seen as a trace.",
            "paths": ["results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl"],
            "math_sample_ids_in_teacher_corpus": math_in_corpus,
            "gsm8k_sample_ids_in_teacher_corpus": gsm_in_corpus,
        },
    }
    receipt_path = os.path.join(ROOT, args.receipt)
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.out} sha256={sample_hash}")
    print(f"wrote {args.receipt}")
    print("math pool: train=%d removed_by_id=%d removed_by_test_question=%d eligible=%d"
          % (len(math_items), math_by_id, math_by_question, len(math_pool)))
    print("gsm8k pool: train=%d removed_by_id=%d removed_by_test_question=%d eligible=%d"
          % (len(gsm8k_items), gsm_by_id, gsm_by_question, len(gsm_pool)))
    print(f"math sample levels: {receipt['math_sample_profile']['by_level']}")
    print(f"gsm8k sample levels: {receipt['gsm8k_sample_profile']['by_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
