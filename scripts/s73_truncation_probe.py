#!/usr/bin/env python3
"""S73: run the truncation classifier over stored MATH completions (CPU only).

This is the measurement half of the S73 probe. It never loads a model and never
generates: it reads a stored per-problem JSONL that already carries
``completion``, ``truncated``, ``new_tokens`` and (for correctness) ``gold``, and
writes a receipt with the four-way class breakdown from
``oracle/frontier/truncation.py``.

The registered use is the 158 truncated S67 soup MATH-500 rows at 4,096, as a
**pre-run estimate** of which fault the ladder must fix. The same script runs
unchanged over any other stored completion file; every receipt names its input
path and sha256, so a reader can tell a replay from a new generation.

Usage:
    python3 scripts/s73_truncation_probe.py \
        --input results/stage2/s67-final/soup-raw/math500-non-thinking-greedy.jsonl \
        --label soup_math500 --out results/stage2/s73-truncation/soup-math500.json

Dry run (no receipt):
    python3 scripts/s73_truncation_probe.py --input ... --label ... --stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import truncation  # noqa: E402

SCHEMA = "selvedge/s73-truncation-classification/v1"
CLASSIFIER_REL = "oracle/frontier/truncation.py"


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "completion" not in row:
                raise SystemExit(f"{path}: a row has no 'completion'")
            if "truncated" not in row:
                raise SystemExit(f"{path}: a row has no 'truncated' flag; the "
                                 "class breakdown is defined on it")
            rows.append(row)
    return rows


def classify_rows(rows: list[dict], task: str) -> list[dict]:
    """Classify every row; deterministic given the stored text and gold."""
    records = []
    for index, row in enumerate(rows):
        classified = truncation.classify_completion(
            row["completion"], row.get("gold"), row["truncated"])
        classified["index"] = index
        classified["problem_id"] = row.get("problem_id")
        classified["new_tokens"] = row.get("new_tokens")
        classified["task"] = task
        records.append(classified)
    return records


def build_receipt(rows: list[dict], records: list[dict], label: str,
                  input_path: str, task: str, argv: list[str]) -> dict:
    return {
        "schema": SCHEMA,
        "label": label,
        "task": task,
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "command": " ".join(argv),
        "classifier": {
            "module": CLASSIFIER_REL,
            "sha256": sha256_file(os.path.join(ROOT, CLASSIFIER_REL)),
            "version": truncation.CLASSIFIER_VERSION,
            "box_then_continued_min_nonspace":
                truncation.BOX_THEN_CONTINUED_MIN_NONSPACE,
            "classes": list(truncation.CLASSES),
            "loop_detector": "frontier.termination.classify (S56 operating point)",
        },
        "input": {
            "path": input_path,
            "sha256": sha256_file(input_path),
            "rows": len(rows),
        },
        "all_rows": truncation.summarise(records, only_truncated=False),
        "truncated_rows": truncation.summarise(records, only_truncated=True),
        "per_row": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="stored completion JSONL")
    parser.add_argument("--label", required=True, help="receipt label, e.g. soup_math500")
    parser.add_argument("--task", default="math500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="receipt path")
    parser.add_argument("--stdout", action="store_true",
                        help="print the receipt instead of writing it")
    parser.add_argument("--no-per-row", action="store_true",
                        help="omit per-row records (counts only)")
    parser.add_argument("--expect-rows", type=int, default=None,
                        help="fail closed unless the input has this many rows")
    parser.add_argument("--expect-truncated", type=int, default=None,
                        help="fail closed unless this many rows are truncated; "
                             "pins the replay to the receipt the recon quoted")
    args = parser.parse_args(argv)

    if args.task not in ("math500", "math_full", "math"):
        raise SystemExit("only MATH-family tasks are supported (the classifier "
                         "is defined on \\boxed{} answers)")
    full_argv = ["python3", "scripts/s73_truncation_probe.py"] + list(argv or sys.argv[1:])

    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise SystemExit(f"expected {args.expect_rows} rows, found {len(rows)}")
    if args.expect_truncated is not None:
        found = sum(1 for row in rows if row["truncated"])
        if found != args.expect_truncated:
            raise SystemExit(
                f"expected {args.expect_truncated} truncated rows, found {found}")
    records = classify_rows(rows, args.task)
    receipt = build_receipt(rows, records, args.label, args.input, args.task,
                            full_argv)
    if args.no_per_row:
        receipt.pop("per_row", None)

    summary = receipt["truncated_rows"]
    print(f"s73-truncation: {args.label} rows={receipt['all_rows']['rows']} "
          f"truncated={summary['rows']}")
    print("  all rows by class:      " +
          json.dumps(receipt["all_rows"]["by_class"], sort_keys=True))
    print("  truncated rows by class:" +
          json.dumps(summary["by_class"], sort_keys=True))
    print(f"  truncated correct box anywhere: "
          f"{summary['correct_box_anywhere_rows']} ")
    print(f"  truncated class-correct overlap: "
          f"{json.dumps(summary['correct_box_anywhere_by_class'], sort_keys=True)}")

    if args.stdout:
        print(json.dumps(receipt, indent=1))
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --stdout is given")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=False)
        handle.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
