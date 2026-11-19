#!/usr/bin/env python3
"""S66: build and register the 1,000-item GSM8K-cal and MATH-cal splits.

The split is a strict enlargement of the S60 300-item calibration: it uses the
same shuffle seeds (6000 for GSM8K, 6001 for MATH), the same exclusion files and
a limit of 1,000, so the first 300 items are the items the original coefficient
selection read. The additional 700 per task are train items never used by S58
training, S58 holdout, or any test instrument.

Only the train sources are opened. `frontier.rft.load_items` reads
`gsm8k-train.arrow` and the Hendrycks MATH train parquet; no test split is
loaded, so "disjoint from all test sets" is a property of the source split, not
a comparison against test item ids.

Usage (CPU, oracle court because pyarrow is pinned there):
    SELVEDGE_MEM=6g scripts/court oracle -- \
      python3 scripts/s66_build_calibration.py \
        --out-dir results/stage2/s66-calibration \
        --registration docs/s66-calibration-splits-v1.json
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
sys.path.insert(0, HERE)

from frontier import rft  # noqa: E402
from s60_interpolate import build_calibration, _load_exclusions  # noqa: E402

SCHEMA = "selvedge/s66-calibration-splits/v1"
EXCLUSIONS = [
    "results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl",
    "results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl",
    "results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl",
]


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


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def describe(rows: list[dict], source: str, excluded: set[str]) -> dict:
    ids = [row["problem_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"s66-calibration: duplicate ids in {source}")
    overlap = sorted(set(ids) & excluded)
    if overlap:
        raise SystemExit(
            f"s66-calibration: {source} overlaps {len(overlap)} S58 ids "
            f"(first {overlap[:3]})")
    return {
        "source": source,
        "items": len(rows),
        "unique_ids": len(set(ids)),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
        "subjects": sorted({str(row.get("subject")) for row in rows
                            if row.get("subject") is not None}),
        "levels": sorted({str(row.get("level")) for row in rows
                          if row.get("level") is not None}),
        "s58_overlap": 0,
        "test_split_read": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/stage2/s66-calibration")
    parser.add_argument("--registration", default="docs/s66-calibration-splits-v1.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6000)
    args = parser.parse_args(argv)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    excluded = _load_exclusions(EXCLUSIONS)
    gsm = build_calibration("gsm8k", excluded, args.limit, args.seed)
    math = build_calibration("math", excluded, args.limit, args.seed + 1)
    if len(gsm) != args.limit or len(math) != args.limit:
        raise SystemExit(
            f"s66-calibration: requested {args.limit}, got {len(gsm)} GSM8K and "
            f"{len(math)} MATH after exclusions")

    # The S60 split was built with the same seeds and limit 300. The first 300
    # rows here are therefore the original selection set; verify rather than
    # assume it.
    old_gsm = build_calibration("gsm8k", excluded, 300, args.seed)
    old_math = build_calibration("math", excluded, 300, args.seed + 1)
    gsm_ids = {row["problem_id"] for row in gsm}
    math_ids = {row["problem_id"] for row in math}
    gsm_old_ids = [row["problem_id"] for row in old_gsm]
    math_old_ids = [row["problem_id"] for row in old_math]
    if not set(gsm_old_ids) <= gsm_ids or not set(math_old_ids) <= math_ids:
        raise SystemExit(
            "s66-calibration: the enlarged split is not a superset of the "
            "S60 300-item calibration")

    gsm_path = os.path.join(out_dir, "gsm8k-cal-1000.jsonl")
    math_path = os.path.join(out_dir, "math-cal-1000.jsonl")
    write_jsonl(gsm_path, gsm)
    write_jsonl(math_path, math)

    registration = {
        "schema": SCHEMA,
        "identifier": "S66-CALIBRATION-SPLITS-V1",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "purpose": (
            "enlarge the train-split interpolation calibration from 300 to "
            "1,000 items per task and test whether the alpha selection is "
            "stable; selection remains calibration-only"),
        "source": {
            "gsm8k": "gsm8k-train.arrow (openai/gsm8k, split=train)",
            "math": "Hendrycks MATH train parquet (split=train)",
            "test_split_read": False,
            "test_disjointness": (
                "test instruments share no split with these sources; no test "
                "file or raw test receipt was opened while building the split"),
        },
        "seed": {
            "gsm8k": args.seed,
            "math": args.seed + 1,
            "rule": "random.Random(seed).shuffle(train items), take first N, "
                    "then sort by id",
        },
        "exclusions": {
            "files": EXCLUSIONS,
            "sha256": {path: sha256_file(os.path.join(ROOT, path))
                       for path in EXCLUSIONS},
            "rule": "every id in any exclusion file is removed before sampling",
        },
        "splits": {
            "gsm8k_cal": {
                **describe(gsm, "gsm8k-train", excluded),
                "path": os.path.relpath(gsm_path, ROOT),
                "sha256": sha256_file(gsm_path),
                "superset_of_s60_300": True,
                "s60_300_ids": gsm_old_ids,
            },
            "math_cal": {
                **describe(math, "math-train", excluded),
                "path": os.path.relpath(math_path, ROOT),
                "sha256": sha256_file(math_path),
                "superset_of_s60_300": True,
                "s60_300_ids": math_old_ids,
            },
        },
        "claims": {
            "calibration_is_train_split_only": True,
            "no_test_used_for_selection": True,
            "superset_of_previous_calibration": True,
        },
    }
    reg_path = os.path.join(ROOT, args.registration)
    os.makedirs(os.path.dirname(reg_path) or ".", exist_ok=True)
    tmp = reg_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(registration, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, reg_path)
    print(f"s66-calibration: GSM8K {len(gsm)} -> {registration['splits']['gsm8k_cal']['path']}")
    print(f"s66-calibration: MATH  {len(math)} -> {registration['splits']['math_cal']['path']}")
    print(f"s66-calibration: registration -> {args.registration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
