#!/usr/bin/env python3
"""S57: paired, problem-level comparison of two stage-1 receipts.

The registered win condition is a prompt-level paired interval: the candidate
must beat the post-trained 0.8B on GSM8K *and* MATH-500 with the 95% interval of
the per-problem difference above zero. This script builds exactly that number
from the `per_problem` rows that every `selvedge/s54-stage1-eval/v1` receipt
already carries.

Rules it keeps, unchanged from `scripts/s54_gap.py` and the S53 registration:

* the resampling unit is the **problem**, never a continuation;
* the two receipts are paired on `problem_id`, and a problem missing from either
  side is dropped and counted (`paired` < `a_items`);
* 4,000 resamples, seed 53, percentile interval at 2.5% / 97.5%;
* the primary verdict is the receipt's own `flexible` flag (the registered
  `accuracy`, i.e. last balanced ``\\boxed{}`` then the documented fallback);
  `strict` and `exact` are reported beside it and never substituted for it.

Usage:
    scripts/court python -- python3 scripts/s57_paired.py \
        --candidate results/stage1/s57-stage2a-final-greedy.json \
        --baseline results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
        --out results/stage1/s57-paired-final-vs-post.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCHEMA = "selvedge/s57-paired-comparison/v1"
RESAMPLES = 4000
SEED = 53
BASES = ("flexible", "strict", "exact")


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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap(deltas: list[float]) -> dict:
    if not deltas:
        return {"problems": 0, "delta": None, "ci95": None}
    rng = random.Random(SEED)
    n = len(deltas)
    draws = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(RESAMPLES)]
    return {
        "problems": n,
        "delta": sum(deltas) / n,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "resamples": RESAMPLES,
        "seed": SEED,
        "basis": ("prompt-level paired bootstrap; the resampling unit is the "
                  "distinct problem, never the continuation"),
    }


def greedy_rows(payload: dict) -> dict[str, dict]:
    """Rows for greedy decoding, keyed by task|mode."""
    rows = {}
    for row in payload.get("tasks", []):
        if row.get("decoding") != "greedy":
            continue
        rows[f"{row['task']}|{row['mode']}"] = row
    return rows


def compare_task(candidate: dict, baseline: dict) -> dict:
    candidate_by_id = {row["problem_id"]: row for row in candidate["per_problem"]}
    baseline_by_id = {row["problem_id"]: row for row in baseline["per_problem"]}
    common = sorted(set(candidate_by_id) & set(baseline_by_id))
    out = {
        "task": candidate["task"],
        "mode": candidate["mode"],
        "candidate_items": len(candidate_by_id),
        "baseline_items": len(baseline_by_id),
        "paired_problems": len(common),
        "candidate_only": len(set(candidate_by_id) - set(baseline_by_id)),
        "baseline_only": len(set(baseline_by_id) - set(candidate_by_id)),
        "termination": {
            "candidate": candidate.get("termination"),
            "baseline": baseline.get("termination"),
        },
        "bases": {},
    }
    for basis in BASES:
        field = f"{basis}_correct"
        deltas = [float(bool(candidate_by_id[key][field]))
                  - float(bool(baseline_by_id[key][field])) for key in common]
        candidate_correct = sum(bool(candidate_by_id[key][field]) for key in common)
        baseline_correct = sum(bool(baseline_by_id[key][field]) for key in common)
        interval = bootstrap(deltas)
        ci = interval["ci95"]
        out["bases"][basis] = {
            "candidate_accuracy": (candidate_correct / len(common)
                                   if common else None),
            "baseline_accuracy": (baseline_correct / len(common)
                                  if common else None),
            "candidate_correct": candidate_correct,
            "baseline_correct": baseline_correct,
            "delta": interval["delta"],
            "ci95": ci,
            "problems": interval["problems"],
            "resamples": interval["resamples"],
            "seed": interval["seed"],
            "basis": interval["basis"],
            "interval_above_zero": bool(ci is not None and ci[0] > 0.0),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task filter (default: every task "
                             "present in both receipts)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    candidate_path = os.path.join(ROOT, args.candidate)
    baseline_path = os.path.join(ROOT, args.baseline)
    candidate = json.load(open(candidate_path, encoding="utf-8"))
    baseline = json.load(open(baseline_path, encoding="utf-8"))

    candidate_rows = greedy_rows(candidate)
    baseline_rows = greedy_rows(baseline)
    keys = sorted(set(candidate_rows) & set(baseline_rows))
    if args.tasks:
        wanted = {value.strip() for value in args.tasks.split(",") if value.strip()}
        keys = [key for key in keys if key.split("|")[0] in wanted]
    if not keys:
        raise SystemExit("s57-paired: no shared greedy task rows")

    rows = [compare_task(candidate_rows[key], baseline_rows[key]) for key in keys]
    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "candidate": {
            "path": os.path.relpath(candidate_path, ROOT),
            "sha256": sha256_file(candidate_path),
            "model": candidate.get("model"),
            "checkpoint": candidate.get("checkpoint"),
        },
        "baseline": {
            "path": os.path.relpath(baseline_path, ROOT),
            "sha256": sha256_file(baseline_path),
            "model": baseline.get("model"),
            "checkpoint": baseline.get("checkpoint"),
        },
        "primary_basis": "flexible (the registered `accuracy` verdict)",
        "rows": rows,
        "command": sys.argv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(os.path.join(ROOT, args.out)))
                or ".", exist_ok=True)
    tmp = os.path.join(ROOT, args.out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, os.path.join(ROOT, args.out))

    for row in rows:
        primary = row["bases"]["flexible"]
        print(f"s57-paired: {row['task']} n={row['paired_problems']} "
              f"candidate={primary['candidate_accuracy']:.4f} "
              f"baseline={primary['baseline_accuracy']:.4f} "
              f"delta={primary['delta']:+.4f} "
              f"ci95=[{primary['ci95'][0]:+.4f},{primary['ci95'][1]:+.4f}] "
              f"above_zero={primary['interval_above_zero']}")
    print(f"s57-paired: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
