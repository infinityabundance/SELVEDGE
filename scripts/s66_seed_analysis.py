#!/usr/bin/env python3
"""S66: aggregate the three seed-0/1/2 interpolation results.

Seed 0 is the S65 receipt set. Seeds 1 and 2 are the S66 replication runs at the
fixed coefficient 0.375; no coefficient is re-selected. The script reads only
receipts, computes the per-seed interval verdicts and the seed-level mean and
spread, and refuses to invent a number that a receipt does not carry.

Usage:
    SELVEDGE_MEM=2g scripts/court python -- \
      python3 scripts/s66_seed_analysis.py \
        --out results/stage2/s66-seeds/three-seed.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEED0_IDENTITY = "results/stage2/s65-interp-selected/identity.json"
SEED1_RESULT = "results/stage2/s66-seeds/seed1/result.json"
SEED2_RESULT = "results/stage2/s66-seeds/seed2/result.json"
BOUND = 0.01


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


def load(rel: str) -> dict:
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def math_row(stage1: dict, task: str) -> dict:
    row = next(row for row in stage1["tasks"]
               if row["task"] == task and row.get("decoding") == "greedy")
    return row


def paired_row(paired: dict, task: str) -> dict:
    return next(row for row in paired["rows"] if row["task"] == task)


def from_seed0() -> dict:
    identity = load(SEED0_IDENTITY)
    full = identity["full_test_evaluation"]
    gsm = full["gsm8k"]
    math = full["math500"]
    mmlu = full["mmlu"]
    return {
        "seed": 0,
        "source": SEED0_IDENTITY,
        "coefficient": identity["selection"]["coefficient"],
        "candidate_checkpoint": identity["source_weights"]["candidate_endpoint"],
        "interpolated_checkpoint": identity["checkpoint"],
        "gsm8k": {
            "accuracy": gsm["candidate_flexible"],
            "baseline": gsm["baseline_flexible"],
            "delta": gsm["delta"],
            "ci95": gsm["ci95"],
            "interval_above_zero": gsm["ci95"][0] > 0.0,
        },
        "math500": {
            "accuracy": math["candidate_flexible"],
            "baseline": math["baseline_flexible"],
            "delta": math["delta"],
            "ci95": math["ci95"],
            "interval_above_zero": math["ci95"][0] > 0.0,
        },
        "mmlu": {
            "accuracy": mmlu["candidate"],
            "baseline": mmlu["baseline"],
            "delta": mmlu["delta"],
            "ci95": mmlu["ci95"],
            "pass": (mmlu["delta"] >= -BOUND and mmlu["ci95"][0] >= -BOUND),
        },
        "receipts": {
            "stage1": gsm["receipt"],
            "paired_math": gsm["paired_receipt"],
            "retention": mmlu["paired_receipt"],
        },
    }


def from_seed_result(rel: str) -> dict:
    result = load(rel)
    stage1 = load(result["receipts"]["stage1"]["path"])
    paired = load(result["receipts"]["paired_math"]["path"])
    retention = load(result["receipts"]["paired_retention"]["path"])
    gsm = math_row(stage1, "gsm8k")
    math = math_row(stage1, "math500")
    gsm_pair = paired_row(paired, "gsm8k")["bases"]["flexible"]
    math_pair = paired_row(paired, "math500")["bases"]["flexible"]
    mmlu_pair = next(row for row in retention["rows"] if row["task"] == "mmlu")
    # The retention receipt reports absolute scores on both sides; the task-1
    # report expects them beside the paired delta.
    mmlu_receipt = load(result["receipts"]["mmlu"]["path"])
    mmlu_task = next(row for row in mmlu_receipt["tasks"] if row["task"] == "mmlu")
    return {
        "seed": result["seed"],
        "source": rel,
        "coefficient": result["coefficient"],
        "candidate_checkpoint": result["candidate_checkpoint"],
        "interpolated_checkpoint": result["interpolated_checkpoint"],
        "gsm8k": {
            "accuracy": gsm["flexible"]["accuracy"],
            "baseline": gsm_pair["baseline_accuracy"],
            "delta": gsm_pair["delta"],
            "ci95": gsm_pair["ci95"],
            "interval_above_zero": gsm_pair["interval_above_zero"],
            "truncation_rate": gsm["termination"]["truncation_rate"],
            "loop_fraction": gsm["termination"]["loop_classified_fraction"],
        },
        "math500": {
            "accuracy": math["flexible"]["accuracy"],
            "baseline": math_pair["baseline_accuracy"],
            "delta": math_pair["delta"],
            "ci95": math_pair["ci95"],
            "interval_above_zero": math_pair["interval_above_zero"],
            "truncation_rate": math["termination"]["truncation_rate"],
            "loop_fraction": math["termination"]["loop_classified_fraction"],
        },
        "mmlu": {
            "accuracy": mmlu_task["all_metrics"].get(
                mmlu_task.get("primary_metric") or "acc,none"),
            "baseline": mmlu_pair["baseline_score"],
            "delta": mmlu_pair["delta"],
            "ci95": mmlu_pair["ci95"],
            "pass": (mmlu_pair["delta"] >= -BOUND
                     and mmlu_pair["ci95"][0] >= -BOUND),
        },
        "receipts": {
            "stage1": result["receipts"]["stage1"]["path"],
            "paired_math": result["receipts"]["paired_math"]["path"],
            "retention": result["receipts"]["paired_retention"]["path"],
            "mmlu": result["receipts"]["mmlu"]["path"],
        },
    }


def summary(rows: list[dict]) -> dict:
    out = {}
    for task in ("gsm8k", "math500", "mmlu"):
        deltas = [row[task]["delta"] for row in rows]
        out[task] = {
            "deltas": deltas,
            "mean_delta": statistics.fmean(deltas),
            "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "all_seeds_above_zero": all(value > 0.0 for value in deltas),
            "all_intervals_above_zero": all(
                row[task].get("interval_above_zero", True) for row in rows),
        }
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", default=SEED1_RESULT)
    parser.add_argument("--seed2", default=SEED2_RESULT)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    paths = [args.seed1, args.seed2]
    missing = [path for path in paths if not os.path.exists(os.path.join(ROOT, path))]
    if missing:
        raise SystemExit(f"s66-seed-analysis: missing seed receipts {missing}")
    rows = [from_seed0()] + [from_seed_result(path) for path in paths]
    stats = summary(rows)
    math_hold = (stats["gsm8k"]["all_seeds_above_zero"]
                 and stats["gsm8k"]["all_intervals_above_zero"]
                 and stats["math500"]["all_seeds_above_zero"]
                 and stats["math500"]["all_intervals_above_zero"])
    retention_hold = all(row["mmlu"]["pass"] for row in rows)
    payload = {
        "schema": "selvedge/s66-three-seed/v1",
        "kind": "three-seed-interpolation-replication",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "coefficient": 0.375,
        "alpha_reselected": False,
        "test_split_used_for_selection": False,
        "seeds": rows,
        "summary": stats,
        "verdict": {
            "both_math_clauses_positive_across_seeds": math_hold,
            "retention_passes_across_seeds": retention_hold,
            "seed_level_win": bool(math_hold and retention_hold),
            "note": ("per-seed paired intervals, not a pooled interval; the "
                     "seed-level mean/spread is descriptive and is not used to "
                     "select any coefficient"),
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    for row in rows:
        print(f"s66-seed: seed {row['seed']} gsm8k={row['gsm8k']['delta']:+.4f} "
              f"[{row['gsm8k']['ci95'][0]:+.4f},{row['gsm8k']['ci95'][1]:+.4f}] "
              f"math500={row['math500']['delta']:+.4f} "
              f"[{row['math500']['ci95'][0]:+.4f},{row['math500']['ci95'][1]:+.4f}] "
              f"mmlu={row['mmlu']['delta']:+.4f} pass={row['mmlu']['pass']}")
    print(f"s66-seed: math across seeds={math_hold} retention={retention_hold} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
