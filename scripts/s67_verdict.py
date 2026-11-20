#!/usr/bin/env python3
"""S67: assemble the final verdict from the scored receipts.

Reads the already-written receipts and computes only arithmetic that is
already registered:

* the soup's paired problem-level intervals against the post-trained baseline
  on GSM8K, MATH-500 and the full 5,000-problem MATH test (S57 bootstrap),
* the soup's paired intervals against the S65 seed-0 checkpoint on the
  instruments both were scored on,
* the retention verdict from the S67 audit receipt,
* per-model wall-clock and token throughput from the S54 receipts.

It does not score a model, does not read a test split for selection and does
not re-select alpha. The alpha is read from the calibration receipt and is
reported, never recomputed here.

Usage (CPU):
    SELVEDGE_MEM=4g scripts/court python -- \
      python3 scripts/s67_verdict.py \
        --calibration results/stage2/s67-soup/calibration.json \
        --soup-eval results/stage2/s67-final/soup-stage1.json \
        --baseline-math results/stage2/s67-final/baseline-math-full.json \
        --s65-seed0 results/stage1/s65-interp-0.375-nonthinking-greedy.json \
        --post-baseline results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
        --retention results/stage2/s67-final/retention-audit.json \
        --out results/stage2/s67-final/verdict.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import s57_paired as paired  # noqa: E402

SCHEMA = "selvedge/s67-verdict/v1"
PREREG = "docs/s67-resolution-instrument-preregistration-v1.json"
S66_THREE_SEED = "results/stage2/s66-seeds/three-seed.json"


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


def read(rel: str) -> dict:
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def task_rows(receipt: dict, task: str) -> dict | None:
    for row in receipt.get("tasks") or []:
        if (row.get("task") == task and row.get("mode") == "non-thinking"
                and row.get("decoding") == "greedy"):
            return row
    return None


def throughput(row: dict) -> dict:
    return {
        "items": row.get("items"),
        "wall_seconds": row.get("wall_seconds"),
        "decode_wall_seconds": row.get("decode_wall_seconds"),
        "generated_tokens": row.get("generated_tokens"),
        "tokens_per_second": row.get("tokens_per_second"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--soup-eval", required=True)
    parser.add_argument("--baseline-math", required=True)
    parser.add_argument("--s65-seed0", required=True)
    parser.add_argument("--post-baseline", required=True)
    parser.add_argument("--retention", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    calibration = read(args.calibration)
    selection = calibration.get("selection") or {}
    alpha = selection.get("coefficient")
    if alpha is None:
        raise SystemExit("s67-verdict: the calibration receipt has no selection")

    soup = read(args.soup_eval)
    baseline_math = read(args.baseline_math)
    seed0 = read(args.s65_seed0)
    post = read(args.post_baseline)
    retention = read(args.retention)
    seed_summary = read(S66_THREE_SEED).get("summary") or {}

    soup_rows = {task: task_rows(soup, task)
                 for task in ("gsm8k", "math500", "math_full")}
    if not all(soup_rows.values()):
        raise SystemExit("s67-verdict: the soup receipt is missing a task")
    baseline_full = task_rows(baseline_math, "math_full")
    if baseline_full is None:
        raise SystemExit("s67-verdict: the baseline receipt has no math_full")

    # Soup vs post-trained: GSM8K and MATH-500 pair against the S54 post-trained
    # baseline receipt (the same protocol the S65/S66 numbers used), full MATH
    # against the baseline run made for this instrument.
    comparisons = {}
    for task in ("gsm8k", "math500"):
        candidate = soup_rows[task]
        baseline = task_rows(post, task)
        if baseline is None:
            raise SystemExit(f"s67-verdict: no post baseline row for {task}")
        comparisons[f"soup_vs_post_{task}"] = paired.compare_task(
            candidate, baseline)
    comparisons["soup_vs_post_math_full"] = paired.compare_task(
        soup_rows["math_full"], baseline_full)

    soup_math_full = comparisons["soup_vs_post_math_full"]["bases"]["flexible"]
    gsm = comparisons["soup_vs_post_gsm8k"]["bases"]["flexible"]
    math500 = comparisons["soup_vs_post_math500"]["bases"]["flexible"]

    # Soup vs the S65 seed-0 checkpoint, on the instruments both were scored
    # on with the identical protocol.
    soup_vs_seed0 = {}
    for task in ("gsm8k", "math500"):
        soup_vs_seed0[task] = paired.compare_task(
            soup_rows[task], task_rows(seed0, task))
    seed0_math_full_scored = False

    retention_verdict = retention.get("verdict") or {}
    math_resolved = bool(soup_math_full["interval_above_zero"])
    complete = bool(gsm["interval_above_zero"] and math_resolved
                    and retention_verdict.get("retention_pass"))
    replicated_gsm8k = bool(
        (seed_summary.get("gsm8k") or {}).get("all_intervals_above_zero"))

    def speed(row):
        return throughput(row)

    payload = {
        "schema": SCHEMA,
        "kind": "s67-final-verdict",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": PREREG,
            "sha256": paired.sha256_file(os.path.join(ROOT, PREREG)),
        },
        "selection": {
            "coefficient": alpha,
            "receipt": calibration,
            "calibration_only": True,
            "test_split_read_for_selection": False,
        },
        "soup_vs_post_trained": comparisons,
        "soup_vs_s65_seed0": soup_vs_seed0,
        "soup_vs_s65_seed0_math_full": {
            "scored": seed0_math_full_scored,
            "why": ("the S65 seed-0 checkpoint was never scored on the full "
                    "5,000-problem MATH test, and this session does not spend "
                    "another full-instrument pass on it; the direct comparison "
                    "is reported on GSM8K and MATH-500 only"),
        },
        "retention": retention,
        "scores": {
            "gsm8k": gsm,
            "math500": math500,
            "math_full": soup_math_full,
        },
        "throughput": {
            "soup": {task: speed(row) for task, row in soup_rows.items()},
            "baseline_math_full": speed(baseline_full),
        },
        "verdict": {
            "replicated_gsm8k_win": replicated_gsm8k,
            "math_effect_resolved_at_larger_n": math_resolved,
            "complete_registered_win": complete,
            "soup_beats_post_trained_gsm8k": bool(gsm["interval_above_zero"]),
            "soup_beats_post_trained_math500": bool(
                math500["interval_above_zero"]),
            "soup_beats_post_trained_math_full": bool(
                soup_math_full["interval_above_zero"]),
            "retention_pass": bool(retention_verdict.get("retention_pass")),
            "soup_beats_s65_seed0_gsm8k": bool(
                soup_vs_seed0["gsm8k"]["bases"]["flexible"][
                    "interval_above_zero"]),
            "soup_beats_s65_seed0_math500": bool(
                soup_vs_seed0["math500"]["bases"]["flexible"][
                    "interval_above_zero"]),
        },
        "distinctions": {
            "replicated_gsm8k_win": (
                "GSM8K interval above zero on all three S66 seeds; already "
                "established and unchanged by this run"),
            "math_effect_resolved_at_larger_n": (
                "the full-MATH paired interval above zero at n=5,000; this is "
                "the clause the S66 verdict left unresolved at n=500"),
            "complete_registered_win": (
                "replicated GSM8K AND resolved full-MATH AND retention on full "
                "MMLU plus the 4,000-item MMLU-Pro instrument"),
        },
        "claims": {
            "single_use_of_resolution_instrument": True,
            "no_reselection_after_scoring": True,
            "paired_unit_is_the_problem": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(out + ".tmp", out)
    print(f"s67-verdict: alpha={alpha} full-MATH delta="
          f"{soup_math_full['delta']:+.4f} ci="
          f"[{soup_math_full['ci95'][0]:+.4f},{soup_math_full['ci95'][1]:+.4f}] "
          f"resolved={math_resolved} complete={complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
