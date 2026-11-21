#!/usr/bin/env python3
"""S68: assemble the depth-ramp test verdict from the scored receipts.

The ramp was selected on train-split calibration only. This script reads the
one-shot test receipts and computes, with the same S57 paired bootstrap used by
S67 (4,000 resamples, seed 53, problem-level pairing):

* ramp vs post-trained Qwen3.5-0.8B on GSM8K, MATH-500 and full MATH;
* ramp vs the S67 constant-alpha soup champion on the same three instruments,
  because the question is whether the ramp improves on the champion, not only
  whether it beats the post-trained baseline;
* the retention verdict from the audit receipt.

No test number is used to choose anything here; the selection receipt already
exists and is named.

Usage (CPU):
    SELVEDGE_MEM=4g scripts/court python -- \
      python3 scripts/s68_ramp_verdict.py \
        --selection results/stage2/s68-ramp/selection.json \
        --test results/stage2/s68-ramp/test-stage1.json \
        --baseline results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
        --baseline-math-full results/stage2/s67-final/baseline-math-full.json \
        --champion results/stage2/s67-final/soup-stage1.json \
        --retention results/stage2/s68-ramp/retention-audit.json \
        --out results/stage2/s68-ramp/verdict.json
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

SCHEMA = "selvedge/s68-ramp-verdict/v1"
TASKS = ("gsm8k", "math500", "math_full")


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


def read(path: str) -> dict:
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def task_row(receipt: dict, task: str) -> dict | None:
    for row in receipt.get("tasks") or []:
        if (row.get("task") == task and row.get("mode") == "non-thinking"
                and row.get("decoding") == "greedy" and "per_problem" in row):
            return row
    return None


def throughput(row: dict) -> dict:
    return {
        "items": row.get("items"),
        "wall_seconds": row.get("wall_seconds"),
        "generated_tokens": row.get("generated_tokens"),
        "tokens_per_second": row.get("tokens_per_second"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-math-full", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--retention", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    selection = read(args.selection)
    candidate = read(args.test)
    baseline = read(args.baseline)
    baseline_full = read(args.baseline_math_full)
    champion = read(args.champion)
    retention = read(args.retention)

    candidate_rows = {task: task_row(candidate, task) for task in TASKS}
    if not all(candidate_rows.values()):
        raise SystemExit("s68-verdict: the test receipt is missing a task")
    champion_rows = {task: task_row(champion, task) for task in TASKS}
    if not all(champion_rows.values()):
        raise SystemExit("s68-verdict: the champion receipt is missing a task")
    baseline_rows = {
        "gsm8k": task_row(baseline, "gsm8k"),
        "math500": task_row(baseline, "math500"),
        "math_full": task_row(baseline_full, "math_full"),
    }
    if not all(baseline_rows.values()):
        raise SystemExit("s68-verdict: the baseline receipts are missing a task")

    vs_post = {task: paired.compare_task(candidate_rows[task],
                                         baseline_rows[task])
               for task in TASKS}
    vs_champion = {task: paired.compare_task(candidate_rows[task],
                                             champion_rows[task])
                   for task in TASKS}
    scores = {task: vs_post[task]["bases"]["flexible"] for task in TASKS}
    champ_scores = {task: vs_champion[task]["bases"]["flexible"]
                    for task in TASKS}
    retention_verdict = retention.get("verdict") or {}
    full_resolved = bool(scores["math_full"]["interval_above_zero"])
    gsm_positive = bool(scores["gsm8k"]["interval_above_zero"])
    retention_pass = bool(retention_verdict.get("retention_pass"))
    config = selection.get("selection") or {}
    payload = {
        "schema": SCHEMA,
        "kind": "depth-ramp-test-verdict",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": selection.get("preregistration"),
        "selection": {
            "receipt": args.selection,
            "tag": config.get("tag"),
            "alpha_shallow": config.get("alpha_shallow"),
            "alpha_deep": config.get("alpha_deep"),
            "calibration_mean": config.get("calibration_mean"),
            "control_calibration_mean": (selection.get("control") or {}).get(
                "calibration_mean"),
            "gain_over_control": config.get("gain_over_control"),
            "adopted": config.get("adopted"),
            "test_split_read_for_selection": False,
        },
        "ramp_vs_post_trained": vs_post,
        "ramp_vs_s67_soup_champion": vs_champion,
        "scores_vs_post_trained": scores,
        "scores_vs_soup_champion": champ_scores,
        "retention": retention,
        "throughput": {
            "ramp": {task: throughput(row)
                     for task, row in candidate_rows.items()},
        },
        "verdict": {
            "beats_post_trained_gsm8k": gsm_positive,
            "beats_post_trained_math500": bool(
                scores["math500"]["interval_above_zero"]),
            "beats_post_trained_math_full": full_resolved,
            "retention_pass": retention_pass,
            "single_run_registered_win": bool(
                gsm_positive and full_resolved and retention_pass),
            "beats_soup_champion_gsm8k": bool(
                champ_scores["gsm8k"]["interval_above_zero"]),
            "beats_soup_champion_math500": bool(
                champ_scores["math500"]["interval_above_zero"]),
            "beats_soup_champion_math_full": bool(
                champ_scores["math_full"]["interval_above_zero"]),
        },
        "distinctions": {
            "seed_replication": (
                "not claimed for the ramp: the three-seed GSM8K replication is "
                "the S66 seed-level result, and the ramp is one interpolation "
                "of the soup. The ramp's own GSM8K interval is reported as a "
                "single-run interval."),
            "full_math": (
                "the full 5,000-problem instrument is scored once for this "
                "ramp; it is not enlarged or re-scored on the basis of the "
                "result."),
            "champion": (
                "the directly comparable champion is the S67 constant-alpha "
                "soup, scored under the same S54 protocol."),
        },
        "claims": {
            "no_reselection_after_scoring": True,
            "paired_unit_is_the_problem": True,
            "no_frontier_claim": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    for task in TASKS:
        row = scores[task]
        print(f"s68-verdict: {task} delta={row['delta']:+.4f} "
              f"ci=[{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}]")
    print(f"s68-verdict: full_math_resolved={full_resolved} "
          f"retention={retention_pass} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
