#!/usr/bin/env python3
"""S69: assemble the preference verdict from the scored receipts.

The preference arm was selected on train-split calibration only. This script
reads the calibration selection receipt and, when the adoption gate passed, the
one-shot test receipts, and computes with the S57 paired bootstrap (4,000
resamples, seed 53, problem-level pairing):

* the candidate against the S67 soup champion on GSM8K, MATH-500 and full MATH;
* the retention comparison against the soup on the same MMLU and MMLU-Pro raw
  dumps it was scored with (no soup rescore);
* the matched SFT control at the same cadence, for attribution, if it was run;
* the registered win bar.

If the calibration gate failed, no test split is read and the verdict records
the calibration-level failure and the measured numbers that produced it. The
cost arithmetic is computed from the receipts named here; no number is typed.

Usage (CPU):
    SELVEDGE_MEM=8g scripts/court python -- python3 scripts/s69_verdict.py \
      --selection results/stage2/s69-calibration/selection.json \
      --test results/stage2/s69-final/test-stage1.json \
      --champion results/stage2/s67-final/soup-stage1.json \
      --champion-mmlu results/stage2/s67-final/raw/20261120T125553Z-mmlu.json.gz \
      --champion-mmlu-pro results/stage2/s67-final/raw/20261120T151153Z-mmlu_pro.json.gz \
      --candidate-mmlu results/stage2/s69-final/raw/<stamp>-mmlu.json.gz \
      --candidate-mmlu-pro results/stage2/s69-final/raw/<stamp>-mmlu_pro.json.gz \
      --out results/stage2/s69-final/verdict.json
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

SCHEMA = "selvedge/s69-preference-verdict/v1"
TASKS = ("gsm8k", "math500", "math_full")
RETENTION_BOUND = 0.01


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


def no_regression(row: dict) -> bool:
    return bool(row["delta"] is not None
                and row["delta"] >= -RETENTION_BOUND
                and row["ci95"] is not None
                and row["ci95"][0] >= -RETENTION_BOUND)


def cost_from_stage1(receipt: dict) -> dict:
    return {
        "items": sum(row.get("items") or 0 for row in receipt.get("tasks") or []),
        "generated_tokens": sum(row.get("generated_tokens") or 0
                                for row in receipt.get("tasks") or []),
        "decode_wall_seconds": sum(row.get("wall_seconds") or 0
                                   for row in receipt.get("tasks") or []),
        "tokens_per_second": (
            (sum(row.get("generated_tokens") or 0
                 for row in receipt.get("tasks") or [])
             / sum(row.get("wall_seconds") or 0
                   for row in receipt.get("tasks") or []))
            if sum(row.get("wall_seconds") or 0
                   for row in receipt.get("tasks") or []) > 0 else None),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--champion", default=None)
    parser.add_argument("--champion-mmlu", default=None)
    parser.add_argument("--champion-mmlu-pro", default=None)
    parser.add_argument("--candidate-mmlu", default=None)
    parser.add_argument("--candidate-mmlu-pro", default=None)
    parser.add_argument("--control", default=None,
                        help="matched SFT control test receipt (gsm8k, math500)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    selection = read(args.selection)
    config = selection.get("selection") or {}
    adopted = bool(config.get("adopted"))
    payload = {
        "schema": SCHEMA,
        "kind": "s69-preference-verdict",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": selection.get("preregistration"),
        "selection": {
            "receipt": args.selection,
            "cadence": config.get("cadence"),
            "checkpoint": config.get("checkpoint"),
            "checkpoint_sha256": config.get("checkpoint_sha256"),
            "math_cal": config.get("math_cal"),
            "gsm8k_cal": config.get("gsm8k_cal"),
            "mmlu_cal": config.get("mmlu_cal"),
            "soup_math_cal": config.get("soup_math_cal"),
            "soup_gsm8k_cal": config.get("soup_gsm8k_cal"),
            "soup_mmlu_cal": config.get("soup_mmlu_cal"),
            "adopted": adopted,
            "test_split_read_for_selection": False,
        },
        "calibration_cost": {
            "checkpoints": [{
                "cadence": row.get("cadence"),
                "wall_seconds": row.get("wall_seconds"),
                "generated_tokens": row.get("generated_tokens"),
            } for row in selection.get("rows") or []],
        },
        "claims": {
            "no_reselection_after_scoring": True,
            "paired_unit_is_the_problem": True,
            "no_frontier_claim": True,
        },
    }

    if not adopted:
        payload["test_read"] = False
        payload["verdict"] = {
            "adopted": False,
            "test_read": False,
            "failure_mode": "calibration adoption gate not met",
            "candidate_math_cal": config.get("math_cal"),
            "soup_math_cal": config.get("soup_math_cal"),
            "candidate_gsm8k_cal": config.get("gsm8k_cal"),
            "soup_gsm8k_cal": config.get("soup_gsm8k_cal"),
            "complete_registered_win": False,
        }
        _write(args.out, payload)
        print(f"s69-verdict: not adopted on calibration; no test split read "
              f"-> {args.out}")
        return 0

    required = {
        "--test": args.test,
        "--champion": args.champion,
        "--champion-mmlu": args.champion_mmlu,
        "--champion-mmlu-pro": args.champion_mmlu_pro,
        "--candidate-mmlu": args.candidate_mmlu,
        "--candidate-mmlu-pro": args.candidate_mmlu_pro,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"s69-verdict: adopted, so {missing} are required")

    candidate = read(args.test)
    champion = read(args.champion)
    candidate_rows = {task: task_row(candidate, task) for task in TASKS}
    champion_rows = {task: task_row(champion, task) for task in TASKS}
    if not all(candidate_rows.values()):
        raise SystemExit("s69-verdict: the test receipt is missing a task")
    if not all(champion_rows.values()):
        raise SystemExit("s69-verdict: the champion receipt is missing a task")

    vs_soup = {task: paired.compare_task(candidate_rows[task],
                                         champion_rows[task])
               for task in TASKS}
    mmlu = paired.compare_lm_eval(os.path.join(ROOT, args.candidate_mmlu),
                                  os.path.join(ROOT, args.champion_mmlu), "acc")
    mmlu_pro = paired.compare_lm_eval(
        os.path.join(ROOT, args.candidate_mmlu_pro),
        os.path.join(ROOT, args.champion_mmlu_pro), "exact_match")

    math500 = vs_soup["math500"]["bases"]["flexible"]
    math_full = vs_soup["math_full"]["bases"]["flexible"]
    gsm8k = vs_soup["gsm8k"]["bases"]["flexible"]
    win = {
        "beats_soup_math500": bool(math500["interval_above_zero"]),
        "beats_soup_math_full": bool(math_full["interval_above_zero"]),
        "gsm8k_no_regression": no_regression(gsm8k),
        "mmlu_no_regression": no_regression(mmlu),
        "mmlu_pro_secondary_pass": no_regression(mmlu_pro),
    }
    win["complete_registered_win"] = bool(
        win["beats_soup_math500"] and win["beats_soup_math_full"]
        and win["gsm8k_no_regression"] and win["mmlu_no_regression"])

    control = None
    if args.control:
        control_receipt = read(args.control)
        control_vs_soup = {}
        control_vs_candidate = {}
        for task in ("gsm8k", "math500"):
            row = task_row(control_receipt, task)
            if row is None:
                continue
            control_vs_soup[task] = paired.compare_task(
                row, champion_rows[task])
            control_vs_candidate[task] = paired.compare_task(
                row, candidate_rows[task])
        control = {
            "receipt": args.control,
            "vs_soup": control_vs_soup,
            "vs_preference": control_vs_candidate,
            "role": ("attribution: the matched SFT control at the same cadence "
                     "shows whether the preference objective adds anything over "
                     "supervised fine-tuning on the same chosen continuations"),
        }

    payload.update({
        "test_read": True,
        "test": {
            "path": args.test,
            "scores_vs_soup": {
                task: vs_soup[task]["bases"]["flexible"] for task in TASKS},
            "all_bases_vs_soup": vs_soup,
            "retention_vs_soup": {"mmlu": mmlu, "mmlu_pro": mmlu_pro},
        },
        "control": control,
        "cost": {
            "test": cost_from_stage1(candidate),
            "control": (cost_from_stage1(control_receipt)
                        if args.control else None),
        },
        "verdict": {
            "adopted": True,
            "test_read": True,
            **win,
            "win_bar": ("MATH-500 and full MATH paired intervals above zero, "
                        "with GSM8K and MMLU point deltas and paired interval "
                        "lower bounds no worse than -1.0 point"),
        },
    })
    _write(args.out, payload)
    for task in TASKS:
        row = payload["test"]["scores_vs_soup"][task]
        print(f"s69-verdict: {task} delta={row['delta']:+.4f} "
              f"ci=[{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}]")
    print(f"s69-verdict: complete_registered_win="
          f"{win['complete_registered_win']} -> {args.out}")
    return 0


def _write(path: str, payload: dict) -> None:
    out = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
