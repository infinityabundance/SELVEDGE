#!/usr/bin/env python3
"""S71: the registered overshoot probe's statistic and verdict.

Reads the calibration receipts produced by
``scripts/s69_milestone_calibration.py score`` and computes, for every
arm/cadence and each task separately:

* the flexible accuracy and its correct count,
* the paired problem-level bootstrap of the delta against the S67 soup,
* the paired bootstrap of the delta against that arm's own 2M endpoint.

The statistic is ``s54_post_trained_eval.paired_prompt_bootstrap`` with its
registered defaults (4000 resamples, seed 53, resampling unit the problem).

The verdict block evaluates the falsification condition in
``docs/s71-overshoot-probe-preregistration-v1.json`` verbatim: the hypothesis is
falsified when every earlier cadence is worse than its 2M endpoint on MATH-cal.

No test split is read, nothing is selected, nothing is adopted.

Usage
-----
    SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
      scripts/court python -- python3 scripts/s71_overshoot_analysis.py \
        --out results/stage2/s71-overshoot/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from s54_post_trained_eval import (  # noqa: E402
    paired_prompt_bootstrap,
    revision,
    sha256_file,
    stamp,
    utcnow,
    write_json,
)

SCHEMA = "selvedge/s71-overshoot-analysis/v1"
REGISTRATION = "docs/s71-overshoot-probe-preregistration-v1.json"
SOUP = "results/stage2/s71-overshoot/soup-calibration.json"
ARMS = {
    "matched-sft": "results/stage2/s69-calibration/matched-sft.json",
    "matched-sft-weighted":
        "results/stage2/s69-calibration/matched-sft-weighted.json",
    "preference": "results/stage2/s69-calibration/preference.json",
}
TASKS = {"gsm8k": "gsm8k_per_problem", "math500": "math_per_problem"}


def load(path: str) -> dict:
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def rows_by_cadence(payload: dict) -> dict[int, dict]:
    return {int(row["cadence"]): row for row in payload["rows"]}


def per_problem(row: dict, key: str) -> dict[str, bool]:
    return {item["problem_id"]: bool(item["flexible_correct"])
            for item in row[key]}


def paired(arm_flags: dict[str, bool], reference_flags: dict[str, bool],
           arm: str, reference: str) -> dict:
    if set(arm_flags) != set(reference_flags):
        missing = sorted(set(arm_flags) ^ set(reference_flags))[:5]
        raise SystemExit(f"s71: paired problem sets differ (e.g. {missing})")
    rows = [{"correct": {arm: arm_flags[pid], reference: reference_flags[pid]}}
            for pid in sorted(arm_flags)]
    result = paired_prompt_bootstrap(rows, arm, reference)
    result["arm_correct"] = sum(1 for value in arm_flags.values() if value)
    result["reference_correct"] = sum(1 for value in reference_flags.values()
                                      if value)
    result["items"] = len(rows)
    return result


def accuracy(flags: dict[str, bool]) -> dict:
    total = len(flags)
    correct = sum(1 for value in flags.values() if value)
    return {"correct": correct, "items": total,
            "accuracy": (correct / total) if total else None}


def loss_trajectory(trace_path: str, threshold: float) -> dict:
    trace = load(trace_path)
    rows = trace["rows"]
    losses = [float(row["loss"]) for row in rows]
    first_below = next((int(row["step"]) for row in rows
                        if float(row["loss"]) < threshold), None)
    summary = trace.get("summary") or {}
    steps = len(rows)
    pairs = summary.get("unique_pairs")
    beta = 0.1
    final = losses[-1]
    margin = None
    if 0.0 < final < 1.0:
        margin = math.log((1.0 - final) / final) / beta
    return {
        "trace": trace_path,
        "trace_sha256": sha256_file(os.path.join(ROOT, trace_path)),
        "schema": trace.get("schema"),
        "steps": steps,
        "unique_pairs": pairs,
        "epochs_over_pairs": (steps / pairs) if pairs else None,
        "first_step_loss": losses[0],
        "steps_above_threshold": sum(1 for value in losses if value >= threshold),
        "threshold": threshold,
        "first_step_below_threshold": first_below,
        "fraction_steps_below_threshold": (sum(1 for value in losses
                                               if value < threshold) / steps),
        "final_loss": final,
        "min_loss": min(losses),
        "final_implied_margin_beta": margin,
        "margin_formula": ("m = log((1 - loss) / loss) / beta, with the loss "
                           "read as -log sigmoid(beta * m) and beta = 0.1"),
        "clipping_active_steps": summary.get("clipping_active_steps"),
        "clipping_active_fraction": (
            (summary.get("clipping_active_steps") or 0) / steps if steps else None),
        "mean_update_l2": summary.get("mean_update_l2"),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def gradient_summary(trace_path: str, grad_clip: float) -> dict:
    """Pre-clip gradient quantiles recomputed from the tracked trace.

    Quantiles use linear interpolation on the sorted step values, the method
    the S69 telemetry receipt states it used, so the two can be compared.
    """
    trace = load(trace_path)
    rows = trace["rows"]
    pre = [float(row["pre_clip_grad_norm"]) for row in rows]
    post = [float(row["post_clip_grad_norm"]) for row in rows]
    update = [float(row["update_l2"]) for row in rows]
    active = sum(1 for value in pre if value > grad_clip)
    return {
        "trace": trace_path,
        "trace_sha256": sha256_file(os.path.join(ROOT, trace_path)),
        "grad_clip": grad_clip,
        "steps": len(rows),
        "clipping_active_steps": active,
        "clipping_active_fraction": (active / len(rows)) if rows else None,
        "pre_clip_grad_norm": {
            "min": min(pre), "p10": _percentile(pre, 0.10),
            "p25": _percentile(pre, 0.25),
            "median": _percentile(pre, 0.50),
            "p75": _percentile(pre, 0.75),
            "p90": _percentile(pre, 0.90), "max": max(pre),
        },
        "post_clip_grad_norm": {
            "median": _percentile(post, 0.50),
            "p90": _percentile(post, 0.90), "max": max(post),
        },
        "update_l2": {
            "median": _percentile(update, 0.50),
            "p90": _percentile(update, 0.90), "max": max(update),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    soup_payload = load(SOUP)
    soup_rows = rows_by_cadence(soup_payload)
    if len(soup_rows) != 1:
        raise SystemExit(f"s71: {SOUP} must hold exactly one soup row")
    soup_row = next(iter(soup_rows.values()))
    soup = {task: per_problem(soup_row, key) for task, key in TASKS.items()}

    soup_agreement = {}
    for task, key in TASKS.items():
        soup_accuracy = accuracy(soup[task])["accuracy"]
        stored_value = (0.609 if task == "gsm8k" else 0.326)
        soup_agreement[task] = {
            "rescored": soup_accuracy,
            "stored_point_estimate": stored_value,
            "delta": soup_accuracy - stored_value,
            "agrees": abs(soup_accuracy - stored_value) <= 0.001,
        }

    results = {}
    for arm, path in ARMS.items():
        payload = load(path)
        rows = rows_by_cadence(payload)
        entry = {
            "receipt": path,
            "receipt_sha256": sha256_file(os.path.join(ROOT, path)),
            "preregistration": payload.get("preregistration"),
            "cadences": {},
        }
        endpoint = rows.get(2000000)
        for cadence in sorted(rows):
            row = rows[cadence]
            cadence_entry = {
                "checkpoint": row["checkpoint"],
                "checkpoint_step": row["checkpoint_step"],
                "checkpoint_processed_tokens": row.get("checkpoint_processed_tokens"),
                "generated_tokens": row.get("generated_tokens"),
                "wall_seconds": row.get("wall_seconds"),
                "tasks": {},
            }
            for task, key in TASKS.items():
                flags = per_problem(row, key)
                task_entry = accuracy(flags)
                task_entry["vs_soup"] = paired(flags, soup[task], arm, "soup")
                if endpoint is not None and cadence != 2000000:
                    task_entry["vs_own_2m_endpoint"] = paired(
                        flags, per_problem(endpoint, key), arm, "2m")
                elif endpoint is not None:
                    task_entry["vs_own_2m_endpoint"] = None
                cadence_entry["tasks"][task] = task_entry
            entry["cadences"][cadence] = cadence_entry
        results[arm] = entry

    # The registered falsification condition, evaluated on point estimates.
    comparisons = []
    for arm in ("matched-sft", "matched-sft-weighted"):
        cadences = results[arm]["cadences"]
        if 2000000 not in cadences:
            raise SystemExit(f"s71: {arm} has no 2M endpoint row")
        endpoint_math = cadences[2000000]["tasks"]["math500"]["accuracy"]
        for cadence in (500000, 1000000):
            if cadence not in cadences:
                raise SystemExit(f"s71: {arm} has no {cadence} row; the "
                                 "falsification condition cannot be evaluated")
            value = cadences[cadence]["tasks"]["math500"]["accuracy"]
            delta = value - endpoint_math
            comparisons.append({
                "arm": arm,
                "cadence": cadence,
                "math_cal": value,
                "endpoint_math_cal": endpoint_math,
                "delta_vs_endpoint": delta,
                "strictly_better_than_endpoint": delta > 0.0,
                "paired_interval": cadences[cadence]["tasks"]["math500"][
                    "vs_own_2m_endpoint"]["ci95"],
            })
    improved = [row for row in comparisons if row["strictly_better_than_endpoint"]]

    payload = {
        "schema": SCHEMA,
        "kind": "s71-overshoot-analysis",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "label": "diagnostic",
        "test_read": False,
        "adopted": False,
        "preregistration": {
            "path": REGISTRATION,
            "sha256": sha256_file(os.path.join(ROOT, REGISTRATION)),
        },
        "soup": {
            "receipt": SOUP,
            "receipt_sha256": sha256_file(os.path.join(ROOT, SOUP)),
            "checkpoint": soup_row["checkpoint"],
            "checkpoint_sha256": soup_row["checkpoint_sha256"],
            "cadence": soup_row["cadence"],
            "tasks": {task: accuracy(soup[task]) for task in TASKS},
            "agreement_with_stored_point_estimates": soup_agreement,
        },
        "results": results,
        "training": {
            "loss_trajectory": {
                arm: loss_trajectory(
                    f"results/stage2/s69-preference/{arm}/trace-{arm}.json",
                    args.loss_threshold)
                for arm in ("preference", "matched-sft",
                            "matched-sft-weighted", "recovery")
            },
            "gradients": {
                arm: gradient_summary(
                    f"results/stage2/s69-preference/{arm}/trace-{arm}.json",
                    1.0)
                for arm in ("preference", "matched-sft",
                            "matched-sft-weighted", "recovery")
            },
            "telemetry_receipt": "results/stage2/s69-preference/telemetry-summary.json",
            "telemetry_receipt_sha256": sha256_file(os.path.join(
                ROOT, "results/stage2/s69-preference/telemetry-summary.json")),
        },
        "falsification": {
            "condition": ("if every earlier cadence is worse than its 2M "
                          "endpoint, the magnitude hypothesis is not supported "
                          "by this corpus"),
            "comparisons": comparisons,
            "earlier_cadences_strictly_better": len(improved),
            "falsified": len(improved) == 0,
            "evaluated_on": "MATH-cal point estimates on the identical 1,000 problems",
        },
        "claims": {
            "selection_is_calibration_only": True,
            "nothing_adopted": True,
            "no_test_split_read": True,
            "diagnostic_only": True,
        },
    }
    write_json(os.path.join(ROOT, args.out), payload)
    print(json.dumps(payload["falsification"], indent=1))
    print(f"s71: wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
