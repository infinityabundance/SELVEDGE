#!/usr/bin/env python3
"""S50: the corrected bar's milestone curve, scores and extension decision.

Reads the three anchored milestone receipts (m5 / m10 / m20), the anchored base
receipt and the training log; emits one JSON receipt with per-task scores and
standard errors, the knowledge and math family means, the S47 sequence-
correctness counts for GSM8K and MATH-500, the ordinary-least-squares math-family
slope over tokens and the registered 100M-extension decision:

    extension justified iff the math-family slope is positive AND the
    10M-to-20M math-family increment is positive.

This is the same rule the S49 registration states; the S50 bar changes only the
optimiser and the learning rate. Nothing is typed by hand: every number comes
from the receipts named in the output.

Usage:
    scripts/court python -- python3 scripts/s50_bar_curve.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUN_DIR = "results/stage2/runs/s50-bar-20m-lr1e-5"
EVAL_ROOT = "results/stage2/eval/s50-bar-20m-lr1e-5"
BASE_RECEIPT = "results/stage2/eval/base/eval.json"
TASKS = ("mmlu", "mmlu_pro", "gsm8k", "minerva_math500")
KNOWLEDGE = ("mmlu", "mmlu_pro")
MATH = ("gsm8k", "minerva_math500")
MILESTONES = (
    ("m5", 2441, 4_999_168),
    ("m10", 4882, 9_998_336),
    ("m20", 9765, 19_998_720),
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tasks(payload: dict) -> dict:
    rows = {}
    for entry in payload["tasks"]:
        task = entry["task"]
        if task not in TASKS:
            continue
        sequence = entry.get("sequence_correctness") or {}
        rows[task] = {
            "score": float(entry["score"]),
            "stderr": float(entry["stderr"]),
            "primary_metric": entry["primary_metric"],
            "samples": entry.get("n_samples"),
            "wall_clock_s": entry.get("wall_clock_s"),
            "cgroup_memory_peak": (entry.get("cgroup") or {}).get("memory_peak"),
            "sequence_correctness": ({
                "accuracy": sequence.get("accuracy"),
                "accuracy_answered": sequence.get("accuracy_answered"),
                "answered": sequence.get("answered"),
                "items": sequence.get("items"),
                "primary_filter": sequence.get("primary_filter"),
            } if sequence else None),
        }
    missing = set(TASKS) - set(rows)
    if missing:
        raise SystemExit(f"s50-bar-curve: receipt is missing tasks {sorted(missing)}")
    return rows


def _family(rows: dict, names: tuple[str, ...]) -> dict:
    values = [rows[name]["score"] for name in names]
    errors = [rows[name]["stderr"] for name in names]
    return {
        "tasks": list(names),
        "mean": sum(values) / len(values),
        "combined_stderr": sum(error ** 2 for error in errors) ** 0.5 / len(errors),
        "per_task": {name: {"score": rows[name]["score"],
                            "stderr": rows[name]["stderr"]} for name in names},
    }


def _slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-dir", default=RUN_DIR,
                        help="training run whose log is read (for a rate rerun)")
    parser.add_argument("--eval-root", default=EVAL_ROOT,
                        help="milestone receipt root matching --run-dir")
    args = parser.parse_args()
    run_dir = args.run_dir
    eval_root = args.eval_root

    base_payload = json.load(open(os.path.join(ROOT, BASE_RECEIPT), encoding="utf-8"))
    base_rows = _tasks(base_payload)
    points = []
    milestones = []
    for name, step, tokens in MILESTONES:
        path = os.path.join(ROOT, eval_root, name, "eval.json")
        if not os.path.isfile(path):
            raise SystemExit(f"s50-bar-curve: missing milestone receipt {path}")
        payload = json.load(open(path, encoding="utf-8"))
        rows = _tasks(payload)
        knowledge = _family(rows, KNOWLEDGE)
        math_family = _family(rows, MATH)
        points.append((tokens, math_family["mean"]))
        milestones.append({
            "name": name,
            "step": step,
            "tokens_seen": tokens,
            "receipt": os.path.join(eval_root, name, "eval.json"),
            "receipt_sha256": _sha256(path),
            "checkpoint_sha256": (payload.get("arm_report") or {}
                                  ).get("checkpoint_sha256"),
            "recorded_checkpoint_step": (payload.get("arm_report") or {}
                                         ).get("checkpoint_step"),
            "tasks": rows,
            "knowledge": knowledge,
            "math": math_family,
            "delta_vs_base": {
                task: rows[task]["score"] - base_rows[task]["score"]
                for task in TASKS},
            "family_delta_vs_base": {
                "knowledge": knowledge["mean"] - _family(base_rows, KNOWLEDGE)["mean"],
                "math": math_family["mean"] - _family(base_rows, MATH)["mean"],
            },
        })

    math_slope = _slope(points)
    increment_10_20 = (milestones[2]["math"]["mean"]
                       - milestones[1]["math"]["mean"])
    knowledge_slope = _slope([(milestone["tokens_seen"],
                               milestone["knowledge"]["mean"])
                              for milestone in milestones])
    justified = bool(math_slope is not None and math_slope > 0.0
                     and increment_10_20 > 0.0)

    log_path = os.path.join(ROOT, run_dir, "log.jsonl")
    log_rows = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    update_values = [row["max_update_over_lr"] for row in log_rows
                     if "max_update_over_lr" in row]
    receipt = {
        "schema": "selvedge/s50-bar-curve/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _revision(),
        "run_dir": run_dir,
        "base_receipt": BASE_RECEIPT,
        "base_receipt_sha256": _sha256(os.path.join(ROOT, BASE_RECEIPT)),
        "base": {"tasks": base_rows,
                 "knowledge": _family(base_rows, KNOWLEDGE),
                 "math": _family(base_rows, MATH)},
        "milestones": milestones,
        "curve": {"math_family": [{"tokens_seen": tokens, "mean": mean}
                                  for tokens, mean in points],
                  "math_family_slope_per_million_tokens": (
                      math_slope * 1_000_000 if math_slope is not None else None),
                  "knowledge_family_slope_per_million_tokens": (
                      knowledge_slope * 1_000_000
                      if knowledge_slope is not None else None),
                  "math_10m_to_20m_increment": increment_10_20},
        "extension": {
            "rule": ("100M extension justified iff the math-family slope over "
                     "tokens is positive and the 10M-to-20M math-family "
                     "increment is positive"),
            "justified": justified,
            "reading": ("the 100M extension is justified by this bar"
                        if justified else
                        "the 100M extension is not justified by this bar; the "
                        "measured slope and increment are reported instead"),
        },
        "training_log": {
            "path": os.path.join(run_dir, "log.jsonl"),
            "sha256": _sha256(log_path),
            "steps": len(log_rows),
            "max_update_over_lr": max(update_values) if update_values else None,
            "median_update_over_lr": (
                sorted(update_values)[len(update_values) // 2]
                if update_values else None),
            "final_knowledge_step": log_rows[-1]["step"] if log_rows else None,
        },
        "command": os.environ.get("SELVEDGE_S50_CURVE_COMMAND"),
    }
    output = args.out or os.path.join(
        ROOT, f"results/stage2/s50-bar-curve-{receipt['stamp']}.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    relative = os.path.relpath(output, ROOT)
    print(f"s50-bar-curve: {relative}")
    print(f"{'milestone':>9} {'tokens':>12} {'MMLU':>7} {'MMLU-Pro':>9} "
          f"{'GSM8K':>7} {'MATH-500':>9} {'knowledge':>10} {'math':>7} "
          f"{'seq GSM8K':>10} {'seq MATH':>10}")
    for milestone in milestones:
        tasks = milestone["tasks"]
        def seq(task):
            value = tasks[task]["sequence_correctness"] or {}
            if value.get("accuracy") is None:
                return "-"
            return f"{value['accuracy']:.4f}"
        print(f"{milestone['name']:>9} {milestone['tokens_seen']:>12} "
              f"{tasks['mmlu']['score']:>7.4f} "
              f"{tasks['mmlu_pro']['score']:>9.4f} "
              f"{tasks['gsm8k']['score']:>7.4f} "
              f"{tasks['minerva_math500']['score']:>9.4f} "
              f"{milestone['knowledge']['mean']:>10.4f} "
              f"{milestone['math']['mean']:>7.4f} "
              f"{seq('gsm8k'):>10} {seq('minerva_math500'):>10}")
    print(f"s50-bar-curve: math slope per 1M tokens "
          f"{receipt['curve']['math_family_slope_per_million_tokens']:+.6f}; "
          f"10M->20M increment "
          f"{receipt['curve']['math_10m_to_20m_increment']:+.6f}; "
          f"extension justified={justified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
