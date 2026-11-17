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
import glob
import hashlib
import json
import os
import re
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

# S53 registered thresholds for "still paying off". The receipt prints them, so a
# reader cannot mistake a measured curve for a chosen conclusion.
FIXED_BATCH_THRESHOLDS = {
    "ce_slope_per_million_tokens": "< 0.0",
    "ce_last_interval": "> 0.0",
    "ce_last_share": ">= 0.10",
    "kl_last_interval": ">= 0.0",
}
CE_LAST_SHARE_MIN = 0.10


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


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _checkpoint_step(label: str) -> int | None:
    if label == "init":
        return None
    match = re.fullmatch(r"ckpt-(\d+)\.pt", label)
    if match is None:
        raise SystemExit(
            f"s50-bar-curve: unexpected checkpoint label {label!r} in the "
            "fixed-batch receipt")
    return int(match.group(1))


def _fixed_batch_trend(receipt_path: str, run_dir: str,
                       tokens_per_step: int) -> dict:
    """Validate and fit the fixed-batch curve over every saved checkpoint."""
    payload = json.load(open(receipt_path, encoding="utf-8"))
    if payload.get("schema") != "selvedge/s50-fixed-batch-kl/v1":
        raise SystemExit("s50-bar-curve: not a fixed-batch KL receipt")
    if payload.get("seq_len") is None or payload.get("batch_count") is None:
        raise SystemExit("s50-bar-curve: fixed-batch receipt is missing its basis")
    results = {entry["checkpoint"]: entry for entry in payload["results"]}
    if "init" not in results:
        raise SystemExit("s50-bar-curve: fixed-batch receipt has no init point")
    directory = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    checkpoint_paths = sorted(glob.glob(os.path.join(directory, "ckpt-*.pt")))
    if len(checkpoint_paths) < 2:
        raise SystemExit(
            "s50-bar-curve: the fixed-batch rule needs at least two saved "
            f"checkpoints, found {len(checkpoint_paths)} in {run_dir}")
    points = []
    for path in checkpoint_paths:
        label = os.path.basename(path)
        if label not in results:
            raise SystemExit(
                f"s50-bar-curve: {label} has no fixed-batch measurement")
        entry = results[label]
        if entry.get("sha256") != _sha256(path):
            raise SystemExit(
                f"s50-bar-curve: {label} sha256 does not match the measurement")
        step = _checkpoint_step(label)
        points.append({
            "checkpoint": label,
            "step": step,
            "tokens_seen": step * tokens_per_step,
            "sha256": entry["sha256"],
            "ce": float(entry["ce"]),
            "kl": float(entry["kl"]),
            "batches": int(entry["batches"]),
        })
    unexpected = set(results) - {"init"} - {point["checkpoint"] for point in points}
    if unexpected:
        raise SystemExit(
            "s50-bar-curve: fixed-batch receipt measures checkpoints that are no "
            f"longer in the run directory: {sorted(unexpected)}")
    ce_slope = _slope([(point["tokens_seen"], point["ce"]) for point in points])
    kl_slope = _slope([(point["tokens_seen"], point["kl"]) for point in points])
    first, last = points[0], points[-1]
    previous = points[-2]
    total_drop = first["ce"] - last["ce"]
    last_drop = previous["ce"] - last["ce"]
    last_share = (last_drop / total_drop
                  if total_drop > 0.0 else None)
    return {
        "receipt": receipt_path,
        "receipt_sha256": _sha256(receipt_path),
        "run_dir": run_dir,
        "batch_count": int(payload["batch_count"]),
        "seq_len": int(payload["seq_len"]),
        "temperature": payload.get("temperature"),
        "start_token": payload.get("start_token"),
        "init": {"ce": float(results["init"]["ce"]),
                 "kl": float(results["init"]["kl"]),
                 "sha256": results["init"].get("sha256")},
        "checkpoints": points,
        "ce_slope_per_million_tokens": (ce_slope * 1_000_000
                                        if ce_slope is not None else None),
        "kl_slope_per_million_tokens": (kl_slope * 1_000_000
                                        if kl_slope is not None else None),
        "ce_last_interval": last_drop,
        "kl_last_interval": previous["kl"] - last["kl"],
        "ce_total_drop_after_first_checkpoint": total_drop,
        "ce_last_share": last_share,
    }


def _extension_decision(fixed: dict, final_math: float,
                        previous_math: float, previous_error: float,
                        base_math: float) -> dict:
    """The registered S53 rule, factored so a CPU test can exercise it."""
    still_paying = bool(
        fixed["ce_slope_per_million_tokens"] is not None
        and fixed["ce_slope_per_million_tokens"] < 0.0
        and fixed["ce_last_interval"] > 0.0
        and fixed["ce_last_share"] is not None
        and fixed["ce_last_share"] >= CE_LAST_SHARE_MIN
        and fixed["kl_last_interval"] >= 0.0)
    harness_confirms = bool(final_math >= previous_math - previous_error
                            and final_math > base_math)
    return {"still_paying_off": still_paying,
            "harness_confirms": harness_confirms,
            "justified": still_paying and harness_confirms}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-dir", default=RUN_DIR,
                        help="training run whose log is read (for a rate rerun)")
    parser.add_argument("--eval-root", default=EVAL_ROOT,
                        help="milestone receipt root matching --run-dir")
    parser.add_argument("--fixed-batch-receipt", default=None,
                        help="selvedge/s50-fixed-batch-kl/v1 receipt over "
                             "every saved checkpoint; required by the S53 rule")
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
    if not args.fixed_batch_receipt:
        raise SystemExit(
            "s50-bar-curve: --fixed-batch-receipt is required by the S53 "
            "decision rule. Produce it with scripts/s50_fixed_batch_kl.py "
            "over this run directory first; the harness-only OLS rule is "
            "superseded and is not used for the decision.")
    run_manifest = json.load(open(os.path.join(ROOT, run_dir, "manifest.json"),
                                  encoding="utf-8"))
    tokens_per_step = run_manifest.get("tokens_per_step")
    if tokens_per_step is None:
        tokens_per_step = run_manifest["config"]["tokens_per_step"]
    fixed = _fixed_batch_trend(_resolve(args.fixed_batch_receipt), run_dir,
                               int(tokens_per_step))
    base_math_mean = _family(base_rows, MATH)["mean"]
    final_math = milestones[-1]["math"]["mean"]
    previous_math = milestones[-2]["math"]["mean"]
    previous_error = milestones[-2]["math"]["combined_stderr"]
    decision = _extension_decision(fixed, final_math, previous_math,
                                   previous_error, base_math_mean)
    still_paying = decision["still_paying_off"]
    harness_confirms = decision["harness_confirms"]
    justified = decision["justified"]

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
                  "math_10m_to_20m_increment": increment_10_20,
                  "fixed_batch": fixed},
        "extension": {
            "rule": ("100M extension justified iff the fixed-batch CE slope "
                     "over every saved checkpoint is negative, the last "
                     "interval still falls, the last interval carries at "
                     "least 10% of the total CE drop, the teacher KL does not "
                     "rise, and the harness endpoints do not contradict the "
                     "fixed-batch reading"),
            "thresholds": FIXED_BATCH_THRESHOLDS,
            "still_paying_off": still_paying,
            "harness_confirms": harness_confirms,
            "harness_condition": (
                "final math family >= previous milestone math family - its "
                "combined standard error AND final math family > base"),
            "harness_endpoints": {
                "final_math_family": final_math,
                "previous_math_family": previous_math,
                "previous_combined_stderr": previous_error,
                "base_math_family": base_math_mean,
            },
            "justified": justified,
            "superseded_harness_points_rule": {
                "math_family_slope_per_million_tokens": (
                    math_slope * 1_000_000 if math_slope is not None else None),
                "math_10m_to_20m_increment": increment_10_20,
                "role": "reported for audit only; it does not decide the "
                        "extension",
            },
            "reading": ("the 100M extension is justified by this bar"
                        if justified else
                        "the 100M extension is not justified by this bar; the "
                        "fixed-batch trend and the harness endpoints are "
                        "reported instead"),
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
    share = fixed["ce_last_share"]
    share_text = f"{share:.4f}" if share is not None else "no total drop"
    print(f"s50-bar-curve: fixed-batch CE slope per 1M tokens "
          f"{fixed['ce_slope_per_million_tokens']:+.6f}; KL slope "
          f"{fixed['kl_slope_per_million_tokens']:+.6f}; last CE interval "
          f"{fixed['ce_last_interval']:+.6f} (share {share_text}); "
          f"still_paying_off={still_paying}; harness_confirms="
          f"{harness_confirms}; extension justified={justified}")
    print(f"s50-bar-curve (superseded harness-only rule): math slope per 1M "
          f"tokens {receipt['curve']['math_family_slope_per_million_tokens']:+.6f}; "
          f"10M->20M increment "
          f"{receipt['curve']['math_10m_to_20m_increment']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
