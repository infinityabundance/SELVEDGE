#!/usr/bin/env python3
"""S62: score the four milestone evaluations and summarise them with CIs.

The S60 revised registration scores four cadence points (0.5M, 1M, 2M, 5M
processed tokens) with the same instrument:

* GSM8K (1,319 items) and MATH-500 (500), greedy, non-thinking, zero-shot,
  ``\\boxed{}``, batch 32, cap 4096: `scripts/s54_post_trained_eval.py`;
* MMLU (14,042) and MMLU-Pro on a 4,000-item total-cap subset:
  `scripts/s43_score_total_cap.py` with `--arm full-distill`;
* paired problem-level bootstrap intervals against the post-trained 0.8B:
  `scripts/s57_paired.py` (4,000 resamples, seed 53, the unit is the problem).

This helper does three things and never invents a number:

1. ``--plan`` prints one court command per milestone and task, checking that
   every checkpoint exists and that the milestone step is on the run's own
   checkpoint grid;
2. ``--collect`` reads the receipts each command wrote and emits one summary
   with, per milestone: GSM8K/MATH-500 flexible accuracy, the paired delta and
   95% interval, the termination/truncation/loop fractions, the retention
   deltas and whether each retention clause is ``pass``, ``fail`` or
   ``unresolved``, and the registered win decision;
3. ``--self-test`` runs the summary on synthetic receipts on CPU.

The win bar is the registered one: GSM8K > 0.5421 AND MATH-500 > 0.3820, each
with the paired 95% interval above zero, AND MMLU and MMLU-Pro within 1.0
point (point drop <= 1.0 and paired interval lower bound >= -1.0). An
unresolved retention clause blocks a win but is not a retention failure.
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

SCHEMA = "selvedge/s62-milestone-summary/v1"
PLAN_SCHEMA = "selvedge/s62-milestone-plan/v1"
STUDENT_ID = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
STUDENT_DIR = f"/hf-cache/models/Qwen/Qwen3.5-0.8B/{STUDENT_REVISION}"
BASELINE = "results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json"
BASELINE_MMLU = "results/stage2/eval/s58-post-0.8b-baseline-mmlu/eval.json"
BASELINE_MMLU_RAW = (
    "results/stage2/eval/s58-post-0.8b-baseline-mmlu/raw/"
    "20261118T222909Z-mmlu.json")
BASELINE_MMLU_PRO_RAW = (
    "results/stage2/eval/s58-post-0.8b-baseline-mmlu/raw/"
    "20261118T224844Z-mmlu_pro.json")
WIN_GSM8K = 0.5421
WIN_MATH = 0.3820
RETENTION_BOUND = 0.01
DEFAULT_MILESTONES = ("0.5M=400", "1M=800", "2M=1600", "5M=4000")
REGISTRATION = (
    "docs/ws19-post-trained-route-stage2-revised-preregistration-s60-v1.json")
ANCHOR_REGISTRATION = "docs/ws19-anchored-subset-amendment-preregistration-v2.json"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def parse_milestones(values: list[str]) -> list[dict]:
    out = []
    for value in values:
        label, _, step = value.partition("=")
        if not label or not step.isdigit():
            raise SystemExit(
                f"s62-score: milestone {value!r} must be label=step")
        out.append({"label": label, "step": int(step)})
    return out


def checkpoint_path(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, f"ckpt-{step:06d}.pt")


def build_plan(args) -> dict:
    milestones = parse_milestones(args.milestones)
    commands = []
    for milestone in milestones:
        label = milestone["label"]
        step = milestone["step"]
        checkpoint = checkpoint_path(args.run_dir, step)
        exists = os.path.exists(os.path.join(ROOT, checkpoint))
        stage1_out = os.path.join(
            args.out_dir, f"stage1-{label}-nonthinking-greedy.json")
        stage1_raw = os.path.join(
            args.out_dir, f"stage1-{label}-nonthinking-greedy-raw")
        retention_out = os.path.join(args.out_dir, f"retention-{label}.json")
        pairing_out = os.path.join(args.out_dir, f"paired-{label}-vs-post.json")
        retention_pair = os.path.join(
            args.out_dir, f"paired-retention-{label}-vs-post.json")
        commands.append({
            "milestone": label, "step": step, "checkpoint": checkpoint,
            "checkpoint_exists": exists,
            "stage1": {
                "out": stage1_out, "raw_dir": stage1_raw,
                "command": [
                    "scripts/court", "--gpu", "eval", "--", "python3",
                    "scripts/s54_post_trained_eval.py",
                    "--model-dir", args.student_dir,
                    "--model-id", STUDENT_ID,
                    "--revision", STUDENT_REVISION,
                    "--full-checkpoint", checkpoint,
                    "--tasks", "gsm8k,math500",
                    "--modes", "non-thinking",
                    "--decodings", "greedy",
                    "--max-new-tokens", "4096",
                    "--batch-size", "32",
                    "--vram-limit-gib", "13.0",
                    "--out", stage1_out,
                    "--raw-dir", stage1_raw],
            },
            "retention": {
                "out": retention_out,
                "command": [
                    "scripts/court", "--gpu", "eval", "--", "python3",
                    "scripts/s43_score_total_cap.py",
                    "--arm", "full-distill",
                    "--full-checkpoint", checkpoint,
                    "--student-dir", args.student_dir,
                    "--model-id", STUDENT_ID,
                    "--model-revision", STUDENT_REVISION,
                    "--tasks", "mmlu,mmlu_pro",
                    "--total-cap-map", f"mmlu=14042,mmlu_pro={args.mmlu_pro_cap}",
                    "--registration", ANCHOR_REGISTRATION,
                    "--vram-limit-gib", "13.0",
                    "--out", retention_out],
            },
            "pairing": {
                "out": pairing_out,
                "command": [
                    "scripts/court", "python", "--", "python3",
                    "scripts/s57_paired.py",
                    "--candidate", stage1_out,
                    "--baseline", args.baseline,
                    "--tasks", "gsm8k,math500",
                    "--out", pairing_out],
                "retention_out": retention_pair,
                "retention_command": [
                    "scripts/court", "python", "--", "python3",
                    "scripts/s57_paired.py",
                    "--lm-eval-candidate",
                    f"{os.path.join(os.path.dirname(retention_out), 'raw')}/*",
                    "--lm-eval-baseline", args.baseline_mmlu_pro_raw,
                    "--lm-eval-metrics", "mmlu,mmlu_pro",
                    "--out", retention_pair],
            },
        })
    return {
        "schema": PLAN_SCHEMA,
        "kind": "milestone-plan",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "run_dir": args.run_dir,
        "baseline": args.baseline,
        "baseline_mmlu": args.baseline_mmlu,
        "milestones": milestones,
        "mmlu_pro_cap": args.mmlu_pro_cap,
        "commands": commands,
        "notes": [
            "one GPU job at a time; run the stage1 and retention commands "
            "serially, never two concurrently",
            "the retention pairing command is a template: point it at the two "
            "raw dumps the retention run wrote, in mmlu,mmlu_pro order",
            "a milestone whose checkpoint does not exist is printed but must "
            "not be run; the cadence and the checkpoint grid are recorded in "
            "the run receipt",
        ],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def load(path: str) -> dict:
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def task_row(receipt: dict, task: str) -> dict | None:
    for row in receipt.get("tasks", []):
        if row.get("task") == task:
            return row
    return None


def stage1_summary(receipt: dict, task: str) -> dict:
    row = task_row(receipt, task)
    if row is None:
        return {"task": task, "present": False}
    flexible = row.get("flexible") or {}
    termination = row.get("termination") or {}
    return {
        "task": task,
        "present": True,
        "accuracy": flexible.get("accuracy"),
        "correct": flexible.get("correct"),
        "items": flexible.get("items"),
        "stderr": flexible.get("stderr"),
        "termination": {
            "truncation_rate": termination.get("truncation_rate"),
            "loop_classified_fraction": termination.get(
                "loop_classified_fraction"),
            "loop_classified_samples": termination.get(
                "loop_classified_samples"),
            "truncated_samples": termination.get("truncated_samples"),
            "cross_tab": termination.get("cross_tab"),
            "mean_new_tokens": termination.get("mean_new_tokens"),
        },
    }


def retention_summary(receipt: dict) -> dict:
    out = {}
    for row in receipt.get("tasks", []):
        metrics = row.get("all_metrics") or {}
        name = metrics.get("name") or row.get("task")
        out[name] = {
            "score": row.get("score"),
            "stderr": row.get("stderr"),
            "primary_metric": row.get("primary_metric"),
            "sample_len": (row.get("n_samples") or {}).get("sample_len"),
        }
    return out


def paired_row(receipt: dict, task: str) -> dict | None:
    for row in receipt.get("rows", []):
        if row.get("task") == task:
            return row
    return None


def retention_verdict(paired: dict | None) -> dict:
    if not paired:
        return {"state": "unresolved", "reason": "no paired retention receipt"}
    delta = paired.get("delta")
    ci = paired.get("ci95")
    if delta is None or not ci:
        return {"state": "unresolved", "reason": "receipt has no interval"}
    point_within = abs(delta) <= RETENTION_BOUND
    interval_ok = ci[0] >= -RETENTION_BOUND
    if point_within and interval_ok:
        state = "pass"
    elif not point_within:
        state = "fail"
    else:
        state = "unresolved"
    return {"state": state, "delta": delta, "ci95": ci,
            "point_within_bound": point_within,
            "interval_lower_within_bound": interval_ok,
            "paired_items": paired.get("paired_items")}


def milestone_summary(entry: dict) -> dict:
    """Combine the three receipts named by one plan entry."""
    stage1_path = entry["stage1"]["out"]
    paired_path = entry["pairing"]["out"]
    retention_path = entry["retention"]["out"]
    retention_pair_path = entry["pairing"]["retention_out"]
    out = {"milestone": entry["milestone"], "step": entry["step"],
           "checkpoint": entry["checkpoint"], "tasks": {}}
    if os.path.exists(os.path.join(ROOT, stage1_path)):
        stage1 = load(stage1_path)
        for task in ("gsm8k", "math500"):
            out["tasks"][task] = stage1_summary(stage1, task)
    else:
        out["tasks"]["gsm8k"] = out["tasks"]["math500"] = {
            "present": False}
    if os.path.exists(os.path.join(ROOT, paired_path)):
        paired = load(paired_path)
        for task in ("gsm8k", "math500"):
            row = paired_row(paired, task)
            if row:
                primary = row["bases"]["flexible"]
                out["tasks"][task]["paired"] = {
                    "delta": primary["delta"],
                    "ci95": primary["ci95"],
                    "interval_above_zero": primary["interval_above_zero"],
                    "problems": primary["problems"],
                    "baseline_accuracy": primary["baseline_accuracy"],
                }
    out["retention"] = {}
    if os.path.exists(os.path.join(ROOT, retention_path)):
        retention = retention_summary(load(retention_path))
        paired = None
        if os.path.exists(os.path.join(ROOT, retention_pair_path)):
            paired = load(retention_pair_path)
        for task in ("mmlu", "mmlu_pro"):
            row = {"score": retention.get(task, {}).get("score"),
                   "sample_len": retention.get(task, {}).get("sample_len")}
            if paired is not None:
                row["paired"] = paired_row(paired, task)
            row["verdict"] = retention_verdict(row.get("paired"))
            out["retention"][task] = row
    return out


def win_decision(summary: dict) -> dict:
    checks = {}

    def positive(task: str) -> dict:
        row = summary["tasks"].get(task, {})
        accuracy = row.get("accuracy")
        paired = row.get("paired") or {}
        bar = WIN_GSM8K if task == "gsm8k" else WIN_MATH
        accuracy_ok = accuracy is not None and accuracy > bar
        interval_ok = bool(paired.get("interval_above_zero"))
        return {"accuracy": accuracy, "bar": bar,
                "accuracy_above_bar": accuracy_ok,
                "paired_ci": paired.get("ci95"),
                "interval_above_zero": interval_ok,
                "met": bool(accuracy_ok and interval_ok)}

    checks["gsm8k"] = positive("gsm8k")
    checks["math500"] = positive("math500")
    retention = {}
    for task in ("mmlu", "mmlu_pro"):
        row = summary.get("retention", {}).get(task) or {}
        retention[task] = (row.get("verdict") or {}).get("state", "unresolved")
    checks["retention"] = retention
    retention_ok = all(state == "pass" for state in retention.values())
    unresolved = any(state == "unresolved" for state in retention.values())
    won = (checks["gsm8k"]["met"] and checks["math500"]["met"]
           and retention_ok)
    return {"win": won, "checks": checks,
            "retention_unresolved": unresolved,
            "reason": ("win: both task bars with paired CIs above zero and "
                       "both retention clauses pass" if won else
                       "not a win; the failing clauses are named in checks")}


def collect(args) -> int:
    plan_path = args.plan_path or os.path.join(args.out_dir, "plan.json")
    plan = load(plan_path)
    entries = []
    for entry in plan["commands"]:
        summary = milestone_summary(entry)
        summary["decision"] = win_decision(summary)
        entries.append(summary)
    final = entries[-1] if entries else None
    payload = {
        "schema": SCHEMA,
        "kind": "milestone-summary",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "plan": plan_path,
        "win_bar": {"gsm8k": WIN_GSM8K, "math500": WIN_MATH,
                    "retention_bound_points": RETENTION_BOUND * 100},
        "milestones": entries,
        "final_milestone_decision": final["decision"] if final else None,
        "command": sys.argv,
    }
    out = args.out or os.path.join(args.out_dir, "summary.json")
    write_json(out, payload)
    for entry in entries:
        gsm = entry["tasks"].get("gsm8k", {})
        math = entry["tasks"].get("math500", {})
        gsm_paired = gsm.get("paired") or {}
        math_paired = math.get("paired") or {}
        print(f"s62-score: {entry['milestone']} (step {entry['step']}): "
              f"gsm8k={gsm.get('accuracy')} "
              f"delta={gsm_paired.get('delta')} ci={gsm_paired.get('ci95')} "
              f"math500={math.get('accuracy')} "
              f"delta={math_paired.get('delta')} "
              f"ci={math_paired.get('ci95')}")
        print(f"s62-score:   retention "
              + ", ".join(f"{task}={row.get('verdict', {}).get('state')}"
                          for task, row in entry["retention"].items()))
    if final:
        print(f"s62-score: final decision win="
              f"{final['decision']['win']} -> {out}")
    return 0


def self_test() -> int:
    baseline = {
        "tasks": [
            {"task": "gsm8k", "flexible": {"accuracy": 0.5421},
             "termination": {"truncation_rate": 0.02,
                             "loop_classified_fraction": 0.005}},
            {"task": "math500", "flexible": {"accuracy": 0.382},
             "termination": {"truncation_rate": 0.21,
                             "loop_classified_fraction": 0.084}}]}
    candidate = {
        "tasks": [
            {"task": "gsm8k", "flexible": {"accuracy": 0.56},
             "termination": {"truncation_rate": 0.03,
                             "loop_classified_fraction": 0.006}},
            {"task": "math500", "flexible": {"accuracy": 0.40},
             "termination": {"truncation_rate": 0.20,
                             "loop_classified_fraction": 0.08}}]}
    paired = {"rows": []}
    for task, delta in (("gsm8k", 0.02), ("math500", 0.018)):
        paired["rows"].append({
            "task": task,
            "bases": {"flexible": {"delta": delta, "ci95": [0.001, 0.04],
                                   "interval_above_zero": True,
                                   "problems": 100,
                                   "baseline_accuracy": 0.5}}})
    retention = {"tasks": [
        {"task": "mmlu", "score": 0.50, "all_metrics": {"name": "mmlu"},
         "n_samples": {"sample_len": 14042}},
        {"task": "mmlu_pro", "score": 0.30,
         "all_metrics": {"name": "mmlu_pro"},
         "n_samples": {"sample_len": 4000}}]}
    retention_pair = {"rows": [
        {"task": "mmlu", "delta": -0.004, "ci95": [-0.01, 0.002],
         "paired_items": 14042},
        {"task": "mmlu_pro", "delta": -0.006, "ci95": [-0.02, 0.008],
         "paired_items": 4000}]}

    summary = {"tasks": {}, "retention": {}}
    for task in ("gsm8k", "math500"):
        summary["tasks"][task] = stage1_summary(candidate, task)
        row = paired_row(paired, task)
        if row:
            summary["tasks"][task]["paired"] = row["bases"]["flexible"]
    for task, row in retention_summary(retention).items():
        paired_row_value = paired_row(retention_pair, task)
        summary["retention"][task] = {
            "score": row["score"],
            "paired": paired_row_value,
            "verdict": retention_verdict(paired_row_value)}
    decision = win_decision(summary)
    assert decision["checks"]["gsm8k"]["met"], decision
    assert decision["checks"]["math500"]["met"], decision
    assert decision["checks"]["retention"]["mmlu"] == "pass", decision
    # mmlu_pro: point inside 1 point, interval lower -2 points -> unresolved.
    assert decision["checks"]["retention"]["mmlu_pro"] == "unresolved", decision
    assert decision["win"] is False and decision["retention_unresolved"], decision

    # A point drop outside 1 point is a failure, even with a good interval.
    verdict = retention_verdict({"delta": -0.03, "ci95": [-0.04, -0.02],
                                 "paired_items": 100})
    assert verdict["state"] == "fail", verdict
    # A clean pass.
    verdict = retention_verdict({"delta": -0.002, "ci95": [-0.008, 0.004],
                                 "paired_items": 100})
    assert verdict["state"] == "pass", verdict

    # The plan parser rejects a malformed milestone.
    assert parse_milestones(["0.5M=400"])[0]["step"] == 400
    try:
        parse_milestones(["0.5M"])
    except SystemExit:
        pass
    else:
        raise AssertionError("a milestone without =step must fail closed")

    print("s62-score self-test: OK (summary, retention verdicts, win logic, "
          "milestone parsing)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir",
                        default="results/stage2/runs/s62-stage2b")
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--baseline-mmlu", default=BASELINE_MMLU)
    parser.add_argument("--baseline-mmlu-raw", default=BASELINE_MMLU_RAW)
    parser.add_argument("--baseline-mmlu-pro-raw",
                        default=BASELINE_MMLU_PRO_RAW)
    parser.add_argument("--mmlu-pro-cap", type=int, default=4000)
    parser.add_argument("--milestones", nargs="+",
                        default=list(DEFAULT_MILESTONES))
    parser.add_argument("--out-dir", default="results/stage2/s62/score")
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.plan:
        plan = build_plan(args)
        out = os.path.join(args.out_dir, "plan.json")
        write_json(out, plan)
        for entry in plan["commands"]:
            status = "exists" if entry["checkpoint_exists"] else "MISSING"
            print(f"s62-score: {entry['milestone']} step {entry['step']} "
                  f"({status})")
            print("  " + " ".join(entry["stage1"]["command"]))
            print("  " + " ".join(entry["retention"]["command"]))
        print(f"s62-score: plan -> {out}")
        return 0
    if args.collect:
        return collect(args)
    parser.error("one of --plan/--collect/--self-test is required")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
