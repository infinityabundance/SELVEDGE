#!/usr/bin/env python3
"""S49: the rerun bar's milestone curve, its scores, and the slope decision.

The registration
(`docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json`)
fixes the milestones (m5 = 6,144,000 tokens, m10 = 10,240,000, m20 =
19,998,720), the anchored basis and the slope rule. This script reads the three
milestone receipts, the base receipt and the diverged bar's receipt, and emits

* per-task scores and standard errors, the knowledge and math family means, and
  the S47 sequence-correctness counts;
* paired per-item deltas against base on every task, from the harness's own raw
  samples, by the same helper the S44/S48 tables use;
* the ordinary-least-squares math-family slope over the three milestones and
  the 10M-to-20M increment, and the registered recommendation;
* if the bar is stable and improves the math family, the matched comparison of
  the already-trained 2M corrective arms and their two controls against the bar
  on GSM8K and MATH-500.

Nothing is typed by hand. The region it owns (`report-s49-bar-curve` in
`results/REPORT.md`) is declared under `external_regions` in
`docs/perf-numbers.json`; `--check` recomputes it from the inputs and fails
closed when a receipt moved.

Usage:
    scripts/court python -- python3 scripts/s49_bar_curve.py --write
    scripts/court python -- python3 scripts/s49_bar_curve.py --check
    scripts/court python -- python3 scripts/s49_bar_curve.py --summary
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import s44_arm_table as table  # noqa: E402

SCHEMA = "selvedge/s49-bar-curve/v1"
REGISTRATION = "docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
REPORT = "results/REPORT.md"
SENTINEL = "report-s49-bar-curve"
RUN_DIR = "results/stage2/runs/full-20m-math-s49"
EVAL_ROOT = "results/stage2/eval/full-20m-math-s49"
BASE_ARM = "base"
DIVERGED_ARM = "full-20m-math"
MILESTONES = (
    ("m5", 3000, 6_144_000),
    ("m10", 5000, 10_240_000),
    ("m20", 9765, 19_998_720),
)
CORRECTIVE_ARMS = (
    "corrective-kl-2m-math",
    "corrective-signed-exp-2m-math",
    "corrective-signed-true-2m-math",
    "adapter-2m-math-matched",
    "gated-dense-2m-math",
)
TASKS = ("mmlu", "mmlu_pro", "gsm8k", "minerva_math500")
MATH = ("gsm8k", "minerva_math500")
KNOWLEDGE = ("mmlu", "mmlu_pro")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def task_rows(payload: dict) -> dict:
    rows = {}
    for entry in payload.get("tasks") or []:
        name = entry.get("task") or entry.get("all_metrics", {}).get("alias")
        if not name:
            continue
        rows[name] = {
            "score": entry.get("score"),
            "stderr": entry.get("stderr"),
            "primary_metric": entry.get("primary_metric"),
            "sample_len": (entry.get("n_samples") or {}).get("sample_len"),
            "sequence_correct": entry.get("sequence_correctness"),
        }
    return rows


def family_mean(rows: dict, tasks: tuple[str, ...]) -> dict:
    values = [rows[name]["score"] for name in tasks
              if name in rows and rows[name].get("score") is not None]
    errors = [rows[name]["stderr"] for name in tasks
              if name in rows and rows[name].get("stderr") is not None]
    if len(values) != len(tasks):
        return {"measured": False}
    return {
        "measured": True,
        "mean": sum(values) / len(values),
        "combined_stderr": math.sqrt(sum(value ** 2 for value in errors)) / len(values),
    }


def raw_path_for(eval_dir: str, task: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(REPO, eval_dir, "raw", f"*-{task}.json")))
    return matches[-1] if matches else None


def paired_from_paths(arm_path: str | None, ref_path: str | None, task: str) -> dict | None:
    if not arm_path or not ref_path:
        return None
    metric, flt = table.TASKS[task]["primary"]
    try:
        keys_a, values_a, hashes_a = table.item_vector(arm_path, task, metric, flt)
        keys_b, values_b, hashes_b = table.item_vector(ref_path, task, metric, flt)
    except (KeyError, ValueError) as error:
        return {"paired": False, "reason": str(error)}
    if keys_a != keys_b:
        return {"paired": False, "reason": "different item sets or order"}
    if hashes_a != hashes_b:
        return {"paired": False, "reason": "different doc-hash sequences"}
    return table.bootstrap_delta_paired(values_a, values_b)


def slope(points: list[tuple[int, float]]) -> dict | None:
    if len(points) < 2:
        return None
    xs = [float(x) for x, _y in points]
    ys = [float(y) for _x, y in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return {
        "slope_per_million_tokens": (numerator / denominator) * 1_000_000,
        "points": [{"tokens": int(x), "math_family_mean": y}
                   for x, y in points],
    }


def collect() -> dict:
    registration = load_json(os.path.join(REPO, REGISTRATION))
    base_payload = load_json(os.path.join(REPO, "results/stage2/eval/base/eval.json"))
    base_rows = task_rows(base_payload)
    base_dir = "results/stage2/eval/base"

    rows = []
    for name, step, tokens in MILESTONES:
        path = os.path.join(REPO, EVAL_ROOT, name, "eval.json")
        if not os.path.exists(path):
            raise SystemExit(f"s49-curve: missing milestone receipt {path}")
        payload = load_json(path)
        tasks = task_rows(payload)
        paired = {}
        for task in TASKS:
            paired[task] = paired_from_paths(
                raw_path_for(os.path.join(EVAL_ROOT, name), task),
                raw_path_for(base_dir, task), task)
        rows.append({
            "milestone": name,
            "step": step,
            "tokens_seen": tokens,
            "receipt": path,
            "stamp": payload.get("stamp"),
            "revision": payload.get("revision"),
            "tasks": tasks,
            "knowledge": family_mean(tasks, KNOWLEDGE),
            "math": family_mean(tasks, MATH),
            "paired_vs_base": paired,
            "checkpoint_step": (payload.get("arm_report") or {}).get("checkpoint_step"),
            "attention": (payload.get("arm_report") or {}).get("attention")
                         or (payload.get("model") or {}).get("attention"),
        })

    divergent_path = os.path.join(REPO, "results/stage2/eval", DIVERGED_ARM, "eval.json")
    divergent = None
    if os.path.exists(divergent_path):
        payload = load_json(divergent_path)
        tasks = task_rows(payload)
        divergent = {
            "arm": DIVERGED_ARM,
            "receipt": divergent_path,
            "stamp": payload.get("stamp"),
            "tasks": tasks,
            "knowledge": family_mean(tasks, KNOWLEDGE),
            "math": family_mean(tasks, MATH),
        }

    points = [(row["tokens_seen"], row["math"]["mean"]) for row in rows
              if row["math"].get("measured")]
    trend = slope(points)
    increment_10_20 = None
    m10 = next((row for row in rows if row["milestone"] == "m10"), None)
    m20 = next((row for row in rows if row["milestone"] == "m20"), None)
    if m10 and m20 and m10["math"].get("measured") and m20["math"].get("measured"):
        increment_10_20 = m20["math"]["mean"] - m10["math"]["mean"]
    positive = bool(trend and trend["slope_per_million_tokens"] > 0
                    and increment_10_20 is not None and increment_10_20 > 0)
    stable = not os.path.exists(os.path.join(REPO, RUN_DIR, "abort.json"))
    improves = bool(m20 and base_rows and m20["math"].get("measured")
                    and family_mean(base_rows, MATH).get("measured")
                    and m20["math"]["mean"] > family_mean(base_rows, MATH)["mean"])

    corrective = None
    if stable and improves:
        corrective = []
        m20_dir = os.path.join(EVAL_ROOT, "m20")
        for arm in CORRECTIVE_ARMS:
            arm_dir = os.path.join("results/stage2/eval", arm)
            path = os.path.join(REPO, arm_dir, "eval.json")
            if not os.path.exists(path):
                corrective.append({"arm": arm, "measured": False,
                                   "reason": "no anchored receipt"})
                continue
            payload = load_json(path)
            tasks = task_rows(payload)
            pairs = {task: paired_from_paths(
                raw_path_for(arm_dir, task), raw_path_for(m20_dir, task), task)
                for task in MATH}
            corrective.append({
                "arm": arm,
                "measured": True,
                "receipt": path,
                "stamp": payload.get("stamp"),
                "tasks": {task: tasks.get(task) for task in MATH},
                "math": family_mean(tasks, MATH),
                "paired_vs_bar": pairs,
                "note": "2M tokens against the bar's 19,998,720; matched stream and teacher, not matched tokens",
            })

    return {
        "registration": {
            "path": REGISTRATION,
            "sha256": hashlib.sha256(
                open(os.path.join(REPO, REGISTRATION), "rb").read()).hexdigest(),
        },
        "run_dir": RUN_DIR,
        "stable": stable,
        "improves_base_math": improves,
        "base": {"receipt": "results/stage2/eval/base/eval.json",
                 "tasks": base_rows,
                 "knowledge": family_mean(base_rows, KNOWLEDGE),
                 "math": family_mean(base_rows, MATH)},
        "diverged_bar": divergent,
        "milestones": rows,
        "slope": trend,
        "increment_10m_to_20m_math": increment_10_20,
        "extension_justified": positive,
        "corrective_comparison": corrective,
        "slope_rule": registration["bar_rerun"]["slope_rule"]["registered_text"],
    }


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_delta(row: dict | None) -> str:
    if not row or not row.get("paired"):
        return "unpaired"
    return f"{row['delta']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]"


def render(payload: dict) -> str:
    lines = [
        "### The rerun bar: milestone curve and scores",
        "",
        "| milestone | step | tokens | MMLU | MMLU-Pro | GSM8K | MATH-500 | knowledge | math | Δ GSM8K vs base [95% CI] | Δ MATH-500 vs base [95% CI] | seq GSM8K | seq MATH-500 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]

    def row_line(label: str, step, tokens, tasks, knowledge, math_family, paired, seq=True):
        def score(name):
            entry = tasks.get(name) or {}
            value = entry.get("score")
            error = entry.get("stderr")
            return "n/a" if value is None else f"{value * 100:.2f} ± {(error or 0) * 100:.2f}"

        def seq_text(name):
            if not seq:
                return "n/a"
            entry = (tasks.get(name) or {}).get("sequence_correct") or {}
            correct, items = entry.get("correct"), entry.get("items")
            return "n/a" if correct is None or not items else f"{correct}/{items}"

        return (
            f"| `{label}` | {step if step is not None else '—'} | "
            f"{tokens if tokens is not None else '—'} | {score('mmlu')} | "
            f"{score('mmlu_pro')} | {score('gsm8k')} | {score('minerva_math500')} | "
            f"{fmt(knowledge.get('mean'))} ± {fmt(knowledge.get('combined_stderr'))} | "
            f"{fmt(math_family.get('mean'))} ± {fmt(math_family.get('combined_stderr'))} | "
            f"{fmt_delta((paired or {}).get('gsm8k'))} | "
            f"{fmt_delta((paired or {}).get('minerva_math500'))} | "
            f"{seq_text('gsm8k')} | {seq_text('minerva_math500')} |"
        )

    base = payload["base"]
    lines.append(row_line("base", None, None, base["tasks"], base["knowledge"],
                          base["math"], None))
    for row in payload["milestones"]:
        lines.append(row_line(row["milestone"], row["step"], row["tokens_seen"],
                              row["tasks"], row["knowledge"], row["math"],
                              row["paired_vs_base"]))
    if payload.get("diverged_bar"):
        diverged = payload["diverged_bar"]
        lines.append(row_line("full-20m-math (diverged)", 9765, 19_998_720,
                              diverged["tasks"], diverged["knowledge"],
                              diverged["math"], None, seq=False))

    trend = payload.get("slope") or {}
    increment = payload.get("increment_10m_to_20m_math")
    lines += [
        "",
        "### The registered slope reading",
        "",
        f"- Math-family slope: **{fmt(trend.get('slope_per_million_tokens'), 4)} points per "
        f"million tokens** (OLS over m5, m10, m20 on the unweighted family mean).",
        f"- 10M-to-20M math increment: **{fmt(increment, 4)} points**.",
        (
            "- The registered 100M extension is **justified** by this bar: the slope is "
            "positive and the 10M-to-20M increment is positive. That is a statement about "
            "direction at this budget, not a projection to 100M."
            if payload["extension_justified"] else
            "- The registered 100M extension is **not justified** by this bar: the slope "
            "or the 10M-to-20M increment is not positive. The next route is named in the "
            "S49 work-log entry (more math tokens, RFT/on-policy, or another recipe), and "
            "no extension is launched on this evidence."
        ),
        (
            "- The bar is stable (no tripwire abort)."
            if payload["stable"] else
            "- The bar **tripped the early-loss tripwire**; its milestone scores are "
            "reported as a divergence, not as a learning curve."
        ),
        (
            "- The bar's math family is above base, so the matched 2M corrective "
            "comparison is reported below."
            if payload["improves_base_math"] else
            "- The bar's math family is not above base, so the matched 2M corrective "
            "comparison is **not** reported: comparing those arms to a bar that does not "
            "improve math would answer a question the registration did not ask."
        ),
        "",
        "Every cell is generated by `scripts/s49_bar_curve.py` from the milestone receipts "
        "under `results/stage2/eval/full-20m-math-s49/`, the base receipt and the diverged "
        "bar's receipt; paired deltas re-read the harness's own raw samples, pair items by "
        "`(leaf, doc_id)`, require identical doc-hash sequences, and use the S44 "
        "percentile bootstrap (4,000 resamples, seed 44). `seq` counts are the S47 exact "
        "final-answer verdicts on the primary filter. The family column is the unweighted "
        "mean of the two task scores with the independent combined standard error.",
    ]
    corrective = payload.get("corrective_comparison")
    if corrective:
        lines += [
            "",
            "### The matched 2M corrective arms against the rerun bar",
            "",
            "| arm | GSM8K | MATH-500 | math | Δ GSM8K vs bar [95% CI] | Δ MATH-500 vs bar [95% CI] |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
        for entry in corrective:
            if not entry.get("measured"):
                lines.append(f"| `{entry['arm']}` | n/a | n/a | n/a | "
                             f"{entry.get('reason') or 'unmeasured'} | n/a |")
                continue
            tasks = entry["tasks"]
            def score(name):
                value = (tasks.get(name) or {}).get("score")
                error = (tasks.get(name) or {}).get("stderr")
                return "n/a" if value is None else f"{value * 100:.2f} ± {(error or 0) * 100:.2f}"
            lines.append(
                f"| `{entry['arm']}` | {score('gsm8k')} | {score('minerva_math500')} | "
                f"{fmt(entry['math'].get('mean'))} | "
                f"{fmt_delta((entry['paired_vs_bar'] or {}).get('gsm8k'))} | "
                f"{fmt_delta((entry['paired_vs_bar'] or {}).get('minerva_math500'))} |")
        lines += [
            "",
            "The corrective arms trained 2M tokens on the same stream and teacher; their "
            "run receipts and anchored receipts are unchanged. The bar trained "
            "19,998,720 tokens, so this is a token-mismatched comparison, and the receipt "
            "says so. A positive delta therefore says a 2M mechanism arm beat a far longer "
            "distillation run on this task set; it does not isolate tokens.",
        ]
    return "\n".join(lines)


BEGIN = "<!-- BEGIN GENERATED: %s -->" % SENTINEL
END = "<!-- END GENERATED: %s -->" % SENTINEL


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def region_body(text: str) -> str:
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"s49-curve: {REPORT} has no {SENTINEL} region")
    return text[start + len(BEGIN):end].strip("\n")


def write_region(body: str) -> bool:
    path = os.path.join(REPO, REPORT)
    text = read_text(path)
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"s49-curve: {REPORT} has no {SENTINEL} region")
    updated = text[:start + len(BEGIN)] + "\n" + body + "\n" + text[end:]
    if updated == text:
        return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.write, args.check, args.summary)) != 1:
        raise SystemExit("choose exactly one of --write, --check, --summary")

    payload = collect()
    body = render(payload)
    if args.summary:
        print(body)
        return 0
    if args.check:
        if region_body(read_text(os.path.join(REPO, REPORT))) != body:
            print("s49-curve: the bar-curve region is stale; run --write", file=sys.stderr)
            return 1
        print("s49-curve: bar-curve region matches the receipts")
        return 0

    receipt = {
        "schema": SCHEMA,
        "kind": "s49-bar-curve",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        **payload,
    }
    out = os.path.join(REPO, "results/stage2", f"s49-bar-curve-{receipt['stamp']}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    changed = write_region(body)
    print(f"s49-curve: receipt {out}")
    print(f"s49-curve: {'wrote' if changed else 'kept'} {SENTINEL} in {REPORT}")
    print(f"s49-curve: extension_justified={payload['extension_justified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
