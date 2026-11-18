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


def lm_eval_rows(path: str, metric: str) -> dict[tuple[str, int], tuple[float, str]]:
    """Per-item primary-metric value from an lm-evaluation-harness raw dump.

    Items are identified by `(leaf task, doc_id)`, the harness's own positional
    identity within a leaf, and the document's `doc_hash` is carried beside it
    so the pairing can refuse two runs that index different documents.
    """
    if path.endswith(".gz"):
        import gzip
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = open(path, "r", encoding="utf-8")
    with handle:
        payload = json.load(handle)
    rows: dict[tuple[str, int], tuple[float, str]] = {}
    for leaf, samples in payload["samples"].items():
        for sample in samples:
            if metric not in sample:
                continue
            rows[(leaf, int(sample["doc_id"]))] = (
                float(sample[metric]), sample.get("doc_hash", ""))
    return rows


def compare_lm_eval(a_path: str, b_path: str, metric: str) -> dict:
    a_rows = lm_eval_rows(a_path, metric)
    b_rows = lm_eval_rows(b_path, metric)
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        raise SystemExit("s57-paired: the two dumps share no (leaf, doc_id) item")
    mismatched_docs = [key for key in common
                       if a_rows[key][1] != b_rows[key][1]]
    if mismatched_docs:
        raise SystemExit(
            "s57-paired: the two dumps index different documents at "
            f"{len(mismatched_docs)} paired positions (first {mismatched_docs[:3]})")
    deltas = [a_rows[key][0] - b_rows[key][0] for key in common]
    a_score = sum(a_rows[key][0] for key in common) / len(common)
    b_score = sum(b_rows[key][0] for key in common) / len(common)
    interval = bootstrap(deltas)
    ci = interval["ci95"]
    return {
        "kind": "lm-eval-primary-metric",
        "metric": metric,
        "candidate_items": len(a_rows),
        "baseline_items": len(b_rows),
        "paired_items": len(common),
        "candidate_only": len(set(a_rows) - set(b_rows)),
        "baseline_only": len(set(b_rows) - set(a_rows)),
        "candidate_score": a_score,
        "baseline_score": b_score,
        "delta": interval["delta"],
        "ci95": ci,
        "items": interval["problems"],
        "resamples": interval["resamples"],
        "seed": interval["seed"],
        "basis": interval["basis"],
        "interval_above_zero": bool(ci is not None and ci[0] > 0.0),
        "interval_no_worse_than_minus_1_point": bool(
            ci is not None and ci[0] >= -0.01),
        "pairing_key": "(leaf task, doc_id), with doc_hash equality enforced",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task filter (default: every task "
                             "present in both receipts)")
    parser.add_argument("--lm-eval-candidate", default=None,
                        help="comma-separated harness raw dumps in the order "
                             "mmlu,mmlu_pro (instead of stage-1 receipts)")
    parser.add_argument("--lm-eval-baseline", default=None)
    parser.add_argument("--lm-eval-metrics", default="mmlu,mmlu_pro",
                        help="retention tasks to pair, in the same order as the "
                             "dumps")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.lm_eval_candidate or args.lm_eval_baseline:
        if not (args.lm_eval_candidate and args.lm_eval_baseline):
            raise SystemExit("s57-paired: give both harness raw dumps")
        rows = []
        candidates = [value.strip() for value in
                      args.lm_eval_candidate.split(",") if value.strip()]
        baselines = [value.strip() for value in
                     args.lm_eval_baseline.split(",") if value.strip()]
        wanted = [value.strip() for value in
                  args.lm_eval_metrics.split(",") if value.strip()]
        plan = []
        for task, metric in (("mmlu", "acc"), ("mmlu_pro", "exact_match")):
            if task in wanted or metric in wanted:
                plan.append((task, metric))
        if len(candidates) != len(plan) or len(baselines) != len(plan):
            raise SystemExit(
                "s57-paired: give one raw dump per retention task, in the order "
                f"{ [task for task, _ in plan] }; got {len(candidates)} "
                f"candidate and {len(baselines)} baseline paths")
        for (task, metric), candidate_path, baseline_path in zip(
                plan, candidates, baselines):
            row = compare_lm_eval(os.path.join(ROOT, candidate_path),
                                  os.path.join(ROOT, baseline_path), metric)
            row["task"] = task
            row["candidate_raw"] = candidate_path
            row["baseline_raw"] = baseline_path
            rows.append(row)
        if not rows:
            raise SystemExit("s57-paired: no retention task matched the metric filter")
        payload = {
            "schema": SCHEMA,
            "created_utc": utcnow(),
            "stamp": stamp(),
            "revision": revision(),
            "kind": "retention-paired",
            "retention_bound": {
                "statement": "within 1.0 point of the post-trained baseline, "
                             "paired 95% interval no worse than -1.0",
            },
            "rows": rows,
            "passed": all(row["interval_no_worse_than_minus_1_point"]
                          and abs(row["delta"]) <= 0.01 for row in rows),
            "command": sys.argv,
        }
        out_path = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(out_path + ".tmp", out_path)
        for row in rows:
            print(f"s57-paired: {row['task']} n={row['paired_items']} "
                  f"candidate={row['candidate_score']:.4f} "
                  f"baseline={row['baseline_score']:.4f} "
                  f"delta={row['delta']:+.4f} "
                  f"ci95=[{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}] "
                  f"no_worse_than_-1pt={row['interval_no_worse_than_minus_1_point']}")
        print(f"s57-paired: passed={payload['passed']} -> {args.out}")
        return 0

    if not (args.candidate and args.baseline):
        raise SystemExit("s57-paired: give --candidate/--baseline or the lm-eval dumps")

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
