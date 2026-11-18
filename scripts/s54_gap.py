#!/usr/bin/env python3
"""S54: paired greedy-vs-maj@16 gap from two stage-1 receipts.

The maj@16 runs were scored on the first N problems of each task; the greedy
runs were scored on the full task. This script intersects the problem ids, pairs
the greedy and maj@16 correctness for each problem, and reports the gap with a
prompt-level paired bootstrap. It never resamples continuations.

Usage:
    scripts/court python -- python3 scripts/s54_gap.py \
        --greedy results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
        --maj16 results/stage1/s54-stage1-post-0.8b-nonthinking-maj16-subset50.json \
        --out results/stage1/s54-gaps.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESAMPLES = 4000
SEED = 53


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(deltas: list[float]) -> dict:
    if not deltas:
        return {"problems": 0, "delta": None, "ci95": None}
    rng = random.Random(SEED)
    n = len(deltas)
    draws = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(RESAMPLES)]
    return {
        "problems": n,
        "delta": sum(deltas) / n,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "resamples": RESAMPLES,
        "seed": SEED,
        "basis": ("prompt-level paired bootstrap; the resampling unit is the "
                  "distinct problem, never the continuation"),
    }


def _task_rows(payload: dict) -> dict:
    return {(row["task"], row["mode"]): row
            for row in payload.get("tasks") or []}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", required=True)
    parser.add_argument("--maj16", required=True)
    parser.add_argument("--out", default="results/stage1/s54-gaps.json")
    args = parser.parse_args(argv)
    greedy = json.load(open(os.path.join(ROOT, args.greedy), encoding="utf-8"))
    maj = json.load(open(os.path.join(ROOT, args.maj16), encoding="utf-8"))
    if greedy.get("complete") is not True or maj.get("complete") is not True:
        raise SystemExit("s54-gap: both receipts must be complete")
    greedy_rows = _task_rows(greedy)
    maj_rows = _task_rows(maj)
    rows = []
    for key in sorted(set(greedy_rows) & set(maj_rows)):
        task, mode = key
        g = greedy_rows[key]
        m = maj_rows[key]
        if g["decoding"] != "greedy" or m["decoding"] != "maj16":
            continue
        g_by_id = {row["problem_id"]: row for row in g["per_problem"]}
        m_by_id = {row["problem_id"]: row for row in m["per_problem"]}
        common = sorted(set(g_by_id) & set(m_by_id))
        if not common:
            continue
        deltas = []
        strict_deltas = []
        exact_deltas = []
        for prompt_id in common:
            g_row = g_by_id[prompt_id]
            m_row = m_by_id[prompt_id]
            deltas.append(float(bool(m_row["vote_correct"]))
                          - float(bool(g_row["flexible_correct"])))
            strict_deltas.append(float(bool(m_row["strict_vote_correct"]))
                                 - float(bool(g_row["strict_correct"])))
            exact_deltas.append(float(bool(m_row["exact_vote_correct"]))
                                - float(bool(g_row["exact_correct"])))
        greedy_acc = sum(bool(g_by_id[p]["flexible_correct"])
                         for p in common) / len(common)
        maj_acc = sum(bool(m_by_id[p]["vote_correct"]) for p in common) / len(common)
        rows.append({
            "model_id": (greedy.get("model") or {}).get("id"),
            "task": task,
            "mode": mode,
            "common_problems": len(common),
            "greedy_accuracy": greedy_acc,
            "maj16_accuracy": maj_acc,
            "gap": _bootstrap(deltas),
            "strict_gap_point": sum(strict_deltas) / len(strict_deltas),
            "exact_gap_point": sum(exact_deltas) / len(exact_deltas),
            "greedy_receipt": args.greedy,
            "greedy_receipt_sha256": _sha256(os.path.join(ROOT, args.greedy)),
            "maj16_receipt": args.maj16,
            "maj16_receipt_sha256": _sha256(os.path.join(ROOT, args.maj16)),
        })
    if not rows:
        raise SystemExit("s54-gap: no common task rows")
    payload = {
        "schema": "selvedge/s54-stage1-gap/v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "rows": rows,
        "reading": ("the gap is the free-compute headroom for stage 3; the "
                    "maj@16 subset is labelled by its common problem count and "
                    "is not a full-task score"),
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in rows:
        gap = row["gap"]
        print(f"s54-gap: {row['model_id']} {row['task']} {row['mode']} "
              f"greedy={row['greedy_accuracy']:.4f} "
              f"maj16={row['maj16_accuracy']:.4f} "
              f"gap={gap['delta']:+.4f} "
              f"[{gap['ci95'][0]:+.4f}, {gap['ci95'][1]:+.4f}] "
              f"n={row['common_problems']}")
    print(f"s54-gap: {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
