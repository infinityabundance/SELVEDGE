#!/usr/bin/env python3
"""S48: recompute sequence-level correctness on the registered primary filter.

The S47 correctness block in `s40_score.py` was fed every stored harness sample.
GSM8K stores one row per item *per filter* (`strict-match` and
`flexible-extract`), so the block counted 2,638 verdicts for 1,319 items and
reported an accuracy over both filters. The driver is fixed
(`primary_filter_samples`), but the five screen receipts and the reference
receipts were produced before the fix. Rather than edit those receipts, this
script recomputes the block from the receipts' own raw sample files, on the
registered primary filter, and writes a separate receipt per arm.

Usage:
    scripts/court oracle -- python3 scripts/s48_seq_correct.py
    scripts/court oracle -- python3 scripts/s48_seq_correct.py --arm base
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

import s40_score as driver  # noqa: E402
from frontier import correctness  # noqa: E402

SCHEMA = "selvedge/s48-seq-correct/v1"
ARMS = ("base", "adapter-2m", "corrective-kl-2m-math",
        "corrective-signed-exp-2m-math", "corrective-signed-true-2m-math",
        "adapter-2m-math-matched", "gated-dense-2m-math")
TASKS = ("gsm8k", "minerva_math500")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def newest(directory: str, task: str) -> str | None:
    import glob
    matches = sorted(glob.glob(os.path.join(REPO, directory, "raw",
                                            f"*-{task}.json")))
    return matches[-1] if matches else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    arms = args.arm or list(ARMS)
    prereg = json.load(open(os.path.join(REPO, "docs",
                                         "anchor-preregistration-v1.json"),
                            encoding="utf-8"))
    declared = {task["name"]: task for task in prereg["tasks"]}
    report = {
        "schema": SCHEMA,
        "kind": "sequence_correctness_primary_filter",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "why": "the S47 block counted every stored harness sample; GSM8K stores "
               "one row per item per filter, so the block double-counted items. "
               "This recomputes on the primary filter, from the same raw files.",
        "selection": "rows whose `filter` equals the task's registered primary "
                     "metric filter (gsm8k strict-match; minerva_math500 none)",
        "arms": {},
    }
    ok = True
    for arm in arms:
        directory = os.path.join("results/stage2/eval", arm)
        eval_path = os.path.join(directory, "eval.json")
        if not os.path.exists(os.path.join(REPO, eval_path)):
            print(f"s48-seq: {arm}: no eval receipt, skipped")
            ok = False
            continue
        receipt = json.load(open(os.path.join(REPO, eval_path), encoding="utf-8"))
        tasks = {}
        for entry in receipt.get("tasks", []):
            name = entry["task"]
            if name not in TASKS:
                continue
            path = newest(directory, name)
            if path is None:
                tasks[name] = {"status": "missing raw samples"}
                ok = False
                continue
            result = json.load(open(path, encoding="utf-8"))
            key = next(iter(result["samples"]))
            samples = result["samples"][key]
            kept = driver.primary_filter_samples(samples, declared[name])
            summary = correctness.harness_samples(kept, name)
            block = receipt.get("tasks", [])
            recorded = next((row.get("sequence_correctness")
                             for row in block if row["task"] == name), None)
            tasks[name] = {
                "status": "ok",
                "raw": os.path.relpath(path, REPO),
                "raw_sha256": sha256_file(path),
                "stored_samples": len(samples),
                "kept_samples": len(kept),
                "predicted_receipt_items": (recorded or {}).get("items"),
                "correct": summary["correct"],
                "items": summary["items"],
                "accuracy": summary["accuracy"],
                "accuracy_stderr": summary["accuracy_stderr"],
                "verdict_counts": summary["verdict_counts"],
                "answered": summary["answered"],
                "accuracy_answered": summary["accuracy_answered"],
            }
            print(f"s48-seq: {arm} {name} {summary['correct']}/{summary['items']} "
                  f"= {summary['accuracy']:.4f} (kept {len(kept)} of {len(samples)} "
                  f"rows; receipt block had {(recorded or {}).get('items')} items)")
        report["arms"][arm] = {
            "eval": eval_path,
            "eval_sha256": sha256_file(os.path.join(REPO, eval_path)),
            "tasks": tasks,
        }
    report["ok"] = ok
    out = args.out or os.path.join("results/stage2",
                                   f"s48-seq-correct-{report['stamp']}.json")
    full = os.path.join(REPO, out)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s48-seq: receipt {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
