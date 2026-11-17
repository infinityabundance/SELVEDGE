#!/usr/bin/env python3
"""S50: consolidate the 200-step revalidation receipts into one record."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNS = [
    ("lr1e-4", "results/stage2/runs/s50-revalidate-200-lr1e-4-curve",
     "results/stage2/s50-fixed-batch-kl-math-20261117T205500Z.json",
     "results/stage2/s50-fixed-batch-kl-20261117T204500Z.json"),
    ("lr1e-5", "results/stage2/runs/s50-revalidate-200-lr1e-5",
     "results/stage2/s50-fixed-batch-kl-math-lr1e-5-20261117T205600Z.json",
     None),
]
STEPS = (1, 50, 100, 200)


def sha256(path):
    digest = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    runs = []
    for label, run_dir, training_kl, heldout_kl in RUNS:
        rows = [json.loads(line) for line in
                open(os.path.join(ROOT, run_dir, "log.jsonl"), encoding="utf-8")]
        receipt = json.load(open(os.path.join(ROOT, run_dir, "receipt.json"),
                                 encoding="utf-8"))
        training = json.load(open(os.path.join(ROOT, training_kl),
                                  encoding="utf-8"))
        record = {
            "label": label,
            "run_dir": run_dir,
            "lr": max(row["lr"] for row in rows),
            "steps": len(rows),
            "log_kl_at_steps": {
                str(step): next(row["kl"] for row in rows if row["step"] == step)
                for step in STEPS},
            "log_ce_at_steps": {
                str(step): next(row["ce"] for row in rows if row["step"] == step)
                for step in STEPS},
            "max_update_over_lr": max(row["max_update_over_lr"] for row in rows),
            "median_update_over_lr": sorted(
                row["max_update_over_lr"] for row in rows)[len(rows) // 2],
            "update_tripwire": receipt.get("update_tripwire"),
            "peak_allocated_gib": receipt.get("peak_allocated_gib"),
            "peak_reserved_gib": receipt.get("peak_reserved_gib"),
            "container_memory": receipt.get("container_memory"),
            "fixed_batch_kl_training_distribution": training["results"],
            "fixed_batch_kl_training_receipt": training_kl,
            "fixed_batch_kl_training_sha256": sha256(training_kl),
            "fixed_batch_kl_heldout": (
                json.load(open(os.path.join(ROOT, heldout_kl),
                               encoding="utf-8"))["results"]
                if heldout_kl else None),
            "fixed_batch_kl_heldout_receipt": heldout_kl,
            "fixed_batch_kl_heldout_sha256": (
                sha256(heldout_kl) if heldout_kl else None),
            "receipt": run_dir + "/receipt.json",
            "receipt_sha256": sha256(run_dir + "/receipt.json"),
        }
        first, last = (record["fixed_batch_kl_training_distribution"][0],
                       record["fixed_batch_kl_training_distribution"][-1])
        record["fixed_batch_kl_training_direction"] = (
            "fall" if last["kl"] < first["kl"] else "rise")
        runs.append(record)

    chosen = next(run for run in runs
                  if run["fixed_batch_kl_training_direction"] == "fall")
    record = {
        "schema": "selvedge/s50-revalidation/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": revision(),
        "criterion": (
            "200 optimiser steps at the candidate rate with the corrected "
            "optimiser; the fixed-batch teacher KL on the training distribution "
            "must fall. The per-step training KL is also recorded but is not the "
            "gate: every step sees a different batch and the pinned stream's "
            "difficulty varies."),
        "fixed_batch_basis": (
            "8 batches x 512 positions, offsets 2,000,000.. from the registered "
            "math stream; KL(teacher_T || student_T) at T=2.0"),
        "update_tripwire_factor": 10.0,
        "runs": runs,
        "chosen": {
            "label": chosen["label"],
            "lr": chosen["lr"],
            "rule": ("1e-4 did not satisfy the KL-fall gate; 1e-5 did. Neither "
                     "run tripped the update tripwire, so the rate was lowered "
                     "by the stated KL criterion, not by a tripwire abort."),
            "run_dir": chosen["run_dir"],
        },
        "supersedes_rate": "1e-4 (rejected) and the invalidated S49 swept rates",
    }
    out = os.path.join(ROOT, f"results/stage2/s50-revalidation-{record['stamp']}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s50-revalidation: {os.path.relpath(out, ROOT)}")
    for run in runs:
        print(f"  {run['label']}: training KL "
              f"{run['fixed_batch_kl_training_distribution'][0]['kl']:.6f} -> "
              f"{run['fixed_batch_kl_training_distribution'][-1]['kl']:.6f} "
              f"({run['fixed_batch_kl_training_direction']}), "
              f"max|u|/lr={run['max_update_over_lr']:.3f}, "
              f"peak={run['peak_allocated_gib']:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
