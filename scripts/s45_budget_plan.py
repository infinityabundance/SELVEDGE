#!/usr/bin/env python3
"""S45: longer-budget arithmetic and checkpoint plan for the next route.

Every rate and size in the emitted receipt is read from a measured artifact; no
number is typed here.

* the training rate is the **median of the last 50 lines** of the in-flight
  full-20M math-weighted run's `log.jsonl` (the bar's own shape), with the last
  line's instantaneous rate also recorded;
* the step size is 2,048 tokens (512 sequence x 4 accumulation), the
  registered stage-2/2b geometry;
* the checkpoint size is the on-disk size of the run's first 1,000-step
  checkpoint;
* the sampling bound is `prompts x K x max_new_tokens`; the accepted-token count
  is **not** predicted here, because the acceptance rate is the measurement the
  generation job produces.

Usage
-----
    scripts/court python -- python3 scripts/s45_budget_plan.py
    scripts/court python -- python3 scripts/s45_budget_plan.py --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = "results/stage2/runs/full-20m-math/log.jsonl"
MANIFEST = "results/stage2/runs/full-20m-math/manifest.json"
CHECKPOINT = "results/stage2/runs/full-20m-math/ckpt-001000.pt"
OUT_DIR = "results/stage2"

TOKENS_PER_STEP = 2048
HORIZONS = (20_000_000, 40_000_000, 60_000_000, 100_000_000)
MILESTONES = (20_000_000, 40_000_000, 60_000_000, 100_000_000)
GSM8K_TRAIN_ROWS = 7473
MATH_TRAIN_ROWS = 7500
SAMPLES_PER_PROMPT = 8
MAX_NEW_TOKENS = 512


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_rate() -> dict:
    path = os.path.join(REPO, LOG)
    lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    if not lines:
        raise SystemExit(f"{LOG} is empty; cannot size the plan")
    tail = lines[-50:]
    rates = [float(row["tokens_per_second"]) for row in tail]
    return {
        "path": LOG,
        "sha256": sha256_file(path),
        "lines": len(lines),
        "last_step": int(lines[-1]["step"]),
        "last_tokens_seen": int(lines[-1]["tokens_seen"]),
        "last_tokens_per_second": rates[-1],
        "median_last_50_tokens_per_second": statistics.median(rates),
        "min_last_50_tokens_per_second": min(rates),
        "max_last_50_tokens_per_second": max(rates),
        "measured_utc": utcnow(),
    }


def arithmetic(rate: float) -> dict:
    horizons = {}
    for tokens in HORIZONS:
        steps = -(-tokens // TOKENS_PER_STEP)
        realised = steps * TOKENS_PER_STEP
        seconds = tokens / rate
        horizons[f"{tokens // 1_000_000}M"] = {
            "tokens": tokens,
            "steps": steps,
            "realised_tokens": realised,
            "shortfall_tokens": realised - tokens,
            "seconds": seconds,
            "hours": seconds / 3600.0,
        }
    return horizons


def checkpoints(rate: float, checkpoint_bytes: int, run_tokens: int) -> dict:
    plan = []
    for tokens in MILESTONES:
        if tokens > run_tokens:
            continue
        steps = -(-tokens // TOKENS_PER_STEP)
        plan.append({
            "tokens": tokens,
            "step": steps,
            "elapsed_hours_from_start": tokens / rate / 3600.0,
        })
    retained = len(plan)
    # The launcher keeps the milestone plus its rotating `latest` copy of the
    # newest checkpoint; record both the milestone-only and the conservative
    # with-latest figure.
    return {
        "run_tokens": run_tokens,
        "milestones": plan,
        "checkpoint_bytes_measured": checkpoint_bytes,
        "milestone_storage_bytes": retained * checkpoint_bytes,
        "milestone_storage_gib": retained * checkpoint_bytes / float(1 << 30),
        "with_latest_copy_bytes": (retained + 1) * checkpoint_bytes,
        "with_latest_copy_gib": (retained + 1) * checkpoint_bytes / float(1 << 30),
        "note": "the checkpoint size is the measured ckpt-001000.pt; a milestone "
                "is kept in addition to the rotating newest/latest pair",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rate_report = read_rate()
    rate = rate_report["median_last_50_tokens_per_second"]
    checkpoint_path = os.path.join(REPO, CHECKPOINT)
    checkpoint_bytes = os.path.getsize(checkpoint_path)
    manifest = None
    manifest_path = os.path.join(REPO, MANIFEST)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

    prompts = GSM8K_TRAIN_ROWS + MATH_TRAIN_ROWS
    sampling_max_tokens = prompts * SAMPLES_PER_PROMPT * MAX_NEW_TOKENS
    payload = {
        "schema": "selvedge/s45-budget-plan/v1",
        "kind": "longer_budget_plan",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "rate": rate_report,
        "step": TOKENS_PER_STEP,
        "horizons": arithmetic(rate),
        "checkpoints": {
            f"{tokens // 1_000_000}M": checkpoints(rate, checkpoint_bytes, tokens)
            for tokens in (60_000_000, 100_000_000)
        },
        "sampling": {
            "prompts": prompts,
            "gsm8k_train_rows": GSM8K_TRAIN_ROWS,
            "math_train_rows": MATH_TRAIN_ROWS,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_generated_tokens_one_pass": sampling_max_tokens,
            "max_generated_tokens_one_pass_millions": sampling_max_tokens / 1e6,
            "accepted_tokens": "unmeasured until the generation job runs; the "
                               "acceptance rate is the job's receipt",
            "time_at_training_rate_hours_lower_bound": sampling_max_tokens / rate / 3600.0,
            "time_note": "generation is not the training loop; this is an "
                         "arithmetic lower bound at the measured training rate, "
                         "not a prediction",
        },
        "manifest": manifest,
        "checkpoint_path": CHECKPOINT,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "win_bar": "beat base and the full-distillation bar on the math family, "
                   "with the paired item-level interval excluding zero and no "
                   "held-out or knowledge regression",
    }
    path = args.out or os.path.join(OUT_DIR, f"s45-budget-plan-{payload['stamp']}.json")
    write_json(os.path.join(REPO, path), payload)
    print(f"rate median(last 50) {rate:.1f} tok/s "
          f"(last {rate_report['last_tokens_per_second']:.1f}) "
          f"from {LOG} step {rate_report['last_step']} "
          f"tokens {rate_report['last_tokens_seen']}")
    for name, row in payload["horizons"].items():
        print(f"  {name:>5}: {row['steps']:6d} steps, "
              f"{row['hours']:.3f} h")
    for name, plan in payload["checkpoints"].items():
        marks = ", ".join(f"{m['tokens'] // 1_000_000}M@{m['elapsed_hours_from_start']:.2f}h"
                          for m in plan["milestones"])
        print(f"  {name} run checkpoints: {marks}; "
              f"storage {plan['milestone_storage_gib']:.2f} GiB "
              f"(+latest {plan['with_latest_copy_gib']:.2f} GiB)")
    print(f"  sampling bound one pass: "
          f"{payload['sampling']['max_generated_tokens_one_pass_millions']:.1f}M tokens")
    print(f"receipt: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
