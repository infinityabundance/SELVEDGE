#!/usr/bin/env python3
"""S64: recompute the registered rollout shard projection and refuse misfits.

The S63 correction registration fixes the corrected rollout plan:

* GSM8K train: 1,200 prompts in two 600-prompt shards, K=8;
* MATH train: 600 prompts in two 300-prompt shards, K=8;
* each shard must project inside the v1 time-ladder bound of 4.5 h, using the
  **pilot receipt's realised tokens/sample and aggregate token rate**;
* the launch record recomputes the projection and refuses a shard that does
  not fit.

This script does exactly that and writes a launch record. It never types a
throughput number: the rate, the per-sample lengths and the sample counts are
read from the pilot receipt and the projected seconds are computed. The
corrected K=8 shards keep the pilot's 64 concurrent sequences
(`--batch-prompts 8` at K=8, the pilot used `--batch-prompts 16` at K=4), so
the pilot's aggregate token rate is the applicable one.

The projection is arithmetic on a measured pilot, not a promise: the corrected
script measures its own tokens/s again in each shard receipt.

Usage
-----
    SELVEDGE_MEM=3g scripts/court oracle -- python3 \
      scripts/s64_shard_window.py \
        --pilot-receipt results/artifacts/stage2/s59-rft/pilot-receipt.json \
        --out results/artifacts/stage2/s64-rollouts/launch-record.json
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

SCHEMA = "selvedge/s64-rollout-launch-record/v1"
REGISTRATION = "docs/ws19-stage2-correction-s63-preregistration-v1.json"
#: The registered v1 time-ladder bound, in hours (registration window clause).
TIME_LADDER_BOUND_HOURS = 4.5
STUDENT_DIR = ("/hf-cache/models/Qwen/Qwen3.5-0.8B/"
               "2fc06364715b967f1860aea9cf38778875588b17")
HOLDOUT = "results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl"
OUT_DIR = "results/artifacts/stage2/s64-rollouts"
#: (label, dataset flag, select flag, prompts selected, shards, K, cap)
PLAN = (
    ("gsm8k-shard0", "gsm8k-train", "--select-gsm8k", 1200, 2, 0, 8, 2048),
    ("gsm8k-shard1", "gsm8k-train", "--select-gsm8k", 1200, 2, 1, 8, 2048),
    ("math-shard0", "math-train", "--select-math", 600, 2, 0, 8, 2048),
    ("math-shard1", "math-train", "--select-math", 600, 2, 1, 8, 2048),
)


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def project(pilot: dict, plan=PLAN,
            bound_hours: float = TIME_LADDER_BOUND_HOURS) -> dict:
    """Project every shard from the pilot's measured rate and lengths."""
    generation = pilot["generation"]
    rate = float(generation["tokens_per_second"])
    if rate <= 0:
        raise SystemExit("s64-window: the pilot's token rate is not positive")
    by_dataset = pilot["acceptance"]["by_dataset"]
    per_sample = {}
    for name, row in by_dataset.items():
        samples = int(row["sampled"])
        tokens = int(row["generated_tokens"])
        if samples <= 0:
            raise SystemExit(f"s64-window: {name} has no sampled rows")
        per_sample[name] = tokens / samples
    bound_seconds = bound_hours * 3600.0
    shards = []
    for (label, dataset, select_flag, selected, shards_count, shard_index,
         k, cap) in plan:
        key = dataset.split("-")[0]
        if key not in per_sample:
            raise SystemExit(
                f"s64-window: the pilot has no {key} rows; cannot project "
                f"{label}")
        prompts = selected // shards_count
        samples = prompts * k
        projected_tokens = samples * per_sample[key]
        projected_seconds = projected_tokens / rate
        shards.append({
            "label": label,
            "dataset": dataset,
            "prompts": prompts,
            "k": k,
            "samples": samples,
            "cap_tokens": cap,
            "pilot_tokens_per_sample": per_sample[key],
            "pilot_tokens_per_second": rate,
            "projected_generated_tokens": projected_tokens,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600.0,
            "time_ladder_bound_hours": bound_hours,
            "fits": projected_seconds <= bound_seconds,
        })
    return {"pilot_rate": rate, "per_sample": per_sample,
            "shards": shards,
            "all_fit": all(shard["fits"] for shard in shards),
            "bound_hours": bound_hours}


def command(label: str, dataset: str, select_flag: str, selected: int,
            shards_count: int, shard_index: int, k: int, cap: int,
            mem: str = "4g") -> list[str]:
    return [
        "scripts/court", "--gpu", "oracle", "--", "python3",
        "scripts/s59_student_rollouts.py",
        "--policy-dir", STUDENT_DIR,
        "--tokenizer-dir", STUDENT_DIR,
        "--datasets", dataset, select_flag, str(selected),
        "--k-gsm8k", str(k), "--k-math", str(k),
        "--shards", str(shards_count), "--shard-index", str(shard_index),
        "--batch-prompts", "8",
        f"--max-new-tokens-{'gsm8k' if dataset.startswith('gsm8k') else 'math'}",
        str(cap),
        "--exclude-ids", HOLDOUT,
        "--rejected-out", f"{OUT_DIR}/{label}-rejected.jsonl",
        "--out-dir", OUT_DIR, "--label", label,
        "--vram-limit-gib", "12.0",
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-receipt", required=True)
    parser.add_argument("--bound-hours", type=float,
                        default=TIME_LADDER_BOUND_HOURS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    pilot_path = (args.pilot_receipt if os.path.isabs(args.pilot_receipt)
                  else os.path.join(ROOT, args.pilot_receipt))
    pilot = json.load(open(pilot_path, encoding="utf-8"))
    if "generation" not in pilot or "acceptance" not in pilot:
        raise SystemExit("s64-window: the pilot receipt is not a rollout receipt")
    projection = project(pilot, PLAN, args.bound_hours)

    commands = [{
        "label": label,
        "command": command(label, dataset, select_flag, selected,
                           shards_count, shard_index, k, cap),
        "environment": {"SELVEDGE_MEM": "4g", "SELVEDGE_CPUS": "8",
                        "SELVEDGE_PIDS": "1024", "SELVEDGE_TMP": "2g"},
    } for (label, dataset, select_flag, selected, shards_count, shard_index,
           k, cap) in PLAN]

    record = {
        "schema": SCHEMA,
        "kind": "rollout-launch-record",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "pilot_receipt": {
            "path": os.path.relpath(pilot_path, ROOT),
            "stamp": pilot.get("stamp"),
            "revision": pilot.get("revision"),
            "claims": ("the pilot is the old K=4 stopping run; its tokens/s is "
                       "the measured aggregate rate for the same 64 concurrent "
                       "sequences, and its saved-token numbers are textual "
                       "replay only"),
        },
        "plan": projection,
        "launch": {
            "student": STUDENT_DIR,
            "k": 8,
            "batch_prompts": 8,
            "concurrent_sequences": 64,
            "memory_basis": (
                "the running K=4 job's cgroup memory.peak and the pilot "
                "receipt's cgroup_memory.memory_peak; both are far below the "
                "4g limit, which is 1.5x the larger measured peak plus 1 GiB"),
            "one_gpu_job_at_a_time": True,
            "compute_saved_from_loop_cut": False,
        },
        "commands": commands,
        "refused": not projection["all_fit"],
    }
    write_json(os.path.join(ROOT, args.out), record)
    for shard in projection["shards"]:
        print(f"s64-window: {shard['label']}: {shard['samples']} samples, "
              f"projected {shard['projected_hours']:.2f} h "
              f"({'fits' if shard['fits'] else 'REFUSED'})")
    print(f"s64-window: launch record -> {args.out}")
    if not projection["all_fit"]:
        print("s64-window: at least one shard exceeds the registered bound; "
              "no shard may be launched from this plan", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
