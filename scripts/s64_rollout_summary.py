#!/usr/bin/env python3
"""S64: merge the corrected K=8 shard receipts into one honest summary.

Each shard (`scripts/s59_student_rollouts.py`) writes its own receipt, ledger
and failed-continuation sidecar. This helper reads the receipts and:

* refuses a receipt whose `loop_cut.compute_saved` is not exactly `false` (the
  per-row cut is textual replay, not compute saved);
* refuses a receipt whose K is not 8 or whose shard plan is not the registered
  one;
* verifies the ledger and sidecar SHA-256 against the receipt;
* merges the per-problem ledger's p histogram, the p=0 / p=1 counts and the
  `p(1-p)` weight sum;
* reports the measured aggregate tokens/s and the textual-replay token count
  beside the loop-cut and truncation rates, per shard and overall.

Nothing is typed by hand: every merged number is a sum of receipt fields, and
the output records which receipts were read.

Usage
-----
    SELVEDGE_MEM=3g scripts/court oracle -- python3 \
      scripts/s64_rollout_summary.py \
        --receipt results/artifacts/stage2/s64-rollouts/gsm8k-shard0-receipt.json \
        --receipt results/artifacts/stage2/s64-rollouts/gsm8k-shard1-receipt.json \
        --receipt results/artifacts/stage2/s64-rollouts/math-shard0-receipt.json \
        --receipt results/artifacts/stage2/s64-rollouts/math-shard1-receipt.json \
        --out results/artifacts/stage2/s64-rollouts/summary-receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCHEMA = "selvedge/s64-rollout-summary/v1"
REGISTERED_K = 8
REGISTERED_SHARDS = 2


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


def sha256_file(path: str) -> str:
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


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _check_receipt(path: str, receipt: dict) -> dict:
    """Fail closed on anything that would make the summary dishonest."""
    loop = receipt.get("loop_cut") or {}
    if loop.get("compute_saved") is not False:
        raise SystemExit(
            f"s64-summary: {path}: loop_cut.compute_saved is not false; a "
            "saved-compute claim is not accepted")
    sampling = receipt.get("sampling") or {}
    for key in ("k_gsm8k", "k_math"):
        if int(sampling.get(key, 0)) != REGISTERED_K:
            raise SystemExit(
                f"s64-summary: {path}: {key}={sampling.get(key)} is not the "
                f"registered K={REGISTERED_K}")
    if int(sampling.get("shards", 0)) != REGISTERED_SHARDS:
        raise SystemExit(
            f"s64-summary: {path}: shards={sampling.get('shards')} is not the "
            f"registered {REGISTERED_SHARDS}")
    ledger = receipt.get("ledger") or {}
    ledger_path = ledger.get("path")
    if not ledger_path:
        raise SystemExit(f"s64-summary: {path}: no per-problem ledger")
    resolved = _resolve(ledger_path)
    if not os.path.exists(resolved):
        raise SystemExit(f"s64-summary: {path}: ledger {ledger_path} is missing")
    actual = sha256_file(resolved)
    if ledger.get("sha256") and ledger["sha256"] != actual:
        raise SystemExit(
            f"s64-summary: {path}: ledger sha256 does not match the receipt")
    preference = receipt.get("preference_data") or {}
    rejected_path = preference.get("rejected_path")
    if rejected_path:
        resolved_rejected = _resolve(rejected_path)
        if not os.path.exists(resolved_rejected):
            raise SystemExit(
                f"s64-summary: {path}: rejected sidecar {rejected_path} is "
                "missing")
        if (preference.get("sha256")
                and preference["sha256"] != sha256_file(resolved_rejected)):
            raise SystemExit(
                f"s64-summary: {path}: rejected sidecar sha256 does not match")
    return {
        "receipt": path,
        "ledger": ledger_path,
        "ledger_sha256": actual,
        "rejected": rejected_path,
        "rejected_sha256": preference.get("sha256"),
    }


def summarise(receipts: list[tuple[str, dict]]) -> dict:
    checked = [_check_receipt(path, receipt) for path, receipt in receipts]
    totals = {"prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
              "duplicates": 0, "truncated": 0, "loop_cut": 0,
              "generated_tokens": 0, "pre_cut_generated_tokens": 0,
              "textual_replay_tokens_cut": 0, "wall_seconds": 0.0}
    by_dataset: dict[str, dict] = {}
    ledger_rows = 0
    ledger_attempts = 0
    ledger_verified = 0
    weight_sum = 0.0
    zero_p = 0
    one_p = 0
    p_histogram: dict[str, int] = {}
    successes_per_problem: list[int] = []
    shards = []
    for path, receipt in receipts:
        acceptance = receipt["acceptance"]["totals"]
        for key in totals:
            totals[key] += acceptance.get(key, 0)
        generation = receipt["generation"]
        ledger_summary = receipt["ledger"]["summary"]
        ledger_rows += int(receipt["ledger"]["rows"])
        ledger_attempts += int(ledger_summary.get("attempts", 0))
        ledger_verified += int(ledger_summary.get("verified_correct", 0))
        weight_sum += float(ledger_summary.get("sum_weight_p1mp", 0.0))
        zero_p += int(ledger_summary.get("zero_p_problems", 0))
        one_p += int(ledger_summary.get("one_p_problems", 0))
        for key, value in (ledger_summary.get("p_histogram") or {}).items():
            p_histogram[key] = p_histogram.get(key, 0) + int(value)
        successes_per_problem.extend(
            int(value) for value in
            ledger_summary.get("successes_per_problem", []))
        for name, row in (receipt["acceptance"].get("by_dataset") or {}).items():
            entry = by_dataset.setdefault(name, {
                "prompts": 0, "sampled": 0, "correct": 0, "kept": 0,
                "duplicates": 0, "truncated": 0, "loop_cut": 0,
                "generated_tokens": 0, "pre_cut_generated_tokens": 0,
                "textual_replay_tokens_cut": 0, "wall_seconds": 0.0,
                "solution_tokens": 0, "rows": 0})
            for key in entry:
                entry[key] += row.get(key, 0)
        shards.append({
            "receipt": path,
            "stamp": receipt.get("stamp"),
            "revision": receipt.get("revision"),
            "shard_index": receipt["sampling"].get("shard_index"),
            "dataset": next(iter(receipt["acceptance"]["by_dataset"])),
            "tokens_per_second": generation.get("tokens_per_second"),
            "loop_cut_rate": receipt.get("acceptance", {}).get("loop_cut_rate"),
            "truncation_rate": receipt.get(
                "acceptance", {}).get("truncation_rate"),
            "textual_replay_tokens_cut": generation.get(
                "textual_replay_tokens_cut"),
            "textual_replay_token_fraction": generation.get(
                "textual_replay_token_fraction"),
            "ledger_rows": receipt["ledger"]["rows"],
            "mean_p": ledger_summary.get("mean_p"),
        })
    for entry in by_dataset.values():
        entry["solution_length"] = None
    pooled_p = (ledger_verified / ledger_attempts) if ledger_attempts else 0.0
    mean_p = (sum(successes_per_problem) / len(successes_per_problem)
              if successes_per_problem else None)
    return {
        "shards": shards,
        "totals": totals,
        "by_dataset": by_dataset,
        "ledger": {
            "rows": ledger_rows,
            "attempts": ledger_attempts,
            "verified_correct": ledger_verified,
            "pooled_p_from_attempts": pooled_p,
            "mean_successes_per_problem": mean_p,
            "sum_weight_p1mp": weight_sum,
            "zero_p_problems": zero_p,
            "one_p_problems": one_p,
            "p_histogram": dict(sorted(p_histogram.items())),
            "p_basis": "verified_correct / attempts (before deduplication)",
        },
        "throughput": {
            "generated_tokens": totals["generated_tokens"],
            "sampling_wall_seconds": totals["wall_seconds"],
            "tokens_per_second": (totals["generated_tokens"]
                                  / totals["wall_seconds"]
                                  if totals["wall_seconds"] else None),
            "textual_replay_tokens_cut": totals["textual_replay_tokens_cut"],
            "textual_replay_token_fraction": (
                totals["textual_replay_tokens_cut"]
                / max(totals["pre_cut_generated_tokens"], 1)),
            "compute_saved": False,
            "basis": ("textual replay only: a batched generate advances every "
                      "row until the slowest row stops"),
        },
        "inputs": checked,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipts = []
    for path in args.receipt:
        resolved = _resolve(path)
        receipts.append((path, json.load(open(resolved, encoding="utf-8"))))
    merged = summarise(receipts)
    payload = {
        "schema": SCHEMA,
        "kind": "corrected-rollout-summary",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": (
            "docs/ws19-stage2-correction-s63-preregistration-v1.json"),
        "command": sys.argv,
        **merged,
    }
    write_json(_resolve(args.out), payload)
    print("s64-summary: "
          f"{merged['totals']['prompts']} prompts, "
          f"{merged['totals']['sampled']} samples, "
          f"{merged['totals']['kept']} accepted rows, "
          f"pooled p={merged['ledger']['pooled_p_from_attempts']:.4f}")
    print(f"s64-summary: {merged['throughput']['tokens_per_second']:.1f} "
          f"generated tokens/s (compute_saved=false) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
