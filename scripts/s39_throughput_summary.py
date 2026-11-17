#!/usr/bin/env python3
"""Derive the Job-1 budget from the two measured throughput receipts.

AGENTS.md directive 12: no throughput, resource or budget number in a tracked
document is typed. The throughput receipts carry seconds and GiB; the plan needs
milliseconds, GPU-hours for the registered token budget, and the six-hour
capacity. That arithmetic lives here and its output is a receipt, so the document
regions can point at a receipt like every other generated number.

Usage:
    scripts/court python scripts/s39_throughput_summary.py \
        --out results/stage2/throughput-summary-<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY = "results/stage2/throughput-20261117T004502Z.json"
ADAPTER = "results/stage2/throughput-20261117T004619Z-adapter.json"
TOKEN_BUDGET = 20_000_000
TOKENS_PER_STEP = 2_048
SIX_HOURS_S = 6 * 3600


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load(rel: str) -> dict:
    with open(os.path.join(REPO, rel), "r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(rel: str) -> str:
    import hashlib

    handle = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as binary:
        for chunk in iter(lambda: binary.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def summarise(rel: str) -> dict:
    receipt = load(rel)
    throughput = receipt["throughput"]
    vram = receipt["vram"]
    return {
        "receipt": rel,
        "receipt_stamp": receipt["stamp"],
        "receipt_sha256": digest(rel),
        "tokens_per_second": throughput["optimiser_step_tokens_per_second"],
        "median_step_ms": throughput["optimiser_step_seconds"]["median"] * 1000.0,
        "p95_step_ms": throughput["optimiser_step_seconds"]["p95"] * 1000.0,
        "peak_allocated_gib": vram["peak_allocated_gib"],
        "peak_reserved_gib": vram["peak_reserved_gib"],
        "container_memory_peak_bytes": int(
            receipt["container_memory"]["memory_peak"] or 0),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    memory = summarise(MEMORY)
    adapter = summarise(ADAPTER)
    rate = adapter["tokens_per_second"]
    seconds = TOKEN_BUDGET / rate
    steps = TOKEN_BUDGET // TOKENS_PER_STEP
    payload = {
        "schema": "selvedge/stage2-throughput-summary/v1",
        "kind": "throughput-summary",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "memory": memory,
        "adapter": adapter,
        "budget": {
            "tokens": TOKEN_BUDGET,
            "tokens_per_step": TOKENS_PER_STEP,
            "rate_source": ADAPTER,
            "seconds": seconds,
            "hours": seconds / 3600.0,
            "six_hour_capacity_tokens": SIX_HOURS_S * rate,
            "realised_steps": steps,
            "realised_tokens": steps * TOKENS_PER_STEP,
            "shortfall_tokens": TOKEN_BUDGET - steps * TOKENS_PER_STEP,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, args.out)
    print(f"throughput-summary: memory {memory['tokens_per_second']:.1f} tok/s, "
          f"adapter {rate:.1f} tok/s, {TOKEN_BUDGET} tokens = "
          f"{seconds / 3600.0:.3f} h, six-hour capacity "
          f"{SIX_HOURS_S * rate:.0f} tokens -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
