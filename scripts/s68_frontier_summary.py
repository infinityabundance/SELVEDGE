#!/usr/bin/env python3
"""S68: summarise the external-frontier position check from scored receipts.

This reads only receipts that already exist. It produces one table with, per
model, the absolute accuracy on full GSM8K and MATH-500 under the SELVEDGE S54
instrument, a Wilson 95% interval (unpaired, because these are different
models), the licence and pinned revision. It never compares a published vendor
number with ours and never pools instruments.

Usage (CPU):
    SELVEDGE_MEM=2g scripts/court python -- \
      python3 scripts/s68_frontier_summary.py \
        --model champion=results/stage2/s67-final/soup-stage1.json \
        --model qwen3-0.6b=results/stage2/s68-frontier/qwen3-0.6b.json \
        --model reasonlite-0.6b=results/stage2/s68-frontier/reasonlite-0.6b.json \
        --model post-trained-0.8b=results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json \
        --inventory results/models/s68-frontier-inventory.json \
        --out results/stage2/s68-frontier/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SCHEMA = "selvedge/s68-frontier-summary/v1"
TASKS = ("gsm8k", "math500")


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


def wilson(correct: int, items: int) -> list[float]:
    """95% Wilson score interval for a binomial proportion."""
    if items <= 0:
        return [None, None]
    z = 1.959963984540054
    p = correct / items
    denom = 1.0 + z * z / items
    centre = (p + z * z / (2 * items)) / denom
    half = (z * math.sqrt(p * (1 - p) / items + z * z / (4 * items * items))
            / denom)
    return [centre - half, centre + half]


def task_row(receipt: dict, task: str) -> dict | None:
    for row in receipt.get("tasks") or []:
        if (row.get("task") == task and row.get("mode") == "non-thinking"
                and row.get("decoding") == "greedy"):
            return row
    return None


def read(path: str) -> dict:
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="label=receipt")
    parser.add_argument("--inventory",
                        default="results/models/s68-frontier-inventory.json")
    parser.add_argument("--out",
                        default="results/stage2/s68-frontier/summary.json")
    args = parser.parse_args(argv)

    inventory = read(args.inventory)
    identities = {item["id"]: item for item in inventory.get("models") or []}
    rows = []
    for text in args.model:
        if "=" not in text:
            raise SystemExit(f"s68-frontier: expected label=receipt, got {text!r}")
        label, path = text.split("=", 1)
        receipt = read(path)
        model_id = receipt.get("model", {}).get("id") if isinstance(
            receipt.get("model"), dict) else receipt.get("model_id")
        entry = {
            "label": label,
            "receipt": path,
            "receipt_sha256": sha256_file(os.path.join(ROOT, path)),
            "receipt_stamp": receipt.get("stamp"),
            "receipt_revision": receipt.get("revision"),
            "model_id": model_id,
            "model_revision": (receipt.get("model", {}).get("revision")
                               if isinstance(receipt.get("model"), dict)
                               else receipt.get("revision")),
            "tasks": {},
        }
        identity = identities.get(model_id) or {}
        entry["license"] = identity.get("license")
        entry["gated"] = identity.get("gated", False)
        for task in TASKS:
            row = task_row(receipt, task)
            if row is None:
                continue
            flexible = row.get("flexible") or {}
            correct = flexible.get("correct")
            items = flexible.get("items") or row.get("items")
            entry["tasks"][task] = {
                "items": items,
                "correct": correct,
                "accuracy": flexible.get("accuracy"),
                "wilson95": wilson(correct, items) if isinstance(
                    correct, int) and isinstance(items, int) and items else None,
                "strict_accuracy": (row.get("strict") or {}).get("accuracy"),
                "exact_accuracy": (row.get("exact") or {}).get("accuracy"),
                "generated_tokens": row.get("generated_tokens"),
                "wall_seconds": row.get("wall_seconds"),
                "tokens_per_second": row.get("tokens_per_second"),
            }
        rows.append(entry)

    payload = {
        "schema": SCHEMA,
        "kind": "external-frontier-position-check",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "inventory": {"path": args.inventory,
                      "sha256": sha256_file(os.path.join(ROOT, args.inventory))},
        "protocol": inventory.get("protocol"),
        "rows": rows,
        "notes": [
            "These are SELVEDGE S54 numbers (greedy, non-thinking, zero-shot CoT, "
            "4096-token cap, the S54 extraction).",
            "Published vendor numbers for these checkpoints use different "
            "prompts, shots, decoding, sampling and extraction; this table is "
            "our instrument only and is not a reproduction of any vendor claim.",
            "Intervals are Wilson binomial intervals; the models are different, "
            "so no paired interval is computed across them.",
        ],
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    for entry in rows:
        parts = []
        for task in TASKS:
            value = entry["tasks"].get(task)
            if value:
                parts.append(f"{task}={100 * value['accuracy']:.2f}%")
        print(f"s68-frontier: {entry['label']} " + " ".join(parts))
    print(f"s68-frontier: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
