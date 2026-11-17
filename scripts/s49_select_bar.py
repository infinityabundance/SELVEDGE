#!/usr/bin/env python3
"""S49: mark the escalation attempt that completed, and verify its receipt.

The S49b amendment fixes the escalation order of the 20M attempts. This script
walks that order, reads each attempt's run directory, and writes
`results/stage2/runs/full-20m-math-s49-selected.json` naming the first attempt
that completed 19,998,720 tokens without an abort. It fails closed when:

* every earlier attempt has no recorded result (the order does not skip),
* the selected receipt's lr, steps, tokens or stream do not match the
  registration and the amendment,
* the selected receipt records a tripped tripwire,
* no attempt has completed.

The marker is what `scripts/s49_bar_curve.py` reads. It is not a new result: it
names which registered attempt is the bar.

Usage:
    scripts/court python -- python3 scripts/s49_select_bar.py
    scripts/court python -- python3 scripts/s49_select_bar.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

REGISTRATION = "docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
AMENDMENT = "docs/ws19-full-distillation-amendment-stability-s49b-preregistration-v1.json"
MARKER = "results/stage2/runs/full-20m-math-s49-selected.json"
SCHEMA = "selvedge/s49-bar-selected/v1"


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


def attempt_dir(registration: dict, order: list[str], position: int) -> str:
    base = registration["bar_rerun"]["run_dir"]
    if position == 0:
        return base
    suffix = order[position].replace("stability-", "").replace("-2m", "")
    return f"{base}-{suffix}"


def collect() -> dict:
    registration = load_json(os.path.join(REPO, REGISTRATION))
    amendment = load_json(os.path.join(REPO, AMENDMENT))
    order = amendment["selection_amendment"]["escalation_order"]
    arms = {entry["name"]: entry for entry in registration["stability_sweep"]["arms"]}
    if set(order) != set(arms):
        raise SystemExit("s49-select: the escalation order does not cover the registered arms")
    attempts = []
    selected = None
    for position, arm in enumerate(order):
        directory = attempt_dir(registration, order, position)
        abort_path = os.path.join(directory, "abort.json")
        receipt_path = os.path.join(directory, "receipt.json")
        entry = {
            "position": position,
            "arm": arm,
            "lr": arms[arm]["lr"],
            "run_dir": directory,
            "aborted": os.path.exists(abort_path),
            "completed": os.path.exists(receipt_path),
        }
        if entry["aborted"]:
            abort = load_json(abort_path)
            entry["abort"] = {
                "step": abort.get("step"),
                "tokens_seen": abort.get("tokens_seen"),
                "ratio": abort.get("ratio"),
                "stamp": abort.get("stamp"),
            }
        elif entry["completed"]:
            receipt = load_json(receipt_path)
            config = receipt.get("config") or {}
            if float(config.get("lr", -1)) != float(arms[arm]["lr"]):
                raise SystemExit(f"s49-select: {receipt_path} lr does not match {arm}")
            if int(config.get("steps", -1)) != 9765:
                raise SystemExit(f"s49-select: {receipt_path} steps are not 9,765")
            if int(receipt.get("tokens_seen_total", -1)) != 19_998_720:
                raise SystemExit(f"s49-select: {receipt_path} tokens are not 19,998,720")
            if (receipt.get("tripwire") or {}).get("tripped"):
                raise SystemExit(f"s49-select: {receipt_path} records a tripped tripwire")
            entry["receipt"] = receipt_path
            entry["receipt_stamp"] = receipt.get("stamp")
            entry["tokens_per_second"] = receipt.get("tokens_per_second")
            entry["peak_allocated_gib"] = receipt.get("peak_allocated_gib")
            entry["cgroup_peak_bytes"] = (receipt.get("container_memory") or {}).get(
                "memory_peak")
            entry["final_checkpoint_sha256"] = receipt.get("final_checkpoint_sha256")
            selected = entry
        attempts.append(entry)
        if selected is not None:
            break
        if not entry["aborted"]:
            raise SystemExit(
                f"s49-select: attempt {arm} has no recorded result; the registered "
                "order does not skip rates")
    if selected is None:
        raise SystemExit("s49-select: no escalation attempt has completed")
    return {
        "schema": SCHEMA,
        "kind": "s49-bar-selected",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": {
            "path": REGISTRATION,
            "sha256": hashlib.sha256(
                open(os.path.join(REPO, REGISTRATION), "rb").read()).hexdigest(),
        },
        "amendment": {
            "path": AMENDMENT,
            "sha256": hashlib.sha256(
                open(os.path.join(REPO, AMENDMENT), "rb").read()).hexdigest(),
        },
        "escalation_order": order,
        "attempts": attempts,
        "selected": {
            "arm": selected["arm"],
            "lr": selected["lr"],
            "run_dir": selected["run_dir"],
            "receipt": selected["receipt"],
            "rule": "first entry of the registered escalation order whose 20M attempt completed without an abort",
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = collect()
    path = os.path.join(REPO, MARKER)
    if args.check:
        if not os.path.exists(path):
            print(f"s49-select: missing marker {MARKER}", file=sys.stderr)
            return 1
        current = load_json(path)
        if current.get("selected") != payload["selected"]:
            print("s49-select: the selection marker is stale; rerun without --check",
                  file=sys.stderr)
            return 1
        print(f"s49-select: marker matches: {payload['selected']['run_dir']}")
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s49-select: {MARKER}")
    print(f"s49-select: selected {payload['selected']['arm']} lr={payload['selected']['lr']} "
          f"in {payload['selected']['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
