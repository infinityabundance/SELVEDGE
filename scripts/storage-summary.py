#!/usr/bin/env python3
"""Aggregate repeated storage measurements into one variance-bearing summary.

The raw per-request JSONL written by `selvedge store bench --metrics-dir` is
large (tens of MiB per repeat) and is not tracked. This script reads only the
JSON receipts, which are tracked, and emits a small summary that records, for
every arm and pass, the median and range across repeats. The summary is the
single input for the measured storage tables in `docs/STORAGE.md`.

Usage:

    python3 scripts/storage-summary.py \
        --bundle 06b \
        --store results/store/store-bench-06b-r1.tmp.json \
                results/store/store-bench-06b-r2.tmp.json \
                results/store/store-bench-06b-r3.tmp.json \
        --io results/io/io-scan-06b-r1.tmp.json \
             results/io/io-scan-06b-r2.tmp.json \
             results/io/io-scan-06b-r3.tmp.json \
        --out results/store/summary-06b.json \
        --io-out results/io/summary-06b.json

No network, no third-party imports.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def stats(values: list[float]) -> dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "cv": None, "n": 0}
    median = statistics.median(clean)
    mean = statistics.fmean(clean)
    spread = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "median": median,
        "min": min(clean),
        "max": max(clean),
        "cv": (spread / mean) if mean else None,
        "n": len(clean),
    }


def pass_summary(passes: list[dict]) -> dict:
    return {
        "throughput_mib_s": stats([item.get("throughput_mib_s") for item in passes]),
        "bytes_used": stats([item.get("bytes_used") for item in passes]),
        "bytes_requested": stats([item.get("bytes_requested") for item in passes]),
        "bytes_completed": stats([item.get("bytes_completed") for item in passes]),
        "bytes_wasted": stats([item.get("bytes_wasted") for item in passes]),
        "device_bytes": stats([item.get("device_bytes") for item in passes]),
        "read_amplification": stats([item.get("read_amplification") for item in passes]),
        "stall_p50_ns": stats([item.get("stall_p50_ns") for item in passes]),
        "stall_p95_ns": stats([item.get("stall_p95_ns") for item in passes]),
        "stall_p99_ns": stats([item.get("stall_p99_ns") for item in passes]),
        "stall_max_ns": stats([item.get("stall_max_ns") for item in passes]),
        "verify_ns": stats([item.get("verify_ns") for item in passes]),
        "cache_hit_bytes": stats([item.get("cache_hit_bytes") for item in passes]),
        "hits": stats([item.get("hits") for item in passes]),
        "device_probe_mib_s": stats(
            [item.get("device_probe_mib_s") for item in passes]
        ),
        "page_cache_dropped": sorted(
            {bool(item.get("page_cache_dropped")) for item in passes}
        ),
    }


def summarize_store(paths: list[str], label: str) -> dict:
    receipts = [load(path) for path in paths]
    first = receipts[0]
    arms_by_id: dict[str, dict] = {}
    for receipt in receipts:
        for arm in receipt["arms"]:
            arms_by_id.setdefault(arm["arm"], {"pass1": [], "pass2": [], "notes": []})
            arms_by_id[arm["arm"]]["pass1"].append(arm["pass1"])
            if arm.get("pass2"):
                arms_by_id[arm["arm"]]["pass2"].append(arm["pass2"])
            arms_by_id[arm["arm"]]["notes"] = arm.get("notes", [])
    arms = []
    for arm in first["arms"]:
        collected = arms_by_id[arm["arm"]]
        entry = {
            "arm": arm["arm"],
            "policy": arm["policy"],
            "cache_configured_bytes": arm["cache_configured_bytes"],
            "cache_payload_share": arm["cache_payload_share"],
            "coalesced": arm["coalesced"],
            "skipped": arm.get("skipped"),
            "prefetch": {
                key: arm["store"].get(key)
                for key in (
                    "prefetch_requests",
                    "prefetch_admitted",
                    "prefetch_bytes_read",
                    "prefetch_bytes_used",
                    "prefetch_bytes_wasted",
                )
            },
            "pass1": pass_summary(collected["pass1"]),
            "pass2": pass_summary(collected["pass2"]) if collected["pass2"] else None,
            "notes": collected["notes"],
        }
        arms.append(entry)
    return {
        "schema": "selvedge/storage-summary/v1",
        "label": label,
        "repeats": len(receipts),
        "stamp": receipts[0].get("stamp"),
        "revision": receipts[0].get("revision"),
        "receipts": [
            {
                "path": path,
                "stamp": receipt.get("stamp"),
                "revision": receipt.get("revision"),
            }
            for path, receipt in zip(paths, receipts)
        ],
        "bundle": first["bundle"],
        "layout": first["layout"],
        "mount": first["mount"],
        "environment": first["environment"],
        "device_idle_delta": first.get("device_idle_delta"),
        "device_idle_window_ns": first.get("device_idle_window_ns"),
        "arms": arms,
        "notes": first.get("notes", []),
    }


def summarize_io(paths: list[str], label: str) -> dict:
    receipts = [load(path) for path in paths]
    first = receipts[0]

    def sequence(field: str, key: str) -> list[dict]:
        rows = []
        for index in range(len(first[field])):
            item = {"key": first[field][index][key]}
            for candidate in first[field][index]:
                if candidate == key:
                    continue
                values = [
                    receipt[field][index].get(candidate) for receipt in receipts
                ]
                if all(isinstance(value, (int, float)) for value in values):
                    item[candidate] = stats(values)
                else:
                    item[candidate] = first[field][index].get(candidate)
            rows.append(item)
        return rows

    checksum_metrics = {}
    for key in (
        "warm_throughput_mib_s",
        "warm_ns_per_gib",
        "cold_throughput_mib_s",
        "shard_hash_ns",
        "shard_bytes",
    ):
        checksum_metrics[key] = stats(
            [receipt["checksum"].get(key) for receipt in receipts]
        )
    return {
        "schema": "selvedge/io-summary/v1",
        "label": label,
        "repeats": len(receipts),
        "stamp": receipts[0].get("stamp"),
        "revision": receipts[0].get("revision"),
        "receipts": [
            {
                "path": path,
                "stamp": receipt.get("stamp"),
                "revision": receipt.get("revision"),
            }
            for path, receipt in zip(paths, receipts)
        ],
        "file": first["file"],
        "mount": first["mount"],
        "plan": first["plan"],
        "cache_control": first.get("cache_control"),
        "sequential": sequence("sequential", "block_bytes"),
        "sequential_noreadahead": sequence("sequential_noreadahead", "block_bytes"),
        "random": sequence("random", "block_bytes"),
        "queue_depth": sequence("queue_depth", "queue_depth"),
        "alignment": sequence("alignment", "label"),
        "checksum": checksum_metrics,
        "notes": first.get("notes", []),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--store", nargs="+", required=True)
    parser.add_argument("--io", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--io-out", required=True)
    args = parser.parse_args(argv)
    if len(args.store) < 3 or len(args.io) < 3:
        raise SystemExit("at least three repeats are required for a variance report")
    write_json(args.out, summarize_store(args.store, args.bundle))
    write_json(args.io_out, summarize_io(args.io, args.bundle))
    print(f"summary: {args.out}")
    print(f"summary: {args.io_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
