#!/usr/bin/env python3
"""Court F (real model, CPU): cold and warm latency, throughput and bytes/token.

Reads the ``selvedge run --json`` receipts written by
``scripts/runtime-courts.sh court-f`` and writes one table-shaped receipt. Every
number here is a measurement from that run; nothing is modelled.

This is a **CPU baseline**. No GPU number is claimed anywhere in this receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "selvedge/runtime-court-f/v1"


def load(receipt: Path) -> dict:
    document = json.loads(receipt.read_text(encoding="utf-8"))
    if document.get("schema") != "selvedge/runtime-run/v1":
        raise SystemExit(f"{receipt} is not a runtime run receipt")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--dir", default="results/runtime")
    parser.add_argument("--names", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.dir)
    rows = []
    for name in [name for name in args.names.split(",") if name]:
        receipt = out_dir / f"{name}-{args.stamp}.json"
        run = load(receipt)
        peak_path = out_dir / f"{name}-{args.stamp}.peak"
        peak = int(peak_path.read_text().strip()) if peak_path.exists() else None
        tokens = max(1, run["totals"]["tokens"])
        rows.append(
            {
                "name": name,
                "receipt": str(receipt),
                "threads": run["environment"]["threads"],
                "policy": run["stream"]["policy"],
                "cache_bytes": run["stream"]["cache_bytes"],
                "prompt_tokens": len(run["prompt_tokens"]),
                "generated_tokens": len(run["generated_tokens"]),
                "prefill_p50_ns": run["prefill"]["p50_ns"],
                "prefill_p95_ns": run["prefill"]["p95_ns"],
                "decode_p50_ns": run["decode"]["p50_ns"],
                "decode_p95_ns": run["decode"]["p95_ns"],
                "decode_p99_ns": run["decode"]["p99_ns"],
                "decode_tokens_per_second": run["decode_tokens_per_second"],
                "bytes_device_total": run["totals"]["bytes_device"],
                "bytes_device_per_token": run["totals"]["bytes_device"] // tokens,
                "reads_per_token": run["totals"]["read_calls"] // tokens,
                "cache_hits": run["totals"]["cache_hits"],
                "active_unique_max": run["totals"]["active_unique_max"],
                "flops_per_token": run["totals"]["flops_sum"] // tokens,
                "total_accessible_elements": run["total_accessible_elements"],
                "payload_bytes": run["payload_bytes"],
                "kv_cache_bytes": run["environment"]["kv_cache_bytes"],
                "memory_peak_bytes": peak,
                "sequence_digest": run["sequence_digest"],
                "exactness": run["exactness"]["status"],
            }
        )

    # Residency effect: the same work with and without the bundle resident.
    residency = None
    cold = next((row for row in rows if row["name"].endswith("-cold")), None)
    warm = next((row for row in rows if row["name"].endswith("-warm")), None)
    if cold and warm:
        residency = {
            "device_bytes_per_token_cold": cold["bytes_device_per_token"],
            "device_bytes_per_token_warm": warm["bytes_device_per_token"],
            "device_byte_reduction_factor": (
                cold["bytes_device_per_token"] / warm["bytes_device_per_token"]
                if warm["bytes_device_per_token"]
                else None
            ),
            "decode_p50_ratio_cold_over_warm": (
                cold["decode_p50_ns"] / warm["decode_p50_ns"] if warm["decode_p50_ns"] else None
            ),
            "same_sequence_digest": cold["sequence_digest"] == warm["sequence_digest"],
        }

    document = {
        "schema": SCHEMA,
        "stamp": args.stamp,
        "revision": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "claim_scope": "CPU f32 baseline on one device; no GPU number is claimed",
        "kernel": "in-repo f32 dot product, eight interleaved accumulators, rows parallelised",
        "reduction_order": "fixed; independent of thread count and cache state",
        "rows": rows,
        "residency": residency,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'variant':<18}{'thr':>4}{'prefill_p50_ms':>16}{'decode_p50_ms':>15}"
        f"{'tok/s':>8}{'dev_MiB/tok':>13}{'peak_GiB':>10}"
    )
    for row in rows:
        print(
            f"{row['name']:<18}{row['threads']:>4}"
            f"{row['prefill_p50_ns'] / 1e6:>16.1f}"
            f"{row['decode_p50_ns'] / 1e6:>15.1f}"
            f"{row['decode_tokens_per_second']:>8.2f}"
            f"{row['bytes_device_per_token'] / 1048576:>13.1f}"
            f"{(row['memory_peak_bytes'] or 0) / 2**30:>10.2f}"
        )
    if residency:
        print(json.dumps(residency, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
