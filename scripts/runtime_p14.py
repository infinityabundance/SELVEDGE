#!/usr/bin/env python3
"""P14 court (real model, CPU): residency, prefetch overlap and batched prefill.

Reads the ``selvedge run --json`` receipts written by
``scripts/runtime-p14-courts.sh`` and writes one table-shaped receipt. Every
number is a measurement from those runs; nothing is modelled.

What this receipt can and cannot show
-------------------------------------
* **Decode passes per token** and **residency hit rate** are counters. They are
  exact.
* **Wall clock, p50/p95 and tokens/s** are CPU measurements on one host. The
  storage is a local NVMe device and the storage cache is RAM, so the latency
  an asynchronous prefetcher can hide on a *warm* run is small. The delayed
  arms inject a per-read delay to make the overlap measurable; the receipt
  reports the injected value next to every number that used it.
* **Batched prefill speedup** is measured at prompt lengths 8, 32 and 128 with
  the scalar path as the oracle, and every batched run must reproduce the
  scalar sequence digest exactly.
* No GPU number is claimed anywhere. GPU overlap and GPU residency are part 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "selvedge/runtime-p14/v1"


def load(directory: Path, name: str, stamp: str) -> dict:
    path = directory / f"{name}-{stamp}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "selvedge/runtime-run/v1":
        raise SystemExit(f"{path} is not a runtime run receipt")
    document["_receipt"] = str(path)
    peak_path = directory / f"{name}-{stamp}.peak"
    document["_peak_bytes"] = (
        int(peak_path.read_text().strip()) if peak_path.exists() else None
    )
    return document


def rows_from_accounting(directory: Path, name: str, stamp: str) -> list[dict]:
    path = directory / f"{name}-{stamp}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean_int(values: list[int]) -> int:
    return int(sum(values) / len(values)) if values else 0


def residency_row(directory: Path, name: str, stamp: str) -> dict:
    run = load(directory, name, stamp)
    rows = rows_from_accounting(directory, name, stamp)
    decodes = [row["decode"]["decodes"] for row in rows]
    hits = [row["decode"]["hits"] for row in rows]
    lookups = [row["decode"]["lookups"] for row in rows]
    tokens = max(1, run["totals"]["tokens"])
    return {
        "name": name,
        "receipt": run["_receipt"],
        "decode_cache_bytes": run["stream"]["decode_cache_bytes"],
        "cache_bytes": run["stream"]["cache_bytes"],
        "threads": run["environment"]["threads"],
        "tokens": run["totals"]["tokens"],
        "decodes_total": sum(decodes),
        "decodes_per_token": sum(decodes) / tokens,
        "decodes_per_token_warm": mean_int(decodes[1:]),
        "decode_hits_total": sum(hits),
        "decode_hit_rate": (sum(hits) / sum(lookups)) if sum(lookups) else 0.0,
        "prefill_p50_ns": run["prefill"]["p50_ns"],
        "decode_p50_ns": run["decode"]["p50_ns"],
        "decode_p95_ns": run["decode"]["p95_ns"],
        "decode_tokens_per_second": run["decode_tokens_per_second"],
        "bytes_device_total": run["totals"]["bytes_device"],
        "read_calls_per_token": run["totals"]["read_calls"] // tokens,
        "residency_peak_bytes": run["environment"]["decode_residency"]["peak_bytes"],
        "residency_entries": run["environment"]["decode_residency"]["entries"],
        "evictions": run["environment"]["decode_residency"]["evictions"],
        "memory_peak_bytes": run["_peak_bytes"],
        "sequence_digest": run["sequence_digest"],
        "exactness": run["exactness"]["status"],
    }


def prefetch_row(directory: Path, name: str, stamp: str) -> dict:
    run = load(directory, name, stamp)
    rows = rows_from_accounting(directory, name, stamp)
    prefetch = [row.get("prefetch") for row in rows]
    prefetch = [item for item in prefetch if item]
    stall_p50 = [item["stall_p50_ns"] for item in prefetch]
    stall_p95 = [item["stall_p95_ns"] for item in prefetch]
    exposed = [item["exposed_stall_ns"] for item in prefetch]
    hidden = [item["hidden_ns"] for item in prefetch]
    wall = [row["elapsed_ns"] for row in rows]
    tokens = max(1, run["totals"]["tokens"])
    return {
        "name": name,
        "receipt": run["_receipt"],
        "async_prefetch": run["stream"]["async_prefetch"],
        "sync_prefetch": run["stream"]["prefetch"],
        "prefetch_depth": run["stream"]["prefetch_depth"],
        "cache_bytes": run["stream"]["cache_bytes"],
        "injected_read_delay_ns": run["stream"]["injected_read_delay_ns"],
        "tokens": run["totals"]["tokens"],
        "fetched_bytes": sum(item["fetched_bytes"] for item in prefetch),
        "used_bytes": sum(item["used_bytes"] for item in prefetch),
        "wasted_bytes": sum(item["wasted_bytes"] for item in prefetch),
        "device_bytes": sum(item["device_bytes"] for item in prefetch),
        "exposed_stall_ns_total": sum(exposed),
        "exposed_stall_ns_per_token": sum(exposed) / tokens,
        "hidden_ns_total": sum(hidden),
        "hidden_ns_per_token": sum(hidden) / tokens,
        "stall_p50_ns_mean": mean_int(stall_p50),
        "stall_p95_ns_mean": mean_int(stall_p95),
        "stall_max_ns": max(stall_p95) if stall_p95 else 0,
        "staging_high_water_bytes": max(
            (item["staging_high_water_bytes"] for item in prefetch), default=0
        ),
        "wall_ns_total": sum(wall),
        "wall_p50_ns": sorted(wall)[len(wall) // 2] if wall else 0,
        "decode_p50_ns": run["decode"]["p50_ns"],
        "decode_p95_ns": run["decode"]["p95_ns"],
        "decode_tokens_per_second": run["decode_tokens_per_second"],
        "memory_peak_bytes": run["_peak_bytes"],
        "sequence_digest": run["sequence_digest"],
        "exactness": run["exactness"]["status"],
    }


def prefill_row(directory: Path, name: str, stamp: str) -> dict:
    run = load(directory, name, stamp)
    return {
        "name": name,
        "receipt": run["_receipt"],
        "prefill_mode": run["stream"]["prefill"],
        "prefill_batch": run["stream"]["prefill_batch"],
        "prompt_tokens": len(run["prompt_tokens"]),
        "prefill_p50_ns": run["prefill"]["p50_ns"],
        "prefill_p95_ns": run["prefill"]["p95_ns"],
        "prefill_tokens_per_second": run["prefill_tokens_per_second"],
        "decode_cache_bytes": run["stream"]["decode_cache_bytes"],
        "memory_peak_bytes": run["_peak_bytes"],
        "sequence_digest": run["sequence_digest"],
        "logits_digests": run["logits_digests"],
        "exactness": run["exactness"]["status"],
    }


def all_equal(values: list[str]) -> bool:
    return len(set(values)) <= 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--dir", default="results/runtime")
    parser.add_argument("--residency", default="")
    parser.add_argument("--prefetch", default="")
    parser.add_argument("--prefill", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    directory = Path(args.dir)
    names = lambda value: [item for item in value.split(",") if item]  # noqa: E731

    residency = [residency_row(directory, name, args.stamp) for name in names(args.residency)]
    prefetch = [prefetch_row(directory, name, args.stamp) for name in names(args.prefetch)]
    prefill = [prefill_row(directory, name, args.stamp) for name in names(args.prefill)]

    # Prefill speedup per prompt length, scalar as the oracle.
    speedups = []
    lengths = sorted({row["prompt_tokens"] for row in prefill})
    for length in lengths:
        scalar = next(
            (
                row
                for row in prefill
                if row["prompt_tokens"] == length and row["prefill_mode"] == "scalar"
            ),
            None,
        )
        batched = next(
            (
                row
                for row in prefill
                if row["prompt_tokens"] == length and row["prefill_mode"] == "batched"
            ),
            None,
        )
        if not scalar or not batched:
            continue
        speedups.append(
            {
                "prompt_tokens": length,
                "scalar_prefill_p50_ns": scalar["prefill_p50_ns"],
                "batched_prefill_p50_ns": batched["prefill_p50_ns"],
                "speedup": (
                    scalar["prefill_p50_ns"] / batched["prefill_p50_ns"]
                    if batched["prefill_p50_ns"]
                    else None
                ),
                "scalar_tokens_per_second": scalar["prefill_tokens_per_second"],
                "batched_tokens_per_second": batched["prefill_tokens_per_second"],
                "same_logits": scalar["logits_digests"] == batched["logits_digests"],
                "same_sequence_digest": scalar["sequence_digest"] == batched["sequence_digest"],
            }
        )

    import subprocess

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    document = {
        "schema": SCHEMA,
        "stamp": args.stamp,
        "revision": revision,
        "claim_scope": (
            "CPU f32 measurements on one host; no GPU number is claimed, and no GPU "
            "overlap is claimed"
        ),
        "reduction_order": (
            "fixed: eight interleaved accumulators over k per output element, rows only "
            "parallelised; the batched path reuses the same dot per row"
        ),
        "residency": {
            "rows": residency,
            "same_sequence_digest": all_equal([row["sequence_digest"] for row in residency]),
        },
        "prefetch": {
            "rows": prefetch,
            "same_sequence_digest": all_equal([row["sequence_digest"] for row in prefetch]),
            "byte_accounting_conserves": all(
                row["fetched_bytes"] == row["used_bytes"] + row["wasted_bytes"]
                for row in prefetch
            ),
        },
        "prefill": {
            "rows": prefill,
            "speedups": speedups,
            "same_logits_at_every_length": all(item["same_logits"] for item in speedups),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    if residency:
        print(
            f"{'residency arm':<26}{'budget_MiB':>11}{'dec/tok':>9}{'warm':>7}"
            f"{'hit':>7}{'dec_p50_ms':>12}{'tok/s':>7}"
        )
        for row in residency:
            print(
                f"{row['name']:<26}{row['decode_cache_bytes'] / 2**20:>11.0f}"
                f"{row['decodes_per_token']:>9.1f}{row['decodes_per_token_warm']:>7}"
                f"{row['decode_hit_rate']:>7.2f}"
                f"{row['decode_p50_ns'] / 1e6:>12.1f}{row['decode_tokens_per_second']:>7.2f}"
            )
    if prefetch:
        print(
            f"{'prefetch arm':<26}{'delay_us':>9}{'stall_p50_us':>13}{'stall_p95_us':>13}"
            f"{'hidden_ms':>11}{'dec_p50_ms':>12}{'tok/s':>7}"
        )
        for row in prefetch:
            print(
                f"{row['name']:<26}{row['injected_read_delay_ns'] / 1e3:>9.0f}"
                f"{row['stall_p50_ns_mean'] / 1e3:>13.0f}"
                f"{row['stall_p95_ns_mean'] / 1e3:>13.0f}"
                f"{row['hidden_ns_total'] / 1e6:>11.2f}"
                f"{row['decode_p50_ns'] / 1e6:>12.1f}"
                f"{row['decode_tokens_per_second']:>7.2f}"
            )
    if speedups:
        print(
            f"{'prefill':<14}{'scalar_ms':>12}{'batched_ms':>12}{'speedup':>9}"
            f"{'same_logits':>13}"
        )
        for item in speedups:
            print(
                f"len={item['prompt_tokens']:<9}"
                f"{item['scalar_prefill_p50_ns'] / 1e6:>12.1f}"
                f"{item['batched_prefill_p50_ns'] / 1e6:>12.1f}"
                f"{item['speedup']:>9.2f}"
                f"{str(item['same_logits']):>13}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
