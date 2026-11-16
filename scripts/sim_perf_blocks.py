#!/usr/bin/env python3
"""Derive the simulator's perf-number manifest blocks from its receipts.

Directive 12: every published number is emitted from a receipt. The simulator
already writes receipts, so the manifest entries are *derived* here rather than
typed by hand, including each receipt's stamp and revision.

Usage (inside the python court):

    python3 scripts/sim_perf_blocks.py            # rewrite the manifest
    python3 scripts/sim_perf_blocks.py --check    # fail if the file would change

Only blocks whose id starts with `sim-` are touched; every other block is
preserved verbatim and the array is written sorted by id, which is how the file
is already laid out.
"""

from __future__ import annotations

import argparse
import json
import sys

MANIFEST = "docs/perf-numbers.json"
DOC = "docs/SIMULATOR.md"
CAL06 = "results/sim/calibration-06b.json"
VAL_LRU1OF2 = "results/sim/validation-06b-lru-1of2.json"
VAL_LS = "results/sim/validation-06b-least-stale-1of4.json"
VAL_NONE = "results/sim/validation-06b-none-cold.json"
VAL17 = "results/sim/validation-17b-lru-1of2.json"
COMPARE = "results/sim/compare-store-06b.json"
PARETO = "results/sim/pareto-m2-17b.json"
PLACEMENT = "results/sim/placement-synthetic-alternating.json"
ENVELOPE = "results/sim/envelope-utility-synthetic-alternating.json"
SIMULATE = "results/sim/simulate-m2-17b-lru025.json"

# (id, receipt, pointer, format)
BLOCKS = [
    ("sim-cal-transfer-p50", CAL06, "/transfer/p50", "{value}"),
    ("sim-cal-transfer-p95", CAL06, "/transfer/p95", "{value}"),
    ("sim-cal-verify-p50", CAL06, "/verify/p50", "{value}"),
    ("sim-cal-hit-copy-p50", CAL06, "/hit_copy/p50", "{value}"),
    ("sim-cal-transfer-exp-maxerr", CAL06, "/fits/0/max_abs_rel_error", "{value:.4f}"),
    ("sim-cal-transfer-lognormal-maxerr", CAL06, "/fits/1/max_abs_rel_error", "{value:.4f}"),
    ("sim-cal-verify-exp-maxerr", CAL06, "/fits/2/max_abs_rel_error", "{value:.4f}"),
    ("sim-cal-verify-lognormal-maxerr", CAL06, "/fits/3/max_abs_rel_error", "{value:.4f}"),
    ("sim-val-06b-lru1of2-p50", VAL_LRU1OF2, "/stall_discrepancy_pct/p50", "{value:+.2f}%"),
    ("sim-val-06b-lru1of2-p95", VAL_LRU1OF2, "/stall_discrepancy_pct/p95", "{value:+.2f}%"),
    ("sim-val-06b-lru1of2-p99", VAL_LRU1OF2, "/stall_discrepancy_pct/p99", "{value:+.2f}%"),
    ("sim-val-06b-lru1of2-bytes", VAL_LRU1OF2, "/bytes/discrepancy_pct", "{value:+.2f}%"),
    ("sim-val-06b-lru1of2-mismatches", VAL_LRU1OF2, "/hit_mismatches", "{value}"),
    ("sim-val-06b-leaststale-p50", VAL_LS, "/stall_discrepancy_pct/p50", "{value:+.2f}%"),
    ("sim-val-06b-leaststale-p95", VAL_LS, "/stall_discrepancy_pct/p95", "{value:+.2f}%"),
    ("sim-val-06b-leaststale-p99", VAL_LS, "/stall_discrepancy_pct/p99", "{value:+.2f}%"),
    ("sim-val-06b-none-p50", VAL_NONE, "/stall_discrepancy_pct/p50", "{value:+.2f}%"),
    ("sim-val-06b-none-p95", VAL_NONE, "/stall_discrepancy_pct/p95", "{value:+.2f}%"),
    ("sim-val-06b-none-p99", VAL_NONE, "/stall_discrepancy_pct/p99", "{value:+.2f}%"),
    ("sim-val-17b-lru1of2-p50", VAL17, "/stall_discrepancy_pct/p50", "{value:+.2f}%"),
    ("sim-val-17b-lru1of2-p95", VAL17, "/stall_discrepancy_pct/p95", "{value:+.2f}%"),
    ("sim-val-17b-lru1of2-p99", VAL17, "/stall_discrepancy_pct/p99", "{value:+.2f}%"),
    ("sim-compare-cold-abs-p50", COMPARE, "/cold/p50_discrepancy_pct", "{value:+.2f}%"),
    ("sim-compare-cold-shape-p95", COMPARE, "/cold/shape_p95_discrepancy_pct", "{value:+.2f}%"),
    ("sim-compare-cold-bytes", COMPARE, "/cold/bytes_discrepancy_pct", "{value:+.2f}%"),
    ("sim-compare-warm-shape-p95", COMPARE, "/warm/shape_p95_discrepancy_pct", "{value:+.2f}%"),
    ("sim-pareto-none-bytes", PARETO, "/rows/0/bytes_per_record", "{value:.0f} B"),
    ("sim-pareto-lru05-bytes", PARETO, "/rows/3/bytes_per_record", "{value:.0f} B"),
    ("sim-pareto-lru05-p50", PARETO, "/rows/3/p50_stall_ns", "{value}"),
    ("sim-pareto-lru05-hitrate", PARETO, "/rows/3/hit_rate", "{value:.4f}"),
    ("sim-pareto-lru10-p50", PARETO, "/rows/4/p50_stall_ns", "{value}"),
    ("sim-pareto-lru10-p95", PARETO, "/rows/4/p95_stall_ns", "{value}"),
    ("sim-placement-seq-p95", PLACEMENT, "/rows/0/p95_stall_ns", "{value}"),
    ("sim-placement-coaccess-p95", PLACEMENT, "/rows/2/p95_stall_ns", "{value}"),
    ("sim-placement-replicated-p95", PLACEMENT, "/rows/3/p95_stall_ns", "{value}"),
    ("sim-placement-replicated-amp", PLACEMENT, "/rows/3/amplification", "{value:.4f}"),
    ("sim-envelope-bytes-saved", ENVELOPE, "/utility/bytes_not_transferred", "{value}"),
    ("sim-envelope-time-saved", ENVELOPE, "/utility/exposed_time_saved_ns", "{value}"),
    ("sim-envelope-cold-charged", ENVELOPE, "/utility/cold_bytes_charged", "{value}"),
    ("sim-envelope-unused", ENVELOPE, "/utility/unused_prefetch_bytes", "{value}"),
    ("sim-envelope-invalidations", ENVELOPE, "/utility/invalidations", "{value}"),
    ("sim-simulate-m2-p95", SIMULATE, "/summary/stall/p95_ns", "{value}"),
    ("sim-simulate-m2-hit-bytes", SIMULATE, "/summary/bytes_hit", "{value}"),
]


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = load(MANIFEST)
    receipts: dict[str, dict] = {}
    blocks = []
    for block in manifest["blocks"]:
        if block["id"].startswith("sim-"):
            continue
        blocks.append(block)
    for block_id, receipt_path, pointer, fmt in BLOCKS:
        receipt = receipts.setdefault(receipt_path, load(receipt_path))
        stamp = receipt.get("stamp")
        revision = receipt.get("revision")
        if not isinstance(stamp, str) or not stamp:
            raise SystemExit(f"{receipt_path} has no stamp")
        if not isinstance(revision, str) or not revision:
            raise SystemExit(f"{receipt_path} has no revision")
        blocks.append(
            {
                "declared_receipt_stamp": stamp,
                "declared_revision": revision,
                "doc": DOC,
                "format": fmt,
                "id": block_id,
                "sentinel": block_id,
                "source": {"pointer": pointer, "receipt": receipt_path},
            }
        )
    seen = set()
    for block in blocks:
        if block["id"] in seen:
            raise SystemExit(f"duplicate block id {block['id']}")
        seen.add(block["id"])
    # Preserve the existing order (the file is grouped by phase, not sorted) and
    # append the simulator blocks at the end so the diff is an insertion only.
    manifest["blocks"] = blocks
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        current = handle.read()
    if args.check:
        if current != rendered:
            print("manifest is stale; run without --check", file=sys.stderr)
            return 1
        print("manifest is current")
        return 0
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    print(f"wrote {len(blocks)} blocks ({len(BLOCKS)} simulator blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
