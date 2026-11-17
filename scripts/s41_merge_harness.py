#!/usr/bin/env python3
"""Merge anchored-harness receipts into an existing held-out comparison receipt.

`scripts/s40_compare.py` writes both halves in one receipt when the per-arm
harness receipts already exist. The 2M-token pass runs the held-out instrument
first (it is the cheap signal check), so its receipt is written before any
anchored receipt exists. This script fills the `harness` section afterwards from
the same per-arm receipts the instrument would have read, without re-running the
held-out forward.

It never invents a score: an arm with no receipt is recorded as missing, exactly
as `s40_compare.harness_section` does.

Usage:
    python3 scripts/s41_merge_harness.py \
        --heldout-receipt results/stage2/s41-heldout-comparison.json \
        --harness-dir results/stage2/eval

The receipt is rewritten atomically in place unless --out names another path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import s40_compare as comparison  # noqa: E402


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-receipt", required=True)
    parser.add_argument("--harness-dir", default="results/stage2/eval")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    with open(args.heldout_receipt, encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("schema") != "selvedge/stage2-arm-comparison/v1":
        raise SystemExit(f"{args.heldout_receipt}: unexpected schema "
                         f"{receipt.get('schema')!r}")
    arms = receipt.get("selected_arms")
    reference = receipt.get("reference_arm")
    if not arms or reference not in arms:
        raise SystemExit("the held-out receipt does not name a selected reference")
    receipt["harness"] = comparison.harness_section(args.harness_dir, arms, reference)
    receipt["harness_merged_utc"] = utcnow()
    out = args.out or args.heldout_receipt
    comparison.write_json(out, receipt)
    status = receipt["harness"]["status"]
    missing = receipt["harness"].get("missing_arms") or []
    print(f"merge: harness {status}; missing {missing}; receipt {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
