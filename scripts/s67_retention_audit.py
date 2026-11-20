#!/usr/bin/env python3
"""S67: retention audit for the selected soup on the registered instruments.

Pairs the candidate and the post-trained baseline raw lm-evaluation-harness
dumps three ways for full MMLU:

* ``full``      -- every paired item (14,042);
* ``selection`` -- the 2,000 items the S60 coefficient selection read;
* ``unused``    -- the complement, never read during selection;

and once for the registered 4,000-item MMLU-Pro total-cap instrument. The
registered bound is a drop bound: point drop <= 1.0 point and paired interval
lower bound >= -1.0 point; an improvement is inside the bound.

Arithmetic is imported from ``scripts/s66_retention_audit.py`` (same bootstrap,
seed, resampling unit and verdict rule), not re-implemented.

Usage (CPU):
    SELVEDGE_MEM=4g scripts/court python -- \
      python3 scripts/s67_retention_audit.py \
        --candidate-mmlu  <soup raw mmlu.json> \
        --candidate-mmlu-pro <soup raw mmlu_pro.json> \
        --baseline-mmlu <post-trained raw mmlu.json> \
        --baseline-mmlu-pro <post-trained 4000-item raw mmlu_pro.json> \
        --selection-mmlu <s60 selection raw mmlu.json> \
        --out results/stage2/s67-final/retention-audit.json
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
sys.path.insert(0, HERE)

import s66_retention_audit as audit  # noqa: E402

SCHEMA = "selvedge/s67-retention-audit/v1"
PREREG = "docs/s67-resolution-instrument-preregistration-v1.json"


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-mmlu", required=True)
    parser.add_argument("--candidate-mmlu-pro", required=True)
    parser.add_argument("--baseline-mmlu", required=True)
    parser.add_argument("--baseline-mmlu-pro", required=True)
    parser.add_argument("--selection-mmlu", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    candidate = audit.rows(audit.load_raw(args.candidate_mmlu), "acc")
    baseline = audit.rows(audit.load_raw(args.baseline_mmlu), "acc")
    selection = audit.rows(audit.load_raw(args.selection_mmlu), "acc")
    selection_keys = set(selection)
    if not selection_keys <= set(candidate):
        raise SystemExit(
            "s67-retention: the 2,000 selection items are not a subset of the "
            "full MMLU candidate dump")
    unused = set(candidate) - selection_keys

    mmlu = {
        "full": audit.compare("acc", candidate, baseline),
        "selection": audit.compare("acc", candidate, baseline, selection_keys),
        "unused": audit.compare("acc", candidate, baseline, unused),
        "counts": {
            "full": len(set(candidate) & set(baseline)),
            "selection": len(selection_keys),
            "unused": len(unused),
        },
    }

    pro_candidate = audit.rows(audit.load_raw(args.candidate_mmlu_pro),
                               "exact_match")
    pro_baseline = audit.rows(audit.load_raw(args.baseline_mmlu_pro),
                              "exact_match")
    pro = audit.compare("exact_match", pro_candidate, pro_baseline)
    pro["sample_size"] = len(pro_candidate)
    pro["registered_sample_size"] = 4000
    pro["registered_instrument_present"] = len(pro_candidate) == 4000
    if not pro["registered_instrument_present"]:
        pro["verdict"] = {
            "state": "unresolved",
            "why": (f"the registered instrument is a 4,000-item total-cap "
                    f"subset; this dump has {len(pro_candidate)} items"),
        }

    mmlu_pass = (mmlu["full"]["verdict"]["state"] == "pass"
                 and mmlu["unused"]["verdict"]["state"] == "pass")
    pro_pass = pro["verdict"]["state"] == "pass"
    payload = {
        "schema": SCHEMA,
        "kind": "retention-audit",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": PREREG,
            "sha256": audit.sha256_file(os.path.join(ROOT, PREREG)),
        },
        "inputs": {
            "candidate_mmlu": args.candidate_mmlu,
            "candidate_mmlu_sha256": audit.sha256_file(
                os.path.join(ROOT, args.candidate_mmlu)),
            "candidate_mmlu_pro": args.candidate_mmlu_pro,
            "candidate_mmlu_pro_sha256": audit.sha256_file(
                os.path.join(ROOT, args.candidate_mmlu_pro)),
            "baseline_mmlu": args.baseline_mmlu,
            "baseline_mmlu_sha256": audit.sha256_file(
                os.path.join(ROOT, args.baseline_mmlu)),
            "baseline_mmlu_pro": args.baseline_mmlu_pro,
            "baseline_mmlu_pro_sha256": audit.sha256_file(
                os.path.join(ROOT, args.baseline_mmlu_pro)),
            "selection_mmlu": args.selection_mmlu,
            "selection_mmlu_sha256": audit.sha256_file(
                os.path.join(ROOT, args.selection_mmlu)),
        },
        "metric": "MMLU acc; MMLU-Pro exact_match; paired bootstrap over items",
        "mmlu": mmlu,
        "mmlu_pro": pro,
        "verdict": {
            "mmlu_pass": mmlu_pass,
            "mmlu_pro_4000_pass": pro_pass,
            "retention_pass": bool(mmlu_pass and pro_pass),
            "rule": ("point drop <= 1.0 point and paired interval lower bound "
                     ">= -1.0 point, on full MMLU, the 12,042 unused MMLU "
                     "items, and the 4,000-item MMLU-Pro instrument"),
        },
        "claims": {
            "selection_items_are_a_subset_of_full": True,
            "unused_partition_is_the_complement": True,
            "no_item_is_counted_in_both_partitions": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(out + ".tmp", out)
    print(f"s67-retention: mmlu={mmlu['full']['verdict']['state']} "
          f"unused={mmlu['unused']['verdict']['state']} "
          f"mmlu_pro={pro['verdict']['state']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
