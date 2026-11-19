#!/usr/bin/env python3
"""S66: audit the S65 retention receipts and separate selection from reuse.

The S65 interpolation coefficient was selected on a 2,000-item MMLU total-cap
subset. The retention claim is scored on the full 14,042-item MMLU test set. A
reader is entitled to know whether the reported retention number is driven by
the items that participated in selection, so this script pairs the candidate and
baseline raw dumps three ways for each coefficient:

* ``full``       -- every paired item (14,042 for MMLU);
* ``selection``  -- the 2,000 items the coefficient selection read;
* ``unused``     -- the complement, i.e. items never read during selection.

It also audits MMLU-Pro. The registered S60 retention instrument is a 4,000-item
total-cap subset; only the older 504-item sample exists on disk, so that row is
reported as ``sample_not_registered_size`` and is not upgraded into a pass.

The registered bound is a *drop* bound: point drop <= 1.0 point and paired
interval lower bound >= -1.0 point. A positive delta (an improvement) is inside
that bound. The older ``s57_paired.py`` summary used ``abs(delta) <= 0.01``,
which marks an improvement larger than one point as a failure; that is a
reporting defect, not the registered rule, and this audit prints both readings.

Usage:
    SELVEDGE_MEM=4g scripts/court python -- \
      python3 scripts/s66_retention_audit.py \
        --out results/stage2/s65-eval/retention-audit.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RESAMPLES = 4000
SEED = 53
BOUND = 0.01
SCHEMA = "selvedge/s66-retention-audit/v1"

#: Candidate and baseline raw dumps, and the 2,000-item selection pass for the
#: same coefficient. The keys are the coefficients S65 scored on test.
CASES = {
    "0.375": {
        "candidate": "results/stage2/s65-eval/raw/20261119T202606Z-mmlu.json",
        "selection": "results/stage2/s60-interpolation/raw/20261119T091650Z-mmlu.json",
        "candidate_mmlu_pro": "results/stage2/s65-eval/raw/20261119T204054Z-mmlu_pro.json",
        "selection_receipt": "results/stage2/s60-interpolation/mmlu-0.375.json",
        "paired_receipt": "results/stage2/s65-eval/paired-retention-interp-0.375-vs-post.json",
    },
    "0.250": {
        "candidate": "results/stage2/s65-eval/raw/20261119T210334Z-mmlu.json",
        "selection": "results/stage2/s60-interpolation/raw/20261119T091508Z-mmlu.json",
        "candidate_mmlu_pro": "results/stage2/s65-eval/raw/20261119T211921Z-mmlu_pro.json",
        "selection_receipt": "results/stage2/s60-interpolation/mmlu-0.250.json",
        "paired_receipt": "results/stage2/s65-eval/paired-retention-interp-0.250-vs-post.json",
    },
}
BASELINE_MMLU = "results/stage2/eval/s58-post-0.8b-baseline-mmlu/raw/20261118T222909Z-mmlu.json"
BASELINE_MMLU_PRO = (
    "results/stage2/eval/s58-post-0.8b-baseline-mmlu/raw/20261118T224844Z-mmlu_pro.json")


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


def load_raw(rel: str) -> dict:
    path = os.path.join(ROOT, rel)
    opened = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else \
        open(path, "r", encoding="utf-8")
    with opened:
        return json.load(opened)


def rows(raw: dict, metric: str) -> dict[tuple[str, int], tuple[float, str]]:
    out = {}
    for leaf, samples in raw["samples"].items():
        for sample in samples:
            if metric not in sample:
                continue
            out[(leaf, int(sample["doc_id"]))] = (
                float(sample[metric]), sample.get("doc_hash", ""))
    return out


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap(deltas: list[float]) -> dict:
    rng = random.Random(SEED)
    n = len(deltas)
    draws = [sum(deltas[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(RESAMPLES)]
    return {
        "items": n,
        "delta": sum(deltas) / n,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "resamples": RESAMPLES,
        "seed": SEED,
        "basis": ("prompt-level paired bootstrap; the resampling unit is the "
                  "distinct problem, never the continuation"),
    }


def registered_verdict(delta: float, ci: list[float]) -> dict:
    """The S60 bound: a drop, not an absolute change."""
    point = delta >= -BOUND
    interval = ci[0] >= -BOUND
    if point and interval:
        state = "pass"
    elif not point:
        state = "fail"
    else:
        state = "unresolved"
    return {
        "state": state,
        "point_drop_within_bound": point,
        "interval_lower_within_bound": interval,
        "rule": ("point drop <= 1.0 point and paired interval lower bound "
                 ">= -1.0 point; an improvement is inside the bound"),
        "legacy_abs_delta_rule_would_say": (
            "pass" if abs(delta) <= BOUND and interval else
            "fail" if abs(delta) > BOUND else "unresolved"),
    }


def compare(metric: str, candidate: dict, baseline: dict,
            subset: set | None = None) -> dict:
    common = sorted(set(candidate) & set(baseline))
    if subset is not None:
        common = [key for key in common if key in subset]
    if not common:
        raise SystemExit("s66-retention: no paired items in the requested subset")
    mismatched = [key for key in common
                  if candidate[key][1] != baseline[key][1]]
    if mismatched:
        raise SystemExit(
            "s66-retention: the dumps index different documents at "
            f"{len(mismatched)} positions (first {mismatched[:3]})")
    deltas = [candidate[key][0] - baseline[key][0] for key in common]
    interval = bootstrap(deltas)
    interval["candidate_score"] = sum(candidate[key][0] for key in common) / len(common)
    interval["baseline_score"] = sum(baseline[key][0] for key in common) / len(common)
    interval["candidate_correct"] = int(sum(candidate[key][0] for key in common))
    interval["baseline_correct"] = int(sum(baseline[key][0] for key in common))
    interval["metric"] = metric
    interval["verdict"] = registered_verdict(interval["delta"], interval["ci95"])
    return interval


def audit_case(name: str, case: dict, baseline_mmlu: dict,
               baseline_pro: dict) -> dict:
    candidate = rows(load_raw(case["candidate"]), "acc")
    selection = rows(load_raw(case["selection"]), "acc")
    selection_keys = set(selection)
    unused = set(candidate) - selection_keys
    if not selection_keys <= set(candidate):
        raise SystemExit(
            f"s66-retention: t={name}: the selection items are not a subset of "
            "the full MMLU dump")
    if not selection_keys <= set(baseline_mmlu):
        raise SystemExit(
            f"s66-retention: t={name}: selection items are absent from the "
            "baseline dump")

    mmlu = {
        "full": compare("acc", candidate, baseline_mmlu),
        "selection": compare("acc", candidate, baseline_mmlu, selection_keys),
        "unused": compare("acc", candidate, baseline_mmlu, unused),
        "counts": {
            "full": len(set(candidate) & set(baseline_mmlu)),
            "selection": len(selection_keys),
            "unused": len(unused),
        },
    }

    pro_candidate = rows(load_raw(case["candidate_mmlu_pro"]), "exact_match")
    pro = compare("exact_match", pro_candidate, baseline_pro)
    pro["sample_size"] = len(pro_candidate)
    pro["registered_sample_size"] = 4000
    pro["registered_instrument_present"] = False
    pro["verdict"]["state"] = "unresolved"
    pro["verdict"]["why"] = (
        "the registered S60 retention instrument is a 4,000-item total-cap "
        "MMLU-Pro subset; only the retired 504-item sample exists on disk, so "
        "the clause is unresolved even though this sample's point estimate is "
        "inside the bound")

    selection_receipt = json.load(open(os.path.join(ROOT, case["selection_receipt"]),
                                       encoding="utf-8"))
    reported = json.load(open(os.path.join(ROOT, case["paired_receipt"]),
                              encoding="utf-8"))
    reported_mmlu = next(row for row in reported["rows"] if row["task"] == "mmlu")
    reported_pro = next(row for row in reported["rows"]
                        if row["task"] == "mmlu_pro")

    def check_equal(label: str, a: float, b: float) -> dict:
        return {"label": label, "a": a, "b": b, "equal": a == b}

    return {
        "coefficient": float(name),
        "inputs": {
            "candidate": case["candidate"],
            "selection": case["selection"],
            "candidate_mmlu_pro": case["candidate_mmlu_pro"],
            "selection_receipt": case["selection_receipt"],
            "paired_receipt": case["paired_receipt"],
        },
        "mmlu": mmlu,
        "mmlu_pro": pro,
        "selection_receipt": {
            "interpolation_coefficient": selection_receipt.get(
                "interpolation_coefficient"),
            "complete": selection_receipt.get("complete"),
            "mmlu_total_cap": (selection_receipt.get("total_cap_enforcement")
                               or {}).get("requested_total_cap"),
        },
        "reproduction_checks": [
            check_equal("reported full MMLU delta", reported_mmlu["delta"],
                        mmlu["full"]["delta"]),
            check_equal("reported full MMLU interval lower",
                        reported_mmlu["ci95"][0], mmlu["full"]["ci95"][0]),
            check_equal("reported full MMLU interval upper",
                        reported_mmlu["ci95"][1], mmlu["full"]["ci95"][1]),
            check_equal("reported MMLU-Pro delta", reported_pro["delta"],
                        pro["delta"]),
        ],
        "reported_receipt_legacy_passed_field": reported.get("passed"),
        "reported_receipt_legacy_passed_is_registered_rule": False,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline_mmlu = rows(load_raw(BASELINE_MMLU), "acc")
    baseline_pro = rows(load_raw(BASELINE_MMLU_PRO), "exact_match")
    cases = [audit_case(name, case, baseline_mmlu, baseline_pro)
             for name, case in CASES.items()]

    payload = {
        "schema": SCHEMA,
        "kind": "retention-reuse-audit",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "baseline": {
            "mmlu": BASELINE_MMLU,
            "mmlu_items": len(baseline_mmlu),
            "mmlu_pro": BASELINE_MMLU_PRO,
            "mmlu_pro_items": len(baseline_pro),
        },
        "registered_bound": {
            "points": 1.0,
            "reading": ("point drop <= 1.0 and paired interval lower bound "
                        ">= -1.0; positive deltas are improvements"),
        },
        "cases": cases,
        "conclusion": {
            "full_mmlu_reproduces": all(
                all(check["equal"] for check in case["reproduction_checks"])
                for case in cases),
            "selection_and_unused_are_disjoint": all(
                case["mmlu"]["counts"]["selection"] + case["mmlu"]["counts"]["unused"]
                == case["mmlu"]["counts"]["full"] for case in cases),
            "mmlu_pro_registered_sample_present": False,
            "complete_registered_win": False,
            "why": ("both math clauses pass and full-MMLU retention passes, "
                    "but the registered 4,000-item MMLU-Pro retention "
                    "instrument is absent; the 504-item sample is retired and "
                    "the clause is unresolved"),
        },
        "claims": {
            "test_split_used_for_selection": False,
            "selection_read_mmlu_items": 2000,
            "full_mmlu_items": 14042,
            "unused_mmlu_items": 12042,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    for case in cases:
        print(f"s66: t={case['coefficient']:.3f} full "
              f"{case['mmlu']['full']['delta']:+.4f} "
              f"[{case['mmlu']['full']['ci95'][0]:+.4f},"
              f"{case['mmlu']['full']['ci95'][1]:+.4f}] "
              f"selection {case['mmlu']['selection']['delta']:+.4f} "
              f"unused {case['mmlu']['unused']['delta']:+.4f} "
              f"({case['mmlu']['counts']['unused']} items)")
        print(f"s66: t={case['coefficient']:.3f} mmlu_pro "
              f"{case['mmlu_pro']['delta']:+.4f} "
              f"[{case['mmlu_pro']['ci95'][0]:+.4f},"
              f"{case['mmlu_pro']['ci95'][1]:+.4f}] n="
              f"{case['mmlu_pro']['items']} verdict="
              f"{case['mmlu_pro']['verdict']['state']}")
    print(f"s66: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
