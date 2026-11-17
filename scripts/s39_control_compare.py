#!/usr/bin/env python3
"""Compare the distilled control with the anchored scoreboard (WS19 Job 1).

Inputs: the control's evaluation receipt and the anchor-gap receipt produced by
`scripts/s36_anchor.py gap` under the same instrument. Output: a receipt with
per-task deltas, per-family means, the registered half-gap thresholds and an
explicit verdict, so no comparison number in a document is typed by hand.

The per-task delta standard error is the unpaired combination
`sqrt(se_control^2 + se_anchor^2)`. The two runs score the same items, so the
unpaired form is conservative for a correlated pair and is labelled as such; it
is a bound on the sampling error, not a paired test.

Usage:
    scripts/court python scripts/s39_control_compare.py \
        --control results/stage2/eval/control-20m/eval.json \
        --gap results/frontier/anchor-gap-20261116T225500Z.json \
        --out results/stage2/control-comparison-<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def sha256(rel: str) -> str:
    import hashlib

    handle = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as binary:
        for chunk in iter(lambda: binary.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def load(rel: str) -> dict:
    with open(os.path.join(REPO, rel), encoding="utf-8") as handle:
        return json.load(handle)


def combined_se(one: float | None, two: float | None) -> float | None:
    if one is None or two is None:
        return None
    return math.sqrt(one ** 2 + two ** 2)


def family_members(gap: dict) -> dict[str, list[str]]:
    return {row["family"]: row["members"] for row in gap["families"]}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--gap", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    control = load(args.control)
    gap = load(args.gap)
    if not control.get("complete"):
        raise SystemExit(f"{args.control}: the control run is not complete")
    if control["harness"]["commit"] != gap["harness_commit"]:
        raise SystemExit("the control and the anchor gap were measured under different harness commits")

    control_by_task = {row["task"]: row for row in control["tasks"]}
    rows = []
    for reference in gap["tasks"]:
        name = reference["task"]
        scored = control_by_task.get(name)
        if scored is None:
            continue
        se = combined_se(scored["stderr"], reference["anchor_stderr"])
        rows.append(
            {
                "task": name,
                "family": reference["family"],
                "metric": reference["metric"],
                "control_score": scored["score"],
                "control_stderr": scored["stderr"],
                "anchor_score": reference["anchor_score"],
                "anchor_stderr": reference["anchor_stderr"],
                "upper_score": reference["upper_score"],
                "upper_stderr": reference["upper_stderr"],
                "delta_vs_anchor_pp": (scored["score"] - reference["anchor_score"]) * 100.0,
                "delta_vs_anchor_combined_stderr_pp": (se * 100.0 if se is not None else None),
                "delta_vs_anchor_note": (
                    "unpaired combination sqrt(se_control^2 + se_anchor^2); the runs "
                    "share items, so this overstates the paired sampling error"
                ),
                "delta_vs_upper_pp": (scored["score"] - reference["upper_score"]) * 100.0,
                "half_gap_pp": reference["half_gap_pp"],
                "control_reaches_half_gap": (
                    scored["score"] >= reference["anchor_plus_half_gap_pp"] / 100.0),
            }
        )

    families = []
    members = family_members(gap)
    for family, names in members.items():
        present = [row for row in rows if row["task"] in names]
        if sorted(row["task"] for row in present) != sorted(names):
            continue
        control_mean = sum(row["control_score"] for row in present) / len(present)
        anchor_mean = sum(row["anchor_score"] for row in present) / len(present)
        upper_mean = sum(row["upper_score"] for row in present) / len(present)
        family_se = math.sqrt(
            sum((row["control_stderr"] or 0.0) ** 2 for row in present)) / len(present)
        threshold = anchor_mean + (upper_mean - anchor_mean) / 2.0
        families.append(
            {
                "family": family,
                "members": names,
                "control_mean": control_mean,
                "control_combined_stderr_independent": family_se,
                "anchor_mean": anchor_mean,
                "upper_mean": upper_mean,
                "delta_vs_anchor_pp": (control_mean - anchor_mean) * 100.0,
                "delta_vs_upper_pp": (control_mean - upper_mean) * 100.0,
                "gap_pp": (upper_mean - anchor_mean) * 100.0,
                "half_gap_threshold": threshold,
                "control_reaches_half_gap": control_mean >= threshold,
                "fraction_of_gap_closed": (
                    (control_mean - anchor_mean) / (upper_mean - anchor_mean)
                    if upper_mean != anchor_mean else None),
            }
        )

    verdict = {
        "control_reaches_half_gap_on_any_family": any(
            row["control_reaches_half_gap"] for row in families),
        "families_reaching_half_gap": [
            row["family"] for row in families if row["control_reaches_half_gap"]],
        "reading": (
            "the registered win condition is 'close at least half the gap on one "
            "family'; this receipt is the distilled control's reading of it. The "
            "mechanism stages must beat this control, not the frozen anchor."
        ),
    }

    payload = {
        "schema": "selvedge/stage2-control-comparison/v1",
        "kind": "control-comparison",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "control": {
            "receipt": args.control,
            "sha256": sha256(args.control),
            "stamp": control["stamp"],
            "checkpoint": control["arm"].get("checkpoint"),
            "checkpoint_sha256": control["arm"].get("checkpoint_sha256"),
            "checkpoint_step": control["arm"].get("checkpoint_step"),
            "checkpoint_tokens_seen": control["arm"].get("checkpoint_tokens_seen"),
        },
        "reference": {"gap_receipt": args.gap, "sha256": sha256(args.gap)},
        "harness_commit": gap["harness_commit"],
        "tasks": rows,
        "families": families,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, args.out)
    for row in rows:
        print(f"  {row['task']:<18} control={row['control_score'] * 100:6.2f} "
              f"anchor={row['anchor_score'] * 100:6.2f} "
              f"2b={row['upper_score'] * 100:6.2f} "
              f"delta_anchor={row['delta_vs_anchor_pp']:+.2f}pp")
    for row in families:
        print(f"  family {row['family']:<10} control={row['control_mean'] * 100:6.2f} "
              f"anchor={row['anchor_mean'] * 100:6.2f} "
              f"2b={row['upper_mean'] * 100:6.2f} "
              f"closed={row['fraction_of_gap_closed'] * 100:6.1f}% "
              f"reaches_half_gap={row['control_reaches_half_gap']}")
    print(f"control-compare: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
