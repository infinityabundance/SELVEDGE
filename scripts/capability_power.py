#!/usr/bin/env python3
"""Emit the power receipt for the registered capability composite (P14.4).

    python3 scripts/capability_power.py --out results/capability/power-<stamp>.json

The estimator is `oracle/capability/power.py`; this script only assembles the
payload, adds the registered suite binding and the revision, and records the
simulation parameters so the number can be reproduced.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO, "oracle")
sys.path.insert(0, ORACLE)

from capability import power, stats, suite  # noqa: E402

SCHEMA = "selvedge/capability-power/v1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    value = os.environ.get("SELVEDGE_REVISION", "")
    if value:
        return value
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures", "capability"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument("--regression-fraction", type=float, default=0.2)
    parser.add_argument("--simulations", type=int, default=120)
    parser.add_argument("--resamples", type=int, default=400)
    args = parser.parse_args(argv)

    manifest = suite.build_manifest(args.suite)
    counts = manifest["counts"]["by_category"]
    categories = [name for name in counts
                  if name not in ("trio", "probe")]
    n_items = manifest["counts"]["suite"]
    n_per_category = n_items // len(categories)
    if n_items % len(categories) != 0:
        raise SystemExit("the suite is not balanced across categories")

    optimistic = power.composite_mde(
        n_per_category=n_per_category, categories=len(categories),
        baseline=args.baseline, regression_fraction=0.0)
    pessimistic = power.composite_mde(
        n_per_category=n_per_category, categories=len(categories),
        baseline=args.baseline, regression_fraction=args.regression_fraction)
    simulation = power.simulate_power(
        optimistic["absolute"], n_per_category=n_per_category,
        categories=len(categories), regression_fraction=0.0,
        replications=args.simulations, resamples=args.resamples)
    empirical = power.empirical_mde(
        n_per_category=n_per_category, categories=len(categories),
        regression_fraction=0.0, replications=args.simulations,
        resamples=args.resamples)
    empirical_pessimistic = power.empirical_mde(
        n_per_category=n_per_category, categories=len(categories),
        regression_fraction=args.regression_fraction,
        replications=args.simulations, resamples=args.resamples)

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "revision": revision(),
        "command": " ".join(sys.argv),
        "suite": {
            "suite_sha256": manifest["suite_sha256"],
            "items": n_items,
            "categories": len(categories),
            "n_per_category": n_per_category,
            "category_names": sorted(categories),
        },
        "registered": {
            "relative_threshold": power.REGISTERED_RELATIVE_THRESHOLD,
            "bootstrap_seed": stats.SEED,
            "bootstrap_resamples": stats.RESAMPLES,
            "level": stats.LEVEL,
            "decision_rule": (
                "improve the composite by at least 10% relative with a paired 95% CI "
                "lower bound above zero, and regress no category by more than 0.05 "
                "absolute"),
        },
        "per_category_ci": {
            "proportion": args.baseline,
            "level": stats.LEVEL,
            "half_width": power.marginal_ci_half_width(
                n_per_category, proportion=args.baseline, level=stats.LEVEL),
            "categories": [
                {"name": name, "n": n_per_category,
                 "half_width": power.marginal_ci_half_width(
                     n_per_category, proportion=args.baseline, level=stats.LEVEL)}
                for name in sorted(categories)
            ],
            "note": (
                "single-proportion 95% CI half-width for one category's success "
                "rate at this n; the paired composite CI is narrower because the "
                "comparison is paired. Every category has the same n and therefore "
                "the same marginal half-width at the same proportion."),
        },
        "mde": {
            "optimistic_no_regressions": optimistic,
            "with_regressions": pessimistic,
            "simulation_at_analytic_mde": simulation,
            "empirical_optimistic": empirical,
            "empirical_with_regressions": empirical_pessimistic,
            "relative_threshold": power.REGISTERED_RELATIVE_THRESHOLD,
            "analytic_is_conservative": (
                simulation["decision_power"] >= 0.8),
        },
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(out + ".tmp", out)
    print(json.dumps({
        "out": os.path.relpath(out, REPO),
        "items": n_items,
        "n_per_category": n_per_category,
        "per_category_ci_half_width": round(payload["per_category_ci"]["half_width"], 4),
        "mde_absolute": round(optimistic["absolute"], 4),
        "mde_relative": round(optimistic["relative_to_baseline"], 4),
        "mde_relative_with_regressions": round(pessimistic["relative_to_baseline"], 4),
        "simulated_decision_power_at_analytic_mde": round(
            simulation["decision_power"], 3),
        "empirical_mde_absolute": (None if empirical["absolute"] is None
                                   else round(empirical["absolute"], 4)),
        "empirical_mde_relative": (None if empirical["relative_to_baseline"] is None
                                   else round(empirical["relative_to_baseline"], 4)),
        "empirical_mde_relative_with_regressions": (
            None if empirical_pessimistic["relative_to_baseline"] is None
            else round(empirical_pessimistic["relative_to_baseline"], 4)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
