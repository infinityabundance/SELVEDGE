#!/usr/bin/env python3
"""Amend the capability court's endpoint, with the power arithmetic in the receipt.

The registered endpoint has been a **10% relative** improvement of the
category-balanced composite since v1. The calibration pass then measured what
that endpoint costs on this instrument: a 10% relative effect at the measured
baseline needs a six-figure item count *per category*, and the suite has the
registered per-category minimum. A threshold that cannot be resolved at any
feasible n is not a strict endpoint, it is an unreachable one, and leaving it in
place means the court reports "no result" for every arm forever.

This script does not relax anything. It:

1. recomputes the required-n arithmetic from the calibration analysis receipt
   (quoting its variance estimates rather than re-deriving them);
2. inverts the same formula at the *registered* n to obtain the smallest
   absolute effect this instrument can detect, under both the planning and the
   conservative variance;
3. takes the **conservative** one, rounded up to the next 0.01, as the
   pre-registered minimum absolute effect — so the new endpoint is reachable at
   the registered n, and reachable under the pessimistic variance estimate;
4. registers a discrimination endpoint (candidate-ranking accuracy) that is
   explicitly **not available** until a receipt binds its pair fixture, and that
   is never pooled with the absolute endpoint or across categories;
5. writes `docs/capability-preregistration-v5.json` from those computed values,
   superseding v4 **without editing it** (v4 stays byte-identical on disk).

Every number in the amendment documents is emitted from the receipt this script
writes; nothing is typed into prose.

Usage:
    scripts/court python -- python3 scripts/capability_endpoint_amendment.py \
        --out results/capability/endpoint-amendment-$(date -u +%Y%m%dT%H%M%SZ).json \
        --write-registration docs/capability-preregistration-v5.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

SCHEMA = "selvedge/capability-endpoint-amendment/v1"
PREREG_SCHEMA = "selvedge/capability-preregistration/v1"


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def revision_of(root: str) -> str:
    result = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=repo_root())
    parser.add_argument("--preregistration", default="docs/capability-preregistration-v4.json")
    parser.add_argument("--analysis",
                        default="results/capability/calibration-analysis-20261116T165500Z.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--write-registration", default=None)
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args(argv)

    root = os.path.abspath(args.repo)
    prereg_path = os.path.join(root, args.preregistration)
    analysis_path = os.path.join(root, args.analysis)
    prereg = read_json(prereg_path)
    analysis = read_json(analysis_path)
    if prereg.get("schema") != PREREG_SCHEMA:
        print(f"capability_endpoint_amendment: {args.preregistration} has schema "
              f"{prereg.get('schema')!r}", file=sys.stderr)
        return 1
    primary = prereg.get("primary") or {}
    if primary.get("endpoint_kind", "relative") == "absolute":
        print("capability_endpoint_amendment: the registration is already on the "
              "absolute endpoint; this script amends a relative one", file=sys.stderr)
        return 1

    power = analysis.get("power") or {}
    baseline = power["measured_baseline_pass_rate"]
    var_planning = power["planning_variance"]
    var_conservative = power["conservative_variance"]
    required_n_planning = power["required_n_planning"]
    required_n_conservative = power["required_n_conservative"]
    available = power["items_per_category_available"]
    # The analysis receipt's own MDE used the available (degenerate) count; its
    # formula names z_total, so z_total is recovered from the receipt rather than
    # re-typed: mde = z * sqrt(var / (categories * n)). The category count is the
    # one the receipt's own MDE block used.
    categories = int(power["minimum_detectable_effect"]["planning"]["categories"])
    if not categories:
        print("capability_endpoint_amendment: the analysis receipt names no categories",
              file=sys.stderr)
        return 1
    n_available = sorted(available.values())[0]
    mde_planning_receipt = power["minimum_detectable_effect"]["planning"]["absolute_effect"]
    z_total = mde_planning_receipt / math.sqrt(var_planning / (categories * n_available))

    n_registered_items = int(primary["suite_items"])
    n_per_category = int(primary["category_minimum_items"])
    if n_per_category * categories != n_registered_items:
        print("capability_endpoint_amendment: the registered per-category minimum "
              "does not multiply out to the registered item count", file=sys.stderr)
        return 1

    mde_planning = z_total * math.sqrt(var_planning / (categories * n_per_category))
    mde_conservative = z_total * math.sqrt(
        var_conservative / (categories * n_per_category))
    minimum_absolute_effect = round(
        float(int(max(mde_planning, mde_conservative) * 100 + 1e-9) + 1) / 100.0, 2)
    ten_percent_relative_effect = 0.1 * baseline

    stamp = args.stamp or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    payload = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": revision_of(root),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": {
            "path": args.preregistration,
            "sha256": sha256_file(prereg_path),
            "reason": "the 10%-relative endpoint cannot be resolved at any feasible "
                      "item count on this instrument; the endpoint becomes an "
                      "absolute effect whose minimum is derived from the measured "
                      "variance at the registered n",
        },
        "source_receipt": {
            "path": args.analysis,
            "stamp": analysis.get("stamp"),
            "revision": analysis.get("revision"),
        },
        "registered_n": {
            "suite_items": n_registered_items,
            "categories": categories,
            "per_category": n_per_category,
            "relative_threshold": primary.get("relative_threshold"),
        },
        "measured_variance": {
            "baseline_pass_rate": baseline,
            "planning_variance": var_planning,
            "conservative_variance": var_conservative,
        },
        "old_endpoint_unreachable": {
            "relative_effect_at_baseline_absolute": ten_percent_relative_effect,
            "required_n_per_category_planning": required_n_planning,
            "required_n_per_category_conservative": required_n_conservative,
            "registered_n_per_category": n_per_category,
            "shortfall_multiple_planning": required_n_planning / n_per_category,
            "shortfall_multiple_conservative": required_n_conservative / n_per_category,
            "statement": "a 10% relative effect at the measured baseline needs the "
                         "required n above per category; the suite registers the "
                         "per-category minimum printed beside it",
        },
        "endpoint": {
            "kind": "absolute",
            "estimand": "category-balanced mean verified task success, paired",
            "z_total_recovered_from_receipt": z_total,
            "mde_absolute_planning": mde_planning,
            "mde_absolute_conservative": mde_conservative,
            "mde_absolute_planning_over_baseline": mde_planning / baseline,
            "mde_absolute_conservative_over_baseline": mde_conservative / baseline,
            "minimum_absolute_effect": minimum_absolute_effect,
            "minimum_absolute_effect_derivation":
                "ceil(max(planning MDE, conservative MDE) at the registered n to the "
                "next 0.01); the conservative variance is the one that must be "
                "reachable, so the minimum is set from it",
            "decision_rule": "WIN iff the paired 95% CI lower bound is at or above the "
                             "minimum absolute effect. Per-category regressions beyond "
                             "0.05 absolute are reported and block a transfer claim; "
                             "they are not a conjunct of the primary, because the "
                             "conjunct is measured to cost about a quarter of the "
                             "decision power at the same effect.",
        },
        "discrimination": {
            "endpoint": "candidate-ranking accuracy: the fraction of registered pairs "
                        "in which the model assigns a higher likelihood to the correct "
                        "candidate than to a mutated one",
            "status": "not_available",
            "why": "no pair fixture exists and no receipt binds one, so no "
                   "discrimination number may be reported",
            "binding_required": "a receipt naming the pair fixture's sha256, the "
                                "mutation rule and the scoring rule; until then the "
                                "endpoint is registered and unusable",
            "never_pooled_with": "the absolute primary endpoint, and never pooled "
                                 "across categories",
        },
    }

    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")

    if args.write_registration:
        registration = json.loads(json.dumps(prereg))
        registration["registered_utc"] = payload["created_utc"]
        registration["supersedes"] = payload["supersedes"]
        new_primary = dict(primary)
        new_primary.pop("relative_threshold", None)
        new_primary["endpoint_kind"] = "absolute"
        new_primary["minimum_absolute_effect"] = minimum_absolute_effect
        new_primary["minimum_absolute_effect_source"] = {
            "receipt": os.path.relpath(out_path, root),
            "pointer": "/endpoint/minimum_absolute_effect",
            "derivation": payload["endpoint"]["minimum_absolute_effect_derivation"],
        }
        new_primary["dominant_variance_at_registration"] = "conservative"
        new_primary["decision_rule"] = payload["endpoint"]["decision_rule"]
        new_primary["reporting"] = (
            "every category is reported separately; per-category regressions beyond "
            "the registered margin are disclosed and block a transfer claim; the "
            "absolute endpoint and the discrimination endpoint are never pooled, and "
            "neither is pooled across categories")
        registration["primary"] = new_primary
        registration["discrimination"] = payload["discrimination"]
        reg_path = args.write_registration
        reg_path = reg_path if os.path.isabs(reg_path) else os.path.join(root, reg_path)
        with open(reg_path, "w", encoding="utf-8") as handle:
            json.dump(registration, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(f"capability_endpoint_amendment: wrote {os.path.relpath(reg_path, root)}")
        payload["written_registration"] = {
            "path": os.path.relpath(reg_path, root),
            "sha256": sha256_file(reg_path),
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")

    print(
        "capability_endpoint_amendment: 10% relative at the measured baseline needs "
        f"{required_n_planning:,} (planning) / {required_n_conservative:,} "
        f"(conservative) items per category against {n_per_category} registered; "
        f"at n={n_registered_items} the absolute MDE is {mde_planning:.4f} "
        f"(planning) / {mde_conservative:.4f} (conservative), so the registered "
        f"minimum absolute effect is {minimum_absolute_effect:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
