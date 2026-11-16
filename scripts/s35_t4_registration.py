#!/usr/bin/env python3
"""Register the K18 trade question (T4) and receipt the numbers it quotes.

T4 is a *new* question about the trade `X` measured, not a re-threshold of K18:
K18's verdict stands, no T2 threshold moves, and this script never re-decides it.
What T4 needs before any arm is trained is a fixed decision rule, and a decision
rule that quotes numbers must quote them from a receipt rather than from prose.

This script reads `results/t2-verdict.json` and writes
`results/t4/preregistration-<stamp>.json` containing:

* the `X` vs `D` primary-composite and pooled-KL differences with their paired
  intervals, exactly as the verdict receipt records them;
* the paired interval's half-width, and the **pre-registered minimum absolute
  effect** for T4 derived from it (ceil to the next 0.01): an effect smaller
  than the instrument's own resolution is not a win condition, it is noise with
  a sign;
* the two arms' active-parameter counts and their relative difference, so
  "matched active parameters" is a printed fact rather than an assertion;
* the KL value `D_kl` must reach, and the compute cap T4 registers for it.

It computes nothing that depends on a run that has not happened. Every field is
either a quotation from the verdict receipt or arithmetic on those quotations.

Usage:
    scripts/court python -- python3 scripts/s35_t4_registration.py \
        --out results/t4/preregistration-$(date -u +%Y%m%dT%H%M%SZ).json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

VERDICT_REL = "results/t2-verdict.json"
PILOT_REL = "results/t2/pilot-state.json"
SCHEMA = "selvedge/t4-preregistration-numbers/v1"

#: Pre-registered by T4 (a choice, not a measurement): the most training compute
#: `D_kl` may spend to reach `X`'s pooled teacher KL. If matching KL costs more
#: than this multiple of the T2 budget, the trade is recorded as a capacity
#: effect rather than a training-cost effect and T4's route changes.
COMPUTE_CAP_MULTIPLE = 4.0


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def revision_of(root: str) -> str:
    result = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=repo_root())
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args(argv)

    root = os.path.abspath(args.repo)
    verdict_path = os.path.join(root, VERDICT_REL)
    if not os.path.exists(verdict_path):
        print(f"s35_t4_registration: {VERDICT_REL} is missing", file=sys.stderr)
        return 1
    with open(verdict_path, "r", encoding="utf-8") as handle:
        verdict = json.load(handle)
    if verdict.get("schema") != "selvedge/t2-verdict/v1":
        print(
            f"s35_t4_registration: {VERDICT_REL} has schema "
            f"{verdict.get('schema')!r}, not the verdict schema",
            file=sys.stderr,
        )
        return 1

    comparison = (verdict.get("comparisons") or {}).get("X-D")
    if comparison is None:
        print("s35_t4_registration: the verdict receipt has no X-D comparison",
              file=sys.stderr)
        return 1
    pc = comparison["pc"]["difference"]
    kl = comparison["kl"]["difference"]
    pc_low, pc_high = pc["ci_low"], pc["ci_high"]
    half_width = (pc_high - pc_low) / 2.0
    # ceil to the next 0.01 with a float-safe epsilon
    minimum_absolute_effect = round(
        float(int(half_width * 100 + 1e-9) + 1) / 100.0, 2
    )

    arms = verdict["arms"]
    x_active = verdict["model_resources"]["X"]["active_unique_params"]
    d_active = verdict["model_resources"]["D"]["active_unique_params"]
    x_kl = arms["X"]["kl_mixed"]["mean"]
    d_kl = arms["D"]["kl_mixed"]["mean"]
    with open(os.path.join(root, PILOT_REL), "r", encoding="utf-8") as handle:
        pilot = json.load(handle)
    budget = pilot.get("budget_tokens")
    if not isinstance(budget, int) or budget <= 0:
        print(f"s35_t4_registration: {PILOT_REL} has no positive budget_tokens",
              file=sys.stderr)
        return 1
    compute_cap_tokens = int(COMPUTE_CAP_MULTIPLE * budget)

    stamp = args.stamp or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    payload = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": revision_of(root),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "question": "is X's task gain at a measured pooled-KL regression "
                    "Pareto-useful under matched inference cost?",
        "source_receipt": {
            "path": VERDICT_REL,
            "stamp": verdict.get("stamp"),
            "revision": verdict.get("revision"),
            "comparison": "X-D",
        },
        "k18_quotation": {
            "note": "K18 was decided FAIL against the T2 registrations. These are "
                    "its numbers, quoted; T4 does not revisit the verdict.",
            "pc_difference": pc["mean_difference"],
            "pc_ci_low": pc_low,
            "pc_ci_high": pc_high,
            "pc_ci_half_width": half_width,
            "kl_difference": kl["mean_difference"],
            "kl_ci_low": kl["ci_low"],
            "kl_ci_high": kl["ci_high"],
            "kl_relative_improvement": comparison["kl"]["relative_improvement"],
            "k18_verdict": comparison["verdict"],
        },
        "t4_registration": {
            "primary": "PC (the T2.9 composite), paired, category-weighted",
            "primary_difference_against": "D_kl",
            "secondary": "pooled teacher KL (mixed interleaving), lower is better",
            "minimum_absolute_effect": minimum_absolute_effect,
            "minimum_absolute_effect_derivation": "ceil(paired CI half-width of the "
                                                 "K18 PC comparison to the next 0.01): "
                                                 "an effect below the instrument's own "
                                                 "resolution is not a win condition",
            "decision_rule": "WIN iff the paired PC difference (X - D_kl) has a 95% CI "
                             "lower bound at or above the minimum absolute effect AND "
                             "the pooled-KL difference's 95% CI is compatible with zero "
                             "at the registered tolerance. Otherwise MISS, with the "
                             "recorded next route.",
            "kl_target": x_kl,
            "kl_target_source": "X's pooled teacher KL, from the same receipt",
            "dense_kl_reference": d_kl,
            "compute_cap_target_tokens": compute_cap_tokens,
            "compute_cap_multiple": COMPUTE_CAP_MULTIPLE,
            "t2_budget_tokens": budget,
            "active_parameters": {
                "X": x_active,
                "D": d_active,
                "absolute_difference": x_active - d_active,
                "relative_difference": (x_active - d_active) / d_active,
            },
            "resamples": verdict.get("resamples"),
            "seed": verdict.get("registered_seed"),
        },
    }

    out_path = os.path.join(root, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"s35_t4_registration: wrote {os.path.relpath(out_path, root)}")
    print(
        "s35_t4_registration: PC {:+0.4f} [{:+0.4f}, {:+0.4f}], KL {:+0.4f}, "
        "minimum absolute effect {:.2f}".format(
            pc["mean_difference"], pc_low, pc_high, kl["mean_difference"],
            minimum_absolute_effect,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
