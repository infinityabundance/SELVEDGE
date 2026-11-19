#!/usr/bin/env python3
"""S59: apply the stage-2b registration's measured time ladder.

The registration fixes the ladder before the pilot exists:

    1. K=4 on 1500+1500 prompts;
    2. if the projected pass is above 4.5 h, MATH drops to K=3;
    3. if that is still above 4.5 h, each source drops to 1200 prompts.

This script replaces the projection with the pilot's own measured numbers —
mean generated tokens per rollout, per dataset, and decode tokens/s — and prints
the projected wall time of each step, plus the first step at or below the
bound. The decision is arithmetic over a receipt, not a judgement call, and the
output is written beside the receipt.

Usage
-----
    python3 scripts/s59_ladder.py \
        --pilot results/artifacts/stage2/s59-rft/pilot-receipt.json \
        --out results/stage2/s59-pilot-ladder.json
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
BOUND_SECONDS = 4.5 * 3600
#: The registration's step table: (label, gsm8k prompts, gsm8k K, math prompts,
#: math K).
STEPS = (
    ("step-1", 1500, 4, 1500, 4),
    ("step-2", 1500, 4, 1500, 3),
    ("step-3", 1200, 4, 1200, 3),
)


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(os.path.join(ROOT, args.pilot), "r", encoding="utf-8") as handle:
        pilot = json.load(handle)
    totals = pilot["acceptance"]["totals"]
    rate = pilot["generation"]["tokens_per_second"]
    by_dataset = pilot["acceptance"]["by_dataset"]
    # Mean generated tokens per rollout, including truncated rollouts: the
    # pilot counts every generated token, so this is the cost the pass pays.
    mean_tokens = {
        name: by_dataset[name]["generated_tokens"] / by_dataset[name]["sampled"]
        for name in by_dataset
    }
    # Accepted rollouts per prompt at the pilot's K, used to project corpus size.
    accepted_per_prompt = {
        name: by_dataset[name]["kept"] / by_dataset[name]["prompts"]
        for name in by_dataset
    }

    steps = []
    chosen = None
    for label, gsm_prompts, gsm_k, math_prompts, math_k in STEPS:
        gsm_tokens = gsm_prompts * gsm_k * mean_tokens["gsm8k"]
        math_tokens = math_prompts * math_k * mean_tokens["math"]
        total = gsm_tokens + math_tokens
        seconds = total / rate
        # K scaling on the accepted-rollout projection: the pilot measured K=4;
        # scaling it by K/4 is the linear expectation and is labelled as such.
        gsm_rows = gsm_prompts * accepted_per_prompt["gsm8k"] * gsm_k / 4.0
        math_rows = math_prompts * accepted_per_prompt["math"] * math_k / 4.0
        entry = {
            "step": label,
            "gsm8k": {"prompts": gsm_prompts, "k": gsm_k,
                      "decode_tokens": round(gsm_tokens),
                      "projected_accepted_rows": round(gsm_rows)},
            "math": {"prompts": math_prompts, "k": math_k,
                     "decode_tokens": round(math_tokens),
                     "projected_accepted_rows": round(math_rows)},
            "decode_tokens": round(total),
            "projected_seconds": round(seconds),
            "projected_hours": round(seconds / 3600.0, 3),
            "within_bound": seconds <= BOUND_SECONDS,
        }
        steps.append(entry)
        if chosen is None and entry["within_bound"]:
            chosen = entry

    payload = {
        "schema": "selvedge/s59-stage2b-ladder/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": revision(),
        "registration": "docs/ws19-post-trained-route-stage2b-preregistration-v1.json",
        "pilot": {"path": args.pilot, "stamp": pilot.get("stamp"),
                  "revision": pilot.get("revision")},
        "measured": {
            "decode_tokens_per_second": rate,
            "mean_generated_tokens_per_rollout": mean_tokens,
            "accepted_rollouts_per_prompt": accepted_per_prompt,
            "pilot_prompts": totals["prompts"],
            "pilot_sampled": totals["sampled"],
            "pilot_truncation_rate": pilot["acceptance"]["truncation_rate"],
            "pilot_loop_cut_rate": pilot["acceptance"]["loop_cut_rate"],
        },
        "bound_seconds": BOUND_SECONDS,
        "steps": steps,
        "chosen": chosen["step"] if chosen else None,
        "rule": "the first step whose projected wall time is at or below the "
                "bound; if no step qualifies, the smallest registered step is "
                "run and the overrun is reported rather than hidden",
        "k_scaling_note": "accepted rows are projected by scaling the pilot's "
                          "K=4 rate linearly; that is an expectation, and the "
                          "realised count comes from the pass's own receipt",
    }
    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for entry in steps:
        print(f"s59-ladder: {entry['step']}: {entry['decode_tokens']} decode "
              f"tokens -> {entry['projected_hours']:.2f} h "
              f"({'within' if entry['within_bound'] else 'above'} the bound), "
              f"~{entry['gsm8k']['projected_accepted_rows']} gsm8k + "
              f"{entry['math']['projected_accepted_rows']} math rows")
    print(f"s59-ladder: chosen {payload['chosen']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
