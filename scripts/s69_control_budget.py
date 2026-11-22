#!/usr/bin/env python3
"""S69: the matched-SFT control budget that reproduces the preference steps.

The preference launcher budgets in forward-token positions
(``2 * prompt + chosen + rejected`` per pair) while the matched SFT control
budgets in supervised positions (``prompt + chosen``). Passing the same token
number to both would give the control a different number of optimisation steps
on the same pair order.

This script replays the launcher's own accounting on the pair file: the same
``weight > 0`` filter, the same ``random.Random(seed).shuffle`` order, the same
re-shuffle when the order is exhausted, and the same per-arm token counts. It
reports the preference step count to the budget and, for a chosen step, the
control's cumulative supervised budget. The control is then launched with
``--budget-tokens <that value> --max-steps <step>`` so both arms take the
identical step sequence over the identical problem set.

It reads no score and writes no model; it is a CPU receipt.

Usage:
    SELVEDGE_MEM=4g scripts/court python -- python3 scripts/s69_control_budget.py \
      --pairs results/artifacts/stage2/s69-preference/soup-with-reference.jsonl \
      --budget-tokens 2000000 --seed 6900 \
      --cadence-report-out results/stage2/s69-calibration/control-budget.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import preference  # noqa: E402


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


def load_pairs(path: str) -> list[preference.PreferencePair]:
    pairs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append(preference.PreferencePair(
                problem_id=str(row["problem_id"]),
                prompt_ids=tuple(int(x) for x in row["prompt_ids"]),
                chosen_ids=tuple(int(x) for x in row["chosen_ids"]),
                rejected_ids=tuple(int(x) for x in row["rejected_ids"]),
                weight=float(row.get("weight", 1.0))))
    # The launcher drops weight <= 0 for the preference and matched arms.
    return [pair for pair in pairs if pair.weight > 0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--budget-tokens", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=6900)
    parser.add_argument("--steps", default=None,
                        help="comma-separated preference step numbers whose "
                             "control budget is wanted; default: every cadence")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    pairs = load_pairs(os.path.join(ROOT, args.pairs))
    if not pairs:
        raise SystemExit("s69-control: no positive-weight pairs")

    rng = random.Random(args.seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cursor = 0
    processed = 0
    supervised = 0
    step = 0
    schedule: list[dict] = []
    while processed < args.budget_tokens:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        pair = pairs[order[cursor]]
        cursor += 1
        step += 1
        processed += preference.preference_processed_tokens([pair])
        supervised += preference.sft_processed_tokens([pair])
        schedule.append({
            "step": step,
            "pair_index": order[cursor - 1],
            "processed_tokens": processed,
            "supervised_tokens": supervised,
        })

    wanted = [int(part) for part in (args.steps or "").split(",") if part.strip()]
    selected = [row for row in schedule if row["step"] in wanted]
    missing = sorted(set(wanted) - {row["step"] for row in selected})
    if missing:
        raise SystemExit(f"s69-control: step(s) {missing} are beyond the "
                         f"preference run ({step} steps)")

    payload = {
        "schema": "selvedge/s69-control-budget/v1",
        "kind": "matched-sft-control-budget",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "pairs": {
            "path": args.pairs,
            "sha256": sha256_file(os.path.join(ROOT, args.pairs)),
            "positive_weight_rows": len(pairs),
        },
        "seed": args.seed,
        "budget_tokens": args.budget_tokens,
        "preference": {
            "steps": step,
            "processed_tokens": processed,
            "basis": "2*prompt + chosen + rejected per pair (the launcher)",
        },
        "matched_sft": {
            "budget_basis": "cumulative prompt + chosen tokens at the requested "
                            "preference step; passes --budget-tokens and "
                            "--max-steps so the control takes the identical "
                            "step sequence over the identical pair order",
            "full_processed_tokens": supervised,
            "selected": selected,
        },
        "schedule": schedule if args.steps is None else None,
        "claims": {
            "same_pair_filter_as_launcher": True,
            "same_order_rule_as_launcher": True,
            "no_score_read": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    print(f"s69-control: preference {step} steps, "
          f"{processed} processed tokens; selected {selected} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
