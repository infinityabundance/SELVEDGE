#!/usr/bin/env python3
"""S69/S70: the control budget that reproduces the preference arm's work.

The preference launcher budgets in forward-token positions
(``2 * prompt + chosen + rejected`` per pair) while the matched SFT control
budgets in supervised positions (``prompt + chosen``). Passing the same token
number to both would give the control a different number of optimisation steps
on the same pair order.

This script replays the launcher's own accounting on the pair file: the same
``weight > 0`` filter, the same pair order (``shuffled`` or the S70
difficulty-proportional ``weighted`` permutation, both from
``frontier.preference.pair_stream`` with the same seed), the same re-shuffle or
re-draw when the order is exhausted, and the same per-arm token counts.

Two bases are emitted together so the difference between them is explicit:

``--basis steps``
    Today's behaviour: the SFT arm takes the identical step count over the
    identical pair order. Its ``--budget-tokens`` is the control's cumulative
    supervised positions at the selected preference step. Compute is **not**
    matched: the preference arm forwards the chosen and rejected continuations,
    the SFT arm forwards the chosen continuation only.

``--basis compute``
    The SFT step count is chosen so that the SFT arm's total processed forward
    positions reach the preference arm's total. Preference per step is
    ``2 * prompt + chosen + rejected``; SFT per step is ``prompt + chosen``.
    The counts are derived from the pair file itself, never estimated.

The receipt always carries both bases, whichever was requested, so a reader can
see the compute gap without re-running the script. It reads no score and writes
no model; it is a CPU receipt.

Usage:
    SELVEDGE_MEM=4g scripts/court oracle -- python3 scripts/s69_control_budget.py \
      --pairs results/artifacts/stage2/s69-preference/soup-with-reference.jsonl \
      --budget-tokens 2000000 --seed 6900 --basis compute \
      --out results/stage2/s69-calibration/control-budget.json

The oracle court is required: the script imports ``frontier.preference``, which
imports torch. The python court has no torch and refuses at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def iter_rows(pairs, *, seed: int, pair_order: str, batch_pairs: int):
    """Yield one row per optimizer step forever, with running totals.

    The ordering comes from `frontier.preference.pair_stream`, which is what the
    launcher itself consumes; the epoch/batch/budget logic is written out here a
    second time on purpose, so the agreement test compares two implementations
    rather than one function with itself. Restarting with the same seed replays
    the identical sequence, which is what the compute basis relies on.
    """
    stream = preference.pair_stream(pairs, seed=seed, order=pair_order)
    processed = 0
    supervised = 0
    step = 0
    while True:
        indices = [next(stream) for _ in range(batch_pairs)]
        batch = [pairs[index] for index in indices]
        step += 1
        step_processed = preference.preference_processed_tokens(batch)
        step_supervised = preference.sft_processed_tokens(batch)
        processed += step_processed
        supervised += step_supervised
        yield {
            "step": step,
            "pair_indices": indices,
            "processed_tokens": processed,
            "supervised_tokens": supervised,
            "step_processed_tokens": step_processed,
            "step_forward_positions_sft": step_supervised,
        }


def preference_schedule(pairs, *, seed: int, pair_order: str,
                        batch_pairs: int, budget_tokens: int,
                        max_steps: int | None = None) -> list[dict]:
    """The launcher's step loop: stop at the processed-token budget."""
    rows: list[dict] = []
    for row in iter_rows(pairs, seed=seed, pair_order=pair_order,
                         batch_pairs=batch_pairs):
        rows.append(row)
        if row["processed_tokens"] >= budget_tokens:
            break
        if max_steps is not None and row["step"] >= max_steps:
            break
    return rows


def steps_at(rows: list[dict], step: int) -> dict:
    for row in rows:
        if row["step"] == step:
            return row
    raise SystemExit(f"s69-control: step {step} is beyond the preference run "
                     f"({len(rows)} steps)")


def compute_basis(pairs, *, target: int, seed: int, pair_order: str,
                  batch_pairs: int) -> dict:
    """The SFT step whose cumulative forward positions cover ``target``.

    The SFT arm is launched with the same seed and order, so its first ``k``
    steps are the first ``k`` rows of the same deterministic sequence. The count
    is derived from the pair file's own token lengths.
    """
    for row in iter_rows(pairs, seed=seed, pair_order=pair_order,
                         batch_pairs=batch_pairs):
        if row["supervised_tokens"] >= target:
            return {
                "sft_steps": row["step"],
                "sft_forward_positions": row["supervised_tokens"],
                "preference_forward_positions": target,
                "overshoot_positions": row["supervised_tokens"] - target,
            }
    raise SystemExit("s69-control: unreachable")  # pragma: no cover


def arm_table(label: str, rows: list[dict], steps: int) -> dict:
    row = steps_at(rows, steps)
    return {
        "basis": label,
        "steps": steps,
        "pairs": sum(len(item["pair_indices"]) for item in rows[:steps]),
        "forward_positions": (row["processed_tokens"] if label == "preference"
                              else row["supervised_tokens"]),
        "preference_forward_positions": row["processed_tokens"],
        "sft_forward_positions": row["supervised_tokens"],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--budget-tokens", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=6900)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--pair-order", choices=preference.PAIR_ORDERS,
                        default="shuffled")
    parser.add_argument("--basis", choices=("steps", "compute"),
                        default="steps")
    parser.add_argument("--steps", default=None,
                        help="comma-separated preference step numbers whose "
                             "control budget is wanted; default: the last step")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    pairs = load_pairs(os.path.join(ROOT, args.pairs))
    if not pairs:
        raise SystemExit("s69-control: no positive-weight pairs")
    if args.batch_pairs < 1:
        raise SystemExit("s69-control: --batch-pairs must be >= 1")

    rows = preference_schedule(
        pairs, seed=args.seed, pair_order=args.pair_order,
        batch_pairs=args.batch_pairs, budget_tokens=args.budget_tokens)
    if not rows:
        raise SystemExit("s69-control: the budget produced no steps")

    wanted = [int(part) for part in (args.steps or "").split(",") if part.strip()]
    if not wanted:
        wanted = [rows[-1]["step"]]
    missing = sorted(set(wanted) - {row["step"] for row in rows})
    if missing:
        raise SystemExit(f"s69-control: step(s) {missing} are beyond the "
                         f"preference run ({len(rows)} steps)")

    selected = []
    bases = []
    for step in wanted:
        row = steps_at(rows, step)
        compute = compute_basis(
            pairs, target=row["processed_tokens"], seed=args.seed,
            pair_order=args.pair_order, batch_pairs=args.batch_pairs)
        selected.append({
            "step": step,
            "processed_tokens": row["processed_tokens"],
            "supervised_tokens": row["supervised_tokens"],
            "supervised_budget_for_matched_steps": row["supervised_tokens"],
            "sft_steps_for_matched_compute": compute["sft_steps"],
            "sft_forward_positions_for_matched_compute": compute[
                "sft_forward_positions"],
            "compute_ratio": compute["sft_steps"] / step,
        })
        bases.append({
            "preference_step": step,
            "steps": {
                "preference": arm_table("preference", rows, step),
                "matched_sft": {
                    "basis": "matched_sft", "steps": step,
                    "pairs": sum(len(item["pair_indices"])
                                 for item in rows[:step]),
                    "forward_positions": row["supervised_tokens"],
                    "budget_tokens_for_control": row["supervised_tokens"],
                    "max_steps_for_control": step,
                },
            },
            "compute": {
                "preference": arm_table("preference", rows, step),
                "matched_sft": {
                    "basis": "matched_sft",
                    "steps": compute["sft_steps"],
                    "pairs": compute["sft_steps"] * args.batch_pairs,
                    "forward_positions": compute["sft_forward_positions"],
                    "budget_tokens_for_control": compute[
                        "sft_forward_positions"],
                    "max_steps_for_control": compute["sft_steps"],
                    "overshoot_positions": compute["overshoot_positions"],
                },
            },
        })

    payload = {
        "schema": "selvedge/s69-control-budget/v2",
        "kind": "matched-sft-control-budget",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "pairs": {
            "path": args.pairs,
            "sha256": sha256_file(os.path.join(ROOT, args.pairs)),
            "positive_weight_rows": len(pairs),
            "mean_weight": preference.corpus_mean_weight(pairs),
        },
        "seed": args.seed,
        "budget_tokens": args.budget_tokens,
        "batch_pairs": args.batch_pairs,
        "pair_order": args.pair_order,
        "basis_requested": args.basis,
        "preference": {
            "steps": rows[-1]["step"],
            "processed_tokens": rows[-1]["processed_tokens"],
            "basis": "2*prompt + chosen + rejected per pair (the launcher)",
        },
        "matched_sft": {
            "budget_basis": (
                "cumulative prompt + chosen tokens at the requested "
                "preference step; passes --budget-tokens and --max-steps so "
                "the control takes the identical step sequence over the "
                "identical pair order"
                if args.basis == "steps" else
                "the chosen SFT step count whose total processed forward "
                "positions cover the preference arm's; each SFT step forwards "
                "prompt + chosen once, while a preference step forwards the "
                "prompt twice plus both continuations"),
            "full_processed_tokens": rows[-1]["supervised_tokens"],
            "selected": selected,
            "selected_basis": args.basis,
            "launch_for_selected_basis": (
                bases[0][args.basis]["matched_sft"]
                if args.basis in ("steps", "compute") else None),
        },
        "bases": bases,
        "compute_basis_rule": (
            "sft_steps = min{k : sum(prompt+chosen over steps 1..k) >= "
            "sum(2*prompt+chosen+rejected over steps 1..N)}; the counts come "
            "from the pair file, not from a per-token estimate"),
        "schedule": None if args.steps else rows,
        "claims": {
            "same_pair_filter_as_launcher": True,
            "same_order_rule_as_launcher": True,
            "same_order_mode_asked_for": args.pair_order,
            "no_score_read": True,
            "compute_is_matched_only_under_basis_compute": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)

    print(f"{'basis':8s} {'arm':12s} {'steps':>7s} {'pairs':>7s} "
          f"{'forward_positions':>18s}")
    for step, block in zip(wanted, bases):
        for label in ("steps", "compute"):
            pref = block[label]["preference"]
            sft = block[label]["matched_sft"]
            print(f"{label:8s} {'preference':12s} {pref['steps']:7d} "
                  f"{pref['pairs']:7d} {pref['forward_positions']:18d}")
            print(f"{label:8s} {'matched-sft':12s} {sft['steps']:7d} "
                  f"{sft['pairs']:7d} {sft['forward_positions']:18d}")
        print(f"         preference step {step}: compute-basis SFT has "
              f"{block['compute']['matched_sft']['steps']} steps vs "
              f"{step} under the step basis")
    print(f"s69-control: basis={args.basis} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
