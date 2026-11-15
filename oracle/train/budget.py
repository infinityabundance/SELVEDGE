#!/usr/bin/env python3
"""Budget reality check: what the measured smoke throughput buys.

Reads one or more smoke receipts, the registered token budget and a stated
wall-clock window, and writes a receipt-shaped table of tokens/GPU-hour, the
time the registered budget needs, and the largest honest budget inside the
window. No number in `results/T1-PLAN.md` is typed by hand; they all come from
this receipt.

    python3 oracle/train/budget.py --smoke results/train/smoke-*.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def largest_power_of_two(value: float) -> int:
    exponent = 0
    while 2 ** (exponent + 1) <= value:
        exponent += 1
    return 2 ** exponent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", nargs="+", required=True)
    parser.add_argument("--index", default="results/artifacts/train/train.index.json")
    parser.add_argument("--window-hours", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    index_path = os.path.join(REPO, args.index)
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    registered = int(index["registered_budget_tokens"])

    paths: list[str] = []
    for pattern in args.smoke:
        matches = sorted(glob.glob(os.path.join(REPO, pattern)))
        paths.extend(matches)
    if not paths:
        raise SystemExit("budget: no smoke receipts matched")

    arms: dict[str, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            receipt = json.load(handle)
        for arm, payload in receipt["arms"].items():
            train = payload["train"]
            seconds = args.window_hours * 3600.0
            tokens_per_second = float(train["tokens_per_second"])
            arms[arm] = {
                "smoke_receipt": os.path.relpath(path, REPO),
                "smoke_stamp": receipt["stamp"],
                "tokens_per_second": tokens_per_second,
                "tokens_per_gpu_hour": tokens_per_second * 3600.0,
                "vram_peak_gib": float(train["vram_peak_gib"]),
                "trainable_elements": int(train["trainable_elements"]),
                "admitted_active_unique": int(train["admitted_active_unique"]),
                "registered_budget_tokens": registered,
                "registered_budget_hours": registered / tokens_per_second / 3600.0,
                "max_tokens_in_window": int(tokens_per_second * seconds),
                "largest_power_of_two_in_window": largest_power_of_two(
                    tokens_per_second * seconds * 0.9),
                "fits_registered_budget": bool(
                    registered / tokens_per_second / 3600.0 <= args.window_hours),
                "resume_delta": payload["resume"]["max_abs_loss_delta"],
                "zero_point_ok": payload["zero_point"]["ok"],
                "eval_determinism_identical": (
                    payload.get("eval") or {}).get("determinism", {}).get("identical"),
                "eval_humaneval_pass_at_1": (
                    (payload.get("eval") or {}).get("humaneval") or {}).get("pass_at_1"),
                "eval_kl_mixed": (
                    ((payload.get("eval") or {}).get("distributional") or {})
                    .get("mixed", {}).get("kl", {}).get("mean")),
            }

    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/train-budget/v1",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "generated_utc": utc_now(),
        "window_hours": args.window_hours,
        "registered_budget_tokens": registered,
        "stream_tokens": index["tokens"],
        "arms": arms,
        "verdict": {
            arm: ("fits" if payload["fits_registered_budget"] else "does not fit")
            for arm, payload in arms.items()
        },
        "notes": [
            "throughput is measured on training steps only, excluding model load, "
            "the step-0 identity check and checkpoint writes",
            "tokens per GPU-hour = measured tokens/second x 3600",
            "largest_power_of_two_in_window leaves 10% of the window for overheads",
        ],
    }
    out = args.out or os.path.join(REPO, "results", "train", f"budget-{stamp}.json")
    write_json_atomic(out, receipt)
    print(json.dumps({"stamp": stamp, "arms": {
        arm: {"hours": round(payload["registered_budget_hours"], 2),
              "fits": payload["fits_registered_budget"],
              "tokens_per_gpu_hour": round(payload["tokens_per_gpu_hour"])}
        for arm, payload in arms.items()}}, indent=2, sort_keys=True))
    print(f"budget: receipt={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
