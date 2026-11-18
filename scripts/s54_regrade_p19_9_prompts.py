#!/usr/bin/env python3
"""S54: expand the P19.9 screen holdout from 8 to 24 prompts.

The S52 prompt set graded the first 64 GSM8K-train prompts 48/8/8. The screen is
a mechanism validation, and eight holdout problems cannot support a
prompt-level paired interval. This re-grades the *same committed prompts* by
file order -- 32 fit / 8 calib / 24 holdout -- without changing a question, an
answer, an id or the sampling plan. No prompt text moves between splits, so the
loader's leakage refusal is unchanged.

Usage:
    python3 scripts/s54_regrade_p19_9_prompts.py \
        --in docs/s52-p19.9-calibration-prompts-v1.json \
        --out docs/s52-p19.9-calibration-prompts-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source",
                        default="docs/s52-p19.9-calibration-prompts-v1.json")
    parser.add_argument("--out", dest="out",
                        default="docs/s52-p19.9-calibration-prompts-v2.json")
    parser.add_argument("--fit", type=int, default=32)
    parser.add_argument("--calib", type=int, default=8)
    args = parser.parse_args()

    payload = json.load(open(args.source, encoding="utf-8"))
    prompts = payload["prompts"]
    if args.fit + args.calib >= len(prompts):
        raise SystemExit("the holdout would be empty")
    for index, row in enumerate(prompts):
        if index < args.fit:
            row["split"] = "fit"
        elif index < args.fit + args.calib:
            row["split"] = "calib"
        else:
            row["split"] = "holdout"
    counts = {name: sum(1 for row in prompts if row["split"] == name)
              for name in ("fit", "calib", "holdout")}
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise SystemExit("duplicate prompt id")
    if len({row["prompt"].strip().lower() for row in prompts}) != len(prompts):
        raise SystemExit("duplicate normalised prompt text")
    payload["schema"] = "selvedge/p19.9-prompt-set/v2"
    payload["created_utc"] = "2026-11-18T02:00:00Z"
    payload["purpose"] = (
        "The S52 64-prompt calibration set re-graded 32 fit / 8 calib / 24 "
        "holdout before any screen run. It expands the holdout so the "
        "prompt-level paired interval has 24 distinct problems. This is a "
        "mechanism-validation set, not a capability benchmark.")
    payload["split_rule"] = (
        "file order over the same 64 committed prompts: first 32 fit, next 8 "
        "calib, last 24 holdout; question, answer, id and sampling unchanged")
    payload["supersedes"] = {
        "path": args.source,
        "sha256": hashlib.sha256(open(args.source, "rb").read()).hexdigest(),
        "change": "split counts only: 48/8/8 -> 32/8/24",
    }
    payload["counts"] = counts
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"{args.out}: {counts} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
