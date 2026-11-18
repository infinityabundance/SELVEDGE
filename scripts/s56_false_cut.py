#!/usr/bin/env python3
"""S56: measure early-cut savings and the false-cut rate on stored generations.

The registration (`docs/ws19-post-trained-route-stage2a-amendment-s56-v1.json`)
says a truncated rollout is wrong and a loop tail may be cut early. "May" is
decided here: the calibrated detector is replayed over every stored stage-1
completion, the prefix it would have cut is verified with the same verifier, and
a cut is counted **false** when the uncut answer was correct and the cut prefix
is not.

No generation is run. For a deterministic rollout the tokens before the cut are
identical to the uncut run, so replaying the stored text is the run; the only
approximation is the character stride the detector is checked at.

Usage:
    scripts/court oracle -- python3 scripts/s56_false_cut.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import rollouts, termination  # noqa: E402

RAW_GLOB = "results/stage1/*-raw/*.jsonl"
OUT = "results/stage1/s56-false-cut.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "completion" not in row or "gold" not in row:
                raise SystemExit(f"{path}: a row lacks completion/gold")
            row["dataset"] = "math" if row.get("task") == "math500" else "gsm8k"
            rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=RAW_GLOB)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--stride-chars", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="the registered stage-2b rollout cap")
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    if not paths:
        raise SystemExit(f"s56-false-cut: no raw generations match {args.glob}")
    rows, sources = [], []
    for path in paths:
        loaded = load_rows(path)
        rows.extend(loaded)
        sources.append({"path": os.path.relpath(path, ROOT),
                        "rows": len(loaded), "sha256": sha256_file(path)})
    analysis = rollouts.false_cut_analysis(
        rows, stride_chars=args.stride_chars)
    # The stored completions were generated at a 4096-token cap. Rescale the
    # saving to the registered 2048 rollout cap as a separate, labelled field;
    # the raw saving above is what the stored text supports.
    payload = {
        "schema": "selvedge/s56-false-cut/v1",
        "kind": "false-cut-analysis",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "sources": sources,
        "registered_rollout_cap_tokens": args.max_new_tokens,
        "stored_generation_cap_tokens": 4096,
        "analysis": analysis,
        "reading": (
            "The cut is applied only to a tail the calibrated detector calls a "
            "loop. A false cut is an uncut-correct answer whose cut prefix no "
            "longer verifies correct. The saving is an estimate from the text "
            "length and the row's own token count, not a re-run. If the rate is "
            "too high, the cut is disabled by configuration; the reward still "
            "counts a truncated rollout as wrong."),
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s56-false-cut: {analysis['samples']} stored generations, "
          f"{analysis['cut_samples']} would be cut "
          f"({analysis['cut_fraction']:.4f})")
    print(f"s56-false-cut: saved {analysis['saved_tokens_estimate']:.0f} tokens "
          f"(estimate), false cuts {analysis['false_cuts']} of "
          f"{analysis['correct_uncut']} uncut-correct "
          f"(rate {analysis['false_cut_rate']})")
    print(f"s56-false-cut: thresholds {termination.DEFAULT_THRESHOLDS}")
    print(f"s56-false-cut: {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
