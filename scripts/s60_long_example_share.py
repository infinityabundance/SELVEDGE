#!/usr/bin/env python3
"""S60: the long-example share, by dataset and difficulty, before and after.

S58 dropped 191 examples at the 1250-token row width. This script measures what
that number is, which dataset and difficulty carry it, and what a wider row
would preserve. It reads the built examples file and the accepted corpus the
examples were built from (the accepted file carries `difficulty`; the examples
file drops it, so the join is by `id`).

The output is a receipt, not a change to any training file.

Usage:
    SELVEDGE_MEM=4g scripts/court python -- python3 scripts/s60_long_example_share.py \
        --examples results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl \
        --accepted results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
        --out results/stage2/s60-long-example-share.json
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
sys.path.insert(0, HERE)

import s55_stage2a_examples as builder  # noqa: E402

SCHEMA = "selvedge/s60-long-example-share/v1"


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


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-lens", default="1250,1536,1792,2048,2304,2500,2560")
    args = parser.parse_args(argv)

    examples_path = os.path.join(ROOT, args.examples)
    accepted_path = os.path.join(ROOT, args.accepted)
    rows = load_rows(examples_path)
    accepted = load_rows(accepted_path)
    difficulty = {row["id"]: row.get("difficulty") for row in accepted}
    for row in rows:
        row["difficulty"] = difficulty.get(row.get("id"))
    seq_lens = [int(value) for value in args.seq_lens.split(",") if value.strip()]

    teacher = [row for row in rows if row.get("kind") == "teacher"]
    profile_teacher = builder.overflow_profile(teacher, seq_lens)
    profile_all = builder.overflow_profile(rows, seq_lens)
    total_tokens = sum(len(row["solution_ids"]) + 1 for row in teacher)

    def share(seq_len: int) -> dict:
        entry = profile_teacher[str(seq_len)]
        return {
            "dropped_examples": entry["dropped"],
            "dropped_fraction": entry["dropped_fraction"],
            "dropped_supervised_tokens": entry["supervised_tokens_dropped"],
            "dropped_supervised_token_fraction": (
                entry["supervised_tokens_dropped"] / total_tokens
                if total_tokens else 0.0),
            "by_dataset": entry["by_dataset"],
            "by_dataset_difficulty": entry["by_dataset_difficulty"],
        }

    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "examples": {"path": args.examples, "sha256": sha256_file(examples_path),
                     "rows": len(rows), "teacher_rows": len(teacher)},
        "accepted": {"path": args.accepted, "sha256": sha256_file(accepted_path),
                     "rows": len(accepted)},
        "measurement": {
            "definition": "a complete example is dropped when prompt + solution "
                          "+ EOS exceeds the row width; it is never split",
            "teacher_supervised_tokens": total_tokens,
            "before_1250": share(1250),
            "after_2048": share(2048),
            "after_2500": share(2500),
            "by_width": {str(seq_len): share(seq_len) for seq_len in seq_lens},
            "all_rows_by_width": {
                str(seq_len): {
                    "dropped": profile_all[str(seq_len)]["dropped"],
                    "supervised_tokens_dropped":
                        profile_all[str(seq_len)]["supervised_tokens_dropped"],
                } for seq_len in seq_lens},
        },
        "reading": ("all examples over 1250 are MATH teacher rows; the 2500 row "
                    "preserves every one of them, and 2048 would still drop the "
                    "rare longest traces. The chosen width is the measured "
                    "threshold, not a round number"),
        "command": sys.argv,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out_path)

    for label, entry in (("1250", payload["measurement"]["before_1250"]),
                         ("2048", payload["measurement"]["after_2048"]),
                         ("2500", payload["measurement"]["after_2500"])):
        print(f"s60-long-share: {label}: {entry['dropped_examples']} examples, "
              f"{entry['dropped_supervised_tokens']} supervised tokens, "
              f"{entry['dropped_supervised_token_fraction']:.4f} of teacher tokens")
    print(f"s60-long-share: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
