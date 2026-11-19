#!/usr/bin/env python3
"""S59 stage 2b: build one arm's matched corpus and its SFT examples.

The stage-2b registration defines three arms and the meaning of "matched":

* `rft-only`        the student's verified trajectories only;
* `rft-teacher-5050` student trajectories and stage-2a teacher solutions, mixed
  50/50 by supervised solution tokens;
* `control-teacher` teacher solutions only, subsampled to the same supervised
  solution-token count as `rft-only`.

`matched` is a token-level property, not a prompt-level one: identical
supervised solution tokens (to within one row), identical replay fraction,
identical training geometry (the launcher enforces the rest).

Every arm excludes the frozen stage-2a hold-out ids, so
`results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl` remains a
clean cross-arm CE instrument. Replay is the pinned fineweb-edu shard at 20% of
supervised tokens with the stage-2a contamination screen, produced by the same
function stage 2a used (`scripts/s55_stage2a_examples.py`), never a copy of it.

Nothing in the receipt is typed: row counts, token counts, achieved shares and
the overflow profile are computed from the rows written.

Usage
-----
    SELVEDGE_MEM=6g scripts/court python -- python3 scripts/s59_build_corpus.py \
        --arm rft-only \
        --rollouts results/artifacts/stage2/s59-rft/rft-accepted.jsonl \
        --rollouts-receipt results/artifacts/stage2/s59-rft/rft-receipt.json \
        --holdout results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
        --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc0636... \
        --label stage2b-rft-only --out-dir results/artifacts/stage2/s59-corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import rft  # noqa: E402
from s55_stage2a_examples import replay_documents  # noqa: E402

SCHEMA = "selvedge/s59-stage2b-corpus/v1"
TEACHER_SOURCE = "results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_rows(path: str, excluded: set[str], label: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    excluded_rows = 0
    duplicates = 0
    seen_global: set[tuple[str, str]] = set()
    seen_prompt: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["id"]) in excluded:
                excluded_rows += 1
                continue
            key = (str(row["id"]), hashlib.sha256(
                ",".join(str(i) for i in row["solution_ids"]).encode()
            ).hexdigest())
            if key in seen_global:
                duplicates += 1
                continue
            prompt_seen = seen_prompt.setdefault(str(row["id"]), set())
            if key[1] in prompt_seen:
                duplicates += 1
                continue
            prompt_seen.add(key[1])
            seen_global.add(key)
            rows.append({"kind": label, **row})
    rows.sort(key=lambda row: (row["dataset"], str(row["id"]),
                              int(row.get("sample_index", 0))))
    return rows, {"source": path, "rows": len(rows),
                  "excluded_holdout_rows": excluded_rows,
                  "duplicate_rows_removed": duplicates,
                  "supervised_solution_tokens": sum(
                      len(row["solution_ids"]) for row in rows)}


def subsample_to_tokens(rows: list[dict], target: int,
                        seed: int) -> tuple[list[dict], dict]:
    """Deterministically take whole rows up to `target` supervised tokens.

    The final row is included only when including it moves the total closer to
    `target` than leaving it out, so the achieved count is within one row of the
    request. The row order used for the decision is a seeded permutation, so the
    choice is reproducible and not a prefix of the file.
    """
    if target >= sum(len(row["solution_ids"]) for row in rows):
        return rows, {"target": target, "achieved": sum(
            len(row["solution_ids"]) for row in rows), "dropped": 0,
            "rule": "kept everything; the source is at or below the target"}
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    chosen: list[int] = []
    total = 0
    dropped = 0
    for index in order:
        length = len(rows[index]["solution_ids"])
        if total + length <= target:
            chosen.append(index)
            total += length
        else:
            if target - total > (total + length) - target:
                chosen.append(index)
                total += length
            dropped += 1
    selected = sorted((rows[index] for index in chosen),
                      key=lambda row: (row["dataset"], str(row["id"]),
                                       int(row.get("sample_index", 0))))
    return selected, {"target": target, "achieved": total,
                      "rows_kept": len(selected), "rows_dropped": dropped,
                      "rule": "seeded permutation, whole rows, final row chosen "
                              "by smaller absolute deviation"}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        choices=["rft-only", "rft-teacher-5050",
                                 "control-teacher"])
    parser.add_argument("--rollouts", default=None,
                        help="student-policy accepted jsonl")
    parser.add_argument("--rollouts-receipt", default=None)
    parser.add_argument("--teacher", default=TEACHER_SOURCE)
    parser.add_argument("--holdout", default=(
        "results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl"))
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1250)
    parser.add_argument("--replay-fraction", type=float, default=0.20)
    parser.add_argument("--replay-max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=5900)
    args = parser.parse_args(argv)

    excluded = set()
    with open(os.path.join(ROOT, args.holdout), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                excluded.add(str(json.loads(line)["id"]))

    student_rows: list[dict] = []
    student_report = None
    if args.arm in ("rft-only", "rft-teacher-5050"):
        if not args.rollouts:
            raise SystemExit(f"s59-corpus: {args.arm} needs --rollouts")
        student_rows, student_report = load_rows(
            os.path.join(ROOT, args.rollouts), excluded, "student")
        if not student_rows:
            raise SystemExit("s59-corpus: no student rows survive")
    teacher_rows: list[dict] = []
    teacher_report = None
    if args.arm in ("control-teacher", "rft-teacher-5050"):
        teacher_rows, teacher_report = load_rows(
            os.path.join(ROOT, args.teacher), excluded, "teacher")
        if not teacher_rows:
            raise SystemExit("s59-corpus: no teacher rows survive")

    student_tokens = sum(len(row["solution_ids"]) for row in student_rows)
    teacher_tokens = sum(len(row["solution_ids"]) for row in teacher_rows)
    mix_report: dict = {}
    if args.arm == "rft-only":
        chosen = student_rows
        mix_report = {"student_tokens": student_tokens,
                      "teacher_tokens": 0,
                      "basis": "student rows only"}
    elif args.arm == "control-teacher":
        chosen, mix_report = subsample_to_tokens(
            teacher_rows, student_tokens, args.seed)
        mix_report.update({"student_tokens": 0,
                           "teacher_tokens": sum(
                               len(row["solution_ids"]) for row in chosen),
                           "basis": "teacher rows subsampled to the student "
                                    "arm's supervised solution tokens"})
    else:
        if student_tokens <= teacher_tokens:
            chosen = student_rows
            extra, extra_report = subsample_to_tokens(
                teacher_rows, student_tokens, args.seed)
            chosen = chosen + extra
            mix_report = {"student_tokens": student_tokens,
                          "teacher_tokens": extra_report["achieved"],
                          "teacher_subsample": extra_report,
                          "basis": "teacher subsampled to the student total"}
        else:
            chosen = teacher_rows
            extra, extra_report = subsample_to_tokens(
                student_rows, teacher_tokens, args.seed)
            chosen = chosen + extra
            mix_report = {"student_tokens": extra_report["achieved"],
                          "teacher_tokens": teacher_tokens,
                          "student_subsample": extra_report,
                          "basis": "student subsampled to the teacher total"}
        mix_report["achieved_student_share"] = (
            mix_report["student_tokens"]
            / (mix_report["student_tokens"] + mix_report["teacher_tokens"]))
        mix_report["requested_student_share"] = 0.5

    supervised = sum(len(row["solution_ids"]) for row in chosen)
    if supervised <= 0:
        raise SystemExit("s59-corpus: no supervised tokens")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, trust_remote_code=False)

    replay_target = int(supervised * args.replay_fraction
                        / (1.0 - args.replay_fraction))
    replay, replay_report = replay_documents(
        tokenizer, max_tokens=args.replay_max_tokens,
        target_tokens=replay_target, seed=args.seed)
    contamination = rft.contamination_check(
        [{"id": doc["id"], "prompt": "",
          "solution": tokenizer.decode(doc["solution_ids"],
                                       skip_special_tokens=True)}
         for doc in replay])
    if contamination["fail_closed"]:
        raise SystemExit(f"s59-corpus: replay crosses the fail threshold: "
                         f"{contamination['examples']}")
    replay_tokens = sum(len(doc["solution_ids"]) for doc in replay)

    ordered = chosen + replay
    random.Random(args.seed).shuffle(ordered)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    examples_path = os.path.join(out_dir, f"{args.label}-examples.jsonl")
    with open(examples_path, "w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps({
                "id": row.get("id"), "kind": row.get("kind", "teacher"),
                "dataset": row.get("dataset"),
                "prompt_ids": row["prompt_ids"],
                "solution_ids": row["solution_ids"]}, sort_keys=True) + "\n")

    lengths = [len(row["prompt_ids"]) + len(row["solution_ids"]) + 1
               for row in chosen]
    over = [length for length in lengths if length > args.seq_len]
    profile = {
        "seq_len": args.seq_len,
        "rows": len(lengths),
        "over_seq_len": len(over),
        "over_seq_len_fraction": len(over) / len(lengths) if lengths else 0.0,
        "over_seq_len_supervised_tokens": sum(
            len(chosen[index]["solution_ids"])
            for index, length in enumerate(lengths) if length > args.seq_len),
        "mean_example_tokens": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_example_tokens": max(lengths) if lengths else 0,
    }
    by_kind: dict[str, dict] = {}
    for row in chosen:
        entry = by_kind.setdefault(row["kind"], {"rows": 0, "solution_tokens": 0,
                                                 "prompt_tokens": 0})
        entry["rows"] += 1
        entry["solution_tokens"] += len(row["solution_ids"])
        entry["prompt_tokens"] += len(row["prompt_ids"])

    total_supervised = supervised + replay_tokens
    payload = {
        "schema": SCHEMA,
        "kind": "stage2b-arm-corpus",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": "docs/ws19-post-trained-route-stage2b-preregistration-v1.json",
        "arm": args.arm,
        "label": args.label,
        "tokenizer_dir": args.tokenizer_dir,
        "holdout_excluded": {"path": args.holdout, "ids": len(excluded)},
        "sources": {"student": student_report, "teacher": teacher_report,
                    "teacher_default": TEACHER_SOURCE},
        "mixture": mix_report,
        "by_kind": by_kind,
        "supervised_solution_tokens": supervised,
        "mix": {
            "requested_replay_fraction": args.replay_fraction,
            "achieved_replay_fraction": (replay_tokens / total_supervised
                                         if total_supervised else 0.0),
            "total_supervised_tokens": total_supervised,
            "basis": "supervised positions after prompt masking",
        },
        "replay": {**replay_report, "contamination": contamination,
                   "supervised_tokens": replay_tokens},
        "length_profile": profile,
        "examples": {"path": os.path.relpath(examples_path, ROOT),
                     "sha256": sha256_file(examples_path),
                     "rows": len(ordered),
                     "order": "seeded shuffle, seed %d" % args.seed},
        "command": sys.argv,
    }
    write_json(os.path.join(out_dir, f"{args.label}-corpus-receipt.json"),
               payload)
    print(f"s59-corpus: {args.arm}: {len(chosen)} corpus rows "
          f"({supervised} supervised tokens) + {len(replay)} replay "
          f"({replay_tokens}) = {total_supervised} supervised, replay "
          f"{payload['mix']['achieved_replay_fraction']:.3f}, "
          f"{len(ordered)} rows -> {payload['examples']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
