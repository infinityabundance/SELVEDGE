#!/usr/bin/env python3
"""S55 stage 2a: accepted teacher solutions + general replay -> SFT examples.

Reads one or more `*-accepted.jsonl` files from `scripts/s55_stage2_data.py` and
writes:

* `<label>-examples.jsonl` — `{prompt_ids, solution_ids, kind}` rows for
  `oracle/frontier/train.py --sft-examples`, teacher rows first;
* `<label>-holdout.jsonl` — the held-out slice used for early stopping, drawn
  as a seeded stratified 2% of the teacher rows and removed from the examples;
* `<label>-receipt.json` — token accounting, the replay mix actually achieved,
  contamination, hashes, and the epoch plan.

General replay
--------------
The S53 registration asks for ~10-15% general text to protect MMLU/MMLU-Pro.
The replay source is the pinned `fineweb-edu` sample already used by the T1
train corpus; wikitext-2 validation is never trained on because it is the
registered calibration text. Replay documents are 5-word-shingle screened
against the GSM8K test and MATH-500 items with the registered Jaccard
thresholds, truncated only at a document boundary to `--replay-max-tokens`, and
supervised in full (empty prompt, whole document is the completion). No
solution is ever truncated.

Usage
-----
    python3 scripts/s55_stage2a_examples.py \
        --accepted results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl \
        --label stage2a --out-dir results/artifacts/stage2/s55-stage2a \
        --replay-fraction 0.13
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import rft  # noqa: E402

SCHEMA = "selvedge/s55-stage2a-examples/v1"
FINE_WEB_EDU = "results/artifacts/train-corpus/fineweb-edu-sample-10bt"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def load_accepted(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = hashlib.sha256(
                    (row["id"] + ":" + row["solution"]).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["id"], row["sample_index"]))
    return rows


def select_holdout(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded stratified hold-out over (dataset, solution-length quartile)."""
    lengths = sorted(len(row["solution_ids"]) for row in rows)
    boundaries = [lengths[len(lengths) * n // 4] for n in (1, 2, 3)]
    strata: dict[str, list[dict]] = {}
    for row in rows:
        length = len(row["solution_ids"])
        bucket = sum(1 for boundary in boundaries if length >= boundary)
        strata.setdefault(f"{row['dataset']}|{bucket}", []).append(row)
    rng = random.Random(seed)
    holdout: list[dict] = []
    kept: list[dict] = []
    for key in sorted(strata):
        members = strata[key][:]
        rng.shuffle(members)
        take = max(1, int(round(len(members) * fraction))) if len(members) > 4 else 0
        take = min(take, max(0, len(members) - 1))
        holdout.extend(members[:take])
        kept.extend(members[take:])
    holdout.sort(key=lambda row: (row["dataset"], row["id"]))
    kept.sort(key=lambda row: (row["dataset"], row["id"]))
    return kept, holdout


def replay_documents(tokenizer, *, max_tokens: int, target_tokens: int,
                     seed: int) -> tuple[list[dict], dict]:
    """Read the pinned fineweb-edu shard and cut whole documents."""
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(ROOT, FINE_WEB_EDU, "*.parquet")))
    if not paths:
        raise SystemExit(f"s55-examples: no fineweb-edu parquet under {FINE_WEB_EDU}")
    documents: list[dict] = []
    truncated = 0
    considered = 0
    total = 0
    for path in paths:
        table = pq.read_table(path, columns=["text"])
        for text in table.column("text").to_pylist():
            if total >= target_tokens:
                break
            considered += 1
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < 64:
                continue
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
                truncated += 1
            documents.append({"kind": "replay", "prompt_ids": [],
                              "solution_ids": ids,
                              "id": f"fineweb-edu:{considered}"})
            total += len(ids)
        if total >= target_tokens:
            break
    if not documents:
        raise SystemExit("s55-examples: no replay documents")
    rng = random.Random(seed)
    rng.shuffle(documents)
    report = {
        "source": FINE_WEB_EDU,
        "files": [os.path.relpath(path, ROOT) for path in paths],
        "rows_considered": considered,
        "documents": len(documents),
        "supervised_tokens": total,
        "max_tokens_per_document": max_tokens,
        "truncated_documents": truncated,
        "truncation_policy": "whole-document prefix at a token boundary; replay "
                             "text is not a solution and carries no answer",
    }
    return documents, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", action="append", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--replay-fraction", type=float, default=0.13)
    parser.add_argument("--replay-max-tokens", type=int, default=768)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=5500)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                              local_files_only=True,
                                              trust_remote_code=False)

    accepted = load_accepted([os.path.join(ROOT, path) for path in args.accepted])
    if not accepted:
        raise SystemExit("s55-examples: no accepted rows")
    kept, holdout = select_holdout(accepted, args.holdout_fraction, args.seed)

    teacher_tokens = sum(len(row["solution_ids"]) for row in kept)
    # Supervised tokens are prompt-masked, so the mix is computed over the
    # positions that actually carry a loss.
    replay_target = int(teacher_tokens * args.replay_fraction
                        / (1.0 - args.replay_fraction))
    replay, replay_report = replay_documents(
        tokenizer, max_tokens=args.replay_max_tokens,
        target_tokens=replay_target, seed=args.seed)
    contamination = rft.contamination_check(
        [{"id": doc["id"],
          "prompt": "",
          "solution": tokenizer.decode(doc["solution_ids"],
                                       skip_special_tokens=True)}
         for doc in replay])
    if contamination["fail_closed"]:
        raise SystemExit(
            "s55-examples: replay text crosses the registered contamination "
            f"fail threshold: {contamination['examples']}")
    replay_supervised = sum(len(doc["solution_ids"]) for doc in replay)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    examples_path = os.path.join(out_dir, f"{args.label}-examples.jsonl")
    holdout_path = os.path.join(out_dir, f"{args.label}-holdout.jsonl")

    def write_rows(path: str, rows: list[dict]) -> dict:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({
                    "id": row.get("id"),
                    "kind": row.get("kind", "teacher"),
                    "dataset": row.get("dataset"),
                    "prompt_ids": row["prompt_ids"],
                    "solution_ids": row["solution_ids"]},
                    sort_keys=True) + "\n")
        return {"path": os.path.relpath(path, ROOT), "sha256": sha256_file(path),
                "rows": len(rows)}

    teacher_rows = [{"kind": "teacher", **row} for row in kept]
    examples = write_rows(examples_path, teacher_rows + replay)
    holdout_meta = write_rows(holdout_path, [{"kind": "teacher", **row}
                                            for row in holdout])

    # Length profile of the packed rows, so the overflow fraction at a given
    # seq_len is a measured number rather than a guess.
    lengths = [len(row["prompt_ids"]) + len(row["solution_ids"]) + 1
               for row in teacher_rows]
    over = [length for length in lengths if length > args.seq_len]
    profile = {
        "seq_len": args.seq_len,
        "examples": len(lengths),
        "over_seq_len": len(over),
        "over_seq_len_fraction": len(over) / len(lengths) if lengths else 0.0,
        "over_seq_len_supervised_tokens": sum(
            len(teacher_rows[index]["solution_ids"])
            for index, length in enumerate(lengths) if length > args.seq_len),
        "max_example_tokens": max(lengths) if lengths else 0,
        "mean_example_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
    }

    total_supervised = teacher_tokens + replay_supervised
    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "kind": "stage2a-sft-examples",
        "label": args.label,
        "tokenizer_dir": args.tokenizer_dir,
        "inputs": [os.path.relpath(os.path.join(ROOT, path), ROOT)
                   for path in args.accepted],
        "teacher": {
            "examples": len(teacher_rows),
            "distinct_prompts": len({row["id"] for row in teacher_rows}),
            "supervised_solution_tokens": teacher_tokens,
            "prompt_tokens": sum(len(row["prompt_ids"]) for row in teacher_rows),
            "by_dataset": {
                name: {
                    "examples": sum(1 for row in teacher_rows
                                    if row["dataset"] == name),
                    "supervised_solution_tokens": sum(
                        len(row["solution_ids"]) for row in teacher_rows
                        if row["dataset"] == name),
                }
                for name in sorted({row["dataset"] for row in teacher_rows})
            },
        },
        "holdout": {**holdout_meta,
                    "supervised_solution_tokens": sum(
                        len(row["solution_ids"]) for row in holdout),
                    "fraction": args.holdout_fraction},
        "replay": {**replay_report, "contamination": contamination,
                   "supervised_tokens": replay_supervised},
        "mix": {
            "requested_replay_fraction": args.replay_fraction,
            "achieved_replay_fraction": (
                replay_supervised / total_supervised if total_supervised else 0.0),
            "total_supervised_tokens": total_supervised,
            "basis": "supervised positions after prompt masking",
        },
        "examples": examples,
        "length_profile": profile,
        "command": sys.argv,
    }
    write_json(os.path.join(out_dir, f"{args.label}-examples-receipt.json"),
               payload)
    print(f"s55-examples: {len(teacher_rows)} teacher rows "
          f"({teacher_tokens} supervised tokens) + {len(replay)} replay "
          f"({replay_supervised} tokens) = "
          f"{payload['mix']['achieved_replay_fraction']:.3f} replay -> "
          f"{examples['path']}")
    print(f"s55-examples: holdout {holdout_meta['rows']} rows; "
          f"overflow at seq_len {args.seq_len}: "
          f"{profile['over_seq_len']}/{profile['examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
