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


def load_rows(path: str, excluded: set[str], label: str,
              seen_global: set[tuple[str, str]] | None = None,
              seen_prompt: dict[str, set[str]] | None = None
              ) -> tuple[list[dict], dict]:
    """Read one accepted file, sharing the dedup sets across shards.

    The same prompt may legitimately contribute several distinct sampled
    trajectories (K>1); only an exact generated-token sequence is removed. The
    sets are passed in so a sharded pass is deduplicated across shards exactly
    as one pass would be.
    """
    if seen_global is None:
        seen_global = set()
    if seen_prompt is None:
        seen_prompt = {}
    rows: list[dict] = []
    excluded_rows = 0
    duplicates = 0
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


def read_budget_receipt(path: str) -> dict:
    """Read the supervised-token budget from a separate corpus receipt.

    The `control-teacher` arm has no student rollouts of its own, so its budget
    cannot be computed from rows inside that invocation. It is taken from a
    previously built student arm's corpus receipt instead of defaulting to zero
    (which silently produced an empty control). The receipt must be the same
    schema, name a student arm, carry a positive supervised-solution-token
    count, name a revision, and (when the examples file is present) its
    recorded SHA-256 must match the file. Any mismatch fails closed.
    """
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(full):
        raise SystemExit(f"s59-corpus: budget receipt {path} does not exist")
    with open(full, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != SCHEMA:
        raise SystemExit(
            f"s59-corpus: budget receipt {path} has schema "
            f"{payload.get('schema')!r}, expected {SCHEMA!r}")
    arm = payload.get("arm")
    if arm not in ("rft-only", "rft-teacher-5050"):
        raise SystemExit(
            f"s59-corpus: budget receipt {path} names arm {arm!r}; the budget "
            "must come from a student arm")
    budget = payload.get("supervised_solution_tokens")
    if not isinstance(budget, int) or budget <= 0:
        raise SystemExit(
            f"s59-corpus: budget receipt {path} has supervised_solution_tokens "
            f"{budget!r}; a matched control needs a positive budget")
    revision = payload.get("revision")
    if not revision or revision == "unknown":
        raise SystemExit(
            f"s59-corpus: budget receipt {path} has no usable revision")
    examples = payload.get("examples") or {}
    example_path = examples.get("path")
    if example_path and examples.get("sha256"):
        candidate = (example_path if os.path.isabs(example_path)
                     else os.path.join(ROOT, example_path))
        if os.path.isfile(candidate):
            got = sha256_file(candidate)
            if got != examples["sha256"]:
                raise SystemExit(
                    f"s59-corpus: budget receipt {path} records examples "
                    f"sha256 {examples['sha256']}, file has {got}")
    return {
        "path": os.path.relpath(full, ROOT) if full.startswith(ROOT) else full,
        "sha256": sha256_file(full),
        "arm": arm,
        "supervised_solution_tokens": budget,
        "stamp": payload.get("stamp"),
        "revision": revision,
        "examples": examples,
        "basis": "the matched control's teacher budget is read from this "
                 "student arm's corpus receipt, not recomputed as zero",
    }


def plan_mixture(arm: str, student_rows: list[dict], teacher_rows: list[dict],
                 student_budget: int | None, seed: int
                 ) -> tuple[list[dict], dict]:
    """Build the chosen rows and the mixture report for one attribution arm.

    `rft-only` keeps every student row. `control-teacher` subsamples the teacher
    rows to `student_budget`, which must be the supervised-token count of a
    separately receipted student corpus. `rft-teacher-5050` matches the two
    sources to each other by supervised solution tokens, so the mixture is
    exact rather than approximate. The whole-row rule keeps the achieved count
    within one row of the request.
    """
    student_tokens = sum(len(row["solution_ids"]) for row in student_rows)
    teacher_tokens = sum(len(row["solution_ids"]) for row in teacher_rows)
    if arm == "rft-only":
        if not student_rows:
            raise SystemExit("s59-corpus: rft-only needs student rows")
        return student_rows, {
            "student_tokens": student_tokens,
            "teacher_tokens": 0,
            "basis": "student rows only",
        }
    if arm == "control-teacher":
        if not teacher_rows:
            raise SystemExit("s59-corpus: control-teacher needs teacher rows")
        if student_budget is None or student_budget <= 0:
            raise SystemExit(
                "s59-corpus: control-teacher needs a positive budget from a "
                "student corpus receipt; refusing a zero-token control")
        chosen, report = subsample_to_tokens(teacher_rows, student_budget, seed)
        report.update({
            "student_tokens": 0,
            "teacher_tokens": sum(len(row["solution_ids"]) for row in chosen),
            "basis": "teacher rows subsampled to the separately receipted "
                     "student arm's supervised solution tokens",
        })
        return chosen, report
    if arm == "rft-teacher-5050":
        if not student_rows or not teacher_rows:
            raise SystemExit("s59-corpus: rft-teacher-5050 needs both sources")
        if student_tokens <= teacher_tokens:
            extra, report = subsample_to_tokens(teacher_rows, student_tokens, seed)
            chosen = list(student_rows) + extra
            mixture = {
                "student_tokens": student_tokens,
                "teacher_tokens": report["achieved"],
                "teacher_subsample": report,
                "basis": "teacher subsampled to the student total",
            }
        else:
            extra, report = subsample_to_tokens(student_rows, teacher_tokens, seed)
            chosen = list(teacher_rows) + extra
            mixture = {
                "student_tokens": report["achieved"],
                "teacher_tokens": teacher_tokens,
                "student_subsample": report,
                "basis": "student subsampled to the teacher total",
            }
        mixture["achieved_student_share"] = (
            mixture["student_tokens"]
            / (mixture["student_tokens"] + mixture["teacher_tokens"]))
        mixture["requested_student_share"] = 0.5
        return chosen, mixture
    raise SystemExit(f"s59-corpus: unknown arm {arm!r}")


def write_examples(path: str, rows: list[dict]) -> dict:
    """Write the trainer's examples file, preserving the replay EOS mask.

    A truncated replay document must reach the packer with
    `supervise_eos=False`; dropping the flag here would silently re-supervise a
    synthetic EOS and undo the S60 fix at the last step.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "id": row.get("id"), "kind": row.get("kind", "teacher"),
                "dataset": row.get("dataset"),
                "prompt_ids": row["prompt_ids"],
                "solution_ids": row["solution_ids"]}
            if row.get("supervise_eos") is False:
                payload["supervise_eos"] = False
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return {"path": path, "sha256": sha256_file(path), "rows": len(rows)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        choices=["rft-only", "rft-teacher-5050",
                                 "control-teacher"])
    parser.add_argument("--rollouts", action="append", default=None,
                        help="student-policy accepted jsonl; repeatable, merged "
                             "in order with cross-shard exact-token dedup")
    parser.add_argument("--rollouts-receipt", action="append", default=[],
                        help="receipt(s) of the generation pass(es) that wrote "
                             "the rollouts; recorded, not re-derived")
    parser.add_argument("--budget-receipt", default=None,
                        help="corpus receipt of a student arm; required for "
                             "--arm control-teacher, whose teacher rows are "
                             "subsampled to that receipt's supervised-token "
                             "budget")
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
        seen_global: set[tuple[str, str]] = set()
        seen_prompt: dict[str, set[str]] = {}
        per_source = []
        for raw in args.rollouts:
            path = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
            rows, report = load_rows(path, excluded, "student",
                                     seen_global, seen_prompt)
            per_source.append(report)
            student_rows.extend(rows)
        student_rows.sort(key=lambda row: (row["dataset"], str(row["id"]),
                                           int(row.get("sample_index", 0))))
        student_report = {
            "sources": per_source,
            "rows": len(student_rows),
            "excluded_holdout_rows": sum(
                item["excluded_holdout_rows"] for item in per_source),
            "duplicate_rows_removed": sum(
                item["duplicate_rows_removed"] for item in per_source),
            "supervised_solution_tokens": sum(
                item["supervised_solution_tokens"] for item in per_source),
            "dedup": "shared across sources; exact solution-token sequence "
                     "within a prompt",
        }
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
    budget_receipt = None
    if args.arm == "control-teacher":
        if not args.budget_receipt:
            raise SystemExit(
                "s59-corpus: --arm control-teacher requires --budget-receipt "
                "naming a built student corpus; otherwise the control has no "
                "matched supervised-token budget")
        budget_receipt = read_budget_receipt(args.budget_receipt)
        student_budget = budget_receipt["supervised_solution_tokens"]
    elif args.arm in ("rft-only", "rft-teacher-5050"):
        student_budget = student_tokens
    else:
        student_budget = None
    chosen, mix_report = plan_mixture(
        args.arm, student_rows, teacher_rows, student_budget, args.seed)
    mix_report["student_source_tokens"] = student_tokens
    mix_report["teacher_source_tokens"] = teacher_tokens
    if budget_receipt is not None:
        mix_report["budget_receipt"] = budget_receipt
        mix_report["matched_budget"] = student_budget
        mix_report["achieved_minus_requested"] = (
            sum(len(row["solution_ids"]) for row in chosen) - student_budget)

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
    if student_report is not None and student_rows:
        merged_accepted_path = os.path.join(
            out_dir, f"{args.label}-accepted.jsonl")
        with open(merged_accepted_path, "w", encoding="utf-8") as handle:
            for row in student_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        student_report["merged"] = {
            "path": os.path.relpath(merged_accepted_path, ROOT),
            "sha256": sha256_file(merged_accepted_path),
            "rows": len(student_rows),
            "basis": "the deduplicated, hold-out-excluded rows the corpus was "
                     "built from, with every accepted-row field preserved so "
                     "the verifier can re-run the answer check",
        }
    examples_path = os.path.join(out_dir, f"{args.label}-examples.jsonl")
    write_examples(examples_path, ordered)

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
    supervised_positions_with_eos = sum(
        len(row["solution_ids"]) + (0 if row.get("supervise_eos") is False
                                    else 1)
        for row in ordered)
    one_epoch_processed_tokens = len(ordered) * args.seq_len
    payload = {
        "schema": SCHEMA,
        "kind": "stage2b-arm-corpus",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": "docs/ws19-post-trained-route-stage2b-preregistration-v1.json",
        "corrections_registration": "docs/ws19-stage2-correction-s63-preregistration-v1.json",
        "arm": args.arm,
        "label": args.label,
        "tokenizer_dir": args.tokenizer_dir,
        "holdout_excluded": {"path": args.holdout, "ids": len(excluded)},
        "sources": {"student": student_report, "teacher": teacher_report,
                    "teacher_default": TEACHER_SOURCE,
                    "budget_receipt": budget_receipt,
                    "generation_receipts": [
                        {"path": path,
                         "stamp": json.load(open(os.path.join(ROOT, path),
                                                 encoding="utf-8")).get("stamp"),
                         "revision": json.load(open(os.path.join(ROOT, path),
                                                    encoding="utf-8")).get("revision"),
                         "acceptance_totals": json.load(open(
                             os.path.join(ROOT, path),
                             encoding="utf-8")).get("acceptance", {}).get("totals")}
                        for path in args.rollouts_receipt]},
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
        "training_arithmetic": {
            "rows": len(ordered),
            "seq_len": args.seq_len,
            "one_epoch_processed_tokens": one_epoch_processed_tokens,
            "supervised_positions_with_eos": supervised_positions_with_eos,
            "cadence_points_tokens": [500000, 1000000, 2000000, 5000000],
            "cadence_5m_reachable_in_one_epoch": (
                one_epoch_processed_tokens >= 5000000),
            "basis": "processed tokens are rows x seq_len at two rows per "
                     "optimiser step, the same nominal accounting as the S58 "
                     "cadence; supervised_positions_with_eos is the actual "
                     "number of loss positions",
        },
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
