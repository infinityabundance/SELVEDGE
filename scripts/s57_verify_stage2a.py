#!/usr/bin/env python3
"""S57: verify the stage-2a teacher corpus before anything is trained on it.

The S55 amendment is only worth what its artifacts say, so this script recounts
the accepted corpus from the JSONL itself instead of quoting the generation
receipt:

* rows, distinct prompts, per-source counts and the acceptance arithmetic
  (kept / sampled, correct / sampled);
* truncation of teacher solutions: a completion is truncated when it reached
  `max_new_tokens` without EOS, and such a completion is never a target. The
  per-source truncated fraction is reported beside the kept rate, because a
  high kept rate on heavily truncated data would mean the cap is doing the
  selection;
* the supervised token total and the prompt total, per source;
* solution length profile (mean, p50, p90, p99, max, fraction at the cap) so the
  SFT `seq_len` is chosen from a measurement;
* duplicate detection across both sources (exact generated-token sequence);
* an independent contamination pass over the accepted rows with the registered
  5-word-shingle Jaccard guard against the cached GSM8K/MATH-500 test items.

The `verdict` block applies the S57 mission gate: acceptance at or above 40% and
no evidence that the cap is truncating the solutions being kept. It does not
gate on a number it did not measure.

S60 adds the two checks the launcher needs before training: `frontier.rft.verify`
is re-run on **every** accepted solution (a stored `extracted` field is not
proof, and a generator-side bug would be invisible to a recount), and the
built examples file is compared byte-for-byte with the SHA-256 its own receipt
declares. The corpus gate fails closed unless every re-verified solution is
`correct` and the examples hash matches.

Usage:
    SELVEDGE_MEM=4g scripts/court python -- python3 scripts/s57_verify_stage2a.py \
        --receipt results/artifacts/stage2/s55-stage2a/stage2a-receipt.json \
        --accepted results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl \
        --examples results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl \
        --examples-receipt results/artifacts/stage2/s58-combined/stage2a-combined-examples-receipt.json \
        --out results/stage2/s57-stage2a-verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import rft  # noqa: E402

SCHEMA = "selvedge/s57-stage2a-verification/v1"
ACCEPTANCE_FLOOR = 0.40
TRUNCATION_CEILING = 0.20
# The stage-2b corpus gate. The registration sets no truncation ceiling for a
# student-policy pass: a completion that reaches the cap without EOS is counted
# and can never become a target, so truncation is a reported diagnostic rather
# than a corpus failure. The acceptance floor is the operator's "too low to
# train" line (25%); the pilot measured ~50%.
STAGE2B_ACCEPTANCE_FLOOR = 0.25
STAGE2B_TRUNCATION_CEILING = None


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


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), ROOT)


def reverify(rows: list[dict]) -> dict:
    """Re-run the verifier on every accepted solution.

    The generator stores an `extracted` value and a `verification_method`; this
    pass ignores both and recomputes the verdict from the solution text and the
    train label with `frontier.rft.verify`, the same extractor every score uses.
    """
    verdicts: dict[str, int] = {}
    failures: list[dict] = []
    for row in rows:
        dataset = "gsm8k" if row.get("dataset") == "gsm8k" else "math"
        result = rft.verify(row["solution"], row["target"], dataset)
        verdict = result["verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        stored = row.get("extracted")
        if verdict != "correct":
            failures.append({
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "verdict": verdict,
                "extracted": result["extracted"],
                "target": result["target"],
                "stored_extracted": stored,
                "reason": result["reason"],
            })
    return {
        "rows": len(rows),
        "verdicts": verdicts,
        "all_correct": not failures,
        "failure_count": len(failures),
        "failures": failures[:50],
        "verifier": "frontier.rft.verify, recomputed for every accepted row",
    }


def examples_check(examples: str | None, examples_receipt: str | None,
                   accepted_rel: str, accepted_sha: str) -> dict | None:
    """Compare the built examples file with its receipt and the accepted input."""
    if not examples:
        return None
    path = os.path.join(ROOT, examples)
    if not os.path.isfile(path):
        raise SystemExit(f"s57-verify: examples file is missing: {examples}")
    actual = sha256_file(path)
    block = {
        "path": _relative(path),
        "sha256": actual,
        "receipt": None,
        "receipt_sha256": None,
        "matches": None,
        "accepted_input_matches": None,
    }
    if examples_receipt:
        receipt_path = os.path.join(ROOT, examples_receipt)
        if not os.path.isfile(receipt_path):
            raise SystemExit(
                f"s57-verify: examples receipt is missing: {examples_receipt}")
        built = json.load(open(receipt_path, encoding="utf-8"))
        recorded = (built.get("examples") or {}).get("sha256")
        block["receipt"] = _relative(receipt_path)
        block["receipt_sha256"] = sha256_file(receipt_path)
        block["receipt_recorded_sha256"] = recorded
        block["matches"] = bool(recorded) and recorded == actual
        block["declared_inputs"] = _declared_inputs(built)
        block["accepted_input_matches"] = any(
            _relative(os.path.join(ROOT, entry)) == accepted_rel
            and _accepted_sha_for(entry) == accepted_sha
            for entry in block["declared_inputs"])
    return block


def _declared_inputs(built: dict) -> list[str]:
    """Every accepted file a corpus receipt declares as an input.

    The stage-2a builder writes `inputs`; the stage-2b builder records the
    rollouts under `sources.student`. Both are read here so the examples-hash
    gate works on either receipt without a second verifier.
    """
    declared: list[str] = []
    for entry in built.get("inputs") or []:
        if isinstance(entry, str):
            declared.append(entry)
    student = (built.get("sources") or {}).get("student") or {}
    if isinstance(student, dict):
        if student.get("source"):
            declared.append(student["source"])
        for item in student.get("sources") or []:
            if item.get("source"):
                declared.append(item["source"])
        merged = student.get("merged") or {}
        if isinstance(merged, dict) and merged.get("path"):
            declared.append(merged["path"])
    teacher = (built.get("sources") or {}).get("teacher") or {}
    if isinstance(teacher, dict) and teacher.get("source"):
        declared.append(teacher["source"])
    seen: set[str] = set()
    unique: list[str] = []
    for path in declared:
        key = _relative(os.path.join(ROOT, path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def acceptance_block(receipt: dict) -> dict:
    """The acceptance block of a generation or a built-corpus receipt.

    A single generation receipt carries `acceptance` at the top level. A built
    stage-2b corpus receipt records the generation receipts it consumed, so the
    exact per-dataset counts and denominators are re-summed from those files
    rather than inferred from row counts.
    """
    if isinstance(receipt.get("acceptance"), dict):
        return receipt["acceptance"]
    entries = (receipt.get("sources") or {}).get("generation_receipts") or []
    if not entries:
        return {}
    by_dataset: dict[str, dict] = {}
    totals: dict[str, float] = {}
    for entry in entries:
        path = os.path.join(ROOT, entry.get("path", ""))
        if not os.path.isfile(path):
            continue
        payload = json.load(open(path, encoding="utf-8"))
        block = payload.get("acceptance") or {}
        for name, item in (block.get("by_dataset") or {}).items():
            target = by_dataset.setdefault(name, {})
            for key, value in item.items():
                if key == "solution_lengths":
                    target.setdefault(key, []).extend(value)
                elif key == "verdicts":
                    verdicts = target.setdefault(key, {})
                    for verdict, count in value.items():
                        verdicts[verdict] = verdicts.get(verdict, 0) + count
                elif isinstance(value, (int, float)):
                    target[key] = target.get(key, 0) + value
        for key, value in (block.get("totals") or {}).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    sampled = totals.get("sampled") or sum(
        (item.get("sampled") or 0) for item in by_dataset.values())
    kept = totals.get("kept") or sum(
        (item.get("kept") or 0) for item in by_dataset.values())
    truncated = totals.get("truncated") or sum(
        (item.get("truncated") or 0) for item in by_dataset.values())
    loop_cut = totals.get("loop_cut") or sum(
        (item.get("loop_cut") or 0) for item in by_dataset.values())
    correct = totals.get("correct") or sum(
        (item.get("correct") or 0) for item in by_dataset.values())
    return {
        "by_dataset": by_dataset,
        "totals": {**totals, "sampled": sampled, "kept": kept,
                   "correct": correct, "truncated": truncated,
                   "loop_cut": loop_cut},
        "kept_rate": (kept / sampled) if sampled else None,
        "verified_rate": (correct / sampled) if sampled else None,
        "truncation_rate": (truncated / sampled) if sampled else None,
        "loop_cut_rate": (loop_cut / sampled) if sampled else None,
        "reconstructed_from": [entry.get("path") for entry in entries],
    }


def _accepted_sha_for(entry: str) -> str | None:
    path = os.path.join(ROOT, entry)
    return sha256_file(path) if os.path.isfile(path) else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--examples", default=None,
                        help="the built SFT examples file; its SHA-256 must match "
                             "the examples receipt before training")
    parser.add_argument("--examples-receipt", default=None,
                        help="the examples builder's receipt (declares "
                             "examples.sha256 and the accepted inputs)")
    parser.add_argument("--skip-contamination", action="store_true",
                        help="only for a dry pass over a smoke corpus")
    parser.add_argument("--skip-reverification", action="store_true",
                        help="only for an explicitly labelled smoke pass")
    parser.add_argument("--corpus-kind", choices=("stage2a", "stage2b"),
                        default="stage2a",
                        help="stage2a applies the 40%% acceptance floor and the "
                             "20%% truncation ceiling; stage2b applies the 25%% "
                             "floor and reports truncation without a ceiling, "
                             "because a truncated completion is never a target")
    args = parser.parse_args(argv)

    receipt_path = os.path.join(ROOT, args.receipt)
    accepted_path = os.path.join(ROOT, args.accepted)
    receipt = json.load(open(receipt_path, encoding="utf-8"))

    rows = []
    with open(accepted_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("s57-verify: the accepted file is empty")

    by_source: dict[str, dict] = {}
    seen: dict[str, str] = {}
    duplicates = []
    for row in rows:
        source = row["dataset"]
        entry = by_source.setdefault(source, {
            "rows": 0, "distinct_prompts": set(), "solution_tokens": 0,
            "prompt_tokens": 0, "lengths": [], "max_new_tokens": set()})
        entry["rows"] += 1
        entry["distinct_prompts"].add(row["id"])
        entry["solution_tokens"] += len(row["solution_ids"])
        entry["prompt_tokens"] += len(row["prompt_ids"])
        entry["lengths"].append(len(row["solution_ids"]))
        entry["max_new_tokens"].add(row.get("max_new_tokens"))
        key = hashlib.sha256(
            ",".join(str(value) for value in row["solution_ids"]).encode()
        ).hexdigest()
        if key in seen:
            duplicates.append({"id": row["id"], "dataset": source,
                               "same_as": seen[key]})
        else:
            seen[key] = f"{source}:{row['id']}"

    acceptance = acceptance_block(receipt)
    per_dataset_acceptance = acceptance.get("by_dataset", {})
    sources = {}
    for source, entry in sorted(by_source.items()):
        lengths = entry["lengths"]
        cap = sorted(value for value in entry["max_new_tokens"]
                     if value is not None)
        cap_value = cap[0] if cap else None
        generated = per_dataset_acceptance.get(source, {})
        sampled = generated.get("sampled")
        correct = generated.get("correct")
        kept = generated.get("kept")
        truncated = generated.get("truncated")
        sources[source] = {
            "rows": entry["rows"],
            "distinct_prompts": len(entry["distinct_prompts"]),
            "prompt_tokens_accepted": entry["prompt_tokens"],
            "solution_tokens_accepted": entry["solution_tokens"],
            "solution_tokens": {
                "mean": statistics.fmean(lengths),
                "p50": percentile(lengths, 0.50),
                "p90": percentile(lengths, 0.90),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
                "cap": cap_value,
                "fraction_at_cap": (sum(1 for value in lengths
                                        if cap_value is not None
                                        and value >= cap_value) / len(lengths)),
                "fraction_at_or_above_90_percent_of_cap": (
                    sum(1 for value in lengths
                        if cap_value is not None
                        and value >= 0.9 * cap_value) / len(lengths)),
            },
            "sampling": {
                "prompts_generated": generated.get("prompts"),
                "prompts_with_a_kept_solution": len(entry["distinct_prompts"]),
                "completions_sampled": sampled,
                "correct": correct,
                "kept": kept,
                "duplicates": generated.get("duplicates"),
                "truncated_without_eos": truncated,
                "loop_cut": generated.get("loop_cut"),
                "acceptance_kept_over_sampled": (
                    kept / generated["sampled"]
                    if generated.get("sampled") else None),
                "correct_over_sampled": (
                    correct / generated["sampled"]
                    if generated.get("sampled") and correct is not None else None),
                "truncated_over_sampled": (
                    truncated / generated["sampled"]
                    if generated.get("sampled") and truncated is not None
                    else None),
                "verdicts": generated.get("verdicts"),
                "generated_tokens": generated.get("generated_tokens"),
            },
        }

    totals = acceptance.get("totals", {})
    overall = {
        "rows": len(rows),
        "distinct_prompts": len({(row["dataset"], row["id"]) for row in rows}),
        "solution_tokens_accepted": sum(
            len(row["solution_ids"]) for row in rows),
        "prompt_tokens_accepted": sum(len(row["prompt_ids"]) for row in rows),
        "supervised_positions_with_eos": sum(
            len(row["solution_ids"]) + 1 for row in rows),
        "duplicates_across_sources": duplicates,
        "acceptance_kept_over_sampled": acceptance.get("kept_rate"),
        "acceptance_correct_over_sampled": acceptance.get("verified_rate"),
        "truncated_over_sampled": (
            totals.get("truncated") / totals.get("sampled")
            if totals.get("sampled") else None),
        "loop_cut_over_sampled": (
            totals.get("loop_cut") / totals.get("sampled")
            if totals.get("sampled") else None),
        "generated_tokens_total": totals.get("generated_tokens"),
    }

    if args.skip_contamination:
        contamination = {"status": "skipped"}
    else:
        contamination = rft.contamination_check(
            [{"id": row["id"], "prompt": row["prompt"],
              "solution": row["solution"]} for row in rows])

    kept_rate = overall["acceptance_kept_over_sampled"] or 0.0
    truncated_rate = overall["truncated_over_sampled"] or 0.0
    if args.corpus_kind == "stage2b":
        acceptance_floor = STAGE2B_ACCEPTANCE_FLOOR
        truncation_ceiling = STAGE2B_TRUNCATION_CEILING
        gate_basis = ("stage-2b corpus gate: acceptance floor 0.25 (the "
                      "operator's too-low-to-train line); no truncation "
                      "ceiling is registered because a completion that hits "
                      "the cap without EOS is counted and can never be a "
                      "target; re-verification and the examples hash are the "
                      "training preconditions")
    else:
        acceptance_floor = ACCEPTANCE_FLOOR
        truncation_ceiling = TRUNCATION_CEILING
        gate_basis = ("stage-2a corpus gate: acceptance floor 0.40 and "
                      "truncation ceiling 0.20")
    sampled_total = (totals.get("sampled")
                     or sum((entry.get("sampled") or 0)
                            for entry in per_dataset_acceptance.values()))
    acceptance_measured = bool(sampled_total)
    if args.skip_reverification:
        reverification = {"status": "skipped"}
    else:
        reverification = reverify(rows)
    accepted_rel = _relative(accepted_path)
    accepted_sha = sha256_file(accepted_path)
    examples_block = examples_check(args.examples, args.examples_receipt,
                                     accepted_rel, accepted_sha)
    gates = {
        "corpus_kind": args.corpus_kind,
        "acceptance_floor": acceptance_floor,
        "acceptance_measured": kept_rate if acceptance_measured else None,
        "acceptance_passes": (kept_rate >= acceptance_floor
                              if acceptance_measured else None),
        "truncation_ceiling": truncation_ceiling,
        "truncation_measured": truncated_rate if acceptance_measured else None,
        "truncation_passes": ((truncated_rate < truncation_ceiling)
                              if (acceptance_measured and
                                  truncation_ceiling is not None) else None),
        "acceptance_gate_basis": (
            ("kept/sampled over every sampled completion" if acceptance_measured
             else "not_applicable: this receipt carries no sampled/kept block; "
                  "the generation receipts named on the corpus carry those "
                  "rates")),
        "gate_basis": gate_basis,
        "contamination_fail_closed": contamination.get("fail_closed"),
        "reverification_all_correct": reverification.get("all_correct"),
        "examples_sha256_matches": (examples_block or {}).get("matches"),
        "examples_accepted_input_matches": (
            (examples_block or {}).get("accepted_input_matches")),
        "solution_cap_reached_by_kept": {
            source: entry["solution_tokens"]["fraction_at_cap"]
            for source, entry in sources.items()},
        "reading": ("acceptance is kept/sampled over every sampled completion; "
                    "a completion that hit the cap without EOS is counted "
                    "truncated and can never be a target, so a high kept rate "
                    "beside a high truncated rate would mean the cap is doing "
                    "the selection. The re-verification and examples-hash "
                    "clauses are the S60 training preconditions."),
    }
    gates["corpus_usable"] = bool(
        gates["acceptance_passes"] is not False
        and gates["truncation_passes"] is not False
        and not contamination.get("fail_closed", False)
        and not duplicates
        and reverification.get("all_correct") is not False
        and (examples_block is None or
             (examples_block.get("matches") is not False
              and examples_block.get("accepted_input_matches") is not False)))

    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "receipt": {"path": os.path.relpath(receipt_path, ROOT),
                    "sha256": sha256_file(receipt_path)},
        "acceptance_reconstructed": (
            acceptance.get("reconstructed_from") if
            isinstance(acceptance.get("reconstructed_from"), list) else None),
        "accepted": {"path": os.path.relpath(accepted_path, ROOT),
                     "sha256": sha256_file(accepted_path)},
        "generation": receipt.get("generation"),
        "sampling": receipt.get("sampling"),
        "overall": overall,
        "sources": sources,
        "contamination": contamination,
        "reverification": reverification,
        "examples": examples_block,
        "gates": gates,
        "command": sys.argv,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out_path)

    print(f"s57-verify: {len(rows)} rows from {len(sources)} sources, "
          f"{overall['solution_tokens_accepted']} accepted solution tokens, "
          f"{overall['prompt_tokens_accepted']} prompt tokens")
    for source, entry in sources.items():
        sampling = entry["sampling"]
        lengths = entry["solution_tokens"]
        kept_over = sampling["acceptance_kept_over_sampled"]
        truncated_over = sampling["truncated_over_sampled"]
        kept_text = "n/a" if kept_over is None else f"{kept_over:.3f}"
        truncated_text = ("n/a" if truncated_over is None
                          else f"{truncated_over:.3f}")
        print(f"s57-verify: {source} rows={entry['rows']} "
              f"prompts={entry['distinct_prompts']} "
              f"kept/sampled={kept_text} "
              f"truncated={truncated_text} "
              f"mean_len={lengths['mean']:.0f} p90={lengths['p90']:.0f} "
              f"max={lengths['max']} at_cap={lengths['fraction_at_cap']:.3f}")
    print(f"s57-verify: contamination max_jaccard="
          f"{contamination.get('max_jaccard')} fail_closed="
          f"{contamination.get('fail_closed')}")
    if reverification.get("status") != "skipped":
        print(f"s57-verify: re-verification {reverification['verdicts']} "
              f"all_correct={reverification['all_correct']}")
    if examples_block is not None:
        print(f"s57-verify: examples sha256 {examples_block['sha256'][:16]}… "
              f"receipt_match={examples_block['matches']} "
              f"accepted_input_match={examples_block['accepted_input_matches']}")
    print(f"s57-verify: corpus_usable={gates['corpus_usable']} -> {args.out}")
    return 0 if gates["corpus_usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
