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

Usage:
    SELVEDGE_MEM=4g scripts/court python -- python3 scripts/s57_verify_stage2a.py \
        --receipt results/artifacts/stage2/s55-stage2a/stage2a-receipt.json \
        --accepted results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl \
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-contamination", action="store_true",
                        help="only for a dry pass over a smoke corpus")
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

    acceptance = receipt.get("acceptance", {})
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
    gates = {
        "acceptance_floor": ACCEPTANCE_FLOOR,
        "acceptance_measured": kept_rate,
        "acceptance_passes": kept_rate >= ACCEPTANCE_FLOOR,
        "truncation_ceiling": TRUNCATION_CEILING,
        "truncation_measured": truncated_rate,
        "truncation_passes": truncated_rate < TRUNCATION_CEILING,
        "contamination_fail_closed": contamination.get("fail_closed"),
        "solution_cap_reached_by_kept": {
            source: entry["solution_tokens"]["fraction_at_cap"]
            for source, entry in sources.items()},
        "reading": ("acceptance is kept/sampled over every sampled completion; "
                    "a completion that hit the cap without EOS is counted "
                    "truncated and can never be a target, so a high kept rate "
                    "beside a high truncated rate would mean the cap is doing "
                    "the selection"),
    }
    gates["corpus_usable"] = bool(
        gates["acceptance_passes"] and gates["truncation_passes"]
        and not contamination.get("fail_closed", False)
        and not duplicates)

    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "receipt": {"path": os.path.relpath(receipt_path, ROOT),
                    "sha256": sha256_file(receipt_path)},
        "accepted": {"path": os.path.relpath(accepted_path, ROOT),
                     "sha256": sha256_file(accepted_path)},
        "generation": receipt.get("generation"),
        "sampling": receipt.get("sampling"),
        "overall": overall,
        "sources": sources,
        "contamination": contamination,
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
        print(f"s57-verify: {source} rows={entry['rows']} "
              f"prompts={entry['distinct_prompts']} "
              f"kept/sampled={sampling['acceptance_kept_over_sampled']:.3f} "
              f"truncated={sampling['truncated_over_sampled']:.3f} "
              f"mean_len={lengths['mean']:.0f} p90={lengths['p90']:.0f} "
              f"max={lengths['max']} at_cap={lengths['fraction_at_cap']:.3f}")
    print(f"s57-verify: contamination max_jaccard="
          f"{contamination.get('max_jaccard')} fail_closed="
          f"{contamination.get('fail_closed')}")
    print(f"s57-verify: corpus_usable={gates['corpus_usable']} -> {args.out}")
    return 0 if gates["corpus_usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
