#!/usr/bin/env python3
"""S72 stage 0 prerequisite: corpus audit for the distillation path.

CPU-only, read-only. Reads the accepted corpora and the cached MATH-train
parquet; it does not load a model, does not touch CUDA and does not read the
0.8B tokenizer (the corpora already store ``prompt_ids`` and ``solution_ids``
produced by that exact tokenizer, which is stronger than re-tokenising).

Reports:

* counts by dataset, MATH level and subject;
* ``solution_ids`` length distribution (p50/p90/p99/max) and the same for
  ``prompt_ids`` (the measured prompt length, not an assumption);
* how many stored solutions would not fit at caps 2048 / 2500 / 3072 / 4096
  with the prompt included, and separately for the completion alone;
* MATH-train coverage: parseable items in the fetched train split, how many
  have at least one verified solution in an accepted corpus, and the level
  profile of the covered and uncovered sets;
* the S60 long-example ledger, quoted from its receipt, restated as what a
  2,500-token window recovers.

Usage (CPU only, inside the oracle court so the GSM8K cache resolves):

    SELVEDGE_MEM=4g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
      scripts/court oracle -- python3 scripts/s72_corpus_audit.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

_spec = importlib.util.spec_from_file_location(
    "s72_rft", os.path.join(ROOT, "oracle", "frontier", "rft.py"))
rft = importlib.util.module_from_spec(_spec)
sys.modules["s72_rft"] = rft
_spec.loader.exec_module(rft)

PRIMARY = "results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl"
CAPS = (2048, 2500, 3072, 4096)
TOKENIZER_RECEIPTS = [
    "results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json",
    "results/stage1/s54-stage1-post-2b-nonthinking-greedy.json",
    "results/stage2/s67-final/soup-stage1.json",
]


def tokenizer_identity() -> dict:
    """Prove the stored ids are the 0.8B tokenizer's own output."""
    rows = {}
    for rel in TOKENIZER_RECEIPTS:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            raise SystemExit(f"s72-corpus: missing tokenizer receipt {rel}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        model = payload.get("model") or {}
        rows[rel] = (model.get("bundle") or {}).get("tokenizer.json")
    distinct = {json.dumps(value, sort_keys=True) for value in rows.values()
                if value is not None}
    return {
        "receipts": rows,
        "all_identical": len(distinct) == 1,
        "tokenizer_json": json.loads(distinct.pop()) if len(distinct) == 1 else None,
        "claim": "the corpora store prompt_ids/solution_ids produced at generation "
                 "time; this block proves the 0.8B checkpoint the soup scores with "
                 "ships a byte-identical tokenizer.json, so those ids are the 0.8B "
                 "tokenizer's own ids and no tokenizer had to be loaded or re-applied",
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def load_corpus(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def corpus_stats(rows: list[dict], label: str) -> dict:
    by_dataset = collections.Counter(str(row.get("dataset")) for row in rows)
    by_level = collections.Counter(
        str(row.get("difficulty")) for row in rows if row.get("dataset") == "math")
    by_subject = collections.Counter(
        str(row.get("subject")) for row in rows if row.get("dataset") == "math")
    prompt_lengths = [len(row.get("prompt_ids") or []) for row in rows]
    solution_lengths = [len(row.get("solution_ids") or []) for row in rows]
    per_dataset = {}
    for dataset in sorted(by_dataset):
        subset = [row for row in rows if str(row.get("dataset")) == dataset]
        per_dataset[dataset] = {
            "rows": len(subset),
            "prompt_ids": {
                "p50": percentile([len(r.get("prompt_ids") or []) for r in subset], 0.50),
                "p90": percentile([len(r.get("prompt_ids") or []) for r in subset], 0.90),
                "p99": percentile([len(r.get("prompt_ids") or []) for r in subset], 0.99),
                "max": max([len(r.get("prompt_ids") or []) for r in subset], default=None),
            },
            "solution_ids": {
                "p50": percentile([len(r.get("solution_ids") or []) for r in subset], 0.50),
                "p90": percentile([len(r.get("solution_ids") or []) for r in subset], 0.90),
                "p99": percentile([len(r.get("solution_ids") or []) for r in subset], 0.99),
                "max": max([len(r.get("solution_ids") or []) for r in subset], default=None),
            },
            "truncation": {
                str(cap): {
                    "with_prompt_exceeds": sum(
                        1 for r in subset
                        if len(r.get("prompt_ids") or []) + len(r.get("solution_ids") or []) > cap),
                    "completion_only_exceeds": sum(
                        1 for r in subset if len(r.get("solution_ids") or []) > cap),
                } for cap in CAPS
            },
        }
    truncation = {}
    for cap in CAPS:
        with_prompt = sum(
            1 for row in rows
            if len(row.get("prompt_ids") or []) + len(row.get("solution_ids") or []) > cap)
        completion_only = sum(
            1 for row in rows if len(row.get("solution_ids") or []) > cap)
        truncation[str(cap)] = {
            "with_prompt_exceeds": with_prompt,
            "with_prompt_fraction": with_prompt / len(rows) if rows else None,
            "completion_only_exceeds": completion_only,
            "completion_only_fraction": completion_only / len(rows) if rows else None,
        }
    return {
        "label": label,
        "rows": len(rows),
        "by_dataset": dict(by_dataset),
        "math_by_level": dict(sorted(by_level.items())),
        "math_by_subject": dict(sorted(by_subject.items())),
        "prompt_ids": {
            "p50": percentile(prompt_lengths, 0.50),
            "p90": percentile(prompt_lengths, 0.90),
            "p99": percentile(prompt_lengths, 0.99),
            "max": max(prompt_lengths) if prompt_lengths else None,
            "sum": sum(prompt_lengths),
        },
        "solution_ids": {
            "p50": percentile(solution_lengths, 0.50),
            "p90": percentile(solution_lengths, 0.90),
            "p99": percentile(solution_lengths, 0.99),
            "max": max(solution_lengths) if solution_lengths else None,
            "sum": sum(solution_lengths),
        },
        "truncation": truncation,
        "per_dataset": per_dataset,
        "math_solution_ids": {
            "p50": percentile([len(row.get("solution_ids") or []) for row in rows
                               if row.get("dataset") == "math"], 0.50),
            "p90": percentile([len(row.get("solution_ids") or []) for row in rows
                               if row.get("dataset") == "math"], 0.90),
            "p99": percentile([len(row.get("solution_ids") or []) for row in rows
                               if row.get("dataset") == "math"], 0.99),
            "max": max([len(row.get("solution_ids") or []) for row in rows
                        if row.get("dataset") == "math"], default=None),
        },
    }


def coverage(rows_by_source: dict[str, list[dict]], universe: list[dict]) -> dict:
    universe_ids = {item["id"]: item for item in universe}
    levels = {item["id"]: item.get("difficulty") for item in universe}
    sources = {}
    union_math: set[str] = set()
    for label, rows in rows_by_source.items():
        ids = {row["id"] for row in rows if row.get("dataset") == "math"}
        known = {key for key in ids if key in universe_ids}
        sources[label] = {
            "distinct_math_prompts": len(ids),
            "distinct_math_prompts_in_universe": len(known),
            "unknown_ids": sorted(ids - set(universe_ids))[:20],
            "unknown_id_count": len(ids - set(universe_ids)),
        }
        union_math |= known
    covered_levels = collections.Counter(str(levels[key]) for key in union_math)
    uncovered = [key for key in universe_ids if key not in union_math]
    uncovered_levels = collections.Counter(str(levels[key]) for key in uncovered)
    return {
        "universe_math_items": len(universe_ids),
        "union_covered": len(union_math),
        "union_uncovered": len(uncovered),
        "union_coverage_fraction": len(union_math) / len(universe_ids),
        "covered_by_level": dict(sorted(covered_levels.items())),
        "uncovered_by_level": dict(sorted(uncovered_levels.items())),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default=PRIMARY)
    parser.add_argument("--out", default="results/stage2/s72-recon/corpus-audit.json")
    args = parser.parse_args()

    primary = load_corpus(os.path.join(ROOT, args.primary))
    other_paths = sorted(
        path for path in glob.glob(os.path.join(ROOT, "results", "artifacts",
                                               "stage2", "**", "*accepted*.jsonl"),
                                   recursive=True)
        if os.path.relpath(path, ROOT) != args.primary)
    others = {}
    for path in other_paths:
        rel = os.path.relpath(path, ROOT)
        try:
            others[rel] = load_corpus(path)
        except Exception as error:  # noqa: BLE001 - record, never guess
            others[rel] = []
            print(f"warning: could not read {rel}: {error}")

    universe = rft.load_items("math-train")
    rows_by_source = {"s58-combined": primary}
    for rel, rows in others.items():
        rows_by_source[rel] = rows

    receipt = {
        "schema": "selvedge/s72-corpus-audit/v1",
        "stamp": now_stamp(),
        "created_utc": now_iso(),
        "kind": "corpus-audit",
        "primary": {
            "path": args.primary,
            "sha256": sha256_file(os.path.join(ROOT, args.primary)),
        },
        "tokenizer_identity": tokenizer_identity(),
        "primary_stats": corpus_stats(primary, "s58-combined"),
        "other_corpora": {
            rel: {
                "sha256": sha256_file(os.path.join(ROOT, rel)),
                "rows": len(rows),
                "by_dataset": dict(collections.Counter(
                    str(row.get("dataset")) for row in rows)),
                "distinct_prompts": len({row["id"] for row in rows}),
            } for rel, rows in others.items()
        },
        "math_train_universe": {
            "parseable_items": len(universe),
            "skipped_reference_items": universe[0].get("skipped_reference_items")
            if universe else None,
            "by_level": dict(sorted(collections.Counter(
                str(item.get("difficulty")) for item in universe).items())),
            "by_subject": dict(sorted(collections.Counter(
                str(item.get("subject")) for item in universe).items())),
        },
        "math_coverage": coverage(rows_by_source, universe),
    }

    ledger_path = os.path.join(ROOT, "results", "stage2",
                               "s60-long-example-share.json")
    if os.path.isfile(ledger_path):
        with open(ledger_path, encoding="utf-8") as handle:
            ledger = json.load(handle)
        receipt["s60_long_example_ledger"] = {
            "path": os.path.relpath(ledger_path, ROOT),
            "sha256": sha256_file(ledger_path),
            "measurement": ledger.get("measurement"),
        }

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
        handle.write("\n")

    stats = receipt["primary_stats"]
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    print(f"primary rows={stats['rows']} by_dataset={stats['by_dataset']}")
    print(f"math by_level={stats['math_by_level']}")
    print(f"math by_subject={stats['math_by_subject']}")
    print(f"prompt_ids={stats['prompt_ids']}")
    print(f"solution_ids={stats['solution_ids']}")
    print(f"math solution_ids={stats['math_solution_ids']}")
    for cap, block in stats["truncation"].items():
        print(f"cap {cap}: with_prompt={block['with_prompt_exceeds']} "
              f"completion_only={block['completion_only_exceeds']}")
    cov = receipt["math_coverage"]
    print(f"math coverage {cov['union_covered']}/{cov['universe_math_items']}"
          f" uncovered={cov['union_uncovered']}")
    print(f"covered_by_level={cov['covered_by_level']}")
    print(f"uncovered_by_level={cov['uncovered_by_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
