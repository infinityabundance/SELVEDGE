#!/usr/bin/env python3
"""S56: calibrate the termination/loop detector on stored stage-1 generations.

The detector in `oracle/frontier/termination.py` has two cheap signals: how much
better the final 2,000 characters compress than the whole answer, and how many
word n-grams in that tail repeat. Both thresholds are chosen here, not by taste.

Input is the stored raw generation JSONL for the stage-1 receipts. Each row
carries the completion, the harness's own `truncated` flag and the answer
verdict, so the label is the harness's: **truncated** means the model did not
emit EOS within the cap. The detector never sees the cap; it only sees text.

The operating point maximises the detection rate on truncated answers subject to
a false-positive rate of at most 5% on finished answers. The receipt prints the
whole grid frontier, so a reader can choose another point rather than trust this
one. Nothing is written into a run directory and no decoding is changed.

Usage:
    scripts/court python -- python3 scripts/s56_termination_calibration.py
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

from frontier import termination  # noqa: E402

RAW_GLOB = "results/stage1/*-raw/*.jsonl"
OUT = "results/stage1/s56-termination-calibration.json"
MAX_FALSE_POSITIVE_RATE = 0.05
#: A threshold is only a loop claim if the effect is non-trivial: the tail must
#: compress at least 20% better than the whole answer, and at least one in five
#: word 6-grams in the tail must be a duplicate. Without a minimum effect a
#: sweep can drift to a threshold that fires on almost nothing but still wins on
#: a single extra detection.
MIN_COMPRESSION_EFFECT = 0.20
MIN_REPETITION_EFFECT = 0.20


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
            required = ("completion", "new_tokens", "truncated")
            missing = [key for key in required if key not in row]
            if missing:
                raise SystemExit(f"{path}: row lacks {missing}")
            rows.append(row)
    return rows


def _group_key(path: str, row: dict) -> tuple[str, str, str]:
    return (row.get("task") or "unknown",
            row.get("mode") or "unknown",
            row.get("decoding") or "unknown")


def _metrics(row: dict) -> dict:
    signal = termination.classify(row["completion"])
    compression = signal["compression"]
    repetition = signal["repetition"]
    return {
        "task": row.get("task"),
        "mode": row.get("mode"),
        "decoding": row.get("decoding"),
        "truncated": bool(row["truncated"]),
        "correct": bool(row.get("flexible_correct")),
        "new_tokens": int(row["new_tokens"]),
        "tail_over_full": compression["tail_over_full"],
        "ngram_repeat_rate": repetition["ngram_repeat_rate"],
        "period": repetition["period"] is not None,
        "period_chars": (repetition["period"] or {}).get("period_chars"),
        "loop": signal["loop"],
        "loop_reason": signal["reason"],
        "measurable": compression["measurable"],
    }


def _distribution(rows: list[dict]) -> dict:
    measurable = [row for row in rows if row["measurable"]]
    return {
        "samples": len(rows),
        "measurable": len(measurable),
        "tail_over_full": termination.percentiles(
            [row["tail_over_full"] for row in measurable
             if row["tail_over_full"] is not None]),
        "ngram_repeat_rate": termination.percentiles(
            [row["ngram_repeat_rate"] for row in measurable]),
        "new_tokens": termination.percentiles([row["new_tokens"] for row in rows]),
        "exact_period_fraction": (
            sum(1 for row in rows if row["period"]) / len(rows) if rows else None),
        "loop_fraction_at_default": (
            sum(1 for row in rows if row["loop"]) / len(rows) if rows else None),
    }


def _sweep(rows: list[dict]) -> list[dict]:
    """The (tail_over_full_max, ngram_repeat_min) grid on the measured rows."""
    measurable = [row for row in rows if row["measurable"]]
    truncated = [row for row in measurable if row["truncated"]]
    finished = [row for row in measurable if not row["truncated"]]
    frontier = []
    for tail_max in (0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.62, 0.66, 0.72,
                     0.78, 0.84, 0.90):
        for ngram_min in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                          0.50, 0.60, 0.70, 0.80):
            def fires(row: dict) -> bool:
                if (row["tail_over_full"] is not None
                        and row["tail_over_full"] <= tail_max
                        and row["ngram_repeat_rate"] >= ngram_min):
                    return True
                return row["period"]

            detected = sum(1 for row in truncated if fires(row))
            false = sum(1 for row in finished if fires(row))
            tpr = detected / len(truncated) if truncated else None
            fpr = false / len(finished) if finished else None
            frontier.append({
                "tail_over_full_max": tail_max,
                "ngram_repeat_min": ngram_min,
                "detected_truncated": detected,
                "truncated": len(truncated),
                "false_finished": false,
                "finished": len(finished),
                "detection_rate": tpr,
                "false_positive_rate": fpr,
            })
    return frontier


def _choose(frontier: list[dict]) -> dict:
    eligible = [point for point in frontier
                if point["false_positive_rate"] is not None
                and point["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE
                and (1.0 - point["tail_over_full_max"]) >= MIN_COMPRESSION_EFFECT
                and point["ngram_repeat_min"] >= MIN_REPETITION_EFFECT]
    if not eligible:
        raise SystemExit("calibration: no grid point meets the false-positive bound")
    # Maximise detection, then prefer the wider compression threshold (more
    # tolerant of unusual notation) and the higher repetition threshold.
    eligible.sort(key=lambda point: (
        point["detection_rate"] or 0.0,
        point["tail_over_full_max"],
        -point["ngram_repeat_min"],
    ), reverse=True)
    return eligible[0]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default=RAW_GLOB)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max-false-positive-rate", type=float,
                        default=MAX_FALSE_POSITIVE_RATE)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    if not paths:
        raise SystemExit(f"calibration: no raw generations match {args.glob}")
    sources, groups, all_rows = [], {}, []
    for path in paths:
        rows = load_rows(path)
        metrics = [_metrics(row) for row in rows]
        all_rows.extend(metrics)
        rel = os.path.relpath(path, ROOT)
        sources.append({"path": rel, "rows": len(rows),
                        "sha256": sha256_file(path)})
        for row in metrics:
            key = _group_key(path, row)
            groups.setdefault(key, []).append(row)

    measurable = [row for row in all_rows if row["measurable"]]
    truncated = [row for row in measurable if row["truncated"]]
    finished = [row for row in measurable if not row["truncated"]]
    frontier = _sweep(all_rows)
    chosen = _choose(frontier)
    thresholds = {
        "tail_over_full_max": chosen["tail_over_full_max"],
        "ngram_repeat_min": chosen["ngram_repeat_min"],
        "ngram_n": termination.DEFAULT_THRESHOLDS["ngram_n"],
        "period_repeats_min": termination.DEFAULT_THRESHOLDS["period_repeats_min"],
        "period_chars_max": termination.DEFAULT_THRESHOLDS["period_chars_max"],
    }

    def fires(row: dict) -> bool:
        return (
            (row["tail_over_full"] is not None
             and row["tail_over_full"] <= thresholds["tail_over_full_max"]
             and row["ngram_repeat_rate"] >= thresholds["ngram_repeat_min"])
            or row["period"])

    at_chosen_rows = [{**row, "loop": fires(row)} for row in measurable]

    def rates(rows: list[dict]) -> dict:
        def accuracy(subset: list[dict]) -> dict:
            correct = sum(1 for row in subset if row["correct"])
            value = correct / len(subset) if subset else None
            return {"samples": len(subset), "correct": correct,
                    "accuracy": value}
        return {
            "samples": len(rows),
            "loop_fraction": (
                sum(1 for row in rows if row["loop"]) / len(rows) if rows else None),
            "loop_true_positives": sum(1 for row in rows if row["loop"]),
            "truncation_rate": (
                sum(1 for row in rows if row["truncated"]) / len(rows)
                if rows else None),
            "accuracy": {
                "all": accuracy(rows),
                "terminated": accuracy([row for row in rows
                                        if not row["truncated"]]),
                "truncated": accuracy([row for row in rows
                                       if row["truncated"]]),
                "loop": accuracy([row for row in rows if row["loop"]]),
                "clean": accuracy([row for row in rows if not row["loop"]]),
            },
        }

    by_group = []
    for key, rows in sorted(groups.items()):
        group_chosen = [{**row, "loop": fires(row)} for row in rows]
        by_group.append({
            "task": key[0],
            "mode": key[1],
            "decoding": key[2],
            "finished": _distribution(
                [row for row in rows if not row["truncated"]]),
            "truncated": _distribution(
                [row for row in rows if row["truncated"]]),
            "at_chosen": {
                "finished": rates([row for row in group_chosen
                                   if not row["truncated"]]),
                "truncated": rates([row for row in group_chosen
                                    if row["truncated"]]),
            },
        })

    payload = {
        "schema": "selvedge/s56-termination-calibration/v1",
        "kind": "termination-detector-calibration",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "sources": sources,
        "label": ("truncated is the harness's own flag: no EOS within the "
                  "generation cap. The detector only sees text."),
        "operating_point": {
            **thresholds,
            "selection_rule": (
                "maximise detection rate on truncated answers subject to "
                f"false-positive rate <= {args.max_false_positive_rate} on "
                "finished answers, a compression effect of at least "
                f"{MIN_COMPRESSION_EFFECT:.0%} and a repetition rate of at "
                f"least {MIN_REPETITION_EFFECT:.0%}"),
            "measured": {
                "detection_rate_truncated": chosen["detection_rate"],
                "false_positive_rate_finished": chosen["false_positive_rate"],
                "detected_truncated": chosen["detected_truncated"],
                "truncated": chosen["truncated"],
                "false_finished": chosen["false_finished"],
                "finished": chosen["finished"],
            },
        },
        "samples": {
            "total": len(all_rows),
            "measurable": len(measurable),
            "truncated": len(truncated),
            "finished": len(finished),
        },
        "distribution": {
            "finished": _distribution(finished),
            "truncated": _distribution(truncated),
            "all": _distribution(all_rows),
        },
        "at_chosen_operating_point": {
            "all": rates(at_chosen_rows),
            "truncated": rates([row for row in at_chosen_rows if row["truncated"]]),
            "finished": rates([row for row in at_chosen_rows
                               if not row["truncated"]]),
        },
        "by_task_mode_decoding": by_group,
        "frontier": frontier,
        "reading": (
            "These thresholds are a claim about text, not about correctness. "
            "A legitimate repeated operation can trip the period branch; the "
            "false-positive rate is measured on finished stage-1 answers and "
            "printed here. Accuracy conditioned on termination uses each raw "
            "row's own `flexible_correct` (sample level; for maj@16 this is not "
            "the vote), so it is comparable across checkpoints with the same "
            "eval receipt. The eval receipt itself reports the same block per "
            "task, mode and decoding, with the receipt's scored verdict, beside "
            "the unchanged decoding."),
    }
    # Fill the per-group `at_chosen` rates with the same chosen point.
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s56-termination: {len(all_rows)} samples from {len(sources)} files")
    print(f"s56-termination: finished {len(finished)}, truncated {len(truncated)}")
    print(f"s56-termination: chosen tail_over_full_max="
          f"{thresholds['tail_over_full_max']} ngram_repeat_min="
          f"{thresholds['ngram_repeat_min']}")
    print(f"s56-termination: detection {chosen['detection_rate']:.3f} on truncated, "
          f"false positive {chosen['false_positive_rate']:.3f} on finished")
    print(f"s56-termination: {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
