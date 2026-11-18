#!/usr/bin/env python3
"""Termination and looping signals for generated answers (CPU only).

A generation has three observable ends:

* **terminated** — the model emitted EOS before the token cap;
* **truncated** — the harness hit the cap, so the answer is cut by the harness,
  not by the model;
* **looping** — the tail of the answer is degenerate: it compresses far better
  than the body, and/or the same n-grams repeat, and/or the final characters are
  an exact repetition of one period.

Truncation and looping are different failures and the receipts report both. A
long but useful derivation can be truncated without looping; a short answer can
loop without reaching the cap. The detector therefore never decides "truncated"
from text, and never decides "loop" from length.

The two cheap signals are:

``tail compression``
    ``zlib`` level-9 compressed bytes per byte, computed for the whole answer
    and for its final ``tail_chars`` characters. A loop compresses its tail much
    more than its body, so ``tail_ratio / full_ratio`` falls. Comparing the two
    ratios (rather than an absolute tail ratio) makes the signal insensitive to
    how compressible the subject's notation is.
``repetition``
    the fraction of word n-grams in the tail that are duplicates of an earlier
    n-gram, plus an exact-period test on the final characters.

This is a **textual** loop detector. It cannot tell a legitimate repeated
operation (a sum written out term by term) from a degenerate cycle; the
thresholds are calibrated on the stored stage-1 generations and the receipt
publishes the false-positive rate on finished answers beside the detection rate
on truncated ones. No decoding parameter is changed to make the number look
better.

The module is importable without ``torch``/``transformers``.
"""

from __future__ import annotations

import math
import re
import zlib

TAIL_CHARS = 2000
MIN_CHARS = 400

#: Calibrated by `scripts/s56_termination_calibration.py` on the stored stage-1
#: generations; see `results/stage1/s56-termination-calibration.json` for the
#: distributions and the chosen operating point. Changing a value here without
#: regenerating that receipt is a defect.
DEFAULT_THRESHOLDS = {
    "tail_over_full_max": 0.78,
    "ngram_repeat_min": 0.20,
    "ngram_n": 6,
    "period_repeats_min": 3,
    "period_chars_max": 160,
}

_WORD_RE = re.compile(r"\S+")


def _ratio(data: bytes) -> float:
    if not data:
        return 0.0
    return len(zlib.compress(data, 9)) / len(data)


def compression_profile(text: str, tail_chars: int = TAIL_CHARS) -> dict:
    """Whole-answer and tail compression, as compressed bytes per byte."""
    data = text.encode("utf-8")
    if len(data) < MIN_CHARS:
        return {
            "measurable": False,
            "chars": len(text),
            "tail_chars": 0,
            "full_ratio": None,
            "tail_ratio": None,
            "tail_over_full": None,
        }
    tail_text = text[-tail_chars:] if tail_chars > 0 else text
    tail = tail_text.encode("utf-8")
    full_ratio = _ratio(data)
    tail_ratio = _ratio(tail)
    return {
        "measurable": True,
        "chars": len(text),
        "tail_chars": len(tail_text),
        "full_ratio": full_ratio,
        "tail_ratio": tail_ratio,
        "tail_over_full": (tail_ratio / full_ratio) if full_ratio else None,
    }


def repetition_profile(text: str, n: int = 6,
                       tail_chars: int = TAIL_CHARS) -> dict:
    """Duplicate word-n-gram rate in the tail and the exact repeating period."""
    window = text[-tail_chars:] if tail_chars > 0 else text
    words = _WORD_RE.findall(window)
    if len(words) < n + 1:
        rate, grams = 0.0, 0
    else:
        observed = [tuple(words[index:index + n])
                    for index in range(len(words) - n + 1)]
        grams = len(observed)
        rate = 1.0 - (len(set(observed)) / grams) if grams else 0.0
    return {
        "ngram_n": n,
        "ngram_count": grams,
        "ngram_repeat_rate": rate,
        "period": repeated_period(text),
    }


def repeated_period(text: str, max_period_chars: int = 160,
                    min_repeats: int = 3) -> dict | None:
    """The smallest exact period repeated at least `min_repeats` times at the end.

    Whitespace is preserved: `"abcabcabc"` is a period 3, while a period that
    only matches after normalisation is not reported. The test is exact, so a
    loop whose numbers change is caught by compression/repetition instead.
    """
    if max_period_chars <= 0 or min_repeats < 2:
        return None
    limit = min(max_period_chars, len(text) // min_repeats)
    for period in range(8, limit + 1):
        block = text[-period * min_repeats:]
        if len(block) < period * min_repeats:
            continue
        if all(block[index] == block[index % period]
               for index in range(len(block))):
            return {
                "period_chars": period,
                "repeats": min_repeats,
                "block_chars": len(block),
            }
    return None


def classify(text: str, thresholds: dict | None = None) -> dict:
    """Classify one answer's tail as looping or not, with the evidence kept."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    compression = compression_profile(text, tail_chars=TAIL_CHARS)
    repetition = repetition_profile(
        text, n=int(limits["ngram_n"]), tail_chars=TAIL_CHARS)
    result = {
        "loop": False,
        "reason": "too_short" if not compression["measurable"] else "no_loop",
        "compression": compression,
        "repetition": repetition,
        "thresholds": limits,
    }
    if not compression["measurable"]:
        return result
    over = compression["tail_over_full"]
    compressed = over is not None and over <= limits["tail_over_full_max"]
    repetitive = repetition["ngram_repeat_rate"] >= limits["ngram_repeat_min"]
    period = repetition["period"]
    if compressed and repetitive:
        result["loop"] = True
        result["reason"] = "compressed_repetitive_tail"
    elif period is not None:
        # An exact period repeated at least `period_repeats_min` times is itself
        # the evidence. It needs no compression gain: a wholly repetitive answer
        # compresses its body as well as its tail, so the ratio is ~1 even
        # though the tail is degenerate.
        result["loop"] = True
        result["reason"] = "exact_period_tail"
    elif compressed and not repetitive:
        result["reason"] = "compressed_but_not_repetitive"
    elif repetitive and not compressed:
        result["reason"] = "repetitive_but_not_compressed"
    return result


def percentiles(values: list[float], points=(0.0, 0.01, 0.05, 0.25, 0.5,
                                              0.75, 0.95, 0.99, 1.0)) -> dict:
    """Deterministic linear-interpolated percentiles; empty -> all None."""
    if not values:
        return {str(point): None for point in points}
    ordered = sorted(values)
    result = {}
    for point in points:
        position = point * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        result[str(point)] = value
    return result


def _condition(rows: list[dict]) -> dict:
    """Accuracy and tokens over a sample-conditioned subset."""
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    tokens = [row["new_tokens"] for row in rows if row.get("new_tokens") is not None]
    accuracy = correct / total if total else None
    stderr = (math.sqrt(accuracy * (1.0 - accuracy) / total)
              if total and accuracy is not None else None)
    return {
        "samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "stderr": stderr,
        "mean_new_tokens": (sum(tokens) / len(tokens)) if tokens else None,
    }


def summarise(samples: list[dict], thresholds: dict | None = None) -> dict:
    """Termination metrics for one task/mode/decoding block.

    Each sample is a mapping with `completion`, `new_tokens`, `truncated`,
    `correct` and optionally `problem_id` and `problem_correct`. The
    `correct` field is the same verdict the eval receipt scores with, so the
    conditional accuracy never introduces a different metric.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    classified = []
    for sample in samples:
        signal = classify(sample["completion"], limits)
        classified.append({**sample, "loop": signal["loop"],
                           "loop_reason": signal["reason"],
                           "terminated": not bool(sample["truncated"]),
                           "signal": signal})
    total = len(classified)
    terminated = [row for row in classified if row["terminated"]]
    truncated = [row for row in classified if not row["terminated"]]
    looping = [row for row in classified if row["loop"]]
    clean = [row for row in classified if not row["loop"]]
    new_tokens = [row["new_tokens"] for row in classified
                  if row.get("new_tokens") is not None]

    # Problem-level conditioning: a problem is `all_terminated` only when every
    # sample it contributed terminated; its `problem_correct` is the receipt's
    # own scored verdict (greedy: flexible; maj@16: the flexible vote).
    problems: dict[str, dict] = {}
    for row in classified:
        key = row.get("problem_id")
        if key is None or row.get("problem_correct") is None:
            continue
        entry = problems.setdefault(key, {"all_terminated": True,
                                          "any_truncated": False,
                                          "correct": bool(row["problem_correct"])})
        entry["all_terminated"] = entry["all_terminated"] and row["terminated"]
        entry["any_truncated"] = entry["any_truncated"] or not row["terminated"]
    problem_rows = list(problems.values())

    def problem_condition(rows: list[dict]) -> dict:
        correct = sum(1 for row in rows if row["correct"])
        return {"problems": len(rows), "correct": correct,
                "accuracy": (correct / len(rows)) if rows else None}

    return {
        "detector": {
            **limits,
            "tail_chars": TAIL_CHARS,
            "min_chars": MIN_CHARS,
            "kind": "zlib_tail_compression_plus_word_ngram_repetition",
            "decoding_unchanged": True,
        },
        "samples": total,
        "truncated_samples": len(truncated),
        "truncation_rate": (len(truncated) / total) if total else None,
        "loop_classified_samples": len(looping),
        "loop_classified_fraction": (len(looping) / total) if total else None,
        "loop_reasons": {
            reason: sum(1 for row in classified if row["loop_reason"] == reason)
            for reason in sorted({row["loop_reason"] for row in classified})
        },
        "mean_new_tokens": {
            "all": (sum(new_tokens) / len(new_tokens)) if new_tokens else None,
            "terminated": _condition(terminated)["mean_new_tokens"],
            "truncated": _condition(truncated)["mean_new_tokens"],
            "loop": _condition(looping)["mean_new_tokens"],
            "clean": _condition(clean)["mean_new_tokens"],
        },
        "sample_level": {
            "all": _condition(classified),
            "terminated": _condition(terminated),
            "truncated": _condition(truncated),
            "loop": _condition(looping),
            "clean": _condition(clean),
        },
        "problem_level": {
            "all_terminated": problem_condition(
                [row for row in problem_rows if row["all_terminated"]]),
            "any_truncated": problem_condition(
                [row for row in problem_rows if row["any_truncated"]]),
        },
        "cross_tab": {
            "terminated_clean": sum(1 for row in classified
                                    if row["terminated"] and not row["loop"]),
            "terminated_loop": sum(1 for row in classified
                                   if row["terminated"] and row["loop"]),
            "truncated_clean": sum(1 for row in classified
                                   if not row["terminated"] and not row["loop"]),
            "truncated_loop": sum(1 for row in classified
                                  if not row["terminated"] and row["loop"]),
        },
    }


__all__ = [
    "TAIL_CHARS", "MIN_CHARS", "DEFAULT_THRESHOLDS", "compression_profile",
    "repetition_profile", "repeated_period", "classify", "percentiles",
    "summarise",
]
