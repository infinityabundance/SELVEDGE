#!/usr/bin/env python3
"""Sequence-level correctness beside next-token CE and teacher KL.

The stage-2 held-out instrument measures mean next-token CE and
`KL(teacher_T || student_T)`. Those are likelihood quantities. A lower mean CE
does **not** imply a higher exact-answer rate: the S45 diagnosis measured a
persistent reasoning-path bias in `memory-2m` that changed the extracted final
answer on 203 of 208 newly-wrong GSM8K items while the next-token statistics
moved by far less, and the S44 arm table has arms whose held-out CE is
indistinguishable from base while their math family mean is 3–4 points lower.

This module adds the missing measurement: take a generated completion, extract
its final answer with the S45 verifier (exact numeric or canonical expression),
and report the verdict beside the likelihood metrics. It is deliberately not a
symbolic verifier; the false-reject tradeoff is the same one `frontier.rft`
documents.

CPU-only. No generation happens here; the caller supplies completions.
"""

from __future__ import annotations

import math
from collections import Counter

from . import rft

#: Tasks whose final answer can be extracted by the registered verifier.
VERIFIABLE_TASKS = frozenset({"gsm8k", "math", "minerva_math500"})
VERDICTS = ("correct", "wrong", "ambiguous", "unparseable")


def extractor_dataset(dataset: str) -> str:
    """Map a harness task name to the S45 extractor's dataset name."""
    if dataset == "gsm8k":
        return "gsm8k"
    if dataset in ("math", "minerva_math500"):
        return "math"
    raise ValueError(f"{dataset!r} has no registered answer extractor")


def reference_answer(target: str, dataset: str) -> str:
    """Use the reference answer inside `target`, or the stripped target itself.

    Harness targets are sometimes the bare answer and sometimes a full solution
    ending in `####` or `\\boxed{}`. Extracting first keeps both forms on the
    same basis; if extraction cannot find an answer the stripped text is used,
    which is the harness's own target string.
    """
    extraction = rft.extract(target, extractor_dataset(dataset))
    if extraction.value is not None and not extraction.ambiguous:
        return extraction.value
    return target.strip()


def verify_completion(completion: str, reference: str, dataset: str) -> dict:
    """One sequence-level verdict under the registered exact extractor."""
    verdict = rft.verify(completion, reference, extractor_dataset(dataset))
    return {
        "verdict": verdict["verdict"],
        "extracted": verdict.get("extracted"),
        "reference": reference,
        "method": verdict.get("method"),
        "reason": verdict.get("reason"),
    }


def summarise(records: list[dict]) -> dict:
    """Count verdicts and report accuracy beside optional CE/KL.

    `records` items contain `verdict` and optionally `ce` and `kl`. Accuracy is
    reported over all items and over answered items (`correct + wrong`), because
    an ambiguity or an unparseable completion is a distinct failure and hiding it
    in one denominator would overstate the answer rate. The standard error is the
    binomial standard error on the all-items denominator.
    """
    total = len(records)
    counts = Counter(record["verdict"] for record in records)
    for verdict in VERDICTS:
        counts.setdefault(verdict, 0)
    correct = counts["correct"]
    answered = counts["correct"] + counts["wrong"]
    accuracy = correct / total if total else None
    stderr = (math.sqrt(accuracy * (1.0 - accuracy) / total)
              if total and accuracy is not None else None)
    summary = {
        "items": total,
        "correct": correct,
        "wrong": counts["wrong"],
        "ambiguous": counts["ambiguous"],
        "unparseable": counts["unparseable"],
        "answered": answered,
        "accuracy": accuracy,
        "accuracy_answered": correct / answered if answered else None,
        "accuracy_stderr": stderr,
        "verdict_counts": dict(counts),
        "reading": (
            "accuracy is over every item; accuracy_answered excludes ambiguous "
            "and unparseable completions. CE/KL are likelihood means and are "
            "reported beside this, never substituted for it."),
    }
    for metric in ("ce", "kl"):
        values = [float(record[metric]) for record in records
                  if record.get(metric) is not None]
        summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
        summary[f"{metric}_items"] = len(values)
    return summary


def sequence_correctness(items: list[dict], dataset: str) -> dict:
    """Verify a batch of `{completion, reference}` (or `{completion, target}`).

    Each item may also carry `ce` and `kl`; they are summarised beside the
    correctness counts so one receipt reports both. An item missing a completion
    is `unparseable` rather than silently dropped.
    """
    records = []
    for index, item in enumerate(items):
        completion = item.get("completion")
        reference = item.get("reference")
        if reference is None and item.get("target") is not None:
            reference = reference_answer(str(item["target"]), dataset)
        if not isinstance(completion, str) or reference is None:
            records.append({"verdict": "unparseable", "index": index})
            continue
        verdict = verify_completion(completion, str(reference), dataset)
        records.append({
            "index": index,
            "verdict": verdict["verdict"],
            "extracted": verdict["extracted"],
            "ce": item.get("ce"),
            "kl": item.get("kl"),
        })
    summary = summarise(records)
    summary["dataset"] = dataset
    summary["items_detail"] = records
    return summary


def combined_report(sequence: dict, heldout: dict) -> dict:
    """Report exact-answer correctness and likelihood side by side.

    `heldout` is the companion receipt's CE/KL block; `sequence` is this
    module's output. Neither is substituted for the other: the result keeps
    both objects and the measured reason they can disagree.
    """
    return {
        "sequence_correctness": sequence,
        "heldout_likelihood": heldout,
        "both_reported": True,
        "why_both": (
            "next-token CE/KL and exact final-answer accuracy are different "
            "measurements; the S45 collapse diagnosis measured a reasoning-path "
            "bias that changed the extracted answer on most newly-wrong GSM8K "
            "items, and the S44 arm table has arms whose held-out CE is "
            "indistinguishable from base while their math family mean is lower"),
    }


def harness_samples(samples: list[dict], dataset: str) -> dict:
    """Verify lm-evaluation-harness samples without trusting its accuracy filter.

    lm-eval stores generated continuations in `resps` (a list per sample, each a
    list of strings) and the gold target in `target`. This function re-extracts
    the final answer and reports its own verdict, so the harness's
    `exact_match`/`math_verify` filter and the registered extractor are both
    visible. The two need not agree; a disagreement is a measurement, not an
    error to hide.
    """
    items = []
    for sample in samples:
        responses = sample.get("resps") or []
        completion = ""
        if responses and isinstance(responses[0], (list, tuple)) and responses[0]:
            completion = str(responses[0][-1])
        elif responses and isinstance(responses[0], str):
            completion = responses[0]
        target = sample.get("target")
        reference = None
        if isinstance(target, str):
            reference = reference_answer(target, dataset)
        items.append({"completion": completion, "reference": reference})
    summary = sequence_correctness(items, dataset)
    summary["harness_samples"] = len(samples)
    summary["note"] = (
        "this verdict is the S45 exact extractor applied to the harness's own "
        "generations; it is reported beside, not instead of, the harness metric")
    return summary
