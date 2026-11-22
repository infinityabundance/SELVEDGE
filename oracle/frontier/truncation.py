#!/usr/bin/env python3
"""S73: four-way classification of a stored MATH completion (CPU only).

The S72 recon found that 158 of our S67 soup's 500 MATH-500 completions hit the
4,096-token cap, while the model we must beat (Qwen3-0.6B) hit it 18 times and
the 2B teacher 115 times. Before spending GPU time on a cap ladder, the truncated
rows need to be described: did the model never produce an answer, did it produce
one and keep talking, or did it degenerate into a loop?

The partition is fixed before any ladder run and is *exclusive* by a declared
precedence:

1. ``loop_degenerate``
    the S56 loop detector fires on the text (``termination.classify`` with its
    calibrated operating point). Degeneration is the mechanism, so it is named
    first whether or not a box appeared; a longer cap is the least likely fix.
    This class is unreachable last, because a looping tail is long enough to
    pre-empt the "box then continued" test whenever a box is present.
2. ``no_box``
    no balanced ``\\boxed{}`` answer anywhere in the completion and no loop. A
    ``\\boxed{`` cut off by the cap (no closing brace) counts as no box, because
    no answer was emitted.
3. ``boxed_then_continued``
    a balanced boxed answer exists **and** at least
    ``BOX_THEN_CONTINUED_MIN_NONSPACE`` non-whitespace characters follow the
    closing brace of the *last* box, and the text is not degenerate. This is the
    "it had an answer and did not stop" class: the terminal behaviour is the
    candidate fault.
4. ``other``
    everything that remains (a box with nothing meaningful after it).

Precedence is ``loop_degenerate`` -> ``no_box`` -> ``boxed_then_continued`` ->
``other``. A row in class 1 that also carries a box keeps ``boxed_present`` and
``correct_box_anywhere`` true, so that overlap is reported rather than hidden.

Extraction is **not** reimplemented here. Boxes come from ``rft._boxed_answers``
and answer equality from ``rft.answers_equal``; the flexible verdict is
``rft.extract``. The strict wrapper mirrors ``extract_strict`` in
``scripts/s54_post_trained_eval.py`` (the scorer the receipts use) using only
those primitives, and ``oracle/tests/test_s73_truncation.py`` pins the two to
agreement on a synthetic battery so the mirror cannot drift unnoticed.

"Verified-correct answer anywhere before the cap" means: at least one balanced
boxed answer in the stored text equals the gold answer under
``rft.answers_equal``. That is a property of the text that survived the cap, so
it is measured, not inferred from a longer generation.

Deterministic and importable; no torch, no network, no model load.
"""

from __future__ import annotations

import re

from . import rft, termination

CLASSIFIER_VERSION = "s73-truncation-v1"

#: Non-whitespace characters that must follow the last box's closing brace for
#: the completion to count as "answer emitted, generation continued". Declared
#: here and hashed into the preregistration; changing it is a new registration.
BOX_THEN_CONTINUED_MIN_NONSPACE = 20

#: Exclusive classes, in declared precedence order.
CLASSES = ("no_box", "boxed_then_continued", "loop_degenerate", "other")

_BOX_START = re.compile(r"\\boxed\s*\{")


def box_spans(text: str) -> list[tuple[str, int, int]]:
    """Balanced boxed answers as ``(content, open_brace, close_brace)``.

    Iteration mirrors ``rft._boxed_answers`` exactly so the contents agree with
    the registered extractor; the indices are what this module adds.
    """
    spans: list[tuple[str, int, int]] = []
    if not isinstance(text, str):
        return spans
    for match in _BOX_START.finditer(text):
        brace = text.index("{", match.start())
        parsed = rft._balanced(text, brace)
        if parsed is not None:
            content, close = parsed
            spans.append((content, brace, close))
    return spans


def _distinct(values: list[str]) -> list[str]:
    distinct: list[str] = []
    for candidate in values:
        if not any(rft.answers_equal(candidate, other) for other in distinct):
            distinct.append(candidate)
    return distinct


def strict_extraction(text: str) -> dict:
    """The registered strict basis: the last balanced box, or why not.

    Same decision sequence as ``extract_strict`` in
    ``scripts/s54_post_trained_eval.py``; every primitive is ``rft``'s.
    """
    answers = rft._boxed_answers(text) if isinstance(text, str) else []
    if not answers:
        return {"value": None, "method": "none", "status": "unparseable",
                "reason": "no \\boxed{} answer"}
    distinct = _distinct(answers)
    if len(distinct) > 1:
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous",
                "reason": f"{len(distinct)} distinct boxed answers"}
    if len(answers) == 1 and rft._OR.search(answers[0]):
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous",
                "reason": "boxed alternative ('or')"}
    return {"value": answers[-1], "method": "last \\boxed{}", "status": "ok",
            "reason": "single boxed answer"}


def flexible_extraction(text: str) -> dict:
    """The registered flexible basis: strict, else the S45 fallback chain."""
    strict = strict_extraction(text)
    if strict["value"] is not None or strict["status"] == "ambiguous":
        return strict
    extraction = rft.extract(text if isinstance(text, str) else "", "math")
    status = ("ambiguous" if extraction.ambiguous
              else "unparseable" if extraction.value is None else "ok")
    return {"value": extraction.value, "method": extraction.method,
            "status": status, "reason": extraction.reason}


def classify_completion(text: str, gold: str | None = None,
                        truncated: bool | None = None,
                        thresholds: dict | None = None) -> dict:
    """One completion -> its class, extraction verdicts and loop evidence."""
    spans = box_spans(text)
    answers = [content for content, _open, _close in spans]
    boxed_present = bool(spans)

    tail_nonspace = None
    if spans:
        tail = text[spans[-1][2] + 1:]
        tail_nonspace = sum(1 for char in tail if not char.isspace())

    loop_signal = termination.classify(text if isinstance(text, str) else "",
                                       thresholds)

    if loop_signal["loop"]:
        klass = "loop_degenerate"
    elif not boxed_present:
        klass = "no_box"
    elif tail_nonspace >= BOX_THEN_CONTINUED_MIN_NONSPACE:
        klass = "boxed_then_continued"
    else:
        klass = "other"

    strict = strict_extraction(text)
    flexible = flexible_extraction(text)
    correct_box_anywhere = None
    if gold is not None:
        correct_box_anywhere = any(rft.answers_equal(answer, str(gold))
                                   for answer in answers)
        strict_correct = (strict["value"] is not None
                          and rft.answers_equal(strict["value"], str(gold)))
        flexible_correct = (flexible["value"] is not None
                            and rft.answers_equal(flexible["value"], str(gold)))
    else:
        strict_correct = None
        flexible_correct = None

    return {
        "class": klass,
        "classifier_version": CLASSIFIER_VERSION,
        "boxed_present": boxed_present,
        "boxed_count": len(answers),
        "boxed_answers": answers,
        "tail_after_last_box_nonspace": tail_nonspace,
        "box_then_continued_min_nonspace": BOX_THEN_CONTINUED_MIN_NONSPACE,
        "loop": bool(loop_signal["loop"]),
        "loop_reason": loop_signal["reason"],
        "strict_value": strict["value"],
        "strict_status": strict["status"],
        "strict_correct": strict_correct,
        "flexible_value": flexible["value"],
        "flexible_status": flexible["status"],
        "flexible_correct": flexible_correct,
        "correct_box_anywhere": correct_box_anywhere,
        "truncated": None if truncated is None else bool(truncated),
    }


def _counts(rows: list[dict]) -> dict:
    by_class = {name: 0 for name in CLASSES}
    for row in rows:
        by_class[row["class"]] += 1
    return by_class


def summarise(records: list[dict], only_truncated: bool = False) -> dict:
    """Class and correctness counts over a list of classified rows.

    ``records`` are the dictionaries returned by ``classify_completion``.
    ``only_truncated`` keeps rows whose ``truncated`` is true and refuses a
    mixed/unknown set rather than silently counting a missing flag as false.
    """
    if only_truncated:
        unknown = [index for index, row in enumerate(records)
                   if row.get("truncated") is None]
        if unknown:
            raise ValueError(
                "only_truncated was requested but rows "
                f"{unknown[:5]} have no truncated flag")
        selected = [row for row in records if row["truncated"]]
    else:
        selected = list(records)

    correct_box = [row for row in selected
                   if row.get("correct_box_anywhere") is True]
    in_class_correct = {name: 0 for name in CLASSES}
    in_class_boxed = {name: 0 for name in CLASSES}
    for row in selected:
        if row.get("correct_box_anywhere") is True:
            in_class_correct[row["class"]] += 1
        if row["boxed_present"]:
            in_class_boxed[row["class"]] += 1

    return {
        "rows": len(selected),
        "by_class": _counts(selected),
        "boxed_present_rows": sum(1 for row in selected if row["boxed_present"]),
        "correct_box_anywhere_rows": len(correct_box),
        "correct_box_anywhere_by_class": in_class_correct,
        "boxed_present_by_class": in_class_boxed,
        "loop_rows": sum(1 for row in selected if row["loop"]),
        "strict_correct_rows": sum(1 for row in selected
                                   if row.get("strict_correct") is True),
        "flexible_correct_rows": sum(1 for row in selected
                                     if row.get("flexible_correct") is True),
        "reading": (
            "by_class is an exclusive partition in the declared precedence "
            "loop_degenerate > no_box > boxed_then_continued > other; "
            "correct_box_anywhere is any balanced boxed answer equal to gold; "
            "loop_rows and correct_box_anywhere_by_class expose the overlap."),
    }


__all__ = [
    "CLASSIFIER_VERSION",
    "BOX_THEN_CONTINUED_MIN_NONSPACE",
    "CLASSES",
    "box_spans",
    "strict_extraction",
    "flexible_extraction",
    "classify_completion",
    "summarise",
]
