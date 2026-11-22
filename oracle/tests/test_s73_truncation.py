"""S73 — four-way truncation classifier, CPU only.

Every class is exercised by a synthetic case constructed to hit it, including
the two cases the probe's decision rule turns on: a boxed answer followed by a
ramble (class 2) and a repetition loop (class 3). The strict/flexible verdicts
are pinned to the registered S54 scorer so the local mirror cannot drift, and
the registered classifier hash is checked against the preregistration when that
file exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from frontier import rft, termination, truncation  # noqa: E402

REGISTRATION = os.path.join(
    ROOT, "docs", "s73-truncation-ceiling-preregistration-v1.json")

NO_BOX = (
    "Let r = \\sqrt{x^2 + y^2}. With x = 3 and y = 4 we get r = \\sqrt{9 + 16} "
    "= \\sqrt{25} = 5. The angle is \\theta = \\arctan(4/3), which is about "
    "0.927 radians, and the problem asks for exact polar form."
)

BOX_THEN_RAMBLE = (
    "The radius is 5 and the angle is \\arctan(4/3), so the polar form is "
    "\\boxed{(5, \\arctan(4/3))}. "
    "Converting back gives x = 5 cos(0.9273) = 3.0001 and y = 5 sin(0.9273) "
    "= 3.9998, which round to the original point (3, 4). A second check with "
    "the tangent relation gives tan(theta) = 4/3, so theta is 53.13 degrees "
    "or 0.9273 radians, and the radius squared is 9 + 16 = 25. A third check "
    "in the complex plane writes z = 3 + 4i, whose modulus is the square root "
    "of 25 and whose argument has cosine 3/5 and sine 4/5. None of these "
    "checks contradicts the boxed expression, so the derivation is complete."
)

LOOP_NO_BOX = "Adding one to the counter gives the next value. " * 80

BOX_THEN_LOOP = (
    "The answer follows from the sum, which simplifies to the value below. "
    "\\boxed{12} "
    + "then we check the invariant again and again. " * 40
)

BOX_AT_END = "The total is 17 + 8 + 16 = 41, so the answer is \\boxed{41}."

BOX_CUT_AT_CAP = "We solve step by step and the final answer is \\boxed{41"


def test_no_box_is_class_one():
    result = truncation.classify_completion(NO_BOX, "5", truncated=True)
    assert result["class"] == "no_box"
    assert result["boxed_present"] is False
    assert result["correct_box_anywhere"] is False


def test_box_then_ramble_is_class_two():
    result = truncation.classify_completion(
        BOX_THEN_RAMBLE, "(5, \\arctan(4/3))", truncated=True)
    assert result["boxed_present"] is True
    assert result["tail_after_last_box_nonspace"] >= (
        truncation.BOX_THEN_CONTINUED_MIN_NONSPACE)
    assert result["class"] == "boxed_then_continued"
    assert result["correct_box_anywhere"] is True


def test_box_at_the_end_is_not_continued():
    result = truncation.classify_completion(BOX_AT_END, "41", truncated=False)
    assert result["boxed_present"] is True
    assert result["tail_after_last_box_nonspace"] < (
        truncation.BOX_THEN_CONTINUED_MIN_NONSPACE)
    assert result["class"] == "other"


def test_repetition_loop_without_box_is_class_three():
    signal = termination.classify(LOOP_NO_BOX)
    assert signal["loop"] is True
    result = truncation.classify_completion(LOOP_NO_BOX, "1", truncated=True)
    assert result["boxed_present"] is False
    assert result["class"] == "loop_degenerate"
    assert result["loop"] is True


def test_loop_after_a_box_is_class_three_and_the_box_overlap_is_visible():
    signal = termination.classify(BOX_THEN_LOOP)
    assert signal["loop"] is True, signal["reason"]
    result = truncation.classify_completion(BOX_THEN_LOOP, "12", truncated=True)
    assert result["class"] == "loop_degenerate"
    assert result["boxed_present"] is True
    assert result["loop"] is True
    assert result["correct_box_anywhere"] is True


def test_a_correct_box_anywhere_survives_a_later_wrong_box():
    text = ("One route gives \\boxed{5} but then the derivation continues and "
            "a second route is attempted, which ends at \\boxed{7}.")
    result = truncation.classify_completion(text, "5", truncated=True)
    assert result["boxed_count"] == 2
    assert result["correct_box_anywhere"] is True
    assert result["strict_status"] == "ambiguous"
    assert result["strict_correct"] is False


def test_a_box_cut_off_by_the_cap_is_not_a_box():
    result = truncation.classify_completion(BOX_CUT_AT_CAP, "41", truncated=True)
    assert result["boxed_present"] is False
    assert result["class"] == "no_box"


def test_the_flexible_fallback_is_not_a_box():
    text = "After simplifying, the final answer is 5."
    result = truncation.classify_completion(text, "5", truncated=True)
    assert result["boxed_present"] is False
    assert result["class"] == "no_box"
    assert result["flexible_correct"] is True
    assert result["correct_box_anywhere"] is False


def test_unknown_gold_leaves_correctness_fields_null():
    result = truncation.classify_completion(NO_BOX, None, truncated=False)
    assert result["correct_box_anywhere"] is None
    assert result["strict_correct"] is None
    assert result["flexible_correct"] is None


def test_box_spans_agree_with_the_registered_extractor():
    for text in (BOX_THEN_RAMBLE, BOX_THEN_LOOP,
                 "a \\boxed{1} b \\boxed{ 2 } c \\boxed{\\frac{3}{4}}" + " " * 40):
        spans = truncation.box_spans(text)
        assert [content for content, _o, _c in spans] == rft._boxed_answers(text)
        for content, open_brace, close_brace in spans:
            assert text[open_brace] == "{" and text[close_brace] == "}"


def test_classification_is_deterministic():
    first = truncation.classify_completion(BOX_THEN_LOOP, "12", truncated=True)
    second = truncation.classify_completion(BOX_THEN_LOOP, "12", truncated=True)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_summarise_partitions_every_row_exactly_once():
    rows = [
        truncation.classify_completion(NO_BOX, "5", truncated=True),
        truncation.classify_completion(BOX_THEN_RAMBLE, "x", truncated=True),
        truncation.classify_completion(LOOP_NO_BOX, "1", truncated=True),
        truncation.classify_completion(BOX_AT_END, "41", truncated=False),
    ]
    summary = truncation.summarise(rows)
    assert summary["rows"] == 4
    assert sum(summary["by_class"].values()) == 4
    assert set(summary["by_class"]) == set(truncation.CLASSES)
    truncated = truncation.summarise(rows, only_truncated=True)
    assert truncated["rows"] == 3
    assert sum(truncated["by_class"].values()) == 3


def test_summarise_refuses_a_missing_truncated_flag():
    row = truncation.classify_completion(NO_BOX, "5", truncated=None)
    with pytest.raises(ValueError):
        truncation.summarise([row], only_truncated=True)


def test_strict_and_flexible_match_the_registered_s54_scorer():
    """The local mirror must agree with the scorer the receipts use."""
    import s54_post_trained_eval as s54

    battery = [
        NO_BOX,
        BOX_THEN_RAMBLE,
        BOX_AT_END,
        BOX_CUT_AT_CAP,
        "One route gives \\boxed{5} and another gives \\boxed{7}.",
        "\\boxed{5 \\text{ or } 7}",
        "After simplifying, the final answer is 5.",
        "Nothing extractable here.",
    ]
    for text in battery:
        ours_strict = truncation.strict_extraction(text)
        theirs_strict = s54.extract_strict(text)
        assert ours_strict["status"] == theirs_strict["status"], text
        assert ours_strict["value"] == theirs_strict["value"], text
        assert ours_strict["method"] == theirs_strict["method"], text

        ours_flexible = truncation.flexible_extraction(text)
        theirs_flexible = s54.extract_flexible(text, "math500")
        assert ours_flexible["status"] == theirs_flexible["status"], text
        assert ours_flexible["value"] == theirs_flexible["value"], text
        assert ours_flexible["method"] == theirs_flexible["method"], text


def test_registered_classifier_hash_matches_the_file():
    if not os.path.exists(REGISTRATION):
        pytest.skip("S73 preregistration is not present at this revision")
    with open(REGISTRATION, "r", encoding="utf-8") as handle:
        registration = json.load(handle)
    registered = registration["classifier"]
    assert registered["module"] == "oracle/frontier/truncation.py"
    digest = hashlib.sha256(
        open(os.path.join(ROOT, registered["module"]), "rb").read()).hexdigest()
    assert registered["sha256"] == digest
    assert registered["version"] == truncation.CLASSIFIER_VERSION
    assert registered["box_then_continued_min_nonspace"] == (
        truncation.BOX_THEN_CONTINUED_MIN_NONSPACE)
    assert list(registration["classifier"]["classes"]) == list(truncation.CLASSES)
