"""S56 — textual termination/loop detector, CPU only.

The detector must distinguish three states without changing decoding:

* a finished answer (EOS) that is long but varied is not a loop;
* a truncated answer (harness cap) is not automatically a loop;
* a degenerate tail — an exact period, or a tail that compresses far better than
  the body while its n-grams repeat — is a loop.

The summary keeps the harness's `truncated` flag as the truncation label and
only ever adds the textual loop verdict beside it.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import termination  # noqa: E402


VARIED = (
    "Let x be the number of apples. The first basket holds 17 apples, the "
    "second holds 9 fewer than the first, and the third holds twice the second "
    "minus four. Then x = 17 + (17 - 9) + 2 * (17 - 9) - 4. Compute each term: "
    "17 - 9 = 8, twice that is 16, and 8 minus 4 is 4. So x = 17 + 8 + 16 + 4 = "
    "45. We should check the arithmetic again because the third basket's rule "
    "is easy to misread: 2 * 8 = 16, then subtract 4 to get 12, not 4. Recompute "
    "the total with 12: 17 + 8 + 16 + 12 = 53. Therefore the final answer is "
    "\\boxed{53}."
)
LOOP = ("Adding one to the counter gives the next value. " * 80)
PERIOD = "x = x + 1; then we check the invariant again; " * 30


def test_varied_answer_is_not_a_loop():
    signal = termination.classify(VARIED)
    assert signal["compression"]["measurable"] is True
    assert signal["loop"] is False
    assert signal["reason"] in {"no_loop", "compressed_but_not_repetitive",
                                "repetitive_but_not_compressed"}


def test_exact_repeated_text_is_a_loop():
    signal = termination.classify(LOOP)
    assert signal["loop"] is True
    assert signal["reason"] in {"compressed_repetitive_tail", "exact_period_tail"}


def test_period_is_found_for_a_short_repeated_tail():
    signal = termination.classify(PERIOD)
    assert signal["repetition"]["period"] is not None
    assert signal["repetition"]["period"]["period_chars"] == len(
        "x = x + 1; then we check the invariant again; ")
    assert signal["loop"] is True


def test_short_text_is_not_classified():
    signal = termination.classify("Answer: 4.")
    assert signal["loop"] is False
    assert signal["reason"] == "too_short"
    assert signal["compression"]["measurable"] is False


def test_truncation_is_the_harness_flag_not_a_text_inference():
    long_answer = VARIED * 12  # long, varied, would exceed most caps
    summary = termination.summarise([
        {"problem_id": "p1", "completion": long_answer, "new_tokens": 4096,
         "truncated": True, "correct": False, "problem_correct": False},
    ])
    assert summary["truncation_rate"] == 1.0
    assert summary["loop_classified_fraction"] == 0.0
    assert summary["sample_level"]["truncated"]["samples"] == 1
    assert summary["sample_level"]["terminated"]["samples"] == 0


def test_summary_conditions_accuracy_on_termination():
    rows = [
        {"problem_id": "p1", "completion": VARIED, "new_tokens": 300,
         "truncated": False, "correct": True, "problem_correct": True},
        {"problem_id": "p2", "completion": LOOP, "new_tokens": 4096,
         "truncated": True, "correct": False, "problem_correct": False},
        {"problem_id": "p3", "completion": LOOP, "new_tokens": 4096,
         "truncated": True, "correct": False, "problem_correct": False},
    ]
    summary = termination.summarise(rows)
    assert summary["samples"] == 3
    assert summary["truncation_rate"] == 2 / 3
    assert summary["loop_classified_samples"] == 2
    assert summary["sample_level"]["terminated"]["accuracy"] == 1.0
    assert summary["sample_level"]["truncated"]["accuracy"] == 0.0
    assert summary["sample_level"]["loop"]["accuracy"] == 0.0
    assert summary["sample_level"]["clean"]["accuracy"] == 1.0
    assert summary["cross_tab"] == {
        "terminated_clean": 1, "terminated_loop": 0,
        "truncated_clean": 0, "truncated_loop": 2,
    }
    # The mean-token block separates the groups rather than averaging them away.
    assert summary["mean_new_tokens"]["terminated"] == 300.0
    assert summary["mean_new_tokens"]["truncated"] == 4096.0


def test_problem_level_conditioning_uses_the_scored_verdict():
    rows = [
        {"problem_id": "p1", "completion": VARIED, "new_tokens": 300,
         "truncated": False, "correct": True, "problem_correct": True},
        {"problem_id": "p2", "completion": LOOP, "new_tokens": 4096,
         "truncated": True, "correct": True, "problem_correct": True},
    ]
    summary = termination.summarise(rows)
    assert summary["problem_level"]["all_terminated"] == {
        "problems": 1, "correct": 1, "accuracy": 1.0}
    assert summary["problem_level"]["any_truncated"] == {
        "problems": 1, "correct": 1, "accuracy": 1.0}


def test_thresholds_are_honoured():
    # A stricter compression threshold cannot classify more answers.
    loose = termination.summarise(
        [{"problem_id": "p", "completion": LOOP, "new_tokens": 500,
          "truncated": False, "correct": False, "problem_correct": False}],
        thresholds={"tail_over_full_max": 0.99, "ngram_repeat_min": 0.0},
    )
    strict = termination.summarise(
        [{"problem_id": "p", "completion": LOOP, "new_tokens": 500,
          "truncated": False, "correct": False, "problem_correct": False}],
        thresholds={"tail_over_full_max": 0.10, "ngram_repeat_min": 0.99},
    )
    assert loose["loop_classified_samples"] >= strict["loop_classified_samples"]


def test_percentiles_are_deterministic_and_bounded():
    values = [1.0, 2.0, 3.0, 4.0]
    result = termination.percentiles(values)
    assert result["0.0"] == 1.0
    assert result["1.0"] == 4.0
    assert termination.percentiles([])["0.5"] is None


def test_default_thresholds_match_the_calibration_receipt():
    """A threshold changed without regenerating the receipt is a defect."""
    import json

    path = os.path.join(ROOT, "results/stage1/s56-termination-calibration.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip("calibration receipt is not present in this checkout")
    with open(path, "r", encoding="utf-8") as handle:
        operating = json.load(handle)["operating_point"]
    for key in ("tail_over_full_max", "ngram_repeat_min", "ngram_n",
                "period_repeats_min", "period_chars_max"):
        assert termination.DEFAULT_THRESHOLDS[key] == operating[key], key
