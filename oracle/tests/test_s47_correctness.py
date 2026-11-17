"""CPU tests for the sequence-level correctness measurement.

The evaluation pipeline already reports next-token CE and teacher KL. These
tests check that the companion measurement extracts the final answer, reports a
verdict per sequence, and presents accuracy beside the likelihood means.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import correctness  # noqa: E402


def test_gsm8k_verdicts_are_exact_and_separate_the_failure_modes():
    correct = correctness.verify_completion(
        "She has 3 apples and buys 2. #### 5", "5", "gsm8k")
    wrong = correctness.verify_completion(
        "She has 3 apples and buys 2. #### 6", "5", "gsm8k")
    ambiguous = correctness.verify_completion(
        "First #### 5 then #### 6", "5", "gsm8k")
    unparseable = correctness.verify_completion(
        "no answer marker here at all", "5", "gsm8k")
    assert correct["verdict"] == "correct"
    assert wrong["verdict"] == "wrong"
    assert ambiguous["verdict"] == "ambiguous"
    assert unparseable["verdict"] == "unparseable"


def test_math_and_minerva_use_the_same_boxed_extractor():
    completion = r"The value is \boxed{\frac{1}{2}}."
    math = correctness.verify_completion(completion, "\\frac{1}{2}", "math")
    minerva = correctness.verify_completion(
        completion, "\\frac{1}{2}", "minerva_math500")
    assert math["verdict"] == "correct"
    assert minerva["verdict"] == "correct"
    # the numeric forms are compared as numbers, not as strings
    assert correctness.verify_completion(
        r"\boxed{0.5}", r"\frac{1}{2}",
        "minerva_math500")["verdict"] == "correct"


def test_summary_reports_accuracy_beside_ce_and_kl():
    records = [
        {"verdict": "correct", "ce": 1.0, "kl": 0.2},
        {"verdict": "correct", "ce": 3.0, "kl": 0.4},
        {"verdict": "wrong", "ce": 2.0, "kl": 0.3},
        {"verdict": "unparseable", "ce": 4.0, "kl": 0.5},
    ]
    summary = correctness.summarise(records)
    assert summary["items"] == 4
    assert summary["correct"] == 2
    assert summary["wrong"] == 1
    assert summary["unparseable"] == 1
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["accuracy_answered"] == pytest.approx(2 / 3)
    assert summary["accuracy_stderr"] == pytest.approx(0.25)
    assert summary["mean_ce"] == pytest.approx(2.5)
    assert summary["mean_kl"] == pytest.approx(0.35)
    assert summary["ce_items"] == summary["kl_items"] == 4


def test_sequence_correctness_accepts_a_target_and_keeps_ce_kl():
    result = correctness.sequence_correctness(
        [
            {"completion": "#### 12", "target": "12", "ce": 1.0, "kl": 0.1},
            {"completion": "#### 9", "target": "12", "ce": 2.0, "kl": 0.2},
        ],
        "gsm8k")
    assert result["dataset"] == "gsm8k"
    assert result["correct"] == 1 and result["wrong"] == 1
    assert result["mean_ce"] == pytest.approx(1.5)
    assert result["mean_kl"] == pytest.approx(0.15)


def test_harness_samples_reports_its_own_verdict():
    samples = [
        {"resps": [["reasoning... #### 5"]], "target": "5"},
        {"resps": [["reasoning... #### 4"]], "target": "5"},
    ]
    result = correctness.harness_samples(samples, "gsm8k")
    assert result["harness_samples"] == 2
    assert result["correct"] == 1
    assert result["wrong"] == 1
    assert "harness metric" in result["note"]


def test_unverifiable_task_is_rejected():
    with pytest.raises(ValueError):
        correctness.verify_completion("#### 1", "1", "mmlu")
