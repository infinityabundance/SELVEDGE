"""S56 — termination-aware rollouts: reward, early cut, false-cut accounting.

These are CPU tests of the policy the registration states:

* a truncated rollout is wrong even when its prefix parses;
* the early cut fires only after the minimum and only on a loop-classified tail;
* every cut records emitted/saved tokens;
* a false cut is an uncut-correct answer whose prefix no longer verifies.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import rollouts, termination  # noqa: E402

LOOP_TEXT = "we repeat the same step again and again " * 40
CORRECT = ("The first basket has 17 apples, the second has 8, and the third "
           "has twice the second minus four, which is 12. The total is "
           "17 + 8 + 12 = 37, so the final answer is \\boxed{37}.")


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, text: str):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text


def test_cut_criteria_waits_for_the_minimum_and_stride():
    tokenizer = FakeTokenizer(LOOP_TEXT)
    criteria = rollouts.LoopCutCriteria(
        tokenizer, prompt_length=4, check_every_tokens=8, min_tokens=16)
    prompt = torch.zeros(1, 4, dtype=torch.long)
    # Before the minimum: no cut.
    assert not bool(criteria(torch.cat([prompt, torch.zeros(1, 12, dtype=torch.long)],
                                       dim=1)).any())
    # At the minimum but not on the stride: no cut.
    assert not bool(criteria(torch.cat([prompt, torch.zeros(1, 15, dtype=torch.long)],
                                       dim=1)).any())
    # On the stride with a loop tail: cut, and the evidence is recorded.
    stop = criteria(torch.cat([prompt, torch.zeros(1, 16, dtype=torch.long)], dim=1))
    assert bool(stop.any())
    assert criteria.cut[0] is True
    assert criteria.cut_tokens[0] == 16
    assert criteria.cut_reason[0] in {"exact_period_tail",
                                      "compressed_repetitive_tail"}


def test_cut_criteria_does_not_fire_on_varied_text():
    tokenizer = FakeTokenizer(CORRECT)
    criteria = rollouts.LoopCutCriteria(
        tokenizer, prompt_length=4, check_every_tokens=8, min_tokens=8)
    stop = criteria(torch.zeros(1, 4 + 64, dtype=torch.long))
    assert not bool(stop.any())
    assert criteria.cut[0] is False


def test_truncated_rollout_is_wrong_even_when_the_prefix_parses():
    result = rollouts.verified_reward(CORRECT, "37", "gsm8k", truncated=True)
    assert result["reward"] == 0.0
    assert result["verdict"] == "truncated"


def test_terminated_correct_rollout_gets_reward_one():
    result = rollouts.verified_reward(CORRECT, "37", "gsm8k", truncated=False)
    assert result["reward"] == 1.0
    assert result["verified"] is True


def test_simulate_cut_finds_the_loop_tail():
    assert rollouts.simulate_cut(CORRECT) is None
    cut = rollouts.simulate_cut(LOOP_TEXT)
    assert cut is not None
    assert cut["cut_chars"] < len(LOOP_TEXT)


def test_false_cut_is_an_uncut_correct_answer_losing_its_answer():
    # Correct answer, but a long repeated tail before the boxed answer: the cut
    # removes the answer, so the uncut-correct row is a false cut.
    bad_cut = "The total is 12. " + LOOP_TEXT + " Therefore \\boxed{37}."
    rows = [
        {"completion": bad_cut, "new_tokens": 2000, "flexible_correct": True,
         "gold": "37", "task": "gsm8k", "mode": "non-thinking",
         "decoding": "greedy"},
        {"completion": LOOP_TEXT, "new_tokens": 2048, "flexible_correct": False,
         "gold": "999", "task": "gsm8k", "mode": "non-thinking",
         "decoding": "greedy"},
    ]
    result = rollouts.false_cut_analysis(rows, stride_chars=32)
    assert result["samples"] == 2
    assert result["cut_samples"] == 2
    assert result["correct_uncut"] == 1
    assert result["false_cuts"] == 1
    assert result["false_cut_rate"] == 1.0
    assert result["false_cut_rate_of_cuts"] == 0.5
    assert result["saved_tokens_estimate"] > 0
    assert result["false_cut_examples"]


def test_cut_free_rows_are_not_counted():
    rows = [{"completion": CORRECT, "new_tokens": 100,
             "flexible_correct": True, "gold": "37", "task": "gsm8k",
             "mode": "non-thinking", "decoding": "greedy"}]
    result = rollouts.false_cut_analysis(rows)
    assert result["cut_samples"] == 0
    assert result["false_cuts"] == 0
    assert result["false_cut_rate"] == 0.0
    assert result["saved_tokens_estimate"] == 0.0


def test_thresholds_come_from_the_calibration_module():
    assert (rollouts.termination.DEFAULT_THRESHOLDS
            == termination.DEFAULT_THRESHOLDS)
