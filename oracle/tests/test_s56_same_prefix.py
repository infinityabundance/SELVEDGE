"""S56 — the corrected stage-2a objective and the renamed divergence.

Two claims are pinned here:

* **completion-only SFT** — prompt positions are ``IGNORE_INDEX`` and contribute
  neither CE nor KL, and an all-masked batch fails closed;
* **same-prefix teacher KL** — the KL is computed at the positions the student
  was supervised on, from logits that describe the same causal prefix. A teacher
  run over a different prefix is rejected, and the divergence helper reports a
  *textual* divergence only.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import same_prefix  # noqa: E402

IGNORE = same_prefix.IGNORE_INDEX


def _logits(rows: int, sequence: int, vocab: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, sequence, vocab, generator=generator)


def test_completion_ce_ignores_prompt_positions():
    student = _logits(1, 4, 5)
    targets = torch.tensor([[IGNORE, IGNORE, 2, 3]])
    first = same_prefix.completion_ce(student, targets)
    perturbed = student.clone()
    perturbed[0, 0] = perturbed[0, 0] + 100.0
    perturbed[0, 1] = perturbed[0, 1] - 100.0
    second = same_prefix.completion_ce(perturbed, targets)
    assert torch.allclose(first, second)


def test_completion_ce_is_per_supervised_position():
    # Two supervised positions, each a uniform distribution over 5 classes:
    # mean CE = log 5.
    student = torch.zeros(1, 3, 5)
    targets = torch.tensor([[IGNORE, 1, 2]])
    value = same_prefix.completion_ce(student, targets)
    assert torch.allclose(value, torch.log(torch.tensor(5.0)), atol=1e-6)


def test_all_masked_batch_fails_closed():
    student = _logits(1, 3, 5)
    with pytest.raises(ValueError):
        same_prefix.completion_ce(student, torch.full((1, 3), IGNORE))


def test_same_prefix_kl_is_zero_for_identical_logits():
    logits = _logits(1, 5, 7)
    targets = torch.tensor([[IGNORE, IGNORE, 3, 4, 5]])
    value = same_prefix.same_prefix_teacher_kl(logits, logits.clone(), targets, 2.0)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_same_prefix_kl_ignores_masked_positions():
    student = _logits(1, 5, 7, seed=1)
    teacher = _logits(1, 5, 7, seed=2)
    targets = torch.tensor([[IGNORE, IGNORE, 3, 4, 5]])
    baseline = same_prefix.same_prefix_teacher_kl(student, teacher, targets, 2.0)
    student = student.clone()
    teacher = teacher.clone()
    student[0, 0] += 50.0   # prompt positions: must not move the KL
    teacher[0, 1] -= 50.0
    assert torch.allclose(
        baseline, same_prefix.same_prefix_teacher_kl(student, teacher, targets, 2.0))


def test_kl_direction_is_student_to_teacher():
    student = torch.zeros(1, 1, 2)
    teacher = torch.tensor([[[0.0, 10.0]]])
    targets = torch.tensor([[0]])
    value = float(same_prefix.same_prefix_teacher_kl(
        student, teacher, targets, 1.0))
    # KL(student || teacher) is large: the teacher puts almost all mass on 1.
    assert value > 2.0
    reverse = float(same_prefix.same_prefix_teacher_kl(
        teacher, student, targets, 1.0))
    assert reverse < value


def test_combined_objective_matches_the_formula():
    student = _logits(1, 4, 6, seed=3)
    teacher = _logits(1, 4, 6, seed=4)
    targets = torch.tensor([[IGNORE, 2, 3, 4]])
    temperature, weight = 2.0, 0.5
    loss, ce, kl = same_prefix.sft_with_same_prefix_kl(
        student, teacher, targets, temperature, weight)
    assert torch.allclose(loss, ce + weight * temperature ** 2 * kl)


def test_assert_same_prefix_fails_closed():
    left = torch.tensor([[1, 2, 3, 4]])
    same_prefix.assert_same_prefix(left, left.clone())
    with pytest.raises(ValueError):
        same_prefix.assert_same_prefix(left, torch.tensor([[1, 2, 9, 4]]))
    with pytest.raises(ValueError):
        same_prefix.assert_same_prefix(left, torch.tensor([[1, 2, 3]]))


def test_first_textual_divergence_is_textual_not_reasoning():
    assert same_prefix.first_textual_divergence([1, 2, 3], [1, 9, 3]) == 1
    # A strict prefix has no differing token; it is not a "reasoning error".
    assert same_prefix.first_textual_divergence([1, 2], [1, 2, 3]) is None
    assert same_prefix.first_textual_divergence([1, 2, 3], [1, 2]) is None
    assert same_prefix.first_textual_divergence([], [1]) is None
    # Identical continuations have no divergence.
    assert same_prefix.first_textual_divergence([4, 5], [4, 5]) is None
