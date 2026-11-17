"""CPU contract for the S53 model-level optimiser comparison harness.

The GPU driver is queued; these tests exercise the harness with fake states so
the arithmetic, the matched-init rule, the evaluation schedule and the paired
interval cannot regress before a GPU is spent.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import optim_compare  # noqa: E402


def _evaluate(state):
    return {
        "ce": float(((state - 1.0) ** 2).sum()),
        "kl": float(state.abs().sum()),
    }


def _spec(delta, calls):
    def make():
        calls["make"] += 1
        return torch.zeros(4)

    def step(state, _index):
        calls["step"] += 1
        return state + delta

    def release(state):
        calls["release"].append(float(state.mean()))

    return optim_compare.ArmSpec(make=make, step=step, release=release)


def test_run_comparison_evaluates_init_points_and_final():
    calls = {"make": 0, "step": 0, "release": []}
    comparison = optim_compare.run_comparison(
        {"a": _spec(0.1, calls), "b": _spec(0.05, calls)},
        _evaluate, steps=10, eval_every=5)
    assert comparison.eval_steps == [0, 5, 10]
    assert sorted(comparison.series) == ["a", "b"]
    for arm in ("a", "b"):
        assert [point["step"] for point in comparison.points(arm)] == [0, 5, 10]
    # matched init: both arms started at the same exact state
    assert comparison.series["a"][0] == comparison.series["b"][0]
    assert calls["make"] == 2
    assert calls["step"] == 20
    # one release per arm, after its final state
    assert len(calls["release"]) == 2
    assert calls["release"][0] == pytest.approx(1.0)


def test_run_comparison_is_sequential_and_matched():
    events = []

    def make():
        events.append("make")
        return torch.zeros(2)

    def release(_state):
        events.append("release")

    spec = optim_compare.ArmSpec(make=make, step=lambda s, i: s + 0.01,
                                 release=release)
    optim_compare.run_comparison({"a": spec, "b": spec}, _evaluate,
                                 steps=2, eval_every=1)
    assert events == ["make", "release", "make", "release"]


def test_summary_pairs_candidate_minus_reference_with_a_seeded_interval():
    calls = {"make": 0, "step": 0, "release": []}
    comparison = optim_compare.run_comparison(
        {"custom": _spec(0.1, calls), "adamw": _spec(0.05, calls)},
        _evaluate, steps=10, eval_every=5)
    summary = optim_compare.summarise_comparison(
        comparison, reference="adamw", candidate="custom", seed=53)
    assert summary["matched_init"] is True
    assert summary["finite"] is True
    assert summary["eval_steps"] == [0, 5, 10]
    paired = summary["paired"]["custom_minus_adamw"]
    assert paired["ce"]["n"] == 3
    # eval points: +0 at init, -1.25 at step 5, -1.0 at step 10
    assert paired["ce"]["mean"] == pytest.approx(-0.75)
    again = optim_compare.summarise_comparison(
        comparison, reference="adamw", candidate="custom", seed=53)
    assert again["paired"] == summary["paired"]
    assert paired["ce"]["ci95"][0] <= paired["ce"]["mean"] <= paired["ce"]["ci95"][1]


def test_summary_refuses_an_unknown_arm():
    calls = {"make": 0, "step": 0, "release": []}
    comparison = optim_compare.run_comparison(
        {"a": _spec(0.1, calls)}, _evaluate, steps=2, eval_every=1)
    with pytest.raises(KeyError):
        optim_compare.summarise_comparison(comparison, "b", "a")


def test_run_comparison_rejects_non_positive_schedule():
    spec = _spec(0.1, {"make": 0, "step": 0, "release": []})
    with pytest.raises(ValueError):
        optim_compare.run_comparison({"a": spec}, _evaluate, steps=0)
    with pytest.raises(ValueError):
        optim_compare.run_comparison({"a": spec}, _evaluate, steps=1,
                                     eval_every=0)
