"""P9.3 — CPU tests for the combination-utility court on synthetic ground truth.

The fixture knows `u_m`, `u_e` and `I` by construction, so every estimator has an
answer to check against: sign recovery on synergy and antagonism, a CI that
covers zero under independence, CI coverage across replications, and a
controller comparison where the cross term is what separates the two arms.
No 0.6B number is claimed here.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ORACLE, "eval"))

import combination_utility as cu  # noqa: E402


# ---------------------------------------------------------------------------
# the artifact, the causal view and leakage


def test_the_four_condition_artifact_is_complete_and_deterministic():
    first = cu.synthetic_dataset(kind="synergy", count=24, seed=5)
    second = cu.synthetic_dataset(kind="synergy", count=24, seed=5)
    assert first["items"] == 24
    assert len(first["records"]) == 24
    for row in first["records"].values():
        assert set(cu.CONDITIONS).issubset(row)
        assert row["interaction"] == pytest.approx(
            row["both"] - row["m"] - row["e"] + row["none"])
    assert first["manifest"] == second["manifest"]
    assert first["records"] == second["records"]
    manifest = first["manifest"]
    assert manifest["leakage"]["causal_inputs_only"] is True
    assert manifest["item_id_sha256"] and manifest["feature_sha256"]
    assert "label" in manifest["leakage"]["blocked_keys"]
    train, test = first["split"]["train"], first["split"]["test"]
    assert sorted(train + test) == sorted(first["records"])
    assert not set(train) & set(test)


def test_the_runner_cannot_read_an_outcome_key():
    item = {"id": "x", "features": [0.1, 0.2, 0.3], "label": 7}

    def runner(view, use_m, use_e):
        return float(view["label"])

    with pytest.raises(ValueError, match="outcome keys"):
        cu.generate_four_condition([item], runner)
    view = cu.CausalView({"id": "x", "features": [0.1], "label": 7})
    with pytest.raises(KeyError, match="outcome key"):
        view["label"]
    assert view["id"] == "x"
    assert "label" not in view


# ---------------------------------------------------------------------------
# the interaction estimator


def test_the_sign_conventions_agree_between_rewards_and_losses():
    rewards = {"none": 0.0, "m": 1.0, "e": 1.0, "both": 2.5}
    losses = {key: -value for key, value in rewards.items()}
    assert cu.interaction_from_rewards(rewards) == pytest.approx(0.5)
    assert cu.interaction_from_losses(losses) == pytest.approx(0.5)
    # a sub-additive pair has a negative interaction in both conventions
    rewards["both"] = 1.5
    losses = {key: -value for key, value in rewards.items()}
    assert cu.interaction_from_rewards(rewards) == pytest.approx(-0.5)
    assert cu.interaction_from_losses(losses) == pytest.approx(-0.5)


def test_the_estimator_recovers_the_known_sign_of_each_regime():
    expectations = {"synergy": (1, "synergy"), "antagonism": (-1, "antagonism")}
    for kind, (sign, label) in expectations.items():
        dataset = cu.synthetic_dataset(kind=kind, count=96, seed=7)
        summary = cu.interaction_summary(dataset, resamples=2000)
        assert sign * summary["mean"] > 0
        assert summary["label"] == label
        assert sign * summary["low"] > 0 or sign * summary["high"] > 0
        if sign > 0:
            assert summary["low"] > 0
        else:
            assert summary["high"] < 0
        assert summary["se"] is not None and summary["se"] > 0
    dataset = cu.synthetic_dataset(kind="independence", count=96, seed=7)
    summary = cu.interaction_summary(dataset, resamples=2000)
    assert summary["label"] == "no detectable interaction"
    assert summary["low"] <= 0 <= summary["high"]
    assert abs(summary["mean"]) < 0.05


def test_the_interval_covers_the_truth_at_close_to_its_nominal_rate():
    study = cu.coverage_study("synergy", replications=40, count=80,
                              resamples=500, seed=11)
    assert study["replications"] == 40
    assert study["coverage"] >= 0.85
    assert study["mean_estimate"] == pytest.approx(study["truth"], abs=0.03)


def test_the_summary_is_deterministic():
    dataset = cu.synthetic_dataset(kind="synergy", count=32, seed=9)
    first = cu.interaction_summary(dataset, resamples=500)
    second = cu.interaction_summary(dataset, resamples=500)
    assert first == second


# ---------------------------------------------------------------------------
# the controllers and the combinatorial oracle


def test_the_interaction_controller_beats_the_additive_one_under_synergy():
    dataset = cu.synthetic_dataset(kind="synergy", count=96, seed=7)
    evaluation = cu.evaluate_controllers(dataset)
    additive = evaluation["controllers"]["additive-utility"]
    interaction = evaluation["controllers"]["interaction-aware"]
    assert interaction["accuracy"] > additive["accuracy"]
    assert interaction["mean_reward"] > additive["mean_reward"]
    difference = evaluation["interaction_minus_additive"]
    assert difference["mean"] > 0 and difference["low"] > 0
    assert interaction["cost"]["learned_parameters"] > \
        additive["cost"]["learned_parameters"]
    # the oracle is the bound and is not deployable
    oracle = evaluation["controllers"]["combinatorial-oracle"]
    assert oracle["oracle_gap"] == 0.0
    assert oracle["cost"]["deployable"] is False
    for row in evaluation["controllers"].values():
        assert row["mean_reward"] <= oracle["mean_reward"] + 1e-9


def test_without_an_interaction_the_two_controllers_are_equivalent():
    dataset = cu.synthetic_dataset(kind="independence", count=128, seed=7)
    evaluation = cu.evaluate_controllers(dataset)
    difference = evaluation["interaction_minus_additive"]
    additive = evaluation["controllers"]["additive-utility"]
    interaction = evaluation["controllers"]["interaction-aware"]
    assert abs(difference["mean"]) < 0.02
    assert interaction["accuracy"] >= additive["accuracy"] - 0.05


def test_the_controls_have_the_documented_cost_and_behaviour():
    dataset = cu.synthetic_dataset(kind="synergy", count=48, seed=7)
    evaluation = cu.evaluate_controllers(dataset)
    balanced = evaluation["controllers"]["load-balanced"]
    gates = evaluation["controllers"]["independent-gates"]
    assert balanced["picks"]["both"] == evaluation["items"]
    assert balanced["cost"]["learned_parameters"] == 0
    assert gates["cost"]["learned_parameters"] > 0
    assert sum(balanced["picks"].values()) == evaluation["items"]


def test_missing_features_are_refused():
    dataset = cu.synthetic_dataset(kind="synergy", count=8, seed=7)
    for row in dataset["records"].values():
        row.pop("features")
    with pytest.raises(ValueError, match="no features"):
        cu.evaluate_controllers(dataset)
