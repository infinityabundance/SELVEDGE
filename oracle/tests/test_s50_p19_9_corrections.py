"""S50 corrections to P19.9: pre-read decisions and counterfactual outcome labels.

Two ordering/supervision defects are pinned here:

* the original `UtilityRouter` consumes `r_t`, so it cannot decide until after
  the memory has been fetched. `PreReadUtilityRouter` must decide from `h_t`
  and compact metadata alone, and must refuse post-read or label inputs;
* the original outcome source used an absolute correctness label, which cannot
  tell a helpful token from a token in a sequence that happened to be correct.
  `outcome_delta` must be a paired probability difference from separately
  verified rollouts.

No training, sampling or generation runs in these tests.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import utility_routing as routing  # noqa: E402


def _metadata(batch: int = 2, positions: int = 5):
    return {
        "position_fraction": torch.linspace(0.0, 1.0, positions).expand(batch, positions),
        "budget_fraction": torch.full((batch, positions), 1.0),
    }


def test_pre_read_predictor_uses_hidden_state_and_compact_metadata_only():
    router = routing.PreReadUtilityRouter(hidden=6, tau=1.0)
    hidden = torch.randn(2, 5, 6)
    predicted = router.predict(hidden, _metadata())
    assert predicted.shape == (2, 5)
    # h_t (6) + hidden_norm (1) + two metadata fractions (2)
    assert router.feature_dim == 9

    for bad in ("r_t", "read_norm", "gate_t", "target_id", "label"):
        metadata = _metadata()
        metadata[bad] = torch.zeros(2, 5)
        with pytest.raises(ValueError):
            router.predict(hidden, metadata)


def test_pre_read_metadata_is_required_and_must_be_a_fraction():
    router = routing.PreReadUtilityRouter(hidden=4)
    hidden = torch.randn(1, 3, 4)
    with pytest.raises(ValueError):
        router.predict(hidden, {"position_fraction": torch.zeros(1, 3)})
    metadata = _metadata(batch=1, positions=3)
    metadata["position_fraction"] = torch.full((1, 3), 4.0)
    with pytest.raises(ValueError):
        router.predict(hidden, metadata)


def test_decision_precedes_the_fetch_and_can_refuse_every_read():
    router = routing.PreReadUtilityRouter(hidden=4, tau=1.0)
    hidden = torch.randn(1, 5, 4)
    metadata = _metadata(batch=1, positions=5)
    store = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    fetches = {"count": 0}

    def fetch(mask):
        fetches["count"] += int(mask.sum())
        return store[mask]

    # zero prediction, positive cost: sigmoid(-0.35) < 0.5, so nothing is read
    mask, _, gate = router.select(hidden, metadata, cost=0.35, threshold=0.5)
    assert not bool(mask.any())
    fetch(mask)
    assert fetches["count"] == 0
    assert float(gate.max().detach()) < 0.5

    # a positive prediction opens every gate; only then does the fetch happen
    with torch.no_grad():
        router.predictor[-1].bias.fill_(0.8)
    mask, _, gate = router.select(hidden, metadata, cost=0.0, threshold=0.5)
    assert bool(mask.all())
    fetch(mask)
    assert fetches["count"] == 5
    assert float(gate.min().detach()) > 0.5


def test_post_read_gate_is_optional_and_cannot_run_without_the_read():
    hidden = torch.randn(1, 4, 6)
    read = torch.randn(1, 4, 6)
    metadata = _metadata(batch=1, positions=4)

    pre_only = routing.TwoStageUtilityRouter(hidden=6, use_post_read=False)
    mask, _, _ = pre_only.select(hidden, metadata, cost=0.0)
    pre_only(hidden, metadata, read, mask, cost=0.0)
    gate_a = pre_only.last_gate.clone()
    assert torch.equal(pre_only.last_gate, pre_only.last_pre_gate)
    # changing the read cannot change a pre-read-only gate
    pre_only(hidden, metadata, read * 1000.0, mask, cost=0.0)
    assert torch.equal(gate_a, pre_only.last_gate)

    # with a fitted post-read stage the read does move the prediction
    two_stage = routing.TwoStageUtilityRouter(hidden=6, use_post_read=True)
    mask, _, _ = two_stage.select(hidden, metadata, cost=0.0)
    with torch.no_grad():
        two_stage.post.predictor[0].weight.fill_(0.05)
        two_stage.post.predictor[2].weight.fill_(0.05)
    two_stage(hidden, metadata, read, mask, cost=0.0)
    prediction_one = two_stage.last_prediction.clone()
    two_stage(hidden, metadata, read * 3.0, mask, cost=0.0)
    assert not torch.allclose(prediction_one, two_stage.last_prediction,
                              atol=1e-9, rtol=1e-6)


def test_outcome_delta_is_a_paired_probability_difference():
    pairs = [
        {"rollout_id": "a", "base": {"verified": True},
         "memory": {"verified": True}},
        {"rollout_id": "b", "base": {"verified": False},
         "memory": {"verified": True}},
        {"rollout_id": "c", "base": {"verified": True},
         "memory": {"verified": False}},
        {"rollout_id": "d", "base": {"verified": False},
         "memory": {"verified": False}},
    ]
    summary = routing.outcome_delta(pairs)
    assert summary["pairs"] == 4
    assert summary["base_probability"] == pytest.approx(0.5)
    assert summary["memory_probability"] == pytest.approx(0.5)
    assert summary["delta"] == pytest.approx(0.0)

    helped = routing.outcome_delta([
        {"base": {"verified": False}, "memory": {"verified": True}},
        {"base": {"verified": False}, "memory": {"verified": True}},
    ])
    assert helped["delta"] == pytest.approx(1.0)
    hurt = routing.outcome_delta([
        {"base": {"verified": True}, "memory": {"verified": False}},
    ])
    assert hurt["delta"] == pytest.approx(-1.0)


def test_outcome_delta_rejects_absolute_labels_and_unpaired_rollouts():
    with pytest.raises(ValueError):
        routing.outcome_delta([])
    with pytest.raises(ValueError):
        routing.outcome_delta([{"base": True, "memory": False}])
    with pytest.raises(ValueError):
        routing.outcome_delta([{"base": {"verified": True}}])
    with pytest.raises(ValueError):
        routing.outcome_delta([
            {"base": {"verified": 0.9}, "memory": {"verified": True}}])
    with pytest.raises(ValueError):
        routing.outcome_delta([
            {"rollout_id": "a", "base": {"verified": True},
             "memory": {"verified": True}},
            {"rollout_id": "a", "base": {"verified": False},
             "memory": {"verified": False}},
        ])


def test_counterfactual_outcome_records_use_the_paired_delta():
    records = [{
        "prompt_id": "p1", "split": "fit", "token_count": 12,
        "features": {"h_t": [0.1], "r_t": [0.2]},
        "paired_rollouts": [
            {"base": {"verified": False}, "memory": {"verified": True}},
            {"base": {"verified": False}, "memory": {"verified": True}},
        ],
    }]
    labels = routing.labels_from_records(records, "counterfactual-outcome")
    assert labels[0]["label"] == pytest.approx(1.0)
    assert labels[0]["supervision"] == "counterfactual-outcome"
    assert labels[0]["pairs"] == 2
    assert labels[0]["token_count"] == 12

    # the absolute control's field is not accepted by the outcome route
    absolute = dict(records[0], verified=True)
    with pytest.raises(ValueError):
        routing.labels_from_records([absolute], "counterfactual-outcome")
    with pytest.raises(ValueError):
        routing.labels_from_records(
            [dict(records[0], teacher_logprob=-0.1)],
            "counterfactual-outcome")
    # the old absolute source is still a registered control
    control = routing.labels_from_records(
        [{"prompt_id": "p2", "split": "fit",
          "features": {"h_t": [0.1]}, "verified": True}],
        "verified-outcome")
    assert control[0]["label"] == 1.0


def test_register_lists_the_corrected_sources():
    assert "counterfactual-outcome" in routing.SUPERVISION_SOURCES
    assert "verified-outcome" in routing.SUPERVISION_SOURCES
    assert {"position_fraction", "budget_fraction"} == set(
        routing.PRE_READ_METADATA_KEYS)
    assert {"r_t", "read_norm", "gate_t"} <= set(routing.POST_READ_ONLY_KEYS)
