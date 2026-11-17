"""P19.9 CPU tests: counterfactual utility routing skeleton.

The route predicts `Delta_t = l_base - l_{base+mem}` from causal features and
gates the residual by `sigmoid((Delta_hat - cost) / tau)`. These tests check the
label arithmetic, the gate's sign, the cost monotonicity and the leakage rules.
No training or generation runs.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import utility_routing as routing  # noqa: E402


def test_counterfactual_improvement_has_the_registered_sign():
    base = torch.tensor([2.0, 2.0, 2.0])
    memory = torch.tensor([1.5, 2.0, 2.5])
    delta = routing.counterfactual_improvement(base, memory)
    assert delta.tolist() == pytest.approx([0.5, 0.0, -0.5])
    # positive where the memory lowers the loss, negative where it raises it
    assert delta[0] > 0 and delta[2] < 0


def test_gate_is_above_half_exactly_when_predicted_improvement_beats_cost():
    router = routing.UtilityRouter(hidden=4, tau=1.0)
    predicted = torch.tensor([0.5, 0.0, -0.5])
    gate = router.gate(predicted, cost=0.0)
    assert gate[0] > 0.5
    assert gate[1] == pytest.approx(0.5)
    assert gate[2] < 0.5
    # cost moves the gate down monotonically
    higher = router.gate(predicted, cost=0.25)
    assert torch.all(higher < gate)


def test_router_is_zero_at_init_and_gates_the_read():
    torch.manual_seed(7)
    router = routing.UtilityRouter(hidden=8, tau=0.5)
    hidden = torch.randn(1, 5, 8)
    read = torch.randn(1, 5, 8)
    predicted = router.predict(hidden, read)
    assert torch.equal(predicted, torch.zeros_like(predicted))
    output = router(hidden, read, cost=0.0)
    assert output.shape == read.shape
    assert torch.equal(router.last_gate, torch.full_like(router.last_gate, 0.5))


def test_measured_cost_is_monotone_and_non_negative():
    read = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    cheap = routing.measured_cost(read, price_per_element=1.0)
    dear = routing.measured_cost(read, price_per_element=2.0)
    assert torch.all(dear >= cheap)
    with pytest.raises(ValueError):
        routing.measured_cost(read, price_per_element=-1.0)


def _record(prompt_id="p1", split="fit", **extra):
    record = {
        "prompt_id": prompt_id,
        "split": split,
        "features": {"h_t": [0.1], "r_t": [0.2]},
    }
    record.update(extra)
    return record


def test_counterfactual_labels_are_token_loss_differences():
    records = [
        _record(base_loss=2.0, memory_loss=1.5),
        _record(prompt_id="p2", base_loss=2.0, memory_loss=2.5),
    ]
    labels = routing.labels_from_records(records, "counterfactual-token-loss")
    assert [item["label"] for item in labels] == pytest.approx([0.5, -0.5])
    assert all(item["supervision"] == "counterfactual-token-loss"
               for item in labels)


def test_verified_outcome_is_sequence_level_and_rejects_likelihoods():
    records = [
        _record(verified=True, token_count=12),
        _record(prompt_id="p2", verified=False, token_count=7),
    ]
    labels = routing.labels_from_records(records, "verified-outcome")
    assert [item["label"] for item in labels] == [1.0, 0.0]
    assert [item["token_count"] for item in labels] == [12, 7]
    with pytest.raises(ValueError):
        routing.labels_from_records(
            [_record(verified=True, teacher_logprob=-0.1)], "verified-outcome")
    with pytest.raises(ValueError):
        routing.labels_from_records(
            [_record(verified=0.9)], "verified-outcome")


def test_leakage_rules_reject_label_features_and_split_overlap():
    with pytest.raises(ValueError):
        routing.labels_from_records(
            [_record(base_loss=1.0, memory_loss=0.5,
                     features={"h_t": [0.1], "target_id": 7})],
            "counterfactual-token-loss")
    with pytest.raises(ValueError):
        routing.labels_from_records(
            [_record(base_loss=1.0, memory_loss=0.5),
             _record(prompt_id="p1", split="holdout",
                     base_loss=1.0, memory_loss=0.5)],
            "counterfactual-token-loss")


def test_causal_feature_allow_list_is_explicit():
    assert "h_t" in routing.CAUSAL_FEATURE_KEYS
    assert "r_t" in routing.CAUSAL_FEATURE_KEYS
    assert "target_id" in routing.FORBIDDEN_FEATURE_KEYS
    assert "y_t" in routing.FORBIDDEN_FEATURE_KEYS
