"""P11.1 — CPU tests for the causal repair court.

The fixture (`TinyLayeredStudent`) has one non-identity block: it deletes the
class subspace at a planted layer. Every assertion below is fixture arithmetic:
the bottleneck is known by construction, the oracle's correction is a scaled
class direction, and the controls see the same items and tokens. No 0.6B number
is claimed anywhere in this file.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORACLE, "eval"))

import causal_repair as repair  # noqa: E402


def test_the_fixture_student_fails_every_item_and_deletes_the_class_subspace():
    student = repair.TinyLayeredStudent(seed=1001)
    items = repair.fixture_items(student)
    verifier = repair.Verifier()
    for item in items:
        assert repair.verify(student, item, None, verifier)["correct"] is False
    hidden = student.hidden_state(items[0].input_ids.unsqueeze(0),
                                  student.bottleneck)
    assert float(hidden[0, -1, :student.classes].detach().abs().max()) == 0.0
    before = student.hidden_state(items[0].input_ids.unsqueeze(0),
                                  student.bottleneck - 1)
    assert float(before[0, -1, :student.classes].detach().abs().max()) == 1.0


def test_the_oracle_finds_the_planted_bottleneck_layer_and_stays_in_budget():
    student = repair.TinyLayeredStudent(seed=1001)
    items = repair.failed_items(student, repair.fixture_items(student))
    epsilon = 2.0
    report = repair.oracle_repair(student, items, epsilon=epsilon, seed=1001)
    assert report["repaired"] == len(items)
    assert report["bottleneck_histogram"] == {str(student.bottleneck): len(items)}
    for row in report["per_item"].values():
        assert row["bottleneck_layer"] == student.bottleneck
        for layer, result in row["layers"].items():
            assert result["delta_norm"] <= epsilon + 1e-5
        # every earlier layer fails: the deletion is downstream of it
        for layer in range(student.bottleneck):
            assert row["layers"][layer]["success"] is False


def test_the_verifier_driven_variant_repairs_without_gradients():
    student = repair.TinyLayeredStudent(seed=1001)
    items = repair.failed_items(student, repair.fixture_items(student))[:8]
    report = repair.oracle_repair(student, items, epsilon=2.0, variant="verifier",
                                  seed=1001)
    assert report["repaired"] >= len(items) // 2
    assert all(row["layers"][student.bottleneck]["variant"] == "verifier"
               for row in report["per_item"].values())


def test_the_deployed_layer_is_the_median_earliest_success():
    student = repair.TinyLayeredStudent(seed=1001)
    items = repair.failed_items(student, repair.fixture_items(student))
    report = repair.oracle_repair(student, items, epsilon=2.0)
    assert repair.choose_deployed_layer(report) == student.bottleneck


def test_the_predictor_learns_the_repair_and_the_controls_do_not():
    receipt = repair.run_fixture(seed=1001)
    assert receipt["oracle_test"]["repaired"] == receipt["items"]["test"]
    assert receipt["learned"]["repaired"] == receipt["items"]["test"]
    assert receipt["capture"]["capture"] == pytest.approx(1.0)
    assert receipt["verdict"]["killed"] is False
    controls = receipt["controls"]
    # In this fixture the bottleneck destroys the only class-carrying hidden
    # signal, so an adaptation that reads hidden states alone cannot repair.
    assert controls["none"]["repaired"] == 0
    assert controls["distributed_lora_equal"]["repaired"] == 0
    assert controls["all_layer_adapters"]["repaired"] == 0


def test_the_equal_budget_control_is_actually_matched():
    receipt = repair.run_fixture(seed=1001)
    matching = receipt["parameter_matching"]
    assert matching["distributed_lora_parameters"] <= matching["predictor"]
    assert matching["distributed_lora_parameters"] > 0
    # the next rank up would exceed the predictor's budget
    student = receipt["student"]
    next_rank = matching["distributed_lora_rank"] + 1
    assert 2 * student["layers"] * student["hidden"] * next_rank \
        > matching["predictor"]
    assert matching["all_layer_rank"] == repair.ALL_LAYER_RANK
    assert matching["all_layer_parameters"] == (
        2 * student["layers"] * student["hidden"] * repair.ALL_LAYER_RANK)
    # the control sees the same items/tokens/steps as the predictor
    lora = receipt["controls"]["distributed_lora_equal"]["training"]
    predictor = receipt["predictor_training"]
    assert lora["steps"] == predictor["steps"]
    assert lora["tokens"] == predictor["tokens"]


def test_the_predictor_never_reads_the_label():
    student = repair.TinyLayeredStudent(seed=1001)
    predictor = repair.CausalRepairPredictor(student.hidden, student.vocab,
                                             student.layers)
    tokens = torch.tensor([0, 0, 0, 0, 2])
    item_a = repair.RepairItem("a", tokens, label=1)
    item_b = repair.RepairItem("b", tokens, label=3)
    first = predictor.predict(student, item_a, student.bottleneck, 2.0)
    second = predictor.predict(student, item_b, student.bottleneck, 2.0)
    assert torch.equal(first["delta"], second["delta"])


def test_the_fixture_is_deterministic():
    first = repair.run_fixture(seed=1001)
    second = repair.run_fixture(seed=1001)
    for key in ("learned", "capture", "oracle_test", "deployed_layer"):
        assert first[key] == second[key]


def test_the_transfer_gap_is_reported_on_a_structurally_different_student():
    receipt = repair.run_fixture(seed=1001)
    transfer = receipt["transfer"]
    assert transfer["target"].startswith("student with bottleneck layer 2")
    assert transfer["oracle"]["repaired"] == 12
    assert transfer["capture"]["capture"] >= repair.TRANSFER_CAPTURE_KILL


def test_the_kill_condition_is_mechanical():
    weak = repair.kill_condition(
        {"capture": 0.4, "learned": 4, "oracle": 10, "baseline": 0})
    assert weak["killed"] is True
    assert "capture" in weak["reasons"][0]
    failed_transfer = repair.kill_condition(
        {"capture": 0.9}, {"capture": 0.1})
    assert failed_transfer["killed"] is True
    passed = repair.kill_condition({"capture": 0.9},
                                   {"capture": repair.TRANSFER_CAPTURE_KILL})
    assert passed["killed"] is False
    # a zero denominator is reported as null, never as a perfect capture
    assert repair.capture({"repaired": 0}, {"repaired": 0})["capture"] is None
