"""CPU tests for the S46 oracle-gate bounds.

The bounds are the decision input for the next gate, so their arithmetic has to
be checkable on data whose answer is known:

* the per-token oracle equals the per-position minimum, exactly;
* a gate that is uniformly right or uniformly wrong recovers nothing;
* the class oracle recovers a planted per-class structure in-sample and, when
  the structure exists only in the fitted half, the cross-fitted number does
  not inherit it;
* a feature that separates the helpful positions perfectly is recovered
  out-of-fold, and a feature that does not separate them is not;
* the recovery fraction is undefined (not zero) when memory helps overall;
* `_flatten` drops the control prompts and keeps positions aligned.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import s46_oracle_gate as gate  # noqa: E402


def row(split, name, base, memory, feature=None, position=0):
    return {
        "doc_id": 0,
        "split": split,
        "class": name,
        "position": position,
        "ce_base": base,
        "ce_memory": memory,
        "kl_base": base,
        "kl_memory": memory,
        "base_entropy": feature,
        "memory_residual_norm": feature,
    }


def test_per_token_oracle_is_the_minimum():
    rows = [row("A", "digit", 1.0, 0.5), row("A", "digit", 0.5, 1.5),
            row("B", "letter", 2.0, 2.5), row("B", "letter", 1.0, 0.25)]
    result = gate._per_token_oracle(rows, "ce")
    assert result["level"]["all_off"] == pytest.approx(1.125)
    assert result["level"]["all_on"] == pytest.approx(1.1875)
    assert result["value"] == pytest.approx((0.5 + 0.5 + 2.0 + 0.25) / 4)
    assert result["level"]["positions_memory_helps"] == 2
    assert result["gate_on_fraction"] == pytest.approx(0.5)
    damage = result["level"]["all_on"] - result["level"]["all_off"]
    assert result["recovery_fraction"] == pytest.approx(
        (result["level"]["all_on"] - result["value"]) / damage)


def test_uniform_gates_recover_the_whole_damage_or_nothing():
    # Memory uniformly worse: the oracle turns it off everywhere and removes
    # 100% of memory-always-on's net damage. That is the "suppress it" answer,
    # and the fraction is 1 by construction.
    worse = [row("A", "digit", 1.0, 1.5), row("B", "letter", 2.0, 2.5)]
    result = gate._per_token_oracle(worse, "ce")
    assert result["value"] == pytest.approx(result["level"]["all_off"])
    assert result["gate_on_fraction"] == 0.0
    assert result["recovery_fraction"] == pytest.approx(1.0)

    better = [row("A", "digit", 1.0, 0.5), row("B", "letter", 2.0, 1.5)]
    result = gate._per_token_oracle(better, "ce")
    assert result["value"] == pytest.approx(result["level"]["all_on"])
    assert result["gate_on_fraction"] == 1.0
    # Memory helps overall: the recovery fraction has no positive denominator
    # and is reported as undefined rather than as a number.
    assert result["recovery_defined"] is False
    assert result["recovery_fraction"] is None


def test_class_oracle_recovers_a_planted_structure():
    rows = []
    for _ in range(10):
        rows.append(row("A", "digit", 1.0, 0.5))
        rows.append(row("B", "digit", 1.0, 0.5))
        rows.append(row("A", "letter", 1.0, 3.0))
        rows.append(row("B", "letter", 1.0, 3.0))
    result = gate._class_oracle(rows, "ce")
    assert result["in_sample"]["on_classes"] == ["digit"]
    assert result["in_sample"]["value"] == pytest.approx(0.75)
    # The gate removes the letter damage *and* keeps the digit gain, so the
    # recovery can exceed 1: 1.75 -> 0.75 against a 0.75 damage denominator.
    assert result["in_sample"]["recovery_fraction"] == pytest.approx(4 / 3)
    # Both folds see the same structure, so the cross-fitted number matches.
    assert result["cross_fitted"]["value"] == pytest.approx(0.75)
    assert result["cross_fitted"]["recovery_fraction"] == pytest.approx(4 / 3)


def test_cross_fitting_does_not_leak_a_fold_local_structure():
    # Digits help in fold A and are slightly hurt in fold B, so the pooled
    # decision is "on" while a B-fitted gate turns them off. Letters hurt
    # everywhere. The A-fit gate is therefore wrong on fold B's digits, and the
    # B-fit gate leaves fold A entirely off.
    rows = []
    for _ in range(10):
        rows.append(row("A", "digit", 1.0, 0.5))
        rows.append(row("B", "digit", 1.0, 1.1))
        rows.append(row("A", "letter", 1.0, 2.0))
        rows.append(row("B", "letter", 1.0, 2.0))
    result = gate._class_oracle(rows, "ce")
    assert result["in_sample"]["on_classes"] == ["digit"]
    a_to_b = result["cross_fitted"]["folds"]["A->B"]
    b_to_a = result["cross_fitted"]["folds"]["B->A"]
    assert a_to_b["on_classes"] == ["digit"]
    # Fold B's digits are hurt by memory, so the A-fit gate is worse than
    # leaving memory off on fold B.
    assert a_to_b["value"] == pytest.approx(1.05)
    assert b_to_a["on_classes"] == []
    assert b_to_a["value"] == pytest.approx(1.0)
    assert result["cross_fitted"]["value"] == pytest.approx(1.025)


def test_feature_oracle_recovers_a_perfect_separator_out_of_fold():
    rows = []
    for index in range(20):
        # The helpful positions share the same feature range in both folds, so
        # a threshold fitted on either half separates the evaluation half.
        rows.append(row("A", "digit", 1.0, 0.5, feature=10.0 + index))
        rows.append(row("B", "digit", 1.0, 0.5, feature=10.0 + index))
        rows.append(row("A", "letter", 1.0, 2.0, feature=float(index % 5)))
        rows.append(row("B", "letter", 1.0, 2.0, feature=float(index % 5)))
    result = gate._feature_oracle(rows, "base_entropy", "ce")
    assert result["value"] == pytest.approx(0.75)
    for name, fold in result["folds"].items():
        assert fold["direction"] == "high"
        assert fold["value"] == pytest.approx(0.75)
    # 1.0 -> 1.25 net under memory-always-on, and the gate reaches 0.75, i.e.
    # it removes twice the net damage: the letter damage and the digit gain.
    assert result["recovery_fraction"] == pytest.approx(2.0)


def test_feature_oracle_finds_no_separation_when_there_is_none():
    rows = []
    for index in range(20):
        # The feature has no relation to which branch is better.
        rows.append(row("A", "digit", 1.0, 0.5, feature=float(index % 3)))
        rows.append(row("B", "letter", 1.0, 2.0, feature=float(index % 3)))
    result = gate._feature_oracle(rows, "base_entropy", "ce")
    # Whatever threshold is fitted, it cannot beat the per-token 0.75 without
    # the labels; out of fold it must be no better than the best constant
    # policy on the evaluation half, which is all-off at 1.0.
    assert result["value"] >= 1.0 - 1e-12


def test_flatten_drops_controls_and_keeps_alignment():
    items = [{
        "doc_id": 7,
        "split": "A",
        "control": False,
        "positions": {
            "class": ["digit", "space"],
            "position": [0, 1],
            "ce_base": [1.0, 2.0],
            "ce_memory": [0.5, 3.0],
            "kl_base_teacher": [0.1, 0.2],
            "kl_memory_teacher": [0.05, 0.3],
            "base_entropy": [4.0, 5.0],
            "memory_residual_norm": [0.7, 0.8],
        },
    }, {
        "doc_id": 8,
        "split": "B",
        "control": True,
        "positions": {
            "class": ["letter"],
            "position": [0],
            "ce_base": [1.0],
            "ce_memory": [1.0],
            "kl_base_teacher": [0.0],
            "kl_memory_teacher": [0.0],
            "base_entropy": [1.0],
            "memory_residual_norm": None,
        },
    }]
    rows = gate._flatten(items, damage_only=True)
    assert len(rows) == 2
    assert [r["class"] for r in rows] == ["digit", "space"]
    assert rows[1]["ce_memory"] == 3.0
    assert rows[1]["memory_residual_norm"] == 0.8
    all_rows = gate._flatten(items, damage_only=False)
    assert len(all_rows) == 3
    assert all_rows[2]["memory_residual_norm"] is None


def test_class_gate_reads_the_token_at_the_gated_position():
    """The gate on hidden position t must use token t, not token t-1.

    A one-token lag puts the gate on the wrong token class, which is the exact
    difference between "apply the memory while writing a number" and "apply it
    one token too early".
    """

    torch = pytest.importorskip("torch")

    class _Tok:
        table = {1: "5", 2: "x"}

        def decode(self, ids):
            return self.table[ids[0]]

    tok = _Tok()
    # The call covers the last position only; that token is the digit.
    on = gate._gate_tensor(torch, "class", {"on_classes": ["digit"]}, tok,
                           [2, 1], [0.0])
    assert float(on.reshape(-1)[0]) == 1.0
    # Same call, last token is a letter -> off.
    off = gate._gate_tensor(torch, "class", {"on_classes": ["digit"]}, tok,
                            [1, 2], [0.0])
    assert float(off.reshape(-1)[0]) == 0.0
