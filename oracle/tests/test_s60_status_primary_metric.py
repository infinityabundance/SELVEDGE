"""S60 — the status block must report the receipt's declared primary metric.

`results/stage2/eval/s50-bar-20m-lr1e-5/m20/eval.json` carries both
`exact_match,none` (0.088) and `math_verify,none` (0.186) for
`minerva_math500`, and declares `primary_metric = math_verify,none`. The status
generator's preference list put `exact_match` first, so the README reported
8.8% where the registered primary is 18.6%. These tests pin the fix and require
the secondary metric to be labelled, not silently dropped.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import status_consistency as sc  # noqa: E402


def _receipt(primary_metric="math_verify,none", declare=True) -> dict:
    task = {
        "task": "minerva_math500",
        "all_metrics": {
            "exact_match,none": 0.088,
            "math_verify,none": 0.186,
        },
    }
    if declare:
        task["primary_metric"] = primary_metric
    return {"tasks": [task]}


def test_declared_primary_metric_wins_over_the_preference_list():
    receipt = _receipt()
    assert sc._milestone_score(receipt, "minerva_math500") == 0.186
    metric = sc._milestone_metric(receipt, "minerva_math500")
    assert metric["primary_metric"] == "math_verify,none"
    assert metric["secondary"] == {"exact_match,none": 0.088}


def test_declared_primary_is_used_even_if_it_is_not_first_in_the_list():
    receipt = _receipt(primary_metric="exact_match,none")
    metric = sc._milestone_metric(receipt, "minerva_math500")
    assert metric["primary_metric"] == "exact_match,none"
    assert metric["primary"] == 0.088
    assert metric["secondary"] == {"math_verify,none": 0.186}


def test_a_receipt_without_a_declared_primary_falls_back_to_math_verify():
    receipt = _receipt(declare=False)
    assert sc._milestone_score(receipt, "minerva_math500") == 0.186


def test_summary_labels_the_metric_and_the_secondary_number():
    milestone = {
        "label": "m20",
        "step": 20,
        "tokens": 100,
        "stamp": "20260104T000020Z",
        "metrics": {
            "minerva_math500": sc._milestone_metric(
                _receipt(), "minerva_math500"),
        },
        "scores": {"minerva_math500": 0.186},
    }
    line = sc._milestone_summary([milestone])
    assert "math_verify,none" in line
    assert "18.6%" in line
    assert "secondary" in line
    assert "exact_match,none" in line
    assert "8.8%" in line
