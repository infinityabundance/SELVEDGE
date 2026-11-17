"""CPU tests for the S49 stability decision and its table.

The registered rule is the only thing that may choose a learning rate. These
tests pin it: an arm that trips the tripwire is ineligible, an arm whose
held-out CE is worse than base + 0.01 nats is ineligible, the highest eligible
learning rate wins, and an empty eligible set requires the second factor.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import s49_bar_curve as curve  # noqa: E402
import s49_stability_table as sweep  # noqa: E402


def make_log(trailing_ce: float, steps: int = 976) -> list[dict]:
    rows = []
    for step in range(1, steps + 1):
        ce = 3.0 if step <= 100 else trailing_ce
        rows.append({"step": step, "tokens_seen": step * 2048,
                     "ce": ce, "tokens_per_second": 1850.0,
                     "peak_allocated_gib": 11.4})
    return rows


def make_row(arm: str, lr: float, trailing: float, delta: float | None,
             aborted: bool = False) -> dict:
    stats = sweep.log_stats(make_log(trailing))
    return {
        "arm": arm,
        "lr": lr,
        "run_dir": f"results/stage2/runs/{arm}",
        "complete": True,
        "aborted": aborted,
        "log_stats": stats,
        "heldout": None if delta is None else {
            "ce_mean": 2.81 + delta,
            "delta_ce": delta,
            "delta_ce_ci95": [delta - 0.01, delta + 0.01],
            "kl_mean": 0.27,
        },
    }


def test_log_stats_measures_the_registered_baseline_and_window():
    stats = sweep.log_stats(make_log(3.0))
    assert stats["median_ce_first_100"] == pytest.approx(3.0)
    assert stats["median_ce_trailing_50"] == pytest.approx(3.0)
    assert stats["trailing_ratio"] == pytest.approx(1.0)
    assert stats["first_step_trailing_over_factor"] is None


def test_log_stats_finds_the_first_crossing():
    rows = make_log(3.0)
    for row in rows[100:]:
        row["ce"] = 6.0
    stats = sweep.log_stats(rows)
    assert stats["trailing_ratio"] == pytest.approx(2.0)
    assert stats["first_step_trailing_over_factor"] == 150


def test_the_highest_eligible_learning_rate_wins():
    rows = [
        make_row("low", 5e-06, 3.0, 0.001),
        make_row("mid", 1e-05, 3.0, 0.002),
        make_row("high", 3e-05, 3.0, 0.003),
    ]
    decision = sweep.decide(rows)
    assert decision["chosen"]["arm"] == "high"
    assert decision["second_factor_required"] is False


def test_a_tripped_arm_is_ineligible_even_at_the_highest_lr():
    rows = [
        make_row("high", 3e-05, 6.0, 0.001, aborted=True),
        make_row("mid", 1e-05, 3.0, 0.001),
    ]
    for row in rows:
        row["evaluation"] = sweep.evaluate(row)
    decision = sweep.decide(rows)
    assert decision["chosen"]["arm"] == "mid"


def test_a_held_out_regression_disqualifies_an_arm():
    rows = [
        make_row("high", 3e-05, 3.0, 0.02),
        make_row("mid", 1e-05, 3.0, 0.005),
    ]
    decision = sweep.decide(rows)
    assert decision["chosen"]["arm"] == "mid"
    reasons = [reason for row in rows if row["arm"] == "high"
               for reason in row["evaluation"]["reasons"]]
    assert any("held-out CE" in reason for reason in reasons)


def test_no_eligible_arm_requires_the_second_factor():
    rows = [make_row("high", 3e-05, 3.0, 0.05),
            make_row("low", 5e-06, 6.0, 0.001, aborted=True)]
    decision = sweep.decide(rows)
    assert decision["chosen"] is None
    assert decision["second_factor_required"] is True


def test_missing_held_out_receipt_is_not_eligible():
    row = make_row("only", 1e-05, 3.0, None)
    assert sweep.evaluate(row)["eligible"] is False


def test_bar_slope_is_positive_when_the_curve_rises():
    trend = curve.slope([(6_144_000, 30.0), (10_240_000, 31.0),
                         (19_998_720, 33.0)])
    assert trend["slope_per_million_tokens"] > 0
    flat = curve.slope([(6_144_000, 30.0), (10_240_000, 30.0),
                        (19_998_720, 30.0)])
    assert flat["slope_per_million_tokens"] == pytest.approx(0.0)
