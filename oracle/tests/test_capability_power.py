"""CPU tests for the capability power arithmetic (P14.4).

The analytic MDE is a normal approximation; these tests pin its behaviour and
check it against the registered bootstrap at the registered n. They run in the
python court (standard library only).
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from capability import power, stats  # noqa: E402


def test_the_marginal_ci_matches_the_closed_form():
    half = power.marginal_ci_half_width(18, proportion=0.5, level=0.95)
    expected = 1.959963984540054 * math.sqrt(0.25 / 18)
    assert abs(half - expected) < 1e-12


def test_a_larger_n_narrows_the_marginal_ci():
    assert power.marginal_ci_half_width(9) > power.marginal_ci_half_width(18)


def test_the_mde_shrinks_with_more_items():
    small = power.composite_mde(n_per_category=9, categories=8)
    large = power.composite_mde(n_per_category=18, categories=8)
    assert large["absolute"] < small["absolute"]
    assert large["n_effective"] == 144


def test_the_mde_is_the_effect_whose_ci_lower_bound_sits_at_zero():
    result = power.composite_mde(n_per_category=18, categories=8)
    z_total = power.z_value(1 - 0.025) + power.z_value(0.8)
    implied = z_total * result["paired_standard_deviation"] / math.sqrt(144)
    assert abs(result["absolute"] - implied) < 1e-9


def test_regressions_widen_the_mde():
    optimistic = power.composite_mde(n_per_category=18, regression_fraction=0.0)
    pessimistic = power.composite_mde(n_per_category=18, regression_fraction=0.2)
    assert pessimistic["absolute"] > optimistic["absolute"]


def test_the_registered_threshold_is_imported_not_chosen():
    assert power.REGISTERED_RELATIVE_THRESHOLD == 0.10
    assert stats.LEVEL == 0.95
    assert stats.RESAMPLES == 10000
    assert stats.SEED == 20261115


def test_the_simulation_power_is_near_the_requested_power():
    mde = power.composite_mde(n_per_category=18, categories=8)
    result = power.simulate_power(mde["absolute"], n_per_category=18, categories=8,
                                  replications=60, resamples=200)
    assert 0.6 <= result["ci_power"] <= 1.0
    assert result["mean_observed_difference"] > 0


def test_a_zero_effect_is_refused():
    for call in (lambda: power.simulate_power(0.0),
                 lambda: power.composite_mde(n_per_category=0),
                 lambda: power.composite_mde(regression_fraction=1.0)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a degenerate input was accepted")
