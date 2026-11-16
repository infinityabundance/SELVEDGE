"""CPU tests for the capability court's statistics.

The percentile rule is compared against numpy, which exists in the python court
but not in the rust court where the checker runs; the point is that the two
definitions agree exactly, so the stdlib implementation is not a second
estimand.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from capability import stats  # noqa: E402


def test_percentile_matches_numpy_linear():
    rng = np.random.default_rng(7)
    sample = sorted(rng.normal(size=257).tolist())
    for q in (0.0, 0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999, 1.0):
        assert abs(stats.percentile(sample, q) - float(np.percentile(sample, q * 100))
                   ) < 1e-12


def test_bootstrap_mean_is_deterministic_and_centred():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    first = stats.bootstrap_mean(values, resamples=2000)
    second = stats.bootstrap_mean(values, resamples=2000)
    assert first == second
    assert first["mean"] == pytest.approx(sum(values) / len(values))
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_bootstrap_interval_narrows_with_more_items():
    small = stats.bootstrap_mean([1.0] * 4 + [0.0] * 4, resamples=2000)
    large = stats.bootstrap_mean([1.0] * 200 + [0.0] * 200, resamples=2000)
    assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


def test_paired_bootstrap_uses_paired_differences():
    a = [1, 1, 0, 1]
    b = [0, 0, 0, 1]
    result = stats.paired_bootstrap(a, b, resamples=2000)
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["n"] == 4


def test_composite_is_category_balanced_not_item_pooled():
    unbalanced = {"a": [1.0] * 10, "b": [0.0, 0.0]}
    assert stats.composite(unbalanced) == pytest.approx(0.5)
    pooled = sum(value for values in unbalanced.values() for value in values) / 12
    assert pooled != pytest.approx(0.5)


def test_composite_refuses_a_missing_category():
    assert stats.composite({}) is None
    with pytest.raises(ValueError):
        stats.stratified_mean({})


def test_paired_category_bootstrap_weights_categories_equally():
    arm = {"a": [1.0] * 10, "b": [1.0, 1.0]}
    base = {"a": [0.0] * 10, "b": [0.0, 0.0]}
    result = stats.paired_category_bootstrap(arm, base, resamples=2000)
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["arm_composite"] == pytest.approx(1.0)
    assert result["ci_low"] > 0


def test_paired_category_bootstrap_rejects_unpaired_input():
    with pytest.raises(ValueError):
        stats.paired_category_bootstrap({"a": [1.0]}, {"a": [1.0, 0.0]})
    with pytest.raises(ValueError):
        stats.paired_category_bootstrap({"a": [1.0]}, {"b": [1.0]})
