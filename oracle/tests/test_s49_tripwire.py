"""CPU tests for the S49 early-loss tripwire in `oracle/frontier/train.py`.

The tripwire is the registered stop for a full-backbone run that leaves the
initialised loss. These tests pin the three properties the registration names:
it is silent until both a full baseline and a full trailing window exist, it
does not fire on a stable series, and it fires when the trailing-window median
exceeds the registered factor of the baseline median.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import train  # noqa: E402


def test_tripwire_is_silent_without_a_full_window():
    history = [1.0] * 149
    assert train.tripwire_decision(history, 1.5, 100, 50) is None
    assert train.tripwire_decision([1.0] * 150, 1.5, 100, 50) is not None


def test_tripwire_is_disabled_by_a_zero_factor():
    assert train.tripwire_decision([1.0] * 1000, 0.0, 100, 50) is None
    assert train.tripwire_decision([1.0] * 1000, -1.0, 100, 50) is None


def test_tripwire_does_not_fire_on_a_stable_series():
    history = [3.0 + 0.01 * (index % 3) for index in range(400)]
    decision = train.tripwire_decision(history, 1.5, 100, 50)
    assert decision is not None
    assert decision["fired"] is False
    assert decision["ratio"] == pytest.approx(1.0, abs=0.01)


def test_tripwire_fires_when_the_trailing_median_doubles():
    history = [2.0] * 100 + [5.0] * 60
    decision = train.tripwire_decision(history, 1.5, 100, 50)
    assert decision is not None
    assert decision["fired"] is True
    assert decision["baseline_median_ce"] == pytest.approx(2.0)
    assert decision["trailing_median_ce"] == pytest.approx(5.0)
    assert decision["ratio"] == pytest.approx(2.5)


def test_tripwire_uses_the_baseline_not_the_whole_history():
    # A series that starts high and decays is not a divergence, even though its
    # trailing median is far below its own start.
    history = [8.0] * 100 + [3.0] * 200
    decision = train.tripwire_decision(history, 1.5, 100, 50)
    assert decision is not None
    assert decision["fired"] is False


def test_tripwire_fires_on_the_diverged_bar_trajectory():
    # The recorded 20M bar's shape: a warmup trough near 2.77, then a run above
    # 5.7 for the rest of training. The baseline (first 100) is about 3.81, so
    # 1.5x is 5.71 and the trailing window is above it from about step 150.
    history = (
        [2.8] * 50
        + [4.8] * 50
        + [6.0 + 0.2 * (index % 5) for index in range(200)]
    )
    decision = train.tripwire_decision(history, 1.5, 100, 50)
    assert decision is not None
    assert decision["fired"] is True
    assert decision["ratio"] > 1.5
