#!/usr/bin/env python3
"""Power arithmetic for the registered capability composite (P14.4).

The registered estimand is the category-balanced mean of paired, per-item
differences, with categories weighted equally. For a paired design the variance
of that composite is

    Var(composite) = Var(d) / (categories * n_per_category)

so the minimum detectable effect (MDE) at a two-sided level and a target power
solves

    E[d] = (z_{1-alpha/2} + z_{power}) * sqrt(Var(d) / (categories * n))

with `d` the per-item paired difference. A favourable model assumes only gains
(d in {0, +1}, P(+1) = q, so E[d] = q and Var = q(1-q)); a pessimistic model
reverses a fraction of the gains (P(-1) = r*q). Both are reported. The normal
approximation is stated rather than assumed exact, and `simulate_power` checks
the resulting MDE against the *registered* paired category bootstrap, so the
analytic number is not the only evidence.

The threshold itself is not touched: `REGISTERED_RELATIVE_THRESHOLD` is the
contract's 0.10 and is imported, not chosen here.
"""

from __future__ import annotations

import math
import random
import statistics

from . import stats

REGISTERED_RELATIVE_THRESHOLD = 0.10
SUITE_CATEGORIES = 8
SUITE_ITEMS = 144


def z_value(probability: float) -> float:
    return statistics.NormalDist().inv_cdf(probability)


def marginal_ci_half_width(n: int, proportion: float = 0.5, level: float = 0.95) -> float:
    """Half-width of a single-proportion 95% CI at `n` items.

    This is what a *per-category* row can say about one category's success rate
    on its own; the paired composite CI is narrower because the comparison is
    paired. Both are reported, never substituted for one another.
    """
    if not 0.0 < proportion < 1.0:
        raise ValueError("proportion must be in (0, 1)")
    return z_value(1.0 - (1.0 - level) / 2.0) * math.sqrt(proportion * (1.0 - proportion) / n)


def _variance(q: float, regression_fraction: float) -> float:
    mean = (1.0 - regression_fraction) * q
    second = q * (1.0 + regression_fraction)
    return max(0.0, second - mean * mean)


def _solve_q(target: float, regression_fraction: float, n_effective: int) -> float:
    """Solve mean(q) = target * sqrt(Var(q)) by bisection on (0, 1)."""
    low, high = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        middle = (low + high) / 2.0
        value = (1.0 - regression_fraction) * middle - target * math.sqrt(
            _variance(middle, regression_fraction) / n_effective)
        if value < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def composite_mde(n_per_category: int = SUITE_ITEMS // SUITE_CATEGORIES,
                  categories: int = SUITE_CATEGORIES,
                  level: float = 0.95, power: float = 0.8,
                  regression_fraction: float = 0.0,
                  baseline: float = 0.5) -> dict:
    if n_per_category < 1 or categories < 1:
        raise ValueError("n_per_category and categories must be positive")
    if not 0.0 <= regression_fraction < 1.0:
        raise ValueError("regression_fraction must be in [0, 1)")
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must be in (0, 1)")
    n_effective = n_per_category * categories
    z_total = z_value(1.0 - (1.0 - level) / 2.0) + z_value(power)
    q = _solve_q(z_total, regression_fraction, n_effective)
    absolute = (1.0 - regression_fraction) * q
    variance = _variance(q, regression_fraction)
    return {
        "absolute": absolute,
        "relative_to_baseline": absolute / baseline,
        "gain_probability": q,
        "paired_standard_deviation": math.sqrt(variance),
        "level": level,
        "power": power,
        "regression_fraction": regression_fraction,
        "baseline": baseline,
        "n_per_category": n_per_category,
        "categories": categories,
        "n_effective": n_effective,
        "approximation": "normal approximation to the paired composite mean",
    }


def simulate_power(absolute_effect: float, n_per_category: int = SUITE_ITEMS // SUITE_CATEGORIES,
                   categories: int = SUITE_CATEGORIES, regression_fraction: float = 0.0,
                   level: float = 0.95, replications: int = 120, resamples: int = 400,
                   seed: int = stats.SEED) -> dict:
    """Check an MDE against the registered bootstrap by simulation.

    Each replication draws paired item outcomes with the requested effect and
    runs `stats.paired_category_bootstrap` exactly as the registered comparison
    does, then counts how often the 95% CI lower bound is above zero and the
    relative improvement reaches the registered threshold.
    """
    if absolute_effect <= 0 or absolute_effect >= 1:
        raise ValueError("absolute_effect must be in (0, 1)")
    gain = absolute_effect / max(1e-12, 1.0 - regression_fraction)
    loss = gain * regression_fraction
    neutral = 1.0 - gain - loss
    if neutral < -1e-9:
        raise ValueError("effect too large for the regression fraction")
    rng = random.Random(seed)
    ci_hits = 0
    threshold_hits = 0
    observed = []
    for _ in range(replications):
        arm: dict[str, list] = {}
        baseline: dict[str, list] = {}
        for category in range(categories):
            name = f"c{category}"
            arm[name] = []
            baseline[name] = []
            for _item in range(n_per_category):
                draw = rng.random()
                if draw < gain:
                    arm[name].append(1.0)
                    baseline[name].append(0.0)
                elif draw < gain + loss:
                    arm[name].append(0.0)
                    baseline[name].append(1.0)
                else:
                    arm[name].append(0.0)
                    baseline[name].append(0.0)
        result = stats.paired_category_bootstrap(arm, baseline, level=level,
                                                 resamples=resamples)
        observed.append(result["mean_difference"])
        if result["ci_low"] > 0:
            ci_hits += 1
            relative = result["relative_difference"]
            if relative is None or relative >= REGISTERED_RELATIVE_THRESHOLD:
                threshold_hits += 1
    return {
        "absolute_effect": absolute_effect,
        "replications": replications,
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "regression_fraction": regression_fraction,
        "ci_power": ci_hits / replications,
        "decision_power": threshold_hits / replications,
        "mean_observed_difference": sum(observed) / len(observed),
        "method": "registered paired category-stratified bootstrap",
    }
