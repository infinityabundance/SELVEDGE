#!/usr/bin/env python3
"""Paired and category-stratified bootstrap, standard library only.

Why not numpy: the check stage runs in the rust court, which has `cargo` and a
bare Python. A second implementation would be a second definition of the
confidence interval, so this module is the only one; `oracle/tests/` compares
its percentile rule against numpy inside the python court, where numpy exists.

The registered method (P12.11) is a paired percentile bootstrap over items,
resampling within each capability category and weighting categories equally, so
a suite with more items in one category cannot move the composite by size alone.
"""

from __future__ import annotations

import math
import random

SEED = 20261115
RESAMPLES = 10_000
LEVEL = 0.95


def percentile(sorted_values: list, q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default `linear`.

    `q` is a fraction in [0, 1].
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _indices(rng: random.Random, n: int, resamples: int) -> list[list[int]]:
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def bootstrap_mean(values: list, seed: int = SEED, resamples: int = RESAMPLES,
                   level: float = LEVEL) -> dict:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("bootstrap_mean needs a non-empty sample")
    rng = random.Random(seed)
    draws = []
    indexes = _indices(rng, len(values), resamples)
    for draw in indexes:
        draws.append(sum(values[index] for index in draw) / len(values))
    draws.sort()
    alpha = (1.0 - level) / 2.0
    return {
        "mean": sum(values) / len(values),
        "ci_low": percentile(draws, alpha),
        "ci_high": percentile(draws, 1.0 - alpha),
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "n": len(values),
    }


def paired_bootstrap(a: list, b: list, seed: int = SEED, resamples: int = RESAMPLES,
                     level: float = LEVEL) -> dict:
    """Mean of (a - b) over paired items, with a percentile CI."""
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs equal non-empty samples")
    differences = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    draws = []
    indexes = _indices(rng, len(differences), resamples)
    for draw in indexes:
        draws.append(sum(differences[index] for index in draw) / len(differences))
    draws.sort()
    alpha = (1.0 - level) / 2.0
    return {
        "mean_difference": sum(differences) / len(differences),
        "ci_low": percentile(draws, alpha),
        "ci_high": percentile(draws, 1.0 - alpha),
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "n": len(differences),
    }


def category_means(by_category: dict) -> dict:
    means = {}
    for category in sorted(by_category):
        values = [float(value) for value in by_category[category]]
        if values:
            means[category] = sum(values) / len(values)
    return means


def composite(by_category: dict) -> float | None:
    """Equal-weight mean over categories; None if any category is empty.

    A missing category is not silently dropped: the composite would then be a
    different estimand from the registered one.
    """
    means = category_means(by_category)
    if not means:
        return None
    return sum(means.values()) / len(means)


def paired_category_bootstrap(arm: dict, baseline: dict, seed: int = SEED,
                              resamples: int = RESAMPLES,
                              level: float = LEVEL) -> dict:
    """Category-balanced paired bootstrap of arm minus baseline.

    `arm` and `baseline` map category -> per-item values in the same order.
    """
    categories = sorted(set(arm) | set(baseline))
    if not categories:
        raise ValueError("paired_category_bootstrap needs at least one category")
    for category in categories:
        if category not in arm or category not in baseline:
            raise ValueError(f"category {category!r} is missing from one arm")
        if len(arm[category]) != len(baseline[category]):
            raise ValueError(f"category {category!r} is not paired by item")
        if not arm[category]:
            raise ValueError(f"category {category!r} has no items")
    rng = random.Random(seed)
    per_category_draws = {}
    observed = {}
    for category in categories:
        differences = [float(x) - float(y)
                       for x, y in zip(arm[category], baseline[category])]
        observed[category] = sum(differences) / len(differences)
        draws = []
        indexes = _indices(rng, len(differences), resamples)
        for draw in indexes:
            draws.append(sum(differences[index] for index in draw) / len(differences))
        per_category_draws[category] = draws
    combined = [sum(per_category_draws[category][index] for category in categories)
                / len(categories) for index in range(resamples)]
    combined.sort()
    alpha = (1.0 - level) / 2.0
    arm_composite = sum(category_means(
        {c: arm[c] for c in categories}).values()) / len(categories)
    base_composite = sum(category_means(
        {c: baseline[c] for c in categories}).values()) / len(categories)
    relative = None
    if base_composite > 0:
        relative = (arm_composite - base_composite) / base_composite
    return {
        "arm_composite": arm_composite,
        "baseline_composite": base_composite,
        "mean_difference": arm_composite - base_composite,
        "relative_difference": relative,
        "ci_low": percentile(combined, alpha),
        "ci_high": percentile(combined, 1.0 - alpha),
        "per_category_difference": observed,
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "categories": categories,
        "n_items": sum(len(arm[c]) for c in categories),
    }


def stratified_mean(by_category: dict, seed: int = SEED, resamples: int = RESAMPLES,
                    level: float = LEVEL) -> dict:
    means = category_means(by_category)
    if not means:
        raise ValueError("stratified_mean needs at least one category")
    rng = random.Random(seed)
    draws = [0.0] * resamples
    for category in sorted(by_category):
        values = [float(value) for value in by_category[category]]
        indexes = _indices(rng, len(values), resamples)
        for index, draw in enumerate(indexes):
            draws[index] += sum(values[i] for i in draw) / len(values) / len(means)
    draws.sort()
    alpha = (1.0 - level) / 2.0
    return {
        "mean": sum(means.values()) / len(means),
        "ci_low": percentile(draws, alpha),
        "ci_high": percentile(draws, 1.0 - alpha),
        "resamples": resamples,
        "seed": seed,
        "level": level,
        "n": sum(len(values) for values in by_category.values()),
    }
