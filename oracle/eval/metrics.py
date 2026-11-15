#!/usr/bin/env python3
"""Paired percentile bootstrap over items; no external dependency."""

from __future__ import annotations

import numpy as np

SEED = 20261115
RESAMPLES = 10_000


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = SEED,
                     resamples: int = RESAMPLES) -> dict:
    """Mean of (a - b) with a 95% percentile CI over paired items."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap needs equal non-empty arrays")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, a.size, size=(resamples, a.size))
    differences = (a - b)[indexes].mean(axis=1)
    return {
        "mean_difference": float((a - b).mean()),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "n": int(a.size),
    }


def bootstrap_mean(values: np.ndarray, seed: int = SEED,
                   resamples: int = RESAMPLES) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap_mean needs a non-empty array")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, values.size, size=(resamples, values.size))
    means = values[indexes].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "n": int(values.size),
    }


def stratified_bootstrap(per_domain: dict[str, np.ndarray], seed: int = SEED,
                         resamples: int = RESAMPLES) -> dict:
    """Equal-weight mixture over domains, resampling items within each domain."""
    domains = sorted(per_domain)
    generator = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    for domain in domains:
        values = np.asarray(per_domain[domain], dtype=np.float64)
        indexes = generator.integers(0, values.size, size=(resamples, values.size))
        draws += values[indexes].mean(axis=1) / len(domains)
    return {
        "mean": float(np.mean([np.asarray(per_domain[domain]).mean() for domain in domains])),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "n": int(sum(np.asarray(per_domain[domain]).size for domain in domains)),
    }
