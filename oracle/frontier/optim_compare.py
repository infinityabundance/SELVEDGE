#!/usr/bin/env python3
"""S53: a model-level optimiser comparison harness with a CPU-testable core.

The existing optimiser evidence is a per-tensor parity test and a 200-step
training log whose KL sees a different batch every step. Neither answers the
model-level question: at matched initialisation, matched stream and matched
learning rate, does the custom int8-state optimiser track fp32 AdamW on the
same fixed-batch teacher KL and held-out CE?

This module contains the part that can be tested without a GPU:

* `run_comparison` runs one arm at a time, so both arms can share one GPU and
  one teacher; `make()` is responsible for restoring the exact initial state,
  and `release()` is called between arms.
* `summarise_comparison` turns the recorded series into per-arm final metrics
  and a paired bootstrap interval on `custom - adamw` over the evaluation
  points.

The GPU driver is `scripts/s53_optimizer_comparison.py`. It is queued, not run;
its container limit must come from its own smoke peak.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass
class ArmSpec:
    """One optimiser arm: a fresh matched-init state and one update function."""

    make: Callable[[], object]
    step: Callable[[object, int], object]
    release: Callable[[object], None] | None = None


@dataclass
class Comparison:
    steps: int
    eval_steps: list[int]
    series: dict[str, dict[int, dict[str, float]]]

    def points(self, arm: str) -> list[dict]:
        return [{"step": step, **metrics}
                for step, metrics in sorted(self.series[arm].items())]


def run_comparison(specs: Mapping[str, ArmSpec],
                   evaluate: Callable[[object], dict[str, float]],
                   steps: int, eval_every: int = 25) -> Comparison:
    """Evaluate each arm at init and every `eval_every` steps, then at `steps`.

    Arms are executed sequentially; `make()` is called afresh, so a caller that
    returns an exact copy of the same initial state gives a matched-init
    comparison without holding two students at once.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive")
    eval_steps = sorted({0, steps, *range(eval_every, steps, eval_every)})
    series: dict[str, dict[int, dict[str, float]]] = {}
    for name, spec in specs.items():
        state = spec.make()
        points: dict[int, dict[str, float]] = {0: dict(evaluate(state))}
        try:
            for step in range(steps):
                state = spec.step(state, step)
                if step + 1 in eval_steps:
                    points[step + 1] = dict(evaluate(state))
        finally:
            if spec.release is not None:
                spec.release(state)
        series[name] = points
    return Comparison(steps=steps, eval_steps=eval_steps, series=series)


def _bootstrap_interval(values: list[float], seed: int = 53,
                        resamples: int = 4000) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "ci95": [means[int(0.025 * resamples)],
                 means[min(resamples - 1, int(0.975 * resamples))]],
        "n": n,
    }


def summarise_comparison(comparison: Comparison, reference: str,
                         candidate: str, seed: int = 53) -> dict:
    """Final metrics per arm and paired `candidate - reference` intervals."""
    if reference not in comparison.series or candidate not in comparison.series:
        raise KeyError(f"comparison needs arms {reference!r} and {candidate!r}")
    metrics = sorted({key for step in comparison.eval_steps
                      for arm in (reference, candidate)
                      for key in comparison.series[arm][step]})
    paired = {}
    for metric in metrics:
        differences = [
            comparison.series[candidate][step][metric]
            - comparison.series[reference][step][metric]
            for step in comparison.eval_steps
        ]
        paired[metric] = _bootstrap_interval(differences, seed=seed)
    return {
        "schema": "selvedge/optimiser-comparison/v1",
        "steps": comparison.steps,
        "eval_steps": comparison.eval_steps,
        "matched_init": True,
        "arms": {
            arm: {
                "final": comparison.series[arm][comparison.steps],
                "series": comparison.points(arm),
            }
            for arm in comparison.series
        },
        "paired": {f"{candidate}_minus_{reference}": paired},
        "reading": ("teacher KL and held-out CE are fixed-batch measurements; "
                    "the paired interval is over the shared evaluation steps, "
                    "not over the training steps, whose batches differ"),
        "finite": all(math.isfinite(value)
                      for arm in comparison.series.values()
                      for point in arm.values()
                      for value in point.values()),
    }


__all__ = ["ArmSpec", "Comparison", "run_comparison",
           "summarise_comparison"]
