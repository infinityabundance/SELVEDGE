#!/usr/bin/env python3
"""P19.9 skeleton: counterfactual utility routing for the corrective residual.

The corrective gate currently opens on a learned function of `[h_t ; r_t]`
without an explicit estimate of whether the memory read *helps* at that token.
P19.9 routes on a predicted per-token improvement instead:

    Delta_t = l_base(t) - l_{base+mem}(t)                 # positive = helps

`Delta_t` is regressed from causally available features, and the residual is
gated by the predicted improvement minus a measured cost:

    g_t = sigmoid((Delta_hat_t - c_t) / tau)

The point of the route is that the label can come from either

* a **counterfactual token loss** measured by two forward passes (the base and
  the base-plus-residual), or
* a **verified math outcome** (the final answer was exact), which is a
  sequence-level label and is therefore broadcast to the sequence's tokens.

The second source is deliberately not teacher-forced likelihood: a token that
raises the teacher's probability is not automatically a token that helps the
final answer. The two sources are registered as separate arms.

This module is the CPU-testable interface only. It does not train, sample or
generate; the runner is a later GPU job.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Features the router may consume. They are all causal at token `t`: the
#: hidden state after the block's own causal attention and the memory read
#: computed from tokens at or before `t`. The target token and any label are
#: forbidden.
CAUSAL_FEATURE_KEYS = frozenset({
    "h_t", "r_t", "read_norm", "hidden_norm", "gate_t", "position",
})
FORBIDDEN_FEATURE_KEYS = frozenset({
    "target", "target_id", "y_t", "answer", "gold", "label", "delta",
    "base_loss", "memory_loss", "teacher_logprob",
})
SUPERVISION_SOURCES = ("counterfactual-token-loss", "verified-outcome")
DEFAULT_COST = 0.0
DEFAULT_TAU = 0.5


class UtilityRouter(nn.Module):
    """Predict `Delta_t` from causal features and gate the residual by it."""

    def __init__(self, hidden: int = 1024, read_dim: int | None = None,
                 tau: float = DEFAULT_TAU):
        super().__init__()
        self.hidden = int(hidden)
        self.read_dim = int(read_dim if read_dim is not None else hidden)
        self.tau = float(tau)
        self.feature_dim = self.hidden + self.read_dim
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 64, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(64, 1, dtype=torch.float32),
        )
        # Exactly zero at init: the untrained router predicts no improvement and
        # therefore opens a gate at sigmoid(-cost/tau), which is reported rather
        # than assumed to be identity. The arm's identity guarantee comes from
        # the residual gate being zero-initialised separately.
        with torch.no_grad():
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)

    def predict(self, hidden_states: torch.Tensor,
                read: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden_states.float(), read.float()], dim=-1)
        return self.predictor(features).squeeze(-1)

    def gate(self, predicted: torch.Tensor, cost: torch.Tensor | float = 0.0
             ) -> torch.Tensor:
        """`sigmoid((Delta_hat - cost) / tau)`; 0.5 when improvement equals cost."""
        return torch.sigmoid((predicted - cost) / max(self.tau, 1e-6))

    def forward(self, hidden_states: torch.Tensor, read: torch.Tensor,
                cost: torch.Tensor | float = 0.0) -> torch.Tensor:
        predicted = self.predict(hidden_states, read)
        gated = self.gate(predicted, cost)
        self.last_prediction = predicted.detach()
        self.last_gate = gated.detach()
        return gated[..., None] * read


def counterfactual_improvement(base_losses: torch.Tensor,
                               memory_losses: torch.Tensor) -> torch.Tensor:
    """`Delta_t = l_base(t) - l_{base+mem}(t)`; positive means memory helps."""
    if base_losses.shape != memory_losses.shape:
        raise ValueError("base and memory losses must have the same shape")
    return base_losses - memory_losses


def measured_cost(read: torch.Tensor, price_per_element: float = 1.0,
                  latency: float = 0.0) -> torch.Tensor:
    """A measured per-token cost: read magnitude times a price, plus latency.

    The registration fixes `price_per_element` and `latency` from the runtime
    receipt before the arm runs. This function only defines the arithmetic; it
    does not invent a price.
    """
    if price_per_element < 0.0 or latency < 0.0:
        raise ValueError("cost terms must be non-negative")
    return price_per_element * read.float().abs().sum(-1) + float(latency)


def outcome_label(verified: bool) -> float:
    """Sequence-level verified outcome: 1.0 correct, 0.0 otherwise."""
    if not isinstance(verified, bool):
        raise ValueError("verified outcome must be a bool, not a likelihood")
    return 1.0 if verified else 0.0


def _check_features(features: dict) -> None:
    unknown = set(features) - CAUSAL_FEATURE_KEYS
    forbidden = set(features) & FORBIDDEN_FEATURE_KEYS
    if forbidden:
        raise ValueError(
            f"features contain non-causal or label keys: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"features contain unregistered keys: {sorted(unknown)}")


def labels_from_records(records: list[dict], source: str) -> list[dict]:
    """Validate records and emit token-level labels for one supervision source.

    Required record fields: `prompt_id`, `split` (`fit` or `holdout`), and
    `features` (causal keys only). Then:

    * `counterfactual-token-loss`: `base_loss` and `memory_loss` are token
      losses from the two forward passes; the label is their difference.
    * `verified-outcome`: `verified` is a bool for the sequence and `token_count`
      gives how many tokens receive the broadcast label; a float likelihood is
      rejected.

    Leakage rules enforced here: a prompt id may not appear in both splits, a
    label key may not appear in features, and the outcome source may not carry a
    teacher-forced likelihood field. Splitting is by prompt, not by token.
    """
    if source not in SUPERVISION_SOURCES:
        raise ValueError(f"unknown supervision source {source!r}")
    seen: dict[str, set] = {"fit": set(), "holdout": set()}
    labels: list[dict] = []
    for index, record in enumerate(records):
        split = record.get("split")
        if split not in seen:
            raise ValueError(f"record {index}: split must be fit or holdout")
        prompt_id = record.get("prompt_id")
        if prompt_id is None:
            raise ValueError(f"record {index}: prompt_id is required")
        seen[split].add(prompt_id)
        features = record.get("features") or {}
        _check_features(features)
        if source == "counterfactual-token-loss":
            base = float(record["base_loss"])
            memory = float(record["memory_loss"])
            if not math.isfinite(base) or not math.isfinite(memory):
                raise ValueError(f"record {index}: losses must be finite")
            labels.append({
                "prompt_id": prompt_id, "split": split,
                "label": base - memory,
                "supervision": source,
                "token_count": 1,
            })
        else:
            if "verified" not in record:
                raise ValueError(f"record {index}: verified is required")
            if "teacher_logprob" in record or "likelihood" in record:
                raise ValueError(
                    "verified-outcome supervision may not use a teacher-forced "
                    "likelihood")
            value = outcome_label(record["verified"])
            count = int(record.get("token_count", 1))
            if count < 1:
                raise ValueError(f"record {index}: token_count must be positive")
            labels.append({
                "prompt_id": prompt_id, "split": split,
                "label": value,
                "supervision": source,
                "token_count": count,
            })
    overlap = seen["fit"] & seen["holdout"]
    if overlap:
        raise ValueError(f"prompt ids appear in both splits: {sorted(overlap)}")
    return labels
