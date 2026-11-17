#!/usr/bin/env python3
"""P19.9: counterfactual utility routing for the corrective residual.

The corrective gate originally opened on a learned function of `[h_t ; r_t]`
without an explicit estimate of whether the memory read *helps* at that token,
and it could only decide **after** the read had already been fetched. P19.9
routes on a predicted per-token improvement instead:

    Delta_t = l_base(t) - l_{base+mem}(t)                 # positive = helps

The S50 corrections make the ordering and the supervision explicit:

* `PreReadUtilityRouter` predicts Delta from the hidden state and compact
  metadata only (`h_t`, position, budget). It cannot see `r_t`, `read_norm` or
  the post-read gate, so its decision is available **before** memory is fetched.
  `select` returns the tokens that should be read; a caller fetches only those.
* the existing `UtilityRouter` is retained as the **optional post-read gate**.
  `TwoStageUtilityRouter` combines the pre-read prediction with the post-read
  refinement in logit space, so a route that fetches memory can still close a
  gate the pre-read estimate opened, but cannot open one it closed.
* outcome supervision is the **paired counterfactual difference**

      Delta_outcome = P(correct | memory) - P(correct | base)

  estimated from paired rollouts that were each verified separately. The old
  absolute `verified-outcome` label (1.0/0.0 broadcast to a sequence) is kept as
  a registered control, not as the outcome route: it cannot distinguish a token
  that helps from a token in a sequence that happened to be correct.

The label can also come from a **counterfactual token loss** measured by two
forward passes. The sources are registered separately.

This module is the CPU-testable interface only. It does not train, sample or
generate; the runner is a later GPU job.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

#: Features the router may consume. They are all causal at token `t`: the
#: hidden state after the block's own causal attention and the memory read
#: computed from tokens at or before `t`. The target token and any label are
#: forbidden.
CAUSAL_FEATURE_KEYS = frozenset({
    "h_t", "r_t", "read_norm", "hidden_norm", "gate_t", "position",
})
#: Compact metadata a pre-read router may see. Nothing here depends on the
#: memory values, so the decision is computable before the fetch.
PRE_READ_METADATA_KEYS = frozenset({"position_fraction", "budget_fraction"})
#: Features that require the read to have happened. Feeding any of these to the
#: pre-read predictor is an ordering error, not a nuisance.
POST_READ_ONLY_KEYS = frozenset({"r_t", "read_norm", "gate_t"})
FORBIDDEN_FEATURE_KEYS = frozenset({
    "target", "target_id", "y_t", "answer", "gold", "label", "delta",
    "base_loss", "memory_loss", "teacher_logprob",
})
SUPERVISION_SOURCES = ("counterfactual-token-loss", "verified-outcome",
                       "counterfactual-outcome")
DEFAULT_COST = 0.0
DEFAULT_TAU = 0.5
DEFAULT_PRE_READ_THRESHOLD = 0.5


class UtilityRouter(nn.Module):
    """Post-read gate: predict `Delta_t` from `[h_t ; r_t]` and gate the read.

    Retained as the optional second stage of `TwoStageUtilityRouter`. Using it
    alone fetches every read before it can decide, which is the ordering S50
    corrects.
    """

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


def _metadata_columns(metadata: Mapping[str, torch.Tensor | float],
                      key_shape: torch.Size, dtype: torch.dtype,
                      device: torch.device) -> list[torch.Tensor]:
    """Validate compact metadata and expand each value to `key_shape`."""
    if metadata is None:
        raise ValueError("pre-read metadata is required")
    keys = set(metadata)
    forbidden = keys & (POST_READ_ONLY_KEYS | FORBIDDEN_FEATURE_KEYS)
    if forbidden:
        raise ValueError(
            "pre-read metadata contains post-read or label keys: "
            f"{sorted(forbidden)}; the decision must precede the fetch")
    unknown = keys - PRE_READ_METADATA_KEYS
    if unknown:
        raise ValueError(f"unregistered pre-read metadata keys: {sorted(unknown)}")
    missing = PRE_READ_METADATA_KEYS - keys
    if missing:
        raise ValueError(f"missing pre-read metadata keys: {sorted(missing)}")
    columns = []
    for key in sorted(PRE_READ_METADATA_KEYS):
        value = torch.as_tensor(metadata[key], dtype=dtype, device=device)
        value = value.expand(key_shape).unsqueeze(-1)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"metadata {key!r} is not finite")
        if bool((value < 0.0).any()) or bool((value > 1.0).any()):
            raise ValueError(f"metadata {key!r} must be a fraction in [0, 1]")
        columns.append(value)
    return columns


class PreReadUtilityRouter(nn.Module):
    """Predict `Delta_t` before the read, from hidden state + compact metadata.

    The predictor consumes `h_t` and the registered fraction metadata only. It
    has no access to `r_t`, `read_norm` or `gate_t`, so `select` can decide which
    tokens are worth a memory fetch before any fetch happens. `hidden_norm` is
    computed from `h_t` inside the module (it needs no read).
    """

    def __init__(self, hidden: int = 1024,
                 metadata_keys: Sequence[str] = tuple(sorted(PRE_READ_METADATA_KEYS)),
                 tau: float = DEFAULT_TAU):
        super().__init__()
        self.hidden = int(hidden)
        self.metadata_keys = tuple(metadata_keys)
        if set(self.metadata_keys) != set(PRE_READ_METADATA_KEYS):
            raise ValueError(
                "the pre-read router must consume exactly the registered "
                f"metadata {sorted(PRE_READ_METADATA_KEYS)}")
        self.tau = float(tau)
        # h_t plus the compact metadata; hidden_norm is one extra scalar.
        self.feature_dim = self.hidden + len(self.metadata_keys) + 1
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, 64, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(64, 1, dtype=torch.float32),
        )
        with torch.no_grad():
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)

    def features(self, hidden_states: torch.Tensor,
                 metadata: Mapping[str, torch.Tensor | float]) -> torch.Tensor:
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden:
            raise ValueError(
                f"hidden states must be [..., {self.hidden}], got "
                f"{tuple(hidden_states.shape)}")
        hidden = hidden_states.float()
        shape = hidden.shape[:-1]
        columns = _metadata_columns(metadata, shape, hidden.dtype,
                                    hidden.device)
        norm = hidden.norm(dim=-1, keepdim=True)
        # A bounded scalar, not the raw norm: the linear layer sees a feature
        # with the same scale as the fractions.
        norm = torch.tanh(norm / math.sqrt(max(self.hidden, 1)))
        return torch.cat([hidden, norm, *columns], dim=-1)

    def predict(self, hidden_states: torch.Tensor,
                metadata: Mapping[str, torch.Tensor | float]) -> torch.Tensor:
        return self.predictor(self.features(hidden_states, metadata)).squeeze(-1)

    def gate(self, predicted: torch.Tensor, cost: torch.Tensor | float = 0.0
             ) -> torch.Tensor:
        return torch.sigmoid((predicted - cost) / max(self.tau, 1e-6))

    def select(self, hidden_states: torch.Tensor,
               metadata: Mapping[str, torch.Tensor | float],
               cost: torch.Tensor | float = 0.0,
               threshold: float = DEFAULT_PRE_READ_THRESHOLD
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(mask, predicted, gate)`; fetch memory only where `mask` is true."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        predicted = self.predict(hidden_states, metadata)
        gated = self.gate(predicted, cost)
        mask = gated >= threshold
        self.last_prediction = predicted.detach()
        self.last_pre_gate = gated.detach()
        return mask, predicted, gated


class TwoStageUtilityRouter(nn.Module):
    """Optional post-read refinement of the pre-read gate.

    The pre-read prediction is always computed first and without the read. When
    `use_post_read` is true the post-read router's prediction is added in logit
    space before the gate, so the refinement can close a gate but cannot open
    one the pre-read stage rejected (the caller only fetches where the pre-read
    stage selected). With `use_post_read=False` the module is exactly the
    pre-read router and the read is never consumed by the gate.
    """

    def __init__(self, hidden: int = 1024, read_dim: int | None = None,
                 tau: float = DEFAULT_TAU, use_post_read: bool = True):
        super().__init__()
        self.tau = float(tau)
        self.pre = PreReadUtilityRouter(hidden=hidden, tau=tau)
        self.post = (UtilityRouter(hidden=hidden, read_dim=read_dim, tau=tau)
                     if use_post_read else None)
        self.use_post_read = bool(use_post_read)

    def pre_gate(self, hidden_states: torch.Tensor,
                 metadata: Mapping[str, torch.Tensor | float],
                 cost: torch.Tensor | float = 0.0) -> torch.Tensor:
        return self.pre.gate(self.pre.predict(hidden_states, metadata), cost)

    def select(self, hidden_states: torch.Tensor,
               metadata: Mapping[str, torch.Tensor | float],
               cost: torch.Tensor | float = 0.0,
               threshold: float = DEFAULT_PRE_READ_THRESHOLD
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre.select(hidden_states, metadata, cost, threshold)

    def forward(self, hidden_states: torch.Tensor,
                metadata: Mapping[str, torch.Tensor | float],
                read: torch.Tensor,
                cost: torch.Tensor | float = 0.0) -> torch.Tensor:
        pre_prediction = self.pre.predict(hidden_states, metadata)
        prediction = pre_prediction
        if self.post is not None:
            prediction = prediction + self.post.predict(hidden_states, read)
        gated = torch.sigmoid((prediction - cost) / max(self.tau, 1e-6))
        self.last_prediction = prediction.detach()
        self.last_pre_gate = self.pre.gate(pre_prediction, cost).detach()
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
    """Absolute sequence label (1.0 correct, 0.0 otherwise).

    Registered control only. The outcome route uses `outcome_delta`, because an
    absolute correctness label cannot tell a helpful token from a token in a
    sequence that happened to be correct.
    """
    if not isinstance(verified, bool):
        raise ValueError("verified outcome must be a bool, not a likelihood")
    return 1.0 if verified else 0.0


def _verified(side: object, where: str) -> bool:
    if not isinstance(side, Mapping) or "verified" not in side:
        raise ValueError(
            f"{where} must be a separately verified rollout object with a "
            "`verified` bool")
    value = side["verified"]
    if not isinstance(value, bool):
        raise ValueError(f"{where}.verified must be a bool, not a likelihood")
    return value


def outcome_delta(paired_rollouts: Sequence[Mapping[str, object]]) -> dict:
    """`P(correct | memory) - P(correct | base)` from paired rollouts.

    Each element of `paired_rollouts` is one prompt's pair:

        {"rollout_id": "r0", "base": {"verified": bool},
         "memory": {"verified": bool}}

    Both sides are verified separately; if the same source label were reused for
    both, the difference would be identically zero and the check would be
    vacuous. The probability is the success frequency over the pairs, not a
    teacher-forced likelihood.
    """
    if not paired_rollouts:
        raise ValueError("paired_rollouts must contain at least one pair")
    base_successes = 0
    memory_successes = 0
    seen: set[str] = set()
    for index, pair in enumerate(paired_rollouts):
        where = f"paired_rollouts[{index}]"
        if not isinstance(pair, Mapping):
            raise ValueError(f"{where} must be an object")
        missed = {"base", "memory"} - set(pair)
        if missed:
            raise ValueError(f"{where} is missing {sorted(missed)}")
        rollout_id = pair.get("rollout_id")
        if rollout_id is not None:
            key = str(rollout_id)
            if key in seen:
                raise ValueError(f"{where}: duplicate rollout_id {key!r}")
            seen.add(key)
        base_successes += int(_verified(pair["base"], f"{where}.base"))
        memory_successes += int(_verified(pair["memory"], f"{where}.memory"))
    pairs = len(paired_rollouts)
    base_probability = base_successes / pairs
    memory_probability = memory_successes / pairs
    return {
        "pairs": pairs,
        "base_successes": base_successes,
        "memory_successes": memory_successes,
        "base_probability": base_probability,
        "memory_probability": memory_probability,
        "delta": memory_probability - base_probability,
    }


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
      rejected. This is the absolute-label control.
    * `counterfactual-outcome`: `paired_rollouts` holds paired rollouts, each
      side separately verified; the label is
      `P(correct | memory) - P(correct | base)`. An absolute top-level
      `verified` is rejected for this source, so the route cannot silently fall
      back to the control's supervision.

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
        elif source == "counterfactual-outcome":
            if "verified" in record:
                raise ValueError(
                    "counterfactual-outcome uses paired rollouts, not an "
                    "absolute correctness label")
            if "teacher_logprob" in record or "likelihood" in record:
                raise ValueError(
                    "counterfactual-outcome supervision may not use a "
                    "teacher-forced likelihood")
            if "paired_rollouts" not in record:
                raise ValueError(f"record {index}: paired_rollouts is required")
            summary = outcome_delta(record["paired_rollouts"])
            count = int(record.get("token_count", 1))
            if count < 1:
                raise ValueError(f"record {index}: token_count must be positive")
            labels.append({
                "prompt_id": prompt_id, "split": split,
                "label": summary["delta"],
                "supervision": source,
                "token_count": count,
                **summary,
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
