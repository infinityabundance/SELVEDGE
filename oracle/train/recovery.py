#!/usr/bin/env python3
"""Omission-conditioned recovery (plan item P13.14).

A sparse layer computes `F_sparse(h, S)` for the selected set `S`. The part of
the dense layer's output that `S` omits is not recoverable by a fixed model, so
the corrected layer is

    h' = F_sparse(h, S) + C_theta(h, S)

where `C_theta` is **conditioned on which set was selected** (equivalently, on
which groups were omitted). Three properties are requirements, not
conveniences:

* **Cheap.** `C_theta` is a rank-`r` bottleneck over `[h; enc(S)]`, so its
  element count is `m*d + (hidden+d)*r + r*hidden` per layer.
* **Causally available.** Its only inputs are the layer's own input hidden
  state and the router's own selection, both of which exist at inference time
  before the layer runs. It never sees a label, a teacher logit, a future token
  or the omitted groups' activations. `causal_input_signature()` returns those
  two inputs and the tests pin it.
* **Counted.** `active_elements()` is the per-token active count and is added to
  the arm's ledger (`ledger()`), which is what the M2 rule charges: every
  learned selector or compensator weight is part of the active unique element
  count, and two arms are comparable only at identical realized counts.

The baseline is the hard one: **layer-wise output reconstruction**, the step
GPTQ performs for quantisation error (Frantar et al., 2210.17323, VERIFIED in
`docs/PRIOR-ART-AND-NOVELTY.md` row 127). GPTQ chooses quantised weights `W'`
minimising `||W X - W' X||^2` and compensates the residual through the Hessian
inverse; the omission analogue implemented here fits a **low-rank linear map**
`A` minimising `||X A - (Y_dense - Y_sparse)||^2` on calibration activations
and truncates it to the rank that the compensator's own element count allows.
That correspondence is a *reading* of GPTQ, not an identity: GPTQ alters
weights, this alters a residual map. `fit(mode="ot")` additionally offers the
Gaussian moment-matching map, which matches the teacher's output mean and
covariance exactly; both are reported so the baseline cannot be dismissed as
weak.

Nothing in this module trains anything by itself: it defines the modules, the
closed-form fits, the accounting and the arm specification.
`oracle/eval/recovery_court.py` is the runner, and its `--smoke` path builds a
tiny model so the whole pipeline is CPU-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Set-conditioning modes. `selected` encodes the k chosen groups, `omitted` the
#: m-k others, `shuffled` encodes an independently drawn set of the same size,
#: which is the falsifying control (knows the rate, not the identity).
SET_MODES = ("selected", "omitted", "shuffled")


@dataclass(frozen=True)
class RecoveryConfig:
    """Everything that decides the compensator's size and its conditioning."""

    rank: int = 8
    set_dim: int = 8
    set_mode: str = "omitted"
    activation: str = "silu"
    shuffle_seed: int = 20261116

    def declared(self) -> dict:
        return {"rank": self.rank, "set_dim": self.set_dim,
                "set_mode": self.set_mode, "activation": self.activation}


def _activation(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"unknown activation {name!r}")


def set_embedding(experts: int, set_dim: int, seed: int | None = None) -> nn.Embedding:
    """A per-expert vector; a set's code is the sum of its members' vectors."""
    embedding = nn.Embedding(experts, set_dim, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0 if seed is None else int(seed))
    with torch.no_grad():
        embedding.weight.copy_(torch.empty(experts, set_dim).normal_(
            mean=0.0, std=0.02, generator=generator))
    return embedding


def encode_sets(embedding: nn.Embedding, selected: torch.Tensor, experts: int,
                mode: str) -> torch.Tensor:
    """`[N, k]` indices -> `[N, set_dim]` set code.

    `selected` are the top-k expert indices, distinct and in selection order.
    `omitted` is the complement, computed here from the same tensor so the
    compensator can never see an omission list the router did not produce.
    """
    if mode == "selected":
        indices = selected
    elif mode == "omitted":
        mask = torch.zeros(selected.shape[0], experts, dtype=torch.bool,
                           device=selected.device)
        mask.scatter_(1, selected, True)
        indices = mask.logical_not().nonzero(as_tuple=False)[:, 1]
        indices = indices.reshape(selected.shape[0], experts - selected.shape[1])
    else:
        raise ValueError(f"{mode!r} is encoded by the caller, not here")
    vectors = embedding(indices)
    return vectors.sum(dim=1)


def shuffle_sets(selected: torch.Tensor, experts: int, generator: torch.Generator | None = None,
                 seed: int | None = None, replace: bool = False) -> torch.Tensor:
    """Independent sets of the same size: the shuffled-condition control.

    With `replace=False` each position gets a proper k-subset, so the control
    knows the omission *rate* exactly and nothing about the identity.
    """
    positions, k = selected.shape
    if generator is None:
        generator = torch.Generator(device=selected.device)
        generator.manual_seed(20261116 if seed is None else int(seed))
    keys = torch.rand(positions, experts, generator=generator,
                      device=selected.device)
    if not replace:
        return keys.topk(k, dim=-1).indices
    return torch.randint(0, experts, (positions, k), generator=generator,
                         device=selected.device)


class OmissionCompensator(nn.Module):
    """`C_theta(h, S)`: a rank-`r` correction conditioned on the selected set."""

    def __init__(self, hidden: int, experts: int, config: RecoveryConfig,
                 seed: int | None = None):
        super().__init__()
        self.hidden = int(hidden)
        self.experts = int(experts)
        self.config = config
        self.embedding = set_embedding(experts, config.set_dim, seed)
        self.down = nn.Linear(hidden + config.set_dim, config.rank, bias=True,
                              dtype=torch.float32)
        self.up = nn.Linear(config.rank, hidden, bias=True, dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else int(seed) + 7919)
        with torch.no_grad():
            nn.init.normal_(self.down.weight, std=0.02, generator=generator)
            nn.init.zeros_(self.down.bias)
            # Zero last layer: the corrected layer equals the sparse layer at
            # step 0, bit for bit, which is what makes the zero-point testable.
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    # -- accounting ---------------------------------------------------------
    def active_elements(self) -> int:
        """Learned elements one output token reads through this module."""
        return (self.embedding.weight.numel()
                + self.down.weight.numel() + self.down.bias.numel()
                + self.up.weight.numel() + self.up.bias.numel())

    def accounting(self) -> dict:
        return {
            "kind": "omission_compensator",
            "set_mode": self.config.set_mode,
            "rank": self.config.rank,
            "set_dim": self.config.set_dim,
            "set_embedding_elements": self.embedding.weight.numel(),
            "down_elements": self.down.weight.numel() + self.down.bias.numel(),
            "up_elements": self.up.weight.numel() + self.up.bias.numel(),
            "active_elements": self.active_elements(),
        }

    @staticmethod
    def causal_input_signature() -> tuple[str, str]:
        return ("hidden_states", "selected")

    # -- forward ------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1]).to(torch.float32)
        code = encode_sets(self.embedding, selected, self.experts, self.config.set_mode)
        hidden = _activation(self.config.activation)(
            self.down(torch.cat([flat, code], dim=-1)))
        return self.up(hidden).reshape(shape)


class RecoveredSparse(nn.Module):
    """`F_sparse(h, S) + C_theta(h, S)` around an existing sparse module.

    The selection is read from the arm's own router on the same hidden state the
    sparse module consumes, so the conditioning cannot disagree with the
    computation that actually ran.
    """

    def __init__(self, sparse: nn.Module, router: nn.Linear, compensator: nn.Module,
                 k: int):
        super().__init__()
        self.sparse = sparse
        self.router = router
        self.compensator = compensator
        self.alignment = None
        self.k = int(k)

    def selection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return F.linear(flat, self.router.weight).topk(self.k, dim=-1).indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        selected = self.selection(hidden_states)
        output = self.sparse(hidden_states)
        corrections = []
        if self.compensator is not None:
            corrections.append(self.compensator(hidden_states, selected))
        if self.alignment is not None:
            corrections.append(self.alignment(hidden_states))
        for delta in corrections:
            if delta.dtype != output.dtype:
                # Same reason as `modeling.AdditiveFFN`: an fp32 residual would
                # promote the whole stream and change every downstream numeric.
                delta = delta.to(output.dtype)
            output = output + delta
        return output


class LayerAlignment(nn.Module):
    """Affine low-rank map fitted on calibration activations, set-independent."""

    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.left = nn.Parameter(torch.zeros(hidden, rank, dtype=torch.float32))
        self.right = nn.Parameter(torch.zeros(hidden, rank, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))

    def active_elements(self) -> int:
        return self.left.numel() + self.right.numel() + self.bias.numel()

    def accounting(self) -> dict:
        return {"kind": "layer_alignment", "rank": self.rank,
                "elements": self.active_elements()}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float32)
        mapped = flat @ self.right @ self.left.T + self.bias
        return mapped.reshape(hidden_states.shape)

    @torch.no_grad()
    def fit(self, features: torch.Tensor, targets: torch.Tensor,
            ridge: float = 1e-3, mode: str = "l2") -> dict:
        """Fit and truncate to `self.rank` in one closed-form step.

        `features` is `[N, hidden]` (the layer's input), `targets` is
        `[N, hidden]` (dense minus sparse output on the same inputs).

        * `mode="l2"` — ridge least squares, the GPTQ-style layer-wise output
          reconstruction.
        * `mode="ot"` — the Gaussian moment-matching map, which reproduces the
          target's mean and covariance exactly (up to the ridge) and is reported
          as the stronger distributional variant.
        """
        x = features.detach().to(torch.float64)
        y = targets.detach().to(torch.float64)
        if x.shape != y.shape:
            raise ValueError("alignment needs features and targets of one shape")
        count = x.shape[0]
        if count == 0:
            raise ValueError("alignment needs at least one calibration position")
        if mode == "l2":
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            xc = x - x_mean
            yc = y - y_mean
            gram = xc.T @ xc / count
            gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype) \
                * float(gram.diagonal().mean().clamp_min(1e-12))
            full = torch.linalg.solve(gram, xc.T @ yc / count)
            offset = y_mean - x_mean @ full
            # Rank truncation of the affine map, offset folded into the mean.
            u, s, vh = torch.linalg.svd(full)
            rank = min(self.rank, s.numel())
            # `full = u diag(s) vh`, and the forward map is `x @ right @ left.T`,
            # so `right = u diag(s)` and `left = vh.T`.
            right = u[:, :rank] * s[:rank].unsqueeze(0)
            left = vh[:rank].T
            residual = xc @ full - yc
            tail = (xc @ (full - right @ left.T) - yc)
        elif mode == "ot":
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            xc = x - x_mean
            yc = y - y_mean
            cov_x = xc.T @ xc / count
            cov_y = yc.T @ yc / count
            eye = torch.eye(cov_x.shape[0], dtype=cov_x.dtype)
            scale = float(cov_x.diagonal().mean().clamp_min(1e-12))
            cov_x = cov_x + ridge * scale * eye
            cov_y = cov_y + ridge * scale * eye
            root_x = torch.linalg.cholesky(cov_x)
            root_y = torch.linalg.cholesky(cov_y)
            middle = root_x @ cov_y @ root_x.T
            values, vectors = torch.linalg.eigh(middle)
            values = values.clamp_min(1e-18)
            middle_root = vectors @ torch.diag(values.sqrt()) @ vectors.T
            inverse_root_x = torch.linalg.solve_triangular(
                root_x, eye, upper=False)
            full = torch.linalg.solve_triangular(
                root_x.transpose(0, 1), inverse_root_x.T @ middle_root.T,
                upper=True).T
            offset = y_mean - x_mean @ full
            u, s, vh = torch.linalg.svd(full)
            rank = min(self.rank, s.numel())
            right = u[:, :rank] * s[:rank].unsqueeze(0)
            left = vh[:rank].T
            residual = xc @ full - yc
            tail = xc @ (full - right @ left.T) - yc
        else:
            raise ValueError(f"unknown alignment mode {mode!r}")

        with torch.no_grad():
            self.left.copy_(left.to(torch.float32))
            self.right.copy_(right.to(torch.float32))
            self.bias.copy_(offset.reshape(-1).to(torch.float32))
        return {
            "mode": mode,
            "positions": int(count),
            "rank": int(rank),
            "full_rank_error_rms": float(residual.pow(2).mean().sqrt()),
            "truncated_error_rms": float((xc @ (right @ left.T) - yc).pow(2).mean().sqrt()),
            "truncation_tail_rms": float(tail.pow(2).mean().sqrt()),
            "target_rms": float(yc.pow(2).mean().sqrt()),
            "offset_norm": float(offset.norm()),
            "singular_values": [float(item) for item in s[:max(8, rank)]],
        }

    @torch.no_grad()
    def corrected(self, hidden_states: torch.Tensor,
                  offset: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(hidden_states)


def matched_alignment_rank(compensator_elements: int, hidden: int) -> int:
    """The rank at which an affine set-independent map costs what the compensator does.

    The baseline is compared at matched bytes (plan P13.14). An affine low-rank
    map costs `2 * hidden * rank + hidden` elements, so the rank is floored to
    keep the baseline at or below the compensator's count, and a budget that
    cannot afford rank one *fails closed* rather than quietly exceeding it.
    """
    budget = int(compensator_elements) - int(hidden)
    denominator = 2 * int(hidden)
    if denominator <= 0:
        raise ValueError("hidden must be positive")
    if budget < denominator:
        raise ValueError(
            f"a {compensator_elements}-element compensator cannot fund a rank-1 "
            f"affine baseline at hidden {hidden}; the comparison would not be "
            "at matched bytes")
    return budget // denominator


# ---------------------------------------------------------------------------
# arm construction and the ledger


@dataclass
class RecoveryArm:
    """One experimental arm: the sparse layer plus what corrects it."""

    name: str
    wrappers: list[RecoveredSparse] = field(default_factory=list)
    alignments: list[nn.Module] = field(default_factory=list)
    compensators: list[nn.Module] = field(default_factory=list)
    reference_full_bank: bool = False

    def active_elements(self) -> int:
        return sum(item.active_elements() for item in self.compensators) \
            + sum(item.active_elements() for item in self.alignments)

    def ledger(self, base_active: int, bank_k: int, bank_experts: int) -> dict:
        """The M2-style accounting line for this arm."""
        compensator = sum(item.active_elements() for item in self.compensators)
        alignment = sum(item.active_elements() for item in self.alignments)
        return {
            "arm": self.name,
            "base_active": int(base_active),
            "layers": len(self.wrappers),
            "compensator_elements": int(compensator),
            "alignment_elements": int(alignment),
            "recovery_elements": int(compensator + alignment),
            "realized_active": int(base_active + compensator + alignment),
            "omission_rate": float((bank_experts - bank_k) / bank_experts),
            "reference_full_bank": bool(self.reference_full_bank),
        }


def attach_recovery(arm_model, *, mode: str, config: RecoveryConfig,
                    seed: int = 1001) -> RecoveryArm:
    """Wrap every specialist bank of an arm with the requested correction.

    `mode` is `"none"` (sparse only), `"compensator"`, `"alignment"` or
    `"both"`. The arm's own weights are only ever *read*: the wrapper adds
    modules and never rewrites a T2 tensor.
    """
    if mode not in ("none", "compensator", "alignment", "both"):
        raise ValueError(f"unknown recovery mode {mode!r}")
    if config.set_mode not in SET_MODES:
        raise ValueError(f"unknown set mode {config.set_mode!r}")

    compensator_elements = OmissionCompensator(
        arm_model.model.config.hidden_size, 8, config).active_elements()
    rank = matched_alignment_rank(compensator_elements,
                                  arm_model.model.config.hidden_size)

    arm = RecoveryArm(name=mode)
    banks = arm_model.banks()
    for index, bank in enumerate(banks):
        hidden = bank.hidden
        reference = bank.router
        compensator = None
        if mode in ("compensator", "both"):
            compensator = OmissionCompensator(
                hidden, bank.experts, config, seed=seed + 11000 + index)
            arm.compensators.append(compensator)
        alignment = None
        if mode in ("alignment", "both"):
            alignment = LayerAlignment(hidden, rank)
            arm.alignments.append(alignment)
        wrapper = RecoveredSparse(bank, reference, compensator, bank.k)
        wrapper.alignment = alignment  # attached, not a submodule of the bank
        arm.wrappers.append(wrapper)
    arm.config = config
    arm.alignment_rank = rank
    return arm


def recovery_parameters(arm: RecoveryArm):
    for compensator in arm.compensators:
        yield from compensator.parameters()
    for alignment in arm.alignments:
        yield from alignment.parameters()


def install_recovery(arm_model, arm: RecoveryArm) -> list[tuple]:
    """Put each wrapper where its bank was, so the model actually runs it.

    Returns the installation record; `remove_recovery` restores the original
    modules, which is what lets one process fit several arms in turn without
    nesting wrappers.
    """
    by_identity = {id(wrapper.sparse): wrapper for wrapper in arm.wrappers}
    installations: list[tuple] = []
    # Snapshot the parents before mutating: a wrapper contains its bank, so
    # installing while walking would find the bank again inside the wrapper.
    candidates = []
    for parent in arm_model.model.modules():
        for name, child in parent.named_children():
            candidates.append((parent, name, child))
    for parent, name, child in candidates:
        wrapper = by_identity.get(id(child))
        if wrapper is not None:
            setattr(parent, name, wrapper)
            installations.append((parent, name, child))
    if len(installations) != len(arm.wrappers):
        raise RuntimeError(
            f"installed {len(installations)} of {len(arm.wrappers)} wrappers; a "
            "bank is not reachable from the model's module tree")
    return installations


def remove_recovery(installations: list[tuple]) -> None:
    for parent, name, original in reversed(installations):
        setattr(parent, name, original)


def freeze_arm_but_recovery(arm_model, arm: RecoveryArm, enable: str = "compensator") -> dict:
    """Freeze every arm tensor, then enable only the requested correction.

    This is the leakage boundary: after it, the only parameters that can carry
    information are the compensator's (and, if asked, the alignment's), and both
    see only `(h, S)`.
    """
    for parameter in arm_model.model.parameters():
        parameter.requires_grad_(False)
    enabled = []
    if enable in ("compensator", "both"):
        for compensator in arm.compensators:
            for parameter in compensator.parameters():
                parameter.requires_grad_(True)
                enabled.append(parameter)
    if enable in ("alignment", "both"):
        for alignment in arm.alignments:
            for parameter in alignment.parameters():
                parameter.requires_grad_(True)
                enabled.append(parameter)
    trainable = sum(parameter.numel() for parameter in
                    arm_model.model.parameters() if parameter.requires_grad)
    return {"trainable_elements": trainable, "enabled_tensors": len(enabled),
            "enable": enable}


def describe_arms(base_active: int, bank_k: int, bank_experts: int,
                  hidden: int = 1024, config: RecoveryConfig | None = None) -> dict:
    """The pre-registered arm matrix with its element counts.

    Pure arithmetic from the registered constants, so the registration and the
    runner cannot drift apart.
    """
    config = config or RecoveryConfig()
    layers = 28
    per_compensator = (bank_experts * config.set_dim
                       + (hidden + config.set_dim) * config.rank + config.rank
                       + config.rank * hidden + hidden)
    compensator_total = per_compensator * layers
    rank = matched_alignment_rank(per_compensator, hidden)
    alignment_total = (2 * hidden * rank + hidden) * layers
    arms = {
        "sparse-only": {"recovery_elements": 0},
        "sparse+alignment": {"recovery_elements": alignment_total},
        "sparse+compensator": {"recovery_elements": compensator_total},
        "sparse+alignment+compensator": {
            "recovery_elements": alignment_total + compensator_total},
        "dense-teacher-reference": {
            "recovery_elements": 0,
            "note": "all m groups at unit weight; not deployable, it is the "
                    "reference the other arms are scored against",
        },
    }
    for name, row in arms.items():
        row["realized_active"] = base_active + row["recovery_elements"]
        row["within_ceiling"] = row["realized_active"] <= 1_000_000_000
    return {
        "hidden": hidden,
        "layers": layers,
        "experts": bank_experts,
        "k": bank_k,
        "omission_rate": (bank_experts - bank_k) / bank_experts,
        "compensator_per_layer": per_compensator,
        "compensator_total": compensator_total,
        "alignment_rank": rank,
        "alignment_per_layer": 2 * hidden * rank + hidden,
        "alignment_total": alignment_total,
        "config": config.declared(),
        "arms": arms,
        "ceiling": 1_000_000_000,
        "matched_rule": "alignment rank is floored so its element count is at "
                        "most the compensator's, so every comparison is at or "
                        "below the compensator's bytes",
    }
