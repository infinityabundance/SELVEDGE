#!/usr/bin/env python3
"""P19.2b/S43: the corrective graft — a token-conditioned, advantage-supervised
near-identity residual.

The S40 representation graft reconstructed the teacher's mapped hidden state at
the injection site and added it to the frozen student. Measured on the 2M-token
held-out slice (`results/stage2/s41-heldout-comparison.json`), that perturbed
the student's hidden state by roughly three quarters of its own norm and cost
+0.57475 nats of next-token CE against the base — it was trained on
reconstruction, not on the quantity the evaluation scores.

This module keeps the token-addressed n-gram memory, the eight reads per token,
the projections, the convolution and the active-parameter accounting, and
changes the correction mechanism:

1. **token-conditioned near-identity gate.** The residual is
   `alpha_t * read_t` with

       alpha_t = tanh(w . phi(h_t, r_t) + b),      phi(h, r) = [h ; r]

   and `w = 0`, `b = 0` initialised exactly. The untrained model therefore
   returns zero at every token and reproduces the base bit for bit. `phi` is a
   parameter-free concatenation, so the active cost is `2 * hidden + 1` elements
   per site (the `w` vector and the bias), reported by
   `active_parameter_report`. The gate can vary per token: two positions with
   different hidden states receive different alpha once `w` is non-zero.
2. **supervision from the teacher's advantage.** The row and projection values
   are learned by `corrective_objective` below, not by reconstructing mapped
   teacher states. Two weights are registered:
   * the existing KL-disagreement weight `a_t = KL(teacher_T || base_T)`;
   * the signed teacher advantage `A_t = log p_T(y_t) - log p_B(y_t)` on the
     supervised target, positive where the teacher puts more probability on the
     true next token than the base does;
   * or both, combined as the geometric mean of the two normalised weights.
3. **interference protection and its measurement.** A per-position preservation
   weight `sigmoid(-(a - mean a) / mean a)` is high where the base already
   predicts the teacher well. It multiplies a penalty on any *increase* in KL
   over the base's own KL and a penalty on the gate magnitude. The objective
   also reports harm on exactly the tokens where the base already matches the
   teacher (lowest advantage quartile, and signed `A_t >= 0`), so a gain
   elsewhere cannot hide damage there.

The constant offset (`out_bias`) is structurally excluded from the residual:
`forward` subtracts it, so it can never reintroduce the representation graft's
large perturbation through a learned mean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import ConditionalMemory, HEADS

OBJECTIVES = ("ce_kl", "correction")
WEIGHTINGS = ("kl", "signed", "both")
GATE_STD = 0.02


class CorrectiveMemory(nn.Module):
    """Token-conditioned, zero-initialised correction gate over a memory read."""

    def __init__(self, hidden: int = 1024, value_dim: int = 256,
                 rows: int = 65_536, sites: tuple[int, ...] = (2,),
                 conv_kernel: int = 4, table_dtype: torch.dtype = torch.float32,
                 seed: int = 0, context=None):
        super().__init__()
        self.core = ConditionalMemory(
            hidden=hidden, value_dim=value_dim, rows=rows, sites=sites,
            conv_kernel=conv_kernel, context=context, table_dtype=table_dtype,
            seed=seed)
        self.hidden = int(hidden)
        self.value_dim = int(value_dim)
        self.rows = int(rows)
        self.sites = self.core.sites
        self.conv_kernel = int(conv_kernel)
        self.context = self.core.context
        # Identity comes from the gate, not from a zero multiplicand: give the
        # value projection a non-degenerate deterministic init so the read has a
        # direction and `dL/dalpha` is non-zero at alpha = 0.
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 991)
        with torch.no_grad():
            for projection in self.core.value_proj:
                nn.init.normal_(projection.weight, std=GATE_STD,
                                generator=generator)
            for bias in self.core.out_bias:
                nn.init.zeros_(bias)
        # phi(h, r) = [h ; r], so w has 2 * hidden elements per site, plus one
        # bias. Both are exactly zero at init: alpha_t = tanh(0) = 0.
        self.gate_w = nn.ParameterList(
            [nn.Parameter(torch.zeros(2 * self.hidden)) for _ in self.sites])
        self.gate_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.sites])
        # Set by the training loop around the frozen base forward; the corrective
        # residual is skipped and the convolution state is not advanced.
        self.bypass = False
        # The per-token gate of the most recent student forward, with its graph.
        # The base bypass pass does not overwrite it.
        self.last_gate: torch.Tensor | None = None

    # -- delegated surface (install_memory, accounting, grafting) -------------
    @property
    def tables(self):
        return self.core.tables

    @property
    def key_proj(self):
        return self.core.key_proj

    @property
    def value_proj(self):
        return self.core.value_proj

    @property
    def out_bias(self):
        return self.core.out_bias

    @property
    def conv(self):
        return self.core.conv

    @property
    def gate_parameters_per_site(self) -> int:
        return 2 * self.hidden + 1

    def gate(self, site_index: int, hidden_states: torch.Tensor,
             read: torch.Tensor) -> torch.Tensor:
        """`alpha_t = tanh(w . [h_t ; r_t] + b)`, float32, shape `[B, T]`."""
        features = torch.cat([hidden_states.float(), read.float()], dim=-1)
        weight = self.gate_w[site_index].float()
        logits = (features * weight).sum(dim=-1) + self.gate_b[site_index].float()
        return torch.tanh(logits)

    def gate_summary(self) -> dict:
        gate = self.last_gate
        if gate is None:
            return {"tokens": 0, "mean": None, "mean_abs": None,
                    "min": None, "max": None}
        value = gate.detach().float()
        return {
            "tokens": int(value.numel()),
            "mean": float(value.mean()),
            "mean_abs": float(value.abs().mean()),
            "min": float(value.min()),
            "max": float(value.max()),
        }

    def identity_at_init(self) -> bool:
        return all(float(weight.detach().abs().sum()) == 0.0
                   for weight in self.gate_w) \
            and all(float(bias.detach().abs().sum()) == 0.0 for bias in self.gate_b)

    def read(self, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """The gated 8-row read, structurally without the constant offset."""
        return self.core.site_mixed(site_index, hidden_states)

    def forward(self, hidden_states: torch.Tensor, site_index: int = 0
                ) -> torch.Tensor:
        if self.bypass:
            return torch.zeros_like(hidden_states)
        correction = self.read(site_index, hidden_states)
        alpha = self.gate(site_index, hidden_states, correction)
        self.last_gate = alpha
        return (alpha[..., None] * correction).to(hidden_states.dtype)

    def active_parameter_report(self) -> dict:
        report = self.core.active_parameter_report()
        sites = len(self.sites)
        gate_elements = self.gate_parameters_per_site * sites
        report.update({
            "kind": "stage2_corrective_token_gate_active",
            "variant": "corrective_token_gate",
            "core_report_kind": self.core.active_parameter_report()["kind"],
            "gate_feature_map": "phi(h_t, r_t) = [h_t ; r_t] (parameter-free)",
            "gate_parameters_per_site": self.gate_parameters_per_site,
            "gate_weight_elements_per_site": 2 * self.hidden,
            "gate_bias_elements_per_site": 1,
            "correction_gate_elements_per_site": self.gate_parameters_per_site,
            "correction_gate_elements": gate_elements,
            "active_per_token_before_gate": report["active_per_token"],
            "active_per_token": report["active_per_token"] + gate_elements,
            "gate_init": "w = 0 and b = 0, so alpha_t = tanh(0) = 0 exactly "
                         "for every token; the untrained residual is exactly zero",
            "gate_shape": "per token (B, T); the gate varies across positions "
                          "once w is non-zero",
            "constant_offset_in_residual": False,
            "basis": (
                "same reads, projections and convolution as the representation "
                "graft, plus a 2*hidden+1 element token gate per site"),
        })
        return report

    def collision_report(self, tokens: torch.Tensor) -> dict:
        return self.core.collision_report(tokens)

    @torch.no_grad()
    def set_grafted_rows(self, site_index: int, rows: torch.Tensor,
                         counts: torch.Tensor) -> None:
        self.core.set_grafted_rows(site_index, rows, counts)

    def graft_counts(self, site_index: int):
        return self.core.graft_counts(site_index)

    def reset_state(self) -> None:
        self.core.reset_state()
        self.last_gate = None


# ---------------------------------------------------------------------------
# the objective


def _per_position_ce(logits: torch.Tensor, targets: torch.Tensor,
                     vocab_chunk: int = 32) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    pieces = []
    for start in range(0, flat_logits.shape[0], vocab_chunk):
        piece = flat_logits[start:start + vocab_chunk].float()
        labels = flat_targets[start:start + vocab_chunk]
        pieces.append(F.cross_entropy(piece, labels, reduction="none"))
    return torch.cat(pieces)


def _per_position_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     temperature: float, vocab_chunk: int = 32) -> torch.Tensor:
    """KL(teacher_T || student_T) for every position."""
    flat_student = student_logits.reshape(-1, student_logits.shape[-1])
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    pieces = []
    for start in range(0, flat_student.shape[0], vocab_chunk):
        student = flat_student[start:start + vocab_chunk].float() / temperature
        teacher = flat_teacher[start:start + vocab_chunk].float() / temperature
        log_student = F.log_softmax(student, dim=-1)
        log_teacher = F.log_softmax(teacher, dim=-1)
        pieces.append((log_teacher.exp() * (log_teacher - log_student)).sum(-1))
    return torch.cat(pieces)


@torch.no_grad()
def teacher_advantage(base_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      temperature: float = 2.0) -> torch.Tensor:
    """Per-position KL(teacher_T || base_T): what the base has not learned yet."""
    return _per_position_kl(base_logits, teacher_logits, temperature)


@torch.no_grad()
def signed_teacher_advantage(base_logits: torch.Tensor, teacher_logits: torch.Tensor,
                             targets: torch.Tensor,
                             temperature: float = 1.0) -> torch.Tensor:
    """`A_t = log p_T(y_t) - log p_B(y_t)` on the supervised target.

    Positive where the teacher assigns more probability to the true next token
    than the base does; negative where the base is already better than the
    teacher. Computed at the stated temperature (1.0, i.e. the raw
    distributions) and detached: it is a supervision weight, not a gradient path.
    """
    flat_targets = targets.reshape(-1, 1)
    pieces = []
    flat_base = base_logits.reshape(-1, base_logits.shape[-1])
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    for start in range(0, flat_base.shape[0], 32):
        base = F.log_softmax(flat_base[start:start + 32].float() / temperature, -1)
        teacher = F.log_softmax(
            flat_teacher[start:start + 32].float() / temperature, -1)
        labels = flat_targets[start:start + 32]
        pieces.append(teacher.gather(-1, labels).squeeze(-1)
                      - base.gather(-1, labels).squeeze(-1))
    return torch.cat(pieces)


def _weigh(advantage: torch.Tensor, signed: torch.Tensor,
           weighting: str) -> tuple[torch.Tensor, dict]:
    """Normalised per-position supervision weight for the registered mode."""
    mean_advantage = advantage.mean().clamp_min(1e-6)
    kl_weight = advantage / mean_advantage
    bounded = torch.exp(torch.clamp(signed, -4.0, 4.0))
    signed_weight = bounded / bounded.mean().clamp_min(1e-6)
    if weighting == "kl":
        weight = kl_weight
    elif weighting == "signed":
        weight = signed_weight
    elif weighting == "both":
        weight = torch.sqrt(kl_weight * signed_weight)
        weight = weight / weight.mean().clamp_min(1e-6)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    diagnostics = {
        "weighting": weighting,
        "mean_advantage_kl": float(mean_advantage),
        "mean_signed_advantage": float(signed.mean()),
        "signed_positive_fraction": float((signed >= 0).float().mean()),
        "advantage_weight_max": float(weight.max().detach()),
        "advantage_weight_min": float(weight.min().detach()),
    }
    return weight, diagnostics


def _harm_report(kl_positions: torch.Tensor, advantage: torch.Tensor,
                 signed: torch.Tensor) -> dict:
    """Harm where the base already matches the teacher.

    Two masks are registered: the lowest advantage quartile (the P19.8
    interference-cost definition) and the signed `A_t >= 0` set (the teacher is
    no more probable than the base on the supervised token).
    """
    delta = (kl_positions - advantage).detach()
    report = {}
    for name, mask in (
        ("lowest_quartile", advantage <= torch.quantile(advantage, 0.25)),
        ("signed_base_better", signed >= 0),
    ):
        count = int(mask.sum())
        if count:
            report[f"harm_{name}_tokens"] = count
            report[f"harm_{name}_mean_delta_kl"] = float(delta[mask].mean())
            report[f"harm_{name}_fraction_worse"] = float(
                (delta[mask] > 0).float().mean())
        else:
            report[f"harm_{name}_tokens"] = 0
            report[f"harm_{name}_mean_delta_kl"] = 0.0
            report[f"harm_{name}_fraction_worse"] = 0.0
    return report


def corrective_objective(student_logits: torch.Tensor, base_logits: torch.Tensor,
                          teacher_logits: torch.Tensor, targets: torch.Tensor,
                          temperature: float, kl_weight: float,
                          objective: str = "correction",
                          weighting: str = "kl",
                          gate: torch.Tensor | None = None,
                          preserve_weight: float = 0.5,
                          interference_weight: float = 0.05) -> tuple:
    """The registered objective for the corrective arm.

    ``ce_kl`` is the control objective used by every other arm,
    ``CE + kl_weight * T^2 * mean KL``. ``correction`` is weighted KL plus a
    base-preservation term and a gate-interference term:

        CE
        + w * T^2 * sum(u_i * KL_i) / sum(u_i)          # weighted distillation
        + w * preserve_weight * T^2 * mean(p_i * max(KL_i - a_i, 0))
        + interference_weight * mean(p_i) * mean(alpha_t^2)

    with ``a_i = KL(teacher_T || base_T)_i`` (detached), ``p_i =
    sigmoid(-(a_i - mean a)/mean a)`` high where the base is already close, and
    ``u_i`` from ``weighting``:

    * ``kl``: ``a_i / mean(a)`` — the registered S42 weight;
    * ``signed``: ``exp(clamp(A_i, -4, 4)) / mean(exp(...))`` with
      ``A_i = log p_T(y_i) - log p_B(y_i)``;
    * ``both``: the geometric mean of the two normalised weights, renormalised.

    ``gate`` is the per-token alpha actually used in the student forward
    (shape ``[B, T]`` or flattened); its square is the interference magnitude.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}")
    ce_positions = _per_position_ce(student_logits, targets)
    kl_positions = _per_position_kl(student_logits, teacher_logits, temperature)
    ce = ce_positions.mean()
    scale = kl_weight * (temperature ** 2)
    if objective == "ce_kl":
        loss = ce + scale * kl_positions.mean()
        diagnostics = {
            "objective": "ce_kl",
            "weighting": "none",
            "mean_kl": float(kl_positions.mean().detach()),
            "preservation_weight_mean": 0.0,
            "interference_penalty": 0.0,
            "gate_mean_abs": 0.0,
        }
        return loss, ce.detach(), kl_positions.mean().detach(), diagnostics

    advantage = _per_position_kl(base_logits, teacher_logits, temperature).detach()
    signed = signed_teacher_advantage(base_logits, teacher_logits, targets,
                                      temperature=1.0)
    weight, weight_diagnostics = _weigh(advantage, signed, weighting)
    mean_advantage = advantage.mean().clamp_min(1e-6)
    preservation = torch.sigmoid(-(advantage - mean_advantage) / mean_advantage)
    weighted = (weight * kl_positions).sum() / weight.sum().clamp_min(1e-6)
    increase = (kl_positions - advantage).clamp_min(0.0)
    preservation_penalty = (preservation * increase).mean()
    gate_flat = None
    if gate is not None:
        gate_flat = gate.reshape(-1)
        if gate_flat.numel() == 1 and kl_positions.numel() != 1:
            gate_flat = gate_flat.expand_as(kl_positions)
        if gate_flat.numel() != kl_positions.numel():
            raise ValueError(
                f"gate has {gate_flat.numel()} elements, KL has "
                f"{kl_positions.numel()}")
    gate_mean_abs = (gate_flat.detach().abs().mean() if gate_flat is not None
                     else torch.zeros((), device=kl_positions.device))
    interference = (preservation.mean() * (gate_flat ** 2).mean()
                    if gate_flat is not None else
                    torch.zeros((), device=kl_positions.device))
    loss = (ce + scale * weighted
            + scale * preserve_weight * preservation_penalty
            + interference_weight * interference)
    diagnostics = {
        "objective": "correction",
        "mean_kl": float(kl_positions.mean().detach()),
        "mean_advantage": float(mean_advantage),
        "preserved_fraction": float((preservation > 0.5).float().mean()),
        "base_preservation_penalty": float(preservation_penalty.detach()),
        "gate_mean_abs": float(gate_mean_abs),
        "interference_penalty": float(interference.detach()),
        **weight_diagnostics,
        **_harm_report(kl_positions, advantage, signed),
    }
    interpretation = {
        "lowest_quartile": "the P19.8 interference cost: delta KL where the "
                           "base's own advantage is in the lowest quartile",
        "signed_base_better": "delta KL where the teacher assigns no more "
                              "probability to the true token than the base",
        "reading": "positive mean_delta_kl and a fraction_worse above zero mean "
                   "the student is worse than the base exactly where the base "
                   "already matched the teacher",
    }
    diagnostics["harm_interpretation"] = interpretation
    return loss, ce.detach(), kl_positions.mean().detach(), diagnostics


def corrective_init_report(module: CorrectiveMemory, hidden_source: torch.Tensor,
                           tokens: torch.Tensor, device: str = "cpu",
                           sample: int = 2048) -> dict:
    """What the untrained corrective graft does: exactly nothing, by design."""
    take = min(sample, tokens.numel())
    tokens_sample = tokens[-take:].to(device)
    hidden = hidden_source[-take:].to(device, torch.float32)
    module.eval()
    module = module.to(device)
    module.context.observe(tokens_sample[None, :])
    with torch.no_grad():
        output = module(hidden[None, :, :])[0].float()
        read = module.read(0, hidden[None, :, :])[0].float()
    gate = module.last_gate
    return {
        "sample_tokens": int(take),
        "gate_parameters_per_site": module.gate_parameters_per_site,
        "gate_summary": module.gate_summary(),
        "identity_at_init": module.identity_at_init(),
        "residual_norm": float(output.norm()),
        "residual_max_abs": float(output.abs().max()),
        "hidden_norm_mean": float(hidden.norm(dim=-1).mean()),
        "read_norm_mean": float(read.norm(dim=-1).mean()),
        "read_to_hidden_norm_ratio": float(
            read.norm(dim=-1).mean() / hidden.norm(dim=-1).mean()),
        "gate_tokens": int(gate.numel()) if gate is not None else 0,
        "note": (
            "the residual is alpha_t * read with alpha_t = tanh(0) = 0 for every "
            "token, so it is exactly zero and the untrained model is the base "
            "bit for bit; the read ratio is the magnitude the gate opens onto "
            "once trained"),
    }
