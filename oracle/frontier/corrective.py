#!/usr/bin/env python3
"""P19.2b: the corrective graft — a near-identity residual trained on advantage.

The S40 representation graft reconstructed the teacher's mapped hidden state at
the injection site and added it to the frozen student. Measured on the 2M-token
held-out slice (`results/stage2/s41-heldout-comparison.json`), that perturbed
the student's hidden state by roughly three quarters of its own norm and cost
+0.57475 nats of next-token CE against the base — it was trained on
reconstruction, not on the quantity the evaluation scores.

This module is the registered replacement. It keeps the same token-addressed
n-gram memory, the same eight reads per token, the same projections, the same
convolution and the same active-parameter accounting, and changes three things:

1. **near-identity initialisation.** The residual is multiplied by a correction
   gate `alpha(site) = tanh(gate_raw)` with `gate_raw` initialised to exactly
   zero. The value projection is *not* zero (a zero multiplicand would also
   block the gate's own gradient), but the gate is, so an untrained corrective
   graft returns exactly zero and the injected model reproduces the base
   bit for bit. The gate's gradient at zero is `-<dL/dh, read>`, which is
   non-zero, so the gate opens on the first optimiser step and the rows and
   projections receive gradient from then on.
2. **supervision from the teacher's advantage.** The row and projection values
   are learned by `corrective_objective` below, not by reconstructing mapped
   teacher states. The distillation term is weighted per position by
   `KL(teacher_T || base_T)`: the correction is asked to explain where the base
   is far from the teacher, not to reproduce the teacher's representation.
3. **interference protection.** A per-position preservation weight
   `sigmoid(-(advantage - mean advantage) / mean advantage)` is high where the
   base already predicts the teacher well. It multiplies (a) a penalty on any
   *increase* in KL over the base's own KL, and (b) a penalty on the gate
   magnitude, so opening the correction on tokens the base already handles is
   what costs loss.

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
GATE_STD = 0.02


class CorrectiveMemory(nn.Module):
    """Conditional memory read with a zero-initialised correction gate."""

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
        self.gate_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.sites])
        # Set by the training loop around the frozen base forward; the corrective
        # residual is skipped and the convolution state is not advanced.
        self.bypass = False

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

    def gate(self, site_index: int = 0) -> torch.Tensor:
        return torch.tanh(self.gate_raw[site_index])

    def gate_values(self) -> list[float]:
        return [float(gate.detach().reshape(-1)[0]) for gate in
                (torch.tanh(raw) for raw in self.gate_raw)]

    def identity_at_init(self) -> bool:
        return all(value == 0.0 for value in self.gate_values())

    def read(self, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """The gated 8-row read, structurally without the constant offset."""
        return self.core.site_mixed(site_index, hidden_states)

    def forward(self, hidden_states: torch.Tensor, site_index: int = 0
                ) -> torch.Tensor:
        if self.bypass:
            return torch.zeros_like(hidden_states)
        correction = self.read(site_index, hidden_states)
        return (self.gate(site_index) * correction).to(hidden_states.dtype)

    def active_parameter_report(self) -> dict:
        report = self.core.active_parameter_report()
        sites = len(self.sites)
        report.update({
            "kind": "stage2_corrective_active",
            "variant": "corrective",
            "core_report_kind": self.core.active_parameter_report()["kind"],
            "correction_gate_elements_per_site": 1,
            "correction_gate_elements": sites,
            "active_per_token_before_gate": report["active_per_token"],
            "active_per_token": report["active_per_token"] + sites,
            "gate_init": "tanh(0) = 0 exactly; untrained residual is exactly zero",
            "constant_offset_in_residual": False,
            "basis": (
                "same reads, projections and convolution as the representation "
                "graft, plus one correction-gate scalar per site"),
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


def corrective_objective(student_logits: torch.Tensor, base_logits: torch.Tensor,
                          teacher_logits: torch.Tensor, targets: torch.Tensor,
                          temperature: float, kl_weight: float,
                          objective: str = "correction",
                          gate: torch.Tensor | None = None,
                          preserve_weight: float = 0.5,
                          interference_weight: float = 0.05) -> tuple:
    """The registered objective for the corrective arm.

    ``ce_kl`` is the control objective used by every other arm,
    ``CE + kl_weight * T^2 * mean KL``. ``correction`` is advantage-weighted KL
    plus a base-preservation term and a gate-interference term:

        CE
        + w * T^2 * sum(a_i * KL_i) / sum(a_i)          # advantage-weighted
        + w * preserve_weight * T^2 * mean(p_i * max(KL_i - a_i, 0))
        + interference_weight * mean(p_i) * mean(alpha^2)

    with ``a_i = KL(teacher || base)_i`` the base's own advantage (detached) and
    ``p_i = sigmoid(-(a_i - mean(a)) / mean(a))`` high where the base is already
    close to the teacher.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    ce_positions = _per_position_ce(student_logits, targets)
    kl_positions = _per_position_kl(student_logits, teacher_logits, temperature)
    ce = ce_positions.mean()
    scale = kl_weight * (temperature ** 2)
    if objective == "ce_kl":
        loss = ce + scale * kl_positions.mean()
        diagnostics = {
            "objective": "ce_kl",
            "mean_kl": float(kl_positions.mean().detach()),
            "preservation_weight_mean": 0.0,
            "interference_penalty": 0.0,
            "gate_mean_abs": 0.0,
        }
        return loss, ce.detach(), kl_positions.mean().detach(), diagnostics

    advantage = _per_position_kl(base_logits, teacher_logits, temperature).detach()
    mean_advantage = advantage.mean().clamp_min(1e-6)
    advantage_weight = advantage / mean_advantage
    preservation = torch.sigmoid(-(advantage - mean_advantage) / mean_advantage)
    weighted = (advantage_weight * kl_positions).sum() / advantage_weight.sum()
    increase = (kl_positions - advantage).clamp_min(0.0)
    preservation_penalty = (preservation * increase).mean()
    gate_mean_abs = (gate.detach().abs().mean() if gate is not None
                     else torch.zeros((), device=kl_positions.device))
    interference = preservation.mean() * (gate ** 2).mean() if gate is not None else \
        torch.zeros((), device=kl_positions.device)
    loss = (ce + scale * weighted
            + scale * preserve_weight * preservation_penalty
            + interference_weight * interference)
    diagnostics = {
        "objective": "correction",
        "mean_kl": float(kl_positions.mean().detach()),
        "mean_advantage": float(mean_advantage),
        "preserved_fraction": float((preservation > 0.5).float().mean()),
        "advantage_weight_max": float(advantage_weight.max()),
        "base_preservation_penalty": float(preservation_penalty.detach()),
        "gate_mean_abs": float(gate_mean_abs),
        "interference_penalty": float(interference.detach()),
    }
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
    return {
        "sample_tokens": int(take),
        "gate_values": module.gate_values(),
        "identity_at_init": module.identity_at_init(),
        "residual_norm": float(output.norm()),
        "residual_max_abs": float(output.abs().max()),
        "hidden_norm_mean": float(hidden.norm(dim=-1).mean()),
        "read_norm_mean": float(read.norm(dim=-1).mean()),
        "read_to_hidden_norm_ratio": float(
            read.norm(dim=-1).mean() / hidden.norm(dim=-1).mean()),
        "note": (
            "the residual is alpha * read with alpha = tanh(0) = 0, so it is "
            "exactly zero and the untrained model is the base bit for bit; the "
            "read ratio is the magnitude the gate opens onto once trained"),
    }
