#!/usr/bin/env python3
"""S45: objective assembly for the RFT / on-policy route.

This module adds no new mechanism.  It wires the terms already registered in
S42/S43/S44 into the next route:

* ``sft`` — cross-entropy on verified teacher solutions (the rejection-sampling
  fine-tuning arm);
* ``kd`` — cross-entropy plus teacher KL on student rollouts (on-policy
  distillation);
* ``kd_correction`` — the registered corrective objective: advantage-weighted
  KL, the per-token base-preservation penalty, and the gate-interference term
  (``corrective.corrective_objective``), reused verbatim;
* ``both`` — a weighted sum of an SFT batch and a KD batch.

The math-format curriculum is target-side only.  It prefixes ``Solution:``
and/or numbers non-empty reasoning lines with ``Step k:``; the evaluation
prompt is untouched.  It is an optional registered arm, not a default.

Nothing here is trained in this session.  The CPU tests exercise the arithmetic
on toy logits and the formatter on hand-built strings.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import corrective  # noqa: E402
from frontier.rft import format_curriculum  # noqa: E402

MODES = ("sft", "kd", "kd_correction", "both")


def sft_loss(student_logits: torch.Tensor, targets: torch.Tensor,
             sample_weight: torch.Tensor | None = None,
             label_smoothing: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Completion-only token cross-entropy, optionally weighted per position.

    Positions whose target is `IGNORE_INDEX` (-100) are prompt or padding and
    contribute neither to the loss nor to the normalising count. A batch with no
    supervised position returns a zero loss that still carries the graph.
    """
    flat_logits = student_logits.reshape(-1, student_logits.shape[-1]).float()
    flat_targets = targets.reshape(-1)
    valid = flat_targets != -100
    if not bool(valid.any()):
        zero = flat_logits.sum() * 0.0
        return zero, zero.detach()
    supervised_logits = flat_logits[valid]
    supervised_targets = flat_targets[valid]
    positions = F.cross_entropy(supervised_logits, supervised_targets,
                                reduction="none",
                                label_smoothing=label_smoothing)
    if sample_weight is not None:
        weight = sample_weight.reshape(-1).float()[valid]
        if weight.numel() != positions.numel():
            raise ValueError(
                f"sample_weight has {weight.numel()} supervised elements, loss has "
                f"{positions.numel()}")
        loss = (positions * weight).sum() / weight.sum().clamp_min(1e-6)
    else:
        loss = positions.mean()
    return loss, positions.mean().detach()


def distill_loss(student_logits: torch.Tensor, base_logits: torch.Tensor,
                 teacher_logits: torch.Tensor, targets: torch.Tensor,
                 mode: str = "kd", temperature: float = 2.0,
                 kl_weight: float = 0.5, weighting: str = "kl",
                 preserve_weight: float = 0.0,
                 interference_weight: float = 0.05,
                 gate: torch.Tensor | None = None) -> tuple:
    """Return ``(loss, ce, kl, diagnostics)`` for a KD batch."""
    if mode == "kd":
        if weighting != "kl" or preserve_weight != 0.0:
            # ``kd`` is the plain control; the correction terms are a separate
            # registered mode so a result is attributable.
            raise ValueError("mode 'kd' uses the plain CE+KL control; use "
                             "'kd_correction' for the weighted/preserved form")
        return corrective.corrective_objective(
            student_logits, base_logits, teacher_logits, targets,
            temperature=temperature, kl_weight=kl_weight,
            objective="ce_kl", weighting="kl", gate=None)
    if mode == "kd_correction":
        return corrective.corrective_objective(
            student_logits, base_logits, teacher_logits, targets,
            temperature=temperature, kl_weight=kl_weight,
            objective="correction", weighting=weighting, gate=gate,
            preserve_weight=preserve_weight,
            interference_weight=interference_weight)
    raise ValueError(f"distill_loss does not handle mode {mode!r}")


def combined_loss(sft_batch: dict | None, kd_batch: dict | None,
                  sft_weight: float = 0.5, kd_weight: float = 0.5,
                  temperature: float = 2.0, kl_weight: float = 0.5,
                  weighting: str = "kl", preserve_weight: float = 0.5,
                  interference_weight: float = 0.05,
                  sft_label_smoothing: float = 0.0) -> tuple:
    """The registered ``both`` arm: weighted SFT batch plus KD batch."""
    if sft_batch is None and kd_batch is None:
        raise ValueError("combined_loss needs at least one batch")
    total = None
    components = {}
    if sft_batch is not None:
        sft, sft_ce = sft_loss(sft_batch["student_logits"],
                               sft_batch["targets"],
                               sft_batch.get("sample_weight"),
                               label_smoothing=sft_label_smoothing)
        total = sft_weight * sft
        components["sft"] = float(sft.detach())
        components["sft_ce"] = float(sft_ce)
    if kd_batch is not None:
        kd_mode = kd_batch.get("mode", "kd")
        kd_preserve = preserve_weight if kd_mode == "kd_correction" else 0.0
        kd, kd_ce, kd_kl, diagnostics = distill_loss(
            kd_batch["student_logits"], kd_batch["base_logits"],
            kd_batch["teacher_logits"], kd_batch["targets"],
            mode=kd_mode, temperature=temperature,
            kl_weight=kl_weight, weighting=weighting,
            preserve_weight=kd_preserve,
            interference_weight=interference_weight,
            gate=kd_batch.get("gate"))
        contribution = kd_weight * kd
        total = contribution if total is None else total + contribution
        components["kd"] = float(kd.detach())
        components["kd_ce"] = float(kd_ce)
        components["kd_kl"] = float(kd_kl)
        components["kd_diagnostics"] = diagnostics
    return total, components


__all__ = ["MODES", "sft_loss", "distill_loss", "combined_loss",
           "format_curriculum"]
