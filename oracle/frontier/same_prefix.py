#!/usr/bin/env python3
"""Completion-only SFT with teacher KL on the SAME causal student prefix.

The S54 registration proposed correcting a failed rollout from its "first
error". That phrase named the first *token* at which the student and a verified
teacher continuation differ, which is a textual divergence: it can be a
paraphrase, an equivalent algebraic form, or a different but valid step. It is
not a reasoning error and it is not a step boundary. This module uses the
correct name, ``first_textual_divergence``, and keeps the quantity as a
diagnostic rather than a loss switch.

The immediate objective it implements is:

* **CE** on the verified teacher solution, completion-only: prompt and padding
  positions carry ``IGNORE_INDEX`` and are neither a loss nor part of the
  normalising count;
* **teacher KL** over the same supervised positions, with the teacher logits
  produced by a forward pass over the *student's own token ids* — the same
  causal prefix the student saw. The teacher is never run over its own generated
  history and no teacher tokens are inserted into the student's context. In a
  teacher-forced batch this is exact by construction: ``inputs`` is the student
  prefix, and both models consume it.

If a future experiment wants reward-weighted on-policy distillation from teacher
continuations, the teacher must still be conditioned on the student's prefix;
running it on the teacher's own continuation is a different objective and needs
its own registration (and its own alignment rule).

CPU-testable with small tensors; no model or tokenizer is loaded here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def completion_mask(targets: torch.Tensor) -> torch.Tensor:
    """Positions that contribute a loss: everything except prompt/padding."""
    return targets.reshape(-1) != IGNORE_INDEX


def completion_ce(student_logits: torch.Tensor, targets: torch.Tensor,
                  vocab_chunk: int = 32) -> torch.Tensor:
    """Mean cross-entropy over supervised positions only.

    Fails closed when no position is supervised: an all-masked batch would
    otherwise return a silent zero gradient.
    """
    flat = student_logits.reshape(-1, student_logits.shape[-1])
    labels = targets.reshape(-1)
    mask = labels != IGNORE_INDEX
    if not bool(mask.any()):
        raise ValueError("no supervised positions; refusing an all-masked CE")
    total = torch.zeros((), device=student_logits.device)
    for start in range(0, flat.shape[0], vocab_chunk):
        chunk_logits = flat[start:start + vocab_chunk].float()
        chunk_labels = labels[start:start + vocab_chunk]
        chunk_mask = mask[start:start + vocab_chunk]
        if not bool(chunk_mask.any()):
            continue
        total = total + F.cross_entropy(
            chunk_logits[chunk_mask], chunk_labels[chunk_mask], reduction="sum")
    return total / mask.sum()


def same_prefix_teacher_kl(student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor,
                           targets: torch.Tensor, temperature: float,
                           vocab_chunk: int = 32) -> torch.Tensor:
    """KL(student_T || teacher_T) over supervised positions, same prefix.

    Both logit tensors must have the same shape and must have been produced
    from the same input ids. The caller enforces the input-id equality with
    :func:`assert_same_prefix`; this function enforces the tensor contract.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits differ in shape; they cannot describe the "
            "same causal prefix")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat_student = student_logits.reshape(-1, student_logits.shape[-1])
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    labels = targets.reshape(-1)
    mask = labels != IGNORE_INDEX
    if not bool(mask.any()):
        raise ValueError("no supervised positions; refusing an all-masked KL")
    total = torch.zeros((), device=student_logits.device)
    for start in range(0, flat_student.shape[0], vocab_chunk):
        chunk_mask = mask[start:start + vocab_chunk]
        if not bool(chunk_mask.any()):
            continue
        student = flat_student[start:start + vocab_chunk][chunk_mask].float()
        teacher = flat_teacher[start:start + vocab_chunk][chunk_mask].float()
        log_student = F.log_softmax(student / temperature, dim=-1)
        log_teacher = F.log_softmax(teacher / temperature, dim=-1)
        # Reverse KL, as registered: KL(student_T || teacher_T), the student is
        # the distribution being integrated over (S54: "the student is the
        # sampling policy; reverse KL keeps the objective on the student's own
        # state distribution").
        total = total + (log_student.exp()
                         * (log_student - log_teacher)).sum()
    return total / mask.sum()


def sft_with_same_prefix_kl(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            targets: torch.Tensor, temperature: float,
                            kl_weight: float) -> tuple[torch.Tensor, ...]:
    """The registered stage-2a objective: completion CE + same-prefix KL."""
    ce = completion_ce(student_logits, targets)
    if kl_weight == 0.0:
        zero = torch.zeros((), device=student_logits.device)
        return ce, ce.detach(), zero
    kl = same_prefix_teacher_kl(student_logits, teacher_logits, targets,
                                temperature)
    return ce + kl_weight * (temperature ** 2) * kl, ce.detach(), kl.detach()


def assert_same_prefix(student_input_ids: torch.Tensor,
                       teacher_input_ids: torch.Tensor) -> None:
    """Fail closed unless the teacher consumed the student's exact prefix."""
    if student_input_ids.shape != teacher_input_ids.shape:
        raise ValueError(
            "teacher and student prefixes differ in shape: the teacher is not "
            "conditioned on the student's causal prefix")
    if not torch.equal(student_input_ids, teacher_input_ids):
        difference = (student_input_ids != teacher_input_ids).reshape(-1).nonzero()
        first = int(difference[0].item()) if difference.numel() else -1
        raise ValueError(
            "teacher and student prefixes differ (first differing flat index "
            f"{first}): the teacher must consume the student's own prefix")


def first_textual_divergence(student_ids, teacher_ids) -> int | None:
    """Index of the first token where the two continuations differ.

    **This is a textual divergence, not a reasoning error.** It says nothing
    about whether either continuation is correct, and an equivalent step
    written differently diverges here. It is reported as a diagnostic only.

    Returns ``None`` when one continuation is a prefix of the other (there is no
    differing token), matching the S54 rule "the first position after the
    shorter prefix is the first divergence" only when a suffix actually exists;
    a pure prefix relationship is reported as no divergence.
    """
    shorter = min(len(student_ids), len(teacher_ids))
    for index in range(shorter):
        if student_ids[index] != teacher_ids[index]:
            return index
    return None


__all__ = [
    "IGNORE_INDEX", "completion_mask", "completion_ce",
    "same_prefix_teacher_kl", "sft_with_same_prefix_kl", "assert_same_prefix",
    "first_textual_divergence",
]
