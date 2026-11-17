#!/usr/bin/env python3
"""Held-out CE/KL and comparison-table arithmetic (S40).

The instrument answers one question for every arm: on tokens never seen in
training or grafting, how close is the arm to the online teacher, and how much
next-token cross-entropy does it pay? It reports both because they disagree when
an arm is sharp but wrong: CE is the student's own likelihood, KL is its
agreement with the teacher at the registered temperature.

Chunking. `token_loss` in `oracle/frontier/train.py` computes the same two
quantities for the training loop in position chunks of 32 so the full-vocabulary
softmax never materialises at full length. The evaluation path uses the same
chunked arithmetic (and a test checks the two agree) but averages *per chunk*
so a standard error across chunks can be reported. Chunks are non-overlapping
and the last partial chunk is dropped; the count is recorded.

Nothing here touches a GPU by itself: `score_stream` takes two callables that
return logits for `input_ids`, so the CPU self-test uses tiny fake models.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_VOCAB_CHUNK = 32


def chunk_ce_kl(student_logits: torch.Tensor, targets: torch.Tensor,
                teacher_logits: torch.Tensor | None = None,
                temperature: float = 2.0,
                vocab_chunk: int = DEFAULT_VOCAB_CHUNK) -> tuple[float, float, int]:
    """Mean CE and mean KL(T) over one chunk's positions.

    Shapes: student `[B, T, V]`, teacher `[B, T, V]` or None, targets `[B, T]`.
    Matches `oracle/frontier/train.py:token_loss` element for element.
    """
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    if teacher_logits is not None:
        teacher = teacher_logits.reshape(-1, vocab)
        if teacher.shape != student.shape:
            raise ValueError("student and teacher logits disagree in shape")
    ce_sum = torch.zeros((), device=student.device, dtype=torch.float32)
    kl_sum = torch.zeros((), device=student.device, dtype=torch.float32)
    positions = 0
    for start in range(0, student.shape[0], vocab_chunk):
        piece = student[start:start + vocab_chunk].float()
        target = flat_targets[start:start + vocab_chunk]
        ce_sum = ce_sum + F.cross_entropy(piece, target, reduction="sum")
        if teacher_logits is not None:
            teacher_piece = teacher[start:start + vocab_chunk].float() / temperature
            student_piece = piece / temperature
            log_student = F.log_softmax(student_piece, dim=-1)
            log_teacher = F.log_softmax(teacher_piece, dim=-1)
            kl_sum = kl_sum + (log_teacher.exp() * (log_teacher - log_student)).sum()
        positions += piece.shape[0]
    return (float(ce_sum) / max(1, positions),
            float(kl_sum) / max(1, positions), positions)


@torch.no_grad()
def score_stream(student, teacher, tokens: torch.Tensor, seq_len: int,
                 device: str, temperature: float = 2.0,
                 vocab_chunk: int = DEFAULT_VOCAB_CHUNK) -> dict:
    """Score `tokens` in non-overlapping next-token chunks.

    `student(input_ids=..., use_cache=False).logits` and the same for `teacher`
    (or None). Returns per-chunk rows plus exact token counts.
    """
    tokens = tokens.to(torch.int64).reshape(-1)
    rows = []
    start = 0
    while start + seq_len + 1 <= tokens.numel():
        chunk = tokens[start:start + seq_len + 1]
        inputs = chunk[:-1].unsqueeze(0).to(device)
        targets = chunk[1:].unsqueeze(0).to(device)
        student_logits = student(input_ids=inputs, use_cache=False).logits
        teacher_logits = None
        if teacher is not None:
            teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
        ce, kl, positions = chunk_ce_kl(student_logits, targets, teacher_logits,
                                        temperature, vocab_chunk)
        rows.append({"start": int(start), "positions": int(positions),
                     "ce": ce, "kl": kl,
                     "perplexity": math.exp(ce) if ce < 700 else None})
        start += seq_len
    if not rows:
        raise ValueError(f"stream has {tokens.numel()} tokens, seq_len is {seq_len}")
    return {"rows": rows, "chunks": len(rows),
            "tokens_scored": sum(row["positions"] for row in rows)}


def summary(values: list[float]) -> dict:
    """Mean, sample standard error and interval over per-chunk values."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "stderr": None, "sd": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "stderr": None, "sd": None}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    sd = math.sqrt(variance)
    stderr = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "stderr": stderr, "sd": sd,
            "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr]}


def paired_delta(values: list[float], reference: list[float]) -> dict:
    """Paired per-chunk difference and its standard error."""
    if len(values) != len(reference):
        raise ValueError("paired delta needs equal chunk counts")
    differences = [value - base for value, base in zip(values, reference)]
    result = summary(differences)
    result["basis"] = "per-chunk paired difference; the same chunks for both arms"
    return result


def arm_summary(scored: dict, reference: dict | None = None) -> dict:
    ce = summary([row["ce"] for row in scored["rows"]])
    kl = summary([row["kl"] for row in scored["rows"]])
    payload = {
        "chunks": scored["chunks"],
        "tokens_scored": scored["tokens_scored"],
        "ce": ce,
        "kl": kl,
        "perplexity": math.exp(ce["mean"]) if ce["mean"] is not None and ce["mean"] < 700 else None,
        "per_chunk": {
            "ce": [row["ce"] for row in scored["rows"]],
            "kl": [row["kl"] for row in scored["rows"]],
        },
    }
    if reference is not None:
        payload["delta_ce_vs_reference"] = paired_delta(
            [row["ce"] for row in scored["rows"]],
            [row["ce"] for row in reference["rows"]])
        payload["delta_kl_vs_reference"] = paired_delta(
            [row["kl"] for row in scored["rows"]],
            [row["kl"] for row in reference["rows"]])
    return payload


def family_means(per_arm: dict, families: dict[str, list[str]]) -> dict:
    """Family mean and independent combined SE from per-task harness scores.

    Same arithmetic as `scripts/s39_control_compare.py`: the family standard
    error is `sqrt(sum(se_i^2)) / n`, labelled independent.
    """
    out = {}
    for arm, tasks in per_arm.items():
        rows = {}
        for family, members in families.items():
            if any(name not in tasks for name in members):
                continue
            present = [tasks[name] for name in members]
            scores = [row["score"] for row in present]
            errors = [row.get("stderr") or 0.0 for row in present]
            mean = sum(scores) / len(scores)
            se = math.sqrt(sum(value ** 2 for value in errors)) / len(errors)
            rows[family] = {"members": members, "mean": mean, "stderr": se}
        out[arm] = rows
    return out


def format_table(rows: list[dict], reference: str | None) -> str:
    """Plain-text table: CE and KL with standard errors, paired delta vs reference."""
    header = (f"{'arm':<22} {'CE':>9} {'SE':>8} {'KL':>9} {'SE':>8} "
              f"{'dCE':>9} {'SE':>8}")
    lines = [header, "-" * len(header)]
    for row in rows:
        ce = row["ce"]
        kl = row["kl"]
        delta = row.get("delta_ce_vs_reference") or {}
        lines.append(
            f"{row['arm']:<22} {ce['mean']:>9.5f} {fmt(ce['stderr']):>8} "
            f"{kl['mean']:>9.5f} {fmt(kl['stderr']):>8} "
            f"{fmt(delta.get('mean'), '+.5f'):>9} {fmt(delta.get('stderr')):>8}")
    if reference:
        lines.append(f"delta columns are paired per-chunk differences against {reference}")
    return "\n".join(lines)


def fmt(value, spec: str = ".5f") -> str:
    return "n/a" if value is None else format(value, spec)
