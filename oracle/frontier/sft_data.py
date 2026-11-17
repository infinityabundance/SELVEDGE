#!/usr/bin/env python3
"""S53: example-boundary-safe packing and completion-only supervision.

The stage-2 trainer consumes an untyped `uint32` token stream and cuts it into
fixed 512-token chunks. That is correct for causal-language-model distillation
(every token is a target) but wrong for verified-solution SFT: a chunk can end
in the middle of one solution and begin in the middle of an unrelated one, and
there is no way to tell the loss to ignore the prompt.

This module makes the SFT unit an **example**, not a chunk:

* `completion_example(prompt_ids, solution_ids, eos_id)` keeps the prompt
  unsupervisised (`label = IGNORE_INDEX`) and supervises every solution token
  plus the trailing EOS. The last prompt token still predicts the first solution
  token, because the model sees the whole row and the labels are shifted by one.
* `pack_examples(...)` places **complete** examples in a row and pads the rest.
  No example is split across rows. The default is one example per row, so no
  example can attend to an unrelated one. Packing more than one complete
  example per row is available, but it is not boundary-safe without a
  per-example attention/mask reset; the report says so, and the later
  boundary-safe option is documented here rather than silently assumed.
* An example longer than `seq_len` fails closed by default. `overflow="drop"`
  is explicit and counted; there is no silent truncation of a solution.

Later boundary-safe packing option (registered, not implemented here): pack
several complete examples per row and reset attention at each boundary (a
document mask / varlen attention call). That keeps the token budget without
letting one example attend to another. It is a change to the model call, so it
needs its own registration and a parity test against one-example-per-row.

The module is CPU-only and has no dependency on `transformers`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class PackedExamples:
    """Packed rows plus the accounting needed to describe the run."""

    input_ids: torch.Tensor          # [rows, seq_len] int64
    labels: torch.Tensor             # [rows, seq_len] int64, IGNORE on prompt/pad
    report: dict


def _as_ids(values, name: str) -> list[int]:
    ids = [int(value) for value in values]
    if any(value < 0 for value in ids):
        raise ValueError(f"{name} contains a negative token id")
    return ids


def completion_example(prompt_ids, solution_ids, eos_id: int
                       ) -> tuple[list[int], list[int]]:
    """Return `(tokens, labels)` with the prompt masked and the EOS supervised."""
    prompt = _as_ids(prompt_ids, "prompt_ids")
    solution = _as_ids(solution_ids, "solution_ids")
    if not solution:
        raise ValueError("a verified solution must have at least one token")
    eos = int(eos_id)
    if eos < 0:
        raise ValueError("eos_id must be non-negative")
    tokens = prompt + solution + [eos]
    labels = [IGNORE_INDEX] * len(prompt) + solution + [eos]
    return tokens, labels


def pack_examples(examples, seq_len: int, pad_id: int, eos_id: int,
                  examples_per_row: int = 1, overflow: str = "error"
                  ) -> PackedExamples:
    """Pack complete examples into padded rows of `seq_len` tokens.

    `examples` is an iterable of mappings with `prompt_ids` and `solution_ids`.
    With `examples_per_row == 1` each row holds exactly one complete example and
    the remaining positions are padding. A larger value greedily places complete
    examples while they fit; the report marks that mode as not boundary-safe
    (no per-example attention reset), which is the documented later option.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if examples_per_row <= 0:
        raise ValueError("examples_per_row must be positive")
    if overflow not in ("error", "drop"):
        raise ValueError("overflow must be 'error' or 'drop'")
    seq_len = int(seq_len)
    pad_id = int(pad_id)
    if pad_id < 0:
        raise ValueError("pad_id must be non-negative")

    rows_tokens: list[list[int]] = []
    rows_labels: list[list[int]] = []
    current_tokens: list[int] = []
    current_labels: list[int] = []
    placed_in_row = 0
    counts = {
        "examples": 0,
        "packed_examples": 0,
        "dropped_overflow": 0,
        "prompt_tokens": 0,
        "supervised_tokens": 0,
        "padding_tokens": 0,
        "max_example_tokens": 0,
    }

    def flush() -> None:
        nonlocal current_tokens, current_labels, placed_in_row
        if not current_tokens:
            return
        padding = seq_len - len(current_tokens)
        rows_tokens.append(current_tokens + [pad_id] * padding)
        rows_labels.append(current_labels + [IGNORE_INDEX] * padding)
        counts["padding_tokens"] += padding
        current_tokens, current_labels, placed_in_row = [], [], 0

    for example in examples:
        counts["examples"] += 1
        tokens, labels = completion_example(
            example["prompt_ids"], example["solution_ids"], eos_id)
        length = len(tokens)
        counts["max_example_tokens"] = max(counts["max_example_tokens"], length)
        if length > seq_len:
            if overflow == "error":
                raise ValueError(
                    f"example {example.get('id', counts['examples'] - 1)!r} has "
                    f"{length} tokens > seq_len {seq_len}; refusing to split an "
                    "example across rows (see the later boundary-safe option)")
            counts["dropped_overflow"] += 1
            continue
        if placed_in_row >= examples_per_row or len(current_tokens) + length > seq_len:
            flush()
        current_tokens.extend(tokens)
        current_labels.extend(labels)
        placed_in_row += 1
        counts["packed_examples"] += 1
        counts["prompt_tokens"] += len(tokens) - sum(
            1 for value in labels if value != IGNORE_INDEX)
        counts["supervised_tokens"] += sum(
            1 for value in labels if value != IGNORE_INDEX)
    flush()

    if not rows_tokens:
        raise ValueError("no example fits in a row; nothing to train on")

    report = {
        "kind": "sft_example_packing",
        "seq_len": seq_len,
        "pad_id": pad_id,
        "eos_id": int(eos_id),
        "examples_per_row": int(examples_per_row),
        "overflow": overflow,
        "rows": len(rows_tokens),
        "boundary_safe": (examples_per_row == 1),
        "boundary_safe_note": (
            "one complete example per row: no example attends to another"
            if examples_per_row == 1 else
            "multiple complete examples per row share the attention window; "
            "the boundary-safe option (per-example attention reset) is a later "
            "registered change"),
        "completion_only_loss": True,
        "padding_tokens": counts["padding_tokens"],
        **{key: counts[key] for key in (
            "examples", "packed_examples", "dropped_overflow", "prompt_tokens",
            "supervised_tokens", "max_example_tokens")},
    }
    return PackedExamples(
        input_ids=torch.tensor(rows_tokens, dtype=torch.int64),
        labels=torch.tensor(rows_labels, dtype=torch.int64),
        report=report)


def load_examples(path: str, vocab_size: int | None = None) -> list[dict]:
    """Load a JSONL examples file written by `scripts/s45_rft_stream.py`."""
    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "prompt_ids" not in payload or "solution_ids" not in payload:
                raise ValueError(f"{path}:{line_number}: missing prompt/solution ids")
            prompt = _as_ids(payload["prompt_ids"], "prompt_ids")
            solution = _as_ids(payload["solution_ids"], "solution_ids")
            if vocab_size is not None:
                if prompt and max(prompt) >= vocab_size:
                    raise ValueError(f"{path}:{line_number}: prompt id outside vocab")
                if solution and max(solution) >= vocab_size:
                    raise ValueError(f"{path}:{line_number}: solution id outside vocab")
            examples.append({**payload, "prompt_ids": prompt,
                             "solution_ids": solution})
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def example_batches(packed: PackedExamples, micro_batch: int, grad_accum: int,
                    steps: int):
    """Yield `(inputs, targets)` for `steps` optimiser steps, cycling rows.

    `inputs` are the row's tokens without the last position and `targets` are the
    row's labels shifted by one, so the first solution token is predicted from the
    last prompt token. Padding and prompt labels are already `IGNORE_INDEX`.
    """
    if micro_batch <= 0 or grad_accum <= 0:
        raise ValueError("micro_batch and grad_accum must be positive")
    rows = packed.input_ids.shape[0]
    cursor = 0
    for _ in range(steps):
        batches = []
        for _ in range(grad_accum):
            indices = torch.arange(cursor, cursor + micro_batch) % rows
            cursor += micro_batch
            inputs = packed.input_ids[indices, :-1]
            targets = packed.labels[indices, 1:]
            batches.append((inputs, targets))
        yield batches


def describe_source(path: str) -> dict:
    """Path, byte size and SHA-256 for a manifest entry."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return {"path": path, "bytes": os.path.getsize(path),
            "sha256": digest.hexdigest()}


__all__ = ["IGNORE_INDEX", "PackedExamples", "completion_example",
           "pack_examples", "load_examples", "example_batches",
           "describe_source"]
