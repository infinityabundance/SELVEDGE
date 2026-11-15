#!/usr/bin/env python3
"""Independent reference for the routed-capacity dispatch and combine (P12.1).

This module is an audit artefact, not a fast path. It transcribes the intended
semantics of `modeling._RoutedDispatch` in the dumbest possible way (explicit
Python loops over pairs, explicit buffer rows, no `index_copy_`, no scatter, no
reshape-and-reduce) so that the shipped vectorised implementation can be checked
against something that was written from the specification rather than from the
code.

Semantics (the specification, restated):

* A forward pass over `tokens` positions selects `k` experts per position. The
  `tokens * k` (token, slot) pairs are ordered **token-major**: pair
  `p = token * k + slot`, where `slot` is the router's ranking order.
* The pairs are grouped by expert id with a **stable** sort, so an expert's
  block is in ascending pair order and the blocks are in ascending expert
  order. A pair's **slot within its expert** is its index in that block.
* Capacity is `max(1, ceil(tokens * k / experts * factor))` rows per expert.
  A pair is **kept** iff its slot within its expert is below capacity; every
  other pair is **dropped**: it contributes exactly zero to its token's output
  and receives exactly zero gradient. No pair is ever re-routed to another
  expert.
* A token's output is the sum over its kept pairs of
  `pair_weight[p] * expert_output[top_index[p]](hidden[token])`, reduced in the
  fixed slot order. Dropped pairs are not silently replaced by another
  expert's output.

Two mask placements are modelled, because the audit has to distinguish them:

``mask_mode="pair"``
    The correct one: `keep` is a property of the pair (its own slot).
``mask_mode="sorted"``
    The shipped behaviour before the P12.1 correction: the boolean vector is
    built in sorted-position order and then applied to tensor positions that
    have already been restored to token order. Unless the sort permutation is
    the identity, pair `p` is then gated by the validity of the pair whose
    *sorted position* is `p`, which is a different pair whenever overflow
    happened.

The buffer content is also modelled, because the wrong gating is not the only
consequence: an overflowing pair that survives the misordered mask reads a
clamped buffer row, which may hold a **different token's** expert activation.
`plan()` therefore exposes the row index each sorted position reads, and
`assemble()` reads exactly that row, so a test can show which value a
mislabelled pair actually receives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: The two mask placements this reference can model.
MASK_MODES = ("pair", "sorted")


def capacity(tokens: int, k: int, experts: int, factor: float) -> int:
    """Rows per expert, exactly as the shipped `modeling._capacity` computes it."""
    if tokens < 0 or k < 0:
        raise ValueError("tokens and k must be non-negative")
    if experts <= 0:
        raise ValueError("experts must be positive")
    if k > experts:
        raise ValueError(
            f"top-k ({k}) exceeds the number of experts ({experts}); a registered "
            "arm always satisfies k <= experts, so this is out of contract")
    if not factor > 0:
        raise ValueError("capacity factor must be positive")
    return max(1, math.ceil(tokens * k / experts * factor))


@dataclass(frozen=True)
class Plan:
    """The assignment computed from a routing decision, in sorted-position order."""

    tokens: int
    k: int
    experts: int
    capacity: int
    #: sorted position -> token-major pair index
    order: tuple[int, ...]
    #: sorted position -> expert id
    expert: tuple[int, ...]
    #: sorted position -> slot within that expert's block
    slot: tuple[int, ...]
    #: sorted position -> buffer row this position reads (clamped when dropped)
    row: tuple[int, ...]
    #: sorted position -> kept by its own slot
    valid: tuple[bool, ...]

    @property
    def pairs(self) -> int:
        return len(self.order)

    def inverse(self) -> tuple[int, ...]:
        """token-major pair index -> sorted position."""
        inverse = [0] * self.pairs
        for position, pair in enumerate(self.order):
            inverse[pair] = position
        return tuple(inverse)

    def dropped(self) -> tuple[int, ...]:
        return tuple(position for position, keep in enumerate(self.valid) if not keep)


def plan(top_index: torch.Tensor, experts: int, factor: float) -> Plan:
    """Compute the assignment for one routing decision.

    `top_index` is `[tokens, k]` (or `[tokens * k]`), token-major: entry
    `[t, s]` is the expert chosen at rank `s` for token `t`.
    """
    flat_index = top_index.reshape(-1)
    if top_index.dim() == 2:
        tokens, k = top_index.shape
    elif top_index.dim() == 1:
        tokens, k = 1, flat_index.numel()
    else:
        raise ValueError("top_index must be 1-D or 2-D")
    rows = capacity(tokens, k, experts, factor)

    pair_expert = [int(value) for value in flat_index.tolist()]
    if any(value < 0 or value >= experts for value in pair_expert):
        raise ValueError(f"expert id outside [0, {experts})")

    # Stable sort by (expert id, pair index): blocks in expert order, each block
    # in ascending pair order.
    order = sorted(range(len(pair_expert)), key=lambda pair: (pair_expert[pair], pair))
    counts = [0] * experts
    for expert in pair_expert:
        counts[expert] += 1
    start = [0] * experts
    running = 0
    for expert, count in enumerate(counts):
        start[expert] = running
        running += count

    expert_of: list[int] = []
    slot: list[int] = []
    valid: list[bool] = []
    row: list[int] = []
    for position, pair in enumerate(order):
        expert = pair_expert[pair]
        within = position - start[expert]
        expert_of.append(expert)
        slot.append(within)
        keep = within < rows
        valid.append(keep)
        # A dropped pair must still index somewhere; the shipped implementation
        # clamps to the last buffer row. Model that exactly, so a test can show
        # which row a mislabelled pair actually reads.
        row.append(expert * rows + within if keep else experts * rows - 1)

    return Plan(tokens=tokens, k=k, experts=experts, capacity=rows,
                order=tuple(order), expert=tuple(expert_of), slot=tuple(slot),
                row=tuple(row), valid=tuple(valid))


def build_buffer(flat: torch.Tensor, specification: Plan) -> torch.Tensor:
    """The dispatch half: place each kept pair's input row in its buffer row."""
    tokens, hidden = flat.shape[0], flat.shape[-1]
    if tokens != specification.tokens:
        raise ValueError("flat and plan disagree on the number of tokens")
    buffer = flat.new_zeros(specification.experts * specification.capacity, hidden)
    for position, pair in enumerate(specification.order):
        if not specification.valid[position]:
            continue
        token = pair // specification.k
        buffer[specification.row[position]] = flat[token]
    return buffer.view(specification.experts, specification.capacity, hidden)


def assemble(buffer_out: torch.Tensor, specification: Plan,
             pair_weight: torch.Tensor, mask_mode: str = "pair") -> torch.Tensor:
    """The combine half: return per-pair outputs to tokens in slot order.

    `buffer_out` is `[experts, capacity, hidden]`, the experts' output on
    `build_buffer`'s buffer. `pair_weight` is `[tokens * k]` in token-major
    order.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    hidden = buffer_out.shape[-1]
    flat_out = buffer_out.reshape(-1, hidden)
    weights = pair_weight.reshape(-1).to(flat_out.dtype)
    if weights.numel() != specification.pairs:
        raise ValueError("pair_weight length does not match the plan")
    inverse = specification.inverse()
    out = flat_out.new_zeros(specification.tokens, hidden)
    for pair in range(specification.pairs):
        token, _slot = divmod(pair, specification.k)
        position = inverse[pair]
        keep = (specification.valid[position] if mask_mode == "pair"
                else specification.valid[pair])
        if not keep:
            continue
        out[token] = out[token] + flat_out[specification.row[position]] * weights[pair]
    return out


def contribution(flat: torch.Tensor, top_index: torch.Tensor, pair_weight: torch.Tensor,
                 expert_fn, experts: int, factor: float,
                 mask_mode: str = "pair") -> torch.Tensor:
    """Dispatch, run the experts, combine — the whole module, explicitly.

    `expert_fn(expert, rows)` must return `[len(rows), hidden]`, where `rows`
    holds the *token* indices whose buffered hidden state this expert reads.
    """
    specification = plan(top_index, experts, factor)
    buffer = build_buffer(flat, specification)
    flat_out = flat.new_zeros(experts * specification.capacity, flat.shape[-1])
    for expert in range(experts):
        reading = [position for position in range(specification.pairs)
                   if specification.expert[position] == expert
                   and specification.valid[position]]
        if not reading:
            continue
        tokens = torch.tensor([specification.order[position] // specification.k
                               for position in reading], dtype=torch.long)
        values = expert_fn(expert, tokens)
        for offset, position in enumerate(reading):
            flat_out[specification.row[position]] = values[offset]
    return assemble(flat_out.view(experts, specification.capacity, -1),
                    specification, pair_weight, mask_mode)
