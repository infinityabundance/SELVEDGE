#!/usr/bin/env python3
"""Hashed memory path (docs/EXPERIMENT-CONTRACT.md T2.2, plan item P3.1).

Deterministic n-gram hashing into a per-layer table of 256-dimensional values.
There is no key table and no learned index: the address is a pure integer
function of the token prefix, so the only learned parameters an executed read
touches are the retrieved value rows, the projection into `d_model` and the
gate. That is exactly what the active union charges.

Two heads read each insertion layer, a bigram (order 2) and a trigram
(order 3) head, so a token performs exactly `2` row reads per insertion layer.
A head collision reads one row twice; the accounting declares the two reads as
two distinct 256-element identities, which is the conservative union.

The projection is zero-initialised so the module contributes exactly zero at
step 0 (the same construction as the specialists' zero `W_down`), which is what
makes step-0 parity to the frozen core exact.

This module is CPU-testable and does not import the model. `TokenContext` is the
only bridge to the token ids: the arm wrapper installs a model-level forward
pre-hook that writes the current `input_ids` into it, so the memory sees the
prefix that the layer is actually processing, including during generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: FNV-1a constants for the 64-bit n-gram fold.
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
#: The hash takes the top `HASH_BITS` bits of the folded 64-bit key.
HASH_BITS = 18


def salt(layer: int, order: int) -> int:
    """Fixed per-(layer, order) seed. Independent of the torch RNG.

    Returned as a *signed* 64-bit integer because `torch.int64` is signed; the
    bit pattern is the same uint64 value.
    """
    mixed = (FNV_OFFSET
             ^ ((layer + 1) * 0x9E3779B97F4A7C15)
             ^ (order * 0xC2B2AE3D27D4EB4F)) & MASK64
    # One more fold so adjacent layers do not share low-bit structure.
    value = ((mixed ^ (mixed >> 33)) * 0xFF51AFD7ED558CCD) & MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def ngram_index(tokens: torch.Tensor, layer: int, order: int, rows: int) -> torch.Tensor:
    """Address of the `order`-gram ending at each position.

    `tokens` is `[B, T]` int64. Returns `[B, T]` int64 in `[0, rows)`.
    Position `t` is addressed by `tokens[0..t]` only, so the index is causal.
    Positions with an incomplete n-gram are left-padded with token id 0.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
    if order < 1:
        raise ValueError("order must be >= 1")
    rows = int(rows)
    if rows <= 0 or rows & (rows - 1):
        raise ValueError("rows must be a positive power of two")
    tokens = tokens.to(torch.int64)
    batch, length = tokens.shape
    padded = F.pad(tokens, (order - 1, 0), value=0)
    key = torch.full((batch, length), salt(layer, order), dtype=torch.int64,
                     device=tokens.device)
    for position in range(order):
        window = padded[:, position:position + length]
        key = (key ^ window) * FNV_PRIME
    shift = 64 - (rows.bit_length() - 1)
    return (key >> shift) & (rows - 1)


class TokenContext:
    """Latest `input_ids` seen by the arm, written by a forward pre-hook."""

    __slots__ = ("ids",)

    def __init__(self) -> None:
        self.ids: torch.Tensor | None = None

    def set(self, ids: torch.Tensor) -> None:
        self.ids = ids.detach()

    def require(self, length: int) -> torch.Tensor:
        if self.ids is None:
            raise RuntimeError("token context is empty: install the arm's pre-hook")
        if self.ids.shape[-1] < length:
            raise RuntimeError(
                f"token context has {self.ids.shape[-1]} positions, layer needs {length}")
        return self.ids[:, -length:]


@dataclass(frozen=True)
class MemoryGeometry:
    """The registered geometry of one insertion layer (T2.2)."""

    hidden: int = 1024
    value_dim: int = 256
    rows: int = 1 << 18
    reads: int = 2
    orders: tuple[int, ...] = (2, 3)

    @property
    def value_elements(self) -> int:
        return self.reads * self.value_dim

    @property
    def projection_elements(self) -> int:
        return self.reads * self.value_dim * self.hidden

    @property
    def gate_elements(self) -> int:
        return self.hidden

    @property
    def index_elements(self) -> int:
        """No learned index: the address is arithmetic (T2.2)."""
        return 0

    @property
    def active_per_layer(self) -> int:
        return (self.value_elements + self.projection_elements
                + self.gate_elements + self.index_elements)

    @property
    def table_elements(self) -> int:
        return self.rows * self.value_dim

    @property
    def total_per_layer(self) -> int:
        # The retrieved value rows are table entries, not extra parameters, so
        # the accessible total is the table plus the projection and the gate.
        return (self.table_elements + self.projection_elements
                + self.gate_elements + self.index_elements)

    def active_report(self, insertions: int) -> dict:
        """The exact export the admission declaration and the tests use."""
        return {
            "kind": "hashed_memory_active",
            "insertions": insertions,
            "reads_per_layer": self.reads,
            "value_elements_per_layer": self.value_elements,
            "projection_elements_per_layer": self.projection_elements,
            "gate_elements_per_layer": self.gate_elements,
            "index_elements_per_layer": self.index_elements,
            "active_per_layer": self.active_per_layer,
            "active_total": insertions * self.active_per_layer,
            "table_elements_per_layer": self.table_elements,
            "table_elements_total": insertions * self.table_elements,
            "total_accessible_added": insertions * self.total_per_layer,
            "rows_total": insertions * self.rows,
            "values_total": insertions * self.table_elements,
        }

    def bytes_report(self, insertions: int, deployment_bytes: int = 2) -> dict:
        """Resident and read bytes, kept separate and labelled.

        `read_bytes_per_token` is the modelled lower bound of the distinct
        identities read at the deployment dtype. It is not measured I/O and is
        labelled `modelled` everywhere it is reported.
        """
        fp32_table = insertions * self.table_elements * 4
        deploy_table = insertions * self.table_elements * deployment_bytes
        read_bytes = insertions * self.active_per_layer * deployment_bytes
        return {
            "kind": "hashed_memory_bytes",
            "dtype_bytes": deployment_bytes,
            "resident_table_bytes_fp32": fp32_table,
            "resident_table_bytes_deployment": deploy_table,
            "read_bytes_per_token": read_bytes,
            "read_bytes_per_token_basis": "modelled",
            "note": "read bytes are distinct identities read at the declared "
                    "dtype; they are a modelled lower bound, never measured I/O",
        }


class HashedMemory(nn.Module):
    """One insertion layer: two hashed reads, a projection and a gate."""

    def __init__(self, hidden: int, layer: int, context: TokenContext,
                 geometry: MemoryGeometry | None = None, seed: int | None = None):
        super().__init__()
        geometry = geometry or MemoryGeometry(hidden=hidden)
        if geometry.reads != len(geometry.orders):
            raise ValueError("read count must equal the number of n-gram heads")
        self.hidden = geometry.hidden
        self.value_dim = geometry.value_dim
        self.rows = geometry.rows
        self.reads = geometry.reads
        self.orders = geometry.orders
        self.layer = int(layer)
        self.context = context
        self.table = nn.Parameter(torch.empty(self.rows, self.value_dim,
                                              dtype=torch.float32))
        self.proj = nn.Linear(self.reads * self.value_dim, self.hidden, bias=False,
                              dtype=torch.float32)
        self.gate = nn.Linear(self.hidden, 1, bias=False, dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        if seed is None:
            # Content-independent, layer-independent of the torch RNG: the
            # values are learned, but their initial state is reproducible.
            generator.manual_seed(0)
        else:
            generator.manual_seed(int(seed))
        with torch.no_grad():
            nn.init.normal_(self.table, std=0.02, generator=generator)
            nn.init.normal_(self.gate.weight, std=0.02, generator=generator)
            # Zero-init the projection: step 0 is exactly the frozen core.
            nn.init.zeros_(self.proj.weight)

    # -- accounting ---------------------------------------------------------
    def geometry(self) -> MemoryGeometry:
        return MemoryGeometry(hidden=self.hidden, value_dim=self.value_dim,
                              rows=self.rows, reads=self.reads, orders=self.orders)

    def active_parameter_report(self) -> dict:
        report = self.geometry().active_report(insertions=1)
        report["layer"] = self.layer
        return report

    # -- forward ------------------------------------------------------------
    def indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, T, reads]` addresses for one layer."""
        return torch.stack([ngram_index(tokens, self.layer, order, self.rows)
                            for order in self.orders], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[-2]
        tokens = self.context.require(length)
        index = self.indices(tokens)
        # Gather the retrieved rows: [B, T, reads, value_dim].
        values = self.table[index]
        mixed = self.proj(values.reshape(*hidden_states.shape[:-1],
                                         self.reads * self.value_dim))
        gate = torch.sigmoid(self.gate(hidden_states))
        return gate * mixed


def state_digest(table: torch.Tensor) -> str:
    """SHA-256 of the table's bytes, for the utility-label manifest."""
    return hashlib.sha256(table.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
