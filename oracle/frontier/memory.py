#!/usr/bin/env python3
"""Stage 2 (WS19 P19.2): teacher-grafted conditional n-gram memory.

The memory is *addressed by token ids only*. There is no learned index, no
suffix scan and no per-token routing network: the address of position `t` is an
exact integer function of the causal token prefix, so a read is a gather and the
whole table is addressable without touching it.

Geometry of one insertion site (the registered default is a single site after
decoder layer index 2; `sites` is configurable):

* 8 heads per token: four bigram heads (order 2) and four trigram heads
  (order 3), each with its own deterministic salt;
* every head reads one row of `value_dim = 256` values from one shared table;
* per site, a key projection `hidden -> value_dim`, a value projection
  `value_dim -> hidden` (shared across the 8 reads, applied per read and
  summed), and a short causal depthwise convolution over the retrieved values;
* the gate is `sigmoid(<key_proj(h_t), row> / sqrt(value_dim))` per read: it is
  conditioned on the hidden state and has no parameters of its own.

Accounting vocabulary used everywhere in this package:

* **accessible** — every parameter the module adds (table rows + projections +
  convolution + bias).
* **active** — what one token reads or executes: the 8 retrieved 256-element
  rows, the two projections, the convolution and its bias. The table is *not*
  charged: a read touches 8 of `rows` rows. The union is conservative: two
  heads that collide on one row are still charged as two distinct identities
  (the same convention as `oracle/train/memory.py`).

The hash is arithmetic on int64 values kept below 2^31 before the final
avalanche, so it cannot overflow and is bit-identical on CPU and CUDA. It is a
pure function of `(token ids, site, order, head, rows)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

#: mask keeping every hash step inside a signed 31-bit range (no overflow).
MASK31 = (1 << 31) - 1
#: multiplication constant of the rolling fold.
FOLD = 1_000_003
#: avalanche constants.
AVALANCHE_MUL = 0x2C1B3C6D
GOLDEN = 0x9E37

#: 4 bigram heads followed by 4 trigram heads: 8 rows read per token.
ORDERS: tuple[int, ...] = (2, 2, 2, 2, 3, 3, 3, 3)
HEADS = len(ORDERS)
DEFAULT_ROWS = 65_536
REGISTERED_ROW_SCALES: tuple[int, ...] = (65_536, 1_048_576, 8_388_608)
DEFAULT_SITE = 2
DEFAULT_VALUE_DIM = 256
DEFAULT_CONV_KERNEL = 4
DEFAULT_HIDDEN = 1024


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def salt(site: int, order: int, head: int) -> int:
    """Deterministic per-(site, order, head) seed in `[0, 2**31)`."""
    value = (((site + 1) * 73_856_093)
             ^ ((order + 1) * 19_349_663)
             ^ ((head + 1) * 83_492_791)) & MASK31
    value = (value ^ (value >> 13)) & MASK31
    value = (value * AVALANCHE_MUL) & MASK31
    return (value ^ (value >> 15)) & MASK31


def address(tokens: torch.Tensor, site: int, order: int, head: int,
            rows: int) -> torch.Tensor:
    """Address of the `order`-gram ending at each position.

    `tokens` is `[B, T]` int64. Returns `[B, T]` int64 in `[0, rows)`.
    Position `t` reads only `tokens[..., :t+1]`: incomplete prefixes are
    left-padded with token id 0, so the address is causal by construction.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
    if order < 1:
        raise ValueError("order must be >= 1")
    if not _is_power_of_two(int(rows)):
        raise ValueError("rows must be a positive power of two")
    tokens = tokens.to(torch.int64)
    batch, length = tokens.shape
    window = F.pad(tokens, (order - 1, 0), value=0)
    key = torch.full((batch, length), salt(site, order, head), dtype=torch.int64,
                     device=tokens.device)
    for position in range(order):
        # Position 0 is the oldest token of the n-gram; multiply it less so
        # that "a b" and "b a" fold differently.
        token = window[:, position:position + length]
        key = (key * FOLD + token * (position + 1) + GOLDEN) & MASK31
    key = (key ^ (key >> 13)) & MASK31
    key = (key * AVALANCHE_MUL) & MASK31
    key = (key ^ (key >> 15)) & MASK31
    bits = rows.bit_length() - 1
    return key >> (31 - bits)


def support(tokens: torch.Tensor, site: int, rows: int) -> torch.Tensor:
    """All 8 addresses for one site: `[B, T, 8]` int64."""
    return torch.stack([address(tokens, site, order, head, rows)
                        for head, order in enumerate(ORDERS)], dim=-1)


# ---------------------------------------------------------------------------
# token context


class TokenContext:
    """Causal token history seen by an arm.

    Installed as a forward pre-hook on the text model. A full-sequence call
    resets the history; a cache-carrying call appends to it, so a generation
    step still sees the true prefix. A cached call may carry one token or a
    chunk: the history is defined by `past_key_values.get_seq_length()`, not by
    the width of the call. `prefix` returns the whole recorded prefix and
    `require` returns only the last `length` tokens; arms that address n-grams
    must use `prefix` (see `ConditionalMemory.site_forward`).

    `generation` increments on every reset so a stateful arm (the convolution
    cache) can drop its carried state when a new sequence starts.
    """

    __slots__ = ("ids", "resets", "appends", "fallbacks", "generation")

    def __init__(self) -> None:
        self.ids: torch.Tensor | None = None
        self.resets = 0
        self.appends = 0
        self.fallbacks = 0
        self.generation = 0

    def observe(self, ids: torch.Tensor | None, past=None) -> None:
        if ids is None:
            return
        ids = ids.detach()
        past_length = 0
        if past is not None:
            try:
                past_length = int(past.get_seq_length())
            except Exception:
                past_length = 0
        if self.ids is None or past_length == 0:
            # Prefill, or a fresh full-sequence call: this *is* the prefix.
            self.ids = ids
            self.resets += 1
            self.generation += 1
            return
        if self.ids.shape[0] != ids.shape[0]:
            self.ids = ids
            self.fallbacks += 1
            self.generation += 1
            return
        if self.ids.shape[-1] < past_length:
            # The cache names more positions than the context recorded, so the
            # true prefix cannot be reconstructed. Fail loudly in the counters
            # rather than address a shorter history as if it were complete.
            self.ids = ids
            self.fallbacks += 1
            self.generation += 1
            return
        # The cache defines the prefix: keep exactly `past_length` positions and
        # append this call's tokens. This is correct for a one-token step, a
        # multi-token chunk, and a replayed call that would otherwise duplicate.
        self.ids = torch.cat([self.ids[:, :past_length], ids], dim=-1)
        self.appends += 1

    def reset(self) -> None:
        self.ids = None
        self.generation += 1

    def prefix(self) -> torch.Tensor:
        """The whole recorded causal prefix, `[B, S]`."""
        if self.ids is None:
            raise RuntimeError("token context is empty: install the arm pre-hook")
        return self.ids

    def require(self, length: int) -> torch.Tensor:
        if self.ids is None:
            raise RuntimeError("token context is empty: install the arm pre-hook")
        if self.ids.shape[-1] < length:
            raise RuntimeError(
                f"token context has {self.ids.shape[-1]} positions, layer needs {length}")
        return self.ids[:, -length:]

    def report(self) -> dict:
        return {"resets": self.resets, "appends": self.appends,
                "fallbacks": self.fallbacks, "generation": self.generation,
                "history": None if self.ids is None else int(self.ids.shape[-1])}


# ---------------------------------------------------------------------------
# exact accounting


@dataclass(frozen=True)
class MemoryGeometry:
    """Exact per-site arithmetic. Every published count is derived here."""

    hidden: int = DEFAULT_HIDDEN
    value_dim: int = DEFAULT_VALUE_DIM
    rows: int = DEFAULT_ROWS
    reads: int = HEADS
    conv_kernel: int = DEFAULT_CONV_KERNEL
    sites: int = 1

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.rows):
            raise ValueError("rows must be a positive power of two")
        if self.reads != HEADS:
            raise ValueError(f"this geometry has exactly {HEADS} reads")

    # -- per token, one site --------------------------------------------------
    @property
    def rows_read_elements(self) -> int:
        return self.reads * self.value_dim

    @property
    def key_projection_elements(self) -> int:
        """`hidden -> value_dim`, no bias: this is also the gate's parameter."""
        return self.hidden * self.value_dim

    @property
    def value_projection_elements(self) -> int:
        """`value_dim -> hidden` plus the once-per-site output bias."""
        return self.value_dim * self.hidden + self.hidden

    @property
    def conv_elements(self) -> int:
        channels = self.reads * self.value_dim
        return channels * self.conv_kernel + channels

    @property
    def gate_elements(self) -> int:
        """The sigmoid gate is parameter-free; `key_projection` is its weight."""
        return 0

    @property
    def active_per_token_per_site(self) -> int:
        return (self.rows_read_elements + self.key_projection_elements
                + self.value_projection_elements + self.conv_elements
                + self.gate_elements)

    @property
    def active_per_token(self) -> int:
        return self.active_per_token_per_site * self.sites

    # -- accessible -----------------------------------------------------------
    @property
    def table_elements_per_site(self) -> int:
        return self.rows * self.value_dim

    @property
    def trainable_per_site(self) -> int:
        return (self.key_projection_elements + self.value_projection_elements
                + self.conv_elements)

    @property
    def accessible_per_site(self) -> int:
        return self.table_elements_per_site + self.trainable_per_site

    @property
    def table_elements(self) -> int:
        return self.table_elements_per_site * self.sites

    @property
    def accessible_added(self) -> int:
        return self.accessible_per_site * self.sites

    def report(self) -> dict:
        return {
            "kind": "stage2_memory_geometry",
            "hidden": self.hidden,
            "value_dim": self.value_dim,
            "rows_per_site": self.rows,
            "reads_per_token_per_site": self.reads,
            "orders": list(ORDERS),
            "sites": self.sites,
            "conv_kernel": self.conv_kernel,
            "rows_read_elements_per_token_per_site": self.rows_read_elements,
            "key_projection_elements_per_site": self.key_projection_elements,
            "value_projection_elements_per_site": self.value_projection_elements,
            "conv_elements_per_site": self.conv_elements,
            "gate_elements_per_site": self.gate_elements,
            "active_per_token_per_site": self.active_per_token_per_site,
            "active_per_token": self.active_per_token,
            "table_elements_per_site": self.table_elements_per_site,
            "trainable_per_site": self.trainable_per_site,
            "accessible_per_site": self.accessible_per_site,
            "table_elements_total": self.table_elements,
            "accessible_added_total": self.accessible_added,
            "resident_table_bytes_bf16": self.table_elements * 2,
            "resident_table_bytes_fp32": self.table_elements * 4,
            "read_bytes_per_token_bf16": self.active_per_token * 2,
            "read_bytes_per_token_basis": "modelled distinct-identity lower bound",
            "table_charged_to_active": False,
        }


def scale_table(rows: int, hidden: int = DEFAULT_HIDDEN,
                value_dim: int = DEFAULT_VALUE_DIM, sites: int = 1) -> dict:
    """The three registered row scales, exact, for the architecture question."""
    geometry = MemoryGeometry(hidden=hidden, value_dim=value_dim, rows=rows,
                              sites=sites)
    report = geometry.report()
    report["labels"] = {
        "development": rows == 65_536,
        "registered_scale": rows in REGISTERED_ROW_SCALES,
    }
    return report


# ---------------------------------------------------------------------------
# modules


class CausalDepthwiseConv1d(nn.Module):
    """Short causal depthwise conv over the retrieved value channels.

    Initialised to the identity at lag 0 (kernel position `kernel-1`), so a
    freshly grafted table injects the retrieved row unchanged until training
    moves the filter.

    The convolution has memory: position `t` sees inputs `t-kernel+1 .. t`.
    A call that is not the start of a sequence must therefore be given the last
    `kernel-1` *input* vectors from the previous call; otherwise each cached
    call would be padded independently and the result would not equal the
    full-sequence convolution. `forward` is the stateless full-sequence path;
    `forward_cached(values, state)` returns `(output, new_state)` and is what
    the memory uses so that token-by-token decoding is exact.
    """

    def __init__(self, channels: int, kernel: int = DEFAULT_CONV_KERNEL):
        super().__init__()
        if kernel < 1:
            raise ValueError("kernel must be >= 1")
        self.channels = channels
        self.kernel = kernel
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        with torch.no_grad():
            self.weight[:, 0, kernel - 1] = 1.0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """`[B, T, channels]` -> `[B, T, channels]`, causal, from a fresh start."""
        output, _ = self.forward_cached(values, None)
        return output

    def forward_cached(self, values: torch.Tensor,
                       state: torch.Tensor | None):
        """One causal step of the same convolution, carrying its history.

        `state` is the last `kernel-1` *inputs* of the previous call, shaped
        `[B, kernel-1, channels]` (or `None` at the start of a sequence). The
        returned new state is detached: it belongs to a completed call and must
        not extend an autograd graph across steps.
        """
        batch, length, channels = values.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        x = values.transpose(1, 2).float()                      # [B, C, T]
        lag = self.kernel - 1
        if lag == 0:
            history = x.new_zeros((batch, channels, 0))
        elif state is None:
            history = x.new_zeros((batch, channels, lag))
        else:
            if tuple(state.shape) != (batch, lag, channels):
                raise ValueError(
                    f"conv state {tuple(state.shape)} != {(batch, lag, channels)}")
            history = state.to(device=x.device, dtype=x.dtype).transpose(1, 2)
        padded = torch.cat([history, x], dim=-1)
        y = F.conv1d(padded, self.weight, self.bias, groups=self.channels)
        new_state = (padded[:, :, -lag:].detach().transpose(1, 2).to(values.dtype)
                     if lag else values.new_zeros((batch, 0, channels)))
        return y.transpose(1, 2).to(values.dtype), new_state


class ConditionalMemory(nn.Module):
    """One table per site, 8 reads per token, projections and a causal conv."""

    def __init__(self, hidden: int = DEFAULT_HIDDEN, value_dim: int = DEFAULT_VALUE_DIM,
                 rows: int = DEFAULT_ROWS, sites: tuple[int, ...] = (DEFAULT_SITE,),
                 conv_kernel: int = DEFAULT_CONV_KERNEL, context: TokenContext | None = None,
                 table_dtype: torch.dtype = torch.bfloat16, seed: int = 0):
        super().__init__()
        if not _is_power_of_two(rows):
            raise ValueError("rows must be a positive power of two")
        if len(set(sites)) != len(sites) or any(site < 0 for site in sites):
            raise ValueError("sites must be distinct non-negative layer indices")
        self.hidden = int(hidden)
        self.value_dim = int(value_dim)
        self.rows = int(rows)
        self.sites = tuple(int(site) for site in sites)
        self.conv_kernel = int(conv_kernel)
        self.context = context or TokenContext()
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.rows, self.value_dim, dtype=table_dtype))
             for _ in self.sites])
        self.key_proj = nn.ModuleList(
            [nn.Linear(self.hidden, self.value_dim, bias=False, dtype=torch.float32)
             for _ in self.sites])
        self.value_proj = nn.ModuleList(
            [nn.Linear(self.value_dim, self.hidden, bias=False, dtype=torch.float32)
             for _ in self.sites])
        # Added once per token, not once per read; equals the grafted mean.
        self.out_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.hidden, dtype=torch.float32))
             for _ in self.sites])
        self.conv = nn.ModuleList(
            [CausalDepthwiseConv1d(HEADS * self.value_dim, self.conv_kernel)
             for _ in self.sites])
        # Carried convolution inputs, one entry per site. Not a buffer: it is
        # inference state, it is reset by the context generation, and it must
        # never enter a checkpoint or a state dict.
        self._conv_state: list[torch.Tensor | None] = [None] * len(self.sites)
        self._conv_generation: int | None = None
        self._conv_context: int | None = None
        # Diagnostic view of the most recent per-read gates (B, T, reads) and
        # the mixed read before the constant offset. Detached; not state.
        self.last_gate: torch.Tensor | None = None
        self.last_mixed: torch.Tensor | None = None
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for table in self.tables:
                # Deterministic small init; grafting overwrites every used row.
                nn.init.normal_(table, std=0.01, generator=generator)
            for proj in self.key_proj:
                nn.init.normal_(proj.weight, std=0.02, generator=generator)
            for proj in self.value_proj:
                # Zero output at step 0 when no graft is installed: the frozen
                # core is exactly reproduced (the step-0 parity test).
                nn.init.zeros_(proj.weight)
            for bias in self.out_bias:
                nn.init.zeros_(bias)

    # -- accounting -----------------------------------------------------------
    def geometry(self) -> MemoryGeometry:
        return MemoryGeometry(hidden=self.hidden, value_dim=self.value_dim,
                              rows=self.rows, reads=HEADS,
                              conv_kernel=self.conv_kernel, sites=len(self.sites))

    def active_parameter_report(self) -> dict:
        return self.geometry().report()

    # -- addresses ------------------------------------------------------------
    def addresses(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        return [support(tokens, site, self.rows) for site in self.sites]

    def collision_report(self, tokens: torch.Tensor) -> dict:
        """Occupancy over the *sampled* stream: reads, distinct identities, rows.

        `distinct_identities_per_token` is the union actually charged: 8 unless
        two heads of the same site collide on one address for that token.
        """
        flat_tokens = tokens.reshape(-1)[None, :]
        reads = 0
        distinct_reads = 0
        used_rows: set[int] = set()
        site_reports = []
        for site in self.sites:
            index = support(flat_tokens, site, self.rows).reshape(-1, HEADS)
            reads += int(index.numel())
            per_row = torch.unique(index, return_counts=True)
            distinct_reads += int(sum(len(torch.unique(index[row]))
                                      for row in range(index.shape[0])))
            used_rows.update(per_row[0].tolist())
            counts = per_row[1]
            site_reports.append({
                "site": site,
                "read_identities": int(index.numel()),
                "distinct_rows_read": int(per_row[0].numel()),
                "max_address_occupancy": int(counts.max().item()),
                "mean_address_occupancy": float(counts.float().mean().item()),
                "table_occupancy_fraction": float(per_row[0].numel() / self.rows),
            })
        return {
            "kind": "stage2_collision_report",
            "tokens": int(flat_tokens.numel()),
            "read_identities_total": reads,
            "distinct_identities_total": distinct_reads,
            "distinct_rows_total": len(used_rows),
            "rows_per_site": self.rows,
            "sites": site_reports,
        }

    # -- forward --------------------------------------------------------------
    def reset_state(self) -> None:
        """Drop carried convolution history between sequences."""
        self._conv_state = [None] * len(self.sites)
        self._conv_generation = None
        self._conv_context = None
        self.last_gate = None
        self.last_mixed = None

    def _state_for(self, site_index: int, batch: int) -> torch.Tensor | None:
        """Carried conv state for this sequence, or `None` on a new sequence."""
        generation = self.context.generation
        identity = id(self.context)
        if self._conv_generation != generation or self._conv_context != identity:
            self._conv_state = [None] * len(self.sites)
            self._conv_generation = generation
            self._conv_context = identity
        state = self._conv_state[site_index]
        if state is not None and state.shape[0] != batch:
            self._conv_state[site_index] = None
            return None
        return state

    def site_mixed(self, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """The gated 8-row read as float32, without the constant offset.

        Split out so a corrective arm can take the read without the learned
        per-token mean; `site_forward` adds it exactly as before, so the
        registered memory arms are numerically unchanged.
        """
        length = hidden_states.shape[-2]
        # Address the n-gram from the whole recorded prefix, then keep the
        # positions this call executes. Addressing `require(length)` directly
        # would turn a cached length-1 call into a zero-padded unigram-like
        # address and change what the table reads during decoding.
        tokens = self.context.prefix()
        if tokens.shape[-1] < length:
            raise RuntimeError(
                f"token context has {tokens.shape[-1]} positions, layer needs {length}")
        addresses = support(tokens, self.sites[site_index], self.rows)[:, -length:]
        table = self.tables[site_index]
        values = table[addresses].float()                      # [B,T,8,VD]
        flat = values.reshape(*values.shape[:-2], HEADS * self.value_dim)
        state = self._state_for(site_index, hidden_states.shape[0])
        flat, new_state = self.conv[site_index].forward_cached(flat, state)
        self._conv_state[site_index] = new_state
        values = flat.reshape(*values.shape)
        key = self.key_proj[site_index](hidden_states.float())  # [B,T,VD]
        score = (values * key[:, :, None, :]).sum(-1) / (self.value_dim ** 0.5)
        gate = torch.sigmoid(score)                             # [B,T,8]
        # Non-persistent diagnostic state, like `CorrectiveMemory.last_gate`.
        # It is detached and never enters a checkpoint or a state dict; the
        # forward arithmetic is unchanged.
        self.last_gate = gate.detach()
        self.last_mixed = None  # set by site_forward after the read is summed
        projected = self.value_proj[site_index](values)         # [B,T,8,H]
        return (gate[..., None] * projected).sum(dim=2)         # [B,T,H]

    def site_forward(self, site_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed = self.site_mixed(site_index, hidden_states)
        self.last_mixed = mixed.detach()
        return (mixed + self.out_bias[site_index]).to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor, site_index: int = 0) -> torch.Tensor:
        return self.site_forward(site_index, hidden_states)

    # -- grafting hooks -------------------------------------------------------
    @torch.no_grad()
    def set_grafted_rows(self, site_index: int, rows: torch.Tensor,
                         counts: torch.Tensor) -> None:
        table = self.tables[site_index]
        if rows.shape != table.shape:
            raise ValueError(f"grafted rows {tuple(rows.shape)} != table "
                             f"{tuple(table.shape)}")
        table.copy_(rows.to(table.dtype))
        self.register_buffer(f"graft_counts_{site_index}", counts.to(torch.int32),
                             persistent=False)

    def graft_counts(self, site_index: int) -> torch.Tensor | None:
        name = f"graft_counts_{site_index}"
        return getattr(self, name, None)


def state_digest(tensor: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(
        tensor.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()
