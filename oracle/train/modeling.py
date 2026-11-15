#!/usr/bin/env python3
"""T1 model wrappers: routed FFN groups, specialist banks and dense adapters.

The frozen backbone is the real Qwen3 checkpoint. Only the modules defined here
have trainable parameters. Each wrapper reads the frozen tensors its admission
config declares.

Routed execution uses *capacity dispatch*: the top-k (token, expert) pairs are
sorted on the device, scattered into a fixed-capacity buffer of shape
`[experts, capacity, hidden]`, and evaluated with one batched matmul per
projection. The selector never leaves the device, so there is no host
synchronisation per layer; a token's real computation still reads only its k
selected experts' weights. Padding rows are zeros and their outputs are
discarded, and the padded FLOPs are reported separately from the realized
per-token set. Capacity is `ceil(tokens * k / experts * factor)` with factor
1.25; overflow pairs are dropped and counted (they are rare under the
load-balancing auxiliary loss).

The combine is *slot-wise* rather than an `index_add_` accumulation: the
per-pair outputs are returned to token-major order with a gather and reduced
over the fixed slot axis. The terms are the same; the result is bitwise
deterministic, which `index_add_`'s CUDA atomics are not (T1.15 A6/A7, T2.4).

Routing is hard top-k in the forward pass and softmax in the backward pass
(straight-through). With `router_grad=False` the mask is a constant for the
distillation loss and the router receives no gradient from it: that is the
`H*` treatment, where the router is trained only on measured utility labels
(T2.5). `soft_mixture` is an evaluation-only ablation that computes all experts
weighted by the softmax; it is reported as a non-compliant routing gap, never
as a compliant arm.

**Overflow validity mask (P12.1).** `_RoutedDispatch.dispatch` returns `valid`
indexed by *sorted* position (the capacity-sorted pair order), while `combine`
restores the pair outputs to *token-major* order before gating them. Applying a
sorted-order mask to token-order positions is only correct when the sort
permutation happens to be the identity. `mask_order="legacy"` keeps that
shipped behaviour byte for byte, because the registered T2 arms were trained and
evaluated with it; `mask_order="pair"` applies the mask before the restore, to
the pair it describes. The corrected path is opt-in, recorded in every run
receipt, and audited by `oracle/eval/dispatch_audit.py` against
`oracle/train/reference_dispatch.py` (see `docs/P12.1-DISPATCH-AUDIT.md`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import HashedMemory, MemoryGeometry, TokenContext

CAPACITY_FACTOR = 1.25
#: Overflow-mask placements (P12.1). `legacy` is the shipped behaviour and the
#: default so no registered arm changes; `pair` is the corrected placement.
MASK_ORDER_LEGACY = "legacy"
MASK_ORDER_PAIR = "pair"
MASK_ORDERS = (MASK_ORDER_LEGACY, MASK_ORDER_PAIR)
#: 0-based depths of the four memory insertions (T2.2).
T2_MEMORY_LAYERS = (3, 10, 17, 24)
#: T2 arm names in their registered spelling (T2.1).
T2_ARMS = ("D", "M", "X", "H", "H*")


def _capacity(tokens: int, k: int, experts: int, factor: float = CAPACITY_FACTOR) -> int:
    return max(1, math.ceil(tokens * k / experts * factor))


class _RoutedDispatch(nn.Module):
    """Shared top-k capacity dispatch and scatter bookkeeping."""

    def __init__(self, hidden: int, experts: int, k: int,
                 mask_order: str = MASK_ORDER_LEGACY):
        super().__init__()
        self.hidden = hidden
        self.experts = experts
        self.k = k
        self.soft_mixture = False
        self.router_grad = True
        self.mask_order = mask_order
        self.capacity_factor = CAPACITY_FACTOR
        self.last_routing: torch.Tensor | None = None
        self.last_probs: torch.Tensor | None = None
        self._dropped_pairs = torch.zeros((), dtype=torch.long)
        #: Running count of dropped pairs over the whole run (P12.1 exposure
        #: counter; a Python int, so it never changes the compute graph).
        self.dropped_pairs_total = 0
        self.pairs_total = 0

    def routing(self, flat: torch.Tensor, router_weight: torch.Tensor):
        logits = F.linear(flat, router_weight)
        probs = logits.float().softmax(dim=-1)
        _values, top_index = probs.topk(self.k, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, top_index, 1.0)
        self.last_routing = top_index.detach()
        self.last_probs = probs
        if self.router_grad:
            straight_gate = mask + probs - probs.detach()
        else:
            # The forward value is identical (the hard 0/1 mask); the mask is
            # simply a constant for the distillation loss. Used by H* (T2.5).
            straight_gate = mask
        pair_weight = straight_gate.gather(-1, top_index).reshape(-1)
        return probs, straight_gate, top_index, pair_weight

    def dispatch(self, flat: torch.Tensor, top_index: torch.Tensor):
        """Return (buffer [experts, capacity, hidden], row, order, valid)."""
        tokens = flat.shape[0]
        device = flat.device
        pairs = tokens * self.k
        pair_expert = top_index.reshape(-1)
        pair_token = torch.arange(tokens, device=device,
                                  dtype=torch.long).unsqueeze(1).expand(tokens, self.k)
        pair_token = pair_token.reshape(-1)
        order = torch.argsort(pair_expert, stable=True)
        sorted_expert = pair_expert.index_select(0, order)
        sorted_token = pair_token.index_select(0, order)
        counts = torch.bincount(pair_expert, minlength=self.experts)
        start = torch.cumsum(counts, dim=0) - counts
        slot = torch.arange(pairs, device=device, dtype=torch.long) - start.index_select(
            0, sorted_expert)
        capacity = _capacity(tokens, self.k, self.experts, self.capacity_factor)
        valid = slot < capacity
        self._dropped_pairs = (~valid).sum()
        self.dropped_pairs_total += int(self._dropped_pairs)
        self.pairs_total += pairs
        # Invalid (overflow) pairs would index past the buffer; clamp their row
        # so the gather is in range. Their contribution is masked out later.
        row = (sorted_expert * capacity + slot).clamp(max=self.experts * capacity - 1)
        buffer = flat.new_zeros(self.experts * capacity, self.hidden)
        gathered = flat.index_select(0, sorted_token)
        buffer.index_copy_(0, row[valid], gathered[valid])
        return buffer.view(self.experts, capacity, self.hidden), row, order, valid

    def combine(self, output: torch.Tensor, row: torch.Tensor, order: torch.Tensor,
                valid: torch.Tensor, pair_weight: torch.Tensor,
                tokens: int) -> torch.Tensor:
        """Deterministic slot-wise combine (T2.4).

        `row` is in sorted-pair order, so the sorted outputs are gathered back
        to token-major pair order with the inverse permutation. Pairs were built
        token-major (`pair = token * k + slot`), so the reshape is
        `[tokens, k, hidden]` and the reduction over slots is in a fixed order.
        Overflow pairs are zeroed; their incoming gradient is zero, so the
        clamped duplicate rows cannot introduce an order-dependent sum.

        `self.mask_order` decides *where* the zeroing happens (P12.1):

        * `"pair"` — `valid` is indexed by sorted position, so it is applied
          **before** the restore, when the tensor is still in sorted order.
          Each pair is gated by its own slot.
        * `"legacy"` — the shipped behaviour: `valid` is applied after the
          restore, so token-major pair `p` is gated by the validity of sorted
          position `p`. That is a different pair whenever any expert overflowed;
          it silently drops a kept pair and keeps an overflowing one, which then
          reads a clamped buffer row (possibly another token's activation).
        """
        pairs = tokens * self.k
        flat_out = output.reshape(-1, self.hidden).index_select(0, row)
        inverse = torch.empty_like(order)
        inverse.scatter_(0, order, torch.arange(pairs, device=order.device,
                                                dtype=order.dtype))
        if self.mask_order == MASK_ORDER_PAIR:
            # Sorted order: the mask and the weight both describe the pair this
            # tensor slot holds (`order[j]` is the pair at sorted position `j`).
            # Restore afterwards.
            weight = pair_weight.index_select(0, order).unsqueeze(-1)
            flat_out = flat_out * weight.to(flat_out.dtype)
            flat_out = flat_out * valid.unsqueeze(-1).to(flat_out.dtype)
            flat_out = flat_out.index_select(0, inverse)
        elif self.mask_order == MASK_ORDER_LEGACY:
            flat_out = flat_out.index_select(0, inverse)
            flat_out = flat_out * pair_weight.unsqueeze(-1).to(flat_out.dtype)
            flat_out = flat_out * valid.unsqueeze(-1).to(flat_out.dtype)
        else:  # pragma: no cover - refused at construction
            raise ValueError(f"unknown mask_order {self.mask_order!r}")
        return flat_out.view(tokens, self.k, self.hidden).sum(dim=1)


class RoutedGroupFFN(_RoutedDispatch):
    """Top-k selection over contiguous groups of a SwiGLU FFN.

    Replaces `Qwen3MLP`. Group slices of the frozen gate/up/down tensors are
    read only for the tokens that select them. A rank-r residual adapter per
    group acts on the group's pre-down activation and is zero-initialised.
    """

    def __init__(self, mlp: nn.Module, groups: int, k: int, adapter_rank: int,
                 router: nn.Linear, mask_order: str = MASK_ORDER_LEGACY):
        super().__init__(mlp.gate_proj.weight.shape[1], groups, k, mask_order=mask_order)
        self.rank = adapter_rank
        gate = mlp.gate_proj.weight
        up = mlp.up_proj.weight
        down = mlp.down_proj.weight
        intermediate, hidden = gate.shape
        if intermediate % groups:
            raise ValueError(f"intermediate {intermediate} not divisible by {groups}")
        self.group_width = intermediate // groups
        # Contiguous cached views; the frozen tensors themselves are unchanged.
        self.register_buffer("gate_groups",
                             gate.detach().view(groups, self.group_width, hidden).clone(),
                             persistent=False)
        self.register_buffer("up_groups",
                             up.detach().view(groups, self.group_width, hidden).clone(),
                             persistent=False)
        self.register_buffer("down_groups",
                             down.detach().view(hidden, groups, self.group_width)
                             .permute(1, 2, 0).contiguous().clone(), persistent=False)
        self.router = router
        self.adapter_down = nn.Parameter(
            torch.empty(groups, adapter_rank, self.group_width, dtype=torch.float32))
        self.adapter_up = nn.Parameter(
            torch.zeros(groups, self.group_width, adapter_rank, dtype=torch.float32))
        with torch.no_grad():
            nn.init.normal_(self.adapter_down, std=0.02)

    def _group_ffn(self, buffer: torch.Tensor,
                   cast: torch.dtype | None = None) -> torch.Tensor:
        gate_w = self.gate_groups if cast is None else self.gate_groups.to(cast)
        up_w = self.up_groups if cast is None else self.up_groups.to(cast)
        down_w = self.down_groups if cast is None else self.down_groups.to(cast)
        gate = torch.bmm(buffer, gate_w.transpose(1, 2))
        up = torch.bmm(buffer, up_w.transpose(1, 2))
        activation = F.silu(gate) * up
        if self.rank:
            delta = torch.bmm(torch.bmm(activation, self.adapter_down.transpose(1, 2)),
                              self.adapter_up.transpose(1, 2))
            activation = activation + delta
        return torch.bmm(activation, down_w)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        probs, straight_through, top_index, pair_weight = self.routing(
            flat, self.router.weight)
        if self.soft_mixture:
            out = torch.zeros_like(flat)
            for group in range(self.experts):
                weight = probs[:, group].unsqueeze(-1)
                gate = F.linear(flat, self.gate_groups[group])
                up = F.linear(flat, self.up_groups[group])
                activation = F.silu(gate) * up
                if self.rank:
                    activation = activation + F.linear(
                        F.linear(activation, self.adapter_down[group]),
                        self.adapter_up[group])
                out = out + F.linear(activation, self.down_groups[group]) * weight
            return out.reshape(shape)
        buffer, row, order, valid = self.dispatch(flat, top_index)
        output = self._group_ffn(buffer)
        combined = self.combine(output, row, order, valid, pair_weight,
                                flat.shape[0])
        return combined.reshape(shape)


class SpecialistBank(_RoutedDispatch):
    """Residual bank of m small SwiGLU FFNs, hard top-k routed per token.

    `full_bank` (with `exclude`) is the counterfactual mode used only by the
    utility labeller: every specialist is executed with unit weight except the
    one named by `exclude`, which is masked out. It is an intervention, not a
    deployed routing mode, and it bypasses the router entirely.
    """

    def __init__(self, hidden: int, width: int, specialists: int, k: int,
                 router: nn.Linear, mask_order: str = MASK_ORDER_LEGACY):
        if mask_order not in MASK_ORDERS:
            raise ValueError(f"mask_order must be one of {MASK_ORDERS}, got {mask_order!r}")
        super().__init__(hidden, specialists, k, mask_order=mask_order)
        self.width = width
        self.full_bank = False
        self.exclude: int | None = None
        self.gate = nn.Parameter(torch.empty(specialists, width, hidden,
                                             dtype=torch.float32))
        self.up = nn.Parameter(torch.empty(specialists, width, hidden,
                                           dtype=torch.float32))
        self.down = nn.Parameter(torch.zeros(specialists, hidden, width,
                                             dtype=torch.float32))
        self.router = router
        with torch.no_grad():
            nn.init.normal_(self.gate, std=0.02)
            nn.init.normal_(self.up, std=0.02)

    def _experts(self, buffer: torch.Tensor) -> torch.Tensor:
        gate = torch.bmm(buffer, self.gate.transpose(1, 2))
        up = torch.bmm(buffer, self.up.transpose(1, 2))
        return torch.bmm(F.silu(gate) * up, self.down.transpose(1, 2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        if self.full_bank:
            out = torch.zeros_like(flat)
            for expert in range(self.experts):
                if self.exclude is not None and expert == self.exclude:
                    continue
                gate = F.linear(flat, self.gate[expert])
                up = F.linear(flat, self.up[expert])
                out = out + F.linear(F.silu(gate) * up, self.down[expert])
            return out.reshape(shape)
        probs, straight_through, top_index, pair_weight = self.routing(
            flat, self.router.weight)
        if self.soft_mixture:
            out = torch.zeros_like(flat)
            for expert in range(self.experts):
                gate = F.linear(flat, self.gate[expert])
                up = F.linear(flat, self.up[expert])
                out = out + F.linear(F.silu(gate) * up, self.down[expert]) * \
                    probs[:, expert].unsqueeze(-1)
            return out.reshape(shape)
        buffer, row, order, valid = self.dispatch(flat, top_index)
        output = self._experts(buffer)
        combined = self.combine(output, row, order, valid, pair_weight,
                                flat.shape[0])
        return combined.reshape(shape)


class DenseAdapter(nn.Module):
    """One dense SwiGLU adapter added to every token (the experimental control)."""

    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(width, hidden, dtype=torch.float32))
        self.up = nn.Parameter(torch.empty(width, hidden, dtype=torch.float32))
        self.down = nn.Parameter(torch.zeros(hidden, width, dtype=torch.float32))
        with torch.no_grad():
            nn.init.normal_(self.gate, std=0.02)
            nn.init.normal_(self.up, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.linear(hidden_states, self.gate)
        up = F.linear(hidden_states, self.up)
        return F.linear(F.silu(gate) * up, self.down)


class AdditiveFFN(nn.Module):
    """Base frozen FFN plus an extra residual module.

    The extra's output is cast to the base's dtype before the add. Without the
    cast, an fp32 projection result would promote the whole residual stream to
    fp32 for the rest of the model: that changes the numerics of every
    downstream layer (and doubles activation memory) even when the extra is
    exactly zero at step 0. The cast makes the zero point exact and keeps the
    registered bf16 compute path.
    """

    def __init__(self, base: nn.Module, extra: nn.Module):
        super().__init__()
        self.base = base
        self.extra = extra

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        extra = self.extra(hidden_states)
        if extra.dtype != base.dtype:
            extra = extra.to(base.dtype)
        return base + extra


class AdditiveExtras(nn.Module):
    """Several residual modules summed in a fixed order (hybrid arms)."""

    def __init__(self, extras: list[nn.Module]):
        super().__init__()
        if not extras:
            raise ValueError("AdditiveExtras needs at least one module")
        self.extras = nn.ModuleList(extras)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first = self.extras[0](hidden_states)
        output = first
        for module in self.extras[1:]:
            extra = module(hidden_states)
            if extra.dtype != output.dtype:
                extra = extra.to(output.dtype)
            output = output + extra
        return output


# ---------------------------------------------------------------------------
# construction


def _install_token_context(model: nn.Module, context: TokenContext) -> None:
    """Give the memory modules the prefix the layer is actually processing."""

    def hook(_module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            context.set(ids)

    model.register_forward_pre_hook(hook, with_kwargs=True)


@dataclass
class ArmModel:
    model: nn.Module
    arm: str
    kind: str
    router_modules: list[nn.Module]
    extra_modules: list[nn.Module]
    admission: dict
    config: dict
    profile: str = "t1"
    context: TokenContext | None = None
    memory_layers: tuple[int, ...] = field(default_factory=tuple)
    trainable_names: list[str] = field(default_factory=list)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def set_soft_mixture(self, value: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, _RoutedDispatch):
                module.soft_mixture = value

    def set_router_grad(self, value: bool) -> None:
        """`False` freezes the distillation path into every router (H*, T2.5)."""
        for module in self.model.modules():
            if isinstance(module, _RoutedDispatch):
                module.router_grad = value

    def set_mask_order(self, value: str) -> None:
        """Select the overflow-mask placement (P12.1).

        `"legacy"` is the shipped behaviour and the default; `"pair"` is the
        corrected placement. A registered run must record which one it used.
        """
        if value not in MASK_ORDERS:
            raise ValueError(f"mask_order must be one of {MASK_ORDERS}, got {value!r}")
        for module in self.model.modules():
            if isinstance(module, _RoutedDispatch):
                module.mask_order = value

    def mask_order(self) -> str:
        orders = {module.mask_order for module in self.model.modules()
                  if isinstance(module, _RoutedDispatch)}
        if len(orders) > 1:
            raise RuntimeError(f"mixed mask orders in one arm: {sorted(orders)}")
        return orders.pop() if orders else MASK_ORDER_LEGACY

    def dispatch_exposure(self) -> dict:
        """Dropped-pair exposure over the whole run (P12.1 accounting).

        `_dropped_pairs` is per forward pass; these totals accumulate. The
        counters change no value, so an arm's numbers are unaffected by reading
        them.
        """
        dropped = 0
        pairs = 0
        for module in self.model.modules():
            if isinstance(module, _RoutedDispatch):
                dropped += int(module.dropped_pairs_total)
                pairs += int(module.pairs_total)
        return {"mask_order": self.mask_order(), "dropped_pairs": dropped,
                "pairs": pairs, "dropped_fraction": (dropped / pairs) if pairs else 0.0}

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Uniform step/forward API for every T1 and T2 arm."""
        return self.model(input_ids=input_ids, use_cache=False, **kwargs)

    def banks(self) -> list[nn.Module]:
        return [module for module in self.model.modules()
                if isinstance(module, SpecialistBank)]

    def memories(self) -> list[nn.Module]:
        return [module for module in self.model.modules()
                if isinstance(module, HashedMemory)]

    def extra_ledger(self) -> dict:
        """Element counts of every added module, by role.

        This is the Python side of the admission declaration: the test
        `test_t2_arm_ledger_matches_admission_config` requires every count here
        to equal the matching `execution_uses` slice in the Rust config.
        """
        roles = {
            "memory_values": 0, "memory_projection": 0, "memory_gate": 0,
            "memory_index": 0, "specialist_gate": 0, "specialist_up": 0,
            "specialist_down": 0, "dense_adapter": 0, "router": 0,
        }
        memories = self.memories()
        banks = self.banks()
        adapters = [module for module in self.model.modules()
                    if isinstance(module, DenseAdapter)]
        for module in memories:
            roles["memory_values"] += module.table.numel()
            roles["memory_projection"] += module.proj.weight.numel()
            roles["memory_gate"] += module.gate.weight.numel()
        for module in banks:
            roles["specialist_gate"] += module.gate.numel()
            roles["specialist_up"] += module.up.numel()
            roles["specialist_down"] += module.down.numel()
        for module in adapters:
            roles["dense_adapter"] += module.gate.numel() + module.up.numel() \
                + module.down.numel()
        roles["router"] = sum(module.weight.numel() for module in self.router_modules)
        return {
            "roles": roles,
            "memory_layers": len(memories),
            "bank_layers": len(banks),
            "dense_adapter_layers": len(adapters),
            "total": sum(roles.values()),
        }

    def active_extra_elements(self) -> int:
        """Active added elements: what the arm reads for one output token."""
        active = 0
        for module in self.memories():
            geometry = module.geometry()
            active += geometry.active_per_layer
        for module in self.banks():
            per_expert = module.width * module.hidden * 3
            active += module.k * per_expert
            active += module.router.weight.numel()
        for module in self.model.modules():
            if isinstance(module, DenseAdapter):
                active += module.gate.numel() + module.up.numel() + module.down.numel()
        return active

    def routing_summary(self) -> dict:
        summary = {}
        for index, module in enumerate(self.model.modules()):
            if isinstance(module, _RoutedDispatch) and module.last_routing is not None:
                counts = torch.bincount(module.last_routing.reshape(-1),
                                        minlength=module.experts)
                summary[str(index)] = {
                    "counts": [int(item) for item in counts.tolist()],
                    "dropped_pairs": int(module._dropped_pairs.item()),
                }
        return summary


def _router(hidden: int, outputs: int, seed: int | None) -> nn.Linear:
    router = nn.Linear(hidden, outputs, bias=False, dtype=torch.float32)
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            router.weight.copy_(torch.empty_like(router.weight).normal_(
                mean=0.0, std=0.02, generator=generator))
    return router


def build_arm(model, arm: str, config: dict, admission: dict, seed: int,
              profile: str = "t1", mask_order: str = MASK_ORDER_LEGACY) -> ArmModel:
    """Wrap a loaded base model in the arm's trainable modules.

    `profile="t1"` keeps T1's arm letters exactly. `profile="t2"` registers the
    five T2 arms; T2's `D` is T1's `C` (the parameter-matched dense control),
    not T1's untrained `D` (T2.1). `mask_order` selects the overflow-mask
    placement of `_RoutedDispatch.combine` (P12.1); the default is the shipped
    behaviour so a registered run is unchanged.
    """
    if mask_order not in MASK_ORDERS:
        raise ValueError(f"mask_order must be one of {MASK_ORDERS}, got {mask_order!r}")
    layers = model.model.layers
    hidden = model.config.hidden_size
    router_modules: list[nn.Module] = []
    extra_modules: list[nn.Module] = []
    context: TokenContext | None = None
    memory_layers: tuple[int, ...] = ()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if profile == "t2":
        if arm not in T2_ARMS:
            raise ValueError(f"unknown T2 arm {arm}")
        wants_memory = arm in ("M", "H", "H*")
        wants_experts = arm in ("X", "H", "H*")
        if wants_memory:
            context = TokenContext()
            _install_token_context(model, context)
        dense_width = 2048 if arm == "D" else 0
        for index, layer in enumerate(layers):
            extras: list[nn.Module] = []
            if wants_experts:
                router = _router(hidden, 8, seed + 2000 + index)
                bank = SpecialistBank(hidden, 512, 8, 4, router)
                router_modules.append(router)
                extra_modules.append(bank)
                extras.append(bank)
            if wants_memory and index in T2_MEMORY_LAYERS:
                memory = HashedMemory(hidden, index, context,
                                      geometry=MemoryGeometry(hidden=hidden),
                                      seed=seed + 3000 + index)
                extra_modules.append(memory)
                extras.append(memory)
            if dense_width:
                adapter = DenseAdapter(hidden, dense_width)
                extra_modules.append(adapter)
                extras.append(adapter)
            if extras:
                extra = extras[0] if len(extras) == 1 else AdditiveExtras(extras)
                layer.mlp = AdditiveFFN(layer.mlp, extra)
        kind = {True: "hybrid", False: "conditional"}[wants_memory or wants_experts]
        if arm == "D":
            kind = "dense_adapter"
        elif arm == "M":
            kind = "memory"
        elif arm == "X":
            kind = "specialists"
        memory_layers = T2_MEMORY_LAYERS if wants_memory else ()
        trainable = ArmModel(model=model, arm=arm, kind=kind,
                             router_modules=router_modules, extra_modules=extra_modules,
                             admission=admission, config=config, profile=profile,
                             context=context, memory_layers=memory_layers)
        if arm == "H*":
            trainable.set_router_grad(False)
        trainable.set_mask_order(mask_order)
        trainable.declared_active = admission["active_unique_params"]
        trainable.trainable_names = [name for name, parameter
                                     in model.named_parameters()
                                     if parameter.requires_grad]
        return trainable

    if arm in ("A", "A0"):
        groups, k, rank = 16, 5, 32
        for index, layer in enumerate(layers):
            router = _router(hidden, groups, seed + 1000 + index)
            wrapper = RoutedGroupFFN(layer.mlp, groups, k, rank, router)
            layer.mlp = wrapper
            router_modules.append(router)
            extra_modules.append(wrapper)
        kind = "routed_groups"
    elif arm == "B":
        specialists, k, width = 8, 4, 512
        for index, layer in enumerate(layers):
            router = _router(hidden, specialists, seed + 2000 + index)
            bank = SpecialistBank(hidden, width, specialists, k, router)
            layer.mlp = AdditiveFFN(layer.mlp, bank)
            router_modules.append(router)
            extra_modules.append(bank)
        kind = "specialists"
    elif arm in ("C", "C2"):
        width = 2048 if arm == "C" else 4608
        for layer in layers:
            adapter = DenseAdapter(hidden, width)
            layer.mlp = AdditiveFFN(layer.mlp, adapter)
            extra_modules.append(adapter)
        kind = "dense_adapter"
    elif arm == "D":
        kind = "dense"
    else:
        raise ValueError(f"unknown arm {arm}")

    trainable = ArmModel(model=model, arm=arm, kind=kind,
                         router_modules=router_modules, extra_modules=extra_modules,
                         admission=admission, config=config)
    trainable.set_mask_order(mask_order)
    trainable.declared_active = admission["active_unique_params"]
    trainable.trainable_names = [name for name, parameter
                                 in model.named_parameters() if parameter.requires_grad]
    return trainable


# ---------------------------------------------------------------------------
# admission records


def load_admission(config_path: str, admission_path: str) -> tuple[dict, dict]:
    import hashlib

    with open(config_path, "rb") as handle:
        config_bytes = handle.read()
    config = json.loads(config_bytes)
    with open(admission_path, "r", encoding="utf-8") as handle:
        admission = json.load(handle)
    digest = hashlib.sha256(config_bytes).hexdigest()
    if admission.get("schema") != "selvedge/admission-record/v1":
        raise ValueError(f"{admission_path}: unexpected schema")
    if not admission.get("admitted"):
        raise ValueError(f"{admission_path}: arm is not admitted: {admission.get('refusal')}")
    if admission.get("config_sha256") != digest:
        raise ValueError(f"{admission_path}: config_sha256 does not match {config_path}")
    if admission["active_unique_params"] > 1_000_000_000:
        raise ValueError(f"{admission_path}: admitted path is over the ceiling")
    return config, admission


def resource_report(arm_model: ArmModel, deployment_bytes: int = 2) -> dict:
    """Active and accessible capacity, and the bytes triple, for one arm.

    `weight_bytes_per_token_modelled` and the memory read bytes are modelled at
    the declared dtype (distinct identities read, bf16 by default). They are
    labelled `modelled` and never presented as measured I/O.
    """
    report = {
        "arm": arm_model.arm,
        "profile": arm_model.profile,
        "active_unique_params": getattr(arm_model, "declared_active", None),
        "total_accessible": (arm_model.config or {}).get("total_accessible"),
        "trainable_elements": arm_model.trainable_elements(),
        "extra_ledger": arm_model.extra_ledger(),
        "dtype_bytes": deployment_bytes,
    }
    memories = arm_model.memories()
    if memories:
        geometry = memories[0].geometry()
        report["memory"] = {
            "insertions": len(memories),
            "layers": list(arm_model.memory_layers),
            "active": geometry.active_report(len(memories)),
            "bytes": geometry.bytes_report(len(memories), deployment_bytes),
        }
    banks = arm_model.banks()
    if banks:
        per_layer = banks[0].k * 3 * banks[0].width * banks[0].hidden
        active = len(banks) * per_layer
        router = sum(bank.router.weight.numel() for bank in banks)
        report["experts"] = {
            "layers": len(banks),
            "m": banks[0].experts,
            "k": banks[0].k,
            "width": banks[0].width,
            "active_elements_per_token": active,
            "weight_bytes_per_token_modelled": (active + router) * deployment_bytes,
            "router_elements": router,
            "bank_total_elements": sum(bank.gate.numel() + bank.up.numel()
                                       + bank.down.numel() for bank in banks),
            "basis": "modelled",
        }
    adapters = [module for module in arm_model.model.modules()
                if isinstance(module, DenseAdapter)]
    if adapters:
        active = sum(module.gate.numel() + module.up.numel() + module.down.numel()
                     for module in adapters)
        report["dense_adapter"] = {
            "layers": len(adapters),
            "active_elements_per_token": active,
            "weight_bytes_per_token_modelled": active * deployment_bytes,
            "basis": "modelled",
        }
    return report


def base_logits_for(base_model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda), \
            torch.inference_mode():
        return base_model(input_ids=input_ids, use_cache=False).logits.float().clone()


def group_reconstruction_report(module: RoutedGroupFFN, tokens: int = 64,
                                seed: int = 20261115) -> dict:
    """f32 layer-level check that the all-groups path equals the dense FFN.

    The end-to-end bf16 logit comparison is not a valid identity test for the
    routed arm: summing 16 group slices changes the reduction order, which
    moves bf16 logits by more than 1e-3 (measured and reported). This check
    widens the weights to f32 and compares the wrapper's all-groups output with
    the dense formula, which is the same instrument M1's Court A used.
    """
    with torch.no_grad():
        groups = module.experts
        gate = module.gate_groups.float().reshape(-1, module.hidden)
        up = module.up_groups.float().reshape(-1, module.hidden)
        down = torch.cat([module.down_groups[index].float().t()
                          for index in range(groups)], dim=1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(tokens, module.hidden, generator=generator,
                        dtype=torch.float32).to(gate.device)
        activation = F.silu(x @ gate.t()) * (x @ up.t())
        reference = activation @ down.t()
        buffer = x.unsqueeze(0).expand(groups, tokens, module.hidden).contiguous()
        saved = [(item.k, item.rank) for item in [module]]
        module.k = groups
        module.rank = 0
        got = module._group_ffn(buffer, cast=torch.float32).sum(dim=0)
        module.k, module.rank = saved[0]
    delta = (got - reference).abs()
    return {
        "kind": "group_reconstruction_f32",
        "groups": groups,
        "tokens": tokens,
        "max_abs_delta": float(delta.max().item()),
        "mean_abs_delta": float(delta.mean().item()),
        "tolerance": 1e-3,
        "ok": bool(float(delta.max().item()) <= 1e-3),
        "note": "f32 widened weights; the bf16 routed sum changes the reduction "
                "order and is not bitwise equal by construction",
    }


def adapter_zero_report(arm_model: ArmModel) -> dict:
    """Every zero-initialised residual projection must be exactly zero."""
    maxima = []
    for module in arm_model.model.modules():
        if isinstance(module, RoutedGroupFFN):
            maxima.append(float(module.adapter_up.abs().max().item()))
        elif isinstance(module, SpecialistBank):
            maxima.append(float(module.down.abs().max().item()))
        elif isinstance(module, DenseAdapter):
            maxima.append(float(module.down.abs().max().item()))
        elif isinstance(module, HashedMemory):
            maxima.append(float(module.proj.weight.abs().max().item()))
    worst = max(maxima) if maxima else 0.0
    return {
        "kind": "adapter_zero_init",
        "modules": len(maxima),
        "max_abs": worst,
        "tolerance": 0.0,
        "ok": bool(worst == 0.0),
    }


def zero_point_report(arm_model: ArmModel, base_logits: torch.Tensor | None,
                      input_ids: torch.Tensor) -> dict:
    """Step-0 identity evidence, dispatching by arm kind.

    B, C and C2 add a zero-initialised residual adapter, so the modified model
    must reproduce the frozen base's logits exactly (tolerance 1e-3). A replaces
    the dense FFN with a routed sum, so its identity test is the f32 layer-level
    reconstruction plus the exact-zero adapter check.
    """
    zero = adapter_zero_report(arm_model)
    if arm_model.profile == "t1" and arm_model.arm in ("A", "A0"):
        modules = [module for module in arm_model.model.modules()
                   if isinstance(module, RoutedGroupFFN)]
        reconstructions = [group_reconstruction_report(module) for module in modules[:3]]
        ok = zero["ok"] and all(item["ok"] for item in reconstructions)
        return {
            "arm": arm_model.arm,
            "kind": "routed_group_reconstruction",
            "adapter_zero": zero,
            "reconstructions": reconstructions,
            "max_abs_reconstruction_delta": max(item["max_abs_delta"]
                                                for item in reconstructions),
            "tolerance": 1e-3,
            "ok": bool(ok),
            "end_to_end_bf16_note": "not used as a pass condition for A: the "
                                    "grouped reduction moves bf16 logits beyond 1e-3 "
                                    "by construction",
        }
    assert base_logits is not None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda), \
            torch.inference_mode():
        modified_logits = arm_model.model(input_ids=input_ids,
                                          use_cache=False).logits.float()
    delta = (modified_logits - base_logits).abs()
    return {
        "arm": arm_model.arm,
        "kind": "end_to_end_bf16_identity",
        "adapter_zero": zero,
        "max_abs_logit_delta": float(delta.max().item()),
        "mean_abs_logit_delta": float(delta.mean().item()),
        "tolerance": 1e-3,
        "ok": bool(float(delta.max().item()) <= 1e-3 and zero["ok"]),
        "device": str(input_ids.device),
    }
