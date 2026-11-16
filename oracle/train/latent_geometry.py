#!/usr/bin/env python3
"""T3 (P10.2 / P12.3) — latent-geometry court.

**What this is.** An accounting replication of published latent-MoE geometry,
not a discovery. LatentMoE (2601.18089) reports accuracy per FLOP/parameter;
Shaping Latent MoE (2610.14644) bounds local Jacobian rank by the expert input and
intermediate widths and finds near-square experts win at fixed per-expert
parameters; Multi-View Latent MoE (2610.40577) shows single-router rankings
disagree with expert-utility rankings. SELVEDGE's contribution here is the
**union/bytes ledger at a fixed active cost** and utility diversity as a
behavioural measurement. The pre-registration says so; see
`docs/EXPERIMENT-CONTRACT.md` (T3).

**Geometry.** A shared projection `P: d_model -> d_latent` and `Q: d_latent ->
d_model` per layer, `m` stored experts in latent space, `k` active. Each expert
is **square**: input `d_latent`, intermediate width `w = d_latent`, output
`d_latent`, i.e. `3 * L^2` elements. At a fixed per-layer active budget `A`:

```
k(L) = largest k such that  2 d L  +  k * 3 L^2  +  d * m  <= A,   m = 2 k
```

so a smaller latent buys more (tiny) active experts and a wider latent buys
fewer (bigger) ones — at the same active cost. The reference point is the T2 `X`
expert form (input `1024`, width `512`, `k = 4`, `m = 8`), whose active cost per
layer is exactly `A`.

**What is reported per point.** The Jacobian rank proxy (a real numerical SVD of
the layer's local input->output Jacobian on a sample; the analytic bound is
`min(k * L, d_model)`), utility diversity under the deployed `k`-read, verified
task success through the same T2 instrument, VRAM, decode overhead, and the
active union from the Rust ledger.

The layer's dispatch writes each `(token, slot)` pair once and sums over a fixed
slot order, so it is bitwise deterministic on any device. All geometry,
arithmetic and utility estimators are CPU-tested in
`oracle/tests/test_t3_latent_geometry.py`; the GPU entry points are `--train`
and `--report`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)

SCHEMA_SPEC = "selvedge/t3-latent-spec/v1"
SCHEMA_SWEEP = "selvedge/t3-latent-sweep/v1"
SCHEMA_RUN = "selvedge/t3-latent-run/v1"
SCHEMA_REPORT = "selvedge/t3-latent-report/v1"
LATENT_DIMS = (128, 256, 384, 512)
BANK_FACTOR = 2
CEILING = 1_000_000_000


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# geometry arithmetic


@dataclass(frozen=True)
class LatentSpec:
    d_model: int = 1024
    d_latent: int = 256
    experts: int = 58
    k: int = 29
    expert_width: int = 256
    layers: int = 28
    bank_factor: int = BANK_FACTOR
    reference: bool = False

    @property
    def kind(self) -> str:
        return "reference" if self.reference else "latent"

    @property
    def latent(self) -> bool:
        return not self.reference

    @property
    def shared_elements(self) -> int:
        return 0 if self.reference else 2 * self.d_model * self.d_latent

    @property
    def expert_elements(self) -> int:
        return 3 * self.d_latent * self.expert_width

    @property
    def router_elements(self) -> int:
        return self.d_model * self.experts

    @property
    def active_per_layer(self) -> int:
        return (self.shared_elements + self.k * self.expert_elements
                + self.router_elements)

    @property
    def active_total(self) -> int:
        return self.layers * self.active_per_layer

    @property
    def accessible_per_layer(self) -> int:
        return (self.shared_elements + self.experts * self.expert_elements
                + self.router_elements)

    @property
    def accessible_total(self) -> int:
        return self.layers * self.accessible_per_layer

    @property
    def near_square(self) -> bool:
        return abs(self.expert_width - self.d_latent) <= max(1, self.d_latent // 8)

    @property
    def expert_rank_bound(self) -> int:
        """Published bound for one expert: min(input width, intermediate width)."""
        return min(self.d_latent, self.expert_width)

    @property
    def jacobian_rank_bound(self) -> int:
        """Layer bound for this geometry.

        A latent layer's experts all read the *same* rank-`L` code produced by
        the shared projection, so the sum of their maps is `Q (sum J_e) P` and
        its local rank is bounded by `min(L, d_model)` — not by `k * L`. The
        reference form reads the native hidden state at every expert, so its
        bound is `min(k * expert width, d_model)`. Reporting the two different
        bounds is the honest version of "latent width bounds rank".
        """
        if self.latent:
            return min(self.d_latent, self.d_model)
        return min(self.k * self.expert_rank_bound, self.d_model)

    def report(self) -> dict:
        return {
            "kind": self.kind, "d_model": self.d_model, "d_latent": self.d_latent,
            "experts": self.experts, "k": self.k,
            "expert_width": self.expert_width, "layers": self.layers,
            "near_square": self.near_square,
            "shared_elements_per_layer": self.shared_elements,
            "expert_elements": self.expert_elements,
            "router_elements": self.router_elements,
            "active_per_layer": self.active_per_layer,
            "active_total": self.active_total,
            "accessible_per_layer": self.accessible_per_layer,
            "accessible_total": self.accessible_total,
            "expert_rank_bound": self.expert_rank_bound,
            "jacobian_rank_bound": self.jacobian_rank_bound,
            "active_flops_per_token": 2 * self.active_total,
        }

    def as_dict(self) -> dict:
        return replace(self, layers=self.layers).__dict__.copy()

    @classmethod
    def load(cls, path: str) -> "LatentSpec":
        with open(os.path.join(REPO, path), encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.pop("schema", None)
        fields = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**fields)

    def save(self, path: str) -> str:
        payload = {"schema": SCHEMA_SPEC, **self.report(), **self.as_dict()}
        full = os.path.join(REPO, path)
        os.makedirs(os.path.dirname(os.path.abspath(full)) or ".", exist_ok=True)
        tmp = full + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, full)
        return path


def reference_spec(d_model: int = 1024, width: int = 512, k: int = 4,
                   experts: int = 8, layers: int = 28) -> LatentSpec:
    """The T2 `X` expert form: native-width input, `k = 4` of `m = 8`."""
    return LatentSpec(d_model=d_model, d_latent=d_model, experts=experts, k=k,
                      expert_width=width, layers=layers, reference=True)


def reference_budget_per_layer(d_model: int = 1024, width: int = 512, k: int = 4,
                               experts: int = 8) -> int:
    return reference_spec(d_model, width, k, experts).active_per_layer


def latent_spec(d_latent: int, budget_per_layer: int | None = None,
                d_model: int = 1024, layers: int = 28,
                bank_factor: int = BANK_FACTOR) -> LatentSpec:
    """Largest `k` of square latent experts that fits the active budget."""
    if budget_per_layer is None:
        budget_per_layer = reference_budget_per_layer(d_model)
    if d_latent <= 0:
        raise ValueError("d_latent must be positive")
    shared = 2 * d_model * d_latent
    if shared >= budget_per_layer:
        raise ValueError(f"latent width {d_latent} alone exceeds the budget "
                         f"({shared} >= {budget_per_layer})")
    k = 0
    while True:
        candidate = k + 1
        experts = bank_factor * candidate
        active = shared + candidate * 3 * d_latent * d_latent + d_model * experts
        if active > budget_per_layer:
            break
        k = candidate
    if k == 0:
        raise ValueError("no active expert fits the budget")
    return LatentSpec(d_model=d_model, d_latent=d_latent,
                      experts=bank_factor * k, k=k,
                      expert_width=d_latent, layers=layers,
                      bank_factor=bank_factor)


def sweep_specs(d_model: int = 1024, layers: int = 28,
                dims: tuple[int, ...] = LATENT_DIMS) -> list[LatentSpec]:
    budget = reference_budget_per_layer(d_model)
    return [latent_spec(dim, budget, d_model, layers) for dim in dims]


def sweep_report(d_model: int = 1024, layers: int = 28) -> dict:
    budget = reference_budget_per_layer(d_model)
    reference = reference_spec(d_model, layers=layers)
    rows = []
    for spec in sweep_specs(d_model, layers):
        row = spec.report()
        row["budget_per_layer"] = budget
        row["active_utilisation"] = spec.active_per_layer / budget
        row["active_slack_per_layer"] = budget - spec.active_per_layer
        row["active_vs_reference"] = spec.active_per_layer / reference.active_per_layer
        row["within_ceiling"] = spec.accessible_total <= CEILING
        row["decode_overhead"] = decode_overhead(spec, reference)
        rows.append(row)
    return {"schema": SCHEMA_SWEEP, "budget_per_layer": budget,
            "reference": reference.report(), "points": rows,
            "rule": "square experts (w = L), m = 2k, largest k that fits the "
                    "reference's per-layer active budget",
            "note": "arithmetic only; measured VRAM, timing and rank come from "
                    "--report on the GPU court"}


# ---------------------------------------------------------------------------
# the layer


class LatentGeometryLayer(nn.Module):
    """Shared P/Q plus `m` square experts in latent space, `k` active per token.

    `reference=True` removes P/Q and puts the experts at native width (the T2 `X`
    form), so one class produces both sides of the comparison. Dispatch is
    capacity-free and deterministic: every `(token, slot)` output is written once
    and the slots are summed in a fixed order.
    """

    def __init__(self, spec: LatentSpec, seed: int | None = None):
        super().__init__()
        self.spec = spec
        d_in = spec.d_model if spec.reference else spec.d_latent
        self.d_in = d_in
        self.project = (nn.Linear(spec.d_model, spec.d_latent, bias=False)
                        if spec.latent else None)
        self.unproject = (nn.Linear(spec.d_latent, spec.d_model, bias=False)
                          if spec.latent else None)
        self.gate = nn.Parameter(torch.empty(spec.experts, spec.expert_width, d_in))
        self.up = nn.Parameter(torch.empty(spec.experts, spec.expert_width, d_in))
        self.down = nn.Parameter(torch.zeros(spec.experts, d_in, spec.expert_width))
        self.router = nn.Linear(spec.d_model, spec.experts, bias=False)
        generator = torch.Generator(device="cpu")

        def draw(offset: int) -> torch.Generator:
            generator.manual_seed((0 if seed is None else int(seed)) + offset)
            return generator

        with torch.no_grad():
            nn.init.normal_(self.gate, std=0.02, generator=draw(1))
            nn.init.normal_(self.up, std=0.02, generator=draw(2))
            nn.init.normal_(self.router.weight, std=0.02, generator=draw(3))
            if self.project is not None:
                nn.init.normal_(self.project.weight, std=0.02, generator=draw(4))
                nn.init.normal_(self.unproject.weight, std=0.02, generator=draw(5))
        # Measurement overrides (deployed-budget utility labelling). They are
        # plain attributes, not buffers, so they never enter a checkpoint.
        self.forced_set: torch.Tensor | None = None
        self.exclude: torch.Tensor | int | None = None

    # -- accounting ----------------------------------------------------------
    def active_elements(self) -> int:
        return self.spec.active_per_layer

    def parameter_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    # -- forward -------------------------------------------------------------
    def _expert(self, latent: torch.Tensor, index: int) -> torch.Tensor:
        gate = F.linear(latent, self.gate[index])
        up = F.linear(latent, self.up[index])
        return F.linear(F.silu(gate) * up, self.down[index])

    def forward(self, hidden: torch.Tensor, forced: torch.Tensor | None = None,
                exclude: torch.Tensor | int | None = None) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        latent = self.project(flat) if self.project is not None else flat
        if forced is None:
            forced = self.forced_set
        if exclude is None:
            exclude = self.exclude
        if forced is None:
            logits = self.router(flat)
            index = torch.topk(logits, self.spec.k, dim=-1).indices
        else:
            index = forced
        tokens = torch.arange(flat.shape[0], device=flat.device)
        token_ids = tokens.repeat_interleave(index.shape[1])
        expert_ids = index.reshape(-1)
        slot_ids = torch.arange(index.shape[1], device=flat.device) \
            .repeat(flat.shape[0])
        keep = torch.ones_like(expert_ids, dtype=torch.bool)
        if exclude is not None:
            if torch.is_tensor(exclude):
                keep = expert_ids != exclude.to(expert_ids.device)[token_ids]
            else:
                keep = expert_ids != int(exclude)
        token_ids, expert_ids, slot_ids = (token_ids[keep], expert_ids[keep],
                                           slot_ids[keep])
        order = torch.argsort(expert_ids, stable=True)
        token_ids, expert_ids, slot_ids = (token_ids[order], expert_ids[order],
                                           slot_ids[order])
        pieces = []
        boundaries = torch.nonzero(expert_ids[1:] != expert_ids[:-1]).reshape(-1) + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.long), boundaries]).tolist()
        ends = starts[1:] + [int(expert_ids.numel())]
        for start, end in zip(starts, ends):
            if start == end:
                continue
            expert = int(expert_ids[start])
            rows = token_ids[start:end]
            output = self._expert(latent.index_select(0, rows), expert)
            pieces.append((rows, slot_ids[start:end], output))
        slots = torch.zeros(flat.shape[0], index.shape[1], latent.shape[-1],
                            dtype=latent.dtype, device=latent.device)
        for rows, slot, output in pieces:
            slots[rows, slot] = output
        combined = slots.sum(dim=1)
        result = self.unproject(combined) if self.unproject is not None else combined
        return result.reshape(shape)


class _Composed(nn.Module):
    def __init__(self, base: nn.Module, extra: nn.Module):
        super().__init__()
        self.base = base
        self.extra = extra

    def forward(self, hidden_states):
        output = self.base(hidden_states) + self.extra(hidden_states)
        return output


class LatentArm:
    """The training/eval surface `baseline.py` and `--train` share."""

    def __init__(self, model, spec: LatentSpec, layers: list[nn.Module]):
        self.model = model
        self.spec = spec
        self.layers = layers
        self.arm = f"T3-L{spec.d_latent}"
        self.kind = "latent_specialists"
        self.profile = "t3"
        self.router_modules = [layer.router for layer in layers]
        self.extra_modules = layers
        self.trainable_names = [name for name, parameter in model.named_parameters()
                                if parameter.requires_grad]

    def trainable_parameters(self):
        return [parameter for parameter in self.model.parameters()
                if parameter.requires_grad]

    def trainable_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def set_mask_order(self, _value: str) -> None:
        pass

    def mask_order(self) -> str:
        return "n/a (no routed-overflow buffer)"

    def banks(self) -> list:
        return []

    def memories(self) -> list:
        return []

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.model(input_ids=input_ids, use_cache=False, **kwargs)


def build_latent_arm(base_model, spec: LatentSpec, seed: int,
                     admission: dict | None = None) -> LatentArm:
    """Freeze the core and add one latent (or reference) expert layer per block."""
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    layers = []
    for index, layer in enumerate(base_model.model.layers):
        geometry = LatentGeometryLayer(spec, seed + 7000 + index)
        layer.mlp = _Composed(layer.mlp, geometry)
        layers.append(geometry)
    arm = LatentArm(base_model, spec, layers)
    arm.admission = admission or {}
    for parameter in arm.trainable_parameters():
        parameter.requires_grad_(True)
    arm.trainable_names = [name for name, parameter in base_model.named_parameters()
                           if parameter.requires_grad]
    return arm


# ---------------------------------------------------------------------------
# measurements


def expert_jacobian_rank(layer: LatentGeometryLayer, latent: torch.Tensor,
                         expert: int, tolerance: float = 1e-4) -> dict:
    """Numerical rank of one expert's local map, `L -> L`."""
    point = latent.detach().reshape(-1)[:layer.d_in].clone()

    def apply(vector):
        return layer._expert(vector.reshape(1, -1), expert).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(apply, point, vectorize=True)
    singular = torch.linalg.svdvals(jacobian.detach().float())
    scale = float(singular[0]) if singular.numel() and float(singular[0]) > 0 else 1.0
    rank = int((singular > tolerance * scale).sum())
    energy = float((singular ** 2).sum())
    return {"expert": int(expert),
            "rank_at_tolerance": rank, "tolerance": tolerance,
            "bound": layer.spec.expert_rank_bound,
            "effective_rank": (float(singular.sum()) ** 2 / energy) if energy else 0.0}


def jacobian_rank_proxy(layer: LatentGeometryLayer, x: torch.Tensor,
                        tolerance: float = 1e-4) -> dict:
    """Numerical singular spectrum of the layer's local map at one token.

    The published bound is `min(k * expert_input_width, d_model)`. This reports
    the *measured* rank at a relative tolerance and the participation ratio
    `(sum sigma)^2 / sum sigma^2`, which is less brittle than a rank at a
    threshold. The probe is a real Jacobian, not an analytic proxy.
    """
    x = x.detach().reshape(1, 1, -1).clone()

    def flat(vector):
        output = layer(vector.reshape(1, 1, -1))
        return output.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(flat, x.reshape(-1),
                                                  vectorize=True)
    singular = torch.linalg.svdvals(jacobian.detach().float())
    scale = float(singular[0]) if singular.numel() and float(singular[0]) > 0 else 1.0
    rank = int((singular > tolerance * scale).sum())
    total = float(singular.sum())
    energy = float((singular ** 2).sum())
    return {"d_model": int(jacobian.shape[0]),
            "rank_at_tolerance": rank, "tolerance": tolerance,
            "bound": layer.spec.jacobian_rank_bound,
            "effective_rank": (total ** 2 / energy) if energy > 0 else 0.0,
            "singular_max": float(singular[0]) if singular.numel() else 0.0,
            "singular_min_nonzero": float(singular[singular > tolerance * scale].min())
            if rank else 0.0,
            "top_singular": [float(value) for value in singular[:8]]}


def utility_diversity(utilities: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    """Behavioural diversity of per-expert utilities under the deployed read.

    `utilities` is `[tokens, experts]`; `mask` marks which (token, expert) pairs
    were actually deployed (a `False` entry is a missing observation, not a zero).
    Returns mean pairwise Pearson correlation across experts with overlapping
    support, the mean over the *mean* utility vector, a JS divergence from the
    uniform share, the effective number of experts and the top-1 share.
    """
    utilities = utilities.detach().float()
    if utilities.dim() != 2:
        raise ValueError("utilities must be [tokens, experts]")
    tokens, experts = utilities.shape
    mask = (torch.ones_like(utilities, dtype=torch.bool) if mask is None
            else mask.detach().to(torch.bool))
    correlations, absolute = [], []
    for first in range(experts):
        for second in range(first + 1, experts):
            support = mask[:, first] & mask[:, second]
            if int(support.sum()) < 3:
                continue
            left = utilities[support, first]
            right = utilities[support, second]
            if float(left.std()) == 0.0 or float(right.std()) == 0.0:
                continue
            value = float(torch.corrcoef(torch.stack([left, right]))[0, 1])
            correlations.append(value)
            absolute.append(abs(value))
    mean_utility = utilities.masked_fill(~mask, float("nan")).nanmean(dim=0)
    mean_utility = torch.nan_to_num(mean_utility, nan=0.0)
    shares = torch.softmax(mean_utility, dim=-1)
    uniform = torch.full_like(shares, 1.0 / experts)
    midpoint = 0.5 * (shares + uniform)
    js = 0.5 * (torch.nn.functional.kl_div(shares.clamp_min(1e-12).log(), uniform,
                                          reduction="sum")
                + torch.nn.functional.kl_div(uniform.log(), midpoint,
                                             reduction="sum")
                + torch.nn.functional.kl_div(shares.clamp_min(1e-12).log(), midpoint,
                                             reduction="sum")) * 0.5
    entropy = float(-(shares * shares.clamp_min(1e-12).log()).sum())
    top1 = float(shares.max())
    return {"tokens": tokens, "experts": experts,
            "mean_pairwise_correlation": (sum(correlations) / len(correlations)
                                          if correlations else None),
            "mean_absolute_pairwise_correlation": (sum(absolute) / len(absolute)
                                                   if absolute else None),
            "pairs": len(correlations),
            "js_from_uniform": float(js),
            "effective_experts": math.exp(entropy),
            "top1_share": top1}


def decode_overhead(spec: LatentSpec, reference: LatentSpec | None = None) -> dict:
    """Per-token extra FLOPs and bytes, decomposed, against the reference form."""
    reference = reference or reference_spec(spec.d_model, layers=spec.layers)

    def flops(layer_spec):
        return 2 * (layer_spec.k * layer_spec.expert_elements
                    + layer_spec.shared_elements + layer_spec.router_elements)

    def reads(layer_spec):
        return layer_spec.k * layer_spec.expert_elements \
            + layer_spec.shared_elements + layer_spec.router_elements

    return {
        "per_layer_flops": flops(spec),
        "per_token_flops_all_layers": spec.layers * flops(spec),
        "reference_per_layer_flops": flops(reference),
        "flops_ratio_vs_reference": flops(spec) / flops(reference),
        "per_layer_read_bytes_bf16": 2 * reads(spec),
        "reference_per_layer_read_bytes_bf16": 2 * reads(reference),
        "bytes_ratio_vs_reference": reads(spec) / reads(reference),
        "experts_per_token": spec.k,
        "reference_experts_per_token": reference.k,
        "routing_overhead_fraction": spec.router_elements
        / max(1, spec.active_per_layer),
        "shared_projection_fraction": spec.shared_elements
        / max(1, spec.active_per_layer),
        "basis": "modelled distinct-identity reads at bf16; not measured traffic",
    }


def vram_model(spec: LatentSpec, tokens: int = 1024, dtype_bytes: int = 2) -> dict:
    """Modelled resident and per-token bytes; labelled `modelled` everywhere."""
    return {
        "basis": "modelled",
        "dtype_bytes": dtype_bytes,
        "trainable_bytes": dtype_bytes * spec.active_total,
        "accessible_bytes": dtype_bytes * spec.accessible_total,
        "per_token_scratch_bytes": dtype_bytes * tokens * (
            spec.d_latent + spec.k * spec.expert_width + spec.d_model),
        "parameters_active": spec.active_total,
        "parameters_accessible": spec.accessible_total,
    }


def measure_utility_diversity(model, spec: LatentSpec, input_ids: torch.Tensor,
                              loss_fn, layers: list[LatentGeometryLayer] | None = None,
                              max_layers: int | None = None) -> dict:
    """Deployed-budget leave-one-expert-out utilities on a calibration batch.

    For each latent layer the router's top-`k` set is executed with the rest of
    the model unchanged; the utility of a selected expert is
    `L(S \\ {e}) - L(S)` per token. Only the deployed set is labelled, which is
    P12.17's reference-set idea applied to T3's geometry, and it is what makes
    "utility diversity" a behavioural quantity. The cost is `1 + k` forwards per
    layer and it is reported in the receipt.
    """
    if loss_fn is None:
        raise ValueError("measure_utility_diversity needs a loss_fn(model, ids)")
    layers = layers or [module for module in model.modules()
                        if isinstance(module, LatentGeometryLayer)]
    if max_layers is not None:
        layers = layers[:max_layers]
    with torch.no_grad():
        baseline = loss_fn(model, input_ids).detach().reshape(-1).float()
    tokens = int(baseline.numel())
    dense = torch.zeros(tokens, spec.experts)
    mask = torch.zeros(tokens, spec.experts, dtype=torch.bool)
    per_layer = []
    passes = 1
    try:
        for layer in layers:
            captured = {}

            def pre_hook(_module, args, captured=captured):
                captured["input"] = args[0].detach()

            handle = layer.register_forward_pre_hook(pre_hook)
            try:
                with torch.no_grad():
                    loss_fn(model, input_ids)
            finally:
                handle.remove()
            if "input" not in captured:
                raise RuntimeError("layer never ran during the calibration forward")
            flat = captured["input"].reshape(-1, spec.d_model)
            with torch.no_grad():
                selected = torch.topk(layer.router(flat), spec.k, dim=-1).indices
            layer.forced_set = selected
            with torch.no_grad():
                full = loss_fn(model, input_ids).detach().reshape(-1).float()
            utilities = torch.zeros_like(selected, dtype=torch.float32)
            for slot in range(spec.k):
                layer.exclude = selected[:, slot]
                with torch.no_grad():
                    ablated = loss_fn(model, input_ids).detach().reshape(-1).float()
                passes += 1
                utilities[:, slot] = ablated - full
            layer.exclude = None
            layer.forced_set = None
            for slot in range(spec.k):
                dense[torch.arange(tokens), selected[:, slot]] = utilities[:, slot]
                mask[torch.arange(tokens), selected[:, slot]] = True
            per_layer.append({"layer_index": len(per_layer),
                              "mean_utility": float(utilities.mean()),
                              "mean_abs_utility": float(utilities.abs().mean()),
                              "full_equals_baseline_max_abs": float(
                                  (full - baseline).abs().max())})
    finally:
        for layer in layers:
            layer.forced_set = None
            layer.exclude = None
    return {"method": "deployed-budget leave-one-selected-expert-out",
            "layers": len(layers), "passes": passes, "tokens": tokens,
            "k": spec.k, "experts": spec.experts,
            "per_layer": per_layer,
            "diversity": utility_diversity(dense, mask),
            "utilities": dense, "mask": mask}


# ---------------------------------------------------------------------------
# admission config: T2 X's core slices, latent expert slices


def latent_admission_config(spec: LatentSpec, size: str = "0.6b") -> dict:
    """Union declaration for a latent arm, reusing the T2 `X` core slices.

    The Rust ledger remains the source of truth; this only declares the slices.
    The shared P/Q are execution slices (they run for every token); the router is
    a selector; the active experts are execution slices, `k` per layer.
    """
    import arms

    base = arms.build_config(arms.T2_ARMS["X"])
    execution = [use for use in base["execution_uses"]
                 if ".specialists." not in use["identity"]["tensor"]]
    selector = [use for use in base["selector_uses"]
                if ".specialists." not in use["identity"]["tensor"]]
    execution_flops = dict(base["execution_flops"])
    selector_flops = dict(base.get("selector_flops", {}))

    removed = [use for use in base["execution_uses"]
               if ".specialists." in use["identity"]["tensor"]]
    removed_selector = [use for use in base["selector_uses"]
                        if ".specialists." in use["identity"]["tensor"]]
    execution_flops["matmul_flops"] -= sum(2 * use["reads"] for use in removed)
    selector_flops["matmul_flops"] = selector_flops.get("matmul_flops", 0) \
        - sum(2 * use["reads"] for use in removed_selector)

    def use(tensor: str, elements: int) -> dict:
        if elements <= 0:
            raise ValueError(f"non-positive element count for {tensor}")
        return {"identity": {"tensor": tensor,
                             "elements": {"start": 0, "end": elements}},
                "reads": elements, "bytes": elements * 2, "tier": "gpu"}

    added_execution = added_selector = 0
    for layer in range(spec.layers):
        if spec.latent:
            for name, elements in (("project", spec.d_model * spec.d_latent),
                                   ("unproject", spec.d_latent * spec.d_model)):
                execution.append(use(f"model.layers.{layer}.latent.{name}.weight",
                                     elements))
                added_execution += 2 * elements
        for slot in range(spec.k):
            for suffix in ("gate", "up", "down"):
                elements = spec.expert_width * spec.d_latent
                execution.append(use(
                    f"model.layers.{layer}.latent.experts.{suffix}.slot{slot}",
                    elements))
                added_execution += 2 * elements
        selector.append(use(f"model.layers.{layer}.latent.router.weight",
                            spec.d_model * spec.experts))
        added_selector += 2 * spec.d_model * spec.experts
    execution_flops["matmul_flops"] += added_execution
    selector_flops["matmul_flops"] = selector_flops.get("matmul_flops", 0) \
        + added_selector

    total_accessible = None
    # The X accessible total = core + m * 3 * w * d + d * m per layer; remove the
    # specialist bank and router, then add this spec's accessible set.
    x_arm = arms.T2_ARMS["X"]
    per_layer_removed = (x_arm.specialists * 3 * x_arm.specialist_width
                         * spec.d_model + spec.d_model * x_arm.specialists)
    total_accessible = (int(base["total_accessible"])
                        - spec.layers * per_layer_removed
                        + spec.accessible_total)
    return {
        "schema": "selvedge/admission-config/v1",
        "arm": f"T3-L{spec.d_latent}",
        "budget": {"active_unique_limit": CEILING, "mode": "compliance"},
        "total_accessible": total_accessible,
        "selector_uses": selector,
        "execution_uses": execution,
        "selector_flops": selector_flops,
        "execution_flops": execution_flops,
        "notes": [
            f"T3 latent geometry: L={spec.d_latent}, square experts w={spec.expert_width}, "
            f"m={spec.experts}, k={spec.k}, shared P/Q per layer",
            "the core slices are T2 X's, so the comparison is the expert geometry "
            "alone",
            f"active per layer {spec.active_per_layer}; the ledger's union is the "
            "authoritative count",
        ],
    }


def python_mirror(config: dict) -> dict:
    """Union arithmetic; the Rust ledger is the source of truth."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for use in config["selector_uses"] + config["execution_uses"]:
        range_start = int(use["identity"]["elements"]["start"])
        range_end = int(use["identity"]["elements"]["end"])
        ranges.setdefault(use["identity"]["tensor"], []).append((range_start, range_end))
    total = 0
    for spans in ranges.values():
        spans.sort()
        cursor = None
        for start, end in spans:
            if cursor is None or start > cursor:
                total += end - start
                cursor = end
            elif end > cursor:
                total += end - cursor
                cursor = end
    return {"active_unique_params": total,
            "within_budget": total <= int(config["budget"]["active_unique_limit"])}


# ---------------------------------------------------------------------------
# GPU entry points


def _require_gpu(purpose: str) -> None:
    if not torch.cuda.is_available():
        raise SystemExit(f"{purpose} requires the GPU court (--gpu)")


def _base_model(hf_root: str, device, size: str = "0.6b"):
    import transformers

    revision = {"0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
                "1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}[size]
    directory = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B"}[size]
    path = os.path.join(hf_root, "models", "Qwen", directory, revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return model


def train_latent(args) -> int:
    """Train one latent arm with the T2 data order, budget and optimiser."""
    from loop import (TargetReader, distillation_loss, load_checkpoint, lr_at,
                      save_checkpoint, set_seed, utc_now)
    from modeling import base_logits_for, zero_point_report

    _require_gpu("T3 training")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    spec = LatentSpec.load(args.spec)
    device = torch.device("cuda")
    set_seed(args.seed)
    from modeling import load_admission

    if not (args.config and args.admission):
        raise SystemExit("--config and --admission are required: T3 runs only an "
                         "admitted config (the Rust ledger is the source of truth)")
    config, admission = load_admission(args.config, args.admission)
    if config["arm"] != f"T3-L{spec.d_latent}":
        raise SystemExit(f"config arm {config['arm']} != T3-L{spec.d_latent}")
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    out_dir = os.path.join(REPO, args.out)
    index = json.load(open(os.path.join(REPO, args.artifact_dir,
                                        "train.index.json"), encoding="utf-8"))
    stream = torch.from_numpy  # only used for clarity below
    import numpy as np

    stream = np.memmap(os.path.join(REPO, args.artifact_dir, "train.bin"),
                       dtype="<u4", mode="r", shape=(index["tokens"],))
    targets = TargetReader(os.path.join(REPO, args.artifact_dir))
    base = _base_model(args.hf_root, device, args.model_size)
    arm = build_latent_arm(base, spec, args.seed, admission)
    probe_ids = probe_base = None
    if args.zero_point:
        probe_ids = torch.from_numpy(np.asarray(stream[0:2 * 1024],
                                                dtype="int64").reshape(2, 1024)).to(device)
        probe_base = base_logits_for(base, probe_ids)
    if args.zero_point:
        report = zero_point_report(arm, probe_base, probe_ids)
        write_json_atomic(os.path.join(out_dir, "zero-point.json"), report)
        if not report["ok"]:
            raise SystemExit("latent zero point is not exactly the frozen core")
        del probe_base, probe_ids
        torch.cuda.empty_cache()
    arm.model.train()
    tokens_per_step = args.micro_batch * 1024 * args.accum
    if tokens_per_step != 64 * 1024:
        raise SystemExit("effective batch must stay 65,536 tokens (T2.7)")
    parameters = arm.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.0)
    start_step = 0
    if args.resume:
        state = load_checkpoint(args.resume, "cpu")
        arm.model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        set_seed(args.seed + start_step)
    log = open(os.path.join(out_dir, "train.jsonl"), "a", encoding="utf-8")
    tokens_done = start_step * tokens_per_step
    peak = 0.0
    train_seconds = 0.0
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        started = time.time()
        for micro in range(args.accum):
            offset = step * tokens_per_step + micro * args.micro_batch * 1024
            block = args.micro_batch * 1024
            splits = block // args.micro_positions
            for sub in range(splits):
                sub_offset = offset + sub * args.micro_positions
                ids = torch.from_numpy(np.asarray(
                    stream[sub_offset:sub_offset + args.micro_positions],
                    dtype="int64").reshape(1, args.micro_positions)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = arm.model(input_ids=ids, use_cache=False).logits
                flat_logits = logits.reshape(-1, logits.shape[-1])
                values, target_ids, lse = targets.read(sub_offset,
                                                       flat_logits.shape[0], device)
                loss = distillation_loss(flat_logits, values, target_ids, lse)
                (loss / args.accum / splits).backward()
                step_loss += float(loss.detach()) / args.accum / splits
                del logits, flat_logits, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        tokens_done += tokens_per_step
        elapsed = time.time() - started
        train_seconds += elapsed
        peak = max(peak, torch.cuda.max_memory_allocated() / (1024 ** 3))
        record = {"step": step + 1, "loss": step_loss, "lr": lr,
                  "grad_norm": float(grad_norm), "tokens": tokens_done,
                  "tokens_per_second": tokens_per_step / max(elapsed, 1e-9),
                  "vram_peak_gib": peak, "step_seconds": elapsed,
                  "trainable_elements": sum(p.numel() for p in parameters),
                  "spec_active_per_layer": spec.active_per_layer}
        log.write(json.dumps(record, sort_keys=True) + "\n")
        log.flush()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"train: step {step + 1}/{args.steps} loss={step_loss:.4f} "
                  f"tok/s={record['tokens_per_second']:.0f} vram={peak:.2f}GiB",
                  flush=True)
        if (step + 1) % args.checkpoint_every == 0 or (step + 1) == args.steps:
            state = {"step": step + 1,
                     "model": {name: parameter.detach().cpu()
                               for name, parameter in arm.model.named_parameters()
                               if name in arm.trainable_names},
                     "optimizer": optimizer.state_dict(), "arm": arm.arm,
                     "seed": args.seed, "tokens": tokens_done}
            path = os.path.join(out_dir, f"step-{step + 1:06d}.pt")
            save_checkpoint(path, state)
            write_json_atomic(os.path.join(out_dir, "latest.json"),
                              {"step": step + 1, "checkpoint": path,
                               "tokens": tokens_done})
    log.close()
    summary = {"schema": SCHEMA_RUN, "arm": arm.arm, "spec": spec.report(),
               "steps": args.steps, "tokens": tokens_done,
               "tokens_per_second": (tokens_done - start_step * tokens_per_step)
               / max(train_seconds, 1e-9),
               "vram_peak_gib": peak,
               "trainable_elements": sum(p.numel() for p in parameters),
               "admitted_active_unique": admission["active_unique_params"],
               "total_accessible": config["total_accessible"],
               "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
               "finished_utc": utc_now(), "log": "train.jsonl",
               "deterministic": bool(args.deterministic)}
    write_json_atomic(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def report_latent(args) -> int:
    """Measure rank, diversity, VRAM and decode time for a trained latent arm."""
    import numpy as np

    from loop import (TargetReader, distillation_loss_per_position, utc_now)

    _require_gpu("T3 report")
    spec = LatentSpec.load(args.spec)
    device = torch.device("cuda")
    base = _base_model(args.hf_root, device, args.model_size)
    arm = build_latent_arm(base, spec, args.seed)
    state = torch.load(os.path.join(REPO, args.checkpoint), map_location="cpu",
                       weights_only=False)
    arm.model.load_state_dict(state["model"], strict=False)
    arm.model.to(device).eval()

    probe = torch.randint(0, base.config.vocab_size, (1, 64),
                          generator=torch.Generator().manual_seed(args.seed)).to(device)
    # The published quantity is the *per-expert* local rank. On the real model the
    # full layer Jacobian is not measured: the dispatch's Python control flow is
    # not vmappable, and a row-wise 1024x1024 Jacobian is 1024 backward passes.
    # The pre-registered sample is therefore `--rank-layers` layers x
    # `--rank-experts` experts per layer, with the layer bound reported
    # analytically from the spec. `--full-layer-jacobian` measures the whole
    # layer map when the caller can afford it (the CPU fixture does).
    with torch.no_grad():
        embedded = arm.model.get_input_embeddings()(probe).reshape(-1, spec.d_model)
    ranks, layer_ranks = [], []
    for index, layer in enumerate(arm.layers[:args.rank_layers]):
        hidden = embedded[index % embedded.shape[0]]
        with torch.no_grad():
            latent = (layer.project(hidden.reshape(1, -1)) if layer.project is not None
                      else hidden.reshape(1, -1))
        for expert in range(min(args.rank_experts, spec.experts)):
            ranks.append({"layer": index, **expert_jacobian_rank(layer, latent[0],
                                                                expert)})
        if args.full_layer_jacobian:
            layer_ranks.append({"layer": index,
                               **jacobian_rank_proxy(layer, hidden)})

    utilities = None
    if args.utilities:
        payload = torch.load(os.path.join(REPO, args.utilities),
                             map_location="cpu", weights_only=False)
        utilities = utility_diversity(payload["utilities"], payload.get("mask"))
    if args.measure_utilities:
        index = json.load(open(os.path.join(REPO, args.artifact_dir,
                                            "train.index.json"), encoding="utf-8"))
        stream = np.memmap(os.path.join(REPO, args.artifact_dir, "train.bin"),
                           dtype="<u4", mode="r", shape=(index["tokens"],))
        targets = TargetReader(os.path.join(REPO, args.artifact_dir))
        positions = min(args.calibration_positions, int(index["tokens"]))
        ids = torch.from_numpy(np.asarray(stream[0:positions], dtype="int64")
                               .reshape(1, positions)).to(device)
        cache = {}

        def loss_fn(model, _ids, cache=cache):
            if "values" not in cache:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=_ids, use_cache=False).logits
                flat = logits.reshape(-1, logits.shape[-1])
                values, target_ids, lse = targets.read(0, flat.shape[0], device)
                cache.update(values=values, target_ids=target_ids, lse=lse)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=_ids, use_cache=False).logits
            flat = logits.reshape(-1, logits.shape[-1])
            return distillation_loss_per_position(flat, cache["values"],
                                                  cache["target_ids"], cache["lse"])

        measured = measure_utility_diversity(arm.model, spec, ids, loss_fn)
        utilities = {key: value for key, value in measured.items()
                     if key not in ("utilities", "mask")}
        if args.utilities_out:
            path = os.path.join(REPO, args.utilities_out)
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            torch.save({"utilities": measured["utilities"],
                        "mask": measured["mask"], "spec": spec.report()}, path)
            utilities["artifact"] = args.utilities_out

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                arm.model(input_ids=probe, use_cache=False)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(args.timing_repeats):
            started = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                arm.model(input_ids=probe, use_cache=False)
            torch.cuda.synchronize()
            times.append(time.time() - started)
    times.sort()
    receipt = {
        "schema": SCHEMA_REPORT, "stamp": utc_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "arm": arm.arm, "spec": spec.report(),
        "checkpoint": args.checkpoint,
        "jacobian": {
            "method": "numerical per-expert Jacobian (SVD rank at a relative "
                      "tolerance plus participation ratio) on the pre-registered "
                      "sample, with the layer bound reported analytically",
            "tolerance": 1e-4,
            "sample": {"layers": min(args.rank_layers, spec.layers),
                       "experts_per_layer": min(args.rank_experts, spec.experts)},
            "per_expert": ranks,
            "mean_expert_rank": (sum(row["rank_at_tolerance"] for row in ranks)
                                 / len(ranks) if ranks else None),
            "expert_bound": spec.expert_rank_bound,
            "layer_bound": spec.jacobian_rank_bound,
            "layer_measured": layer_ranks,
            "note": "a shared projection caps the whole-layer rank at L; the "
                    "reference form is bounded by min(k*w, d_model)",
        },
        "utility_diversity": utilities,
        "decode": {
            "tokens": int(probe.shape[-1]),
            "median_seconds": times[len(times) // 2],
            "min_seconds": times[0],
            "repeats": args.timing_repeats,
            "device": torch.cuda.get_device_name(0),
            "modelled": decode_overhead(spec),
        },
        "vram_peak_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "vram_modelled": vram_model(spec, tokens=int(probe.shape[-1])),
        "active_union": python_mirror(latent_admission_config(spec)),
        "note": "quality is scored by oracle/eval/baseline.py --profile t3, not here",
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({"out": args.out,
                      "mean_expert_rank": receipt["jacobian"]["mean_expert_rank"],
                      "vram_peak_gib": receipt["vram_peak_gib"],
                      "median_seconds": receipt["decode"]["median_seconds"]},
                     indent=2, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="emit the arithmetic sweep (CPU, no model)")
    parser.add_argument("--emit-specs", default=None,
                        help="directory to write the four sweep specs into")
    parser.add_argument("--emit-config", default=None,
                        help="write the admission config for --spec")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--utilities", default=None)
    parser.add_argument("--utilities-out", default=None)
    parser.add_argument("--measure-utilities", action="store_true",
                        help="run the deployed-budget LOO utility measurement")
    parser.add_argument("--calibration-positions", type=int, default=1024)
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--micro-positions", type=int, default=512)
    parser.add_argument("--accum", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--zero-point", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sample-tokens", type=int, default=1)
    parser.add_argument("--rank-layers", type=int, default=4)
    parser.add_argument("--rank-experts", type=int, default=2)
    parser.add_argument("--full-layer-jacobian", action="store_true",
                        help="also measure the whole-layer Jacobian (expensive)")
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args(argv)

    if args.sweep:
        payload = sweep_report()
        if args.out:
            write_json_atomic(os.path.join(REPO, args.out), payload)
        if args.emit_specs:
            directory = os.path.join(REPO, args.emit_specs)
            os.makedirs(directory, exist_ok=True)
            for spec in sweep_specs():
                spec.save(os.path.join(args.emit_specs, f"L{spec.d_latent}.json"))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.emit_config:
        if not args.spec:
            raise SystemExit("--emit-config needs --spec")
        spec = LatentSpec.load(args.spec)
        write_json_atomic(os.path.join(REPO, args.emit_config),
                          latent_admission_config(spec))
        print(json.dumps({"config": args.emit_config,
                          "mirror": python_mirror(
                              latent_admission_config(spec))},
                         indent=2, sort_keys=True))
        return 0
    if args.train:
        return train_latent(args)
    if args.report:
        return report_latent(args)
    parser.error("one of --sweep, --emit-config, --train or --report is required")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
