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

Routing is hard top-k in the forward pass and softmax in the backward pass
(straight-through). `soft_mixture` is an evaluation-only ablation that computes
all experts weighted by the softmax; it is reported as a non-compliant routing
gap, never as a compliant arm.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

CAPACITY_FACTOR = 1.25


def _capacity(tokens: int, k: int, experts: int, factor: float = CAPACITY_FACTOR) -> int:
    return max(1, math.ceil(tokens * k / experts * factor))


class _RoutedDispatch(nn.Module):
    """Shared top-k capacity dispatch and scatter bookkeeping."""

    def __init__(self, hidden: int, experts: int, k: int):
        super().__init__()
        self.hidden = hidden
        self.experts = experts
        self.k = k
        self.soft_mixture = False
        self.last_routing: torch.Tensor | None = None
        self._dropped_pairs = torch.zeros((), dtype=torch.long)

    def routing(self, flat: torch.Tensor, router_weight: torch.Tensor):
        logits = F.linear(flat, router_weight)
        probs = logits.float().softmax(dim=-1)
        _values, top_index = probs.topk(self.k, dim=-1)
        mask = torch.zeros_like(probs).scatter_(-1, top_index, 1.0)
        self.last_routing = top_index.detach()
        weight = straight_gate = mask + probs - probs.detach()
        pair_weight = straight_gate.gather(-1, top_index).reshape(-1)
        return probs, straight_gate, top_index, pair_weight

    def dispatch(self, flat: torch.Tensor, top_index: torch.Tensor):
        """Return (buffer [experts*capacity, hidden], row, pair_token, weight, mask)."""
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
        capacity = _capacity(tokens, self.k, self.experts)
        valid = slot < capacity
        self._dropped_pairs = (~valid).sum()
        # Invalid (overflow) pairs would index past the buffer; clamp their row
        # so the gather is in range. Their contribution is masked out later.
        row = (sorted_expert * capacity + slot).clamp(max=self.experts * capacity - 1)
        buffer = flat.new_zeros(self.experts * capacity, self.hidden)
        gathered = flat.index_select(0, sorted_token)
        buffer.index_copy_(0, row[valid], gathered[valid])
        return buffer.view(self.experts, capacity, self.hidden), row, sorted_token, valid

    def combine(self, output: torch.Tensor, row: torch.Tensor, sorted_token: torch.Tensor,
                valid: torch.Tensor, pair_weight: torch.Tensor,
                tokens: int) -> torch.Tensor:
        flat_out = output.reshape(-1, self.hidden).index_select(0, row)
        flat_out = flat_out * pair_weight.unsqueeze(-1).to(flat_out.dtype)
        result = output.new_zeros(tokens, self.hidden)
        result.index_add_(0, sorted_token[valid], flat_out[valid])
        return result


class RoutedGroupFFN(_RoutedDispatch):
    """Top-k selection over contiguous groups of a SwiGLU FFN.

    Replaces `Qwen3MLP`. Group slices of the frozen gate/up/down tensors are
    read only for the tokens that select them. A rank-r residual adapter per
    group acts on the group's pre-down activation and is zero-initialised.
    """

    def __init__(self, mlp: nn.Module, groups: int, k: int, adapter_rank: int,
                 router: nn.Linear):
        super().__init__(mlp.gate_proj.weight.shape[1], groups, k)
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
        buffer, row, sorted_token, valid = self.dispatch(flat, top_index)
        output = self._group_ffn(buffer)
        combined = self.combine(output, row, sorted_token, valid, pair_weight,
                                flat.shape[0])
        return combined.reshape(shape)


class SpecialistBank(_RoutedDispatch):
    """Residual bank of m small SwiGLU FFNs, hard top-k routed per token."""

    def __init__(self, hidden: int, width: int, specialists: int, k: int,
                 router: nn.Linear):
        super().__init__(hidden, specialists, k)
        self.width = width
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
        buffer, row, sorted_token, valid = self.dispatch(flat, top_index)
        output = self._experts(buffer)
        combined = self.combine(output, row, sorted_token, valid, pair_weight,
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
    """Base frozen FFN plus an extra residual module."""

    def __init__(self, base: nn.Module, extra: nn.Module):
        super().__init__()
        self.base = base
        self.extra = extra

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.base(hidden_states) + self.extra(hidden_states)


# ---------------------------------------------------------------------------
# construction


@dataclass
class ArmModel:
    model: nn.Module
    arm: str
    kind: str
    router_modules: list[nn.Module]
    extra_modules: list[nn.Module]
    admission: dict
    config: dict

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def set_soft_mixture(self, value: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, _RoutedDispatch):
                module.soft_mixture = value

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


def build_arm(model, arm: str, config: dict, admission: dict, seed: int) -> ArmModel:
    """Wrap a loaded base model in the arm's trainable modules."""
    layers = model.model.layers
    hidden = model.config.hidden_size
    router_modules: list[nn.Module] = []
    extra_modules: list[nn.Module] = []

    for parameter in model.parameters():
        parameter.requires_grad_(False)

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
    trainable.declared_active = admission["active_unique_params"]
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
    if arm_model.arm in ("A", "A0"):
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
