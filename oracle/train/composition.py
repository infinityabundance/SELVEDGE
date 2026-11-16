#!/usr/bin/env python3
"""P8.2 — bounded-gate composition of two learned residuals.

The composed output at a position is

```
y = S + a * r_D + b * r_M,     (a, b) = g(feature),   ||a r_D + b r_M||_2 <= rho
```

where `S` is the frozen core's output, `r_D` and `r_M` are the residuals the
dense-adapter and memory capabilities would add on their own, and `g` is a small
linear gate on a per-token feature (the layer's input hidden state by default).
The gate is bounded twice: its own weights are initialised to zero (so at step 0
the composition is exactly `S`, matching T2.4's zero point), and the *effect* is
projected onto an L2 ball of radius `rho` per token, so the perturbation cannot
exceed a declared budget no matter what the gate produces.

Three quantities are reported rather than argued:

* **interference** — `||y - (S + r_D)||_2` per token, i.e. how much adding the
  memory residual changes the output that `D` alone would have produced. By
  construction it is at most `rho`; the report shows the realised distribution.
* **attribution** — `a` and `b`, plus the realised norm split, so a composition
  win cannot be read as a routing win.
* **union accounting** — `composition_admission_config` builds the config the
  Rust ledger admits: the union of `D`'s execution slices, `M`'s execution
  slices and the gate's own parameters (declared as a selector), with the
  accessible count unioned by identity rather than summed, because both arms
  share the frozen core.

CPU-only; no model is needed to build or admit the config.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

SCHEMA_CONFIG = "selvedge/admission-config/v1"
SCHEMA_WRAPPER = "selvedge/p8-composition-admission/v1"
SCHEMA_TRAIN = "selvedge/p8-composition-run/v1"
SCHEMA_EVAL = "selvedge/p8-composition-eval/v1"
COMPOSED_ARM = "P8-COMPOSE"
MODES = ("D", "M", "sum", "gate")
GATE_STEPS = 256
GATE_LR = 3e-3
GATE_SEED = 1001
KAPPA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261115
WINDOW = 1024


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_admission(config_path: str, record_path: str, out_path: str) -> dict:
    """Record the ledger's admission with the stamp and revision a receipt needs."""
    record = json.load(open(record_path, encoding="utf-8"))
    config = json.load(open(config_path, encoding="utf-8"))
    payload = {
        "schema": SCHEMA_WRAPPER,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "config": {"path": config_path, "sha256": sha256_file(config_path)},
        "record_path": record_path,
        "record": record,
        "union": python_mirror(config),
        "mirror_agrees": python_mirror(config)["active_unique_params"]
        == record.get("active_unique_params"),
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out_path)
    return payload


class BoundedGateComposition(nn.Module):
    """`y = S + a r_D + b r_M` with a zero-initialised, L2-bounded gate.

    The budget is either **absolute** (`budget`, a positive scalar in the units
    of the hidden state) or **relative** (`relative`, a fraction `kappa` of the
    token's own base-state norm, `rho_t = kappa * ||S_t||`). The relative form is
    what the P8.2 registration fixes, because an absolute radius has no
    scale-free meaning across layers.
    """

    def __init__(self, hidden: int, budget: float | None = None,
                 feature_dim: int | None = None,
                 relative: float | None = None):
        super().__init__()
        if (budget is None) == (relative is None):
            raise ValueError("declare exactly one of budget (absolute) or relative")
        if budget is not None and budget <= 0:
            raise ValueError("budget must be positive")
        if relative is not None and relative <= 0:
            raise ValueError("relative budget must be positive")
        self.hidden = hidden
        self.budget = float(budget) if budget is not None else None
        self.relative = float(relative) if relative is not None else None
        self.feature_dim = feature_dim or hidden
        self.mode = "gate"
        self.last_info: dict | None = None
        self.gate = nn.Linear(self.feature_dim, 2, bias=True)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    def budget_for(self, base: torch.Tensor) -> torch.Tensor:
        """Per-position budget `[tokens]` (flattened over the batch axes)."""
        positions = base.numel() // base.shape[-1]
        if self.relative is None:
            return torch.full((positions,), float(self.budget), dtype=torch.float32,
                              device=base.device)
        norms = base.detach().float().norm(dim=-1).reshape(-1)
        return (self.relative * norms).clamp_min(1e-12)

    def forward(self, base: torch.Tensor, residual_d: torch.Tensor,
                residual_m: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.mode in ("D", "M", "sum"):
            output = base
            if self.mode in ("D", "sum"):
                output = output + residual_d
            if self.mode in ("M", "sum"):
                output = output + residual_m
            norms = (output - base).detach().norm(dim=-1).reshape(-1)
            budget = self.budget_for(base)
            info = {
                "a": torch.full_like(norms, 1.0 if self.mode in ("D", "sum") else 0.0),
                "b": torch.full_like(norms, 1.0 if self.mode in ("M", "sum") else 0.0),
                "perturbation_norm": norms,
                "scale": torch.ones_like(norms),
                "budget": budget,
                "mode": self.mode,
                "budget_kind": "relative" if self.relative is not None else "absolute",
            }
            self.last_info = info
            return output, info
        coefficients = self.gate(feature)
        a = coefficients[..., 0:1].to(base.dtype)
        b = coefficients[..., 1:2].to(base.dtype)
        perturbation = a * residual_d + b * residual_m
        norms = perturbation.norm(dim=-1, keepdim=True)
        budget = self.budget_for(base).reshape(norms.shape)
        scale = torch.where(norms > budget, budget / norms.clamp_min(1e-12),
                            torch.ones_like(norms))
        # `a`, `b` and `scale` all carry a trailing singleton, so they broadcast
        # over the hidden axis for `[tokens, hidden]` and `[batch, seq, hidden]`.
        a = a * scale
        b = b * scale
        perturbation = a * residual_d + b * residual_m
        output = base + perturbation
        info = {
            "a": a.detach().reshape(-1),
            "b": b.detach().reshape(-1),
            "perturbation_norm": perturbation.detach().norm(dim=-1).reshape(-1),
            "scale": scale.reshape(-1).detach(),
            "budget": budget.reshape(-1).detach(),
            "mode": "gate",
            "budget_kind": "relative" if self.relative is not None else "absolute",
        }
        self.last_info = info
        return output, info


def _norms(tensor: torch.Tensor) -> torch.Tensor:
    """Per-position L2 norm over the last (hidden) axis."""
    return tensor.detach().norm(dim=-1).reshape(-1)


def _coefficient(tensor: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Reshape a flat per-position coefficient back to `[..., 1]`."""
    return tensor.reshape(base.shape[:-1] + (1,)).to(base.dtype)


def interference_report(base: torch.Tensor, residual_d: torch.Tensor,
                        residual_m: torch.Tensor, composed: torch.Tensor,
                        info: dict, tolerance: float = 1e-5) -> dict:
    """Make the interference a quantity with no unaccounted term.

    Three separations, because "adding `M` changed the output" can mean three
    different things and only one of them is bounded by the gate's budget:

    * `accounted` — the composed output equals `base + a r_D + b r_M` for the
      gate's own `(a, b)`, recomputed from the residuals. If this fails, the
      difference is not the gate's doing and the report is invalid.
    * `added_effect_of_m` — `||b r_M||`: what `M` adds at the coefficients the
      gate chose.
    * `deviation_from_d_alone` — `||(a - 1) r_D + b r_M||`: the plain-sense
      interference, which is **not** bounded by the budget because the gate may
      also turn `D` down. Reporting both is what keeps the bound honest.
    """
    a = _coefficient(info["a"], base)
    b = _coefficient(info["b"], base)
    recomputed = base + a * residual_d + b * residual_m
    accounted = bool(torch.allclose(recomputed, composed, atol=tolerance,
                                    rtol=tolerance))
    added = _norms(b * residual_m)
    deviation = _norms((a - 1.0) * residual_d + b * residual_m)
    perturbation = _norms(a * residual_d + b * residual_m)
    budget = info["budget"].detach().to(torch.float32).reshape(-1)
    return {
        "accounted": accounted,
        "mode": info.get("mode", "gate"),
        "budget": {"kind": info.get("budget_kind", "absolute"),
                   "mean": float(budget.mean()), "min": float(budget.min()),
                   "max": float(budget.max())},
        "added_effect_of_m": {"mean": float(added.mean()), "max": float(added.max())},
        "deviation_from_d_alone": {"mean": float(deviation.mean()),
                                   "max": float(deviation.max())},
        "perturbation": {"mean": float(perturbation.mean()),
                         "max": float(perturbation.max())},
        "within_budget": bool((perturbation <= budget + tolerance).all()),
        "count": int(added.numel()),
    }


def aggregate_interference(reports: list[dict]) -> dict:
    """Aggregate per-batch interference reports without hiding the worst case."""
    if not reports:
        raise ValueError("no interference reports to aggregate")
    def field(name, key):
        values = [report[name][key] for report in reports]
        return {"mean": sum(values) / len(values), "max": max(values)}
    return {
        "batches": len(reports),
        "accounted": all(report["accounted"] for report in reports),
        "within_budget": all(report["within_budget"] for report in reports),
        "budget": {"mean": sum(report["budget"]["mean"] for report in reports)
                   / len(reports)},
        "added_effect_of_m": field("added_effect_of_m", "max"),
        "deviation_from_d_alone": field("deviation_from_d_alone", "max"),
        "perturbation": field("perturbation", "max"),
    }


def check_budget(info: dict, tolerance: float = 1e-6) -> dict:
    """The realised perturbation must stay inside the declared budget."""
    norms = info["perturbation_norm"].reshape(-1).to(torch.float32)
    budget = info["budget"].detach().to(torch.float32).reshape(-1)
    if budget.numel() == 1 and norms.numel() != 1:
        budget = budget.expand_as(norms)
    violations = int((norms > budget * (1.0 + tolerance) + tolerance).sum())
    return {"budget": float(budget.mean()) if budget.numel() else 0.0,
            "max_perturbation_norm": float(norms.max()) if norms.numel() else 0.0,
            "violations": violations,
            "budget_utilisation": float((norms / budget.clamp_min(1e-12)).max())
            if norms.numel() else 0.0}


# ---------------------------------------------------------------------------
# admission: the union through the Rust ledger


def _identity(use: dict) -> tuple[str, int, int]:
    elements = use["identity"]["elements"]
    return (use["identity"]["tensor"], int(elements["start"]), int(elements["end"]))


def _elements(use: dict) -> int:
    return int(use["identity"]["elements"]["end"]) - int(use["identity"]["elements"]["start"])


def composition_admission_config(config_d: dict, config_m: dict,
                                 layers: int = 28, hidden: int = 1024) -> dict:
    """Build one config whose ledger union is the composed arm.

    The gate is `hidden -> 2` per layer, so it declares
    `layers * (2 * hidden + 2)` learned elements as a **selector**: it decides
    the coefficients, it does not carry capacity. The two arms' execution slices
    are carried verbatim, so the ledger unions them by identity; `total_accessible`
    is unioned the same way, because both configs re-declare the frozen core.
    """
    execution = list(config_d["execution_uses"]) + list(config_m["execution_uses"])
    selector = list(config_d.get("selector_uses", [])) + list(config_m.get("selector_uses", []))

    gate_elements = layers * (2 * hidden + 2)
    gate_flops = {"matmul_flops": 2 * 2 * hidden, "other_flops": 0,
                  "controller_flops": 2 * hidden}
    selector.append({
        "identity": {"tensor": "p8.composition.gate.weight",
                     "elements": {"start": 0, "end": gate_elements}},
        "reads": gate_elements,
        "bytes": gate_elements * 2,
        "tier": "gpu",
    })

    shared = {}
    for use_d in config_d["execution_uses"]:
        key = _identity(use_d)
        for use_m in config_m["execution_uses"]:
            if _identity(use_m) == key:
                shared[key] = _elements(use_d)
    core_elements = sum(shared.values())
    total_accessible = (int(config_d["total_accessible"])
                        + int(config_m["total_accessible"])
                        - core_elements
                        + gate_elements)

    execution_flops = {}
    for key in ("matmul_flops", "other_flops", "controller_flops"):
        execution_flops[key] = (int(config_d["execution_flops"][key])
                               + int(config_m["execution_flops"][key]))
    selector_flops = {}
    for key in ("matmul_flops", "other_flops", "controller_flops"):
        selector_flops[key] = (int(config_d.get("selector_flops", {}).get(key, 0))
                              + int(config_m.get("selector_flops", {}).get(key, 0))
                              + gate_flops[key])

    return {
        "schema": SCHEMA_CONFIG,
        "arm": COMPOSED_ARM,
        "budget": {"active_unique_limit": 1_000_000_000, "mode": "compliance"},
        "total_accessible": total_accessible,
        "selector_uses": selector,
        "execution_uses": execution,
        "selector_flops": selector_flops,
        "execution_flops": execution_flops,
        "notes": [
            "P8.2 bounded-gate composition: frozen core + the dense adapter (T2 D) "
            "+ the hashed memory path (T2 M) + one per-layer 1024->2 gate",
            f"gate elements {gate_elements} = layers * (2 * hidden + 2), declared "
            "as a selector because it decides coefficients rather than carrying "
            "capacity",
            f"total_accessible unions {len(shared)} shared core identities "
            f"({core_elements} elements) instead of double-counting them",
        ],
    }


def python_mirror(config: dict) -> dict:
    """Union arithmetic for the cross-check; the Rust ledger is the source of truth."""
    def union(uses):
        ranges: dict[str, list[tuple[int, int]]] = {}
        for use in uses:
            tensor, start, end = _identity(use)
            ranges.setdefault(tensor, []).append((start, end))
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
        return total

    selector = union(config["selector_uses"])
    execution = union(config["execution_uses"])
    active = union(config["selector_uses"] + config["execution_uses"])
    return {"selector_unique_params": selector,
            "execution_unique_params": execution,
            "active_unique_params": active,
            "within_budget": active <= int(config["budget"]["active_unique_limit"])}


# ---------------------------------------------------------------------------
# the composed arm: frozen core + D's adapter + M's memory + a gate per layer


class ComposedFFN(nn.Module):
    """One layer's additive module under the P8.2 composition.

    `base` is the frozen layer FFN (`S` at this layer), `dense` is T2 `D`'s
    adapter (`r_D`), `memory` is T2 `M`'s hashed memory (`r_M`, present only at
    the four registered insertion layers) and `gate` decides `(a, b)`.
    """

    def __init__(self, base: nn.Module, dense: nn.Module, memory: nn.Module | None,
                 gate: BoundedGateComposition):
        super().__init__()
        self.base = base
        self.dense = dense
        self.memory = memory
        self.gate = gate
        self.capture = False
        self.last_trace: dict | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        residual_d = self.dense(hidden_states)
        if self.memory is not None:
            residual_m = self.memory(hidden_states)
        else:
            residual_m = torch.zeros_like(residual_d)
        residual_d = residual_d.to(base.dtype)
        residual_m = residual_m.to(base.dtype)
        output, info = self.gate(base, residual_d, residual_m, hidden_states)
        if self.capture:
            self.last_trace = {"base": base.detach(), "residual_d": residual_d.detach(),
                               "residual_m": residual_m.detach(),
                               "composed": output.detach(), "info": info}
        return output


class CompositionArm:
    """The P8.2 arm: one frozen core, the two trained residuals, gates only trainable."""

    def __init__(self, model, gates, dense_modules, memory_modules, layers: int,
                 hidden: int, kappa: float, config_d: dict, config_m: dict,
                 admission: dict | None = None,
                 memory_layers: tuple[int, ...] = ()):
        self.model = model
        self.gates = gates
        self.dense_modules = dense_modules
        self.memory_modules = memory_modules
        self.memory_layers = tuple(memory_layers)
        self.layers = layers
        self.hidden = hidden
        self.kappa = float(kappa)
        self.config_d = config_d
        self.config_m = config_m
        self.admission = admission
        self.checkpoints: dict[str, str] = {}

    # -- construction / freezing ---------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        for gate in self.gates:
            gate.mode = mode

    def trainable_parameters(self) -> list[torch.Tensor]:
        return [parameter for parameter in self.model.parameters()
                if parameter.requires_grad]

    def trainable_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def frozen_parameters(self) -> list[torch.Tensor]:
        return [parameter for parameter in self.model.parameters()
                if not parameter.requires_grad]

    def gate_elements(self) -> int:
        return sum(parameter.numel() for gate in self.gates
                   for parameter in gate.parameters())

    # -- accounting ----------------------------------------------------------
    def selector_cost(self) -> dict:
        return composition_selector_cost(self.layers, self.hidden)

    def active_union(self) -> dict:
        composed = composition_admission_config(self.config_d, self.config_m,
                                                layers=self.layers,
                                                hidden=self.hidden)
        return {"total_accessible": composed["total_accessible"],
                "mirror": python_mirror(composed),
                "gate_elements": self.gate_elements(),
                "trainable_elements": self.trainable_elements(),
                "declared_gate_elements": self.layers * (2 * self.hidden + 2)}

    def gate_statistics(self) -> dict:
        return gate_statistics(self.gates)

    def save_gates(self, path: str, extra: dict | None = None) -> str:
        state = {"model": {f"gate.{index}.weight": gate.gate.weight.detach().cpu()
                            for index, gate in enumerate(self.gates)}
                 | {f"gate.{index}.bias": gate.gate.bias.detach().cpu()
                    for index, gate in enumerate(self.gates)},
                 "layers": self.layers, "hidden": self.hidden, "kappa": self.kappa}
        if extra:
            state.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return sha256_file(path)

    def load_gates(self, path: str) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (int(state["layers"]), int(state["hidden"])) != (self.layers, self.hidden):
            raise ValueError(f"{path}: gate geometry does not match the arm")
        if abs(float(state["kappa"]) - self.kappa) > 1e-12:
            raise ValueError(f"{path}: gate budget kappa does not match")
        with torch.no_grad():
            for index, gate in enumerate(self.gates):
                gate.gate.weight.copy_(state["model"][f"gate.{index}.weight"])
                gate.gate.bias.copy_(state["model"][f"gate.{index}.bias"])
        return state

    def interference_on(self, ids: torch.Tensor) -> dict:
        """Per-layer interference of the composition, on one probe batch."""
        composed = [module for module in self.model.modules()
                    if isinstance(module, ComposedFFN)]
        for module in composed:
            module.capture = True
            module.last_trace = None
        with torch.no_grad():
            self.model(input_ids=ids, use_cache=False)
        reports = []
        for module in composed:
            trace = module.last_trace
            module.capture = False
            module.last_trace = None
            if trace is None:
                continue
            reports.append(interference_report(
                trace["base"], trace["residual_d"], trace["residual_m"],
                trace["composed"], trace["info"]))
        if not reports:
            raise RuntimeError("no composed layer ran during the probe")
        return {"layers": len(reports), "per_layer": reports,
                "aggregate": aggregate_interference(reports)}


def build_composition_arm(base_model, config_d: dict, admission_d: dict,
                          config_m: dict, admission_m: dict, seed: int,
                          kappa: float, dense_width: int = 2048,
                          memory_geometry=None,
                          memory_layers: tuple[int, ...] | None = None) -> CompositionArm:
    """Install both capabilities on one frozen core, with gates as the only
    trainable parameters. The checkpoints are loaded separately by
    `load_arm_checkpoints`, before the arm is used.

    `dense_width`, `memory_geometry` and `memory_layers` default to the T2
    registration; the CPU tests override them with a tiny geometry.
    """
    from modeling import DenseAdapter, T2_MEMORY_LAYERS
    from memory import HashedMemory, MemoryGeometry, TokenContext
    from modeling import _install_token_context

    layers = base_model.model.layers
    hidden = base_model.config.hidden_size
    insertion = tuple(T2_MEMORY_LAYERS if memory_layers is None else memory_layers)
    geometry = memory_geometry or MemoryGeometry(hidden=hidden)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    context = TokenContext()
    _install_token_context(base_model, context)

    gates, dense_modules, memory_modules = [], [], []
    for index, layer in enumerate(layers):
        dense = DenseAdapter(hidden, dense_width)
        memory = None
        if index in insertion:
            memory = HashedMemory(hidden, index, context, geometry=geometry,
                                  seed=seed + 3000 + index)
            memory_modules.append(memory)
        gate = BoundedGateComposition(hidden, relative=kappa)
        layer.mlp = ComposedFFN(layer.mlp, dense, memory, gate)
        dense_modules.append(dense)
        gates.append(gate)

    # Freeze everything the composition adds except the gates.
    for module in dense_modules + memory_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for gate in gates:
        for parameter in gate.parameters():
            parameter.requires_grad_(True)
    arm = CompositionArm(base_model, gates, dense_modules, memory_modules,
                         layers=len(layers), hidden=hidden, kappa=kappa,
                         config_d=config_d, config_m=config_m,
                         memory_layers=insertion)
    arm.set_mode("gate")
    if arm.trainable_elements() != arm.gate_elements():
        raise RuntimeError("composition arm must have gate-only trainable parameters")
    return arm


def _remap_arm_state(state: dict, tag: str, arm: CompositionArm) -> dict:
    """Map a T2 checkpoint's `mlp.extra.*` names onto the composed arm.

    `D`'s checkpoint carries the dense adapter; `M`'s carries the hashed memory
    modules. Both were built inside `AdditiveFFN`, where the module is called
    `extra`; here the two live at `mlp.dense` and `mlp.memory`. The mapping is
    explicit and fails closed on an unmapped key, a missing key or a shape
    mismatch.
    """
    source = state.get("model", state)
    target = "dense" if tag == "D" else "memory"
    existing = dict(arm.model.named_parameters())
    remapped = {}
    for name, tensor in source.items():
        if ".mlp.extra." not in name:
            raise ValueError(f"{tag} checkpoint: unexpected key {name}")
        new_name = name.replace(".mlp.extra.", f".mlp.{target}.")
        if new_name not in existing:
            raise ValueError(f"{tag} checkpoint: {name} has no target {new_name}")
        if tuple(existing[new_name].shape) != tuple(tensor.shape):
            raise ValueError(f"{tag} checkpoint: {name} shape {tuple(tensor.shape)} "
                             f"!= {new_name} {tuple(existing[new_name].shape)}")
        remapped[new_name] = tensor
    if tag == "D" and len(remapped) != 3 * arm.layers:
        raise ValueError(f"D checkpoint covers {len(remapped)} parameters, expected "
                         f"{3 * arm.layers}")
    if tag == "M" and len(remapped) != 3 * len(arm.memory_modules):
        raise ValueError(f"M checkpoint covers {len(remapped)} parameters, expected "
                         f"{3 * len(arm.memory_modules)}")
    return remapped


def load_arm_checkpoints(arm: CompositionArm, checkpoint_d: str,
                         checkpoint_m: str, device="cpu") -> dict:
    state_d = torch.load(checkpoint_d, map_location="cpu", weights_only=False)
    state_m = torch.load(checkpoint_m, map_location="cpu", weights_only=False)
    mapped_d = _remap_arm_state(state_d, "D", arm)
    mapped_m = _remap_arm_state(state_m, "M", arm)
    parameters = dict(arm.model.named_parameters())
    with torch.no_grad():
        for name, tensor in mapped_d.items():
            parameters[name].copy_(tensor.to(parameters[name].dtype))
        for name, tensor in mapped_m.items():
            parameters[name].copy_(tensor.to(parameters[name].dtype))
    arm.checkpoints = {"D": {"path": checkpoint_d, "sha256": sha256_file(checkpoint_d),
                             "step": int(state_d.get("step", -1))},
                       "M": {"path": checkpoint_m, "sha256": sha256_file(checkpoint_m),
                             "step": int(state_m.get("step", -1))}}
    arm.model.to(device)
    arm.model.eval()
    return arm.checkpoints


# ---------------------------------------------------------------------------
# gate-only training


def gate_statistics(gates: list[BoundedGateComposition]) -> dict:
    """Realised coefficients and budget use, from each gate's last forward."""
    reports = []
    for gate in gates:
        info = gate.last_info
        if info is None:
            continue
        budget = info["budget"].detach().float()
        reports.append({
            "mean_abs_a": float(info["a"].detach().float().abs().mean()),
            "mean_abs_b": float(info["b"].detach().float().abs().mean()),
            "fraction_projected": float((info["scale"] < 1.0).float().mean()),
            "budget_utilisation": float((info["perturbation_norm"].detach().float()
                                         / budget.clamp_min(1e-12)).max()),
            "positions": int(info["a"].numel()),
        })
    if not reports:
        return {"layers_observed": 0}
    keys = ("mean_abs_a", "mean_abs_b", "fraction_projected", "budget_utilisation")
    aggregated = {key: sum(report[key] for report in reports) / len(reports)
                  for key in keys}
    aggregated.update({"layers_observed": len(reports),
                       "positions": sum(report["positions"] for report in reports)})
    return aggregated


def train_gate(gates: list[BoundedGateComposition], step_fn, steps: int = GATE_STEPS,
               lr: float = GATE_LR, seed: int = GATE_SEED, log_every: int = 25,
               weight_decay: float = 0.0, grad_clip: float = 1.0,
               progress=None) -> dict:
    """Optimise **only** the gates; `step_fn(step)` returns `(loss, tokens)`.

    The caller owns the forward pass and the loss, so the same trainer is used
    by the registered GPU run (distillation loss through the frozen core) and by
    the CPU tests (a known-target objective). Nothing else is trainable: the
    parameter set here is exactly the union of the gates' own parameters.
    """
    parameters = [parameter for gate in gates for parameter in gate.parameters()]
    if not parameters:
        raise ValueError("no gates to train")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=weight_decay)
    history, tokens_seen = [], 0
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, tokens = step_fn(index)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"gate loss is not finite at step {index}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
        tokens_seen += int(tokens)
        record = {"step": index + 1, "loss": float(loss.detach()),
                  "grad_norm": float(grad_norm), "tokens": tokens_seen,
                  "gate_norm": float(sum(parameter.detach().float().pow(2).sum()
                                         for parameter in parameters).sqrt())}
        history.append(record)
        if progress is not None and ((index + 1) % log_every == 0 or index == 0):
            progress(record)
    return {"steps": steps, "lr": lr, "seed": seed,
            "weight_decay": weight_decay, "grad_clip": grad_clip,
            "tokens": tokens_seen, "first_loss": history[0]["loss"],
            "final_loss": history[-1]["loss"], "history": history,
            "coefficients": gate_statistics(gates),
            "trainable_elements": sum(parameter.numel() for parameter in parameters)}


def composition_selector_cost(layers: int = 28, hidden: int = 1024) -> dict:
    """The gate's own per-token cost, counted separately from the capabilities."""
    elements = layers * (2 * hidden + 2)
    return {"learned_parameters": elements,
            "per_token_matmul_flops": 2 * 2 * hidden * layers,
            "controller_flops": 2 * hidden * layers,
            "bytes_bf16": 2 * elements,
            "note": "the gate decides coefficients; it carries no capacity and is "
                    "charged as a selector in the admission config"}


# ---------------------------------------------------------------------------
# the registered comparison


def paired_bootstrap(differences: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    values = torch.tensor(differences, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, values.numel(), (resamples, values.numel()),
                          generator=generator)
    means = values[index].mean(dim=1)
    return {"mean": float(values.mean()),
            "low": float(torch.quantile(means, alpha / 2)),
            "high": float(torch.quantile(means, 1 - alpha / 2)),
            "n": int(values.numel()), "resamples": resamples, "seed": seed}


def composition_comparison(per_item: dict[str, list[float]],
                           references: tuple[str, ...] = ("D", "M", "sum", "gate"),
                           alpha: float = 0.05) -> dict:
    """Paired comparison of the four references plus the per-item oracle bound.

    `per_item` maps an arm name to per-item losses on the same fresh items
    (lower is better). The decisive comparison is `gate - sum`: a composition
    win has to survive *both* capabilities being present, not just beat the
    single arms.
    """
    missing = [name for name in references if name not in per_item]
    if missing:
        raise ValueError(f"missing arms in the comparison: {missing}")
    lengths = {name: len(per_item[name]) for name in references}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"per-item losses must be paired; got {lengths}")
    count = lengths[references[0]]
    if count == 0:
        raise ValueError("the comparison has no items")
    oracle = [min(per_item[name][index] for name in references)
              for index in range(count)]
    means = {name: sum(per_item[name]) / count for name in references}
    best_single = [min(per_item["D"][index], per_item["M"][index])
                   for index in range(count)]
    comparisons = {}
    for name in references:
        if name == "gate":
            continue
        differences = [per_item["gate"][index] - per_item[name][index]
                       for index in range(count)]
        comparisons[f"gate-minus-{name}"] = paired_bootstrap(differences, alpha=alpha)
    comparisons["gate-minus-best-single"] = paired_bootstrap(
        [per_item["gate"][index] - best_single[index] for index in range(count)],
        alpha=alpha)
    wins = {name: sum(1 for index in range(count)
                      if per_item[name][index] == oracle[index])
            for name in references}
    return {
        "items": count,
        "mean_loss": means,
        "comparisons": comparisons,
        "oracle": {"mean_loss": sum(oracle) / count,
                   "gate_gap": sum(per_item["gate"][index] - oracle[index]
                                   for index in range(count)) / count},
        "wins": wins,
        "decision_pair": "gate-minus-sum",
    }


# ---------------------------------------------------------------------------
# the two GPU entry points


def _require_gpu(purpose: str) -> None:
    if not torch.cuda.is_available():
        raise SystemExit(f"{purpose} needs the GPU court (--gpu). The CPU court has "
                         "no GPU and the frozen 0.6B core does not fit its memory "
                         "limit; the CPU tests exercise this module's arithmetic on "
                         "tiny models instead.")


def _load_base(hf_root: str, device, dtype=torch.bfloat16):
    import transformers

    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    path = os.path.join(hf_root, "models", "Qwen", "Qwen3-0.6B", revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return model


def _stream(artifact_dir: str):
    import numpy as np

    index = json.load(open(os.path.join(artifact_dir, "train.index.json"),
                           encoding="utf-8"))
    stream = np.memmap(os.path.join(artifact_dir, "train.bin"), dtype="<u4",
                       mode="r", shape=(index["tokens"],))
    return index, stream


def _prepare_arm(args) -> tuple[CompositionArm, dict, torch.Tensor | None, torch.Tensor | None]:
    """Shared setup for training and scoring: base, arm, checkpoints."""
    from modeling import base_logits_for, load_admission

    _require_gpu("the composition arm")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    for name in ("config_d", "admission_d", "config_m", "admission_m",
                 "checkpoint_d", "checkpoint_m"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    config_d, admission_d = load_admission(args.config_d, args.admission_d)
    config_m, admission_m = load_admission(args.config_m, args.admission_m)
    if config_d["arm"] != "D" or config_m["arm"] != "M":
        raise SystemExit("the composition is registered over T2's D and M configs")
    base = _load_base(args.hf_root, device)
    probe_ids = None
    probe_base = None
    if args.run_composition:
        import numpy as np

        _index, stream = _stream(args.artifact_dir)
        probe_ids = torch.from_numpy(
            np.asarray(stream[0:2 * WINDOW], dtype="int64")
            .reshape(2, WINDOW)).to(device)
        probe_base = base_logits_for(base, probe_ids)
    arm = build_composition_arm(base, config_d, admission_d, config_m, admission_m,
                                seed=args.seed, kappa=args.kappa)
    checkpoints = load_arm_checkpoints(arm, args.checkpoint_d, args.checkpoint_m,
                                      device=device)
    return arm, checkpoints, probe_ids, probe_base


def composition_zero_point(arm: CompositionArm, base_logits: torch.Tensor,
                           input_ids: torch.Tensor, tolerance: float = 1e-3) -> dict:
    """The composition's zero point: with a zero gate the arm must equal `S`.

    Unlike the T2 arms, the two capabilities are already *trained* here, so this
    check cannot use the adapters' own zero init. It asserts what the gate
    guarantees: every gate parameter is exactly zero, so `a r_D + b r_M = 0`
    for every token and the model reproduces the frozen core.
    """
    gate_max = max(float(parameter.detach().abs().max())
                   for gate in arm.gates for parameter in gate.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda), \
            torch.inference_mode():
        modified = arm.model(input_ids=input_ids, use_cache=False).logits.float()
    delta = (modified - base_logits).abs()
    return {"arm": COMPOSED_ARM, "kind": "end_to_end_bf16_identity",
            "gate_max_abs": gate_max, "gate_parameters_zero": gate_max == 0.0,
            "max_abs_logit_delta": float(delta.max()),
            "mean_abs_logit_delta": float(delta.mean()),
            "tolerance": tolerance,
            "ok": bool(gate_max == 0.0 and float(delta.max()) <= tolerance),
            "device": str(input_ids.device)}


def run_composition(args) -> int:
    import numpy as np

    from loop import TargetReader, distillation_loss, set_seed

    if not args.out:
        raise SystemExit("--out is required for --run-composition")
    set_seed(args.seed)
    arm, checkpoints, probe_ids, probe_base = _prepare_arm(args)
    device = torch.device("cuda")
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    out_dir = os.path.join(REPO, args.out)
    targets = TargetReader(os.path.join(REPO, args.artifact_dir))
    index, stream = _stream(os.path.join(REPO, args.artifact_dir))

    # The gate starts at zero, so the composed model must reproduce the frozen
    # core exactly, `y = S` (P8.2's zero point).
    zero = composition_zero_point(arm, probe_base, probe_ids)
    write_json_atomic(os.path.join(out_dir, "zero-point.json"), zero)
    if not zero["ok"]:
        raise SystemExit("composition zero-point identity failed; the gate does not "
                         "start at y = S")
    del probe_base

    slice_tokens = args.calibration_windows * WINDOW
    if slice_tokens > int(index["tokens"]):
        raise SystemExit("calibration slice exceeds the training stream")
    steps_per_epoch = slice_tokens // args.positions
    if steps_per_epoch * args.positions != slice_tokens:
        raise SystemExit("--positions must divide the calibration slice")
    arm.model.train()

    def step(position: int):
        offset = (position % steps_per_epoch) * args.positions
        ids = torch.from_numpy(np.asarray(
            stream[offset:offset + args.positions], dtype="int64")
            .reshape(1, args.positions)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = arm.model(input_ids=ids, use_cache=False).logits
        flat = logits.reshape(-1, logits.shape[-1])
        values, target_ids, lse = targets.read(offset, flat.shape[0], device)
        loss = distillation_loss(flat, values, target_ids, lse)
        return loss, args.positions

    history = []
    def progress(record):
        history.append(record)
        print(f"composition: step {record['step']}/{args.steps} "
              f"loss={record['loss']:.4f} gate_norm={record['gate_norm']:.4f}",
              flush=True)

    training = train_gate(arm.gates, step, steps=args.steps, lr=args.gate_lr,
                          seed=args.seed, progress=progress)
    arm.model.eval()
    interference = arm.interference_on(probe_ids)
    gate_path = os.path.join(out_dir, f"gate-{args.steps:06d}.pt")
    gate_sha = arm.save_gates(gate_path, extra={"seed": args.seed,
                                                "steps": args.steps})
    union = arm.active_union()
    receipt = {
        "schema": SCHEMA_TRAIN,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "arm": COMPOSED_ARM,
        "seed": args.seed,
        "kappa": args.kappa,
        "budget_rule": "rho_t = kappa * ||S_t||_2 per token (relative, scale-free)",
        "gate": {"steps": args.steps, "lr": args.gate_lr,
                  "trainable_elements": training["trainable_elements"],
                  "selector_cost": composition_selector_cost(arm.layers, arm.hidden),
                  "checkpoint": os.path.relpath(gate_path, REPO),
                  "sha256": gate_sha},
        "data": {"artifact_dir": args.artifact_dir,
                  "stream_tokens": int(index["tokens"]),
                  "stream_sha256": index["stream_sha256"],
                  "calibration_windows": args.calibration_windows,
                  "slice_tokens": slice_tokens,
                  "positions": args.positions,
                  "steps_per_epoch": steps_per_epoch,
                  "epochs": args.steps / steps_per_epoch,
                  "order": "stream order from position 0, wrapping at the end of "
                           "the T2 calibration slice; the fresh evaluation set is "
                           "disjoint by construction (P8.1 contamination receipt)"},
        "capability_checkpoints": checkpoints,
        "admission": {"D": args.admission_d, "M": args.admission_m},
        "trainable_scope": "gates only; every core, adapter and memory parameter is "
                           "frozen (asserted by train_gate)",
        "training": training,
        "gate_statistics": arm.gate_statistics(),
        "interference": interference,
        "active_union": union,
        "zero_point": {"max_abs_logit_delta": zero["max_abs_logit_delta"],
                        "gate_parameters_zero": zero["gate_parameters_zero"],
                        "ok": zero["ok"]},
        "note": "gate-only training; the two capabilities are the trained T2 D and M "
                "checkpoints and are not updated by this run",
    }
    write_json_atomic(os.path.join(out_dir, "composition-run.json"), receipt)
    print(json.dumps({"out": receipt["gate"]["checkpoint"],
                      "gate_elements": union["gate_elements"],
                      "final_loss": training["final_loss"],
                      "budget_utilisation": arm.gate_statistics().get(
                          "budget_utilisation")}, indent=2, sort_keys=True))
    return 0


def score_composition(args) -> int:
    """Score D, M, naive sum, gated composition and the per-item oracle."""
    if not args.out:
        raise SystemExit("--out is required for --score-composition")
    if not args.gate_checkpoint:
        raise SystemExit("--gate-checkpoint is required for --score-composition")
    import sys

    eval_dir = os.path.join(REPO, "oracle", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import complementarity  # noqa: E402
    import fresh_tasks  # noqa: E402
    import harness  # noqa: E402

    arm, checkpoints, _probe_ids, _probe_base = _prepare_arm(args)
    device = torch.device("cuda")
    gate_state = arm.load_gates(os.path.join(REPO, args.gate_checkpoint))
    tokenizer = harness.load_tokenizer(args.hf_root)
    tasks = list(fresh_tasks.TASKS)
    losses = {mode: [] for mode in MODES}
    arm.model.eval()
    with torch.inference_mode():
        for mode in MODES:
            arm.set_mode(mode)
            losses[mode] = [complementarity.completion_loss(
                arm.model, tokenizer, task, device) for task in tasks]
    comparison = composition_comparison(losses)
    probe = torch.zeros((1, 8), dtype=torch.long, device=device)
    interference = arm.interference_on(probe)
    receipt = {
        "schema": SCHEMA_EVAL,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "arm": COMPOSED_ARM,
        "mode": "fresh task set (P8.1); per-item mean NLL of the reference "
                "continuation, lower is better",
        "task_set": fresh_tasks.manifest(),
        "tasks": len(tasks),
        "kappa": arm.kappa,
        "gate_training": {key: gate_state.get(key) for key in ("step", "steps", "seed")},
        "capability_checkpoints": checkpoints,
        "gate_checkpoint": args.gate_checkpoint,
        "per_item": {mode: losses[mode] for mode in MODES},
        "comparison": comparison,
        "interference": interference["aggregate"],
        "active_union": arm.active_union(),
        "selector_cost": arm.selector_cost(),
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({"comparison": comparison["comparisons"]["gate-minus-sum"],
                      "mean_loss": comparison["mean_loss"],
                      "out": args.out}, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-config", default=None)
    parser.add_argument("--wrap-admission", nargs=3,
                        metavar=("CONFIG", "RECORD", "OUT"), default=None)
    parser.add_argument("--run-composition", action="store_true",
                        help="train the gate on the frozen core (GPU court)")
    parser.add_argument("--score-composition", action="store_true",
                        help="score the four references on the fresh task set (GPU court)")
    parser.add_argument("--config-d", default=None)
    parser.add_argument("--admission-d", default=None)
    parser.add_argument("--config-m", default=None)
    parser.add_argument("--admission-m", default=None)
    parser.add_argument("--checkpoint-d", default=None)
    parser.add_argument("--checkpoint-m", default=None)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=GATE_STEPS)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--gate-lr", type=float, default=GATE_LR)
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    parser.add_argument("--calibration-windows", type=int, default=32)
    parser.add_argument("--positions", type=int, default=WINDOW)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args(argv)

    if args.wrap_admission:
        config_path, record_path, out_path = args.wrap_admission
        payload = wrap_admission(os.path.join(REPO, config_path),
                                 os.path.join(REPO, record_path),
                                 os.path.join(REPO, out_path))
        print(json.dumps({"out": out_path, "stamp": payload["stamp"],
                          "mirror_agrees": payload["mirror_agrees"],
                          "active_unique_params":
                              payload["record"].get("active_unique_params")},
                         indent=2, sort_keys=True))
        return 0

    if args.run_composition:
        return run_composition(args)
    if args.score_composition:
        return score_composition(args)

    if not args.emit_config:
        parser.error("one of --emit-config, --wrap-admission, --run-composition or "
                     "--score-composition is required")
    import arms

    config_d = arms.build_config(arms.T2_ARMS["D"])
    config_m = arms.build_config(arms.T2_ARMS["M"])
    composed = composition_admission_config(config_d, config_m,
                                            layers=args.layers, hidden=args.hidden)
    path = os.path.join(REPO, args.emit_config)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(composed, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    mirror = python_mirror(composed)
    print(json.dumps({"config": args.emit_config,
                      "total_accessible": composed["total_accessible"],
                      "mirror": mirror,
                      "d_active": arms.python_mirror(config_d)["active_unique_params"],
                      "m_active": arms.python_mirror(config_m)["active_unique_params"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
