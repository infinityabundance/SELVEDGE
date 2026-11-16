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
COMPOSED_ARM = "P8-COMPOSE"


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
    """`y = S + a r_D + b r_M` with a zero-initialised, L2-bounded gate."""

    def __init__(self, hidden: int, budget: float, feature_dim: int | None = None):
        super().__init__()
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.hidden = hidden
        self.budget = float(budget)
        self.feature_dim = feature_dim or hidden
        self.gate = nn.Linear(self.feature_dim, 2, bias=True)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.zero_()

    def forward(self, base: torch.Tensor, residual_d: torch.Tensor,
                residual_m: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, dict]:
        coefficients = self.gate(feature)
        a = coefficients[..., 0:1].to(base.dtype)
        b = coefficients[..., 1:2].to(base.dtype)
        perturbation = a * residual_d + b * residual_m
        norms = perturbation.norm(dim=-1, keepdim=True)
        scale = torch.where(norms > self.budget, self.budget / norms.clamp_min(1e-12),
                            torch.ones_like(norms))
        # `a`, `b` and `scale` all carry a trailing singleton, so they broadcast
        # over the hidden axis for `[tokens, hidden]` and `[batch, seq, hidden]`.
        a = a * scale
        b = b * scale
        perturbation = a * residual_d + b * residual_m
        output = base + perturbation
        info = {
            "a": a.detach(),
            "b": b.detach(),
            "perturbation_norm": perturbation.detach().norm(dim=-1).reshape(-1),
            "scale": scale.reshape(-1).detach(),
            "budget": self.budget,
        }
        return output, info


def _norms(tensor: torch.Tensor) -> torch.Tensor:
    """Per-position L2 norm over the last (hidden) axis."""
    return tensor.detach().norm(dim=-1).reshape(-1)


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
    a = info["a"].to(base.dtype)
    b = info["b"].to(base.dtype)
    recomputed = base + a * residual_d + b * residual_m
    accounted = bool(torch.allclose(recomputed, composed, atol=tolerance,
                                    rtol=tolerance))
    added = _norms(b * residual_m)
    deviation = _norms((a - 1.0) * residual_d + b * residual_m)
    perturbation = _norms(a * residual_d + b * residual_m)
    return {
        "accounted": accounted,
        "budget": float(info["budget"]),
        "added_effect_of_m": {"mean": float(added.mean()), "max": float(added.max())},
        "deviation_from_d_alone": {"mean": float(deviation.mean()),
                                   "max": float(deviation.max())},
        "perturbation": {"mean": float(perturbation.mean()),
                         "max": float(perturbation.max())},
        "within_budget": bool((perturbation <= float(info["budget"]) + tolerance).all()),
        "count": int(added.numel()),
    }


def check_budget(info: dict, tolerance: float = 1e-6) -> dict:
    """The realised perturbation must stay inside the declared budget."""
    norms = info["perturbation_norm"].reshape(-1)
    budget = float(info["budget"])
    violations = int((norms > budget * (1.0 + tolerance) + tolerance).sum())
    return {"budget": budget, "max_perturbation_norm": float(norms.max())
            if norms.numel() else 0.0, "violations": violations}


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-config", default=None)
    parser.add_argument("--wrap-admission", nargs=3,
                        metavar=("CONFIG", "RECORD", "OUT"), default=None)
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

    if not args.emit_config:
        parser.error("one of --emit-config or --wrap-admission is required")
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
