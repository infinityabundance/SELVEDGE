#!/usr/bin/env python3
"""Build and check the T1 admission configs from the real accounting receipts.

The Rust ledger is the source of truth for the parameter ceiling. This module
only *declares* each arm's learned-parameter slices and its per-token FLOPs; the
`selvedge admit` command builds the union, admits or refuses it, and writes the
record. `check` then verifies the emitted records against the design counts in
`docs/EXPERIMENT-CONTRACT.md` (T1.2, post-A5) and against a Python mirror of the
union. Invoked by `scripts/train-admission.sh`; never used as the source of truth.

Subcommands:

  emit    write results/train/admission-configs/<arm>.json from the receipts
  check   verify every emitted record against the design count and the mirror
  mirror  print the Python union for one config (training's cross-check)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACCOUNTING_GLOB = "results/accounting/*-qwen3-{size}.json"
CONFIG_DIR = "results/train/admission-configs"
ADMISSION_DIR = "results/train/admission"
CEILING = 1_000_000_000
SCHEMA_CONFIG = "selvedge/admission-config/v1"
TRAIN_CONTEXT = 1024

# Design counts fixed in docs/EXPERIMENT-CONTRACT.md T1.2 (after correction A5).
DESIGN_ACTIVE = {
    "A": 998_269_952,
    "A0": 998_269_952,
    "B": 772_440_064,
    "C": 772_210_688,
    "C2": 992_411_648,
    "D": 596_049_920,
}
DESIGN_TOTAL_ACCESSIBLE = {
    "A": 1_732_502_528,
    "A0": 1_732_502_528,
    "B": 948_600_832,
    "C": 772_210_688,
    "C2": 992_411_648,
    "D": 596_049_920,
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def accounting(size: str) -> dict:
    matches = sorted(glob.glob(os.path.join(REPO, ACCOUNTING_GLOB.format(size=size))))
    if not matches:
        raise SystemExit(f"arms: no accounting receipt for {size}: {ACCOUNTING_GLOB}")
    return load_json(matches[-1])


@dataclass(frozen=True)
class Arm:
    name: str
    size: str
    kind: str  # dense | routed_groups | specialists | dense_adapter
    groups: int = 0
    k: int = 0
    adapter_rank: int = 0
    specialists: int = 0
    specialist_width: int = 0
    dense_width: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


ARMS = {
    "D": Arm(name="D", size="0.6b", kind="dense",
             notes=("unmodified dense Qwen3-0.6B baseline; no training",)),
    "A": Arm(name="A", size="1.7b", kind="routed_groups", groups=16, k=5, adapter_rank=32,
             notes=("frozen 1.7B backbone + per-layer router over 16 FFN groups, k=5, "
                    "rank-32 residual adapters on the selected groups",
                    "slot identities stand for the k realized groups; the runtime check "
                    "requires k distinct groups per layer and the Python mirror uses the "
                    "realized group identities")),
    "A0": Arm(name="A0", size="1.7b", kind="routed_groups", groups=16, k=5, adapter_rank=32,
              notes=("untrained A reference: seed-1001 router, zero adapters",)),
    "B": Arm(name="B", size="0.6b", kind="specialists", specialists=8, k=4,
             specialist_width=512,
             notes=("frozen 0.6B core + per-layer 8 SwiGLU specialists of width 512, "
                    "hard top-4 routing, zero-initialised down projections",)),
    "C": Arm(name="C", size="0.6b", kind="dense_adapter", dense_width=2048,
             notes=("frozen 0.6B core + one dense width-2048 SwiGLU adapter per layer; "
                    "parameter-matched to B's active specialists",)),
    "C2": Arm(name="C2", size="0.6b", kind="dense_adapter", dense_width=4608,
              notes=("frozen 0.6B core + one dense width-4608 SwiGLU adapter per layer; "
                     "ceiling-max dense control, not parameter-matched to B",)),
}


def _use(tensor: str, elements: int) -> dict:
    if elements <= 0:
        raise SystemExit(f"arms: non-positive element count for {tensor}")
    return {
        "identity": {"tensor": tensor, "elements": {"start": 0, "end": elements}},
        "reads": elements,
        "bytes": elements * 2,
        "tier": "gpu",
    }


def _flops(matmul: int, other: int) -> dict:
    return {"matmul_flops": matmul, "other_flops": other, "controller_flops": 0}


def build_config(arm: Arm) -> dict:
    receipt = accounting(arm.size)
    hidden = receipt["model"]["hidden"]
    intermediate = receipt["model"]["intermediate"]
    layers = receipt["model"]["layers"]
    tie = receipt["tie"]
    per_token = receipt["flops"]["per_token"]

    ffn_by_layer: dict[int, list[dict]] = {}
    for component in receipt["components"]:
        if component["ffn"]:
            ffn_by_layer.setdefault(component["layer"], []).append(component)

    selector_uses: list[dict] = []
    execution_uses: list[dict] = []
    execution_matmul = 0
    controller_matmul = 0

    for component in receipt["components"]:
        if component["ffn"]:
            continue
        if component["component"] == "lm_head" and tie.get("treated_as_shared"):
            continue  # counted once, under the embedding key
        execution_uses.append(_use(component["tensor"], component["elements"]))
        execution_matmul += 2 * component["elements"]

    execution_other = per_token["norms_per_layer_estimate"] * layers
    # Receipt formula: attention core = 4 * context * (heads * head_dim) for
    # QK^T and AV; softmax estimate = 5 * context * heads. The receipt stores
    # these at contexts 1, 2048 and 40960, so the registered 1024 is computed
    # from the same formula and the model's real q_proj width.
    q_out = next(component["shape"][0] for component in receipt["components"]
                 if component["tensor"].endswith("q_proj.weight"))
    attention_core = 4 * TRAIN_CONTEXT * q_out
    attention_softmax = 5 * TRAIN_CONTEXT * receipt["model"]["heads"]
    execution_other += attention_core + attention_softmax

    additions = 0

    if arm.kind == "routed_groups":
        group_width = intermediate // arm.groups
        group_elements = 3 * group_width * hidden
        adapter_elements = 2 * arm.adapter_rank * group_width
        for layer in range(layers):
            for slot in range(arm.k):
                for suffix in ("gate", "up", "down"):
                    elements = group_width * hidden
                    key = f"model.layers.{layer}.mlp.{suffix}_proj.weight.slot{slot}"
                    execution_uses.append(_use(key, elements))
                    execution_matmul += 2 * elements
                key = f"model.layers.{layer}.mlp.adapter.slot{slot}"
                execution_uses.append(_use(key, adapter_elements))
                execution_matmul += 2 * adapter_elements
            router_elements = hidden * arm.groups
            selector_uses.append(_use(f"model.layers.{layer}.router.weight", router_elements))
            controller_matmul += 2 * router_elements
        execution_other += int(round(per_token["ffn_elementwise_per_layer_estimate"]
                                     * layers * (arm.k / arm.groups)))
        additions = layers * (arm.groups * adapter_elements + hidden * arm.groups)
        # The dense FFN is stored as the 16 groups, so the groups themselves are
        # already inside the receipt's unique count; only adapters and routers are new.
    elif arm.kind == "specialists":
        specialist_elements = 2 * arm.specialist_width * hidden + hidden * arm.specialist_width
        for layer in range(layers):
            for component in ffn_by_layer[layer]:
                execution_uses.append(_use(component["tensor"], component["elements"]))
                execution_matmul += 2 * component["elements"]
            for slot in range(arm.k):
                for suffix in ("gate", "up", "down"):
                    elements = arm.specialist_width * hidden
                    key = f"model.layers.{layer}.specialists.{suffix}.slot{slot}"
                    execution_uses.append(_use(key, elements))
                    execution_matmul += 2 * elements
            router_elements = hidden * arm.specialists
            selector_uses.append(_use(f"model.layers.{layer}.specialists.router.weight",
                                      router_elements))
            controller_matmul += 2 * router_elements
        execution_other += int(round(per_token["ffn_elementwise_per_layer_estimate"]
                                     * layers * (1.0 + arm.k * arm.specialist_width
                                                 / intermediate)))
        additions = layers * (arm.specialists * specialist_elements + hidden * arm.specialists)
    elif arm.kind == "dense_adapter":
        per_layer = 3 * arm.dense_width * hidden
        for layer in range(layers):
            for component in ffn_by_layer[layer]:
                execution_uses.append(_use(component["tensor"], component["elements"]))
                execution_matmul += 2 * component["elements"]
            for suffix in ("gate", "up", "down"):
                elements = arm.dense_width * hidden
                key = f"model.layers.{layer}.dense_adapter.{suffix}.weight"
                execution_uses.append(_use(key, elements))
                execution_matmul += 2 * elements
        execution_other += int(round(per_token["ffn_elementwise_per_layer_estimate"]
                                     * layers * (1.0 + arm.dense_width / intermediate)))
        additions = layers * per_layer
    elif arm.kind == "dense":
        for layer in range(layers):
            for component in ffn_by_layer[layer]:
                execution_uses.append(_use(component["tensor"], component["elements"]))
                execution_matmul += 2 * component["elements"]
        execution_other += per_token["ffn_elementwise_per_layer_estimate"] * layers
        additions = 0
    else:  # pragma: no cover
        raise SystemExit(f"arms: unknown kind {arm.kind}")

    total_accessible = receipt["budget"]["unique_learned_elements"] + additions

    return {
        "schema": SCHEMA_CONFIG,
        "arm": arm.name,
        "budget": {"active_unique_limit": CEILING, "mode": "compliance"},
        "total_accessible": total_accessible,
        "selector_uses": selector_uses,
        "execution_uses": execution_uses,
        "selector_flops": _flops(controller_matmul, 0),
        "execution_flops": _flops(execution_matmul, execution_other),
        "notes": list(arm.notes),
    }


def python_mirror(config: dict) -> dict:
    """Union of the declared selector and execution identities (cross-check only)."""
    spans: dict[str, list[tuple[int, int]]] = {}
    selector = 0
    execution = 0
    for group, counter in (("selector_uses", "selector"),
                           ("execution_uses", "execution")):
        for use in config[group]:
            key = use["identity"]["tensor"]
            start = use["identity"]["elements"]["start"]
            end = use["identity"]["elements"]["end"]
            spans.setdefault(key, []).append((start, end))
            if counter == "selector":
                selector += end - start
            else:
                execution += end - start
    union = 0
    for ranges in spans.values():
        ranges.sort()
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                union += current_end - current_start
                current_start, current_end = start, end
        union += current_end - current_start
    return {
        "active_unique_params": union,
        "selector_elements": selector,
        "execution_elements": execution,
        "identities": len(spans),
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(out_dir: str, arms: list[str]) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for name in arms:
        config = build_config(ARMS[name])
        path = os.path.join(out_dir, f"{name}.json")
        write_json_atomic(path, config)
        flops = config["execution_flops"]
        print(f"arms: {name} config={path} total_accessible={config['total_accessible']} "
              f"forward_matmul_flops={flops['matmul_flops']} "
              f"declared_active={DESIGN_ACTIVE[name]}")
    return 0


def check(out_dir: str, admission_dir: str, arms: list[str]) -> int:
    failures = []
    for name in arms:
        config_path = os.path.join(out_dir, f"{name}.json")
        record_path = os.path.join(admission_dir, f"{name}.json")
        if not os.path.exists(config_path):
            failures.append(f"{name}: missing config {config_path}")
            continue
        if not os.path.exists(record_path):
            failures.append(f"{name}: no admission record at {record_path}")
            continue
        config = load_json(config_path)
        record = load_json(record_path)
        mirror = python_mirror(config)
        if not record.get("admitted"):
            failures.append(f"{name}: admission refused: {record.get('refusal')}")
            continue
        if record.get("config_sha256") != sha256_file(config_path):
            failures.append(f"{name}: record config_sha256 does not match the config bytes")
        if record["active_unique_params"] != DESIGN_ACTIVE[name]:
            failures.append(f"{name}: admission union {record['active_unique_params']} != "
                            f"design {DESIGN_ACTIVE[name]}")
        if mirror["active_unique_params"] != record["active_unique_params"]:
            failures.append(f"{name}: Python mirror {mirror['active_unique_params']} "
                            f"disagrees with the Rust record "
                            f"{record['active_unique_params']}")
        if config["total_accessible"] != DESIGN_TOTAL_ACCESSIBLE[name]:
            failures.append(f"{name}: total_accessible {config['total_accessible']} != "
                            f"design {DESIGN_TOTAL_ACCESSIBLE[name]}")
        if record["active_unique_params"] > CEILING:
            failures.append(f"{name}: over ceiling")
        print(f"arms: {name} admitted active={record['active_unique_params']} "
              f"selector={record['selector_unique_params']} "
              f"execution={record['execution_unique_params']} "
              f"total_accessible={config['total_accessible']}")
    if failures:
        for failure in failures:
            print(f"arms: FAIL {failure}", file=sys.stderr)
        return 1
    print("arms: all admission records match the design counts and the Python mirror")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["emit", "check", "mirror"])
    parser.add_argument("--out", default=os.path.join(REPO, CONFIG_DIR))
    parser.add_argument("--admission", default=os.path.join(REPO, ADMISSION_DIR))
    parser.add_argument("--arms", default="D,A,A0,B,C,C2")
    parser.add_argument("--config", help="config to mirror when command=mirror")
    args = parser.parse_args(argv)
    names = [item for item in args.arms.split(",") if item]
    unknown = [item for item in names if item not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}")
    if args.command == "emit":
        return emit(args.out, names)
    if args.command == "check":
        return check(args.out, args.admission, names)
    if not args.config:
        parser.error("--config is required for mirror")
    print(json.dumps(python_mirror(load_json(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
