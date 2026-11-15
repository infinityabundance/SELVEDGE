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
    kind: str  # dense | routed_groups | specialists | dense_adapter | memory | hybrid
    groups: int = 0
    k: int = 0
    adapter_rank: int = 0
    specialists: int = 0
    specialist_width: int = 0
    dense_width: int = 0
    memory_layers: tuple[int, ...] = ()
    memory_rows: int = 1 << 18
    memory_value_dim: int = 256
    memory_reads: int = 2
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def inserts(self) -> int:
        return len(self.memory_layers)


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

# ---------------------------------------------------------------------------
# T2: the five-arm error-directed capacity matrix (docs/EXPERIMENT-CONTRACT.md T2)
#
# T2's `D` is T1's `C` (the parameter-matched dense control); T1's untrained
# dense baseline is `D0` in T2. The T2 names live in their own table so no T1
# definition can be rewritten by accident.

T2_MEMORY_LAYERS = (3, 10, 17, 24)
T2_MEMORY_ROWS = 1 << 18
T2_MEMORY_VALUE_DIM = 256
T2_MEMORY_READS = 2

DESIGN_ACTIVE_T2 = {
    "D": 772_210_688,
    "M": 598_153_216,
    "X": 772_440_064,
    "H": 774_543_360,
    "H*": 774_543_360,
}
DESIGN_TOTAL_T2 = {
    "D": 772_210_688,
    "M": 866_586_624,
    "X": 948_600_832,
    "H": 1_219_137_536,
    "H*": 1_219_137_536,
}

_MEMORY_NOTES = (
    "hashed memory: 2 n-gram heads (order 2 and 3), no learned index, 2 row reads "
    "per token per insertion layer, 256-dim values, 2^18 rows per table, four "
    "insertion layers at 0-based depth 3, 10, 17 and 24, per-layer 512->1024 "
    "projection and 1024->1 sigmoid gate; the projection is zero-initialised",
)
_SPECIALIST_NOTES = (
    "frozen 0.6B core + per-layer 8 SwiGLU specialists of width 512, hard top-4 "
    "routing, zero-initialised down projections",
)

T2_ARMS = {
    "D": Arm(name="D", size="0.6b", kind="dense_adapter", dense_width=2048,
             notes=("T2 dense control = T1 arm C: frozen core + one dense width-2048 "
                    "SwiGLU adapter per layer, parameter-matched to X's active experts",)),
    "M": Arm(name="M", size="0.6b", kind="memory",
             memory_layers=T2_MEMORY_LAYERS, memory_rows=T2_MEMORY_ROWS,
             memory_value_dim=T2_MEMORY_VALUE_DIM, memory_reads=T2_MEMORY_READS,
             notes=_MEMORY_NOTES + ("no expert bank",)),
    "X": Arm(name="X", size="0.6b", kind="specialists", specialists=8, k=4,
             specialist_width=512, notes=_SPECIALIST_NOTES + ("no memory path",)),
    "H": Arm(name="H", size="0.6b", kind="hybrid", specialists=8, k=4,
             specialist_width=512,
             memory_layers=T2_MEMORY_LAYERS, memory_rows=T2_MEMORY_ROWS,
             memory_value_dim=T2_MEMORY_VALUE_DIM, memory_reads=T2_MEMORY_READS,
             notes=_SPECIALIST_NOTES + _MEMORY_NOTES
             + ("plain distillation router plus Switch load balance",)),
    "H*": Arm(name="H*", size="0.6b", kind="hybrid", specialists=8, k=4,
              specialist_width=512,
              memory_layers=T2_MEMORY_LAYERS, memory_rows=T2_MEMORY_ROWS,
              memory_value_dim=T2_MEMORY_VALUE_DIM, memory_reads=T2_MEMORY_READS,
              notes=_SPECIALIST_NOTES + _MEMORY_NOTES
              + ("same modules as H; router trained on measured leave-one-expert-out "
                 "utility labels, no load balance and no distillation gradient to "
                 "the router",)),
}


def memory_elements(arm: Arm, hidden: int = 1024) -> dict:
    """Memory-path element arithmetic for one arm (T2.2).

    Returns per-layer and total counts; `active` excludes the table because a
    read touches two 256-element rows, not the whole table.
    """
    if not arm.inserts:
        return {"inserts": 0, "rows": 0, "values": 0, "projection": 0,
                "gate": 0, "index": 0, "active": 0, "total": 0}
    per_values = arm.memory_reads * arm.memory_value_dim
    per_projection = arm.memory_reads * arm.memory_value_dim * hidden
    per_gate = hidden
    per_layer_active = per_values + per_projection + per_gate
    per_table = arm.memory_rows * arm.memory_value_dim
    return {
        "inserts": arm.inserts,
        "rows": arm.inserts * arm.memory_rows,
        "values": arm.inserts * per_table,
        "projection": arm.inserts * per_projection,
        "gate": arm.inserts * per_gate,
        "index": 0,
        "active": arm.inserts * per_layer_active,
        "total": arm.inserts * (per_table + per_projection + per_gate),
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
    elif arm.kind == "memory":
        memory = memory_elements(arm, hidden)
        for layer in range(layers):
            for component in ffn_by_layer[layer]:
                execution_uses.append(_use(component["tensor"], component["elements"]))
                execution_matmul += 2 * component["elements"]
        execution_other += per_token["ffn_elementwise_per_layer_estimate"] * layers
        for layer in arm.memory_layers:
            # Two distinct 256-element value rows: the conservative union even
            # when the two hash heads collide on one row.
            for slot in range(arm.memory_reads):
                elements = arm.memory_value_dim
                key = f"model.layers.{layer}.memory.table.row.slot{slot}"
                execution_uses.append(_use(key, elements))
                execution_other += elements  # gather (element copies, not FLOPs)
            projection = arm.memory_reads * arm.memory_value_dim * hidden
            key = f"model.layers.{layer}.memory.proj.weight"
            execution_uses.append(_use(key, projection))
            execution_matmul += 2 * projection
            gate = hidden
            key = f"model.layers.{layer}.memory.gate.weight"
            execution_uses.append(_use(key, gate))
            execution_matmul += 2 * gate
        additions = memory["total"]
    elif arm.kind == "hybrid":
        memory = memory_elements(arm, hidden)
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
            if layer in arm.memory_layers:
                for slot in range(arm.memory_reads):
                    elements = arm.memory_value_dim
                    key = f"model.layers.{layer}.memory.table.row.slot{slot}"
                    execution_uses.append(_use(key, elements))
                    execution_other += elements
                projection = arm.memory_reads * arm.memory_value_dim * hidden
                key = f"model.layers.{layer}.memory.proj.weight"
                execution_uses.append(_use(key, projection))
                execution_matmul += 2 * projection
                gate = hidden
                key = f"model.layers.{layer}.memory.gate.weight"
                execution_uses.append(_use(key, gate))
                execution_matmul += 2 * gate
        execution_other += int(round(per_token["ffn_elementwise_per_layer_estimate"]
                                     * layers * (1.0 + arm.k * arm.specialist_width
                                                 / intermediate)))
        additions = (layers * (arm.specialists * specialist_elements
                               + hidden * arm.specialists) + memory["total"])
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


def _tables(profile: str) -> tuple[dict, dict, dict]:
    if profile == "t2":
        return T2_ARMS, DESIGN_ACTIVE_T2, DESIGN_TOTAL_T2
    return ARMS, DESIGN_ACTIVE, DESIGN_TOTAL_ACCESSIBLE


def emit(out_dir: str, arms: list[str], profile: str = "t1") -> int:
    table, design_active, _design_total = _tables(profile)
    os.makedirs(out_dir, exist_ok=True)
    for name in arms:
        config = build_config(table[name])
        path = os.path.join(out_dir, f"{name}.json")
        write_json_atomic(path, config)
        flops = config["execution_flops"]
        print(f"arms: {profile} {name} config={path} "
              f"total_accessible={config['total_accessible']} "
              f"forward_matmul_flops={flops['matmul_flops']} "
              f"declared_active={design_active[name]}")
    return 0


def check(out_dir: str, admission_dir: str, arms: list[str], profile: str = "t1") -> int:
    table, design_active, design_total = _tables(profile)
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
        if record["active_unique_params"] != design_active[name]:
            failures.append(f"{name}: admission union {record['active_unique_params']} != "
                            f"design {design_active[name]}")
        if mirror["active_unique_params"] != record["active_unique_params"]:
            failures.append(f"{name}: Python mirror {mirror['active_unique_params']} "
                            f"disagrees with the Rust record "
                            f"{record['active_unique_params']}")
        if config["total_accessible"] != design_total[name]:
            failures.append(f"{name}: total_accessible {config['total_accessible']} != "
                            f"design {design_total[name]}")
        if record["active_unique_params"] > CEILING:
            failures.append(f"{name}: over ceiling")
        print(f"arms: {profile} {name} admitted active={record['active_unique_params']} "
              f"selector={record['selector_unique_params']} "
              f"execution={record['execution_unique_params']} "
              f"total_accessible={config['total_accessible']}")
    if failures:
        for failure in failures:
            print(f"arms: FAIL {failure}", file=sys.stderr)
        return 1
    print(f"arms: {profile} records match the design counts and the Python mirror")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["emit", "check", "mirror"])
    parser.add_argument("--profile", choices=["t1", "t2"], default="t1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--config", help="config to mirror when command=mirror")
    args = parser.parse_args(argv)
    table, _design_active, _design_total = _tables(args.profile)
    default_out = os.path.join(REPO, CONFIG_DIR, *([] if args.profile == "t1" else ["t2"]))
    default_admission = os.path.join(REPO, ADMISSION_DIR,
                                     *([] if args.profile == "t1" else ["t2"]))
    out = args.out or default_out
    admission = args.admission or default_admission
    default_arms = "D,A,A0,B,C,C2" if args.profile == "t1" else "D,M,X,H,H*"
    names = [item for item in (args.arms or default_arms).split(",") if item]
    unknown = [item for item in names if item not in table]
    if unknown:
        parser.error(f"unknown {args.profile} arms: {unknown}")
    if args.command == "emit":
        return emit(out, names, args.profile)
    if args.command == "check":
        return check(out, admission, names, args.profile)
    if not args.config:
        parser.error("--config is required for mirror")
    print(json.dumps(python_mirror(load_json(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
