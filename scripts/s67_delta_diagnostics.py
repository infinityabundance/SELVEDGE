#!/usr/bin/env python3
"""S67: diagnostics on the three fine-tuning deltas before averaging.

The S66 replication produced three full-arm checkpoints trained on the same
recipe with different seeds. This script measures whether the three updates

    d_s = theta_s - theta_0        (theta_0 = post-trained 0.8B endpoint)

occupy compatible directions before any of them is averaged. Nothing is
written except a receipt; no evaluation model is loaded.

What is reported, per pair of seeds and overall:

* global cosine similarity of the flattened deltas,
* the delta norm per seed and the norm ratio ``||d_s|| / ||theta_0||``,
* per-layer cosine similarity and per-layer delta norms, with the layer index
  parsed out of the parameter name,
* the most anti-correlated layer and the count of layers with negative
  cosine, which is the fail-closed input to the decision below.

Decision rule (fixed here, before the numbers are read): the soup route is
adopted only if every pairwise global cosine is positive and no layer is
*strongly* anti-correlated (cosine < -0.10 for any pair). Otherwise the
registered fallback route is taken: the single S65 seed-0 checkpoint is the
capability endpoint, and the soup is not built.

Usage (CPU, oracle court because this image pins torch):
    SELVEDGE_MEM=16g scripts/court oracle -- \
      python3 scripts/s67_delta_diagnostics.py \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --seed 0:results/stage2/runs/s58-stage2a/ckpt-004000.pt \
        --seed 1:results/stage2/runs/s66-stage2a-seed1-guard20/ckpt-004000.pt \
        --seed 2:results/stage2/runs/s66-stage2a-seed2-guard20/ckpt-004000.pt \
        --out results/stage2/s67-soup/delta-diagnostics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from s60_interpolate import _base_arm_state, torch_load, sha256_file  # noqa: E402

SCHEMA = "selvedge/s67-delta-diagnostics/v1"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
STRONG_ANTI = -0.10


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def layer_of(key: str) -> str:
    match = LAYER_RE.search(key)
    return f"layer.{int(match.group(1)):02d}" if match else "non-layer"


def norms_and_cosines(deltas: list) -> dict:
    """Pairwise cosine/norm arithmetic on a list of flat float32 tensors."""
    import torch

    squares = [float(torch.sum(delta * delta)) for delta in deltas]
    dots = {}
    for left in range(len(deltas)):
        for right in range(left + 1, len(deltas)):
            dots[f"{left}-{right}"] = float(
                torch.sum(deltas[left] * deltas[right]))
    cosines = {}
    for pair, dot in dots.items():
        left, right = (int(part) for part in pair.split("-"))
        denom = (squares[left] * squares[right]) ** 0.5
        cosines[pair] = dot / denom if denom > 0 else None
    return {
        "squared_norms": squares,
        "norms": [value ** 0.5 for value in squares],
        "dots": dots,
        "cosines": cosines,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seed", action="append", required=True,
                        help="<index>:<checkpoint path>; repeat per seed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    seeds = []
    for item in args.seed:
        index, path = item.split(":", 1)
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            raise SystemExit(f"s67-delta: missing checkpoint {path}")
        seeds.append((int(index), path, full))
    seeds.sort()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False)
    checkpoints = [torch_load(path) for _, _, path in seeds]
    for (index, path, _), payload in zip(seeds, checkpoints):
        if payload.get("arm") != "full":
            raise SystemExit(f"s67-delta: seed {index} is not a full arm")
    keys = sorted(checkpoints[0]["arm_state"])
    for (index, path, _), payload in zip(seeds[1:], checkpoints[1:]):
        if sorted(payload["arm_state"]) != keys:
            raise SystemExit(f"s67-delta: seed {index} key set differs")
    base_state = _base_arm_state(model, checkpoints[0]["arm_state"])
    base_squares = 0.0

    # Per-layer accumulators: squared norms per seed, pairwise dots, and the
    # per-tensor cosine extrema.
    layers: dict[str, dict] = {}
    worst = {"cosine": None, "pair": None, "layer": None, "tensor": None}
    total_elements = 0

    for key in keys:
        base = base_state[key]
        if not hasattr(base, "dtype") or not torch.is_floating_point(base):
            continue
        base_flat = base.detach().float().reshape(-1)
        base_squares += float(torch.sum(base_flat * base_flat))
        deltas = [
            (payload["arm_state"][key].detach().float().reshape(-1) - base_flat)
            for payload in checkpoints
        ]
        layer = layer_of(key)
        bucket = layers.setdefault(layer, {
            "tensors": 0,
            "elements": 0,
            "squared_norms": [0.0 for _ in deltas],
            "dots": {f"{left}-{right}": 0.0
                     for left in range(len(deltas))
                     for right in range(left + 1, len(deltas))},
            "min_tensor_cosine": None,
        })
        bucket["tensors"] += 1
        bucket["elements"] += int(base_flat.numel())
        total_elements += int(base_flat.numel())
        local = norms_and_cosines(deltas)
        for index, value in enumerate(local["squared_norms"]):
            bucket["squared_norms"][index] += value
        for pair, value in local["dots"].items():
            bucket["dots"][pair] += value
        for pair, value in local["cosines"].items():
            if value is None:
                continue
            current = bucket["min_tensor_cosine"]
            if current is None or value < current:
                bucket["min_tensor_cosine"] = value
            if worst["cosine"] is None or value < worst["cosine"]:
                worst = {"cosine": value, "pair": pair, "layer": layer,
                         "tensor": key}

    global_stats = {
        "squared_norms": [0.0 for _ in seeds],
        "dots": {},
    }
    for bucket in layers.values():
        for index, value in enumerate(bucket["squared_norms"]):
            global_stats["squared_norms"][index] += value
        for pair, value in bucket["dots"].items():
            global_stats["dots"][pair] = (
                global_stats["dots"].get(pair, 0.0) + value)

    def pair_stats(squared: list[float], dots: dict[str, float]) -> dict:
        norms = [value ** 0.5 for value in squared]
        cosines = {}
        for pair, dot in dots.items():
            left, right = (int(part) for part in pair.split("-"))
            denom = (squared[left] * squared[right]) ** 0.5
            cosines[pair] = dot / denom if denom > 0 else None
        return {"norms": norms, "cosines": cosines}

    global_rows = pair_stats(global_stats["squared_norms"], global_stats["dots"])
    layer_rows = {}
    negative = []
    for name in sorted(layers):
        stats = pair_stats(layers[name]["squared_norms"], layers[name]["dots"])
        layer_rows[name] = {
            "tensors": layers[name]["tensors"],
            "elements": layers[name]["elements"],
            "norms": stats["norms"],
            "cosines": stats["cosines"],
            "min_tensor_cosine": layers[name]["min_tensor_cosine"],
        }
        for pair, value in stats["cosines"].items():
            if value is not None and value < 0.0:
                negative.append({"layer": name, "pair": pair,
                                 "cosine": value})

    pairs = list(global_rows["cosines"])
    strongly_anti = [row for row in negative if row["cosine"] < STRONG_ANTI]
    compatible = (all(value is not None and value > 0.0
                      for value in global_rows["cosines"].values())
                  and not strongly_anti)

    norm_ratio = []
    for norm in global_rows["norms"]:
        norm_ratio.append(norm / (base_squares ** 0.5) if base_squares else None)

    payload = {
        "schema": SCHEMA,
        "kind": "delta-diagnostics",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "model_dir": args.model_dir,
        "base": {
            "endpoint": "post-trained Qwen3.5-0.8B (untrained endpoint)",
            "squared_norm": base_squares,
            "norm": base_squares ** 0.5,
            "elements": total_elements,
        },
        "seeds": [
            {"seed": index, "checkpoint": path,
             "checkpoint_sha256": sha256_file(os.path.join(ROOT, path)),
             "step": checkpoint.get("step")}
            for (index, path, _), checkpoint in zip(seeds, checkpoints)
        ],
        "pairs": pairs,
        "global": {
            "norms": global_rows["norms"],
            "norm_ratio_vs_base": norm_ratio,
            "cosines": global_rows["cosines"],
        },
        "layers": layer_rows,
        "negative_cosine_layers": negative,
        "worst_tensor": worst,
        "decision": {
            "strong_anti_threshold": STRONG_ANTI,
            "any_pair_global_non_positive": not all(
                value is not None and value > 0.0
                for value in global_rows["cosines"].values()),
            "strongly_anti_correlated_layers": strongly_anti,
            "directions_compatible": compatible,
            "route": ("damped soup (theta_0 + alpha * mean_s d_s)"
                      if compatible
                      else "S65 seed-0 checkpoint (registered fallback); "
                           "soup not built"),
        },
        "claims": {
            "reads_no_test_split": True,
            "writes_no_checkpoint": True,
            "arithmetic": "delta = theta_seed - theta_post in float32; "
                          "cosine and norms accumulated per parameter and "
                          "aggregated by layer; layers parsed from names",
        },
    }
    write_json(os.path.join(ROOT, args.out), payload)
    print(f"s67-delta: global cosines "
          f"{ {pair: round(value, 4) for pair, value in global_rows['cosines'].items()} }")
    print(f"s67-delta: negative layers={len(negative)} "
          f"strongly_anti={len(strongly_anti)} compatible={compatible}")
    print(f"s67-delta: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
