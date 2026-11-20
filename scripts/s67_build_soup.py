#!/usr/bin/env python3
"""S67: build the damped model soup from the three seed deltas.

    delta_bar = mean_s (theta_s - theta_0)
    theta_soup(alpha) = theta_0 + alpha * delta_bar

Only the 498,113,344 trainable text-tower elements exported by the full arm are
mixed, in float32, then rounded back to the base tensor's dtype. The tied
embedding/output projection is frozen and identical at every endpoint, so it is
inherited from the post-trained snapshot; the certified active text path stays
752,393,024 elements.

Checkpoints are written with the S60/S65 payload schema and the S60 filename
(`interp-<alpha>-arm.pt`) so the existing calibration, MMLU and evaluation
harnesses load them unchanged. `alpha = 0` is the post-trained endpoint byte for
byte in value; it is written so a calibration grid is self-contained.

This script reads no test split and performs no selection.

Usage (CPU, oracle court):
    SELVEDGE_MEM=24g scripts/court oracle -- \
      python3 scripts/s67_build_soup.py \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --seed results/stage2/runs/s58-stage2a/ckpt-004000.pt \
        --seed results/stage2/runs/s66-stage2a-seed1-guard20/ckpt-004000.pt \
        --seed results/stage2/runs/s66-stage2a-seed2-guard20/ckpt-004000.pt \
        --grid 0,0.25,0.375,0.5,0.625 \
        --out-dir results/stage2/s67-soup \
        --out results/stage2/s67-soup/soup-receipt.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from s60_interpolate import (  # noqa: E402
    _base_arm_state, parse_grid, sha256_file, torch_load, torch_save,
)

SCHEMA = "selvedge/s67-soup/v1"


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seed", action="append", required=True)
    parser.add_argument("--grid", default="0,0.25,0.375,0.5,0.625")
    parser.add_argument("--out-dir", default="results/stage2/s67-soup")
    parser.add_argument("--out", default="results/stage2/s67-soup/soup-receipt.json")
    parser.add_argument("--diagnostics", default=None,
                        help="delta diagnostics receipt; required to be green")
    args = parser.parse_args(argv)

    if len(args.seed) < 2:
        raise SystemExit("s67-soup: at least two seed checkpoints are required")

    if args.diagnostics:
        diagnostics = json.load(open(os.path.join(ROOT, args.diagnostics),
                                     encoding="utf-8"))
        if not (diagnostics.get("decision") or {}).get("directions_compatible"):
            raise SystemExit(
                "s67-soup: the delta diagnostics do not declare compatible "
                "directions; the registered fallback is the seed-0 checkpoint, "
                "not a soup")

    import torch
    from transformers import AutoModelForCausalLM

    grid = parse_grid(args.grid)
    checkpoints = [torch_load(os.path.join(ROOT, path)) for path in args.seed]
    for path, payload in zip(args.seed, checkpoints):
        if payload.get("arm") != "full":
            raise SystemExit(f"s67-soup: {path} is not a full arm")
    keys = sorted(checkpoints[0]["arm_state"])
    for path, payload in zip(args.seed[1:], checkpoints[1:]):
        if sorted(payload["arm_state"]) != keys:
            raise SystemExit(f"s67-soup: {path} key set differs")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False)
    base_state = _base_arm_state(model, checkpoints[0]["arm_state"])

    # delta_bar per parameter, float32, accumulated without a second copy of
    # the three deltas: sum then divide.
    delta_bar = {}
    for key in keys:
        base = base_state[key]
        if not hasattr(base, "dtype") or not torch.is_floating_point(base):
            delta_bar[key] = None
            continue
        total = torch.zeros_like(base, dtype=torch.float32)
        for payload in checkpoints:
            total += payload["arm_state"][key].float() - base.float()
        delta_bar[key] = total / len(checkpoints)
        del total

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    artifacts = []
    for alpha in grid:
        mixed = {}
        for key in keys:
            base = base_state[key]
            delta = delta_bar[key]
            if delta is None:
                mixed[key] = base.clone()
            elif alpha == 0.0:
                mixed[key] = base.clone()
            else:
                mixed[key] = (base.float() + alpha * delta).to(base.dtype)
        payload = {
            "schema": "selvedge/stage2-ckpt/v1",
            "interpolation": {
                "kind": "seed-delta-soup",
                "coefficient": alpha,
                "base": "post-trained 0.8B (untrained endpoint)",
                "candidate": "mean of the three S66 seed deltas",
                "seeds": [
                    {"checkpoint": path, "checkpoint_sha256": sha256_file(
                        os.path.join(ROOT, path))}
                    for path in args.seed],
            },
            "step": checkpoints[0].get("step"),
            "config_sha256": checkpoints[0].get("config_sha256"),
            "arm": "full",
            "arm_state": mixed,
        }
        path = os.path.join(out_dir, f"interp-{alpha:.3f}-arm.pt")
        torch_save(payload, path)
        artifacts.append({
            "coefficient": alpha,
            "path": os.path.relpath(path, ROOT),
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
        print(f"s67-soup: alpha={alpha:.3f} -> {os.path.relpath(path, ROOT)} "
              f"sha256={artifacts[-1]['sha256']}", flush=True)
        # Release the mixed payload before the next alpha.
        del payload, mixed

    receipt = {
        "schema": SCHEMA,
        "kind": "seed-delta-soup",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "formula": "theta_soup(alpha) = theta_0 + alpha * mean_s(theta_s - theta_0)",
        "model_dir": args.model_dir,
        "base": {
            "endpoint": "post-trained Qwen3.5-0.8B (untrained endpoint)",
            "config_sha256": checkpoints[0].get("config_sha256"),
        },
        "seeds": [
            {"checkpoint": path,
             "checkpoint_sha256": sha256_file(os.path.join(ROOT, path))}
            for path in args.seed],
        "diagnostics": args.diagnostics,
        "grid": grid,
        "artifacts": artifacts,
        "state_elements": sum(
            int(value.numel()) for value in delta_bar.values()
            if value is not None),
        "frozen_elements_inherited": (
            "tied embedding/output projection is not mixed; it is inherited "
            "from the post-trained snapshot"),
        "claims": {
            "reads_no_test_split": True,
            "selection_performed": False,
            "arithmetic": "float32 sum of deltas divided by seed count; "
                          "base + alpha * delta_bar rounded to bf16",
        },
    }
    write_json(os.path.join(ROOT, args.out), receipt)
    print(f"s67-soup: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
