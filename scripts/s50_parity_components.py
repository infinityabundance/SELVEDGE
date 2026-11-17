#!/usr/bin/env python3
"""S50: where the remaining AdamW parity error comes from.

Compares three moment-storage layouts on the same heterogeneous gradient
streams (per-element scale lognormal(0, sigma) inside 256-element blocks, 50
steps, fp32 parameters, lr 1e-4, betas 0.9/0.95, eps 1e-8):

* ``shipped``   -- int8 first moment (linear absmax) + bf16 second moment, the
  S50 correction in `oracle/frontier/optim8bit.py`;
* ``exact_m``   -- exact fp32 first moment + bf16 second moment, obtained by
  monkeypatching `quantize` for the first moment only (diagnostic, not a
  supported optimiser);
* ``legacy``    -- int8 first moment + linear absmax int8 second moment, the
  pre-S50 layout, transcribed from the same module's step arithmetic.

The point of the receipt is the attribution: the catastrophic error is the
linear int8 second moment; what remains in the shipped layout is the bounded
absolute quantisation of the first moment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import optim8bit  # noqa: E402

BETAS = (0.9, 0.95)
EPS = 1e-8
BLOCK = 256
LR = 1e-4


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_sim():
    path = os.path.join(HERE, "s50_optim_defect_sim.py")
    spec = importlib.util.spec_from_file_location("s50_optim_defect_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(steps: int, elements: int, sigma: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(elements, generator=generator) * 0.02
    scales = torch.exp(sigma * torch.randn(elements, generator=generator))
    directions = torch.randn(steps, elements, generator=generator)
    return initial, directions * scales[None, :]


def _reference(initial: torch.Tensor, grads: torch.Tensor) -> tuple[torch.Tensor, list]:
    parameter = nn.Parameter(initial.clone().requires_grad_(True))
    optimizer = torch.optim.AdamW([parameter], lr=LR, betas=BETAS, eps=EPS,
                                  weight_decay=0.0)
    updates = []
    for index in range(grads.shape[0]):
        optimizer.zero_grad(set_to_none=True)
        before = parameter.detach().clone()
        parameter.grad = grads[index].clone()
        optimizer.step()
        updates.append((parameter.detach() - before) / LR)
    return parameter.detach(), updates


def _shipped(initial, grads):
    parameter = nn.Parameter(initial.clone())
    optimizer = optim8bit.Adam8bit([parameter], lr=LR, betas=BETAS, eps=EPS,
                                   block=BLOCK, weight_rounding="nearest")
    updates = []
    for index in range(grads.shape[0]):
        optimizer.zero_grad(set_to_none=True)
        before = parameter.detach().clone()
        parameter.grad = grads[index].clone()
        optimizer.step()
        updates.append((parameter.detach() - before) / LR)
    return parameter.detach(), updates, optimizer


def _exact_m(initial, grads):
    """Second moment bf16, first moment exact (diagnostic monkeypatch)."""
    original = optim8bit.quantize

    def lossless(flat, block=BLOCK):
        padded = optim8bit._padded(flat.numel(), block)
        if padded != flat.numel():
            flat = torch.cat([flat, flat.new_zeros(padded - flat.numel())])
        return flat.clone(), torch.full((padded // block,), 127.0)

    optim8bit.quantize = lossless
    try:
        return _shipped(initial, grads)
    finally:
        optim8bit.quantize = original


def _legacy(initial, grads):
    sim = _load_sim()
    result = sim.legacy_updates(initial, grads, LR, BLOCK, "exact")
    return result["parameter"]


def _compare(initial, grads, updates, reference_updates):
    relative = []
    ratios = []
    for ours, theirs in zip(updates, reference_updates):
        relative.append(float((ours - theirs).norm()
                              / theirs.norm().clamp_min(1e-30)))
        ratios.append(float(ours.abs().max() / theirs.abs().max()))
    return {
        "per_step_relative_l2": relative,
        "worst_relative_l2": max(relative),
        "worst_step": int(max(range(len(relative)), key=relative.__getitem__)) + 1,
        "max_update_ratio": max(ratios),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--sigmas", default="0.5,1.0")
    args = parser.parse_args()

    torch.set_num_threads(1)
    runs = {}
    for sigma in (float(value) for value in args.sigmas.split(",")):
        initial, grads = _fixture(args.steps, args.elements, sigma, seed=0)
        reference_parameter, reference_updates = _reference(initial, grads)
        shipped_parameter, shipped_updates, optimizer = _shipped(initial, grads)
        exact_parameter, exact_updates, _ = _exact_m(initial, grads)
        legacy_parameter = _legacy(initial, grads)
        runs[str(sigma)] = {
            "shipped": {
                **_compare(initial, grads, shipped_updates, reference_updates),
                "final_parameter_relative_error": float(
                    (shipped_parameter - reference_parameter).norm()
                    / reference_parameter.norm()),
                "moment_bytes_per_element": optimizer.moment_bytes_per_element(),
            },
            "exact_m": {
                **_compare(initial, grads, exact_updates, reference_updates),
                "final_parameter_relative_error": float(
                    (exact_parameter - reference_parameter).norm()
                    / reference_parameter.norm()),
            },
            "legacy": {
                "final_parameter_relative_error": float(
                    (legacy_parameter - reference_parameter).norm()
                    / reference_parameter.norm()),
            },
        }
    receipt = {
        "schema": "selvedge/s50-parity-components/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _git_revision(),
        "steps": args.steps,
        "elements": args.elements,
        "lr": LR,
        "betas": list(BETAS),
        "eps": EPS,
        "block": BLOCK,
        "fixture": ("per-element scale lognormal(0, sigma) fixed across the run, "
                    "times per-step standard-normal direction; fp32 parameters"),
        "runs": runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for sigma, arms in runs.items():
        print(f"sigma={sigma} shipped worst={arms['shipped']['worst_relative_l2']:.4f} "
              f"exact_m worst={arms['exact_m']['worst_relative_l2']:.4f} "
              f"legacy final_parerr={arms['legacy']['final_parameter_relative_error']:.3e} "
              f"shipped final_parerr={arms['shipped']['final_parameter_relative_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
