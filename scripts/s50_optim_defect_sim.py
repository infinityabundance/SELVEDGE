#!/usr/bin/env python3
"""S50: reproduce the Adam8bit second-moment defect from the optimiser's own arithmetic.

The pre-S50 `Adam8bit` stores both moments as per-256-element linear absmax int8.
For the second moment that is catastrophic: the code spacing inside a block is
`absmax_v / 127`, so any squared gradient below about `absmax_v / 254` rounds to
zero. This script transcribes the step arithmetic of
`oracle/frontier/optim8bit.py` at revision `LEGACY_REVISION` *exactly* (same
operations in the same order) and compares it with fp32 `torch.optim.AdamW` on
the same synthetic gradient stream.

Two orderings are reported because the claim in review depends on which one is
meant:

* ``exact``   -- the order in the module: dequantise the stored moments, apply
  the current gradient, compute the update from the *unquantised* fp32 moments,
  then quantise for the next step. This is what the shipped code does.
* ``reread``  -- the order a reader assumes: quantise the accumulated moments and
  dequantise before forming the denominator (i.e. treat the stored value as the
  value used). This is the pessimistic case and is reported for comparison only.

Gradient model: per-element scale `s_i ~ lognormal(0, sigma)` fixed across the
run, times a per-step standard normal direction. This is the heterogeneous
within-block regime the review names; a uniform gradient spread cannot expose
the defect because every element then has the same v ratio.

Output: a JSON receipt with a per-step trace and the summary numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import optim8bit  # noqa: E402

LEGACY_REVISION = "e57df2f"
BETAS = (0.9, 0.95)
EPS = 1e-8


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def legacy_updates(parameter: torch.Tensor, grads: torch.Tensor, lr: float,
                   block: int = 256, ordering: str = "exact") -> dict:
    """Transcription of the pre-S50 `Adam8bit.step` moment arithmetic.

    `grads` is `[steps, N]`. The parameter is fp32 in both arms so the only
    difference from `torch.optim.AdamW` is the moment storage. Weight decay is
    zero, matching the arm's registered recipe.
    """
    beta1, beta2 = BETAS
    n = parameter.numel()
    padded = optim8bit._padded(n, block)
    zeros = torch.zeros(padded, dtype=torch.float32)
    q_m, absmax_m = optim8bit.quantize(zeros, block)
    q_v, absmax_v = optim8bit.quantize(zeros, block)
    parameter = parameter.clone()
    trace = []
    for step_index in range(grads.shape[0]):
        g = grads[step_index]
        if g.numel() < padded:
            g = torch.cat([g, g.new_zeros(padded - g.numel())])
        # dequantise what is stored (the shipped code does this first)
        moment = optim8bit.dequantize(q_m, absmax_m, padded, block)
        variance = optim8bit.dequantize(q_v, absmax_v, padded, block)
        if ordering == "reread":
            # The pessimistic reading: the stored quantisation is applied before
            # the denominator is formed.
            pass
        moment.mul_(beta1).add_(g, alpha=1.0 - beta1)
        variance.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        step = step_index + 1
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        if ordering == "reread":
            q_m, absmax_m = optim8bit.quantize(moment, block)
            q_v, absmax_v = optim8bit.quantize(variance, block)
            moment = optim8bit.dequantize(q_m, absmax_m, padded, block)
            variance = optim8bit.dequantize(q_v, absmax_v, padded, block)
        denominator = (variance / bias2).sqrt().add_(EPS)
        update = ((moment / bias1) / denominator)[:n]
        parameter.add_(update, alpha=-lr)
        # zero-v statistics of the state the *next* step will read
        v_next = optim8bit.dequantize(q_v, absmax_v, padded, block)[:n]
        m_next = optim8bit.dequantize(q_m, absmax_m, padded, block)[:n]
        v_used = variance[:n]
        trace.append({
            "step": step,
            "v_used_zero": int((v_used == 0).sum()),
            "v_used_zero_with_m": int(((v_used == 0) & (moment[:n] != 0)).sum()),
            "v_stored_zero": int((v_next == 0).sum()),
            "v_stored_zero_with_m": int(((v_next == 0) & (m_next != 0)).sum()),
            "max_abs_update": float(update.abs().max()),
            "max_abs_update_over_lr": float(update.abs().max()) / lr,
            "finite": bool(torch.isfinite(update).all()),
        })
        if ordering == "exact":
            q_m, absmax_m = optim8bit.quantize(moment, block)
            q_v, absmax_v = optim8bit.quantize(variance, block)
    return {"parameter": parameter, "trace": trace}


def reference_updates(parameter: torch.Tensor, grads: torch.Tensor,
                      lr: float) -> dict:
    """fp32 AdamW on the same stream, recording the raw update (before lr)."""
    beta1, beta2 = BETAS
    parameter = parameter.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([parameter], lr=lr, betas=BETAS,
                                  eps=EPS, weight_decay=0.0)
    trace = []
    for step_index in range(grads.shape[0]):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = grads[step_index].clone()
        optimizer.step()
        state = optimizer.state[parameter]
        step = state["step"].item()
        moment = state["exp_avg"]
        variance = state["exp_avg_sq"]
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        update = (moment / bias1) / ((variance / bias2).sqrt() + EPS)
        trace.append({
            "step": step,
            "max_abs_update": float(update.abs().max()),
            "max_abs_update_over_lr": float(update.abs().max()) / lr,
            "finite": bool(torch.isfinite(update).all()),
        })
    return {"parameter": parameter.detach(), "trace": trace}


def run(sigma: float, steps: int, elements: int, block: int, lr: float,
        seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(elements, generator=generator) * 0.02
    # fixed per-element scale, lognormal inside every block
    scales = torch.exp(torch.randn(elements, generator=generator) * sigma)
    directions = torch.randn(steps, elements, generator=generator)
    grads = directions * scales[None, :]
    arms = {}
    for ordering in ("exact", "reread"):
        result = legacy_updates(initial, grads, lr, block, ordering)
        reference = reference_updates(initial, grads, lr)
        steps_out = []
        for legacy, ref in zip(result["trace"], reference["trace"]):
            ratio = (legacy["max_abs_update"] / ref["max_abs_update"]
                     if ref["max_abs_update"] > 0 else math.inf)
            steps_out.append({**legacy, **{
                "reference_max_abs_update": ref["max_abs_update"],
                "max_update_ratio_to_adamw": ratio,
            }})
        final_legacy = result["parameter"]
        final_ref = reference["parameter"]
        divergence = float((final_legacy - final_ref).norm()
                           / final_ref.norm().clamp_min(1e-30))
        arms[ordering] = {
            "steps": steps_out,
            "final_relative_parameter_error": divergence,
            "any_non_finite": not all(item["finite"] for item in steps_out),
            "max_fraction_v_used_zero_with_m": max(
                item["v_used_zero_with_m"] for item in steps_out) / elements,
            "max_fraction_v_stored_zero_with_m": max(
                item["v_stored_zero_with_m"] for item in steps_out) / elements,
            "max_abs_update": max(item["max_abs_update"] for item in steps_out),
            "max_abs_update_over_lr": max(
                item["max_abs_update_over_lr"] for item in steps_out),
            "max_update_ratio_to_adamw": max(
                item["max_update_ratio_to_adamw"] for item in steps_out),
            "step_of_max_update": max(
                steps_out, key=lambda item: item["max_abs_update"])["step"],
            "final_fraction_v_stored_zero": (
                steps_out[-1]["v_stored_zero"] / elements),
            "final_fraction_v_stored_zero_with_m": (
                steps_out[-1]["v_stored_zero_with_m"] / elements),
        }
    return {
        "sigma": sigma,
        "steps": steps,
        "elements": elements,
        "block": block,
        "lr": lr,
        "seed": seed,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigmas", default="1.0")
    args = parser.parse_args()

    torch.set_num_threads(1)
    runs = [run(float(sigma), args.steps, args.elements, args.block, args.lr,
                args.seed) for sigma in args.sigmas.split(",")]
    receipt = {
        "schema": "selvedge/s50-optim-defect-sim/v1",
        "kind": "s50-optim-defect-simulation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _git_revision(),
        "defective_revision": LEGACY_REVISION,
        "module": "oracle/frontier/optim8bit.py",
        "module_sha256": _sha256(
            os.path.join(ROOT, "oracle", "frontier", "optim8bit.py")),
        "script_sha256": _sha256(os.path.abspath(__file__)),
        "torch": torch.__version__,
        "arithmetic": {
            "betas": list(BETAS),
            "eps": EPS,
            "block_elements": args.block,
            "second_moment": "linear absmax int8 per block (pre-S50)",
            "first_moment": "linear absmax int8 per block",
            "ordering_exact": ("dequantise stored moments; add current gradient; "
                               "form the update from the fp32 sums; quantise after"),
            "ordering_reread": ("quantise after accumulation and dequantise "
                                "before forming the denominator"),
        },
        "gradient_model": (
            "per-element scale lognormal(0, sigma) fixed across the run, times a "
            "per-step standard-normal direction; parameter fp32 in both arms so "
            "the only difference is moment storage"),
        "runs": runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    # a compact human line for the court log
    for item in runs:
        exact = item["arms"]["exact"]
        reread = item["arms"]["reread"]
        print(f"sigma={item['sigma']} steps={item['steps']} "
              f"exact: v0+m={exact['max_fraction_v_stored_zero_with_m']:.3f} "
              f"max|u|={exact['max_abs_update']:.3e} "
              f"max|u|/lr={exact['max_abs_update_over_lr']:.3e} "
              f"ratio={exact['max_update_ratio_to_adamw']:.3e} "
              f"step={exact['step_of_max_update']} "
              f"parerr={exact['final_relative_parameter_error']:.3e}; "
              f"reread: v0+m={reread['max_fraction_v_stored_zero_with_m']:.3f} "
              f"max|u|={reread['max_abs_update']:.3e} "
              f"ratio={reread['max_update_ratio_to_adamw']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
