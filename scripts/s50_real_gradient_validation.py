#!/usr/bin/env python3
"""S51: validate the int8 first moment on real Qwen3.5-0.8B gradients.

The S50 correction fixed the second moment (linear int8 -> bfloat16) and left
the first moment as a per-256-element linear absmax int8 code. On synthetic
lognormal gradients that code leaves a bounded direction error and a 0.88%
parameter error after 50 steps
(`results/stage2/s50-parity-components-20261117T203840Z.json`). Synthetic
gradients are not real gradients, so this harness replays a fixture captured
from the registered math stream:

1. `scripts/s50_capture_real_gradients.py` runs one or two forward/backward
   steps on the pinned student/teacher and writes the post-clip gradient of a
   handful of parameter tensors plus the parameter values (GPU, queued).
2. this script runs 50 optimiser steps of the shipped int8-first-moment
   `Adam8bit`, the optional bfloat16-first-moment variant, and fp32 `AdamW` on
   those exact tensors, CPU-side.

The captured gradients are replayed cyclically because the capture is cheap; a
real capture of 50 distinct steps is a later GPU option. The comparison is
therefore a property of the *gradient spectrum* the optimiser sees, not of the
data order.

Decision thresholds (from the synthetic worst case, with headroom):

* worst per-step relative L2 update error <= 0.40 (synthetic sigma=1.0: 0.2789)
* max |update| ratio against AdamW <= 3.0 (synthetic sigma=1.0: 2.46)
* final parameter relative error at the validated lr <= 0.02 (synthetic at
  lr 1e-4: 0.0088; the bar runs at lr 1e-5, where the same direction error
  moves the parameter ~10x less)

If the int8 arm exceeds any threshold and the bfloat16 arm does not, the first
moment must move to bfloat16; that is the registered switch, not a judgement
call. If both arms fail, the first moment is not the cause and a storage switch
must not be used to excuse it.

Usage (CPU; run through `scripts/court python`):

    python3 scripts/s50_real_gradient_validation.py \
        --fixture results/artifacts/stage2/s50-real-gradients-<stamp>/fixture.pt \
        --out results/stage2/s50-real-gradient-validation-<stamp>.json

The fixture is a `.pt` file, which `.gitignore` excludes; the JSON receipt is
the tracked evidence and records the fixture's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
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

FIXTURE_SCHEMA = "selvedge/s50-real-gradient-fixture/v1"
RECEIPT_SCHEMA = "selvedge/s50-real-gradient-validation/v1"
BETAS = (0.9, 0.95)
EPS = 1e-8
BLOCK = 256
DEFAULT_LR = 1e-5          # the corrected 20M bar's rate
DEFAULT_STEPS = 50
#: Full-arm trainable elements and the measured corrected peak (S50 receipt
#: `results/stage2/s50-revalidation-20261117T210326Z.json`: 11.885 GiB allocated).
TRAINABLE_ELEMENTS = 498_113_344
MEASURED_INT8_PEAK_GIB = 11.885
INT8_MOMENT_BYTES_PER_ELEMENT = 3.015625
BF16_MOMENT_BYTES_PER_ELEMENT = 4.0
#: Headroom over the synthetic sigma=1.0 measurements (see module docstring).
TOLERANCE = {
    "worst_relative_l2": 0.40,
    "max_update_ratio": 3.0,
    "final_parameter_relative_error": 0.02,
}
TOLERANCE_BASIS = (
    "synthetic per-element lognormal(0, 1) gradients, 50 steps, "
    "results/stage2/s50-parity-components-20261117T203840Z.json: shipped int8 "
    "first moment worst relative L2 0.2789, max update ratio 2.46, final "
    "parameter error 0.008828 at lr 1e-4; thresholds add headroom")
STORAGES = ("int8", "bfloat16")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_fixture(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise SystemExit(f"{path}: not a {FIXTURE_SCHEMA} fixture")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise SystemExit(f"{path}: fixture has no tensors")
    for name, entry in tensors.items():
        parameter = entry.get("parameter")
        grads = entry.get("grads")
        if not isinstance(parameter, torch.Tensor) or not isinstance(grads, list) \
                or not grads or not all(isinstance(g, torch.Tensor) for g in grads):
            raise SystemExit(f"{path}: tensor {name!r} is malformed")
        if parameter.numel() != grads[0].numel() or any(
                g.numel() != parameter.numel() for g in grads):
            raise SystemExit(f"{path}: tensor {name!r} has inconsistent sizes")
    return payload


def _step_updates(initial: torch.Tensor, gradients: list[torch.Tensor],
                  steps: int, storage: str, lr: float, betas, eps: float,
                  block: int) -> dict:
    """Replay one tensor through one arm and AdamW, keeping only scalars."""
    # The fixture stores the captured tensor at its captured shape (optionally
    # row-capped); the optimiser arms work on one flat vector, so flatten here
    # rather than assuming the 1-D case (S55 fix).
    initial = initial.reshape(-1).contiguous()
    reference = nn.Parameter(initial.clone())
    adamw = torch.optim.AdamW([reference], lr=lr, betas=betas, eps=eps,
                              weight_decay=0.0)
    arm = nn.Parameter(initial.clone())
    optimizer = optim8bit.Adam8bit(
        [arm], lr=lr, betas=betas, eps=eps, weight_decay=0.0, block=block,
        weight_rounding="nearest", first_moment_storage=storage)
    diff_sq = [0.0] * steps
    reference_sq = [0.0] * steps
    max_ratio = 0.0
    for step in range(steps):
        gradient = gradients[step % len(gradients)].to(torch.float32).reshape(-1)
        # AdamW reference
        adamw.zero_grad(set_to_none=True)
        before_reference = reference.detach().clone()
        reference.grad = gradient.clone()
        adamw.step()
        reference_update = (reference.detach() - before_reference) / lr
        # quantized arm
        optimizer.zero_grad(set_to_none=True)
        before_arm = arm.detach().clone()
        arm.grad = gradient.clone()
        optimizer.step()
        arm_update = (arm.detach() - before_arm) / lr
        difference = (arm_update - reference_update).reshape(-1)
        diff_sq[step] = float(difference.pow(2).sum())
        reference_sq[step] = float(reference_update.reshape(-1).pow(2).sum())
        denominator = float(reference_update.abs().max())
        numerator = float(arm_update.abs().max())
        max_ratio = max(max_ratio, numerator / max(denominator, 1e-30))
    final_difference = (arm.detach() - reference.detach()).reshape(-1)
    return {
        "diff_sq": diff_sq,
        "reference_sq": reference_sq,
        "max_update_ratio": max_ratio,
        "final_diff_sq": float(final_difference.pow(2).sum()),
        "final_reference_sq": float(reference.detach().reshape(-1).pow(2).sum()),
        "moment_bytes_per_element": optimizer.moment_bytes_per_element(),
    }


def _evaluate_arm(tensors: dict, steps: int, storage: str, lr: float,
                  betas, eps: float, block: int) -> dict:
    total_diff = [0.0] * steps
    total_reference = [0.0] * steps
    max_ratio = 0.0
    final_diff = 0.0
    final_reference = 0.0
    moment_bytes = 0.0
    for name, entry in sorted(tensors.items()):
        result = _step_updates(
            entry["parameter"], entry["grads"], steps, storage, lr, betas,
            eps, block)
        for step in range(steps):
            total_diff[step] += result["diff_sq"][step]
            total_reference[step] += result["reference_sq"][step]
        max_ratio = max(max_ratio, result["max_update_ratio"])
        final_diff += result["final_diff_sq"]
        final_reference += result["final_reference_sq"]
        moment_bytes = max(moment_bytes, result["moment_bytes_per_element"])
    relative = [
        (value / reference) ** 0.5 if reference > 0 else 0.0
        for value, reference in zip(total_diff, total_reference)]
    worst = max(range(steps), key=relative.__getitem__)
    return {
        "first_moment_storage": storage,
        "per_step_relative_l2": relative,
        "worst_relative_l2": relative[worst],
        "worst_step": worst + 1,
        "max_update_ratio": max_ratio,
        "final_parameter_relative_error": (
            (final_diff / final_reference) ** 0.5 if final_reference > 0 else 0.0),
        "moment_bytes_per_element": moment_bytes,
    }


def decide(arms: dict) -> dict:
    def failures(arm: dict) -> list[str]:
        return [f"{name}={arm[name]:.6g} > {limit:.6g}"
                for name, limit in TOLERANCE.items() if arm[name] > limit]

    int8_failures = failures(arms["int8"])
    bfloat16_failures = failures(arms["bfloat16"])
    if not int8_failures:
        status = "retain-int8-first-moment"
        action = ("the int8 first moment stays the default; no storage switch "
                  "is authorised by this fixture")
    elif not bfloat16_failures:
        status = "switch-first-moment-to-bfloat16"
        action = ("the int8 first moment fails the registered tolerance and the "
                  "bfloat16 first moment passes it; select "
                  "first_moment_storage='bfloat16' for the bar")
    else:
        status = "first-moment-storage-not-the-cause"
        action = ("both first-moment layouts fail; do not switch storage to "
                  "excuse a gradient or step-size regime")
    return {
        "status": status,
        "action": action,
        "int8_failures": int8_failures,
        "bfloat16_failures": bfloat16_failures,
        "tolerance": dict(TOLERANCE),
        "tolerance_basis": TOLERANCE_BASIS,
    }


def evaluate(fixture: dict, steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
             betas=BETAS, eps: float = EPS, block: int = BLOCK) -> dict:
    tensors = fixture["tensors"]
    arms = {
        storage: _evaluate_arm(tensors, steps, storage, lr, betas, eps, block)
        for storage in STORAGES
    }
    decision = decide(arms)
    extra_gib = ((BF16_MOMENT_BYTES_PER_ELEMENT - INT8_MOMENT_BYTES_PER_ELEMENT)
                 * TRAINABLE_ELEMENTS / (1 << 30))
    return {
        "arms": arms,
        "decision": decision,
        "memory": {
            "int8_moment_bytes_per_element": INT8_MOMENT_BYTES_PER_ELEMENT,
            "bfloat16_moment_bytes_per_element": BF16_MOMENT_BYTES_PER_ELEMENT,
            "measured_int8_projected_peak_gib": MEASURED_INT8_PEAK_GIB,
            "bfloat16_projected_peak_gib": MEASURED_INT8_PEAK_GIB + extra_gib,
            "vram_limit_gib": 12.5,
            "bfloat16_fits": MEASURED_INT8_PEAK_GIB + extra_gib < 12.5,
        },
    }


def _summary(path: str) -> dict:
    payload = load_fixture(path)
    metadata = payload.get("metadata") or {}
    tensors = payload["tensors"]
    return {
        "path": path,
        "sha256": _sha256(path),
        "schema": payload["schema"],
        "capture": metadata,
        "tensor_names": sorted(tensors),
        "tensor_count": len(tensors),
        "elements": int(sum(entry["parameter"].numel()
                            for entry in tensors.values())),
        "grad_steps": min(len(entry["grads"]) for entry in tensors.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True,
                        help="fixture written by s50_capture_real_gradients.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads; one keeps the receipt reproducible")
    parser.add_argument("--synthetic", action="store_true",
                        help="label the fixture as synthetic (harness smoke only)")
    args = parser.parse_args(argv)

    if args.steps < 2:
        raise SystemExit("s50-real-grad: at least two steps are required")
    torch.set_num_threads(max(1, args.threads))
    summary = _summary(args.fixture)
    fixture = load_fixture(args.fixture)
    result = evaluate(fixture, steps=args.steps, lr=args.lr, block=args.block)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "kind": "s50-real-gradient-validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _git_revision(),
        "fixture": {**summary, "synthetic": bool(args.synthetic)},
        "steps": args.steps,
        "lr": args.lr,
        "betas": list(BETAS),
        "eps": EPS,
        "block": args.block,
        "parameter_dtype": "float32 (the only difference from AdamW is moment storage)",
        "weight_rounding": "nearest (fp32 parameters; isolates moment storage)",
        "replay": ("captured gradients are replayed cyclically for the requested "
                   "step count; the comparison is over the captured spectrum, "
                   "not the data order"),
        **result,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for storage in STORAGES:
        arm = result["arms"][storage]
        print(f"s50-real-grad: {storage}: worst relL2={arm['worst_relative_l2']:.4f} "
              f"max|u|ratio={arm['max_update_ratio']:.3f} "
              f"final parerr={arm['final_parameter_relative_error']:.5f} "
              f"({arm['moment_bytes_per_element']:.4f} B/element)")
    print(f"s50-real-grad: decision {result['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
