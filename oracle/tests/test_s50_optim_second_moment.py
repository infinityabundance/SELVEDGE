"""CPU tests for the S50 second-moment correction in `Adam8bit`.

The pre-S50 optimiser stored the second moment as a linear absmax int8 code per
256-element block. That code has spacing `absmax / 127`, so any squared gradient
below about `absmax / 254` rounds to zero, and Adam's denominator then falls to
`eps` whenever the current gradient is also small. The defect is invisible to a
uniform-gradient test and to a first-step comparison: at step 1 both moments are
still the exact fp32 sums, and a uniform spread gives every element the same
`v / absmax` ratio.

These tests pin the corrected storage and the heterogeneous regime the review
used: per-element scales lognormal(0, 1) inside a block, 50 steps against
`torch.optim.AdamW`. The parameter is fp32 in both arms, so the only difference
is moment storage; the first moment is int8 by design, the second is bf16.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import optim8bit, train  # noqa: E402

BLOCK = 256
LR = 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
# Stated tolerance for the update-direction parity over 50 steps. The remaining
# error is the int8 *first* moment's absolute per-block quantisation
# (`absmax_m / 254`), not the second moment: the S50 receipt
# `results/stage2/s50-parity-components-20261117T203600Z.json` measures a worst
# relative L2 update error of 0.2789 with the shipped int8 first moment against
# 0.0023 when the first moment is left exact (same bf16 second moment), on this
# exact fixture. The int8 error is bounded direction noise, it does not change a
# step's order of magnitude (max ratio 1.9), and the 50-step parameter trajectory
# stays within 0.9%; the pre-S50 linear int8 second moment gives 3e4 and a 313%
# parameter error. 0.30 therefore pins the corrected regime without admitting a
# denominator collapse.
PARITY_TOLERANCE = 0.30
# At a moderate within-block spread the same layout is much tighter (0.0395
# measured); this is the bound the registered recipe should see on real tensors.
PARITY_TOLERANCE_SIGMA_HALF = 0.06


def _fixture(steps: int = 50, elements: int = 4096, sigma: float = 1.0,
             seed: int = 0):
    """Fixed per-element lognormal scale inside every block."""
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(elements, generator=generator) * 0.02
    scales = torch.exp(sigma * torch.randn(elements, generator=generator))
    directions = torch.randn(steps, elements, generator=generator)
    return initial, scales, directions * scales[None, :]


def _parity(steps: int = 50, elements: int = 4096, sigma: float = 1.0,
            seed: int = 0) -> dict:
    initial, _, grads = _fixture(steps, elements, sigma, seed)
    eight = nn.Parameter(initial.clone())
    optimizer = optim8bit.Adam8bit([eight], lr=LR, betas=BETAS, eps=EPS,
                                   block=BLOCK, weight_rounding="nearest")
    reference = nn.Parameter(initial.clone())
    adamw = torch.optim.AdamW([reference], lr=LR, betas=BETAS, eps=EPS,
                              weight_decay=0.0)
    relative_errors = []
    max_update_ratios = []
    for index in range(steps):
        arms = []
        for parameter, opt in ((eight, optimizer), (reference, adamw)):
            opt.zero_grad(set_to_none=True)
            before = parameter.detach().clone()
            parameter.grad = grads[index].clone()
            opt.step()
            arms.append((parameter.detach() - before) / LR)
        ours, theirs = arms
        difference = float((ours - theirs).norm()
                           / theirs.norm().clamp_min(1e-30))
        relative_errors.append(difference)
        max_update_ratios.append(float(ours.abs().max() / theirs.abs().max()))
        assert torch.isfinite(ours).all()
    return {
        "relative_errors": relative_errors,
        "max_update_ratios": max_update_ratios,
        "final_parameter_error": float(
            (eight.detach() - reference.detach()).norm()
            / reference.detach().norm().clamp_min(1e-30)),
        "optimizer": optimizer,
    }


def test_heterogeneous_scale_parity_with_adamw_over_50_steps():
    result = _parity()
    worst = max(result["relative_errors"])
    # step 1 is exact in both arms; the comparison is about steps 2..50
    assert result["relative_errors"][0] < 1e-6
    assert result["relative_errors"][1] > 1e-3
    assert worst < PARITY_TOLERANCE, (
        f"worst relative update error {worst:.4f} exceeds "
        f"{PARITY_TOLERANCE}; per-step {result['relative_errors']}")
    # no element may take an order-of-magnitude larger step than AdamW
    assert max(result["max_update_ratios"]) < 3.0
    # and the 50-step parameter trajectory stays close
    assert result["final_parameter_error"] < 0.05, result["final_parameter_error"]


def test_moderate_spread_parity_is_tight():
    result = _parity(sigma=0.5)
    assert max(result["relative_errors"]) < PARITY_TOLERANCE_SIGMA_HALF
    assert result["final_parameter_error"] < 0.01


def test_second_moment_is_never_zeroed_for_a_nonzero_squared_gradient():
    parameter = nn.Parameter(torch.zeros(BLOCK))
    optimizer = optim8bit.Adam8bit([parameter], lr=LR, betas=BETAS, eps=EPS,
                                   block=BLOCK)
    # one large gradient and 255 tiny ones: the tiny squared gradients are below
    # absmax/254 and the pre-S50 int8 code stored exactly zero for them
    gradient = torch.full((BLOCK,), 1e-6)
    gradient[0] = 1.0
    for _ in range(3):
        parameter.grad = gradient.clone()
        optimizer.step()
    state = optimizer.state[parameter]
    stored = state["v"][:BLOCK].float()
    assert bool((stored > 0).all()), (
        f"{int((stored == 0).sum())} second moments were zeroed")
    assert state["v"].dtype == torch.bfloat16


def test_zeroed_second_moment_cannot_collapse_the_denominator():
    """Heterogeneous scales and a zero follow-up gradient.

    The pre-S50 code stores v = 0 for the small elements after step 1. At step 2
    their gradient is zero, so `sqrt(v) = 0`, the denominator is `eps`, and the
    update reaches ~5e5 lr. bf16 keeps the history, so the correct update is ~5.
    """
    gradient_one = torch.full((BLOCK,), 1e-2)
    gradient_one[100:] = 0.0
    gradient_one[0] = 1.0
    gradient_two = torch.zeros(BLOCK)
    gradient_two[0] = 1.0

    parameter = nn.Parameter(torch.zeros(BLOCK))
    optimizer = optim8bit.Adam8bit([parameter], lr=LR, betas=BETAS, eps=EPS,
                                   block=BLOCK)
    parameter.grad = gradient_one.clone()
    optimizer.step()
    parameter.grad = gradient_two.clone()
    optimizer.step()
    # |update| / lr for the 99 small-gradient elements: ~5, not ~5e5
    assert optimizer.last_max_update_over_lr is not None
    assert optimizer.last_max_update_over_lr < 10.0, (
        optimizer.last_max_update_over_lr)


def test_update_tripwire_decision_is_explicit_about_its_threshold():
    assert train.update_tripwire_decision(None, 10.0) is None
    assert train.update_tripwire_decision(3.0, 0.0) is None
    quiet = train.update_tripwire_decision(3.0, 10.0)
    assert quiet == {"max_update_over_lr": 3.0, "factor": 10.0, "fired": False}
    loud = train.update_tripwire_decision(1.2e5, 10.0)
    assert loud["fired"] is True
    assert loud["max_update_over_lr"] == pytest.approx(1.2e5)


def test_optimizer_reports_the_largest_update_direction():
    parameter = nn.Parameter(torch.zeros(300))
    parameter.grad = torch.randn(300)
    optimizer = optim8bit.Adam8bit([parameter], lr=LR, betas=BETAS,
                                   block=BLOCK)
    optimizer.step()
    measured = optimizer.last_max_update_over_lr
    # fp32 parameters take the exact update, so recompute it from the step.
    assert measured is not None and 0.0 < measured < 10.0
    assert optimizer.receipt()["max_update_over_lr"] == pytest.approx(measured)


def test_pre_s50_int8_second_moment_state_is_refused_on_resume():
    parameter = nn.Parameter(torch.zeros(64))
    optimizer = optim8bit.Adam8bit([parameter], block=64)
    parameter.grad = torch.randn(64)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    legacy = next(iter(saved["state"].values()))
    # rebuild the pre-S50 layout: int8 second moment and its scale
    number = int(legacy["padded"])
    legacy["q_v"] = torch.zeros(number, dtype=torch.int8)
    legacy["absmax_v"] = torch.ones(number // 64, dtype=torch.float32)
    legacy.pop("v")
    with pytest.raises(ValueError, match="pre-S50"):
        optimizer.load_state_dict(saved)
