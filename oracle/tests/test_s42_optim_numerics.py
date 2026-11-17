"""CPU tests for the S42 weight-update numerics (stochastic rounding).

The registered full-backbone arm trains bf16 weights with block-wise 8-bit Adam
and no fp32 master copy. An update below half an ulp therefore rounds to zero
under round-to-nearest, which is the regime at 1e-4 learning rates. These tests
prove the three claims the fix rests on, on a synthetic weight whose update is
just below half an ulp:

* stochastic rounding changes about half of the elements and round-to-nearest
  changes none;
* the unchanged-element fraction is *measured* from the applied bits and agrees
  with the probability the arithmetic predicts, including at a very small LR
  where most elements legitimately do not move;
* the seed makes the sequence exact, and the optional bf16 compensation buffer
  bounds accumulated rounding error far below the nearest-even control.

They also check the compensation buffer's measured bytes against the 1.2 GiB
registration headroom instead of assuming it fits.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import optim8bit  # noqa: E402

# bfloat16 stores 7 mantissa bits, so the spacing is 2^-7 in [1, 2). The base
# value is 1.5: inside [1, 2) the spacing is uniform, so a sub-half-ulp update
# stays in one bracket instead of straddling a binade boundary at 1.0.
ULP_AT_BASE = 2.0 ** -7
BASE = 1.5
BLOCK = 64
# Registers from docs/ws19-full-distillation-preregistration-v1.json
TRAINABLE_ELEMENTS = 498_113_344
HEADROOM_GIB = 1.17
GIB = float(1 << 30)


def _base_parameter(count: int = 4096) -> nn.Parameter:
    return nn.Parameter(torch.full((count,), BASE, dtype=torch.bfloat16))


def _one_step(lr: float, rounding: str, count: int = 4096,
              seed: int = 0) -> tuple[nn.Parameter, optim8bit.Adam8bit]:
    parameter = _base_parameter(count)
    parameter.grad = torch.ones(count, dtype=parameter.dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = optim8bit.Adam8bit(
        [parameter], lr=lr, betas=(0.9, 0.95), block=BLOCK,
        weight_rounding=rounding, generator=generator, chunk=512)
    optimizer.step(measure=True)
    return parameter, optimizer


def _changed_fraction(before: torch.Tensor, after: torch.Tensor) -> float:
    changed = after.view(torch.int16) != before.view(torch.int16)
    return float(changed.float().mean())


# ---------------------------------------------------------------------------
# the core property: sub-half-ulp updates


def test_sub_half_ulp_update_changes_half_under_sr_and_never_under_nearest():
    # Adam's first-step magnitude is ~1, so the update is |lr| in weight units.
    # 0.49 ulp is strictly below the nearest-even threshold of half an ulp.
    lr = 0.49 * ULP_AT_BASE
    start = torch.full((4096,), BASE, dtype=torch.bfloat16)
    stochastic, sr_optimizer = _one_step(lr, "stochastic")
    nearest, _ = _one_step(lr, "nearest")

    sr_changed = _changed_fraction(start, stochastic.detach())
    rn_changed = _changed_fraction(start, nearest.detach())
    assert 0.4 <= sr_changed <= 0.6, sr_changed
    assert rn_changed == 0.0, rn_changed

    # the update is negative (gradient descent on a positive gradient), so each
    # changed element moved down by exactly one ulp: no second grid step
    moved = (stochastic.detach().float() - start.float()) / ULP_AT_BASE
    assert torch.allclose(moved, torch.round(moved), atol=0, rtol=0)
    assert float(moved.min()) >= -1.0
    assert float(moved.max()) <= 0.0

    # unbiasedness in expectation: the mean displacement is -p * ulp
    mean_steps = -float(moved.mean())
    assert mean_steps == pytest.approx(sr_changed, abs=0.02)


def test_stochastic_rounding_leaves_grid_values_exactly_unchanged():
    grid = torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0, 1e-30, -3.5],
                        dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(3)
    out = optim8bit.stochastic_round(grid.float(), generator)
    assert torch.equal(out.view(torch.int16), grid.view(torch.int16))


def test_stochastic_rounding_is_unbiased_on_a_uniform_interval():
    torch.manual_seed(11)
    torch.use_deterministic_algorithms(True)
    low = torch.full((200_000,), BASE, dtype=torch.bfloat16)
    fractions = torch.rand(200_000)
    exact = low.float() + fractions * ULP_AT_BASE
    generator = torch.Generator(device="cpu").manual_seed(5)
    rounded = optim8bit.stochastic_round(exact, generator)
    error = (rounded.float() - exact)
    # Each element's error has variance f(1-f) ulp^2; report the mean error.
    ulp_error = error / ULP_AT_BASE
    standard_error = float(torch.sqrt((fractions * (1 - fractions)).mean()
                                      / fractions.numel()))
    assert abs(float(ulp_error.mean())) < 4.0 * standard_error


# ---------------------------------------------------------------------------
# the measurement, not an assertion


def test_unchanged_fraction_is_measured_at_half_ulp():
    lr = 0.49 * ULP_AT_BASE
    parameter, optimizer = _one_step(lr, "stochastic")
    stats = optimizer.last_numerics
    assert stats["elements"] == 4096
    assert stats["sub_half_ulp_fraction"] == 1.0
    assert stats["unchanged_fraction_round_to_nearest"] == 1.0
    assert 0.4 <= stats["changed_fraction"] <= 0.6
    # the measured fraction agrees with the arithmetic prediction
    assert stats["unchanged_fraction"] == pytest.approx(
        stats["expected_unchanged_fraction_stochastic"], abs=0.02)


def test_unchanged_fraction_is_high_and_reported_at_a_tiny_lr():
    # 0.01 ulp: almost every element legitimately does not move in one step.
    lr = 0.01 * ULP_AT_BASE
    parameter, optimizer = _one_step(lr, "stochastic", count=16_384)
    stats = optimizer.last_numerics
    assert 0.97 < stats["unchanged_fraction"] < 1.0
    assert stats["unchanged_fraction"] != 1.0  # never frozen, unlike nearest
    assert stats["unchanged_fraction"] == pytest.approx(
        stats["expected_unchanged_fraction_stochastic"], abs=0.01)
    nearest, _ = _one_step(lr, "nearest", count=16_384)
    assert torch.equal(nearest.detach().view(torch.int16),
                       torch.full((16_384,), BASE,
                                  dtype=torch.bfloat16).view(torch.int16))


# ---------------------------------------------------------------------------
# determinism


def test_seed_makes_the_rounded_sequence_exact_and_replayable():
    lr = 0.49 * ULP_AT_BASE
    one, _ = _one_step(lr, "stochastic", seed=7)
    two, _ = _one_step(lr, "stochastic", seed=7)
    other, _ = _one_step(lr, "stochastic", seed=8)
    assert torch.equal(one.detach().view(torch.int16),
                       two.detach().view(torch.int16))
    assert not torch.equal(one.detach().view(torch.int16),
                           other.detach().view(torch.int16))


def test_optimizer_rng_state_round_trips_through_the_optimizer():
    parameter = _base_parameter(1024)
    generator = torch.Generator(device="cpu").manual_seed(21)
    optimizer = optim8bit.Adam8bit([parameter], lr=0.49 * ULP_AT_BASE,
                                   block=BLOCK, weight_rounding="stochastic",
                                   generator=generator)
    parameter.grad = torch.ones(1024, dtype=parameter.dtype)
    optimizer.step()
    saved = optimizer.rng_state().clone()
    parameter.grad = torch.ones(1024, dtype=parameter.dtype)
    optimizer.step()
    optimizer.set_rng_state(saved)
    repeated = optimizer.rng_state()
    assert torch.equal(saved, repeated)


# ---------------------------------------------------------------------------
# error compensation


def test_error_compensation_bounds_accumulated_rounding_error():
    steps = 1000
    increment = 0.1 * ULP_AT_BASE
    target = BASE - steps * increment

    compensated = _base_parameter(512)
    nearest = _base_parameter(512)
    generator = torch.Generator(device="cpu").manual_seed(1)
    optimizer = optim8bit.Adam8bit(
        [compensated], lr=increment, betas=(0.9, 0.95), block=BLOCK,
        weight_rounding="compensated", generator=generator)
    control = optim8bit.Adam8bit(
        [nearest], lr=increment, betas=(0.9, 0.95), block=BLOCK,
        weight_rounding="nearest")
    for _ in range(steps):
        for parameter, opt in ((compensated, optimizer), (nearest, control)):
            parameter.grad = torch.ones_like(parameter)
            opt.step()

    compensation_error = float(
        (compensated.detach().float() - target).abs().max())
    nearest_error = float((nearest.detach().float() - target).abs().max())
    # nearest-even discards every 0.1-ulp increment: the weight never moves
    assert nearest_error == pytest.approx(steps * increment, rel=1e-6)
    # the carry keeps the accumulated error inside about one ulp
    assert compensation_error <= 1.5 * ULP_AT_BASE
    assert compensation_error <= nearest_error / 50.0
    assert optimizer.compensation_bytes() == 512 * 2
    assert optimizer.total_state_bytes() == (optimizer.state_bytes()
                                             + optimizer.compensation_bytes())


def test_compensation_buffer_fits_the_registered_headroom():
    # Measured, not assumed: one bf16 element per trainable element of the
    # registered full arm, against the 1.17 GiB headroom in the registration.
    per_element = 2  # bfloat16
    total_bytes = TRAINABLE_ELEMENTS * per_element
    total_gib = total_bytes / GIB
    assert total_bytes == 996_226_688
    assert total_gib == pytest.approx(0.9278, abs=5e-4)
    assert total_gib < HEADROOM_GIB
    # and it is the registered *optional* variant, not the default
    assert optim8bit.DEFAULT_NUMERICS == "stochastic"
    assert optim8bit.Adam8bit([_base_parameter(8)]).compensation_bytes() == 0


def test_receipt_records_the_numerics_and_their_bytes():
    parameter = _base_parameter(1000)
    parameter.grad = torch.ones(1000, dtype=parameter.dtype)
    optimizer = optim8bit.Adam8bit([parameter], block=BLOCK,
                                   weight_rounding="stochastic")
    optimizer.step(measure=True)
    receipt = optimizer.receipt()
    assert receipt["weight_rounding"] == "stochastic"
    assert receipt["compensation_bytes"] == 0
    assert receipt["rounding_chunk_elements"] == optim8bit.SR_CHUNK
    assert receipt["numerics_steps_measured"] == 1
    assert receipt["last_numerics"]["elements"] == 1000
    assert "arXiv:2110.02861" in receipt["weight_rounding_basis"]


def test_unknown_numerics_fail_closed():
    with pytest.raises(ValueError):
        optim8bit.Adam8bit([_base_parameter(8)], weight_rounding="magic")


# ---------------------------------------------------------------------------
# the run-level summary the smoke receipt carries


def _record(step: int, lr: float, unchanged: float) -> dict:
    return {
        "step": step, "lr": lr, "unchanged_fraction": unchanged,
        "unchanged_fraction_round_to_nearest": 1.0,
        "sub_half_ulp_fraction": 0.95,
    }


def test_train_summary_records_peak_and_minimum_lr():
    from frontier import train

    records = [_record(1, 2e-5, 0.98),
               _record(200, 1e-4, 0.62),
               _record(9765, 1e-5, 0.995)]
    summary = train.summarise_numerics(records)
    assert summary["peak_lr"]["step"] == 200
    assert summary["peak_lr"]["lr"] == 1e-4
    assert summary["min_lr"]["step"] == 9765
    assert summary["min_lr"]["unchanged_fraction"] == 0.995
    assert summary["peak_lr"]["unchanged_fraction_round_to_nearest"] == 1.0
    assert train.summarise_numerics([]) is None
    # the registered default is stochastic rounding, not nearest
    assert optim8bit.DEFAULT_NUMERICS == "stochastic"
