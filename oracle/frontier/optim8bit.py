#!/usr/bin/env python3
"""Block-wise 8-bit Adam for the full-backbone distillation control (S40/S42).

`bitsandbytes` is not in the pinned oracle image, and rebuilding the court image
to add it is a larger change than the control needs. This module implements the
same idea in readable PyTorch: the first moment is stored as int8 with one
float32 scale per block of 256 elements, the second moment is stored in
bfloat16, and the update arithmetic is float32. The first moment follows the
block-wise construction of Dettmers et al. (arXiv:2110.02861, ICLR 2022); the
second-moment storage is a correction registered in S50.

What is stored per trainable element (S50 onward):

* first moment: 1 byte (int8, block-wise absmax scale) by default, or 2 bytes
  (bfloat16) when `first_moment_storage="bfloat16"` is selected
* second moment: 2 bytes (bfloat16, one exponent and eight mantissa bits)
* one float32 scale per 256-element block: 0.015625 bytes (int8 first moment
  only; the bf16 first moment needs no per-block scale)
* the parameter itself stays in its model dtype (bf16 for this arm), so there is
  no fp32 master copy

Total ~3.02 bytes per trainable element for the shipped int8 first moment,
against 8 bytes for fp32 AdamW. For the 498,113,344 trainable elements of the
full arm that is 3.016 B/element = 1.399 GiB of moment state, 0.456 GiB more
than the pre-S50 2.031 B/element (0.943 GiB). The measured full-arm peak was
11.4289 GiB allocated against the 12.5 GiB limit, so the corrected projection is
11.885 GiB and stays under the limit; fp32 second moments (add 1.392 GiB,
projecting 12.821 GiB) would not. A bf16 first moment costs 2 B/element (0.928
GiB of state on its own); replacing the int8 first moment adds 0.984 B/element
(4.000 B/element total, +0.457 GiB) and projects 12.34 GiB, inside the limit by
0.16 GiB. It is kept behind the `first_moment_storage` flag so the real-gradient
validation can make the switch a one-line change rather than a rewrite.

Why the second moment cannot be linear int8 (S50 defect). A linear absmax code
has spacing `absmax / 127` inside each block, so every squared gradient below
about `absmax / 254` rounds to exactly zero. Because Adam forms
`m / (sqrt(v) + eps)`, a zeroed `v` collapses the denominator to `eps = 1e-8`
whenever the current gradient is also small: the element's update becomes up to
`~1e8` times its AdamW value. The reproduction in
`scripts/s50_optim_defect_sim.py` measures 57.7% of elements with a zeroed
stored second moment and a nonzero first moment at lognormal scale spread 1.0,
with a maximum update 3.2e4 times fp32 AdamW's and a 313% parameter error after
50 steps. bfloat16 has a fixed 0.4% relative error at every magnitude (sqrt
halves it to 0.2% in the denominator), so it cannot zero a small moment.

The int8 first moment is retained because its error is absolute within a block
(`absmax_m / 254`) rather than a relative collapse, and an update direction of
order one arises only where `m` is comparable to its block's scale. It costs one
byte per element, and that is the default; a bf16 first moment costs a second
byte and 0 B/element for the block scale, replacing 1.016 B/element with 2.000
B/element and pushing the projected peak to 12.34 GiB, inside the limit by 0.16
GiB. `first_moment_storage` selects the layout, and the real-gradient harness
(`scripts/s50_real_gradient_validation.py`) decides whether the default must
move to bfloat16. The int8 quantization is nearest-even (deterministic),
symmetric in [-127, 127], and per-block so a single large value cannot flatten a
whole tensor. The reported bytes come from the allocated state tensors, not from
arithmetic.

Weight-update numerics (S42). A bf16 parameter has 8 mantissa bits; an update
whose magnitude is below half an ulp of the weight rounds to zero under
round-to-nearest, and that is exactly the regime at a 1e-4 learning rate on
weights of order 0.01-1. `Adam8bit` therefore applies the fp32 update to the
bf16 weight with one of three registered numerics:

* ``stochastic`` (default) -- unbiased stochastic rounding to the bf16 grid:
  the probability of taking the upper neighbour is the fractional position of
  the exact fp32 result inside its bracket. Expected value is the exact update,
  no extra state, and an update of ``q`` ulp changes the weight with probability
  ``q`` instead of never;
* ``nearest`` -- round-to-nearest (the pre-S42 behaviour, kept as a control);
* ``compensated`` -- round-to-nearest plus a bf16 error-compensation buffer
  (`arXiv:1710.03740`-style residual carry) so the fraction of an ulp that
  nearest-even discards is carried into the next step. It costs exactly one
  bf16 element per trainable element.

Every chunk is processed in bounded temporaries (``SR_CHUNK`` elements), so the
transient fp32 working set is tens of MB rather than a second copy of the model.

The control is registered as `adam8bit`; the training receipt records the
optimizer class, the block size, the state bytes, the parameter dtype, the
weight-update numerics, the compensation bytes, and -- on measured steps -- the
fraction of parameter elements left bitwise unchanged.

The construction follows Dettmers et al. (arXiv:2110.02861) for the moments and
the published stochastic-rounding / error-feedback literature for the update;
it is not a bitwise reimplementation of bitsandbytes.
"""

from __future__ import annotations

import torch

DEFAULT_BLOCK = 256
DEFAULT_NUMERICS = "stochastic"
NUMERICS = ("stochastic", "nearest", "compensated")
# S51: the int8 first moment leaves a bounded absolute error inside each block.
# The switch is kept explicit so a real-gradient validation can move it to
# bfloat16 without touching the step arithmetic.
DEFAULT_FIRST_MOMENT_STORAGE = "int8"
FIRST_MOMENT_STORAGES = ("int8", "bfloat16")
FIRST_MOMENT_BYTES = {"int8": 1, "bfloat16": 2}
# S50: the second moment is stored in bfloat16, not linear int8. bfloat16 has a
# fixed relative precision at every magnitude, so a small squared gradient
# cannot quantise to zero (see the module docstring for the defect and bytes).
SECOND_MOMENT_DTYPE = torch.bfloat16
SECOND_MOMENT_BYTES = 2
# Elements per bounded fp32 temporary. 2**22 bf16 elements are 8 MiB; the whole
# per-chunk working set stays under ~40 MiB, so the transient does not compete
# with the 1.2 GiB registration headroom.
SR_CHUNK = 1 << 22


def _padded(numel: int, block: int) -> int:
    return ((numel + block - 1) // block) * block


def quantize(flat: torch.Tensor, block: int = DEFAULT_BLOCK):
    """`flat` float32 `[N]` -> (int8 `[Np]`, float32 absmax `[Np/block]`)."""
    numel = flat.numel()
    padded = _padded(numel, block)
    if padded != numel:
        flat = torch.cat([flat, flat.new_zeros(padded - numel)])
    view = flat.view(-1, block)
    absmax = view.abs().amax(dim=1).clamp_min(1e-30)
    scale = 127.0 / absmax
    quantized = (view * scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(-1), absmax


def dequantize(quantized: torch.Tensor, absmax: torch.Tensor, numel: int,
               block: int = DEFAULT_BLOCK) -> torch.Tensor:
    """Inverse of `quantize`, truncated back to `numel` elements."""
    scale = absmax / 127.0
    flat = (quantized.view(-1, block).float() * scale[:, None]).reshape(-1)
    return flat[:numel]


# ---------------------------------------------------------------------------
# weight-update numerics


def _adjacent_bf16(rounded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjacent bf16 values strictly below and above `rounded`."""
    positive_infinity = torch.full_like(rounded, float("inf"))
    up = torch.nextafter(rounded, positive_infinity)
    down = torch.nextafter(rounded, -positive_infinity)
    return down, up


@torch.no_grad()
def stochastic_round(values: torch.Tensor, generator: torch.Generator | None = None
                     ) -> torch.Tensor:
    """Unbiased stochastic rounding of float32 `values` to bfloat16.

    For each element, `lo` and `hi` are the adjacent bf16 grid points bracketing
    the exact fp32 value; the result is `hi` with probability
    `(x - lo) / (hi - lo)` and `lo` otherwise. Values that are already on the
    grid are returned unchanged: when `x == lo` the probability is 0, when
    `x == hi` it is 1, so exactness needs no special case (and the draw is still
    consumed, which keeps the stream length independent of the data).

    The draw comes from `generator` when given (a seeded generator makes the
    sequence reproducible and resumable) and from the global torch generator
    otherwise.
    """
    rounded = values.to(torch.bfloat16)
    down, up = _adjacent_bf16(rounded)
    reference = rounded.to(torch.float32)
    above = values > reference
    lower = torch.where(above, reference, down.to(torch.float32))
    upper = torch.where(above, up.to(torch.float32), reference)
    span = upper - lower
    safe = span > 0
    unit = torch.ones_like(span)
    probability = torch.where(
        safe, (values - lower) / torch.where(safe, span, unit),
        torch.zeros_like(span))
    draw = torch.rand(values.shape, dtype=torch.float32, device=values.device,
                      generator=generator)
    result = torch.where(draw < probability, upper, lower).to(torch.bfloat16)
    # inf/nan parameters are not a training regime; pass them through unrounded.
    return torch.where(torch.isfinite(values), result, rounded)


def bf16_ulp(values: torch.Tensor) -> torch.Tensor:
    """Spacing of the bf16 grid at each element's nearest bf16 value."""
    rounded = values.to(torch.bfloat16)
    _, up = _adjacent_bf16(rounded)
    ulp = (up.to(torch.float32) - rounded.to(torch.float32)).abs()
    tiny = torch.full_like(ulp, torch.finfo(torch.float32).tiny)
    return torch.maximum(ulp, tiny)


def _empty_stats() -> dict:
    return {
        "elements": 0,
        "changed": 0,
        "changed_nearest": 0,
        "sub_half_ulp": 0,
        "expected_changed": 0.0,
        "absolute_rounding_error": 0.0,
    }


def _merge_stats(accumulator: dict, chunk: dict) -> None:
    for key in ("elements", "changed", "changed_nearest", "sub_half_ulp"):
        accumulator[key] += chunk[key]
    for key in ("expected_changed", "absolute_rounding_error"):
        accumulator[key] += chunk[key]


def _chunk_statistics(before: torch.Tensor, exact: torch.Tensor,
                      applied: torch.Tensor) -> dict:
    """Bitwise outcome of one update chunk, in elements (not fractions)."""
    elements = before.numel()
    nearest = exact.to(before.dtype)
    changed = (applied.view(torch.int16) != before.view(torch.int16))
    changed_nearest = (nearest.view(torch.int16) != before.view(torch.int16))
    before_f32 = before.to(torch.float32)
    delta = (exact - before_f32).abs()
    ulp = bf16_ulp(exact)
    sub_half = delta < 0.5 * ulp
    expected_changed = (delta / ulp).clamp(max=1.0).sum()
    error = (exact - applied.to(torch.float32)).abs().sum()
    return {
        "elements": int(elements),
        "changed": int(changed.sum().item()),
        "changed_nearest": int(changed_nearest.sum().item()),
        "sub_half_ulp": int(sub_half.sum().item()),
        "expected_changed": float(expected_changed.item()),
        "absolute_rounding_error": float(error.item()),
    }


def finalise_stats(stats: dict, lr: float) -> dict:
    """Fractions from the element counts; the measured numbers the receipt uses."""
    elements = max(1, stats["elements"])
    return {
        "lr": float(lr),
        "elements": int(stats["elements"]),
        "changed_fraction": stats["changed"] / elements,
        "unchanged_fraction": 1.0 - stats["changed"] / elements,
        "unchanged_fraction_round_to_nearest": 1.0 - stats["changed_nearest"] / elements,
        "sub_half_ulp_fraction": stats["sub_half_ulp"] / elements,
        "expected_unchanged_fraction_stochastic": 1.0 - stats["expected_changed"] / elements,
        "mean_absolute_rounding_error": stats["absolute_rounding_error"] / elements,
    }


class Adam8bit(torch.optim.Optimizer):
    """AdamW with block-wise 8-bit first and second moments."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 block: int = DEFAULT_BLOCK,
                 weight_rounding: str = DEFAULT_NUMERICS,
                 first_moment_storage: str = DEFAULT_FIRST_MOMENT_STORAGE,
                 generator: torch.Generator | None = None,
                 chunk: int = SR_CHUNK):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if block < 1:
            raise ValueError(f"invalid block size: {block}")
        if weight_rounding not in NUMERICS:
            raise ValueError(f"unknown weight_rounding {weight_rounding!r}; "
                             f"expected one of {NUMERICS}")
        if first_moment_storage not in FIRST_MOMENT_STORAGES:
            raise ValueError(
                f"unknown first_moment_storage {first_moment_storage!r}; "
                f"expected one of {FIRST_MOMENT_STORAGES}")
        if chunk < 1:
            raise ValueError(f"invalid chunk size: {chunk}")
        super().__init__(params, dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            block=int(block), weight_rounding=weight_rounding,
            first_moment_storage=first_moment_storage, chunk=int(chunk)))
        self.generator = generator
        self.last_numerics: dict | None = None
        self.numerics_steps_measured = 0
        # S50 tripwire: the largest per-element |update| / lr over the whole step.
        # A well-conditioned Adam step is of order one; a value above the
        # registered factor is a moment-storage defect, not a training regime.
        self.last_max_update_over_lr: float | None = None
        self.max_update_over_lr_ever: float = 0.0

    # -- RNG (saved beside the torch/numpy states so a resume is exact) -------
    def rng_state(self):
        return None if self.generator is None else self.generator.get_state()

    def set_rng_state(self, state) -> None:
        if self.generator is not None and state is not None:
            self.generator.set_state(state)

    # -- update application ---------------------------------------------------
    @torch.no_grad()
    def _apply_update(self, parameter: torch.Tensor, update: torch.Tensor,
                      lr: float, state: dict, stats: dict | None) -> None:
        group = self.param_groups[0]
        mode = group["weight_rounding"]
        chunk = group["chunk"]
        dtype = parameter.dtype
        if dtype not in (torch.bfloat16, torch.float16) or mode == "nearest":
            if stats is None:
                # Fast path: the registered pre-S42 arithmetic.
                parameter.add_(update.to(dtype), alpha=-lr)
                return
            # Measured nearest path: chunked so the outcome can be counted
            # without a full-size clone of the parameter.
            stats_chunk = _empty_stats()
            flat_parameter = parameter.reshape(-1)
            flat_update = update.reshape(-1)
            for start in range(0, flat_parameter.numel(), chunk):
                end = min(start + chunk, flat_parameter.numel())
                before = flat_parameter[start:end]
                exact = before.to(torch.float32).add_(flat_update[start:end],
                                                      alpha=-lr)
                applied = exact.to(dtype)
                _merge_stats(stats_chunk,
                             _chunk_statistics(before, exact, applied))
                before.copy_(applied)
            _merge_stats(stats, stats_chunk)
            return

        compensation = None
        if mode == "compensated":
            if "compensation" not in state:
                state["compensation"] = torch.zeros(
                    parameter.numel(), dtype=torch.bfloat16,
                    device=parameter.device)
            compensation = state["compensation"]
        stats_chunk = _empty_stats() if stats is not None else None
        flat_parameter = parameter.reshape(-1)
        flat_update = update.reshape(-1)
        for start in range(0, flat_parameter.numel(), chunk):
            end = min(start + chunk, flat_parameter.numel())
            before = flat_parameter[start:end]
            before_f32 = before.to(torch.float32)
            if compensation is None:
                exact = before_f32.add_(flat_update[start:end], alpha=-lr)
                applied = stochastic_round(exact, self.generator)
            else:
                residual = compensation[start:end].to(torch.float32)
                residual.add_(flat_update[start:end] * (-lr))
                exact = before_f32 + residual
                applied = exact.to(dtype)
                # carry the discarded fraction into the next step
                carried = residual - (applied.to(torch.float32) - before_f32)
                compensation[start:end].copy_(carried.to(torch.bfloat16))
            if stats_chunk is not None:
                _merge_stats(stats_chunk,
                             _chunk_statistics(before, exact, applied))
            before.copy_(applied)
        if stats_chunk is not None:
            _merge_stats(stats, stats_chunk)

    @torch.no_grad()
    def step(self, closure=None, measure: bool = False):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        stats = _empty_stats() if measure else None
        measured_lr = None
        max_direction = None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block = group["block"]
            first_moment_storage = group.get(
                "first_moment_storage", DEFAULT_FIRST_MOMENT_STORAGE)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not parameter.is_floating_point():
                    raise TypeError("Adam8bit only supports floating parameters")
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam8bit does not support sparse gradients")
                state = self.state[parameter]
                numel = parameter.numel()
                if not state:
                    state["step"] = 0
                    state["shape"] = tuple(parameter.shape)
                    state["numel"] = numel
                    state["padded"] = _padded(numel, block)
                    zeros = torch.zeros(state["padded"], dtype=torch.float32,
                                        device=parameter.device)
                    if first_moment_storage == "bfloat16":
                        # Unquantised first moment: fixed relative precision at
                        # every magnitude, no per-block absolute code.
                        state["m"] = torch.zeros(
                            state["padded"], dtype=torch.bfloat16,
                            device=parameter.device)
                    else:
                        state["q_m"], state["absmax_m"] = quantize(zeros, block)
                    # Unquantised bf16 second moment: no linear absmax code, so
                    # no squared gradient can round to zero.
                    state["v"] = torch.zeros(state["padded"],
                                             dtype=SECOND_MOMENT_DTYPE,
                                             device=parameter.device)
                state["step"] += 1
                step = state["step"]
                if first_moment_storage == "bfloat16":
                    moment = state["m"].to(torch.float32)
                else:
                    moment = dequantize(state["q_m"], state["absmax_m"],
                                        state["padded"], block)
                variance = state["v"].to(torch.float32)
                if weight_decay != 0.0:
                    # Decoupled AdamW decay, not an L2 term in the gradient.
                    parameter.mul_(1.0 - lr * weight_decay)
                grad_flat = grad.detach().to(torch.float32).reshape(-1)
                if grad_flat.numel() < state["padded"]:
                    grad_flat = torch.cat([
                        grad_flat,
                        grad_flat.new_zeros(state["padded"] - grad_flat.numel())])
                moment.mul_(beta1).add_(grad_flat, alpha=1.0 - beta1)
                variance.mul_(beta2).addcmul_(grad_flat, grad_flat,
                                              value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                denominator = (variance / bias2).sqrt().add_(eps)
                update = (moment / bias1) / denominator
                update = update[:state["numel"]].reshape(state["shape"])
                # `amax`/`amin` read the tensor without an abs() temporary.
                if max_direction is None:
                    max_direction = torch.maximum(update.amax(), -update.amin())
                else:
                    max_direction = torch.maximum(
                        max_direction,
                        torch.maximum(update.amax(), -update.amin()))
                self._apply_update(parameter, update, lr, state, stats)
                if first_moment_storage == "bfloat16":
                    state["m"].copy_(moment.to(torch.bfloat16))
                else:
                    state["q_m"], state["absmax_m"] = quantize(moment, block)
                state["v"].copy_(variance.to(SECOND_MOMENT_DTYPE))
                measured_lr = lr
        if max_direction is not None:
            self.last_max_update_over_lr = float(max_direction.item())
            self.max_update_over_lr_ever = max(self.max_update_over_lr_ever,
                                               self.last_max_update_over_lr)
        else:
            self.last_max_update_over_lr = 0.0
        if stats is not None:
            self.numerics_steps_measured += 1
            self.last_numerics = finalise_stats(stats, lr=measured_lr or 0.0)
        else:
            self.last_numerics = None
        return loss

    def load_state_dict(self, state_dict):
        """Restore int8 moments without PyTorch's dtype cast to the parameter.

        `Optimizer.load_state_dict` casts floating-point state to the parameter
        dtype; that would turn the int8 moments into bf16 and silently undo the
        8-bit storage. Keep the quantized tensors, float32 scales and the bf16
        compensation buffer as they are.
        """
        params = [parameter for group in self.param_groups
                  for parameter in group["params"]]
        storage = self.param_groups[0].get(
            "first_moment_storage", DEFAULT_FIRST_MOMENT_STORAGE)
        preserved = {}
        for index, state in (state_dict.get("state") or {}).items():
            if "q_v" in state or "absmax_v" in state:
                raise ValueError(
                    "pre-S50 Adam8bit state stores the second moment as a linear "
                    "int8 absmax code; that state is invalidated by the S50 "
                    "optimiser defect and cannot be resumed. Rerun from the "
                    "arm's initialisation with the corrected optimizer.")
            has_int8_first = "q_m" in state or "absmax_m" in state
            has_bf16_first = "m" in state
            if storage == "bfloat16" and has_int8_first:
                raise ValueError(
                    "checkpoint stores the first moment as int8 but the optimizer "
                    "is configured for bfloat16; the first-moment storage is part "
                    "of the recipe and cannot be converted on resume")
            if storage == "int8" and has_bf16_first:
                raise ValueError(
                    "checkpoint stores the first moment as bfloat16 but the "
                    "optimizer is configured for int8; select the same "
                    "first_moment_storage on resume")
            preserved[index] = {
                key: value.detach().clone() for key, value in state.items()
                if key in ("q_m", "absmax_m", "m", "v", "compensation")}
        result = super().load_state_dict(state_dict)
        for index, tensors in preserved.items():
            parameter = params[index]
            for key, value in tensors.items():
                self.state[parameter][key] = value.to(device=parameter.device)
        return result

    def state_bytes(self) -> int:
        """Bytes actually allocated for optimizer moments and scales."""
        total = 0
        for state in self.state.values():
            for key in ("q_m", "absmax_m", "m", "v"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total

    def compensation_bytes(self) -> int:
        """Bytes allocated for the optional bf16 error-compensation buffer."""
        total = 0
        for state in self.state.values():
            value = state.get("compensation")
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def total_state_bytes(self) -> int:
        return self.state_bytes() + self.compensation_bytes()

    def moment_bytes_per_element(self) -> float:
        numel = sum(int(state.get("numel", 0)) for state in self.state.values())
        if not numel:
            return 0.0
        return self.state_bytes() / numel

    def receipt(self) -> dict:
        group = self.param_groups[0] if self.param_groups else {}
        block = int(group.get("block", DEFAULT_BLOCK))
        storage = group.get("first_moment_storage",
                            DEFAULT_FIRST_MOMENT_STORAGE)
        numel = sum(int(state.get("numel", 0)) for state in self.state.values())
        return {
            "kind": "stage2_adam8bit",
            "implementation": "oracle/frontier/optim8bit.py",
            "basis": "block-wise quantized Adam moments, Dettmers et al. arXiv:2110.02861",
            "block_elements": block,
            "first_moment_storage": storage,
            "first_moment_bytes_per_element": float(FIRST_MOMENT_BYTES[storage]),
            "first_moment_basis": {
                "int8": ("symmetric int8, nearest-even, per-block float32 "
                         "absmax; bounded absolute block error"),
                "bfloat16": ("unquantised bfloat16; fixed 0.4% relative "
                             "precision at every magnitude, no per-block scale"),
            }[storage],
            "quantization": ("symmetric int8, nearest-even, per-block float32 "
                             "absmax" if storage == "int8" else
                             "none for the first moment (bfloat16 storage)"),
            "second_moment_storage": "bfloat16 (unquantised; S50 correction)",
            "second_moment_basis": (
                "linear absmax int8 zeroes any squared gradient below "
                "absmax/254, collapsing the Adam denominator to eps when the "
                "current gradient is also small; bfloat16 has fixed 0.4% "
                "relative precision at every magnitude (S50)"),
            "second_moment_bytes_per_element": float(SECOND_MOMENT_BYTES),
            "parameter_dtype": "model dtype (bf16 for this arm); no fp32 master copy",
            "weight_rounding": group.get("weight_rounding", DEFAULT_NUMERICS),
            "weight_rounding_basis": {
                "stochastic": "unbiased stochastic rounding; arXiv:1511.00363, "
                              "arXiv:2110.02861 §3 (SR for weight updates)",
                "nearest": "round-to-nearest control (pre-S42 behaviour)",
                "compensated": "round-to-nearest + bf16 residual carry, "
                               "arXiv:1710.03740",
            }[group.get("weight_rounding", DEFAULT_NUMERICS)],
            "rounding_chunk_elements": int(group.get("chunk", SR_CHUNK)),
            "moment_bytes_per_element": self.moment_bytes_per_element(),
            "state_bytes": self.state_bytes(),
            "compensation_dtype": "bfloat16",
            "compensation_bytes": self.compensation_bytes(),
            "compensation_bytes_per_element": (
                self.compensation_bytes() / numel if numel else 0.0),
            "total_state_bytes": self.total_state_bytes(),
            "numerics_steps_measured": self.numerics_steps_measured,
            "last_numerics": self.last_numerics,
            "max_update_over_lr": self.last_max_update_over_lr,
            "max_update_over_lr_ever": self.max_update_over_lr_ever,
        }
