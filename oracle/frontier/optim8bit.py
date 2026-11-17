#!/usr/bin/env python3
"""Block-wise 8-bit Adam for the full-backbone distillation control (S40).

`bitsandbytes` is not in the pinned oracle image, and rebuilding the court image
to add it is a larger change than the control needs. This module implements the
same idea in readable PyTorch: the two Adam moments are stored as int8 with one
float32 scale per block of 256 elements, and the update arithmetic is float32.
It is the documented 8-bit-optimizer construction of Dettmers et al.
(arXiv:2110.02861, ICLR 2022), not a new optimizer.

What is stored per trainable element:

* first moment: 1 byte (int8, block-wise absmax scale)
* second moment: 1 byte
* two float32 scales per 256-element block: 0.03125 bytes
* the parameter itself stays in its model dtype (bf16 for this arm), so there is
  no fp32 master copy

Total ~2.03 bytes per trainable element for the two moments, against 8 bytes for
fp32 AdamW. The quantization is nearest-even (deterministic), symmetric in
[-127, 127], and per-block so a single large value cannot flatten a whole tensor.
The reported bytes come from the allocated state tensors, not from arithmetic.

The control is registered as `adam8bit`; the training receipt records the
optimizer class, the block size, the state bytes and the parameter dtype.
"""

from __future__ import annotations

import torch

DEFAULT_BLOCK = 256


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


class Adam8bit(torch.optim.Optimizer):
    """AdamW with block-wise 8-bit first and second moments."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 block: int = DEFAULT_BLOCK):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if block < 1:
            raise ValueError(f"invalid block size: {block}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps,
                                      weight_decay=weight_decay, block=int(block)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block = group["block"]
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
                    state["q_m"], state["absmax_m"] = quantize(zeros, block)
                    state["q_v"], state["absmax_v"] = quantize(zeros, block)
                state["step"] += 1
                step = state["step"]
                moment = dequantize(state["q_m"], state["absmax_m"],
                                    state["padded"], block)
                variance = dequantize(state["q_v"], state["absmax_v"],
                                      state["padded"], block)
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
                parameter.add_(update.to(parameter.dtype), alpha=-lr)
                state["q_m"], state["absmax_m"] = quantize(moment, block)
                state["q_v"], state["absmax_v"] = quantize(variance, block)
        return loss

    def load_state_dict(self, state_dict):
        """Restore int8 moments without PyTorch's dtype cast to the parameter.

        `Optimizer.load_state_dict` casts floating-point state to the parameter
        dtype; that would turn the int8 moments into bf16 and silently undo the
        8-bit storage. Keep the quantized tensors and float32 scales as they are.
        """
        params = [parameter for group in self.param_groups
                  for parameter in group["params"]]
        preserved = {}
        for index, state in (state_dict.get("state") or {}).items():
            preserved[index] = {
                key: value.detach().clone() for key, value in state.items()
                if key in ("q_m", "q_v", "absmax_m", "absmax_v")}
        result = super().load_state_dict(state_dict)
        for index, tensors in preserved.items():
            parameter = params[index]
            for key, value in tensors.items():
                self.state[parameter][key] = value.to(device=parameter.device)
        return result

    def state_bytes(self) -> int:
        """Bytes actually allocated for optimizer state (moments and scales)."""
        total = 0
        for state in self.state.values():
            for key in ("q_m", "q_v", "absmax_m", "absmax_v"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total

    def moment_bytes_per_element(self) -> float:
        numel = sum(int(state.get("numel", 0)) for state in self.state.values())
        if not numel:
            return 0.0
        return self.state_bytes() / numel

    def receipt(self) -> dict:
        block = int(self.param_groups[0]["block"]) if self.param_groups else DEFAULT_BLOCK
        return {
            "kind": "stage2_adam8bit",
            "implementation": "oracle/frontier/optim8bit.py",
            "basis": "block-wise quantized Adam moments, Dettmers et al. arXiv:2110.02861",
            "block_elements": block,
            "quantization": "symmetric int8, nearest-even, per-block float32 absmax",
            "parameter_dtype": "model dtype (bf16 for this arm); no fp32 master copy",
            "moment_bytes_per_element": self.moment_bytes_per_element(),
            "state_bytes": self.state_bytes(),
        }
