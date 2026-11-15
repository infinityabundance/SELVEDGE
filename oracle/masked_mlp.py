#!/usr/bin/env python3
"""Exact masked-MLP instrument for the M1 group-omission frontier.

The published bundle policy partitions each FFN's intermediate channels into
`G` contiguous groups. H0 proved that, in exact arithmetic,

    FFN(x) = sum_g W_down[:, G_g] (SiLU(W_gate[G_g] x) * (W_up[G_g] x)).

This module provides two implementations of the masked sum over a selected set
of groups:

* `grouped_forward` computes each group's contribution explicitly and adds them
  in ascending group order. This is the reference instrument and the analogue of
  the Rust `ffn_grouped_f32` path.
* `MaskedMlp` keeps the original projection modules and multiplies the
  intermediate activation by a per-channel 0/1 mask before the down projection.
  Excluded channels contribute exactly zero, so this is the same sum with the
  down-projection reduction order of the dense model; it is the fast path used
  by the sweep.

Neither path trains, routes, samples or approximates. With every group selected
both reproduce the dense MLP; the small f32 differences from reduction order are
measured, not assumed away.

Everything runs under `torch.inference_mode()` in f32 with TF32 disabled by the
caller. The module has no CLI and no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ReadAccounting:
    """Parameter-element reads and FLOPs of one selection+execution path.

    `selector_*` counts what the selector read to make its decision (including
    any dense gate/up projection used to score groups). `execution_*` counts
    what the selected computation read. `executed_unique_*` is filled by the
    caller from the union of the two identity sets; the counter itself cannot
    know tensor identities, so it never invents a union.
    """

    selector_parameter_elements: int = 0
    selector_elements_read: int = 0
    execution_elements_read: int = 0
    selection_flops: int = 0
    execution_flops: int = 0
    matmul_calls: int = 0
    tokens: int = 0
    layers: int = 0
    measurements: list[str] = field(default_factory=list)

    def add_selector(self, elements: int, flops: int) -> None:
        self.selector_elements_read += int(elements)
        self.selection_flops += int(flops)

    def add_execution(self, elements: int, flops: int, matmuls: int = 1) -> None:
        self.execution_elements_read += int(elements)
        self.execution_flops += int(flops)
        self.matmul_calls += int(matmuls)

    def as_dict(self) -> dict:
        return {
            "selector_parameter_elements": self.selector_parameter_elements,
            "selector_elements_read": self.selector_elements_read,
            "execution_elements_read": self.execution_elements_read,
            "selection_flops": self.selection_flops,
            "execution_flops": self.execution_flops,
            "matmul_calls": self.matmul_calls,
            "tokens": self.tokens,
            "layers": self.layers,
            "measurements": list(self.measurements),
        }


def group_bounds(intermediate: int, groups: int) -> list[tuple[int, int]]:
    """Contiguous group bounds, matching the converter's `ceil` policy."""
    if intermediate <= 0:
        raise ValueError("intermediate must be positive")
    if groups <= 0:
        raise ValueError("groups must be positive")
    width = -(-intermediate // groups)
    bounds = []
    start = 0
    for _ in range(groups):
        end = min(start + width, intermediate)
        if start >= intermediate:
            break
        bounds.append((start, end))
        start = end
    if start != intermediate:
        raise ValueError(f"groups do not cover {intermediate} channels")
    return bounds


def group_widths(intermediate: int, groups: int) -> list[int]:
    return [end - start for start, end in group_bounds(intermediate, groups)]


def selected_mask(
    intermediate: int, groups: int, selected: list[int], device=None
) -> torch.Tensor:
    """A 0/1 f32 mask over intermediate channels for the selected groups."""
    bounds = group_bounds(intermediate, groups)
    mask = torch.zeros(intermediate, dtype=torch.float32, device=device)
    for index in selected:
        if index < 0 or index >= len(bounds):
            raise ValueError(f"group {index} is out of range 0..{len(bounds)}")
        start, end = bounds[index]
        mask[start:end] = 1.0
    return mask


def masks_from_selection(
    intermediate: int, groups: int, selection: list[list[int]], device=None
) -> list[torch.Tensor]:
    return [selected_mask(intermediate, groups, selected, device=device) for selected in selection]


def dense_channels(
    mlp: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """The intermediate activation `SiLU(gate x) * (up x)` in f32."""
    gate = mlp.gate_proj(hidden_states)
    up = mlp.up_proj(hidden_states)
    return mlp.act_fn(gate) * up


def grouped_forward(
    mlp: torch.nn.Module,
    hidden_states: torch.Tensor,
    groups: int,
    selected: list[int],
) -> torch.Tensor:
    """Sum the selected groups' contributions explicitly, in group order.

    `hidden_states` is any shape ending in the hidden width. The result has the
    same shape. This is the reference path (the Rust grouped reduction order).
    """
    shape = hidden_states.shape
    if shape[-1] != mlp.gate_proj.in_features:
        raise ValueError("hidden width does not match the gate projection")
    flat = hidden_states.reshape(-1, shape[-1])
    activation = dense_channels(mlp, flat)
    intermediate = activation.shape[-1]
    bounds = group_bounds(intermediate, groups)
    out = torch.zeros(
        (flat.shape[0], mlp.down_proj.out_features),
        dtype=torch.float32,
        device=flat.device,
    )
    weight = mlp.down_proj.weight
    for index in sorted(selected):
        start, end = bounds[index]
        out = out + activation[:, start:end] @ weight[:, start:end].transpose(0, 1)
    return out.reshape(*shape[:-1], mlp.down_proj.out_features)


def sparse_grouped_forward(
    mlp: torch.nn.Module,
    hidden_states: torch.Tensor,
    groups: int,
    selection: torch.Tensor,
    counter: ReadAccounting | None = None,
) -> torch.Tensor:
    """Execute only the selected groups, slicing weights before every matmul.

    `selection` is `[T, G]` boolean (one row per token). Tokens with the same
    subset are executed together; for each distinct subset the gate/up rows and
    down columns are sliced with `index_select` *before* the matmul, so no
    unselected channel is ever computed. This is the genuinely sparse execution
    path: its parameter reads are exactly the sliced elements.

    The result is the same sum as the masked-dense path, up to f32 reduction
    order. `oracle/tests/test_masked_mlp.py` checks the two paths against a
    float64 hand computation and against each other.
    """
    if selection.dim() != 2:
        raise ValueError("selection must be [T, G]")
    shape = hidden_states.shape
    if shape[-1] != mlp.gate_proj.in_features:
        raise ValueError("hidden width does not match the gate projection")
    flat = hidden_states.reshape(-1, shape[-1])
    if flat.shape[0] != selection.shape[0]:
        raise ValueError(
            f"selection has {selection.shape[0]} rows for {flat.shape[0]} tokens"
        )
    intermediate = mlp.down_proj.in_features
    out_width = mlp.down_proj.out_features
    bounds = group_bounds(intermediate, groups)
    if selection.shape[1] != len(bounds):
        raise ValueError("selection group count does not match the layer")
    selection = selection.to(device=flat.device, dtype=torch.bool)

    out = torch.zeros(
        (flat.shape[0], out_width), dtype=flat.dtype, device=flat.device
    )
    by_subset: dict[tuple[int, ...], list[int]] = {}
    rows = selection.detach().cpu()
    for token in range(rows.shape[0]):
        key = tuple(int(index) for index in torch.nonzero(rows[token]).flatten())
        by_subset.setdefault(key, []).append(token)

    gate_weight = mlp.gate_proj.weight
    up_weight = mlp.up_proj.weight
    down_weight = mlp.down_proj.weight
    for key, tokens in by_subset.items():
        if not key:
            continue
        channels: list[int] = []
        for index in key:
            start, end = bounds[index]
            channels.extend(range(start, end))
        index_tensor = torch.tensor(channels, dtype=torch.long, device=flat.device)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=flat.device)
        x = flat.index_select(0, token_tensor)
        width = len(channels)
        count = len(tokens)
        gate = x @ gate_weight.index_select(0, index_tensor).transpose(0, 1)
        up = x @ up_weight.index_select(0, index_tensor).transpose(0, 1)
        activation = F.silu(gate) * up
        down = down_weight.index_select(1, index_tensor)
        out.index_copy_(0, token_tensor, activation @ down.transpose(0, 1))
        if counter is not None:
            # One matmul call reads each sliced weight element once, regardless
            # of how many token rows share the subset (a batched kernel reuses
            # the weight). FLOPs still scale with the token rows.
            elements = width * (2 * flat.shape[1] + out_width)
            flops = 2 * count * elements
            counter.add_execution(elements, flops, matmuls=3)
    if counter is not None:
        counter.tokens += flat.shape[0]
        counter.layers += 1
        counter.measurements.append("sliced-weight-element-count")
    return out.reshape(*shape[:-1], out_width)


def dense_execution_accounting(
    tokens: int, hidden: int, intermediate: int, out_width: int, layers: int
) -> tuple[int, int]:
    """Parameter elements read and FLOPs of a dense masked execution.

    One token through one layer reads every gate, up and down element once:
    `I * (2H + O)`. FLOPs are `2 * I * (2H + O)` per token per layer. This is the
    count the masked-dense path physically performs; it is reported as the
    measured-basis for the `dense_computation_masked` verdict.
    """
    per_token_layer = intermediate * (2 * hidden + out_width)
    elements = layers * per_token_layer
    return elements, 2 * tokens * elements


def selector_dense_accounting(
    tokens: int, hidden: int, intermediate: int, layers: int
) -> tuple[int, int]:
    """Parameter elements read and FLOPs of a dense gate/up group-norm selector.

    `dense_group_norms` computes every gate and up channel for every token and
    layer to rank groups; it reads `2 * H * I` elements per token per layer and
    performs `2 * (2 * H * I)` FLOPs. Those reads are outside a selected-group
    active set and must be charged to the selection path.
    """
    per_token_layer = 2 * hidden * intermediate
    elements = layers * per_token_layer
    return elements, 2 * tokens * elements


def audit_sparse_reads(
    mlp: torch.nn.Module,
    hidden_states: torch.Tensor,
    groups: int,
    selection: torch.Tensor,
) -> dict:
    """Cross-check `sparse_grouped_forward` against a per-op tensor audit.

    The sparse path is instrumented with `torch.profiler(record_shapes=True)`.
    Every `aten::mm` whose second operand is a sliced weight contributes
    `k * n` parameter elements, and `2 * m * n * k` FLOPs. The audit compares
    those sums with the counter's by-construction count. A mismatch is returned
    as `matches: false` instead of being hidden.
    """
    counter = ReadAccounting()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as profile:
        with torch.no_grad():
            sparse_grouped_forward(
                mlp, hidden_states, groups, selection, counter=counter
            )
    profiled_elements = 0
    profiled_flops = 0
    matmuls = 0
    for event in profile.key_averages(group_by_input_shape=True):
        # Only the kernel-level `aten::mm` is counted: the dispatcher also
        # records an `aten::matmul` event for the same operation, and summing
        # both would double every byte.
        if event.key != "aten::mm":
            continue
        shapes = event.input_shapes
        if len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2:
            continue
        m, k = shapes[0]
        k2, n = shapes[1]
        if k != k2 or 0 in (m, k, n):
            continue
        profiled_elements += event.count * k * n
        profiled_flops += event.count * 2 * m * n * k
        matmuls += event.count
    return {
        "counted_elements": counter.execution_elements_read,
        "profiled_elements": profiled_elements,
        "counted_flops": counter.execution_flops,
        "profiled_flops": profiled_flops,
        "counted_matmuls": counter.matmul_calls,
        "profiled_matmuls": matmuls,
        "matches": (
            counter.execution_elements_read == profiled_elements
            and counter.execution_flops == profiled_flops
        ),
    }


class MaskedMlp(torch.nn.Module):
    """An FFN wrapper with a mutable per-channel or per-token mask.

    The original projection modules are reused, so no weights are copied and no
    parameter is added. `mask` is a non-persistent buffer of length
    `intermediate`; setting it to ones reproduces the dense MLP exactly (the
    multiplication by 1.0 is exact and the reduction order is unchanged).

    `token_mask` is empty by default. When it is set to a `[T, intermediate]`
    (or `[1, T, intermediate]`) tensor it takes precedence over `mask`, so each
    position can use a different group subset. This is the per-token path used
    by the M2 selectors; the arithmetic is the same masked sum, only the mask
    is position dependent.

    `track_activation` is off by default. When it is on, the pre-mask
    intermediate activation is made a leaf that requires grad and retains it,
    and is stored as `last_activation`. This is used only for the labelled
    non-deployable gradient x activation attribution arms.
    """

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        intermediate = original.down_proj.in_features
        self.register_buffer(
            "mask",
            torch.ones(intermediate, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "token_mask",
            torch.zeros(0, dtype=torch.float32),
            persistent=False,
        )
        self.track_activation = False
        self.last_activation = None
        self.sparse_groups = 0
        self.sparse_selection = None
        self.read_counter = None

    def set_sparse_selection(
        self,
        selection: torch.Tensor | None,
        groups: int = 0,
        counter: ReadAccounting | None = None,
    ) -> None:
        """Execute only the selected groups, slicing weights before the matmul.

        `selection` is a `[T, G]` boolean tensor (or `[1, T, G]`), one row per
        token. When it is set, `forward` uses `sparse_grouped_forward` instead of
        computing dense gate/up projections and masking afterwards. `counter`,
        when supplied, accumulates the exact number of parameter elements the
        sliced matmuls read.
        """
        if selection is None:
            self.sparse_selection = None
            self.sparse_groups = 0
            self.read_counter = None
            return
        if selection.dim() == 3:
            if selection.shape[0] != 1:
                raise ValueError("sparse selection batch dimension must be 1")
            selection = selection[0]
        if selection.dim() != 2:
            raise ValueError("sparse selection must be [T, G]")
        self.sparse_selection = selection.to(device=self.mask.device, dtype=torch.bool)
        self.sparse_groups = int(groups)
        self.read_counter = counter

    def clear_sparse_selection(self) -> None:
        self.set_sparse_selection(None)

    def set_token_mask(self, token_mask: torch.Tensor | None) -> None:
        if token_mask is None:
            self.token_mask = torch.zeros(
                0, dtype=self.mask.dtype, device=self.mask.device
            )
            return
        if token_mask.dim() == 3:
            if token_mask.shape[0] != 1:
                raise ValueError("token mask batch dimension must be 1")
            token_mask = token_mask[0]
        if token_mask.dim() != 2:
            raise ValueError("token mask must be [T, intermediate]")
        if token_mask.shape[1] != self.mask.numel():
            raise ValueError("token mask width does not match the layer")
        self.token_mask = token_mask.to(
            device=self.mask.device, dtype=self.mask.dtype
        )

    def clear_token_mask(self) -> None:
        self.set_token_mask(None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sparse_selection is not None:
            return sparse_grouped_forward(
                self,
                hidden_states,
                self.sparse_groups,
                self.sparse_selection,
                counter=self.read_counter,
            )
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activation = self.act_fn(gate) * up
        self.last_activation = activation
        if self.track_activation:
            activation.requires_grad_(True)
            activation.retain_grad()
        if self.token_mask.numel() != 0:
            if self.token_mask.shape != activation.shape[-2:]:
                raise ValueError(
                    "token mask shape does not match the activation: "
                    f"{tuple(self.token_mask.shape)} vs {tuple(activation.shape[-2:])}"
                )
            return self.down_proj(activation * self.token_mask)
        return self.down_proj(activation * self.mask)


class MaskController:
    """Install `MaskedMlp` on every decoder layer and switch masks in place."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layers = list(model.model.layers)
        self.originals = []
        self.wrappers = []
        for layer in self.layers:
            original = layer.mlp
            wrapper = MaskedMlp(original).to(
                device=original.down_proj.weight.device,
                dtype=original.down_proj.weight.dtype,
            )
            self.originals.append(original)
            self.wrappers.append(wrapper)
            layer.mlp = wrapper
        self.intermediate = self.wrappers[0].mask.numel()
        self.layers_count = len(self.wrappers)

    def set_masks(self, masks: list[torch.Tensor]) -> None:
        if len(masks) != len(self.wrappers):
            raise ValueError(
                f"{len(masks)} masks for {len(self.wrappers)} layers"
            )
        for wrapper, mask in zip(self.wrappers, masks):
            if mask.numel() != wrapper.mask.numel():
                raise ValueError("mask width does not match the layer")
            wrapper.mask.copy_(mask)

    def set_token_masks(self, masks: list[torch.Tensor]) -> None:
        if len(masks) != len(self.wrappers):
            raise ValueError(
                f"{len(masks)} token masks for {len(self.wrappers)} layers"
            )
        for wrapper, mask in zip(self.wrappers, masks):
            wrapper.set_token_mask(mask)

    def clear_token_masks(self) -> None:
        for wrapper in self.wrappers:
            wrapper.clear_token_mask()

    def set_dense(self) -> None:
        self.clear_token_masks()
        for wrapper in self.wrappers:
            wrapper.clear_sparse_selection()
            wrapper.mask.fill_(1.0)

    def set_sparse_selections(
        self,
        selections: list[torch.Tensor],
        groups: int,
        counter: ReadAccounting | None = None,
    ) -> None:
        """Switch every layer to genuinely sparse execution of its selection."""
        if len(selections) != len(self.wrappers):
            raise ValueError(
                f"{len(selections)} selections for {len(self.wrappers)} layers"
            )
        self.clear_token_masks()
        for wrapper, selection in zip(self.wrappers, selections):
            wrapper.set_sparse_selection(selection, groups, counter)

    def clear_sparse_selections(self) -> None:
        for wrapper in self.wrappers:
            wrapper.clear_sparse_selection()

    def suspend(self) -> None:
        """Restore the original MLP modules so the unmodified model can run."""
        for layer, original in zip(self.layers, self.originals):
            layer.mlp = original

    def resume(self) -> None:
        """Reinstall the masked wrappers after a `suspend`."""
        for layer, wrapper in zip(self.layers, self.wrappers):
            layer.mlp = wrapper

    def restore(self) -> None:
        self.suspend()


def active_parameter_report(
    selection: list[list[int]],
    hidden: int,
    intermediate: int,
    groups: int,
    backbone_elements: int,
) -> dict:
    """Realized active unique parameters and derived FLOPs for a mask.

    `backbone_elements` is the receipt-sourced always-active count. Each
    selected group contributes `3 * width * hidden` learned elements; the
    selected intermediate width drives the derived matmul FLOPs
    (`6 * width * hidden` per layer, `docs/ARCHITECTURE.md`).
    """
    bounds = group_bounds(intermediate, groups)
    ffn_elements = 0
    channels = 0
    ffn_flops = 0
    for selected in selection:
        for index in selected:
            start, end = bounds[index]
            width = end - start
            ffn_elements += 3 * width * hidden
            channels += width
            ffn_flops += 6 * width * hidden
    return {
        "group_count": sum(len(selected) for selected in selection),
        "active_ffn_elements": ffn_elements,
        "active_unique_elements": backbone_elements + ffn_elements,
        "active_intermediate_channels": channels,
        "ffn_matmul_flops_per_token": ffn_flops,
    }


def selection_from_masks(masks: list[torch.Tensor], groups: int) -> list[list[int]]:
    """Recover group indices from channel masks (each group is all-or-nothing)."""
    selection = []
    for mask in masks:
        intermediate = mask.numel()
        bounds = group_bounds(intermediate, groups)
        selected = []
        for index, (start, end) in enumerate(bounds):
            block = mask[start:end]
            if bool((block > 0.5).all()):
                selected.append(index)
            elif bool((block > 0.5).any()):
                raise ValueError("mask is not a union of whole groups")
        selection.append(selected)
    return selection


def logits_kl_metrics(
    teacher_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict:
    """Per-position teacher KL, top-1 agreement, entropy and NLLs.

    `teacher_logits` and `masked_logits` are `[1, T, V]`; `input_ids` is
    `[1, T]`. Returns 1-D f32 CPU tensors over the predicting positions
    (and, for the NLL pair used by the perplexity ratio, only positions after
    the first predicting position).
    """
    if teacher_logits.shape != masked_logits.shape:
        raise ValueError("teacher and masked logits differ in shape")
    if teacher_logits.shape[0] != 1:
        raise ValueError("metrics expect a single sequence")
    logits_t = teacher_logits[0, :-1, :].float()
    logits_m = masked_logits[0, :-1, :].float()
    target = input_ids[0, 1:]
    logp_t = F.log_softmax(logits_t, dim=-1)
    logp_m = F.log_softmax(logits_m, dim=-1)
    probs_t = logp_t.exp()
    kl = (probs_t * (logp_t - logp_m)).sum(dim=-1)
    top1 = (logits_t.argmax(dim=-1) == logits_m.argmax(dim=-1)).to(torch.float32)
    entropy = -(probs_t * logp_t).sum(dim=-1)
    nll_t = -logp_t.gather(1, target.unsqueeze(1)).squeeze(1)
    nll_m = -logp_m.gather(1, target.unsqueeze(1)).squeeze(1)
    # The perplexity ratio in the contract excludes the first predicting
    # position of each sequence.
    if nll_t.numel() > 1:
        nll_t = nll_t[1:]
        nll_m = nll_m[1:]
    return {
        "kl": kl.detach().to("cpu", torch.float32),
        "top1": top1.detach().to("cpu", torch.float32),
        "entropy": entropy.detach().to("cpu", torch.float32),
        "nll_teacher": nll_t.detach().to("cpu", torch.float32),
        "nll_masked": nll_m.detach().to("cpu", torch.float32),
    }


def summarize_positions(metrics: dict) -> dict:
    """Aggregate per-position metrics for one window."""
    kl = metrics["kl"]
    nll_t = metrics["nll_teacher"]
    nll_m = metrics["nll_masked"]
    positions = int(kl.numel())
    return {
        "positions": positions,
        "kl_mean": float(kl.mean()) if positions else 0.0,
        "kl_median": float(kl.median()) if positions else 0.0,
        "kl_p95": float(torch.quantile(kl, 0.95)) if positions else 0.0,
        "kl_min": float(kl.min()) if positions else 0.0,
        "kl_max": float(kl.max()) if positions else 0.0,
        "top1_agreement": float(metrics["top1"].mean()) if positions else 0.0,
        "teacher_entropy_mean": float(metrics["entropy"].mean()) if positions else 0.0,
        "nll_teacher_mean": float(nll_t.mean()) if nll_t.numel() else 0.0,
        "nll_masked_mean": float(nll_m.mean()) if nll_m.numel() else 0.0,
        "nll_positions": int(nll_t.numel()),
    }


def calibration_group_statistics(
    model: torch.nn.Module,
    windows: list[list[int]],
    groups: int,
    device: torch.device,
) -> dict:
    """Mean group activation norm and activation frequency on calibration text.

    For token `t`, `a_g(t)` is the group's intermediate activation vector and
    `mass_g(t) = ||a_g(t)|| / sum_j ||a_j(t)||`. The magnitude statistic is the
    mean of `||a_g(t)||`; the frequency statistic is the fraction of tokens with
    `mass_g(t) > 2 / groups` (twice the uniform share). Tokens with zero total
    norm are excluded from the frequency statistic and counted.
    """
    layers = list(model.model.layers)
    layer_count = len(layers)
    sums = torch.zeros((layer_count, groups), dtype=torch.float64, device=device)
    frequency = torch.zeros((layer_count, groups), dtype=torch.float64, device=device)
    tokens = torch.zeros(layer_count, dtype=torch.float64, device=device)
    skipped = torch.zeros(layer_count, dtype=torch.float64, device=device)
    bounds = None

    def make_hook(index: int):
        def hook(module, inputs, output):
            nonlocal bounds
            hidden = inputs[0]
            flat = hidden.detach().reshape(-1, hidden.shape[-1])
            activation = dense_channels(module, flat)
            intermediate = activation.shape[-1]
            if bounds is None:
                bounds = group_bounds(intermediate, groups)
            norms = torch.stack(
                [activation[:, start:end].norm(dim=1) for start, end in bounds],
                dim=1,
            )
            total = norms.sum(dim=1)
            sums[index] += norms.to(torch.float64).sum(dim=0)
            valid = total > 0
            if bool(valid.any()):
                mass = norms[valid] / total[valid].unsqueeze(1)
                frequency[index] += (
                    (mass > 2.0 / groups).sum(dim=0).to(torch.float64)
                )
            tokens[index] += float(flat.shape[0])
            skipped[index] += float((~valid).sum())

        return hook

    handles = [layer.mlp.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            for window in windows:
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    result = {"groups": groups, "layers": layer_count, "bounds": bounds}
    per_layer = []
    for index in range(layer_count):
        count = float(tokens[index])
        valid_count = float(tokens[index] - skipped[index])
        mean_norm = (sums[index] / count).tolist() if count else [0.0] * groups
        freq = (
            (frequency[index] / valid_count).tolist() if valid_count else [0.0] * groups
        )
        per_layer.append(
            {
                "layer": index,
                "tokens": int(count),
                "skipped_zero_mass": int(skipped[index]),
                "mean_norm": mean_norm,
                "frequency": freq,
            }
        )
    result["layers_data"] = per_layer
    return result


def rankings(statistics: dict) -> dict:
    """Descending group rankings per layer for the magnitude and frequency selectors."""
    magnitude = []
    frequency = []
    for layer in statistics["layers_data"]:
        groups = statistics["groups"]
        magnitude.append(
            sorted(range(groups), key=lambda g: (-layer["mean_norm"][g], g))
        )
        frequency.append(
            sorted(range(groups), key=lambda g: (-layer["frequency"][g], g))
        )
    return {"magnitude": magnitude, "frequency": frequency}


def random_rankings(layers: int, groups: int, seed: int) -> list[list[int]]:
    """Deterministic per-layer permutations, generated in layer order."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [torch.randperm(groups, generator=generator).tolist() for _ in range(layers)]


class HiddenCapture:
    """Capture each decoder layer's pre-MLP hidden state during a forward.

    The hook is installed on `layer.mlp`, so `inputs[0]` is the residual-stream
    hidden state `h` that the FFN actually receives. It is detached and cloned;
    no computation is changed. `layers[l]` is a `[1, T, hidden]` tensor for the
    last forward that ran while the capture was active.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.layers = list(model.model.layers)
        self.captured: list[torch.Tensor | None] = [None] * len(self.layers)
        self.handles = []
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.mlp.register_forward_pre_hook(self._make_hook(index))
            )

    def _make_hook(self, index: int):
        def hook(_module, inputs):
            if inputs:
                self.captured[index] = inputs[0].detach().clone()
        return hook

    def clear(self) -> None:
        self.captured = [None] * len(self.layers)

    def get(self, layer: int) -> torch.Tensor:
        value = self.captured[layer]
        if value is None:
            raise RuntimeError(f"no hidden state captured for layer {layer}")
        return value

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


def group_contributions(
    mlp: torch.nn.Module,
    hidden_states: torch.Tensor,
    groups: int,
) -> dict:
    """Per-group FFN output contributions for one layer.

    `hidden_states` is `[T, hidden]` (or any shape ending in hidden). Returns,
    for the flattened token rows `T`:

    * `contributions`: `[T, G, hidden_out]`, where row `g` is
      `a_g @ W_down[:, G_g]^T` and `a_g` is the group's intermediate
      activation;
    * `dense`: `[T, hidden_out]`, the sum over all groups (the dense layer
      output for these hidden states);
    * `gram`: `[T, G, G]`, `contributions . contributions` over the output axis;
    * `dot_dense`: `[T, G]`, `contributions . dense`;
    * `dense_sq`: `[T]`, `||dense||^2`;
    * `group_norms`: `[T, G]`, `||a_g||_2`;
    * `bounds`: the group channel bounds.

    The exact local squared error of a subset `S` is
    `dense_sq - 2 sum_{g in S} dot_dense[g] + sum_{g,j in S} gram[g,j]`.
    These identities are tested in `oracle/tests/test_masked_mlp.py`.
    """
    shape = hidden_states.shape
    if shape[-1] != mlp.gate_proj.in_features:
        raise ValueError("hidden width does not match the gate projection")
    flat = hidden_states.reshape(-1, shape[-1])
    activation = dense_channels(mlp, flat)
    intermediate = activation.shape[-1]
    bounds = group_bounds(intermediate, groups)
    out_width = mlp.down_proj.out_features
    contributions = torch.empty(
        (flat.shape[0], len(bounds), out_width),
        dtype=torch.float32,
        device=flat.device,
    )
    norms = torch.empty(
        (flat.shape[0], len(bounds)), dtype=torch.float32, device=flat.device
    )
    weight = mlp.down_proj.weight
    for index, (start, end) in enumerate(bounds):
        chunk = activation[:, start:end]
        contributions[:, index, :] = chunk @ weight[:, start:end].transpose(0, 1)
        norms[:, index] = chunk.norm(dim=1)
    dense = contributions.sum(dim=1)
    gram = contributions @ contributions.transpose(1, 2)
    dot_dense = (contributions * dense.unsqueeze(1)).sum(dim=2)
    dense_sq = (dense * dense).sum(dim=1)
    return {
        "contributions": contributions,
        "dense": dense,
        "gram": gram,
        "dot_dense": dot_dense,
        "dense_sq": dense_sq,
        "group_norms": norms,
        "bounds": bounds,
        "activation": activation,
    }


def subset_local_error(stats: dict, selected: torch.Tensor) -> torch.Tensor:
    """Exact local squared error for a boolean `[T, G]` selected set."""
    gram = stats["gram"]
    dot = stats["dot_dense"]
    mask = selected.to(torch.float32)
    linear = (dot * mask).sum(dim=1)
    quadratic = torch.einsum("tg,tgj,tj->t", mask, gram, mask)
    return stats["dense_sq"] - 2.0 * linear + quadratic


def subset_selection_from_groups(selection: list[list[int]], groups: int) -> torch.Tensor:
    """A `[T, G]` boolean tensor from per-token group index lists."""
    tokens = len(selection)
    selected = torch.zeros((tokens, groups), dtype=torch.bool)
    for token, indices in enumerate(selection):
        for index in indices:
            if index < 0 or index >= groups:
                raise ValueError(f"group {index} is out of range 0..{groups}")
            selected[token, index] = True
    return selected
