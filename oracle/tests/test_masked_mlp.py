"""Court B unit tests for the masked-MLP instrument.

Every test uses a tiny synthetic model whose group contributions are computed by
hand in float64, so the ground truth does not come from the instrument under
test. The suite runs on CPU and needs no network, model or dataset.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import masked_mlp


class TinyMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
                )
        self.act_fn = F.silu
        self.intermediate_size = intermediate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def hand_grouped(mlp: TinyMlp, x: torch.Tensor, groups: int, selected: list[int]) -> torch.Tensor:
    """Explicit float64 group sum, used as the hand-computed truth."""
    bounds = masked_mlp.group_bounds(mlp.intermediate_size, groups)
    flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
    gate = flat @ mlp.gate_proj.weight.to(torch.float64).T
    up = flat @ mlp.up_proj.weight.to(torch.float64).T
    activation = F.silu(gate) * up
    out = torch.zeros(
        (flat.shape[0], mlp.down_proj.out_features), dtype=torch.float64
    )
    for index in selected:
        start, end = bounds[index]
        out += activation[:, start:end] @ mlp.down_proj.weight[:, start:end].to(
            torch.float64
        ).T
    return out.reshape(*x.shape[:-1], mlp.down_proj.out_features)


def test_court_b_group_bounds_cover_every_channel() -> None:
    bounds = masked_mlp.group_bounds(6144, 16)
    assert len(bounds) == 16
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 6144
    assert all(end - start == 384 for start, end in bounds)
    for (_, end), (start, _) in zip(bounds, bounds[1:]):
        assert end == start


def test_court_b_group_bounds_ceil_policy_matches_converter() -> None:
    bounds = masked_mlp.group_bounds(10, 4)
    assert [end - start for start, end in bounds] == [3, 3, 3, 1]
    assert bounds[-1][1] == 10


def test_court_b_grouped_forward_matches_hand_computation() -> None:
    torch.manual_seed(0)
    mlp = TinyMlp(hidden=8, intermediate=12, seed=7)
    x = torch.randn(3, 5, 8)
    for selected in ([0, 3], [1, 2, 3], [0, 1, 2, 3]):
        got = masked_mlp.grouped_forward(mlp, x, 4, selected)
        want = hand_grouped(mlp, x, 4, selected)
        assert torch.allclose(got.to(torch.float64), want, atol=1e-5, rtol=1e-5)


def test_court_b_partial_sum_is_not_the_dense_output() -> None:
    mlp = TinyMlp(hidden=8, intermediate=12, seed=11)
    x = torch.randn(2, 8)
    dense = hand_grouped(mlp, x, 4, [0, 1, 2, 3])
    partial = hand_grouped(mlp, x, 4, [0, 2])
    assert not torch.allclose(dense, partial, atol=1e-3, rtol=1e-3)


def test_court_b_selected_mask_is_a_union_of_whole_groups() -> None:
    mask = masked_mlp.selected_mask(12, 4, [1, 3])
    assert mask.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        masked_mlp.selected_mask(12, 4, [4])
    selection = masked_mlp.selection_from_masks([mask], 4)
    assert selection == [[1, 3]]


def test_court_b_mask_controller_all_ones_is_bitwise_dense() -> None:
    class Layer(torch.nn.Module):
        def __init__(self, mlp: TinyMlp) -> None:
            super().__init__()
            self.mlp = mlp

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(TinyMlp(8, 12, 3))])

    model = Model()
    x = torch.randn(1, 4, 8)
    with torch.inference_mode():
        dense = model.model.layers[0].mlp(x).clone()
    controller = masked_mlp.MaskController(model)
    try:
        controller.set_dense()
        with torch.inference_mode():
            masked = model.model.layers[0].mlp(x)
        assert torch.equal(masked, dense)
    finally:
        controller.restore()


def test_court_b_mask_controller_subset_matches_grouped_reference() -> None:
    mlp = TinyMlp(8, 12, 5)
    controller_mlp = TinyMlp(8, 12, 5)
    controller_mlp.load_state_dict(mlp.state_dict())

    class Layer(torch.nn.Module):
        def __init__(self, module: TinyMlp) -> None:
            super().__init__()
            self.mlp = module

    class Model(torch.nn.Module):
        def __init__(self, module: TinyMlp) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(module)])

    model = Model(controller_mlp)
    controller = masked_mlp.MaskController(model)
    x = torch.randn(2, 6, 8)
    try:
        controller.set_masks([masked_mlp.selected_mask(12, 4, [0, 2])])
        with torch.inference_mode():
            fast = model.model.layers[0].mlp(x)
    finally:
        controller.restore()
    reference = masked_mlp.grouped_forward(mlp, x, 4, [0, 2])
    assert torch.allclose(fast, reference, atol=1e-5, rtol=1e-5)


def test_court_b_controller_suspend_restores_the_dense_module() -> None:
    class Layer(torch.nn.Module):
        def __init__(self, mlp: TinyMlp) -> None:
            super().__init__()
            self.mlp = mlp

    class Model(torch.nn.Module):
        def __init__(self, mlp: TinyMlp) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(mlp)])

    mlp = TinyMlp(8, 12, 41)
    model = Model(mlp)
    x = torch.randn(1, 4, 8)
    controller = masked_mlp.MaskController(model)
    assert model.model.layers[0].mlp is not mlp
    controller.suspend()
    assert model.model.layers[0].mlp is mlp
    with torch.inference_mode():
        dense = model.model.layers[0].mlp(x)
    controller.resume()
    assert isinstance(model.model.layers[0].mlp, masked_mlp.MaskedMlp)
    controller.set_dense()
    with torch.inference_mode():
        wrapped = model.model.layers[0].mlp(x)
    assert torch.equal(dense, wrapped)


def test_court_b_empty_selection_is_exactly_zero() -> None:
    mlp = TinyMlp(8, 12, 13)
    x = torch.randn(2, 5, 8)
    out = masked_mlp.grouped_forward(mlp, x, 4, [])
    assert torch.count_nonzero(out) == 0


def test_court_b_error_propagates_through_a_second_layer() -> None:
    """A masked first layer feeds a second layer; both are hand-checked.

    The test asserts the composition equals the hand-computed masked
    composition, not the dense one, so a silently "corrected" error would fail.
    """
    torch.manual_seed(2)
    hidden = 8
    mlp1 = TinyMlp(hidden, 12, 21)
    mlp2 = TinyMlp(hidden, 12, 22)
    x = torch.randn(1, 4, hidden)
    selection1 = [0, 1]
    selection2 = [2, 3]
    with torch.inference_mode():
        h1_dense = x + mlp1(x)
        h1_masked = x + masked_mlp.grouped_forward(mlp1, x, 4, selection1)
        dense2 = h1_masked + mlp2(h1_masked)
        masked2 = h1_masked + masked_mlp.grouped_forward(mlp2, h1_masked, 4, selection2)
    want_h1 = x.to(torch.float64) + hand_grouped(mlp1, x, 4, selection1)
    want_h2 = want_h1 + hand_grouped(mlp2, want_h1.to(torch.float32), 4, selection2)
    assert not torch.allclose(h1_dense, h1_masked, atol=1e-3)
    assert not torch.allclose(dense2, masked2, atol=1e-3)
    assert torch.allclose(masked2.to(torch.float64), want_h2, atol=1e-5, rtol=1e-5)


def test_court_b_active_parameter_report_counts_selected_groups() -> None:
    selection = [[0, 1, 2] for _ in range(4)]
    report = masked_mlp.active_parameter_report(
        selection, hidden=8, intermediate=12, groups=3, backbone_elements=100
    )
    assert report["group_count"] == 12


def test_court_b_active_parameter_report_exact_elements() -> None:
    selection = [[0], [1, 2]]
    report = masked_mlp.active_parameter_report(
        selection, hidden=8, intermediate=12, groups=3, backbone_elements=1000
    )
    # Group widths are [4, 4, 4]; three groups selected => 3 * 3 * 4 * 8.
    assert report["active_ffn_elements"] == 3 * 3 * 4 * 8
    assert report["active_unique_elements"] == 1000 + 288
    assert report["ffn_matmul_flops_per_token"] == 6 * 12 * 8


def test_court_b_kl_of_identical_distributions_is_zero() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 0.5, 0.0], [1.0, 1.0, 1.0]]])
    ids = torch.tensor([[0, 1, 2]])
    metrics = masked_mlp.logits_kl_metrics(logits, logits, ids)
    assert torch.allclose(metrics["kl"], torch.zeros_like(metrics["kl"]), atol=1e-7)
    assert float(metrics["top1"].min()) == 1.0
    probs = torch.softmax(logits, dim=-1)[0, :-1, :]
    entropy = -(probs * probs.log()).sum(dim=-1)
    assert torch.allclose(metrics["entropy"], entropy, atol=1e-6)


def test_court_b_kl_matches_hand_computation() -> None:
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.5, 0.3, 0.2])
    expected = float((p * (p.log() - q.log())).sum())
    teacher = p.log().reshape(1, 1, 3).repeat(1, 3, 1)
    masked = q.log().reshape(1, 1, 3).repeat(1, 3, 1)
    ids = torch.tensor([[0, 1, 2]])
    metrics = masked_mlp.logits_kl_metrics(teacher, masked, ids)
    assert torch.allclose(
        metrics["kl"], torch.full_like(metrics["kl"], expected), atol=1e-6
    )
    assert math.isclose(expected, 0.08512282595722162, rel_tol=1e-6)


def test_court_b_nll_pair_excludes_the_first_predicting_position() -> None:
    teacher = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    masked = torch.tensor([[[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])
    ids = torch.tensor([[0, 1, 0]])
    metrics = masked_mlp.logits_kl_metrics(teacher, masked, ids)
    assert metrics["nll_teacher"].numel() == 1
    assert metrics["kl"].numel() == 2


def test_court_b_random_rankings_are_deterministic_and_per_layer() -> None:
    left = masked_mlp.random_rankings(3, 4, 20261114)
    right = masked_mlp.random_rankings(3, 4, 20261114)
    assert left == right
    assert all(sorted(row) == [0, 1, 2, 3] for row in left)


def test_court_b_rankings_put_the_largest_first() -> None:
    statistics = {
        "groups": 2,
        "layers": 1,
        "layers_data": [
            {"layer": 0, "mean_norm": [1.0, 5.0], "frequency": [0.9, 0.1]}
        ],
    }
    result = masked_mlp.rankings(statistics)
    assert result["magnitude"] == [[1, 0]]
    assert result["frequency"] == [[0, 1]]


def _per_token_selection(tokens: int, groups: int, pattern: list[list[int]]) -> torch.Tensor:
    selection = torch.zeros((tokens, groups), dtype=torch.bool)
    for token in range(tokens):
        for index in pattern[token % len(pattern)]:
            selection[token, index] = True
    return selection


def test_court_b_sparse_path_matches_masked_dense_for_per_token_subsets() -> None:
    torch.manual_seed(3)
    mlp = TinyMlp(hidden=8, intermediate=12, seed=11)
    x = torch.randn(1, 7, 8)
    groups = 4
    selection = _per_token_selection(7, groups, [[0, 2], [1], [0, 1, 3], []])
    masked = torch.stack(
        [
            masked_mlp.grouped_forward(mlp, x[0, token : token + 1], groups, sorted(
                torch.nonzero(selection[token]).flatten().tolist()
            )).squeeze(0)
            if bool(selection[token].any())
            else torch.zeros_like(mlp(x[0, token : token + 1])[0])
            for token in range(7)
        ]
    ).unsqueeze(0)
    sparse = masked_mlp.sparse_grouped_forward(mlp, x, groups, selection)
    assert sparse.shape == x.shape[:-1] + (mlp.down_proj.out_features,)
    assert torch.allclose(sparse, masked, atol=1e-5, rtol=1e-5)


def test_court_b_sparse_path_matches_dense_mask_including_empty_rows() -> None:
    torch.manual_seed(5)
    mlp = TinyMlp(hidden=6, intermediate=10, seed=13)
    wrapper = masked_mlp.MaskedMlp(mlp)
    x = torch.randn(1, 5, 6)
    groups = 5
    selection = _per_token_selection(5, groups, [[0], [], [2, 4], [0, 1, 2, 3, 4]])
    bounds = masked_mlp.group_bounds(10, groups)
    mask = torch.zeros((5, 10))
    for index, (start, end) in enumerate(bounds):
        mask[:, start:end] = selection[:, index : index + 1].to(torch.float32)
    wrapper.set_token_mask(mask)
    dense = wrapper(x)
    wrapper.clear_token_mask()
    wrapper.set_sparse_selection(selection, groups)
    sparse = wrapper(x)
    assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)


def test_court_b_sparse_read_count_is_exactly_the_sliced_elements() -> None:
    torch.manual_seed(7)
    mlp = TinyMlp(hidden=8, intermediate=12, seed=17)
    groups = 3
    bounds = masked_mlp.group_bounds(12, groups)  # widths [4, 4, 4]
    selection = torch.zeros((4, groups), dtype=torch.bool)
    selection[0, 0] = True
    selection[1, 0] = True
    selection[2, 2] = True
    selection[3, 1] = True
    counter = masked_mlp.ReadAccounting()
    masked_mlp.sparse_grouped_forward(
        mlp, torch.randn(1, 4, 8), groups, selection, counter=counter
    )
    # Two tokens share group 0 (executed together once), one token each uses
    # groups 2 and 1: three sliced executions of width 4.
    per_row = 4 * (2 * 8 + 8)
    # Three distinct subsets, each executed once: group 0 is shared by two
    # tokens, and a batched matmul reads its weight slice once.
    assert counter.execution_elements_read == 3 * per_row
    assert counter.execution_flops == 2 * 4 * per_row
    assert counter.matmul_calls == 3 * 3


def test_court_b_sparse_read_audit_matches_the_counter() -> None:
    torch.manual_seed(9)
    mlp = TinyMlp(hidden=8, intermediate=12, seed=19)
    groups = 3
    selection = _per_token_selection(5, groups, [[0], [1, 2], [0, 2]])
    audit = masked_mlp.audit_sparse_reads(
        mlp, torch.randn(2, 5, 8), groups, selection.repeat(2, 1)
    )
    assert audit["matches"], audit
    assert audit["counted_elements"] == audit["profiled_elements"]
    assert audit["counted_flops"] == audit["profiled_flops"]


def test_court_b_dense_and_selector_accounting_formulas_match_shapes() -> None:
    elements, flops = masked_mlp.dense_execution_accounting(
        tokens=3, hidden=2048, intermediate=6144, out_width=2048, layers=28
    )
    assert elements == 28 * 6144 * (2 * 2048 + 2048)
    assert flops == 2 * 3 * elements
    selector_elements, selector_flops = masked_mlp.selector_dense_accounting(
        tokens=3, hidden=2048, intermediate=6144, layers=28
    )
    assert selector_elements == 28 * 2 * 2048 * 6144
    assert selector_flops == 2 * 3 * selector_elements
