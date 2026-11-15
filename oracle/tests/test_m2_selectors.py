"""Court B tests for the M2 selector and oracle helpers.

These are CPU tests on a tiny synthetic SwiGLU. The local-error identity, the
best-k search and the extra-group allocation are checked against brute force;
the learned greedy and the deterministic random arm are checked for
determinism and budget shape.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

import masked_mlp
import m2_selectors


class Tiny(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator)
                )
        self.act_fn = F.silu


def small_stats(groups: int, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    mlp = Tiny(8, groups * 4, seed)
    hidden = torch.randn(5, 8)
    stats = masked_mlp.group_contributions(mlp, hidden, groups)
    return stats


def test_local_error_identity_matches_explicit_sum() -> None:
    groups = 4
    stats = small_stats(groups)
    for size in (1, 2, 3):
        for subset in itertools.combinations(range(groups), size):
            selected = torch.zeros((5, groups), dtype=torch.bool)
            for index in subset:
                selected[:, index] = True
            error = masked_mlp.subset_local_error(stats, selected)
            explicit = (
                (
                    stats["dense"]
                    - stats["contributions"][:, list(subset), :].sum(dim=1)
                )
                ** 2
            ).sum(dim=1)
            assert torch.allclose(error, explicit, atol=1e-2, rtol=1e-4)


def test_best_k_selection_matches_brute_force() -> None:
    groups = 4
    stats = small_stats(groups)
    selected, error = m2_selectors.best_k_selection(stats, 2)
    subsets = list(itertools.combinations(range(groups), 2))
    errors = []
    for subset in subsets:
        mask = torch.zeros((5, groups), dtype=torch.bool)
        for index in subset:
            mask[:, index] = True
        errors.append(masked_mlp.subset_local_error(stats, mask))
    brute = torch.stack(errors, dim=1).min(dim=1).values
    assert torch.allclose(error, brute, atol=1e-2, rtol=1e-4)
    assert (selected.sum(dim=1) == 2).all()


def test_extra_group_allocation_is_deterministic() -> None:
    groups = 8
    tokens = 6
    scores = [
        torch.rand((tokens, groups), generator=torch.Generator().manual_seed(layer))
        for layer in range(4)
    ]
    selection = [
        m2_selectors.topk_mask(score, 2) for score in scores
    ]
    first = m2_selectors.best_extra_pairs(scores, selection, 2, largest=True)
    second = m2_selectors.best_extra_pairs(scores, selection, 2, largest=True)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    totals = torch.stack([row.sum(dim=1) for row in first], dim=1).sum(dim=1)
    assert torch.equal(totals, torch.full((tokens,), 2 * len(first) + 2))
    assert all(int(row.sum()) >= 2 for layer in first for row in layer)


def test_learned_greedy_is_deterministic_and_budget_shaped() -> None:
    tokens, groups, features = 3, 6, 5
    unary = torch.rand((tokens, groups), generator=torch.Generator().manual_seed(3))
    phi = torch.randn((tokens, groups, features), generator=torch.Generator().manual_seed(4))
    matrix = torch.eye(features)
    first = m2_selectors.greedy_surrogate_selection(unary, phi, matrix, 0.5, 3)
    second = m2_selectors.greedy_surrogate_selection(unary, phi, matrix, 0.5, 3)
    assert torch.equal(first, second)
    assert (first.sum(dim=1) == 3).all()


def test_topk_tie_breaks_by_group_index() -> None:
    scores = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    selected = m2_selectors.topk_mask(scores, 2)
    assert torch.equal(
        torch.nonzero(selected[0]).flatten(), torch.tensor([0, 1])
    )


def test_random_selection_has_the_registered_budget() -> None:
    accounting = {"group_count": 16}
    input_ids = torch.tensor([[11, 12, 13, 14, 15]])
    selection = m2_selectors.random_selection(input_ids, accounting, 2, torch.device("cpu"))
    counts = torch.stack([row.sum(dim=1) for row in selection], dim=1).sum(dim=1)
    assert torch.equal(counts, torch.full((5,), 142))
    again = m2_selectors.random_selection(input_ids, accounting, 2, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(selection, again))


def test_task_static_selection_has_the_registered_budget() -> None:
    statistics = {
        "groups": 16,
        "layers": 28,
        "documents": {
            "doc": {
                "mean_norm": [
                    [float(group + layer) for group in range(16)]
                    for layer in range(28)
                ]
            }
        },
    }
    selection = m2_selectors.task_static_selection(
        {"statistics": statistics}, "doc", 2
    )
    assert sum(len(row) for row in selection) == 142
    assert all(len(set(row)) == len(row) for row in selection)


def test_onepass_physical_reason_follows_the_evidence() -> None:
    magnitude_reason, magnitude_detail = m2_selectors.onepass_physical_verdict(
        {"selector_elements_read": 4096}
    )
    assert magnitude_reason == "selection_reads_outside_active_set"
    assert "gate/up" in magnitude_detail
    random_reason, random_detail = m2_selectors.onepass_physical_verdict(
        {"selector_elements_read": 0}
    )
    assert random_reason == "dense_computation_masked"
    assert "reads no learned parameters" in random_detail
    assert m2_selectors.onepass_physical_verdict(None)[0] == "dense_computation_masked"


def test_physical_unique_text_never_invents_a_number() -> None:
    assert (
        m2_selectors.physical_unique_text(
            {"accounting": {"executed_unique_elements": 7}}
        )
        == "7"
    )
    assert m2_selectors.physical_unique_text({"accounting": None}) == "absent"
    assert m2_selectors.physical_unique_text({}) == "absent"
