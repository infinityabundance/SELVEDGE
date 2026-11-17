"""S51: the pre-read selection mask is an enforced interface, not a convention.

`TwoStageUtilityRouter` previously computed the post-read prediction for every
token and relied on the caller to have fetched only the selected ones. These
tests pin the corrected contract:

* `select` returns a boolean `selection_mask` and the caller must pass it to
  `forward`;
* an unselected token receives exactly zero contribution, even when the
  post-read refinement would open its gate;
* `execute` calls the read path once, with exactly the selected flat indices, so
  an unselected token is never requested;
* physical-read accounting is a separate interface counted in unique rows,
  tier bytes and cache hits, because `measured_cost(read)` is still a surrogate.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import utility_routing as routing  # noqa: E402


def _metadata(batch: int = 1, positions: int = 4):
    return {
        "position_fraction": torch.linspace(0.0, 1.0, positions).expand(batch, positions),
        "budget_fraction": torch.full((batch, positions), 1.0),
    }


def _router(hidden: int = 4, use_post_read: bool = True):
    return routing.TwoStageUtilityRouter(hidden=hidden, use_post_read=use_post_read)


def test_forward_requires_a_bool_mask_on_the_token_grid():
    router = _router()
    hidden = torch.randn(1, 4, 4)
    read = torch.randn(1, 4, 4)
    metadata = _metadata()
    with pytest.raises(TypeError):
        router(hidden, metadata, read)  # the mask is not optional
    with pytest.raises(ValueError):
        router(hidden, metadata, read, torch.ones(1, 4))  # float mask
    with pytest.raises(ValueError):
        router(hidden, metadata, read, torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        router(hidden, metadata, read, torch.ones(4, dtype=torch.bool))


def test_unselected_tokens_receive_exactly_zero_contribution():
    torch.manual_seed(11)
    router = _router()
    # A post-read stage that would admit everything if the mask were ignored.
    with torch.no_grad():
        router.post.predictor[-1].bias.fill_(20.0)
    hidden = torch.randn(1, 4, 4)
    read = torch.full((1, 4, 4), 100.0)
    mask = torch.tensor([[True, False, True, False]])
    output = router(hidden, _metadata(), read, mask, cost=0.0)
    assert output.shape == read.shape
    assert torch.equal(output[0, 1], torch.zeros(4))
    assert torch.equal(output[0, 3], torch.zeros(4))
    # selected tokens still receive a contribution
    assert bool((output[0, 0].abs() > 0).all())
    assert bool((output[0, 2].abs() > 0).all())


def test_execute_requests_only_selected_indices_once():
    router = _router()
    hidden = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, False, True, False]])
    calls: list[torch.Tensor] = []

    def read_path(indices):
        calls.append(indices.clone())
        return torch.arange(indices.numel() * 4, dtype=torch.float32).reshape(
            indices.numel(), 4)

    # Pin the selection so the spy's expectations are exact.
    router.select = lambda *args, **kwargs: (
        mask, torch.zeros(1, 4), torch.ones(1, 4))

    execution = router.execute(hidden, _metadata(), read_path, cost=0.0,
                               row_bytes_by_tier={"hot": 64},
                               row_ids=torch.tensor([0, 5, 0, 9]))
    assert len(calls) == 1
    assert calls[0].tolist() == [0, 2]
    assert execution.requested_indices.tolist() == [0, 2]
    # only the requested rows were scattered in; unselected reads are zero
    assert torch.equal(execution.contribution[0, 1], torch.zeros(4))
    assert torch.equal(execution.contribution[0, 3], torch.zeros(4))
    assert execution.accounting.requests == 2
    assert execution.accounting.unique_rows_requested == 1  # rows 0 and 1 -> {0}
    assert execution.accounting.bytes_by_tier == {"hot": 64}


def test_execute_rejects_a_read_path_with_the_wrong_shape():
    router = _router()
    hidden = torch.randn(1, 3, 4)
    router.select = lambda *args, **kwargs: (
        torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3),
        torch.ones(1, 3))
    with pytest.raises(ValueError):
        router.execute(hidden, _metadata(positions=3), lambda indices: torch.zeros(3, 5))


def test_two_stage_refinement_cannot_resurrect_a_rejected_token():
    torch.manual_seed(5)
    router = _router()
    hidden = torch.randn(1, 4, 4)
    read = torch.randn(1, 4, 4)
    reject = torch.tensor([[True, False, True, True]])

    # With a strong positive post-read prediction the gate is ~1 everywhere.
    with torch.no_grad():
        router.post.predictor[-1].bias.fill_(8.0)
    output = router(hidden, _metadata(), read, reject, cost=0.0)
    assert float(router.last_gate[0, 1]) > 0.99  # the gate *would* open
    assert torch.equal(output[0, 1], torch.zeros(4))  # but the token is rejected
    # Changing the read at the rejected token cannot create a contribution.
    read_changed = read.clone()
    read_changed[0, 1] = 1e6
    output_changed = router(hidden, _metadata(), read_changed, reject, cost=0.0)
    assert torch.equal(output_changed[0, 1], torch.zeros(4))


def test_routing_is_deterministic():
    torch.manual_seed(3)
    router = _router()
    with torch.no_grad():
        router.pre.predictor[0].weight.normal_(0.0, 0.1)
        router.post.predictor[0].weight.normal_(0.0, 0.1)
    hidden = torch.randn(2, 5, 4)
    read = torch.randn(2, 5, 4)
    metadata = _metadata(batch=2, positions=5)
    first_mask, first_prediction, first_gate = router.select(hidden, metadata)
    first_output = router(hidden, metadata, read, first_mask)
    second_mask, second_prediction, second_gate = router.select(hidden, metadata)
    second_output = router(hidden, metadata, read, second_mask)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_prediction, second_prediction)
    assert torch.equal(first_gate, second_gate)
    assert torch.equal(first_output, second_output)


def test_physical_read_accounting_counts_unique_rows_tiers_and_cache_hits():
    mask = torch.tensor([[True, True, True, True, False, False]])
    row_ids = torch.tensor([7, 7, 9, 11, 11, 11])
    accounting = routing.physical_read_accounting(
        mask, row_bytes_by_tier={"hot": 256, "cold": 1024},
        cache_hits=1, row_ids=row_ids)
    assert accounting.tokens == 6
    assert accounting.requests == 4
    assert accounting.unique_rows_requested == 3  # {7, 9, 11}
    assert accounting.bytes_by_tier == {"hot": 3 * 256, "cold": 3 * 1024}
    assert accounting.cache_hits == 1
    assert accounting.cache_misses == 3
    assert "surrogate" in accounting.basis
    with pytest.raises(ValueError):
        routing.physical_read_accounting(mask, cache_hits=5)
    with pytest.raises(ValueError):
        routing.physical_read_accounting(mask, row_ids=torch.zeros(5, dtype=torch.long))


def test_accounting_without_row_ids_is_conservative_and_says_so():
    mask = torch.tensor([[True, False, True]])
    accounting = routing.physical_read_accounting(mask)
    assert accounting.requests == 2
    assert accounting.unique_rows_requested == 2
    assert accounting.bytes_by_tier == {}
    assert "one row per selected token" in accounting.basis
