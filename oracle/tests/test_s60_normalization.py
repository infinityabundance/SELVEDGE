"""S60 — the unit of learning must match the token accounting.

`token_loss` used to divide by the supervised positions of *each microbatch*
and the loop then divided the accumulated gradient by `grad_accum`. That makes
every example equally weighted regardless of length, while every receipt
reports supervised *tokens*. These tests pin the corrected contract:

* `--loss-unit token` sums CE over the optimiser step's supervised positions
  and divides by that step total, so two unequal-length microbatches have
  exactly the gradient of one global batch;
* the historical example average is still available and is *not* equal to the
  token normalisation on unequal lengths;
* `auto` resolves to `token` for the completion-masked SFT objectives and to
  `example` for the earlier `ce_kl`/`correction` arms, so their recipes and
  config hashes do not move.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import same_prefix, train  # noqa: E402

VOCAB = 11


def _fixture():
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, VOCAB, requires_grad=True)
    tokens_a = torch.randn(1, 3, VOCAB)
    tokens_b = torch.randn(1, 6, VOCAB)
    targets_a = torch.tensor([[-100, 2, 3]])
    targets_b = torch.tensor([[-100, 1, 2, 3, 4, 5]])
    total = int((targets_a != -100).sum() + (targets_b != -100).sum())
    return weight, tokens_a, tokens_b, targets_a, targets_b, total


def test_accumulated_token_normalisation_equals_one_global_batch():
    weight, tokens_a, tokens_b, targets_a, targets_b, total = _fixture()
    assert total == 7

    loss_a, _, _ = train.token_loss(tokens_a @ weight, None, targets_a, 1.0, 0.0,
                                    normaliser=total)
    loss_b, _, _ = train.token_loss(tokens_b @ weight, None, targets_b, 1.0, 0.0,
                                    normaliser=total)
    (loss_a + loss_b).backward()
    accumulated = weight.grad.clone()

    weight.grad = None
    tokens = torch.cat([tokens_a, tokens_b], dim=1)
    targets = torch.cat([targets_a, targets_b], dim=1)
    global_loss, _, _ = train.token_loss(tokens @ weight, None, targets, 1.0, 0.0,
                                         normaliser=total)
    global_loss.backward()

    assert torch.allclose(loss_a + loss_b, global_loss, atol=1e-6)
    assert torch.allclose(accumulated, weight.grad, atol=1e-6), (
        "two token-normalised microbatches must produce the global-batch gradient")


def test_example_average_is_not_token_normalised_on_unequal_lengths():
    weight, tokens_a, tokens_b, targets_a, targets_b, total = _fixture()
    loss_a, _, _ = train.token_loss(tokens_a @ weight, None, targets_a, 1.0, 0.0)
    loss_b, _, _ = train.token_loss(tokens_b @ weight, None, targets_b, 1.0, 0.0)
    example_step = 0.5 * (loss_a + loss_b)

    tokens = torch.cat([tokens_a, tokens_b], dim=1)
    targets = torch.cat([targets_a, targets_b], dim=1)
    token_step, _, _ = train.token_loss(tokens @ weight, None, targets, 1.0, 0.0,
                                        normaliser=total)
    assert not torch.isclose(example_step, token_step, atol=1e-5)


def test_same_prefix_ce_uses_the_step_normaliser():
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, VOCAB, requires_grad=True)
    tokens_a = torch.randn(1, 4, VOCAB)
    tokens_b = torch.randn(1, 7, VOCAB)
    targets_a = torch.tensor([[-100, 1, 2, 3]])
    targets_b = torch.tensor([[-100, 4, 5, 6, 7, 8, 9]])
    total = 3 + 6
    ce_a = same_prefix.completion_ce(tokens_a @ weight, targets_a,
                                     normaliser=total)
    ce_b = same_prefix.completion_ce(tokens_b @ weight, targets_b,
                                     normaliser=total)
    (ce_a + ce_b).backward()
    accumulated = weight.grad.clone()

    weight.grad = None
    tokens = torch.cat([tokens_a, tokens_b], dim=1)
    targets = torch.cat([targets_a, targets_b], dim=1)
    ce = same_prefix.completion_ce(tokens @ weight, targets, normaliser=total)
    ce.backward()
    assert torch.allclose(ce_a + ce_b, ce, atol=1e-6)
    assert torch.allclose(accumulated, weight.grad, atol=1e-6)


def test_auto_unit_keeps_historical_arms_on_example_average():
    assert train.resolve_loss_unit("auto", "sft") == "token"
    assert train.resolve_loss_unit("auto", "sft-kl") == "token"
    assert train.resolve_loss_unit("auto", "ce_kl") == "example"
    assert train.resolve_loss_unit("auto", "correction") == "example"
    assert train.resolve_loss_unit("example", "sft") == "example"
    assert train.resolve_loss_unit("token", "sft") == "token"


def test_length_aware_plan_groups_rows_and_trims_each_microbatch():
    from types import SimpleNamespace

    from frontier import sft_data

    packed = sft_data.pack_examples(
        [{"prompt_ids": [1], "solution_ids": [2] * length}
         for length in (1, 2, 8, 9)],
        seq_len=16, pad_id=0, eos_id=7)
    plan = train.length_aware_plan(packed, rows_per_step=2, seed=0)
    assert plan["steps"] == 2
    assert plan["leftover_rows"] == 0
    assert plan["min_group_width"] == 4
    assert plan["max_group_width"] == 11
    # The plan is a pure function of the rows and the seed, so a resumed run
    # rebuilds the same step order.
    again = train.length_aware_plan(packed, rows_per_step=2, seed=0)
    assert plan["groups"] == again["groups"]

    args = SimpleNamespace(grad_accum=2, micro_batch=1, seq_len=16)
    totals = []
    for step in (0, 1):
        batches, _cursor, _offset = train.materialise_step(
            packed, None, step=step, row_cursor=0, offset=0, plan=plan,
            args=args)
        assert len(batches) == 2
        for inputs, targets in batches:
            # Each microbatch is trimmed to its own width, so no row carries the
            # full padded 15 positions.
            assert inputs.shape[1] < 15
            assert inputs.shape[1] == targets.shape[1]
        totals.append(train.step_supervised_positions(batches))
    # supervised positions: (solution length + EOS) per row, grouped in pairs
    assert sorted(totals) == [2 + 3, 9 + 10]
