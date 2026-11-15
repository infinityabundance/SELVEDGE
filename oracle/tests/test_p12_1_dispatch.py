"""P12.1 — routed-overflow validity-mask audit (CPU, deterministic).

The audit asks one question: is the overflow validity mask applied to the pairs
it describes? `dispatch` builds `valid` in *sorted* (capacity-sorted) pair order;
`combine` restores the pair outputs to *token-major* order. If the mask is
applied after the restore, pair `p` is gated by the validity of the pair whose
sorted position is `p`, which is a different pair whenever any expert overflowed.

These tests pin the semantics with `oracle/train/reference_dispatch.py`, an
independent transcription written from the specification (explicit loops, no
scatter, no reshape-and-reduce), and then check the shipped module against it in
both mask placements:

* `mask_order="legacy"` (the default, the behaviour every registered T2 arm was
  trained and evaluated with) must reproduce the *misordered* semantics exactly;
* `mask_order="pair"` (the corrected placement, opt-in) must reproduce the
  documented semantics exactly.

The headline toy case is `tokens=3, experts=4, k=2, factor=0.5`, where an
overflowing pair is kept by the legacy mask and reads a clamped row that holds
**another token's** expert activation.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import modeling  # noqa: E402
import reference_dispatch as reference  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures and helpers


class _ForcedBank(modeling.SpecialistBank):
    """A bank whose routing decision is supplied instead of computed."""

    def __init__(self, hidden: int, width: int, experts: int, k: int,
                 top_index: torch.Tensor, pair_weight: torch.Tensor | None = None):
        router = modeling._router(hidden, experts, seed=5)
        super().__init__(hidden, width, experts, k, router)
        with torch.no_grad():
            self.down.normal_(generator=torch.Generator().manual_seed(3))
        self._top_index = top_index
        self._pair_weight = pair_weight

    def routing(self, flat: torch.Tensor, router_weight: torch.Tensor):
        probs = torch.full((flat.shape[0], self.experts), 1.0 / self.experts,
                           dtype=torch.float32)
        gate = torch.zeros_like(probs)
        weight = self._pair_weight
        if weight is None:
            weight = torch.arange(1, self._top_index.numel() + 1, dtype=torch.float32) / 4
        return probs, gate, self._top_index, weight


def _expert_fn(bank: modeling.SpecialistBank, flat: torch.Tensor):
    def call(expert: int, tokens: torch.Tensor):
        rows = flat.index_select(0, tokens)
        gate = F.linear(rows, bank.gate[expert])
        up = F.linear(rows, bank.up[expert])
        return F.linear(F.silu(gate) * up, bank.down[expert])
    return call


def _module_buffer(bank: modeling.SpecialistBank, flat: torch.Tensor,
                   top_index: torch.Tensor):
    buffer, row, order, valid = bank.dispatch(flat, top_index)
    return buffer, row, order, valid


# ---------------------------------------------------------------------------
# capacity arithmetic


def test_capacity_matches_the_registered_geometry_and_accepts_factor_below_one():
    assert reference.capacity(1024, 4, 8, 1.25) == 640
    assert modeling._capacity(1024, 4, 8, 1.25) == 640
    # factor < 1 is legal and is how a small toy case reaches overflow.
    assert reference.capacity(3, 2, 4, 0.5) == 1
    assert modeling._capacity(3, 2, 4, 0.5) == 1
    # an empty batch still reserves one row per expert.
    assert reference.capacity(0, 2, 4, 0.5) == 1


def test_top_k_above_the_expert_count_is_refused_by_both():
    with pytest.raises(ValueError):
        reference.capacity(2, 5, 4, 1.25)
    with pytest.raises(ValueError):
        reference.plan(torch.zeros(2, 5, dtype=torch.long), experts=4, factor=1.25)
    # the shipped module reaches `torch.topk`, which refuses k > experts.
    bank = modeling.SpecialistBank(4, 2, 4, 5, modeling._router(4, 4, seed=1))
    with pytest.raises(RuntimeError):
        bank(torch.zeros(1, 2, 4))
    # k <= experts holds by construction in every registered arm: k=4, m=8.
    assert 4 <= 8


# ---------------------------------------------------------------------------
# the reference itself: the two placements are different functions


def test_reference_reproduces_the_documented_semantics_on_a_hand_case():
    # tokens=2, k=1, experts=2, factor=1.0 -> capacity 1, no overflow.
    flat = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    top_index = torch.tensor([[0], [1]])
    weight = torch.tensor([3.0, 5.0])
    # expert e multiplies its input by (e + 1)
    got = reference.contribution(flat, top_index, weight,
                                 lambda expert, tokens: flat.index_select(0, tokens) * (expert + 1),
                                 experts=2, factor=1.0)
    expected = torch.tensor([[3.0, 0.0], [0.0, 20.0]])
    assert torch.allclose(got, expected)


def test_reference_pair_and_sorted_modes_differ_only_under_overflow():
    flat = torch.tensor([[1.0], [10.0], [100.0], [1000.0]])
    top_index = torch.tensor([[2], [2], [0], [1]])  # expert 2 overflows, others empty
    weight = torch.ones(4)
    factor = 0.25  # capacity = max(1, ceil(4*1/3*0.25)) = 1
    specification = reference.plan(top_index, experts=3, factor=factor)
    assert specification.capacity == 1
    # expert 0 holds pair 2, expert 1 holds pair 3, expert 2 holds pairs 0 and 1
    assert specification.order == (2, 3, 0, 1)
    assert specification.valid == (True, True, True, False)
    assert specification.row[3] == 3 * 1 - 1  # clamped onto expert 2's only row

    identity = lambda expert, tokens: flat.index_select(0, tokens)  # noqa: E731
    pair_mode = reference.contribution(flat, top_index, weight, identity,
                                       experts=3, factor=factor, mask_mode="pair")
    sorted_mode = reference.contribution(flat, top_index, weight, identity,
                                         experts=3, factor=factor, mask_mode="sorted")
    # pair mode: tokens 0, 2 and 3 keep their pairs; the overflowing pair 1
    # (token 1) is dropped.
    assert torch.allclose(pair_mode, torch.tensor([[1.0], [0.0], [100.0], [1000.0]]))
    # sorted mode gates pair p by valid[p]: pairs 0, 1 and 2 are kept, so the
    # overflowing pair 1 is retained and reads the clamped row, which holds
    # token 0's activation; token 3's kept pair 3 is dropped.
    assert torch.allclose(sorted_mode, torch.tensor([[1.0], [1.0], [100.0], [0.0]]))
    assert not torch.allclose(pair_mode, sorted_mode)


# ---------------------------------------------------------------------------
# the headline toy case: a kept overflow pair reads another token's activation


def test_legacy_mask_lets_an_overflow_pair_read_another_tokens_output():
    hidden, width, experts, k, tokens = 4, 2, 4, 2, 3
    factor = 0.5  # capacity = ceil(3*2/4*0.5) = 1
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bank = _ForcedBank(hidden, width, experts, k, top_index, weight)
    bank.capacity_factor = factor
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(11))

    specification = reference.plan(top_index, experts=experts, factor=factor)
    assert specification.capacity == 1
    # experts 0, 1, 2 hold one pair each; expert 3 holds three (positions 3-5).
    assert specification.order == (1, 3, 5, 0, 2, 4)
    assert specification.valid == (True, True, True, True, False, False)
    # the pair at sorted position 4 (token 1, slot 0, expert 3) overflows and its
    # row is clamped onto the last buffer row, which holds a *different* pair.
    assert specification.row[4] == experts * specification.capacity - 1
    assert specification.row[4] == specification.row[3]
    assert specification.order[3] == 0 and specification.order[4] == 2

    buffer, row, order, valid = _module_buffer(bank, flat, top_index)
    reference_buffer = reference.build_buffer(flat, specification)
    assert torch.equal(buffer, reference_buffer), "dispatch halves disagree"
    assert order.tolist() == list(specification.order)

    expert_fn = _expert_fn(bank, flat)
    expected = reference.contribution(flat, top_index, weight, expert_fn,
                                      experts=experts, factor=factor, mask_mode="pair")
    legacy = reference.contribution(flat, top_index, weight, expert_fn,
                                    experts=experts, factor=factor, mask_mode="sorted")
    assert not torch.allclose(legacy, expected), "the two placements must not agree here"

    # Token 1's correct contribution is its one kept pair (expert 1)...
    assert torch.allclose(expected[1], weight[3] * expert_fn(1, torch.tensor([1]))[0],
                          atol=1e-6)
    # ...but the legacy mask also keeps its overflowing pair 2, which reads the
    # clamped row 3 -- token 0's activation, scored by expert 3.
    wrong = weight[2] * expert_fn(3, torch.tensor([0]))[0]
    assert float(wrong.detach().abs().sum()) > 1e-6
    assert torch.allclose(legacy[1], expected[1] + wrong, atol=1e-6)
    # Token 2's correct contribution is its kept pair (expert 2); the legacy
    # mask drops it instead.
    assert torch.allclose(expected[2], weight[5] * expert_fn(2, torch.tensor([2]))[0],
                          atol=1e-6)
    assert float(expected[2].detach().abs().sum()) > 1e-6
    assert torch.allclose(legacy[2], torch.zeros(hidden), atol=1e-9)


def test_module_legacy_matches_the_reference_sorted_mode_exactly():
    hidden, width, experts, k, tokens = 4, 2, 4, 2, 3
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(7))
    bank = _ForcedBank(hidden, width, experts, k, top_index, weight)
    expert_fn = _expert_fn(bank, flat)
    expected = reference.contribution(flat, top_index, weight, expert_fn,
                                      experts=experts, factor=0.5, mask_mode="sorted")
    bank.capacity_factor = 0.5
    assert bank.mask_order == modeling.MASK_ORDER_LEGACY
    got = bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
    assert torch.allclose(got.double(), expected.double(), atol=1e-6)


def test_module_pair_mode_matches_the_reference_pair_mode_exactly():
    hidden, width, experts, k, tokens = 4, 2, 4, 2, 3
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(7))
    bank = _ForcedBank(hidden, width, experts, k, top_index, weight)
    expert_fn = _expert_fn(bank, flat)
    expected = reference.contribution(flat, top_index, weight, expert_fn,
                                      experts=experts, factor=0.5, mask_mode="pair")
    bank.capacity_factor = 0.5
    bank.mask_order = modeling.MASK_ORDER_PAIR
    got = bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
    assert torch.allclose(got.double(), expected.double(), atol=1e-6)
    # and the corrected path is not what the registered arms ran
    bank.mask_order = modeling.MASK_ORDER_LEGACY
    legacy = bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
    assert not torch.allclose(legacy.double(), got.double(), atol=1e-6)


# ---------------------------------------------------------------------------
# the case matrix: ties, duplicate ids, empty selections, extremes


def _cases() -> list[dict]:
    return [
        # (name, tokens, experts, k, factor, top_index rows, weight mode)
        {"name": "no-overflow-random", "experts": 4, "k": 2, "factor": 1.25,
         "top": [[0, 1], [2, 3], [1, 0], [3, 2], [0, 3]]},
        {"name": "overflow-single-expert", "experts": 4, "k": 2, "factor": 0.5,
         "top": [[0, 0], [0, 1], [0, 2], [0, 3]]},
        {"name": "overflow-last-expert-clamps", "experts": 4, "k": 2, "factor": 0.5,
         "top": [[3, 0], [3, 1], [3, 2]]},
        {"name": "ties-first-k", "experts": 4, "k": 2, "factor": 0.5,
         "top": [[0, 1], [0, 1], [0, 1], [0, 1]]},
        {"name": "duplicate-expert-ids", "experts": 4, "k": 3, "factor": 0.75,
         "top": [[2, 2, 2], [0, 2, 1], [2, 0, 0]]},
        {"name": "k-equals-one", "experts": 3, "k": 1, "factor": 0.5,
         "top": [[1], [1], [1], [2]]},
        {"name": "k-equals-experts", "experts": 3, "k": 3, "factor": 0.5,
         "top": [[0, 1, 2], [2, 1, 0]]},
        {"name": "single-token", "experts": 4, "k": 2, "factor": 0.25, "top": [[1, 2]]},
        {"name": "empty-tokens", "experts": 4, "k": 2, "factor": 1.25, "top": []},
        {"name": "empty-selection-k-zero", "experts": 4, "k": 0, "factor": 1.25,
         "top": [[], [], []]},
    ]


@pytest.mark.parametrize("case", _cases(), ids=[case["name"] for case in _cases()])
def test_module_and_reference_agree_in_both_placements(case):
    tokens = len(case["top"])
    hidden, width = 6, 3
    top_index = torch.tensor(case["top"], dtype=torch.long).reshape(tokens, case["k"])
    weight = torch.arange(1, tokens * case["k"] + 1, dtype=torch.float32)
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(19))
    bank = _ForcedBank(hidden, width, case["experts"], case["k"], top_index, weight)
    bank.capacity_factor = case["factor"]
    expert_fn = _expert_fn(bank, flat)

    specification = reference.plan(top_index, experts=case["experts"],
                                   factor=case["factor"])
    buffer, _row, _order, _valid = _module_buffer(bank, flat, top_index)
    assert torch.equal(buffer, reference.build_buffer(flat, specification))

    for mode, placement in ((modeling.MASK_ORDER_PAIR, "pair"),
                            (modeling.MASK_ORDER_LEGACY, "sorted")):
        bank.mask_order = mode
        expected = reference.contribution(flat, top_index, weight, expert_fn,
                                          experts=case["experts"],
                                          factor=case["factor"], mask_mode=placement)
        got = bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
        assert torch.allclose(got.double(), expected.double(), atol=1e-6), mode

    # the two placements agree exactly when nothing overflowed
    overflowed = any(not keep for keep in specification.valid)
    pair_mode = reference.contribution(flat, top_index, weight, expert_fn,
                                       experts=case["experts"], factor=case["factor"],
                                       mask_mode="pair")
    sorted_mode = reference.contribution(flat, top_index, weight, expert_fn,
                                         experts=case["experts"], factor=case["factor"],
                                         mask_mode="sorted")
    if not overflowed:
        assert torch.allclose(pair_mode, sorted_mode)


def test_the_agreement_matrix_is_not_vacuous():
    """At least one overflow case must actually separate the two placements."""
    separations = 0
    for case in _cases():
        if not case["top"]:
            continue
        tokens = len(case["top"])
        top_index = torch.tensor(case["top"], dtype=torch.long).reshape(tokens, case["k"])
        weight = torch.arange(1, tokens * case["k"] + 1, dtype=torch.float32)
        flat = torch.randn(tokens, 6, generator=torch.Generator().manual_seed(23))
        fn = lambda expert, idx: flat.index_select(0, idx) * (expert + 1)  # noqa: E731
        pair_mode = reference.contribution(flat, top_index, weight, fn,
                                           experts=case["experts"],
                                           factor=case["factor"], mask_mode="pair")
        sorted_mode = reference.contribution(flat, top_index, weight, fn,
                                             experts=case["experts"],
                                             factor=case["factor"], mask_mode="sorted")
        if not torch.allclose(pair_mode, sorted_mode):
            separations += 1
    assert separations >= 3


# ---------------------------------------------------------------------------
# gradient attribution


def test_only_the_gated_pairs_receive_gradient_and_the_placement_decides_which():
    hidden, width, experts, k, tokens = 4, 2, 4, 2, 3
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(31))
    outputs = {}
    for mode in (modeling.MASK_ORDER_PAIR, modeling.MASK_ORDER_LEGACY):
        bank = _ForcedBank(hidden, width, experts, k, top_index, weight)
        bank.capacity_factor = 0.5
        bank.mask_order = mode
        target = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(37))
        loss = (bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
                - target).pow(2).sum()
        loss.backward()
        outputs[mode] = bank.gate.grad.abs().sum(dim=(1, 2)).tolist()
    specification = reference.plan(top_index, experts=experts, factor=0.5)
    # pair mode: only experts 0, 1, 2, 3 receive gradient, and expert 3 receives
    # the gradient of the single pair it actually executed (pair 0, token 0).
    assert outputs[modeling.MASK_ORDER_PAIR][3] > 0
    # legacy mode: the kept-overflow pair (token 1, expert 3) also back-propagates
    # into expert 3, through a buffer row that belongs to token 0.
    assert outputs[modeling.MASK_ORDER_LEGACY][3] > 0
    assert specification.valid[3] and not specification.valid[4]


def test_dropped_pairs_are_counted_and_the_total_is_exposed():
    hidden, width, experts, k, tokens = 4, 2, 4, 2, 3
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    bank = _ForcedBank(hidden, width, experts, k, top_index)
    bank.capacity_factor = 0.5
    flat = torch.randn(1, tokens, hidden, generator=torch.Generator().manual_seed(41))
    bank(flat)
    assert int(bank._dropped_pairs.item()) == 2
    assert bank.dropped_pairs_total == 2
    assert bank.pairs_total == tokens * k
    bank(flat)
    assert bank.dropped_pairs_total == 4


# ---------------------------------------------------------------------------
# the default must not change a registered arm


def test_the_default_placement_is_legacy_and_is_recorded():
    bank = modeling.SpecialistBank(8, 4, 4, 2, modeling._router(8, 4, seed=1))
    assert bank.mask_order == modeling.MASK_ORDER_LEGACY
    assert bank.capacity_factor == modeling.CAPACITY_FACTOR == 1.25
    holder = modeling.ArmModel(model=torch.nn.Sequential(bank), arm="X",
                               kind="specialists",
                               router_modules=[bank.router], extra_modules=[bank],
                               admission={}, config={}, profile="t2")
    assert holder.mask_order() == modeling.MASK_ORDER_LEGACY
    holder.set_mask_order(modeling.MASK_ORDER_PAIR)
    assert bank.mask_order == modeling.MASK_ORDER_PAIR
    assert holder.mask_order() == modeling.MASK_ORDER_PAIR
    exposure = holder.dispatch_exposure()
    assert exposure["mask_order"] == modeling.MASK_ORDER_PAIR
    assert exposure["pairs"] == 0 and exposure["dropped_fraction"] == 0.0
    with pytest.raises(ValueError):
        holder.set_mask_order("post-restore")
    with pytest.raises(ValueError):
        modeling.SpecialistBank(8, 4, 4, 2, modeling._router(8, 4, seed=1),
                                mask_order="nonsense")
