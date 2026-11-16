"""P13.14 — omission-conditioned recovery, on synthetic ground truth.

The tests check the *mechanism*, not a stored number:

* a zero-initialised compensator reproduces the sparse layer exactly, bit for
  bit, which is the zero point the registered comparison rests on;
* the accounting includes the compensator, the alignment rank is floored so the
  baseline never has more elements than the compensator, and the realized active
  counts stay inside the ceiling;
* the leakage boundary holds: the compensator's declared inputs are
  `(hidden_states, selected)` and after freezing, the only trainable parameters
  are its own;
* on a synthetic case where the omitted groups' contribution is *known*, the
  conditioned compensator learns it while (a) the set-independent alignment
  baseline cannot and (b) a compensator conditioned on a shuffled set of the
  same size cannot. That is the decisive-experiment shape in miniature;
* everything is deterministic under a fixed seed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import recovery  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures


class _Bank(torch.nn.Module):
    """A minimal stand-in exposing what `RecoveredSparse` reads from a bank."""

    def __init__(self, hidden: int, experts: int, k: int, maps: torch.Tensor,
                 offsets: torch.Tensor, weights: torch.Tensor | None = None):
        super().__init__()
        self.hidden = hidden
        self.experts = experts
        self.k = k
        self.router = torch.nn.Linear(hidden, experts, bias=False)
        self.maps = maps          # [experts, hidden, hidden] expert contribution
        self.offsets = offsets    # [experts, hidden]
        self.fixed_weights = weights  # [experts] mixture weights, or softmax

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        selected = self.router(flat).topk(self.k, dim=-1).indices
        out = torch.zeros_like(flat)
        if self.fixed_weights is None:
            weights = torch.softmax(self.router(flat), dim=-1).gather(1, selected)
        else:
            weights = self.fixed_weights.reshape(1, -1).expand(flat.shape[0], -1) \
                .gather(1, selected)
        for slot in range(self.k):
            expert = selected[:, slot]
            contribution = torch.einsum("nh,nhd->nd", flat, self.maps[expert]) \
                + self.offsets[expert]
            out = out + contribution * weights[:, slot].unsqueeze(-1)
        return out.reshape(hidden_states.shape)

    def dense(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Unit-weight full bank: the reference the omission is measured against."""
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.zeros_like(flat)
        for expert in range(self.experts):
            out = out + torch.einsum("nh,hd->nd", flat, self.maps[expert]) \
                + self.offsets[expert]
        return out.reshape(hidden_states.shape)


def _bank(hidden: int = 4, experts: int = 4, k: int = 2, seed: int = 5) -> _Bank:
    """A bank whose experts are rank-one linear maps with a small offset.

    Rank one keeps the omitted groups' contribution exactly representable (two
    omitted experts give a rank-two map), so the synthetic case has a known
    answer rather than an arbitrary one.
    """
    generator = torch.Generator().manual_seed(seed)
    maps = torch.zeros(experts, hidden, hidden)
    for expert in range(experts):
        out = torch.randn(hidden, generator=generator)
        read = torch.randn(hidden, generator=generator)
        maps[expert] = torch.outer(out, read) / (out.norm() * read.norm())
    offsets = torch.randn(experts, hidden, generator=generator) * 0.2
    bank = _Bank(hidden, experts, k, maps, offsets)
    with torch.no_grad():
        bank.router.weight.copy_(torch.randn(experts, hidden, generator=generator))
    return bank


def _features(count: int, hidden: int = 4, seed: int = 11) -> torch.Tensor:
    return torch.randn(count, hidden, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# the zero point


def test_zero_initialised_compensator_reproduces_the_sparse_layer_exactly():
    bank = _bank()
    config = recovery.RecoveryConfig(rank=4, set_dim=4)
    compensator = recovery.OmissionCompensator(4, 4, config, seed=1)
    wrapper = recovery.RecoveredSparse(bank, bank.router, compensator, k=2)
    features = _features(32)
    with torch.no_grad():
        assert torch.equal(wrapper(features), bank(features))
        assert float(wrapper.compensator(features, wrapper.selection(features))
                     .abs().max()) == 0.0


def test_wrapper_without_a_compensator_is_the_bare_sparse_layer():
    bank = _bank()
    wrapper = recovery.RecoveredSparse(bank, bank.router, None, k=2)
    features = _features(16)
    with torch.no_grad():
        assert torch.equal(wrapper(features), bank(features))


# ---------------------------------------------------------------------------
# accounting


def test_compensator_element_count_follows_the_declared_shape():
    config = recovery.RecoveryConfig(rank=8, set_dim=8)
    compensator = recovery.OmissionCompensator(1024, 8, config, seed=1)
    expected = (8 * 8                      # set embedding
                + (1024 + 8) * 8 + 8       # down weight and bias
                + 8 * 1024 + 1024)         # up weight and bias
    assert compensator.active_elements() == expected
    accounting = compensator.accounting()
    assert accounting["active_elements"] == expected
    assert accounting["set_embedding_elements"] == 64


def test_alignment_rank_is_floored_so_the_baseline_never_exceeds_the_compensator():
    for compensator_elements in (4096, 17_544, 32_896, 100_000):
        rank = recovery.matched_alignment_rank(compensator_elements, 1024)
        assert 2 * 1024 * rank + 1024 <= compensator_elements
    # An unaffordable budget fails closed instead of exceeding the ceiling.
    with pytest.raises(ValueError):
        recovery.matched_alignment_rank(100, 1024)


def test_arm_ledger_adds_the_correction_to_the_base_count():
    arm = recovery.RecoveryArm(name="sparse+compensator")
    arm.compensators.append(recovery.OmissionCompensator(
        1024, 8, recovery.RecoveryConfig(rank=8, set_dim=8), seed=1))
    arm.alignments.append(recovery.LayerAlignment(1024, 4))
    row = arm.ledger(base_active=774_543_360, bank_k=4, bank_experts=8)
    assert row["compensator_elements"] == arm.compensators[0].active_elements()
    assert row["alignment_elements"] == 2 * 1024 * 4 + 1024
    assert row["realized_active"] == 774_543_360 + row["recovery_elements"]
    assert row["omission_rate"] == pytest.approx(0.5)


def test_describe_arms_is_inside_the_ceiling_and_matches_the_hand_sum():
    described = recovery.describe_arms(base_active=774_543_360, bank_k=4,
                                       bank_experts=8, hidden=1024)
    arms = described["arms"]
    assert arms["sparse-only"]["realized_active"] == 774_543_360
    assert arms["sparse+compensator"]["recovery_elements"] == \
        described["compensator_total"] == described["compensator_per_layer"] * 28
    assert arms["sparse+alignment"]["recovery_elements"] == \
        (2 * 1024 * described["alignment_rank"] + 1024) * 28
    assert arms["sparse+alignment+compensator"]["recovery_elements"] == \
        arms["sparse+alignment"]["recovery_elements"] \
        + arms["sparse+compensator"]["recovery_elements"]
    for row in arms.values():
        assert row["within_ceiling"], row
    assert arms["sparse-only"]["recovery_elements"] < \
        arms["sparse+compensator"]["recovery_elements"]
    # The two corrections together are still far below the ceiling's headroom.
    headroom = 1_000_000_000 - 774_543_360
    assert arms["sparse+alignment+compensator"]["recovery_elements"] < headroom


# ---------------------------------------------------------------------------
# set conditioning


def test_omitted_code_is_the_complement_of_the_selection():
    embedding = recovery.set_embedding(4, 2, seed=3)
    selected = torch.tensor([[0, 2], [1, 3]])
    chosen = recovery.encode_sets(embedding, selected, 4, "selected")
    omitted = recovery.encode_sets(embedding, selected, 4, "omitted")
    with torch.no_grad():
        assert torch.allclose(chosen, embedding.weight[selected].sum(dim=1))
        assert torch.allclose(omitted, embedding.weight[torch.tensor([[1, 3], [0, 2]])]
                              .sum(dim=1))
        assert torch.allclose(chosen + omitted,
                              embedding.weight.sum(dim=0).expand(2, -1))


def test_shuffled_sets_keep_the_size_and_are_deterministic():
    selected = torch.tensor([[0, 1], [2, 3], [1, 2]])
    first = recovery.shuffle_sets(selected, 4, seed=9)
    again = recovery.shuffle_sets(selected, 4, seed=9)
    other = recovery.shuffle_sets(selected, 4, seed=10)
    assert first.shape == selected.shape
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    for row in first:
        assert len(set(row.tolist())) == selected.shape[1]


def test_selected_and_omitted_conditioning_differ_in_their_codes():
    config = recovery.RecoveryConfig(rank=4, set_dim=4, set_mode="selected")
    selected_mode = recovery.OmissionCompensator(4, 4, config, seed=1)
    omitted_config = recovery.RecoveryConfig(rank=4, set_dim=4, set_mode="omitted")
    omitted_mode = recovery.OmissionCompensator(4, 4, omitted_config, seed=1)
    features = _features(8)
    chosen = torch.tensor([[0, 1]] * 8)
    with torch.no_grad():
        first = selected_mode(features, chosen)
        second = omitted_mode(features, chosen)
    assert first.shape == second.shape == features.shape


def test_unknown_set_mode_is_refused():
    with pytest.raises(ValueError):
        recovery.encode_sets(recovery.set_embedding(4, 2), torch.zeros(1, 2,
                                                                     dtype=torch.long),
                             4, "everything")


# ---------------------------------------------------------------------------
# leakage boundary and freezing


def test_compensator_declares_only_causal_inputs():
    assert recovery.OmissionCompensator.causal_input_signature() == \
        ("hidden_states", "selected")


def test_freezing_leaves_only_the_compensator_trainable():
    bank = _bank()
    wrapper = recovery.RecoveredSparse(
        bank, bank.router,
        recovery.OmissionCompensator(4, 4, recovery.RecoveryConfig(rank=4, set_dim=4),
                                     seed=1), k=2)
    model = torch.nn.Module()
    model.config = type("Config", (), {"hidden_size": 4})()
    model.model = wrapper
    model.banks = lambda: [bank]
    arm = recovery.RecoveryArm(name="sparse+compensator")
    arm.wrappers.append(wrapper)
    arm.compensators.append(wrapper.compensator)
    report = recovery.freeze_arm_but_recovery(model, arm, enable="compensator")
    assert report["trainable_elements"] == wrapper.compensator.active_elements()
    assert report["enabled_tensors"] == len(list(wrapper.compensator.parameters()))
    trainable = [name for name, parameter in wrapper.named_parameters()
                 if parameter.requires_grad]
    assert all(name.startswith("compensator.") for name in trainable)


def test_compensator_output_ignores_anything_but_the_state_and_the_set():
    config = recovery.RecoveryConfig(rank=4, set_dim=4)
    compensator = recovery.OmissionCompensator(4, 4, config, seed=1)
    with torch.no_grad():
        compensator.up.weight.normal_(std=0.1, generator=torch.Generator().manual_seed(4))
    features = _features(8)
    selected = torch.tensor([[0, 1]] * 8)
    first = compensator(features, selected)
    # A different "teacher" or "label" cannot reach it: the call has no other
    # input, and the module holds no buffer that a label could be written into.
    assert set(name for name, _ in compensator.named_buffers()) == set()
    assert torch.equal(first, compensator(features, selected))


# ---------------------------------------------------------------------------
# determinism


def test_compensator_is_deterministic_for_a_seed():
    config = recovery.RecoveryConfig(rank=4, set_dim=4)
    first = recovery.OmissionCompensator(4, 4, config, seed=77)
    second = recovery.OmissionCompensator(4, 4, config, seed=77)
    third = recovery.OmissionCompensator(4, 4, config, seed=78)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
    assert not torch.equal(first.down.weight, third.down.weight)


# ---------------------------------------------------------------------------
# installation into the model


def test_install_replaces_every_bank_and_remove_restores_it():
    bank = _bank()
    holder = torch.nn.ModuleList([bank])
    model = torch.nn.Module()
    model.config = type("Config", (), {"hidden_size": 4})()
    model.model = torch.nn.Sequential(holder)
    wrapper = recovery.RecoveredSparse(
        bank, bank.router,
        recovery.OmissionCompensator(4, 4, recovery.RecoveryConfig(rank=4, set_dim=4),
                                     seed=1), k=2)
    arm = recovery.RecoveryArm(name="both")
    arm.wrappers.append(wrapper)
    arm.compensators.append(wrapper.compensator)
    model.banks = lambda: [bank]
    installations = recovery.install_recovery(model, arm)
    assert len(installations) == 1
    assert model.model[0][0] is wrapper
    # Wrapping twice must not nest: the snapshot walk means the second install
    # cannot find the bank inside the wrapper it just placed.
    recovery.remove_recovery(installations)
    assert model.model[0][0] is bank


def test_installed_wrapper_runs_the_correction_on_the_graph():
    bank = _bank()
    model = torch.nn.Module()
    model.config = type("Config", (), {"hidden_size": 4})()
    model.model = torch.nn.Sequential(torch.nn.ModuleList([bank]))
    model.banks = lambda: [bank]
    wrapper = recovery.RecoveredSparse(
        bank, bank.router,
        recovery.OmissionCompensator(4, 4, recovery.RecoveryConfig(rank=4, set_dim=4),
                                     seed=1), k=2)
    arm = recovery.RecoveryArm(name="sparse+compensator")
    arm.wrappers.append(wrapper)
    arm.compensators.append(wrapper.compensator)
    installations = recovery.install_recovery(model, arm)
    recovery.freeze_arm_but_recovery(model, arm, enable="compensator")
    features = _features(8)
    model.model[0][0](features).pow(2).mean().backward()
    connected = all(parameter.grad is not None and
                    (parameter.grad != 0).any()
                    for parameter in wrapper.compensator.parameters()
                    if parameter is wrapper.compensator.up.weight or
                    parameter is wrapper.compensator.up.bias)
    assert connected
    recovery.remove_recovery(installations)


# ---------------------------------------------------------------------------
# the alignment baseline


def test_alignment_l2_recovers_a_rank_two_linear_map():
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(2000, 8, generator=generator)
    basis = torch.randn(8, 2, generator=generator)
    targets = features @ basis @ torch.randn(2, 8, generator=generator) + 0.7
    alignment = recovery.LayerAlignment(8, 6)
    report = alignment.fit(features, targets, ridge=1e-8, mode="l2")
    assert report["truncated_error_rms"] < 0.05 * report["target_rms"]
    with torch.no_grad():
        corrected = alignment(features)
    residual = (corrected - targets).pow(2).mean().sqrt()
    assert float(residual) < 0.05 * float(targets.pow(2).mean().sqrt())


def test_alignment_truncation_costs_what_the_tail_says():
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(1500, 8, generator=generator)
    singular = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05])
    targets = features @ torch.diag(singular)
    full = recovery.LayerAlignment(8, 8)
    truncated = recovery.LayerAlignment(8, 2)
    full_report = full.fit(features, targets, ridge=1e-10, mode="l2")
    truncated_report = truncated.fit(features, targets, ridge=1e-10, mode="l2")
    assert truncated_report["rank"] == 2
    assert truncated_report["truncated_error_rms"] > full_report["truncated_error_rms"]
    # The truncated map loses exactly the tail energy the SVD says it must:
    # the retained energy is the first two singular values, so the residual per
    # element is the root of the remaining energy spread over all eight dims.
    expected_tail = float((singular[2:] ** 2).sum() / singular.numel()) ** 0.5
    assert truncated_report["truncated_error_rms"] == pytest.approx(
        expected_tail, rel=0.1)


def test_alignment_ot_mode_matches_the_target_moments():
    generator = torch.Generator().manual_seed(19)
    features = torch.randn(4000, 6, generator=generator) * 2.0
    targets = torch.randn(4000, 6, generator=generator) @ torch.diag(
        torch.tensor([3.0, 1.0, 1.0, 0.5, 0.5, 0.2]))
    alignment = recovery.LayerAlignment(6, 6)
    alignment.fit(features, targets, ridge=1e-6, mode="ot")
    with torch.no_grad():
        corrected = alignment(features)
    assert torch.allclose(corrected.mean(dim=0), targets.mean(dim=0), atol=0.05)
    corrected_cov = torch.cov(corrected.T)
    target_cov = torch.cov(targets.T)
    relative = (corrected_cov - target_cov).norm() / target_cov.norm()
    assert float(relative) < 0.15


def test_alignment_refuses_mismatched_shapes():
    alignment = recovery.LayerAlignment(4, 2)
    with pytest.raises(ValueError):
        alignment.fit(torch.zeros(4, 4), torch.zeros(3, 4))


# ---------------------------------------------------------------------------
# the decisive synthetic experiment


def _train_compensator(bank: _Bank, mode: str, steps: int = 600, seed: int = 3,
                       set_mode: str = "omitted") -> tuple[torch.nn.Module, dict]:
    """Learn the omitted groups' contribution for one synthetic bank."""
    config = recovery.RecoveryConfig(rank=8, set_dim=8, set_mode=set_mode)
    compensator = recovery.OmissionCompensator(bank.hidden, bank.experts, config,
                                               seed=seed)
    optimizer = torch.optim.Adam(compensator.parameters(), lr=5e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    losses = []
    for _step in range(steps):
        features = torch.randn(256, bank.hidden, generator=generator)
        with torch.no_grad():
            selected = bank.router(features).topk(bank.k, dim=-1).indices
            weights = torch.softmax(bank.router(features), dim=-1).gather(1, selected)
            sparse = bank(features)
            dense = bank.dense(features)
            if mode == "shuffled":
                conditioned = recovery.shuffle_sets(selected, bank.experts,
                                                    seed=seed)
            else:
                conditioned = selected
        if mode == "alignment":
            continue
        prediction = compensator(features, conditioned)
        loss = (prediction - (dense - sparse)).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return compensator, {"final_loss": losses[-1] if losses else None,
                         "first_loss": losses[0] if losses else None}


def _evaluate(correction, bank: _Bank, mode: str, count: int = 512,
              seed: int = 99) -> float:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, bank.hidden, generator=generator)
    with torch.no_grad():
        selected = bank.router(features).topk(bank.k, dim=-1).indices
        target = bank.dense(features) - bank(features)
        if mode == "alignment":
            prediction = correction(features)
        elif mode == "shuffled":
            prediction = correction(features, recovery.shuffle_sets(
                selected, bank.experts, seed=seed + 5))
        else:
            prediction = correction(features, selected)
    return float((prediction - target).pow(2).mean().sqrt())


def test_conditioned_compensator_learns_the_omitted_function_and_the_controls_do_not():
    # The sparse path runs the selected groups at unit weight, so the omitted
    # contribution is *exactly* the sum of the unselected groups' functions:
    # a known target that depends on (h, S) and nothing else. That is the
    # synthetic form of the decisive experiment.
    bank = _bank(hidden=6, experts=4, k=2, seed=5)
    bank.fixed_weights = torch.ones(bank.experts)

    # (a) the set-independent alignment baseline, fitted closed-form on the
    #     same calibration distribution.
    generator = torch.Generator().manual_seed(21)
    calibration = torch.randn(4096, bank.hidden, generator=generator)
    with torch.no_grad():
        alignment_target = bank.dense(calibration) - bank(calibration)
    alignment = recovery.LayerAlignment(bank.hidden, 4)
    alignment.fit(calibration, alignment_target, ridge=1e-6, mode="l2")
    alignment_error = _evaluate(alignment, bank, "alignment")

    # (b) the conditioned compensator.
    compensator, curve = _train_compensator(bank, mode="conditioned", steps=1500)
    conditioned_error = _evaluate(compensator, bank, "conditioned")

    # (c) the shuffled-condition falsifying control: same rate, wrong identity.
    shuffled, _ = _train_compensator(bank, mode="shuffled", steps=1500)
    shuffled_error = _evaluate(shuffled, bank, "shuffled")

    assert curve["final_loss"] < curve["first_loss"]
    # The set-independent map can only learn the average omitted contribution,
    # so the conditioned compensator must be several times more accurate.
    assert conditioned_error < 0.25 * alignment_error, (
        conditioned_error, alignment_error)
    # And the control that knows the rate but not the identity must fail too.
    assert conditioned_error < 0.5 * shuffled_error, (
        conditioned_error, shuffled_error)


def test_selected_and_omitted_conditioning_are_both_learnable():
    bank = _bank(hidden=6, experts=4, k=2, seed=7)
    bank.fixed_weights = torch.ones(bank.experts)
    omitted, _ = _train_compensator(bank, mode="conditioned", steps=1500, seed=4,
                                    set_mode="omitted")
    selected, _ = _train_compensator(bank, mode="conditioned", steps=1500, seed=4,
                                     set_mode="selected")
    # Both encodings carry the same information about the omitted set when
    # k = m/2, so both must learn it well below the target scale.
    generator = torch.Generator().manual_seed(101)
    features = torch.randn(512, bank.hidden, generator=generator)
    with torch.no_grad():
        chosen = bank.router(features).topk(bank.k, dim=-1).indices
        target = bank.dense(features) - bank(features)
        scale = target.pow(2).mean().sqrt()
        first = (omitted(features, chosen) - target).pow(2).mean().sqrt()
        second = (selected(features, chosen) - target).pow(2).mean().sqrt()
    assert float(first) < 0.25 * float(scale)
    assert float(second) < 0.25 * float(scale)


# ---------------------------------------------------------------------------
# the runner's smoke path (tiny model, CPU)


def test_runner_smoke_path_produces_a_receipt(tmp_path):
    import json

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
    import recovery_court

    out = tmp_path / "smoke.json"
    code = recovery_court.main(["--smoke", "--out", str(out), "--rank", "4",
                                "--set-dim", "4"])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "smoke"
    assert payload["installed"] == payload["layers"] == 2
    assert payload["compensator_graph_connected"] is True
    assert payload["compensator_receives_gradient"] is True
    assert payload["freeze"]["trainable_elements"] == \
        payload["ledger"]["compensator_elements"]
    assert payload["ledger"]["realized_active"] == \
        payload["ledger"]["recovery_elements"]
    assert payload["alignment_reports"], "the alignment must be fitted"


def test_runner_plan_needs_no_model_and_reports_the_matched_budget(capsys):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
    import recovery_court

    code = recovery_court.main(["--plan", "--base-active", "772440064"])
    assert code == 0
    import json

    described = json.loads(capsys.readouterr().out)
    compensator = described["arms"]["sparse+compensator"]["recovery_elements"]
    alignment = described["arms"]["sparse+alignment"]["recovery_elements"]
    assert alignment <= compensator
    assert described["arms"]["sparse+alignment+compensator"]["recovery_elements"] \
        == alignment + compensator
    for row in described["arms"].values():
        assert row["within_ceiling"]
