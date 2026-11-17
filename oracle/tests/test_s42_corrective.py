"""CPU tests for the P19.2b corrective graft.

Three properties the registration depends on:

* **gate-zero parity is bit-exact.** With ``w = 0`` and ``b = 0`` the residual is
  ``tanh(0) * read = 0``, so the injected model is the base model element for
  element — including when the rows and projections already carry the
  representation fit. This is the "gate-zero control" arm.
* **the gate can open and the gradient then reaches the memory rows.** A
  multiplicative gate at exactly zero blocks the pathway's gradient, so the
  ordering is the mechanism: ``dL/dalpha != 0`` at zero, and once alpha is
  non-zero ``tables.grad`` is non-zero. The test states that ordering instead
  of hiding it.
* **interference is suppressed.** The preservation weight is high where the base
  already matches the teacher, the preservation term penalises a KL increase
  there, and it makes the objective's derivative with respect to the gate larger
  when the correction is harmful on those positions.

Plus the accounting: one correction-gate scalar per site, no constant offset in
the residual, and the objective's advantage weighting.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import arms, corrective  # noqa: E402
from frontier import memory as mem  # noqa: E402


# ---------------------------------------------------------------------------
# tiny model, same shape as the stage-2 tests


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden: torch.Tensor):
        return (hidden + torch.tanh(self.linear(hidden)),)


class _Text(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])

    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                use_cache: bool = False):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class _LM(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.model = _Text(vocab, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                use_cache: bool = False):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.lm_head(
            self.model(input_ids, past_key_values, use_cache)))


def _module(hidden: int = 32, value_dim: int = 8, rows: int = 64,
            sites=(0, 2), seed: int = 0) -> corrective.CorrectiveMemory:
    module = corrective.CorrectiveMemory(
        hidden=hidden, value_dim=value_dim, rows=rows, sites=sites,
        conv_kernel=3, table_dtype=torch.float32, seed=seed)
    with torch.no_grad():
        for table in module.tables:
            table.normal_(std=0.05)
    return module


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


# ---------------------------------------------------------------------------
# gate-zero parity


def test_untrained_graft_reproduces_the_base_bit_for_bit():
    torch.manual_seed(5)
    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    tokens = torch.randint(0, 64, (1, 12), dtype=torch.int64)
    with torch.no_grad():
        base = lm(tokens, use_cache=False).logits

    module = _module()
    assert module.identity_at_init()
    handles = arms.install_memory(lm, module)
    try:
        with torch.no_grad():
            injected = lm(tokens, use_cache=False).logits
    finally:
        arms.uninstall(handles)
    assert injected.view(torch.int32).equal(base.view(torch.int32))


def test_gate_zero_is_exact_even_with_grafted_rows_and_a_nonzero_offset():
    torch.manual_seed(6)
    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    tokens = torch.randint(0, 64, (1, 10), dtype=torch.int64)
    with torch.no_grad():
        base = lm(tokens, use_cache=False).logits

    module = _module()
    with torch.no_grad():
        module.set_grafted_rows(0, torch.randn(module.rows, module.value_dim),
                                torch.ones(module.rows, dtype=torch.int64))
        module.set_grafted_rows(1, torch.randn(module.rows, module.value_dim),
                                torch.ones(module.rows, dtype=torch.int64))
        for bias in module.out_bias:
            bias.normal_()          # excluded structurally by forward()
    handles = arms.install_memory(lm, module)
    try:
        with torch.no_grad():
            injected = lm(tokens, use_cache=False).logits
    finally:
        arms.uninstall(handles)
    assert injected.view(torch.int32).equal(base.view(torch.int32))


def test_constant_offset_is_structurally_excluded():
    torch.manual_seed(7)
    module = _module(sites=(0,))
    hidden = torch.randn(1, 5, 32)
    module.context.observe(torch.randint(0, 50, (1, 5), dtype=torch.int64))
    with torch.no_grad():
        module.gate_b[0].fill_(0.5)
        first = module(hidden).clone()
        module.out_bias[0].add_(10.0)
        second = module(hidden)
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# the gate opens, and then the rows learn


def test_gate_gradient_is_nonzero_at_zero_and_rows_follow():
    torch.manual_seed(8)
    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    module = _module()
    handles = arms.install_memory(lm, module)
    tokens = torch.randint(0, 64, (1, 10), dtype=torch.int64)
    try:
        logits = lm(tokens, use_cache=False).logits
        logits.pow(2).sum().backward()
        # at alpha = 0 the gate itself has a non-zero gradient ...
        assert module.gate_w[0].grad is not None
        assert float(module.gate_w[0].grad.abs().sum()) > 0.0
        # ... while the multiplicative pathway carries none yet
        assert all(table.grad is None
                   or float(table.grad.abs().sum()) == 0.0
                   for table in module.tables)

        # open the gate and repeat: gradient now reaches every table
        with torch.no_grad():
            module.gate_w[0].fill_(0.05)
            module.gate_b[0].fill_(0.2)
            module.gate_w[1].fill_(0.05)
            module.gate_b[1].fill_(0.2)
        for table in module.tables:
            table.grad = None
        if module.key_proj[0].weight.grad is not None:
            module.key_proj[0].weight.grad = None
        lm(tokens, use_cache=False).logits.pow(2).sum().backward()
        for table in module.tables:
            assert table.grad is not None
            assert float(table.grad.abs().sum()) > 0.0
        assert float(module.key_proj[0].weight.grad.abs().sum()) > 0.0
        assert float(module.value_proj[0].weight.grad.abs().sum()) > 0.0
    finally:
        arms.uninstall(handles)


def test_gate_can_open_on_the_real_objective():
    torch.manual_seed(9)
    lm = _LM(vocab=32, hidden=32, layers=3)
    _freeze(lm)
    module = _module(sites=(0,))
    handles = arms.install_memory(lm, module)
    tokens = torch.randint(0, 32, (1, 8), dtype=torch.int64)
    try:
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        student = lm(inputs, use_cache=False).logits
        with torch.no_grad():
            module.bypass = True
            base = lm(inputs, use_cache=False).logits
            module.bypass = False
            teacher = base + 0.25 * torch.randn_like(base)
        loss, ce, kl, diagnostics = corrective.corrective_objective(
            student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
            objective="correction", gate=module.last_gate,
            preserve_weight=0.5, interference_weight=0.05)
        loss.backward()
        assert diagnostics["objective"] == "correction"
        assert module.gate_w[0].grad is not None
        assert float(module.gate_w[0].grad.abs().sum()) > 0.0
    finally:
        arms.uninstall(handles)


# ---------------------------------------------------------------------------
# interference suppression


def _synthetic_problem(positions: int = 64, vocab: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(1, positions, vocab, generator=generator)
    small = 0.05 * torch.randn(1, positions, vocab, generator=generator)
    large = 3.0 * torch.randn(1, positions, vocab, generator=generator)
    # first half: the base already matches the teacher; second half: it does not
    base = teacher + torch.cat([small[:, :positions // 2],
                                large[:, positions // 2:]], dim=1)
    return teacher, base


def test_preservation_weight_is_high_where_the_base_is_close():
    teacher, base = _synthetic_problem()
    advantage = corrective.teacher_advantage(base, teacher, temperature=2.0)
    half = advantage.numel() // 2
    assert float(advantage[:half].mean()) < float(advantage[half:].mean())
    mean = advantage.mean()
    preservation = torch.sigmoid(-(advantage - mean) / mean)
    assert float(preservation[:half].mean()) > 0.5
    assert float(preservation[:half].mean()) > float(preservation[half:].mean())


def test_preservation_penalty_suppresses_harmful_correction():
    teacher, base = _synthetic_problem()
    advantage = corrective.teacher_advantage(base, teacher, temperature=2.0)
    half = advantage.numel() // 2
    # a correction that moves the base *away* from the teacher where the base is
    # already right, and towards it where it is wrong
    direction = torch.empty_like(base)
    direction[:, :half] = -(teacher - base)[:, :half]
    direction[:, half:] = (teacher - base)[:, half:]
    targets = torch.zeros((1, base.shape[1]), dtype=torch.int64)
    gates = []
    for preserve_weight in (0.0, 0.5):
        # start away from zero so the preservation penalty's relu is active
        alpha = torch.tensor(0.5, requires_grad=True)
        student = base + alpha * direction
        loss, _, _, diagnostics = corrective.corrective_objective(
            student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
            objective="correction", gate=alpha * torch.ones(1),
            preserve_weight=preserve_weight, interference_weight=0.05)
        loss.backward()
        gates.append((float(alpha.grad), diagnostics))
    without_preservation, with_preservation = gates
    # the preservation term makes opening the gate strictly more costly where
    # the base is already close to the teacher
    assert with_preservation[0] > without_preservation[0]
    assert with_preservation[1]["base_preservation_penalty"] > 0.0
    assert 0.0 < with_preservation[1]["preserved_fraction"] < 1.0


# ---------------------------------------------------------------------------
# S47: corrected sign diagnostics, true signed weighting, aligned interference,
# and base-relative token protection


def test_sign_masks_split_positive_negative_and_zero_explicitly():
    signed = torch.tensor([2.0, 0.0, -2.0, 0.5, -0.5])
    masks = corrective.sign_masks(signed)
    assert masks["teacher_better"].tolist() == [True, False, False, True, False]
    assert masks["base_better"].tolist() == [False, False, True, False, True]
    assert masks["tie"].tolist() == [False, True, False, False, False]
    # the masks partition every token exactly once
    total = masks["teacher_better"] | masks["base_better"] | masks["tie"]
    assert total.all()
    assert int(total.sum()) == signed.numel()


def test_harm_report_uses_the_corrected_sign_labels():
    signed = torch.tensor([2.0, 0.0, -2.0, 3.0, -1.0])
    advantage = torch.zeros(5)
    kl_positions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    report = corrective._harm_report(kl_positions, advantage, signed)
    assert report["harm_sign_counts"]["teacher_better"] == 2
    assert report["harm_sign_counts"]["base_better"] == 2
    assert report["harm_sign_counts"]["tie"] == 1
    # teacher_better indices 0 and 3 -> delta 1 and 4 -> mean 2.5
    assert report["harm_teacher_better_mean_delta_kl"] == pytest.approx(2.5)
    # base_better indices 2 and 4 -> delta 3 and 5 -> mean 4.0
    assert report["harm_base_better_mean_delta_kl"] == pytest.approx(4.0)
    assert report["harm_tie_mean_delta_kl"] == pytest.approx(2.0)
    assert "harm_signed_base_better_tokens" not in report


def test_signed_exp_and_signed_true_have_the_registered_sign_behaviour():
    signed = torch.tensor([2.0, 0.0, -2.0])
    advantage = torch.zeros(3)
    exp_weight, exp_diag = corrective._weigh(advantage, signed, "signed-exp")
    true_weight, true_diag = corrective._weigh(advantage, signed, "signed-true")
    # both are monotone in A; neither is negative
    assert exp_weight[0] > exp_weight[1] > exp_weight[2] > 0
    assert true_weight[0] > true_weight[1] > true_weight[2] > 0
    # the exponential weight is unbounded and never zero: it only suppresses
    assert float(exp_weight[2]) > 0.0
    # the sigmoid weight is bounded, exactly 0.5 at a tie, and symmetric:
    # w(+x) + w(-x) = 1, so the negative region receives the complement of the
    # positive region's pressure (the base-protection term supplies the
    # reversal; the weight supplies the sign split).
    assert float(true_weight[1]) == pytest.approx(0.5)
    assert float(true_weight[0] + true_weight[2]) == pytest.approx(1.0)
    assert true_diag["signed_tie_fraction"] == pytest.approx(1 / 3)
    assert true_diag["signed_positive_fraction"] == pytest.approx(1 / 3)
    assert true_diag["signed_negative_fraction"] == pytest.approx(1 / 3)
    # `signed` remains an accepted alias of `signed-exp`
    alias_weight, alias_diag = corrective._weigh(advantage, signed, "signed")
    assert torch.equal(alias_weight, exp_weight)
    assert alias_diag["weighting"] == "signed"
    assert alias_diag["weighting_canonical"] == "signed-exp"


def test_token_aligned_interference_distinguishes_alignment_where_the_old_one_cannot():
    # Two configurations with identical means: p = [1, 0] and alpha^2 = [0, 1]
    # (anti-aligned) versus p = [1, 0] and alpha^2 = [1, 0] (aligned).
    preservation = torch.tensor([1.0, 0.0])
    anti_aligned = torch.tensor([0.0, 1.0]).sqrt()
    aligned = torch.tensor([1.0, 0.0]).sqrt()
    assert preservation.mean() == preservation.mean()
    assert anti_aligned.square().mean() == aligned.square().mean()
    old_anti = corrective._interference_factorised(preservation, anti_aligned)
    old_aligned = corrective._interference_factorised(preservation, aligned)
    new_anti = corrective._interference(preservation, anti_aligned)
    new_aligned = corrective._interference(preservation, aligned)
    assert float(old_anti) == pytest.approx(float(old_aligned))
    assert float(new_anti) < float(new_aligned)
    assert float(new_anti) == pytest.approx(0.0)
    assert float(new_aligned) == pytest.approx(0.5)


def test_base_protection_is_active_only_where_the_base_is_better():
    signed = torch.tensor([2.0, 0.0, -2.0, -1.0])
    ce = torch.tensor([2.0, 2.0, 2.0, 1.0])
    base_ce = torch.tensor([1.0, 1.0, 1.0, 1.0])
    protected = corrective._base_protection(signed, ce, base_ce)
    # A > 0 and A == 0 contribute nothing; A < 0 contributes only the increase
    assert protected.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    # improving on the base in the A < 0 region is not penalised
    better = corrective._base_protection(signed, 0.5 * base_ce, base_ce)
    assert float(better.abs().sum()) == 0.0


def test_signed_true_objective_carries_the_base_protection_term():
    torch.manual_seed(123)
    vocab, positions = 8, 4
    targets = torch.tensor([[1, 2, 3, 4]])
    teacher = torch.zeros(1, positions, vocab)
    base = torch.zeros(1, positions, vocab)
    # position 0 and 1: teacher better; position 2 and 3: base better
    teacher[0, 0, 1] = 3.0
    teacher[0, 1, 2] = 3.0
    base[0, 2, 3] = 3.0
    base[0, 3, 4] = 3.0
    student = base + 0.1 * torch.randn_like(base)
    loss_off, _, _, diag_off = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="correction", weighting="signed-true",
        base_protection_weight=0.0)
    loss_on, _, _, diag_on = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="correction", weighting="signed-true")
    assert diag_off["base_token_protection_weight"] == 0.0
    assert diag_on["base_token_protection_weight"] == 0.5
    assert diag_on["base_token_protection_tokens"] == 2
    assert float(loss_on) >= float(loss_off)
    # a different weighting does not inherit the term by default
    _, _, _, diag_exp = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="correction", weighting="signed-exp")
    assert diag_exp["base_token_protection_weight"] == 0.0


def test_ce_kl_objective_is_the_control():
    teacher, base = _synthetic_problem()
    student = base + 0.1 * torch.randn_like(base)
    targets = torch.zeros((1, base.shape[1]), dtype=torch.int64)
    loss, ce, kl, diagnostics = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="ce_kl")
    assert diagnostics["objective"] == "ce_kl"
    assert diagnostics["interference_penalty"] == 0.0
    with pytest.raises(ValueError):
        corrective.corrective_objective(student, base, teacher, targets,
                                        temperature=2.0, kl_weight=0.5,
                                        objective="magic")


# ---------------------------------------------------------------------------
# accounting


def test_active_union_is_the_memory_union_plus_one_gate_scalar_per_site():
    geometry = mem.MemoryGeometry()
    module = corrective.CorrectiveMemory(
        hidden=geometry.hidden, value_dim=geometry.value_dim, rows=64,
        sites=(2,), conv_kernel=geometry.conv_kernel)
    report = module.active_parameter_report()
    assert report["active_per_token_before_gate"] == 537_600
    assert report["correction_gate_elements"] == 2 * geometry.hidden + 1
    assert report["active_per_token"] == 537_600 + 2 * geometry.hidden + 1
    assert report["constant_offset_in_residual"] is False
    assert module.identity_at_init()


def test_state_dict_round_trips_gate_and_rows():
    module = _module(sites=(0,))
    with torch.no_grad():
        module.gate_w[0].fill_(0.01)
        module.gate_b[0].fill_(0.3)
    state = module.state_dict()
    assert "gate_w.0" in state and "gate_b.0" in state
    rebuilt = _module(sites=(0,))
    rebuilt.load_state_dict(state, strict=True)
    assert rebuilt.gate_summary() == module.gate_summary()
    assert torch.equal(rebuilt.tables[0], module.tables[0])


# ---------------------------------------------------------------------------
# training-loop wiring


def test_train_builds_the_corrective_arm_from_its_init_checkpoint(tmp_path):
    from types import SimpleNamespace
    from frontier import train

    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    module = _module(sites=(0,))
    checkpoint = {
        "state_dict": module.state_dict(),
        "config": {
            "hidden": 32, "value_dim": 8, "rows": 64, "sites": [0],
            "conv_kernel": 3, "orders": list(mem.ORDERS),
            "table_dtype": "float32", "seed": 0, "variant": "corrective",
        },
        "graft": {"site": 0, "variant": "corrective"},
    }
    path = tmp_path / "memory-init.pt"
    torch.save(checkpoint, path)
    args = SimpleNamespace(memory_init=str(path), site=0, hidden=32)
    arm, handles, report = train.build_arm("corrective", lm, args, "cpu")
    try:
        assert report["kind"] == "stage2_corrective_token_gate_active"
        assert report["identity_at_init"] is True
        assert report["active_per_token"] == report["active_per_token_before_gate"] + 2 * 32 + 1
        assert any(name.startswith("gate_w") for name, _ in arm.named_parameters())
        assert len(handles) == 2  # token context + one site
    finally:
        arms.uninstall(handles)
