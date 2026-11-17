"""CPU tests for the S43 token-conditioned correction gate and advantage weights.

Properties the registration depends on:

* **zero-init is bit-identical.** With `w = 0` and `b = 0`, `alpha_t = tanh(0) =
  0` at every token, so an untrained corrective graft reproduces the base model
  element for element even with grafted rows and a non-zero constant offset.
* **the gate is per token.** Once `w` is non-zero, two positions with different
  hidden states receive different `alpha_t`; at zero it is constant (zero).
* **the signed advantage has the right sign.** `A_t = log p_T(y_t) -
  log p_B(y_t)` is positive when the teacher puts more probability on the true
  token and negative when the base does.
* **gradient reaches the rows once the gate opens**, and a non-zero gate path is
  deterministic for a fixed seed and input.
* **harm is measured where the base already matches the teacher**, not only on
  the mean.
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
            sites=(0,), seed: int = 0) -> corrective.CorrectiveMemory:
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
# zero-init parity and token conditioning


def test_zero_init_bit_identity_with_grafted_rows_and_nonzero_offset():
    torch.manual_seed(11)
    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    tokens = torch.randint(0, 64, (1, 12), dtype=torch.int64)
    with torch.no_grad():
        base = lm(tokens, use_cache=False).logits

    module = _module()
    assert module.identity_at_init()
    with torch.no_grad():
        module.set_grafted_rows(0, torch.randn(module.rows, module.value_dim),
                                torch.ones(module.rows, dtype=torch.int64))
        module.out_bias[0].normal_()
    handles = arms.install_memory(lm, module)
    try:
        with torch.no_grad():
            injected = lm(tokens, use_cache=False).logits
            gate = module.last_gate
    finally:
        arms.uninstall(handles)
    assert injected.view(torch.int32).equal(base.view(torch.int32))
    assert gate is not None and float(gate.abs().max()) == 0.0
    assert gate.numel() == tokens.shape[1]


def test_gate_varies_per_token_once_w_is_nonzero():
    torch.manual_seed(12)
    module = _module()
    tokens = torch.randint(0, 64, (1, 10), dtype=torch.int64)
    module.context.observe(tokens)
    hidden = torch.randn(1, 10, 32)
    with torch.no_grad():
        module(hidden)
        at_init = module.last_gate.clone()
        assert float(at_init.std()) == 0.0
        module.gate_w[0].normal_(std=0.1)
        module(hidden)
        opened = module.last_gate.clone()
    assert float(opened.std()) > 0.0
    assert not torch.equal(at_init, opened)
    # identical (h, r) at three positions gets the same gate value
    same_h = hidden[:, :1].repeat(1, 3, 1)
    same_r = torch.randn(1, 1, 32).repeat(1, 3, 1)
    with torch.no_grad():
        values = module.gate(0, same_h, same_r)[0]
    assert torch.equal(values, values[0].expand_as(values))


def test_gate_reports_the_parameter_cost_of_phi():
    module = corrective.CorrectiveMemory(hidden=1024, value_dim=256, rows=64,
                                         sites=(2,), conv_kernel=4)
    report = module.active_parameter_report()
    assert report["gate_feature_map"].startswith("phi(h_t, r_t)")
    assert report["gate_parameters_per_site"] == 2 * 1024 + 1
    assert report["correction_gate_elements"] == 2 * 1024 + 1
    assert report["active_per_token"] == 537_600 + 2 * 1024 + 1
    assert report["gate_init"].startswith("w = 0 and b = 0")


# ---------------------------------------------------------------------------
# signed advantage


def _logits_with_target_prob(prob: float, vocab: int = 8) -> torch.Tensor:
    """Logits whose softmax probability of token 0 is exactly `prob`."""
    other = (1.0 - prob) / (vocab - 1)
    values = torch.full((1, 1, vocab), float(torch.log(torch.tensor(other))))
    values[..., 0] = float(torch.log(torch.tensor(prob)))
    return values


def test_signed_advantage_sign_is_correct():
    targets = torch.zeros((1, 1), dtype=torch.int64)
    base = _logits_with_target_prob(0.1)
    teacher = _logits_with_target_prob(0.9)
    positive = corrective.signed_teacher_advantage(base, teacher, targets)
    negative = corrective.signed_teacher_advantage(teacher, base, targets)
    assert float(positive) > 0.0
    assert float(negative) < 0.0
    assert float(positive + negative) == pytest.approx(0.0, abs=1e-5)
    expected = float(torch.log(torch.tensor(0.9 / 0.1)))
    assert float(positive) == pytest.approx(expected, rel=1e-4)


def test_signed_weight_orders_positions_by_teacher_advantage():
    advantage = torch.ones(4)
    signed = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    weight, diagnostics = corrective._weigh(advantage, signed, "signed")
    assert torch.all(weight.diff() > 0)
    assert diagnostics["mean_signed_advantage"] == pytest.approx(0.0)
    assert diagnostics["signed_positive_fraction"] == pytest.approx(0.5)
    both, _ = corrective._weigh(advantage, signed, "both")
    assert torch.all(both.diff() > 0)


# ---------------------------------------------------------------------------
# gradient flow and determinism


def _two_identical_modules():
    first = _module(seed=3)
    second = _module(seed=3)
    with torch.no_grad():
        first.gate_w[0].normal_(std=0.05, generator=torch.Generator().manual_seed(7))
        second.gate_w[0].copy_(first.gate_w[0])
        first.gate_b[0].fill_(0.1)
        second.gate_b[0].copy_(first.gate_b[0])
        for a, b in zip(first.tables, second.tables):
            b.copy_(a)
    return first, second


def test_rows_receive_gradient_once_the_gate_is_open_and_the_path_is_deterministic():
    torch.manual_seed(13)
    lm = _LM(vocab=32, hidden=32, layers=3)
    _freeze(lm)
    tokens = torch.randint(0, 32, (1, 8), dtype=torch.int64)
    first, second = _two_identical_modules()
    losses = []
    gate_means = []
    for module in (first, second):
        handles = arms.install_memory(lm, module)
        try:
            logits = lm(tokens, use_cache=False).logits.float()
            loss = logits.pow(2).mean()
            loss.backward()
            losses.append(float(loss.detach()))
            gate_means.append(float(module.last_gate.detach().mean()))
            assert float(module.gate_w[0].grad.abs().sum()) > 0.0
            assert all(table.grad is not None and
                       float(table.grad.abs().sum()) > 0.0
                       for table in module.tables)
            assert float(module.key_proj[0].weight.grad.abs().sum()) > 0.0
        finally:
            arms.uninstall(handles)
    assert losses[0] == losses[1]
    assert gate_means[0] == gate_means[1]
    for a, b in zip(first.tables, second.tables):
        assert torch.equal(a.grad, b.grad)


# ---------------------------------------------------------------------------
# harm where the base already matches


def test_harm_is_reported_where_the_base_already_matches():
    torch.manual_seed(14)
    positions, vocab = 64, 8
    teacher = torch.randn(1, positions, vocab)
    half = positions // 2
    base = teacher.clone()
    base[:, :half] += 0.02 * torch.randn(1, half, vocab)   # base already matches
    base[:, half:] += 3.0 * torch.randn(1, positions - half, vocab)
    # the student moves away from the teacher on the first half only, and is
    # identical to base on the second half
    student = base.clone()
    student[:, :half] = teacher[:, :half] - 2.0 * (teacher[:, :half] - base[:, :half])
    targets = torch.zeros((1, positions), dtype=torch.int64)
    gate = torch.full((1, positions), 0.1)
    loss, _, _, diagnostics = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="correction", weighting="both", gate=gate,
        preserve_weight=0.5, interference_weight=0.05)
    assert diagnostics["harm_lowest_quartile_tokens"] > 0
    assert diagnostics["harm_lowest_quartile_mean_delta_kl"] > 0.0
    assert diagnostics["harm_lowest_quartile_fraction_worse"] > 0.0
    # S47: the old `signed >= 0` bucket is split and the labels are corrected.
    assert diagnostics["harm_teacher_better_tokens"] > 0
    assert diagnostics["harm_base_better_tokens"] > 0
    assert "harm_signed_base_better_tokens" not in diagnostics
    assert diagnostics["weighting"] == "both"


def test_weighting_modes_are_registered_and_unknown_modes_fail():
    assert set(corrective.WEIGHTINGS) == {
        "kl", "signed", "signed-exp", "signed-true", "both"}
    teacher = torch.randn(1, 4, 8)
    base = teacher + 0.5 * torch.randn_like(teacher)
    student = base + 0.1 * torch.randn_like(base)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    for weighting in corrective.WEIGHTINGS:
        loss, _, _, diagnostics = corrective.corrective_objective(
            student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
            objective="correction", weighting=weighting)
        assert torch.isfinite(loss)
        assert diagnostics["weighting"] == weighting
        assert diagnostics["weighting_canonical"] == corrective.canonical_weighting(
            weighting)
    with pytest.raises(ValueError):
        corrective.corrective_objective(
            student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
            objective="correction", weighting="magic")


def test_ce_kl_control_is_unchanged():
    teacher = torch.randn(1, 4, 8)
    base = teacher + 0.5 * torch.randn_like(teacher)
    student = base + 0.1 * torch.randn_like(base)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    loss, ce, kl, diagnostics = corrective.corrective_objective(
        student, base, teacher, targets, temperature=2.0, kl_weight=0.5,
        objective="ce_kl")
    assert diagnostics["objective"] == "ce_kl"
    assert torch.isfinite(loss)
