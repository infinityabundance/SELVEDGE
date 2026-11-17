"""S47 CPU tests for the matched controls.

The stage-2b registration now requires two comparison rows beside the corrective
arms:

* a dense adapter trained on the same stream, same teacher and same optimizer
  budget with **exactly** the corrective arm's 539,649-element active union;
* an identity-gated dense residual with the **same** token gate and no external
  memory, so a gain cannot be attributed to the gate alone.

These tests check the accounting and the identity property; no GPU work runs.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import arms, corrective  # noqa: E402
from frontier import matched_controls as matched  # noqa: E402
from frontier import memory as mem  # noqa: E402


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

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class _LM(nn.Module):
    def __init__(self, vocab: int, hidden: int, layers: int):
        super().__init__()
        self.model = _Text(vocab, hidden, layers)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.lm_head(
            self.model(input_ids, past_key_values, use_cache)))


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def test_matched_dense_adapter_is_exactly_the_corrective_union():
    module = matched.MatchedDenseAdapter(hidden=1024)
    assert matched.CORRECTIVE_ACTIVE == 539_649
    assert module.active_elements == 539_649
    report = module.active_parameter_report()
    assert report["kind"] == "stage2_matched_dense_adapter_active"
    assert report["active_per_token"] == 539_649
    assert report["gate_elements"] == 0
    assert report["match_delta"] == 0


def test_identity_gated_dense_residual_is_exactly_the_corrective_union():
    module = matched.IdentityGatedDenseResidual(hidden=1024)
    assert module.dense.active_elements == 537_600
    assert module.gate_parameters == 2_049
    assert module.active_elements == 539_649
    report = module.active_parameter_report()
    assert report["kind"] == "stage2_identity_gated_dense_active"
    assert report["dense_elements"] == 537_600
    assert report["gate_elements"] == 2_049
    assert report["active_per_token"] == 539_649
    assert report["match_delta"] == 0
    assert module.identity_at_init()


def test_identity_gated_dense_residual_is_the_base_bit_for_bit_at_init():
    torch.manual_seed(47)
    lm = _LM(vocab=32, hidden=32, layers=3)
    _freeze(lm)
    tokens = torch.randint(0, 32, (1, 10), dtype=torch.int64)
    with torch.no_grad():
        base = lm(tokens, use_cache=False).logits
    module = matched.IdentityGatedDenseResidual(hidden=32)
    handles = arms.install_adapter(lm, module, (0,))
    try:
        with torch.no_grad():
            injected = lm(tokens, use_cache=False).logits
            gate = module.last_gate
    finally:
        arms.uninstall(handles)
    assert injected.view(torch.int32).equal(base.view(torch.int32))
    assert float(gate.abs().max()) == 0.0


def test_identity_gated_dense_residual_uses_the_corrective_gate():
    torch.manual_seed(48)
    hidden, value_dim = 32, 8
    control = matched.IdentityGatedDenseResidual(hidden=hidden)
    corrective_module = corrective.CorrectiveMemory(
        hidden=hidden, value_dim=value_dim, rows=64, sites=(0,),
        conv_kernel=3, table_dtype=torch.float32, seed=0)
    with torch.no_grad():
        # the same w and b means the same gate on the same (h, r) pair
        control.gate_w.copy_(torch.randn_like(control.gate_w))
        control.gate_b.copy_(torch.randn_like(control.gate_b))
        corrective_module.gate_w[0].copy_(control.gate_w)
        corrective_module.gate_b[0].copy_(control.gate_b)
    h = torch.randn(1, 5, hidden)
    r = torch.randn(1, 5, hidden)
    gate_control = control.gate(h, r)
    gate_corrective = corrective_module.gate(0, h, r)
    assert torch.equal(gate_control, gate_corrective)


def test_build_control_reports_the_registered_union_for_both_kinds():
    from types import SimpleNamespace

    for kind, expected_kind in (
        ("adapter-matched", "stage2_matched_dense_adapter_active"),
        ("gated-dense", "stage2_identity_gated_dense_active"),
    ):
        # no model: this checks construction and accounting, not installation
        args = SimpleNamespace(hidden=1024, site=0, model=None)
        module, handles, report = matched.build_control(kind, args, "cpu")
        assert handles == []
        assert report["kind"] == expected_kind
        assert report["match_delta"] == 0
        assert report["active_per_token"] == 539_649


def test_build_control_installs_on_a_tiny_model():
    from types import SimpleNamespace

    lm = _LM(vocab=32, hidden=32, layers=3)
    _freeze(lm)
    args = SimpleNamespace(hidden=32, site=0, model=lm)
    module, handles, report = matched.build_control("gated-dense", args, "cpu")
    try:
        assert report["kind"] == "stage2_identity_gated_dense_active"
        assert report["identity_at_init"] is True
        assert len(handles) == 1
    finally:
        arms.uninstall(handles)


def test_matched_dense_adapter_forward_is_finite_and_zero_initialised():
    module = matched.MatchedDenseAdapter(hidden=32,
                                         target_active=28769)
    assert module.active_elements == 28769
    output = module(torch.randn(2, 4, 32))
    assert output.shape == (2, 4, 32)
    assert torch.isfinite(output).all()
    # the zero initialised up-projection makes the residual exactly the bias
    assert float(output.detach().abs().max()) == 0.0
