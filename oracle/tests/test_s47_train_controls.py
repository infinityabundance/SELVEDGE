"""CPU tests for the S47 matched-control wiring in `oracle/frontier/train.py`.

The controls are only worth anything if the training loop builds the same
module the scoring loop builds. These tests run the training-side entry point on
a tiny CPU model and check the properties a registration can silently lose:

* both arm names reach `matched_controls.build_control`, not a dense adapter;
* the active union is exactly 539,649 at the registered geometry;
* the untrained control reproduces the base logits bit for bit (gated-dense
  through `alpha = tanh(0) = 0`, adapter-matched through a zero up-projection);
* one registered CE+KL step produces finite gradients in the control's own
  parameters and in the gate, so the wiring is trainable, not just installable;
* the correction objective and the 8-bit optimiser are refused on a control,
  exactly as they are on the other registered adapter/memory arms.

No GPU work happens here.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import arms, train  # noqa: E402


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
        return SimpleNamespace(logits=self.lm_head(
            self.model(input_ids, past_key_values, use_cache)))


HIDDEN = 1024
VOCAB = 64


def _args(model, **overrides):
    base = dict(hidden=HIDDEN, site=0, model=model, memory_init=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def _tiny(seed: int = 0):
    torch.manual_seed(seed)
    model = _LM(VOCAB, HIDDEN, 3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _logits(model, ids):
    with torch.inference_mode():
        return model(input_ids=ids).logits.clone()


@pytest.mark.parametrize("kind", ["adapter-matched", "gated-dense"])
def test_build_arm_wires_the_registered_union_and_identity(kind):
    model = _tiny()
    module, handles, report = train.build_arm(kind, model, _args(model), "cpu")
    try:
        assert report["active_per_token"] == 539_649, report
        assert report["match_delta"] == 0, report
        assert report["identity_at_init"] is True, report
        assert len(handles) == 1
        assert any(parameter.requires_grad for parameter in module.parameters())
    finally:
        arms.uninstall(handles)


@pytest.mark.parametrize("kind", ["adapter-matched", "gated-dense"])
def test_an_untrained_control_is_the_base_bit_for_bit(kind):
    model = _tiny()
    ids = torch.randint(0, VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    before = _logits(model, ids)
    _module, handles, _report = train.build_arm(kind, model, _args(model), "cpu")
    try:
        after = _logits(model, ids)
    finally:
        arms.uninstall(handles)
    assert torch.equal(before, after), (kind, float((after - before).abs().max()))


@pytest.mark.parametrize("kind", ["adapter-matched", "gated-dense"])
def test_one_ce_kl_step_produces_finite_nonzero_gradients(kind):
    model = _tiny()
    module, handles, _report = train.build_arm(kind, model, _args(model), "cpu")
    try:
        ids = torch.randint(0, VOCAB, (2, 8),
                            generator=torch.Generator().manual_seed(2))
        inputs, targets = ids[:, :-1], ids[:, 1:]
        student = model(input_ids=inputs).logits
        with torch.no_grad():
            teacher = model(input_ids=inputs).logits + 0.1
        loss, ce, _kl = train.token_loss(student, teacher, targets, 2.0, 0.5,
                                         vocab_chunk=64)
        loss.backward()
        grads = [parameter.grad for parameter in module.parameters()
                 if parameter.grad is not None]
        assert grads, kind
        total = sum(float(grad.abs().sum()) for grad in grads)
        assert total > 0.0, f"{kind}: no gradient reached the control"
        assert float(ce) > 0.0
        assert all(torch.isfinite(grad).all() for grad in grads)
    finally:
        arms.uninstall(handles)
        model.zero_grad(set_to_none=True)


def test_the_gated_dense_gate_opens_on_the_first_step():
    model = _tiny()
    module, handles, _report = train.build_arm("gated-dense", model, _args(model), "cpu")
    try:
        ids = torch.randint(0, VOCAB, (2, 8),
                            generator=torch.Generator().manual_seed(3))
        student = model(input_ids=ids[:, :-1]).logits
        student.sum().backward()
        # d alpha / d w at w = 0 is [h ; r] / 1, so the gate weight must move.
        gate_grad = float(module.gate_w.grad.abs().sum())
        assert gate_grad > 0.0, "the gate received no gradient"
    finally:
        arms.uninstall(handles)
        model.zero_grad(set_to_none=True)


def test_the_correction_objective_is_refused_on_a_control():
    with pytest.raises(SystemExit):
        train.validate_objective_args(SimpleNamespace(
            objective="correction", arm="gated-dense", kl_weight=0.5))
    with pytest.raises(SystemExit):
        train.validate_objective_args(SimpleNamespace(
            objective="sft", arm="adapter-matched", kl_weight=0.5))


def test_main_keeps_the_controls_on_the_registered_optimizer():
    argv = ["--arm", "adapter-matched", "--stream-manifest", "unused.json",
            "--run-dir", "unused", "--optimizer", "adam8bit"]
    with pytest.raises(SystemExit) as raised:
        train.main(argv)
    assert "AdamW" in str(raised.value)
