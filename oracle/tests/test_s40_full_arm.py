"""CPU tests for the S40 full-backbone control and its 8-bit Adam (WS19).

The full arm is the bar: it freezes the tied token embedding and trains every
other text-tower identity under the online teacher. These tests prove the parts
a GPU receipt cannot: the frozen/trainable partition is exactly the tied
embedding, the exported state cannot carry the embedding, and the block-wise
8-bit Adam behaves like Adam and stores what it says it stores.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import full, optim8bit  # noqa: E402

BLOCK = 64


# ---------------------------------------------------------------------------
# block-wise 8-bit Adam


def test_adam8bit_first_step_matches_adamw_direction():
    torch.manual_seed(0)
    start = torch.randn(4, 128)
    grad = torch.randn(4, 128)
    one = nn.Parameter(start.clone())
    one.grad = grad.clone()
    two = nn.Parameter(start.clone())
    two.grad = grad.clone()
    eight = optim8bit.Adam8bit([one], lr=1e-3, betas=(0.9, 0.95), block=BLOCK)
    reference = torch.optim.AdamW([two], lr=1e-3, betas=(0.9, 0.95))
    eight.step()
    reference.step()
    # int8 block quantization of the moments is ~1/127 relative
    relative = float((one.detach() - two.detach()).norm() / two.detach().norm())
    assert relative < 0.02


def test_adam8bit_converges_on_a_quadratic():
    torch.manual_seed(1)
    target = torch.zeros(512)
    eight_param = nn.Parameter(torch.ones(512))
    eight = optim8bit.Adam8bit([eight_param], lr=1e-2, betas=(0.9, 0.95), block=BLOCK)
    reference_param = nn.Parameter(torch.ones(512))
    reference = torch.optim.AdamW([reference_param], lr=1e-2, betas=(0.9, 0.95))
    for _ in range(300):
        for parameter, optimizer in ((eight_param, eight),
                                     (reference_param, reference)):
            optimizer.zero_grad()
            loss = ((parameter - target) ** 2).sum()
            loss.backward()
            optimizer.step()
    eight_loss = float(((eight_param.detach() - target) ** 2).sum())
    reference_loss = float(((reference_param.detach() - target) ** 2).sum())
    assert eight_loss < 1e-3
    assert eight_loss <= reference_loss * 10.0 + 1e-6


def test_adam8bit_stores_int8_moments_and_reports_their_bytes():
    parameter = nn.Parameter(torch.randn(1000))
    parameter.grad = torch.randn(1000)
    optimizer = optim8bit.Adam8bit([parameter], block=BLOCK)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["q_m"].dtype == torch.int8
    assert state["q_v"].dtype == torch.int8
    assert state["q_m"].numel() == state["padded"] == 1024
    reported = optimizer.state_bytes()
    # two int8 moments + two float32 scales per 64-element block
    expected = 2 * 1024 + 2 * (1024 // BLOCK) * 4
    assert reported == expected
    assert 2.0 < optimizer.moment_bytes_per_element() < 2.2
    receipt = optimizer.receipt()
    assert receipt["moment_bytes_per_element"] > 2.0
    assert receipt["block_elements"] == BLOCK


def test_adam8bit_quantize_dequantize_round_trip_is_bounded():
    flat = torch.randn(300) * torch.logspace(-3, 3, 300)
    quantized, absmax = optim8bit.quantize(flat, BLOCK)
    restored = optim8bit.dequantize(quantized, absmax, flat.numel(), BLOCK)
    assert restored.shape == flat.shape
    # per-block absmax bounds the error by one quantization step
    error = (restored - flat).abs()
    step = absmax.repeat_interleave(BLOCK)[:flat.numel()] / 127.0
    assert bool((error <= step + 1e-6).all())


# ---------------------------------------------------------------------------
# the full-backbone arm


class _TinyCore(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.model = _TinyCore(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        # tied, exactly as the anchor: one Parameter object, no lm_head tensor
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_full_arm_freezes_only_the_tied_embedding():
    model = _TinyLM(vocab=32, hidden=8)
    arm = full.FullBackboneArm(model)
    assert arm.text_elements == 32 * 8 + 2 * (8 * 8 + 8) + 2 * 8
    assert arm.frozen_elements == 32 * 8
    assert arm.trainable_elements == arm.text_elements - arm.frozen_elements
    assert arm.frozen_identities == ["embed_tokens.weight"]
    report = arm.active_parameter_report()
    assert report["deployed_active_union_elements"] == arm.text_elements
    assert report["tied_output_projection"] is True
    # the tied output projection is frozen with the embedding
    assert model.lm_head.weight.requires_grad is False
    assert all(parameter.requires_grad for parameter in arm.parameters())


def test_full_arm_state_excludes_the_embedding_and_round_trips():
    model = _TinyLM(vocab=32, hidden=8)
    arm = full.FullBackboneArm(model)
    state = arm.state_dict()
    assert "embed_tokens.weight" not in state
    assert all(not name.startswith("embed_tokens") for name in state)
    # move every trainable parameter, then restore
    with torch.no_grad():
        for name, parameter in arm.named_trainable():
            parameter.add_(1.0)
    arm.load_state_dict(state)
    with torch.no_grad():
        for name, parameter in arm.named_trainable():
            assert torch.allclose(parameter, state[name].to(parameter.dtype))


def test_full_arm_rejects_a_state_that_carries_the_embedding():
    model = _TinyLM(vocab=32, hidden=8)
    arm = full.FullBackboneArm(model)
    state = arm.state_dict()
    state["embed_tokens.weight"] = torch.zeros(32, 8)
    with pytest.raises(ValueError):
        arm.load_state_dict(state)


def test_full_arm_bytes_arithmetic_is_exact():
    report = full.bytes_report(text_elements=752_393_024,
                               frozen_embedding_elements=254_279_680,
                               trainable_elements=498_113_344,
                               optimizer_bytes=1_011_792_730)
    assert report["trainable_weights_bytes"] == 996_226_688
    assert report["gradients_bytes"] == 996_226_688
    assert report["frozen_embedding_bytes"] == 508_559_360
    assert (report["text_tower_elements"]
            == report["trainable_elements"] + report["frozen_embedding_elements"])
    assert report["total_training_state_bytes"] == (
        report["student_weights_bytes"] + report["gradients_bytes"]
        + report["optimizer_state_bytes"])
