"""The recurrent-state contract, tested on a tiny random model and the anchor.

The plan made a statement about temporal state that is easy to get exactly
backwards: a pass's state must be **carried across tokens** and reset only
**between sequences**, and each pass must own its own state. This file tests the
three rules on the reference implementation of the text path:

* `multi_pass_state_test` on a tiny randomly-initialised checkpoint, which runs
  anywhere in seconds and needs no model cache;
* the same test on the real `-Base` anchor when its cache is present (it is
  skipped otherwise, so the CPU court still passes on a machine without it).

A third test pins the fail-closed behaviour of the attention cache: a cache may
only be continued one token at a time, because a chunked attention forward is not
implemented and silently producing one would be a different computation.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reference import qwen35 as ref  # noqa: E402

TEXT_PREFIX = ref.TEXT_PREFIX

#: The WS19 anchor is the **-Base** revision (`docs/FRONTIER-TARGET.md` §10).
ANCHOR_CHECKPOINT = os.environ.get(
    "SELVEDGE_QWEN35_BASE",
    "/hf-cache/models/Qwen/Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
)


def tiny_config() -> ref.TextConfig:
    """A four-layer text tower: three DeltaNet layers and one attention layer."""
    return ref.TextConfig(
        hidden=32,
        intermediate=64,
        layers=4,
        vocab=256,
        heads=4,
        kv_heads=2,
        head_dim=16,
        layer_types=(
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ),
        attn_output_gate=True,
        lin_key_heads=2,
        lin_key_dim=8,
        lin_value_heads=2,
        lin_value_dim=8,
        conv_kernel=4,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


def tiny_model(config: ref.TextConfig, seed: int = 7) -> ref.Qwen35TextReference:
    """A model with the checkpoint's own tensor names and random small values."""
    generator = torch.Generator().manual_seed(seed)
    model = ref.Qwen35TextReference(config)
    state = {}
    for name, shape in model._expected_shapes().items():
        tensor = (torch.randn(shape, generator=generator) * 0.1).float()
        if name.endswith("A_log"):
            tensor = -torch.rand(shape, generator=generator) * 0.5
        elif name.endswith("dt_bias"):
            tensor = torch.zeros(shape)
        elif name.endswith("linear_attn.norm.weight"):
            # `rms_norm_gated` uses this gain directly, and it is one-initialised.
            # This branch must precede the generic zero-centred-gain branch below,
            # or the DeltaNet output is multiplied by zero and the state becomes
            # unobservable at the logits.
            tensor = torch.ones(shape)
        elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            # zero-centred gains, so the effective multiplier is `1 + g`
            tensor = torch.zeros(shape)
        state[TEXT_PREFIX + name] = tensor
    model.load_state(state)
    model.eval()
    return model


def test_state_is_carried_across_tokens_reset_between_sequences_and_pass_isolated():
    model = tiny_model(tiny_config())
    result = ref.multi_pass_state_test(model, passes=2, times=8, reset_after=4)
    assert result["carry"]["argmax_equal"], result["carry"]
    assert result["carry"]["ok"], result["carry"]
    assert result["chunk"]["bit_identical"], result["chunk"]
    assert result["reset"]["reset_equals_fresh"], result["reset"]
    assert result["reset"]["not_reset_differs"], result["reset"]
    assert result["isolation"]["pass0_state_bit_identical"], result["isolation"]
    assert result["isolation"]["last_pass_state_differs"], result["isolation"]
    assert result["isolation"]["distinct_storage"], result["isolation"]
    assert result["all_ok"], result


def test_two_passes_do_not_change_the_single_pass_state_or_each_others():
    """Pass 0's trajectory must be independent of how many passes follow it."""
    model = tiny_model(tiny_config())
    config = model.config
    ids = torch.arange(1, 6, dtype=torch.long)

    one = ref.initial_state(config, passes=1, device="cpu", dtype=torch.float32)
    three = ref.initial_state(config, passes=3, device="cpu", dtype=torch.float32)
    for token in ids.tolist():
        _, one = model.step_token(token, one)
        _, three = model.step_token(token, three)
    for layer, tensor in one[0].recurrent.items():
        assert torch.equal(tensor, three[0].recurrent[layer])
    for layer, pair in one[0].kv.items():
        assert torch.equal(pair[0], three[0].kv[layer][0])
        assert torch.equal(pair[1], three[0].kv[layer][1])


def test_attention_cache_refuses_a_multi_token_call():
    model = tiny_model(tiny_config())
    config = model.config
    cos, sin = ref.rope_tables(
        torch.arange(2), config.rotary_dim, config.rope_theta, torch.float32
    )
    x = torch.zeros(2, config.hidden)
    cache = (
        torch.zeros(config.kv_heads, 1, config.head_dim),
        torch.zeros(config.kv_heads, 1, config.head_dim),
    )
    with pytest.raises(ValueError, match="one token at a time"):
        model._attention(x, 3, cos, sin, kv=cache)


@pytest.mark.skipif(
    not os.path.isdir(ANCHOR_CHECKPOINT),
    reason="the -Base anchor checkpoint is not cached on this machine",
)
def test_state_contract_holds_on_the_anchor():
    model, _config = ref.load_reference(ANCHOR_CHECKPOINT, torch.bfloat16, "cpu")
    result = ref.multi_pass_state_test(model, passes=2, times=4, reset_after=2)
    assert result["all_ok"], result
