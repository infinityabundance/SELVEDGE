#!/usr/bin/env python3
"""An independent Qwen3.5-0.8B text-path forward in plain PyTorch.

Why this file exists
--------------------
The Rust runtime must eventually execute this checkpoint. Before kernels are
written, the *arithmetic* has to be pinned by something that is not the thing
being tested. `transformers==5.23.0` (pinned in `docker/requirements-oracle.txt`)
ships `Qwen3_5ForConditionalGeneration`, and it is the vendor reference. This
file is a second, independent derivation written from the config and the tensor
shapes only; the parity driver compares the two. If this file and `transformers`
agree token-for-token, an equation is not a matter of taste.

What is implemented
-------------------
The **text path only**: `model.language_model.*`. The vision tower and the MTP
head are deliberately absent; `selvedge inspect-model --active-path text-only`
accounts exactly this set.

Every constant below is read from the checkpoint's `config.json` at load time
and asserted against the tensor shapes, so a config change breaks loudly rather
than silently changing an equation. Nothing is hard-coded to 0.8B.

The equations, in the order the residual stream passes through them
------------------------------------------------------------------
Let `H` be hidden width, `V` vocabulary, `I` the FFN width, `L` the layer count.

1. **Embedding.** `h0 = embed_tokens[token]` (a gather; no arithmetic).

2. **Per layer** (`l = 0..L-1`), with `y` the running residual:

   `a = rmsnorm(y; g_ln)` where

       rmsnorm(x; g) = x / sqrt(mean(x^2) + eps) * (1 + g)

   The `1 + g` is the checkpoint's convention: the stored gains are
   *zero-centred*, so the effective gain is `1 + g`. The `transformers`
   implementation states this in a comment and it is the single most common way
   to get this model wrong.

   **2a. Gated DeltaNet layer** (`layer_types[l] == "linear_attention"`,
   eighteen of twenty-four layers here):

       qkv = x @ W_qkv.T                 # [T, 2*K + Vd],  K = k_heads*k_dim
       z   = x @ W_z.T                   # [T, Vd]
       b   = x @ W_b.T                   # [T, v_heads]
       a   = x @ W_a.T                   # [T, v_heads]

       qkv = silu(depthwise_causal_conv1d(qkv, W_conv, kernel=4))

       q, k, v = split(qkv, [K, K, Vd])
       beta = sigmoid(b)                                  # per value head
       g    = -exp(A_log) * softplus(a + dt_bias)          # <= 0, log-space decay

       q = l2norm_per_head(q) * k_dim**-0.5
       k = l2norm_per_head(k)

       S = zeros(k_dim, v_dim) per head                   # recurrent state
       for t in range(T):
           S = S * exp(g[t])                              # forget
           delta = (v[t] - S^T k[t]) * beta[t]            # delta rule
           S = S + k[t] (x) delta                         # write
           o[t] = S^T q[t]                                # read

       o = rmsnorm_gated(o; W_norm) * silu(z)             # per value head, over
                                                          # its 128 channels;
                                                          # the norm carries the
                                                          # gate, it is not 1+W
       y = y + o @ W_out.T

   The convolution is *depthwise* over the concatenated `[q;k;v]` channels with
   kernel 4, causal (left-padded by 3), followed by SiLU. It is a short state,
   not a recurrence: token `t` sees `t-3..t` of the projected, pre-conv values.

   **2b. Gated attention layer** (six layers: indices 3, 7, 11, 15, 19, 23):

       qg  = x @ W_q.T                   # [T, heads * 2 * head_dim]
       q, gate = reshape(qg, [T, heads, 2*head_dim]).chunk(2, -1)
       k = x @ W_k.T                     # [T, kv_heads * head_dim]
       v = x @ W_v.T
       q = rmsnorm(q; g_qnorm)           # over head_dim
       k = rmsnorm(k; g_knorm)
       q, k = rope(q, k)                 # rotary_dim = head_dim * 0.25 = 64
       o = softmax(q k^T * head_dim**-0.5, causal) @ v      # 8 query / 2 kv heads
       o = o * sigmoid(gate)             # fused output gate
       y = y + o @ W_o.T

   **2c. FFN (SwiGLU)**, identical on every layer:

       y = y + (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T

   **2d. Residual order.** Attention output is added to the *pre-norm* residual,
   then the FFN is applied to a fresh norm of that sum:

       y = y_pre + mixer(rmsnorm(y_pre; g_in))
       y = y     + mlp(rmsnorm(y;     g_post))

3. **Final norm and head.** `logits = rmsnorm(y; g_final) @ embed_tokens.T`. The
   head is **tied**: the checkpoint has no `lm_head.weight`, and `tie_word_embeddings`
   is true, so the embedding matrix *is* the output projection.

4. **RoPE details.** Frequencies are
   `inv_freq[j] = rope_theta ** (-2j / rotary_dim)` for `j < rotary_dim/2`, and
   `cos/sin` are built at width `rotary_dim` by concatenating the half-width
   tables to themselves. `rotate_half` therefore pairs element `j` with
   `j + rotary_dim/2` — half-split rotation, not interleaved.

   The config also declares mRoPE (`mrope_section [11, 11, 10], interleaved`).
   For a text-only sequence all four position-id rows (text, temporal, height,
   width) are the same `arange`, so the mRoPE recomposition selects identical
   frequencies and the result is *exactly* the standard RoPE above. The parity
   driver verifies this rather than asserting it: the reference and
   `transformers` (which does run the mRoPE machinery) must agree.

5. **Norms are computed in fp32** and the result is cast back to the input dtype;
   the gated norm applies `silu` to the incoming `z` in fp32 as well. Softmax is
   computed in fp32. Everything else follows the weight dtype.

Usage
-----
    python3 oracle/reference/qwen35.py --checkpoint DIR --prompt "..." [--dtype bf16]

The module is importable; `scripts/s33_reference_parity.py` is the driver that
produces the receipts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEFAULT_CHECKPOINT = os.environ.get(
    "SELVEDGE_QWEN35",
    "/hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17",
)

TEXT_PREFIX = "model.language_model."


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """`x / sqrt(mean(x^2) + eps) * (1 + gain)`, computed in fp32.

    The `1 + gain` is not a convention this file chose: the checkpoint stores
    zero-centred norm gains, so the effective multiplier is `1 + gain`. Reading
    the stored value as the gain itself is a silent, near-total-model error.
    """
    input_dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + gain.float()).to(out.dtype)
    return out.to(input_dtype)


def rms_norm_gated(
    x: torch.Tensor, gain: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    """`rmsnorm(x) * gain * silu(gate)`, with both norms and SiLU in fp32.

    The gain is used directly, not as `1 + gain`: this norm is one-initialised
    (and stored in fp32), unlike the zero-centred gains used everywhere else.
    The normalisation runs over the last axis, which is one value head's
    channels, not the whole value vector.
    """
    input_dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = gain.float() * out.to(input_dtype)
    out = out * torch.nn.functional.silu(gate.float())
    return out.to(input_dtype)


def depthwise_causal_conv1d_silu(
    x: torch.Tensor, weight: torch.Tensor, kernel: int
) -> torch.Tensor:
    """Causal depthwise convolution over the channel axis, then SiLU.

    `x` is `[T, C]`, `weight` is `[C, 1, kernel]`. Output token `t` is
    `sum_k weight[c, 0, k] * x[t - (kernel - 1 - k), c]`, with zero padding on the
    left, so no token sees the future.
    """
    if kernel <= 0:
        raise ValueError("conv kernel must be positive")
    times = x.shape[0]
    padded = torch.nn.functional.pad(x.t().unsqueeze(0), (kernel - 1, 0))
    out = torch.nn.functional.conv1d(
        padded, weight, bias=None, groups=x.shape[1]
    )  # [1, C, T]
    out = out[0].t()[:times]
    return torch.nn.functional.silu(out)


def l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x / sqrt(sum(x^2) + eps)` over the last axis, in fp32 (FLA convention)."""
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + eps)).to(x.dtype)


def rope_tables(
    positions: torch.Tensor, rotary_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `cos`/`sin` of width `rotary_dim` for the half-split rotation.

    `transformers` computes the half-width table and concatenates it to itself,
    which makes `rotate_half` pair element `j` with `j + rotary_dim/2`.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device)
            / rotary_dim
        )
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    table = torch.cat((freqs, freqs), dim=-1)
    return table.cos().to(dtype), table.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`cat(-x2, x1)` over the last axis (half-split rotation)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first `cos.shape[-1]` channels of `q` and `k`; pass the rest."""
    rotary = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary], q[..., rotary:]
    k_rot, k_pass = k[..., :rotary], k[..., rotary:]
    # `cos` is [T, rotary]; q and k are [batch, heads, T, head_dim].
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    k_out = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return q_out, k_out


def gated_delta_rule_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule, in its recurrent form, one token at a time.

    `query` / `key` are `[T, H, Dk]` (already L2-normalised and scaled), `value`
    is `[T, H, Dv]`, `decay` is `[T, H]` in log space (<= 0), `beta` is `[T, H]`.

    State `S` is `[H, Dk, Dv]` and starts at zero (or at `initial_state`):

        S   <- S * exp(g_t)
        delta = (v_t - S^T k_t) * beta_t
        S   <- S + k_t (x) delta
        o_t = S^T q_t

    All of it runs in fp32, matching the reference implementation in
    `transformers` and FLA. The final state is returned so a caller can carry it
    into the next chunk (and so the state test can prove it is carried).
    """
    times, heads, k_dim = key.shape
    v_dim = value.shape[-1]
    q = query.float().transpose(0, 1)  # [H, T, Dk]
    k = key.float().transpose(0, 1)
    v = value.float().transpose(0, 1)
    g = decay.float().transpose(0, 1)  # [H, T]
    b = beta.float().transpose(0, 1)
    if initial_state is None:
        state = torch.zeros(
            heads, k_dim, v_dim, dtype=torch.float32, device=key.device
        )
    else:
        state = initial_state.float().clone()
    out = torch.zeros(heads, times, v_dim, dtype=torch.float32, device=key.device)
    for t in range(times):
        state = state * g[:, t].exp()[:, None, None]
        kv_mem = torch.bmm(k[:, t].unsqueeze(1), state).squeeze(1)  # [H, Dv]
        delta = (v[:, t] - kv_mem) * b[:, t].unsqueeze(-1)
        state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(1)
        out[:, t] = torch.bmm(q[:, t].unsqueeze(1), state).squeeze(1)
    return out.transpose(0, 1).to(value.dtype), state


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextConfig:
    """The subset of `text_config` the text path reads."""

    hidden: int
    intermediate: int
    layers: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    attn_output_gate: bool
    lin_key_heads: int
    lin_key_dim: int
    lin_value_heads: int
    lin_value_dim: int
    conv_kernel: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.lin_key_heads * self.lin_key_dim

    @property
    def value_dim(self) -> int:
        return self.lin_value_heads * self.lin_value_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


def load_text_config(checkpoint: str) -> tuple[TextConfig, dict[str, Any]]:
    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    text = raw["text_config"]
    rope = text["rope_parameters"]
    config = TextConfig(
        hidden=int(text["hidden_size"]),
        intermediate=int(text["intermediate_size"]),
        layers=int(text["num_hidden_layers"]),
        vocab=int(text["vocab_size"]),
        heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        layer_types=tuple(text["layer_types"]),
        attn_output_gate=bool(text["attn_output_gate"]),
        lin_key_heads=int(text["linear_num_key_heads"]),
        lin_key_dim=int(text["linear_key_head_dim"]),
        lin_value_heads=int(text["linear_num_value_heads"]),
        lin_value_dim=int(text["linear_value_head_dim"]),
        conv_kernel=int(text["linear_conv_kernel_dim"]),
        rope_theta=float(rope["rope_theta"]),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
        rms_norm_eps=float(text["rms_norm_eps"]),
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
    )
    if len(config.layer_types) != config.layers:
        raise ValueError("layer_types length does not match num_hidden_layers")
    if config.rotary_dim % 2 != 0:
        raise ValueError(f"rotary dim {config.rotary_dim} must be even")
    if not config.tie_word_embeddings:
        raise ValueError("this reference implements the tied head only")
    return config, raw


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class Qwen35TextReference(nn.Module):
    """The text path as a module so weights move between dtypes and devices."""

    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        # A plain module dict: names match the checkpoint keys exactly (minus the
        # stored prefix), so the loader is a rename and never a guess.
        self.weights = nn.ParameterDict()

    # -- loading ---------------------------------------------------------
    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Take the text-path tensors, verifying every shape against the config."""
        config = self.config
        expected = self._expected_shapes()
        for name, shape in expected.items():
            key = TEXT_PREFIX + name
            if key not in state:
                raise KeyError(f"checkpoint has no {key}")
            tensor = state[key]
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"{key} has shape {tuple(tensor.shape)}, config implies {tuple(shape)}"
                )
            self.weights[name.replace(".", "__")] = nn.Parameter(
                tensor.detach().clone(), requires_grad=False
            )
        unexpected = [
            key
            for key in state
            if key.startswith(TEXT_PREFIX)
            and key[len(TEXT_PREFIX) :] not in expected
            and key != TEXT_PREFIX + "embed_tokens.weight"
        ]
        if unexpected:
            raise ValueError(f"unclassified text tensors: {unexpected[:4]}")
        _ = config

    def _expected_shapes(self) -> dict[str, tuple[int, ...]]:
        config = self.config
        hidden, intermediate = config.hidden, config.intermediate
        q_out = config.heads * config.head_dim * (2 if config.attn_output_gate else 1)
        kv_out = config.kv_heads * config.head_dim
        shapes: dict[str, tuple[int, ...]] = {
            "embed_tokens.weight": (config.vocab, hidden),
            "norm.weight": (hidden,),
        }
        for layer in range(config.layers):
            prefix = f"layers.{layer}"
            shapes[f"{prefix}.input_layernorm.weight"] = (hidden,)
            shapes[f"{prefix}.post_attention_layernorm.weight"] = (hidden,)
            shapes[f"{prefix}.mlp.gate_proj.weight"] = (intermediate, hidden)
            shapes[f"{prefix}.mlp.up_proj.weight"] = (intermediate, hidden)
            shapes[f"{prefix}.mlp.down_proj.weight"] = (hidden, intermediate)
            if config.layer_types[layer] == "full_attention":
                shapes[f"{prefix}.self_attn.q_proj.weight"] = (q_out, hidden)
                shapes[f"{prefix}.self_attn.k_proj.weight"] = (kv_out, hidden)
                shapes[f"{prefix}.self_attn.v_proj.weight"] = (kv_out, hidden)
                shapes[f"{prefix}.self_attn.o_proj.weight"] = (
                    hidden,
                    config.heads * config.head_dim,
                )
                shapes[f"{prefix}.self_attn.q_norm.weight"] = (config.head_dim,)
                shapes[f"{prefix}.self_attn.k_norm.weight"] = (config.head_dim,)
            elif config.layer_types[layer] == "linear_attention":
                shapes[f"{prefix}.linear_attn.in_proj_qkv.weight"] = (
                    config.conv_dim,
                    hidden,
                )
                shapes[f"{prefix}.linear_attn.in_proj_z.weight"] = (
                    config.value_dim,
                    hidden,
                )
                shapes[f"{prefix}.linear_attn.in_proj_a.weight"] = (
                    config.lin_value_heads,
                    hidden,
                )
                shapes[f"{prefix}.linear_attn.in_proj_b.weight"] = (
                    config.lin_value_heads,
                    hidden,
                )
                shapes[f"{prefix}.linear_attn.conv1d.weight"] = (
                    config.conv_dim,
                    1,
                    config.conv_kernel,
                )
                shapes[f"{prefix}.linear_attn.A_log"] = (config.lin_value_heads,)
                shapes[f"{prefix}.linear_attn.dt_bias"] = (config.lin_value_heads,)
                shapes[f"{prefix}.linear_attn.norm.weight"] = (config.lin_value_dim,)
                shapes[f"{prefix}.linear_attn.out_proj.weight"] = (
                    hidden,
                    config.value_dim,
                )
            else:
                raise ValueError(f"unknown layer type {config.layer_types[layer]!r}")
        return shapes

    def w(self, name: str) -> torch.Tensor:
        return self.weights[name.replace(".", "__")]

    def named_weights(self) -> dict[str, torch.Tensor]:
        """The loaded tensors under their checkpoint names (without the prefix)."""
        return {name.replace("__", "."): param for name, param in self.weights.items()}

    # -- pieces ----------------------------------------------------------
    def _linear_attention(
        self,
        x: torch.Tensor,
        layer: int,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Gated DeltaNet layer.

        Returns `(output, conv_state, recurrent_state)`, where `conv_state` is
        the last `kernel - 1` columns of the *raw* `[q;k;v]` projection (the
        convolution's memory) and `recurrent_state` is `[heads, k_dim, v_dim]`.
        Passing them back in continues the sequence exactly; passing `None`
        resets to the zero state. The audited `forward` never passes them.
        """
        config = self.config
        prefix = f"layers.{layer}.linear_attn"
        qkv = self.w(f"{prefix}.in_proj_qkv.weight")
        z = self.w(f"{prefix}.in_proj_z.weight")
        w_b = self.w(f"{prefix}.in_proj_b.weight")
        w_a = self.w(f"{prefix}.in_proj_a.weight")
        conv = self.w(f"{prefix}.conv1d.weight")

        times = x.shape[0]
        projected = torch.nn.functional.linear(x, qkv)
        if conv_state is not None:
            projected = torch.cat((conv_state, projected), dim=0)
        new_conv_state = projected[-(config.conv_kernel - 1) :].detach()
        mixed = depthwise_causal_conv1d_silu(projected, conv, config.conv_kernel)
        mixed = mixed[-times:]

        gate_values = torch.nn.functional.linear(x, z)
        beta = torch.nn.functional.linear(x, w_b).sigmoid()
        decay = -self.w(f"{prefix}.A_log").float().exp() * torch.nn.functional.softplus(
            torch.nn.functional.linear(x, w_a).float()
            + self.w(f"{prefix}.dt_bias").float()
        )

        q, k, v = torch.split(
            mixed, [config.key_dim, config.key_dim, config.value_dim], dim=-1
        )
        q = q.view(times, config.lin_key_heads, config.lin_key_dim)
        k = k.view(times, config.lin_key_heads, config.lin_key_dim)
        v = v.view(times, config.lin_value_heads, config.lin_value_dim)
        q = l2_norm(q) * (config.lin_key_dim**-0.5)
        k = l2_norm(k)

        core, new_state = gated_delta_rule_recurrent(
            q, k, v, decay, beta, initial_state=recurrent_state
        )
        core = core.view(times, config.lin_value_heads, config.lin_value_dim)
        gate_values = gate_values.view(times, config.lin_value_heads, config.lin_value_dim)
        gated = rms_norm_gated(
            core,
            self.w(f"{prefix}.norm.weight"),
            gate_values,
            config.rms_norm_eps,
        )
        gated = gated.reshape(times, config.value_dim)
        out = torch.nn.functional.linear(gated, self.w(f"{prefix}.out_proj.weight"))
        return out, new_conv_state, new_state

    def _attention(
        self,
        x: torch.Tensor,
        layer: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        prefix = f"layers.{layer}.self_attn"
        times = x.shape[0]
        gate_width = config.heads * config.head_dim

        qg = torch.nn.functional.linear(x, self.w(f"{prefix}.q_proj.weight"))
        qg = qg.view(times, config.heads, 2 * config.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        k = torch.nn.functional.linear(x, self.w(f"{prefix}.k_proj.weight")).view(
            times, config.kv_heads, config.head_dim
        )
        v = torch.nn.functional.linear(x, self.w(f"{prefix}.v_proj.weight")).view(
            times, config.kv_heads, config.head_dim
        )
        q = rms_norm(q, self.w(f"{prefix}.q_norm.weight"), config.rms_norm_eps)
        k = rms_norm(k, self.w(f"{prefix}.k_norm.weight"), config.rms_norm_eps)

        q = q.transpose(0, 1)  # [heads, T, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        q, k = apply_rope(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q, k = q[0], k[0]

        repeat = config.heads // config.kv_heads
        k = k.repeat_interleave(repeat, dim=0)
        v = v.repeat_interleave(repeat, dim=0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (config.head_dim**-0.5)
        causal = torch.ones(times, times, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal[None, :, :], float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(weights, v)  # [heads, T, head_dim]
        out = out.transpose(0, 1).reshape(times, gate_width)
        out = out * torch.sigmoid(gate.reshape(times, gate_width))
        return torch.nn.functional.linear(out, self.w(f"{prefix}.o_proj.weight"))

    def _mlp(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"layers.{layer}.mlp"
        gate = torch.nn.functional.linear(x, self.w(f"{prefix}.gate_proj.weight"))
        up = torch.nn.functional.linear(x, self.w(f"{prefix}.up_proj.weight"))
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, self.w(f"{prefix}.down_proj.weight")
        )

    # -- forward ---------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits `[T, V]` for one unpadded sequence."""
        config = self.config
        if input_ids.ndim != 1:
            raise ValueError("the reference forward takes one 1-D token sequence")
        times = int(input_ids.shape[0])
        positions = torch.arange(times, device=input_ids.device)
        cos, sin = rope_tables(
            positions, config.rotary_dim, config.rope_theta, self.dtype
        )

        hidden = torch.nn.functional.embedding(input_ids, self.w("embed_tokens.weight"))
        for layer in range(config.layers):
            residual = hidden
            normed = rms_norm(
                hidden, self.w(f"layers.{layer}.input_layernorm.weight"), config.rms_norm_eps
            )
            if config.layer_types[layer] == "full_attention":
                mixed = self._attention(normed, layer, cos, sin)
            else:
                mixed, _, _ = self._linear_attention(normed, layer)
            hidden = residual + mixed

            residual = hidden
            normed = rms_norm(
                hidden,
                self.w(f"layers.{layer}.post_attention_layernorm.weight"),
                config.rms_norm_eps,
            )
            hidden = residual + self._mlp(normed, layer)

        hidden = rms_norm(hidden, self.w("norm.weight"), config.rms_norm_eps)
        embed = self.w("embed_tokens.weight")
        return torch.nn.functional.linear(hidden, embed)

    @property
    def dtype(self) -> torch.dtype:
        return self.w("embed_tokens.weight").dtype

    @torch.no_grad()
    def greedy(self, input_ids: torch.Tensor, steps: int) -> list[int]:
        """Greedy continuation, recomputing the full prefix each step."""
        ids = input_ids.tolist()
        generated: list[int] = []
        for _ in range(steps):
            logits = self.forward(torch.tensor(ids, device=input_ids.device))
            next_id = int(torch.argmax(logits[-1].float()).item())
            ids.append(next_id)
            generated.append(next_id)
        return generated


def load_reference(
    checkpoint: str, dtype: torch.dtype, device: str
) -> tuple[Qwen35TextReference, TextConfig]:
    """Load the text-path weights from the checkpoint's safetensors files.

    Only `model.language_model.*` keys are read; the vision tower and the MTP
    head are not loaded, which is what makes this the text path rather than an
    approximation of the checkpoint.
    """
    from safetensors import safe_open  # local import: only the loader needs it

    config, _raw = load_text_config(checkpoint)
    state: dict[str, torch.Tensor] = {}
    files = sorted(
        name
        for name in os.listdir(checkpoint)
        if name.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"no safetensors files under {checkpoint}")
    for name in files:
        with safe_open(os.path.join(checkpoint, name), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(TEXT_PREFIX):
                    state[key] = handle.get_tensor(key)
    model = Qwen35TextReference(config)
    model.load_state(state)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, config


def linear_attention_state_test(
    model: Qwen35TextReference,
    layer: int = 0,
    times: int = 12,
    split: int = 5,
    seed: int = 20261116,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Prove the DeltaNet state is carried across a chunk boundary, and reset.

    Three runs of one real layer with real weights:

    1. the whole sequence in one pass (the definition);
    2. `[0, split)` then `[split, times)` with the conv and recurrent state
       carried across the boundary;
    3. `[split, times)` again from a *fresh* state.

    Run (2) must reproduce run (1); run (3) must not. Together those two facts
    mean the state is both necessary and correctly carried. The comparison is
    against bf16 activations, so the tolerance is 1e-3 rather than exact.
    """
    config = model.config
    if config.layer_types[layer] != "linear_attention":
        raise ValueError(f"layer {layer} is {config.layer_types[layer]}, not linear_attention")
    generator = torch.Generator().manual_seed(seed)
    x = (
        torch.randn(times, config.hidden, generator=generator)
        .to(device=model.w("embed_tokens.weight").device, dtype=model.dtype)
        * 0.5
    )
    with torch.no_grad():
        full, _, _ = model._linear_attention(x, layer)
        head, conv_state, recurrent_state = model._linear_attention(x[:split], layer)
        tail, _, _ = model._linear_attention(
            x[split:], layer, conv_state=conv_state, recurrent_state=recurrent_state
        )
        fresh_tail, _, _ = model._linear_attention(x[split:], layer)
    chunked = torch.cat((head, tail), dim=0)
    carried = (chunked.float() - full.float()).abs()
    reset = (fresh_tail.float() - full[split:].float()).abs()
    carried_max = float(carried.max().item())
    reset_max = float(reset.max().item())
    return {
        "layer": layer,
        "layer_type": config.layer_types[layer],
        "times": times,
        "split": split,
        "conv_kernel": config.conv_kernel,
        "carry_max_abs": carried_max,
        "carry_mean_abs": float(carried.mean().item()),
        "reset_max_abs": reset_max,
        "reset_mean_abs": float(reset.mean().item()),
        "tolerance": tolerance,
        "carry_matches": carried_max <= tolerance,
        "reset_differs": reset_max > carried_max * 100.0,
        "state_shape": list(recurrent_state.shape),
        "conv_state_shape": list(conv_state.shape),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, config = load_reference(args.checkpoint, dtype, args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"][0].to(args.device)
    logits = model.forward(ids)
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "tokens": int(ids.shape[0]),
                "logits_shape": list(logits.shape),
                "dtype": str(dtype),
                "device": args.device,
                "argmax_next": int(torch.argmax(logits[-1].float()).item()),
                "next_token": tokenizer.decode(
                    [int(torch.argmax(logits[-1].float()).item())]
                ),
                "layers": config.layers,
                "rotary_dim": config.rotary_dim,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
