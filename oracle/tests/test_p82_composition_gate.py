"""P8.2 — CPU tests for gate-only training, the composed arm and the comparison.

The registered GPU run needs the frozen 0.6B core, so this file tests the parts
that can be wrong without a GPU:

* the zero point (`y = S` with a zero gate) on a tiny end-to-end composition;
* reachability (the gate can learn D-only, M-only and both coefficients);
* the L2 projection, including the relative budget `rho_t = kappa ||S_t||`;
* no leakage from the two residuals into the gate's coefficients;
* determinism of gate-only training;
* that training touches the gates and nothing else;
* the checkpoint remap (fail-closed) and the paired comparison with the rubric's
  bootstrap.

Nothing here is a measured P8.2 result; the numbers are fixture arithmetic.
"""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORACLE, "eval"))
sys.path.insert(0, os.path.join(ORACLE, "train"))

import complementarity  # noqa: E402
import composition  # noqa: E402
from memory import MemoryGeometry  # noqa: E402


# ---------------------------------------------------------------------------
# a tiny deterministic transformer-shaped core


class _Layer(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(hidden) * 0.5)


class _Inner(torch.nn.Module):
    def __init__(self, hidden: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(hidden) for _ in range(layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = hidden_states + layer.mlp(hidden_states)
        return hidden_states


class _TinyCore(torch.nn.Module):
    """`forward(input_ids, use_cache=...)` with `.model.layers` and `.config`."""

    def __init__(self, hidden: int = 8, vocab: int = 13, layers: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.model = _Inner(hidden, layers)
        self.config = SimpleNamespace(hidden_size=hidden)
        generator = torch.Generator().manual_seed(4)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, hidden, generator=generator))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        hidden_states = self.embed(input_ids)
        hidden_states = self.model(hidden_states)
        return SimpleNamespace(logits=hidden_states)


def _arm(hidden: int = 8, layers: int = 2, kappa: float = 0.5):
    core = _TinyCore(hidden=hidden, layers=layers)
    config_d = {"arm": "D", "total_accessible": 1,
                "execution_uses": [], "selector_uses": [],
                "execution_flops": {"matmul_flops": 0, "other_flops": 0,
                                    "controller_flops": 0}}
    config_m = {"arm": "M", "total_accessible": 1,
                "execution_uses": [], "selector_uses": [],
                "execution_flops": {"matmul_flops": 0, "other_flops": 0,
                                    "controller_flops": 0}}
    return core, composition.build_composition_arm(
        core, config_d, {}, config_m, {}, seed=1001, kappa=kappa,
        dense_width=4, memory_geometry=MemoryGeometry(hidden=hidden, value_dim=2,
                                                      rows=16, reads=2),
        memory_layers=(1,))


def _tokens(batch: int = 2, length: int = 5, vocab: int = 13, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=generator)


# ---------------------------------------------------------------------------
# the zero point and the trainable scope


def test_zero_gate_reproduces_the_frozen_core_exactly():
    core, arm = _arm()
    ids = _tokens()
    reference_core = _TinyCore(hidden=8, layers=2)
    with torch.no_grad():
        base = reference_core(input_ids=ids, use_cache=False).logits
        composed = arm.model(input_ids=ids, use_cache=False).logits
    assert torch.equal(base, composed)
    report = composition.composition_zero_point(arm, base.double(), ids)
    assert report["gate_parameters_zero"] is True
    assert report["max_abs_logit_delta"] == pytest.approx(0.0, abs=1e-6)
    assert report["ok"] is True


def test_only_the_gates_are_trainable_and_they_are_counted_as_a_selector():
    _core, arm = _arm(hidden=8, layers=2)
    expected = 2 * (2 * 8 + 2)
    assert arm.gate_elements() == expected
    assert arm.trainable_elements() == expected
    assert len(arm.trainable_parameters()) == 2 * len(arm.gates)
    for module in arm.dense_modules + arm.memory_modules:
        assert all(not parameter.requires_grad for parameter in module.parameters())
    cost = arm.selector_cost()
    assert cost["learned_parameters"] == 2 * (2 * 8 + 2)
    # the registered geometry's arithmetic is the same formula
    assert composition.composition_selector_cost()["learned_parameters"] == \
        28 * (2 * 1024 + 2)


# ---------------------------------------------------------------------------
# reachability and the budget


def _residuals(hidden: int = 8, tokens: int = 6, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(tokens, hidden, generator=generator) + 1.0
    residual_d = 0.3 * torch.randn(tokens, hidden, generator=generator)
    residual_m = 0.3 * torch.randn(tokens, hidden, generator=generator)
    return base, residual_d, residual_m


def test_the_gate_reaches_d_only_m_only_and_the_combination():
    hidden, tokens = 8, 6
    base, residual_d, residual_m = _residuals(hidden, tokens)
    feature = torch.zeros(tokens, hidden)
    targets = {
        "D": (base + residual_d, (1.0, 0.0)),
        "M": (base + residual_m, (0.0, 1.0)),
        "sum": (base + residual_d + residual_m, (1.0, 1.0)),
    }
    for name, (target, coefficients) in targets.items():
        gate = composition.BoundedGateComposition(hidden, budget=10.0)

        def step(_index, gate=gate, target=target):
            output, _info = gate(base, residual_d, residual_m, feature)
            return torch.nn.functional.mse_loss(output, target), tokens

        report = composition.train_gate([gate], step, steps=400, lr=0.05)
        with torch.no_grad():
            output, info = gate(base, residual_d, residual_m, feature)
        assert report["final_loss"] < 1e-6, (name, report["final_loss"])
        assert float(info["a"].mean()) == pytest.approx(coefficients[0], abs=5e-3)
        assert float(info["b"].mean()) == pytest.approx(coefficients[1], abs=5e-3)
        assert torch.allclose(output, target, atol=1e-2)


def test_the_relative_budget_is_a_fraction_of_each_token_and_is_respected():
    hidden, tokens, kappa = 8, 7, 0.10
    base, residual_d, residual_m = _residuals(hidden, tokens)
    base = base * torch.linspace(0.5, 2.0, tokens).reshape(-1, 1)
    gate = composition.BoundedGateComposition(hidden, relative=kappa)
    with torch.no_grad():
        gate.gate.weight.normal_(std=5.0)
        gate.gate.bias.fill_(3.0)
    feature = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(5))
    _output, info = gate(base, residual_d, residual_m, feature)
    report = composition.check_budget(info)
    assert report["violations"] == 0
    expected = kappa * base.norm(dim=-1)
    assert torch.allclose(info["budget"], expected, rtol=1e-5)
    assert report["budget_utilisation"] <= 1.0 + 1e-6
    # a larger kappa admits a strictly larger perturbation on the same input
    looser = composition.BoundedGateComposition(hidden, relative=2 * kappa)
    with torch.no_grad():
        looser.gate.weight.copy_(gate.gate.weight)
        looser.gate.bias.copy_(gate.gate.bias)
    _o, loose_info = looser(base, residual_d, residual_m, feature)
    assert float(loose_info["perturbation_norm"].max()) > \
        float(info["perturbation_norm"].max())


# ---------------------------------------------------------------------------
# no leakage, determinism, frozen capabilities


def test_the_coefficients_do_not_read_the_residuals():
    hidden, tokens = 8, 4
    base, residual_d, residual_m = _residuals(hidden, tokens)
    gate = composition.BoundedGateComposition(hidden, budget=50.0)
    with torch.no_grad():
        gate.gate.weight.normal_(std=0.5)
        gate.gate.bias.normal_(std=0.5)
    feature = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(9))
    _out_a, info_a = gate(base, residual_d, residual_m, feature)
    _out_b, info_b = gate(base, 7.0 * residual_d, -3.0 * residual_m, feature)
    assert torch.equal(info_a["a"], info_b["a"])
    assert torch.equal(info_a["b"], info_b["b"])


def test_gate_training_is_deterministic_and_touches_only_the_gates():
    hidden, tokens = 8, 6
    base, residual_d, residual_m = _residuals(hidden, tokens)
    feature = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(11))

    def train_once():
        gate = composition.BoundedGateComposition(hidden, budget=50.0)

        def step(_index):
            output, _info = gate(base, residual_d, residual_m, feature)
            return torch.nn.functional.mse_loss(output, base + 0.5 * residual_d), tokens

        composition.train_gate([gate], step, steps=30, lr=0.02, seed=1001)
        return gate.gate.weight.detach().clone(), gate.gate.bias.detach().clone()

    weight_a, bias_a = train_once()
    weight_b, bias_b = train_once()
    assert torch.equal(weight_a, weight_b)
    assert torch.equal(bias_a, bias_b)

    _core, arm = _arm()
    ids = _tokens()
    # The registered D and M checkpoints have non-zero residuals; the tiny arm's
    # adapters start at their zero-init. Give them a non-zero output path so the
    # gate has a gradient, then keep them frozen (as the real arm is).
    with torch.no_grad():
        for module in arm.dense_modules:
            module.down.normal_(std=0.05, generator=torch.Generator().manual_seed(2))
        for module in arm.memory_modules:
            module.proj.weight.normal_(std=0.05,
                                       generator=torch.Generator().manual_seed(3))
    before = {name: parameter.detach().clone()
              for name, parameter in arm.model.named_parameters()
              if not parameter.requires_grad}

    def step(_index, arm=arm, ids=ids):
        logits = arm.model(input_ids=ids, use_cache=False).logits
        # A moving target with a non-zero gradient at the zero gate.
        return torch.nn.functional.mse_loss(logits, 2.0 * logits.detach()), \
            int(ids.numel())

    composition.train_gate(arm.gates, step, steps=5, lr=0.01)
    after = dict(arm.model.named_parameters())
    for name, value in before.items():
        assert torch.equal(value, after[name].detach()), name
    assert float(arm.gates[0].gate.bias.detach().abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# the checkpoint remap


def _fake_checkpoint(arm, tag: str) -> dict:
    prefix = "dense" if tag == "D" else "memory"
    state = {}
    for index, layer in enumerate(arm.model.model.layers):
        module = getattr(layer.mlp, prefix)
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            key = f"model.layers.{index}.mlp.{prefix}.{name}"
            state[key.replace(f".mlp.{prefix}.", ".mlp.extra.")] = \
                parameter.detach().clone()
    return {"step": 7, "model": state}


def test_the_t2_checkpoints_remap_onto_the_composed_arm():
    _core, arm = _arm()
    state_d = _fake_checkpoint(arm, "D")
    state_m = _fake_checkpoint(arm, "M")
    remapped_d = composition._remap_arm_state(state_d, "D", arm)
    remapped_m = composition._remap_arm_state(state_m, "M", arm)
    assert len(remapped_d) == 3 * arm.layers
    assert len(remapped_m) == 3 * len(arm.memory_modules)
    target = dict(arm.model.named_parameters())
    for key, value in remapped_d.items():
        assert target[key].shape == value.shape
        assert torch.equal(target[key].detach(), value)


def test_an_unexpected_checkpoint_key_fails_closed():
    _core, arm = _arm()
    bad = {"model": {"model.layers.0.mlp.oops.weight": torch.zeros(1)}}
    with pytest.raises(ValueError, match="unexpected key"):
        composition._remap_arm_state(bad, "D", arm)


# ---------------------------------------------------------------------------
# the registered comparison


def test_the_comparison_is_paired_and_reports_the_oracle_bound():
    per_item = {
        "D": [2.0, 2.0, 2.0, 2.0],
        "M": [2.0, 2.0, 2.0, 2.0],
        "sum": [2.0, 3.0, 2.0, 3.0],
        "gate": [1.0, 1.0, 1.0, 1.0],
    }
    report = composition.composition_comparison(per_item)
    assert report["items"] == 4
    assert report["mean_loss"]["gate"] == pytest.approx(1.0)
    assert report["comparisons"]["gate-minus-sum"] == {
        "mean": pytest.approx(-1.5), "low": pytest.approx(-2.0),
        "high": pytest.approx(-1.0), "n": 4,
        "resamples": composition.BOOTSTRAP_RESAMPLES,
        "seed": composition.BOOTSTRAP_SEED}
    assert report["oracle"]["mean_loss"] == pytest.approx(1.0)
    assert report["oracle"]["gate_gap"] == pytest.approx(0.0)
    assert report["wins"] == {"D": 0, "M": 0, "sum": 0, "gate": 4}
    assert report["decision_pair"] == "gate-minus-sum"


def test_the_bootstrap_matches_the_p8_1_harness():
    differences = [0.1, -0.2, 0.3, 0.05, -0.4, 0.25]
    ours = composition.paired_bootstrap(differences)
    theirs = complementarity.paired_bootstrap(differences)
    assert ours == theirs


def test_a_missing_arm_is_refused():
    with pytest.raises(ValueError, match="missing arms"):
        composition.composition_comparison({"D": [1.0], "M": [1.0], "gate": [1.0]})
