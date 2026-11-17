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
