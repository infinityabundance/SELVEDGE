"""CPU tests for the S44 stage-2b corrective entry points.

The stage-2b registration
(`docs/ws19-stage2b-corrective-advantage-preregistration-v1.json`) places the
three S43 weightings on the math-weighted 20M stream. These tests check the
things a registration can silently get wrong:

* **each weighting runs a training step.** `kl`, `signed` and `both` compute the
  registered objective on a frozen base, produce a finite loss, open the gate,
  and back-propagate into the gate, the rows, the key projection and the value
  projection.
* **the step is deterministic.** The same seed twice gives a bit-identical loss
  and bit-identical gradients — the run is resumable and a re-run is evidence.
* **the active union is the registered number.** 539,649 = 537,600 (memory
  union) + 2*1024 + 1 per site, i.e. 2,049 more than the memory arm.
* **zero init is the base.** The untrained corrective arm reproduces the base
  logits bit for bit even with grafted rows and a non-zero constant offset.
* **the signed advantage has the registered sign.**
* **the entry points refuse mis-wiring.** `--objective correction` without
  `--arm corrective` fails before any GPU work, and the launcher's printed
  command matches the registration.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import arms, corrective, memory  # noqa: E402

REGISTRATION = os.path.join(
    ROOT, "docs", "ws19-stage2b-corrective-advantage-preregistration-v1.json")
LAUNCHER = os.path.join(ROOT, "scripts", "s44_corrective_math.sh")


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


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _module(hidden: int = 32, value_dim: int = 8, rows: int = 64,
            sites=(0,), seed: int = 0) -> corrective.CorrectiveMemory:
    module = corrective.CorrectiveMemory(
        hidden=hidden, value_dim=value_dim, rows=rows, sites=sites,
        conv_kernel=3, table_dtype=torch.float32, seed=seed)
    with torch.no_grad():
        for table in module.tables:
            table.normal_(std=0.05)
    return module


def _step(weighting: str, seed: int = 0, steps: int = 3):
    """Registered training steps on a tiny frozen model, CPU only.

    Mirrors `oracle/frontier/train.py`'s inner loop: student forward with the
    residual installed, base forward under `bypass`, teacher forward under
    no_grad, then `corrective_objective`. Three steps, because the first step has
    a gate of exactly zero by registration: the gate receives gradient on step 1
    and the memory rows, key and value projections only once it has opened.
    """
    torch.manual_seed(seed)
    student = _LM(vocab=48, hidden=32, layers=3)
    _freeze(student)
    teacher = _LM(vocab=48, hidden=32, layers=3)
    torch.manual_seed(seed + 1)
    for parameter in teacher.parameters():
        if parameter.dim() > 1:
            nn.init.normal_(parameter, std=0.1)
    _freeze(teacher)
    module = _module(seed=seed)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    handles = arms.install_memory(student, module)
    try:
        optimizer = torch.optim.AdamW(list(module.parameters()), lr=1e-3,
                                      betas=(0.9, 0.95))
        tokens = torch.randint(0, 48, (1, 16), dtype=torch.int64)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        losses, ces, kls, gates, diagnostics = [], [], [], [], None
        grads: dict[str, torch.Tensor] = {}
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(input_ids=inputs, use_cache=False).logits
            with torch.no_grad():
                module.bypass = True
                base_logits = student(input_ids=inputs, use_cache=False).logits
                module.bypass = False
                teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
            gate = module.last_gate
            assert gate is not None
            loss, ce, kl, diagnostics = corrective.corrective_objective(
                student_logits, base_logits, teacher_logits, targets,
                temperature=2.0, kl_weight=0.5, objective="correction",
                weighting=weighting, gate=gate, preserve_weight=0.5,
                interference_weight=0.05)
            loss.backward()
            losses.append(float(loss.detach()))
            ces.append(float(ce))
            kls.append(float(kl))
            gates.append(gate.detach().clone())
            if step == steps - 1:
                grads = {
                    "gate_w": module.gate_w[0].grad.detach().clone(),
                    "gate_b": module.gate_b[0].grad.detach().clone(),
                    "key": module.key_proj[0].weight.grad.detach().clone(),
                    "value": module.value_proj[0].weight.grad.detach().clone(),
                    "table": module.tables[0].grad.detach().clone(),
                }
            optimizer.step()
        # a forward after the steps: the gate is now a per-token function
        with torch.no_grad():
            module.context.observe(tokens)
            module(torch.randn(1, inputs.shape[1], 32))
            trained_gate = module.last_gate.clone()
        return {
            "losses": losses, "ce": ces, "kl": kls, "gates": gates,
            "diagnostics": diagnostics, "grads": grads, "module": module,
            "trained_gate": trained_gate,
        }
    finally:
        arms.uninstall(handles)


# ---------------------------------------------------------------------------
# accounting


def test_active_union_is_the_registered_539649():
    geometry = memory.MemoryGeometry()
    module = corrective.CorrectiveMemory(
        hidden=geometry.hidden, value_dim=geometry.value_dim, rows=64,
        sites=(2,), conv_kernel=geometry.conv_kernel)
    report = module.active_parameter_report()
    assert report["active_per_token_before_gate"] == 537_600
    assert report["correction_gate_elements"] == 2 * geometry.hidden + 1 == 2_049
    assert report["active_per_token"] == 539_649
    assert report["active_per_token"] - 537_600 == 2_049
    assert report["constant_offset_in_residual"] is False
    assert module.identity_at_init()


# ---------------------------------------------------------------------------
# zero-init parity


def test_untrained_corrective_arm_is_the_base_bit_for_bit():
    torch.manual_seed(21)
    lm = _LM(vocab=64, hidden=32, layers=4)
    _freeze(lm)
    tokens = torch.randint(0, 64, (1, 12), dtype=torch.int64)
    with torch.no_grad():
        base = lm(tokens, use_cache=False).logits
    module = _module()
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
    assert float(gate.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# the three entry points


@pytest.mark.parametrize("weighting", ["kl", "signed", "signed-true", "both"])
def test_each_weighting_runs_registered_training_steps(weighting):
    result = _step(weighting)
    assert all(torch.isfinite(torch.tensor(value)) for value in result["losses"])
    assert all(value > 0.0 for value in result["losses"])
    assert all(value > 0.0 for value in result["ce"])
    diagnostics = result["diagnostics"]
    assert diagnostics["objective"] == "correction"
    assert diagnostics["weighting"] == weighting
    assert diagnostics["weighting_canonical"] == corrective.canonical_weighting(
        weighting)
    if weighting == "signed-true":
        # the S47 sign split: the token-loss protection is active only here
        assert diagnostics["base_token_protection_weight"] == 0.5
        assert diagnostics["base_token_protection_tokens"] >= 0
    else:
        assert diagnostics["base_token_protection_weight"] == 0.0
    # step 1's gate is exactly zero by registration; the gate's own gradient is
    # therefore the only path on that step, which is what makes zero-init safe
    assert float(result["gates"][0].abs().max()) == 0.0
    # by the last step the gate has opened and varies across positions
    assert float(result["gates"][-1].abs().max()) > 0.0
    assert float(result["trained_gate"].std()) > 0.0
    for name, grad in result["grads"].items():
        assert torch.isfinite(grad).all(), name
        assert float(grad.abs().sum()) > 0.0, f"{name} received no gradient"


def test_the_signed_advantage_has_the_registered_sign():
    torch.manual_seed(31)
    vocab, positions = 8, 4
    targets = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, positions, vocab)
    # the teacher prefers the true token at positions 0 and 1, the base at 2 and 3
    teacher = logits.clone()
    teacher[0, 0, 1] = 4.0
    teacher[0, 1, 2] = 2.0
    base = logits.clone()
    base[0, 2, 3] = 4.0
    base[0, 3, 4] = 2.0
    advantage = corrective.signed_teacher_advantage(base, teacher, targets)
    assert advantage.shape == (positions,)
    assert advantage[0] > 0 and advantage[1] > 0
    assert advantage[2] < 0 and advantage[3] < 0
    # the signed weight orders positions by that advantage
    weight, diagnostics = corrective._weigh(
        torch.zeros_like(advantage), advantage, "signed")
    assert weight[0] > weight[2]
    assert diagnostics["signed_positive_fraction"] == 0.5


def test_steps_are_deterministic_for_each_weighting():
    for weighting in ("kl", "signed", "signed-true", "both"):
        first = _step(weighting, seed=5)
        second = _step(weighting, seed=5)
        assert first["losses"] == second["losses"], weighting
        assert first["ce"] == second["ce"]
        assert first["kl"] == second["kl"]
        for name in first["grads"]:
            assert torch.equal(first["grads"][name], second["grads"][name]), \
                f"{weighting}:{name}"
        for left, right in zip(first["gates"], second["gates"]):
            assert torch.equal(left, right), weighting


# ---------------------------------------------------------------------------
# entry-point wiring


def test_the_correction_objective_is_refused_on_a_non_corrective_arm():
    from frontier import train

    argv = [
        "--arm", "memory", "--objective", "correction",
        "--stream-manifest", "does-not-matter.json",
        "--memory-init", "does-not-matter.pt",
        "--run-dir", "/tmp/never-used",
    ]
    with pytest.raises(SystemExit) as excinfo:
        train.main(argv)
    assert "requires --arm corrective" in str(excinfo.value)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
@pytest.mark.parametrize("weighting", ["kl", "signed", "signed-true", "both"])
def test_the_launcher_matches_the_registration(weighting):
    registration = json.load(open(REGISTRATION, encoding="utf-8"))
    arm = f"corrective-{weighting}-20m-math"
    registered = registration["arms"][arm]
    assert registered["weighting"] == weighting

    proc = subprocess.run(
        ["bash", LAUNCHER, "--weighting", weighting, "--dry-run"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    command = proc.stdout
    assert f"--advantage-weighting {registered['weighting']}" in command
    assert f"--stream-manifest {registration['stream']['manifest']}" in command
    assert registration["stream"]["sha256"] in command
    assert f"--tokens {registration['common']['tokens']}" in command
    assert f"--seq-len {registration['common']['sequence_length']}" in command
    assert f"--grad-accum {registration['common']['gradient_accumulation']}" in command
    assert f"--lr {registration['common']['lr']}" in command
    assert f"--warmup {registration['common']['warmup_steps']}" in command
    assert f"--seed {registration['common']['seed']}" in command
    assert f"--run-dir {registered['run_dir']}" in command
    assert "--objective correction" in command
    assert "--arm corrective" in command
    assert "--vram-limit-gib 12.5" in command


def test_the_scoring_entry_point_covers_every_registered_arm():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s40_compare", os.path.join(ROOT, "scripts", "s40_compare.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registration = json.load(open(REGISTRATION, encoding="utf-8"))
    for arm in registration["arms"]:
        assert arm in module.SCORED_ARMS, arm
