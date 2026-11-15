"""CPU tests for the T2 capacity modules.

Covers the hashed memory (determinism, causality, bounds, the exact active
accounting and the zero-init cold start), the deterministic slot-wise combine,
the router-gradient treatment that distinguishes H from H*, the absence of a
load-balance term in H*, and the agreement between the declared Rust config and
the modules' real element counts.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

import arms  # noqa: E402
import loop  # noqa: E402
import memory  # noqa: E402
import modeling  # noqa: E402

ROWS = 1 << 18


# ---------------------------------------------------------------------------
# hashing


def _tokens(seed: int = 7, length: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 100_000, (1, length), generator=generator, dtype=torch.int64)


def test_hash_index_is_deterministic_and_seed_independent():
    tokens = _tokens()
    first = memory.ngram_index(tokens, layer=3, order=2, rows=ROWS)
    torch.manual_seed(999)
    second = memory.ngram_index(tokens, layer=3, order=2, rows=ROWS)
    assert torch.equal(first, second)
    # A different layer or order addresses differently.
    assert not torch.equal(first, memory.ngram_index(tokens, layer=4, order=2, rows=ROWS))
    assert not torch.equal(first, memory.ngram_index(tokens, layer=3, order=3, rows=ROWS))


def test_hash_index_is_causal():
    tokens = _tokens()
    order = 3
    base = memory.ngram_index(tokens, layer=10, order=order, rows=ROWS)
    moved_at_five = False
    for delta in range(1, 17):
        changed = tokens.clone()
        changed[0, 5] = (changed[0, 5] + delta) % 100_000
        other = memory.ngram_index(changed, layer=10, order=order, rows=ROWS)
        # Position t depends on tokens[t-order+1 .. t] only: every position
        # before the edited token is bitwise unchanged.
        assert torch.equal(base[0, :5], other[0, :5])
        moved_at_five = moved_at_five or not torch.equal(base[0, 5], other[0, 5])
    # A 2^-18 collision is possible for a single perturbation, so at least one
    # of 16 perturbations must move the address at the edited position.
    assert moved_at_five
    # A token after the decision point cannot move it.
    far = tokens.clone()
    far[0, 11] = (far[0, 11] + 1) % 100_000
    far_index = memory.ngram_index(far, layer=10, order=order, rows=ROWS)
    assert torch.equal(base[0, :11], far_index[0, :11])


def test_hash_index_is_bounded_and_handles_short_sequences():
    tokens = torch.tensor([[5]], dtype=torch.int64)
    for order in (2, 3):
        index = memory.ngram_index(tokens, layer=3, order=order, rows=ROWS)
        assert index.shape == (1, 1)
        assert 0 <= int(index.min()) and int(index.max()) < ROWS
    with pytest.raises(ValueError):
        memory.ngram_index(tokens, layer=3, order=2, rows=1000)  # not a power of two


# ---------------------------------------------------------------------------
# active-parameter accounting


def test_memory_active_accounting_matches_the_contract():
    geometry = memory.MemoryGeometry()
    report = geometry.active_report(insertions=4)
    assert report["value_elements_per_layer"] == 512
    assert report["projection_elements_per_layer"] == 524_288
    assert report["gate_elements_per_layer"] == 1_024
    assert report["index_elements_per_layer"] == 0
    assert report["active_per_layer"] == 525_824
    assert report["active_total"] == 2_103_296
    assert report["table_elements_per_layer"] == 67_108_864
    assert report["table_elements_total"] == 268_435_456
    assert report["total_accessible_added"] == 270_536_704
    assert report["rows_total"] == 4 * ROWS


def test_memory_bytes_are_reported_not_confused():
    geometry = memory.MemoryGeometry()
    report = geometry.bytes_report(insertions=4)
    assert report["resident_table_bytes_fp32_equivalent"] == 1_073_741_824
    assert report["resident_table_bytes_deployment"] == 536_870_912
    assert report["read_bytes_per_token"] == 4_206_592
    assert report["read_bytes_per_token_basis"] == "modelled"


def test_memory_config_declares_two_distinct_rows_per_layer():
    config = arms.build_config(arms.T2_ARMS["H"])
    rows = [use for use in config["execution_uses"]
            if ".memory.table.row.slot" in use["identity"]["tensor"]]
    assert len(rows) == 4 * 2
    assert all(use["identity"]["elements"]["end"]
               - use["identity"]["elements"]["start"] == 256 for use in rows)
    # The memory arm has no learned selector: the address is arithmetic.
    config_m = arms.build_config(arms.T2_ARMS["M"])
    assert config_m["selector_uses"] == []


# ---------------------------------------------------------------------------
# memory module


def _memory_module(layer: int = 3, seed: int = 11) -> memory.HashedMemory:
    context = memory.TokenContext()
    return memory.HashedMemory(1024, layer, context, seed=seed)


def test_memory_is_exactly_zero_at_init_and_cold_starts_the_projection():
    module = _memory_module()
    context = memory.TokenContext()
    module.context = context
    hidden = torch.randn(1, 4, 1024)
    context.set(torch.randint(1, 1000, (1, 4), dtype=torch.int64))
    output = module(hidden)
    assert torch.equal(output, torch.zeros_like(output))
    # A non-zero target gives a non-zero dL/d(output), which is what a real
    # residual loss supplies. The projection then receives gradient.
    ((output - 1.0) ** 2).mean().backward()
    assert module.proj.weight.grad is not None
    assert float(module.proj.weight.grad.abs().sum()) > 0
    # At step 0 the values receive no gradient because the projection is zero;
    # this is the same cold start as the specialists' zero W_down, documented.
    assert module.table.grad is None or float(module.table.grad.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# deterministic combine and routing


def _bank(hidden: int = 8, width: int = 4, experts: int = 4, k: int = 2) -> modeling.SpecialistBank:
    bank = modeling.SpecialistBank(hidden, width, experts, k,
                                   modeling._router(hidden, experts, seed=5))
    with torch.no_grad():
        bank.down.normal_(generator=torch.Generator().manual_seed(3))
    return bank


def test_combine_is_bitwise_repeatable():
    bank = _bank()
    x = torch.randn(1, 7, 8)
    first = bank(x)
    second = bank(x)
    assert torch.equal(first, second)
    assert float(first.detach().abs().sum()) > 0


def test_full_bank_exclusion_is_the_expert_contribution():
    bank = _bank()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        bank.full_bank = True
        bank.exclude = None
        full = bank(x)
        bank.exclude = 0
        without = bank(x)
        bank.full_bank = False
        bank.exclude = None
        flat = x.reshape(-1, 8)
        only0 = F.linear(F.silu(F.linear(flat, bank.gate[0]))
                         * F.linear(flat, bank.up[0]),
                         bank.down[0]).reshape(x.shape)
    assert torch.allclose(full - without, only0, atol=1e-6)


def test_h_star_cuts_the_distillation_gradient_to_the_router():
    routed = _bank()
    utility_trained = _bank()
    utility_trained.router_grad = False
    x = torch.randn(1, 6, 8)
    routed(x).pow(2).sum().backward()
    utility_trained(x).pow(2).sum().backward()
    assert routed.router.weight.grad is not None
    assert float(routed.router.weight.grad.abs().sum()) > 0
    assert utility_trained.router.weight.grad is None
    # The experts still receive the distillation gradient in both arms.
    for bank in (routed, utility_trained):
        assert bank.gate.grad is not None and bank.down.grad is not None


class _Holder(torch.nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.bank = bank


def test_load_balance_is_absent_from_an_arm_without_banks():
    fake = modeling.ArmModel(model=torch.nn.Sequential(), arm="M", kind="memory",
                             router_modules=[], extra_modules=[], admission={},
                             config={}, profile="t2")
    assert loop.load_balance_loss(fake) is None


def test_load_balance_is_finite_and_non_negative():
    bank = _bank(experts=4, k=2)
    holder = _Holder(bank)
    fake = modeling.ArmModel(model=holder, arm="H", kind="hybrid",
                             router_modules=[bank.router], extra_modules=[bank],
                             admission={}, config={}, profile="t2")
    bank(torch.randn(1, 9, 8))
    value = loop.load_balance_loss(fake)
    assert value is not None and torch.isfinite(value.detach()) and float(value.detach()) >= 0.0


# ---------------------------------------------------------------------------
# declaration versus real modules


def test_t2_arm_configs_match_the_registered_design_counts():
    for name in arms.T2_ARMS:
        config = arms.build_config(arms.T2_ARMS[name])
        mirror = arms.python_mirror(config)
        assert mirror["active_unique_params"] == arms.DESIGN_ACTIVE_T2[name], name
        assert config["total_accessible"] == arms.DESIGN_TOTAL_T2[name], name
        assert mirror["active_unique_params"] <= arms.CEILING, name


def test_t2_arm_ledger_matches_the_admission_config():
    """The real modules' element counts equal the declared slices."""
    hidden, layers, width, experts, k = 1024, 28, 512, 8, 4
    router = modeling._router(hidden, experts, seed=1)
    bank = modeling.SpecialistBank(hidden, width, experts, k, router)
    assert bank.gate.numel() == experts * width * hidden
    assert bank.up.numel() == experts * width * hidden
    assert bank.down.numel() == experts * hidden * width
    assert (bank.gate.numel() + bank.up.numel() + bank.down.numel()
            == experts * 3 * width * hidden)
    assert router.weight.numel() == hidden * experts

    context = memory.TokenContext()
    module = memory.HashedMemory(hidden, 3, context)
    assert module.table.numel() == ROWS * 256
    assert module.proj.weight.numel() == 512 * hidden
    assert module.gate.weight.numel() == hidden

    for name in ("X", "H", "H*"):
        config = arms.build_config(arms.T2_ARMS[name])
        per_layer = 3 * width * hidden
        slots = [use for use in config["execution_uses"]
                 if ".specialists." in use["identity"]["tensor"]]
        assert sum(use["identity"]["elements"]["end"]
                   - use["identity"]["elements"]["start"] for use in slots) == \
            layers * k * per_layer, name
        assert sum(use["reads"] for use in config["selector_uses"]) == \
            layers * hidden * experts, name
    for name in ("M", "H", "H*"):
        config = arms.build_config(arms.T2_ARMS[name])
        projection = [use for use in config["execution_uses"]
                      if use["identity"]["tensor"].endswith(".memory.proj.weight")]
        assert sum(use["reads"] for use in projection) == 4 * 512 * hidden, name
        gate = [use for use in config["execution_uses"]
                if use["identity"]["tensor"].endswith(".memory.gate.weight")]
        assert sum(use["reads"] for use in gate) == 4 * hidden, name


def test_dense_control_is_parameter_matched_to_expert_only_active():
    dense = arms.python_mirror(arms.build_config(arms.T2_ARMS["D"]))["active_unique_params"]
    expert = arms.python_mirror(arms.build_config(arms.T2_ARMS["X"]))["active_unique_params"]
    # Only the 229,376-element router separates them, and D is cheaper.
    assert expert - dense == 229_376


# ---------------------------------------------------------------------------
# P2.4 instrument: the compact target against a brute-force full target


def test_instrument_targets_match_a_float64_full_vocabulary_reference(monkeypatch):
    import instrument_check

    monkeypatch.setattr(instrument_check, "CHUNK", 7)  # exercise chunk boundaries
    torch.manual_seed(23)
    positions, vocab, top_k = 6, 96, 16
    teacher = torch.randn(positions, vocab)
    student = torch.randn(positions, vocab)
    got = instrument_check.window_targets(student, teacher)

    teacher64 = teacher.double()
    student64 = student.double()
    log_p_t = torch.log_softmax(teacher64, dim=-1)
    log_p_s = torch.log_softmax(student64, dim=-1)
    p_t = log_p_t.exp()
    values, ids = torch.topk(teacher64, top_k, dim=-1)
    support = torch.zeros_like(p_t, dtype=torch.bool)
    support.scatter_(1, ids, True)
    tail_mask = ~support
    p_tail = (p_t * tail_mask).sum(-1)
    q_support = log_p_s.exp().gather(1, ids).sum(-1)
    q_tail = 1.0 - q_support
    compact = -(p_t.gather(1, ids) * log_p_s.gather(1, ids)).sum(-1) \
        - p_tail * q_tail.log()
    full = -(p_t * log_p_s).sum(-1)
    r = (p_t * tail_mask) / p_tail.unsqueeze(-1)
    s = (log_p_s.exp() * tail_mask) / q_tail.unsqueeze(-1)
    entropy = -(r * r.clamp_min(1e-300).log()).sum(-1)
    kl = (r * (r.clamp_min(1e-300).log()
               - s.clamp_min(1e-300).log())).sum(-1)
    assert torch.allclose(got["compact"].double(), compact, atol=1e-9)
    assert torch.allclose(got["full"].double(), full, atol=1e-9)
    assert torch.allclose(got["bias"].double(), compact - full, atol=1e-9)
    assert torch.allclose(got["bias"].double(), -p_tail * (entropy + kl), atol=1e-7)
    assert torch.allclose(got["tail_entropy"].double(), entropy, atol=1e-7)
    assert torch.allclose(got["tail_kl"].double(), kl, atol=1e-6)
