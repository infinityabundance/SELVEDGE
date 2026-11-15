"""CPU tests for the T1 corpus, accounting, routing, loss and metric pieces.

These run in the oracle court without a GPU. They cover the parts that a
full-run receipt cannot prove by itself: the declared union arithmetic, the
licence and contamination helpers, the packed-target round trip, the routing
wrappers' zero-point behaviour, the exact top-k/tail loss and the bootstrap.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

import arms  # noqa: E402
import corpus  # noqa: E402
import loop  # noqa: E402
import metrics  # noqa: E402
import modeling  # noqa: E402
import teacher_cache  # noqa: E402


# ---------------------------------------------------------------------------
# declared accounting


def test_python_mirror_unions_overlapping_and_adjacent_ranges():
    config = {
        "selector_uses": [
            {"identity": {"tensor": "router", "elements": {"start": 0, "end": 100}}},
        ],
        "execution_uses": [
            {"identity": {"tensor": "ffn", "elements": {"start": 0, "end": 100}}},
            {"identity": {"tensor": "ffn", "elements": {"start": 100, "end": 200}}},
            {"identity": {"tensor": "ffn", "elements": {"start": 50, "end": 150}}},
            {"identity": {"tensor": "norm", "elements": {"start": 0, "end": 10}}},
        ],
    }
    mirror = arms.python_mirror(config)
    # Union: router 100 + ffn [0,200) 200 + norm 10 = 310. The three ffn
    # execution uses overlap, so their sum (300) is not the union.
    assert mirror["active_unique_params"] == 310
    assert mirror["selector_elements"] == 100
    assert mirror["execution_elements"] == 310
    assert mirror["identities"] == 3


def test_python_mirror_reports_union_below_sum_for_one_identity():
    config = {
        "selector_uses": [
            {"identity": {"tensor": "router", "elements": {"start": 0, "end": 40}}},
        ],
        "execution_uses": [
            {"identity": {"tensor": "ffn", "elements": {"start": 0, "end": 100}}},
            {"identity": {"tensor": "ffn", "elements": {"start": 0, "end": 100}}},
            {"identity": {"tensor": "ffn", "elements": {"start": 0, "end": 100}}},
        ],
    }
    mirror = arms.python_mirror(config)
    # Three identical reads of the same 100 elements are one unique identity.
    assert mirror["execution_elements"] == 300
    assert mirror["active_unique_params"] == 140
    assert mirror["identities"] == 2


def test_every_arm_config_matches_its_design_count():
    for name in arms.ARMS:
        config = arms.build_config(arms.ARMS[name])
        mirror = arms.python_mirror(config)
        assert mirror["active_unique_params"] == arms.DESIGN_ACTIVE[name], name
        assert config["total_accessible"] == arms.DESIGN_TOTAL_ACCESSIBLE[name], name
        assert mirror["active_unique_params"] <= arms.CEILING, name


def test_routed_arm_declares_k_slots_not_all_groups():
    config = arms.build_config(arms.ARMS["A"])
    slots = [use for use in config["execution_uses"] if ".slot" in use["identity"]["tensor"]]
    # 28 layers x 5 slots x (3 group projections + 1 adapter)
    assert len(slots) == 28 * 5 * 4
    assert config["execution_flops"]["matmul_flops"] > 0


# ---------------------------------------------------------------------------
# corpus helpers


def test_normalize_and_shingles_are_stable():
    assert corpus.normalize("  a\t b \n c ") == "a b c"
    bag = corpus.shingles(("a", "b", "c", "d", "e", "f"))
    assert "a b c d e" in bag and "b c d e f" in bag
    assert len(corpus.shingles(("short", "doc"))) == 1


def test_row_licence_allowlist():
    source = {"row_license_field": "license",
              "permissive_allowlist": ["mit", "apache-2.0"]}
    assert corpus.row_license_allowed(source, {"license": "MIT"})
    assert corpus.row_license_allowed(source, {"license": " apache-2.0 "})
    assert not corpus.row_license_allowed(source, {"license": "unknown"})
    assert not corpus.row_license_allowed(source, {"license": None})
    assert corpus.row_license_allowed({"row_license_field": None}, {"license": "nope"})


def test_document_text_template():
    source = {"text_fields": ["problem", "solution"],
              "text_template": "Problem: {problem}\nSolution: {solution}"}
    assert corpus.document_text(source, {"problem": "1+1", "solution": "2"}) == (
        "Problem: 1+1\nSolution: 2")
    assert corpus.document_text(source, {"problem": None, "solution": "2"}) is None


# ---------------------------------------------------------------------------
# packed teacher targets


def test_pack_unpack_round_trip_is_exact():
    generator = np.random.default_rng(7)
    values = generator.integers(0, 65535, size=(4, 3, 16), dtype=np.uint16)
    indices = generator.integers(0, 151936, size=(4, 3, 16), dtype=np.int32)
    logsumexp = generator.normal(size=(4, 3)).astype(np.float32)
    packed = teacher_cache.pack(values, indices, logsumexp)
    assert len(packed) == 4 * 3 * teacher_cache.RECORD_BYTES
    got_values, got_indices, got_lse = teacher_cache.unpack(np.frombuffer(packed,
                                                                        dtype=np.uint8))
    assert np.array_equal(got_indices, indices.reshape(-1, 16))
    assert np.array_equal(got_lse, logsumexp.reshape(-1))
    expected = teacher_cache.bf16_to_float32(values.reshape(-1, 16))
    # Random bf16 bit patterns include NaNs, and NaN != NaN under value
    # equality, so the round trip is asserted on the bit patterns. The packer
    # stores logits as raw bf16 bits, so bit equality is the strictest form.
    assert np.array_equal(got_values.view(np.uint32), expected.view(np.uint32))


def test_bf16_widening_is_exact():
    bits = np.array([0x3F80, 0x4000, 0x0000, 0xBF80], dtype=np.uint16)
    widened = teacher_cache.bf16_to_float32(bits)
    assert widened.tolist() == [1.0, 2.0, 0.0, -1.0]


# ---------------------------------------------------------------------------
# routing wrappers


class _TinyMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_capacity_is_ceil_with_factor():
    assert modeling._capacity(1024, 4, 8) == math.ceil(1024 * 4 / 8 * 1.25)
    assert modeling._capacity(1, 1, 1) >= 1


def test_routed_group_all_groups_matches_dense_in_f32():
    torch.manual_seed(0)
    mlp = _TinyMlp(16, 32)
    router = modeling._router(16, 4, seed=1)
    wrapper = modeling.RoutedGroupFFN(mlp, groups=4, k=4, adapter_rank=2, router=router)
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        reference = mlp(x)
        wrapper.k = 4
        wrapper.rank = 0
        # Direct dispatch at k=groups, compared in f32.
        gate = wrapper.gate_groups.float().reshape(-1, 16)
        up = wrapper.up_groups.float().reshape(-1, 16)
        down = torch.cat([wrapper.down_groups[index].float().t() for index in range(4)],
                         dim=1)
        flat = x.reshape(-1, 16).float()
        activation = F.silu(flat @ gate.t()) * (flat @ up.t())
        expected = activation @ down.t()
        buffer = flat.unsqueeze(0).expand(4, 8, 16).contiguous()
        got = wrapper._group_ffn(buffer, cast=torch.float32).sum(dim=0)
    assert torch.allclose(got, expected, atol=1e-5)
    assert reference.shape == x.shape


def test_specialist_bank_is_exactly_zero_at_init():
    bank = modeling.SpecialistBank(hidden=8, width=4, specialists=4, k=2,
                                   router=modeling._router(8, 4, seed=3))
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out = bank(x)
    assert torch.equal(out, torch.zeros_like(out))


def test_dense_adapter_is_exactly_zero_at_init():
    adapter = modeling.DenseAdapter(hidden=8, width=4)
    x = torch.randn(3, 8)
    with torch.no_grad():
        assert torch.equal(adapter(x), torch.zeros_like(x))


def test_adapter_zero_report_is_strict():
    model = torch.nn.Sequential(modeling.DenseAdapter(8, 4))
    fake = modeling.ArmModel(model=model, arm="C2", kind="dense_adapter",
                             router_modules=[], extra_modules=[],
                             admission={}, config={})
    report = modeling.adapter_zero_report(fake)
    assert report["ok"] and report["max_abs"] == 0.0
    assert report["modules"] == 1 and report["tolerance"] == 0.0
    # Leaf Parameters require grad, so the perturbation needs its own no_grad.
    with torch.no_grad():
        model[0].down[0, 0] = 1e-12
    assert not modeling.adapter_zero_report(fake)["ok"]


# ---------------------------------------------------------------------------
# loss
#
# The registered objective is T1.4: top-k support cross-entropy with ONE tail
# bucket. It is not the full-vocabulary cross-entropy, because the compact
# teacher cache stores only the top-16 logits and the tail MASS, never the
# teacher's tail distribution. An earlier version of these tests asserted
# equality with the full-vocabulary cross-entropy and failed (KL 3.4980 vs
# 4.0765 on the fixture). That expectation was wrong: it contradicted the
# registered target. The tests below now pin the registered objective, and
# `test_distillation_loss_equals_full_cross_entropy_at_full_support` proves
# that the implementation is the exact support term when the support is the
# whole vocabulary. See docs/TRAINING.md.


def _bruteforce_topk_tail(logits: torch.Tensor, teacher: torch.Tensor,
                          ids: torch.Tensor) -> torch.Tensor:
    """Independent float64 transcription of the T1.4 objective."""
    logits = logits.double()
    teacher = teacher.double()
    student_logp = torch.log_softmax(logits, dim=-1)
    teacher_p = torch.softmax(teacher, dim=-1)
    support_teacher = teacher_p.gather(1, ids)
    tail_teacher = 1.0 - support_teacher.sum(-1)
    support_student = student_logp.gather(1, ids)
    tail_student = (1.0 - support_student.exp().sum(-1)).clamp_min(1e-7)
    return -((support_teacher * support_student).sum(-1)
             + tail_teacher * tail_student.log()).mean()


def test_distillation_loss_matches_registered_topk_tail_objective():
    torch.manual_seed(5)
    vocab, top_k, positions = 32, 16, 4
    logits = torch.randn(positions, vocab)
    teacher = torch.randn(positions, vocab)
    values, ids = torch.topk(teacher, top_k, dim=-1)
    lse = torch.logsumexp(teacher, dim=-1)
    got = loop.distillation_loss(logits, values, ids, lse)
    expected = _bruteforce_topk_tail(logits, teacher, ids)
    assert torch.allclose(got.double(), expected, atol=1e-9)


def test_distillation_loss_is_invariant_to_support_order():
    torch.manual_seed(11)
    vocab, top_k, positions = 48, 16, 6
    logits = torch.randn(positions, vocab)
    teacher = torch.randn(positions, vocab)
    values, ids = torch.topk(teacher, top_k, dim=-1)
    lse = torch.logsumexp(teacher, dim=-1)
    permutation = torch.randperm(top_k)
    shuffled = loop.distillation_loss(logits, values[:, permutation], ids[:, permutation], lse)
    plain = loop.distillation_loss(logits, values, ids, lse)
    assert torch.allclose(shuffled, plain, atol=1e-9)


def test_distillation_loss_equals_full_cross_entropy_at_full_support():
    """With the whole vocabulary as support the tail is empty and the
    registered objective is the ordinary cross-entropy. This is the check that
    the support term is exact rather than merely self-consistent."""
    torch.manual_seed(7)
    vocab, positions = 64, 5
    logits = torch.randn(positions, vocab, requires_grad=True)
    teacher = torch.randn(positions, vocab)
    values, ids = torch.topk(teacher, vocab, dim=-1)
    lse = torch.logsumexp(teacher, dim=-1)
    got = loop.distillation_loss(logits, values, ids, lse)
    expected = -(torch.softmax(teacher, dim=-1)
                 * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
    assert torch.allclose(got, expected, atol=1e-9)
    got.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_distillation_loss_is_finite_for_concentrated_support():
    logits = torch.full((2, 32), -50.0)
    logits[:, 0] = 50.0
    teacher = torch.full((2, 32), -50.0)
    teacher[:, 0] = 50.0
    values, ids = torch.topk(teacher, 16, dim=-1)
    lse = torch.logsumexp(teacher, dim=-1)
    loss = loop.distillation_loss(logits, values, ids, lse)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# metrics


def test_paired_bootstrap_sign_and_ci():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a - 0.5
    result = metrics.paired_bootstrap(a, b, resamples=500)
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["ci_low"] <= result["mean_difference"] <= result["ci_high"]


def test_stratified_bootstrap_weights_domains_equally():
    per_domain = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([1.0])}
    result = metrics.stratified_bootstrap(per_domain, resamples=200)
    assert result["mean"] == pytest.approx(0.5)
