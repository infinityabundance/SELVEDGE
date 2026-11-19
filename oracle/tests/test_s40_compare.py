"""CPU tests for the S40 comparison instrument (held-out CE/KL and the table)."""

from __future__ import annotations

import math
import os
import sys

import torch
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from frontier import compare, fake, train  # noqa: E402


def test_arm_selection_requires_the_reference_to_be_selected():
    import s40_compare

    class Args:
        pass

    args = Args()
    args.arms = "base,adapter-2m"
    args.reference_arm = "base"
    assert s40_compare.selected_arms(args) == ["base", "adapter-2m"]
    args.reference_arm = "full-distill"
    with pytest.raises(SystemExit):
        s40_compare.selected_arms(args)
    args.arms = "base,not-an-arm"
    with pytest.raises(SystemExit):
        s40_compare.selected_arms(args)


def test_interpolated_arm_is_its_own_label_and_needs_a_coefficient(tmp_path):
    import s40_compare

    assert "interpolated" in s40_compare.SCORED_ARMS
    checkpoint = tmp_path / "interp.pt"
    torch.save({"arm": "full", "arm_state": {}}, checkpoint)

    class Args:
        full_checkpoint = str(checkpoint)

    with pytest.raises(SystemExit) as exc:
        s40_compare._build_named_arm("interpolated", None, Args(), "cpu")
    assert "interpolation block" in str(exc.value)


def test_limit_map_is_per_task_and_refuses_unknown_names():
    import s40_score

    declared = {"mmlu": {}, "mmlu_pro": {}, "gsm8k": {}}
    tasks = ["mmlu_pro", "mmlu", "gsm8k"]
    assert s40_score.parse_limit_map(None, declared, tasks) == {}
    assert s40_score.parse_limit_map("mmlu=2000", declared, tasks) == {"mmlu": 2000}
    assert s40_score.parse_limit_map("mmlu=2000,gsm8k=1319", declared, tasks) == {
        "mmlu": 2000, "gsm8k": 1319}
    for bad, selected in (("arc=1", tasks), ("mmlu=2000", ["gsm8k"]), ("mmlu=x", tasks)):
        with pytest.raises(SystemExit):
            s40_score.parse_limit_map(bad, declared, selected)


def test_chunk_ce_kl_matches_the_training_loss_element_for_element():
    torch.manual_seed(0)
    vocab, length = 32, 7
    student = torch.randn(1, length, vocab)
    teacher = torch.randn(1, length, vocab)
    targets = torch.randint(0, vocab, (1, length))
    temperature, weight = 2.0, 0.5
    ce, kl, positions = compare.chunk_ce_kl(
        student, targets, teacher, temperature, vocab_chunk=3)
    loss, expected_ce, expected_kl = train.token_loss(
        student, teacher, targets, temperature, weight, vocab_chunk=3)
    assert positions == length
    assert math.isclose(ce, float(expected_ce), rel_tol=1e-6)
    assert math.isclose(kl, float(expected_kl), rel_tol=1e-6)
    assert math.isclose(loss, ce + weight * temperature ** 2 * kl, rel_tol=1e-6)


def test_summary_reports_the_sample_standard_error():
    summary = compare.summary([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert math.isclose(summary["mean"], 2.5)
    assert math.isclose(summary["sd"], math.sqrt(5.0 / 3.0))
    assert math.isclose(summary["stderr"], math.sqrt(5.0 / 3.0) / 2.0)
    assert summary["ci95"][0] < summary["mean"] < summary["ci95"][1]
    assert compare.summary([])["mean"] is None


def test_paired_delta_is_the_difference_of_the_same_chunks():
    arm = [2.0, 3.0, 4.0]
    reference = [1.0, 1.0, 1.0]
    delta = compare.paired_delta(arm, reference)
    assert math.isclose(delta["mean"], 2.0)
    assert delta["n"] == 3


def test_format_table_prints_every_arm_with_standard_errors():
    rows = [{"arm": "base", "ce": {"mean": 1.0, "stderr": 0.1},
             "kl": {"mean": 0.2, "stderr": 0.01}, "delta_ce_vs_reference": None},
            {"arm": "full-distill", "ce": {"mean": 0.9, "stderr": 0.2},
             "kl": {"mean": 0.1, "stderr": 0.02},
             "delta_ce_vs_reference": {"mean": 0.0, "stderr": 0.0}}]
    text = compare.format_table(rows, "full-distill")
    assert "base" in text and "full-distill" in text
    assert "0.10000" in text


def test_score_stream_scores_non_overlapping_chunks():
    torch.manual_seed(1)
    vocab, hidden, layers = 32, 16, 2
    student = fake.TinyLM(vocab, hidden, layers)
    teacher = fake.TinyLM(vocab, hidden, layers)
    tokens = torch.randint(0, vocab, (40,))
    scored = compare.score_stream(student, teacher, tokens, seq_len=8, device="cpu",
                                  temperature=2.0, vocab_chunk=4)
    assert scored["chunks"] == 4          # 40 // 9 leaves the partial tail out
    assert scored["tokens_scored"] == 4 * 8
    for row in scored["rows"]:
        assert math.isfinite(row["ce"]) and math.isfinite(row["kl"])


def test_ungrafted_module_is_exactly_the_frozen_base():
    torch.manual_seed(2)
    model = fake.TinyLM(32, 16, 2)
    context, module = fake_memory()
    tokens = torch.randint(0, 32, (20,))
    context.observe(tokens[None, :])
    hidden = torch.randn(1, 20, 16)
    output = module(hidden)
    assert torch.equal(output, torch.zeros_like(output))


def fake_memory():
    from frontier import memory
    context = memory.TokenContext()
    module = memory.ConditionalMemory(hidden=16, value_dim=8, rows=32, sites=(0,),
                                      conv_kernel=3, context=context,
                                      table_dtype=torch.float32)
    return context, module
