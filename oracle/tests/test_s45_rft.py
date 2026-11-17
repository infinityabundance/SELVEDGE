"""CPU tests for the S45 RFT/on-policy route.

The next-route registration depends on things that can be tested without a GPU:

* the verifier's exactness and its refusal to resolve ambiguity;
* the extractor on the LaTeX and numeric forms the two training sets actually
  contain;
* rejection sampling, deduplication and acceptance arithmetic on the fixture;
* the n-gram contamination check failing closed;
* the target-side math-format curriculum, including that a formatted solution
  still verifies;
* the objective assembly: plain SFT CE, the registered CE+KL control, the
  registered preservation/signed-advantage correction (reused from S42/S43),
  and the weighted combination of an SFT and a KD batch.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import corrective, rft, rft_objective, sft_data  # noqa: E402
from frontier import train as train_module  # noqa: E402
from types import SimpleNamespace  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "s45", "rft-samples.jsonl")


@pytest.mark.parametrize(
    "solution,target,dataset,expected",
    [
        ("work\n#### 12\n", "12", "gsm8k", "correct"),
        ("work\n#### 13\n", "12", "gsm8k", "wrong"),
        ("#### 1,000", "1000", "gsm8k", "correct"),
        ("#### 12 and #### 13", "12", "gsm8k", "ambiguous"),
        ("#### 24\n#### 25\n", "24", "gsm8k", "ambiguous"),
        ("I do not know", "12", "gsm8k", "unparseable"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", "math", "correct"),
        ("\\boxed{\\dfrac{3}{4}}", "\\frac{3}{4}", "math", "correct"),
        ("\\boxed{0.5}", "\\frac{1}{2}", "math", "correct"),
        ("\\boxed{1}\\text{ or }\\boxed{2}", "1", "math", "ambiguous"),
        ("\\boxed{\\sqrt{2}}", "\\sqrt2", "math", "correct"),
        ("\\boxed{3}", "4", "math", "wrong"),
        ("no answer here", "4", "math", "unparseable"),
    ],
)
def test_verifier(solution, target, dataset, expected):
    assert rft.verify(solution, target, dataset)["verdict"] == expected


def test_extractor_reports_ambiguity_and_method():
    result = rft.extract("#### 1 and #### 2", "gsm8k")
    assert result.ambiguous
    assert result.value is None
    assert len(result.candidates) == 2
    single = rft.extract("\\boxed{\\frac{2}{4}}", "math")
    assert not single.ambiguous
    assert single.value == "\\frac{2}{4}"
    assert "boxed" in single.method


@pytest.mark.parametrize("mode", ["none", "prefix", "steps", "prefix+steps"])
def test_curriculum_keeps_the_answer_verifiable(mode):
    solution = "First compute 3+4.\nThen report the total.\n#### 7"
    formatted = rft.format_curriculum(solution, mode)
    if mode in ("prefix", "prefix+steps"):
        assert formatted.startswith("Solution:")
    if mode in ("steps", "prefix+steps"):
        assert "Step 1:" in formatted
        assert "Step 2:" in formatted
    assert rft.verify(formatted, "7", "gsm8k")["verdict"] == "correct"


def test_curriculum_does_not_renumber_existing_markers():
    formatted = rft.format_curriculum("Step 9: already labelled", "steps")
    assert formatted == "Step 9: already labelled"


def test_dedupe_key_ignores_whitespace():
    assert rft.dedupe_key("a  b\n c") == rft.dedupe_key("a b c")


def test_rejection_sampling_fixture():
    rows = rft._sample_source(FIXTURE)
    accepted, summary = rft._rejection(rows, "mixed", "none")
    assert summary["prompts"] == 3
    assert summary["sampled_solutions"] == 10
    assert summary["verified_solutions"] == 4
    assert summary["kept_solutions"] == 3
    assert summary["verified_rate"] == pytest.approx(0.4)
    assert summary["dedup_retention"] == pytest.approx(0.75)
    assert summary["prompts_with_a_kept_solution"] == 2
    ids = [entry["id"] for entry in accepted]
    assert ids.count("gsm8k-train:0") == 1
    assert ids.count("math-train:algebra:0") == 2


def test_contamination_check_fails_closed_on_overlap():
    item = {
        "id": "x",
        "prompt": "What is the sum of three and four",
        "solution": "The sum of three and four is seven. #### 7",
    }
    tests = [{"task": "synthetic", "text": item["prompt"] + " " + item["solution"]}]
    report = rft.contamination_check([item], tests)
    assert report["max_jaccard"] == pytest.approx(1.0)
    assert report["fail_closed"]


def test_contamination_check_passes_on_disjoint_text():
    item = {"id": "x", "prompt": "alpha beta gamma delta epsilon",
            "solution": "zeta eta theta iota kappa #### 1"}
    tests = [{"task": "synthetic", "text": "one two three four five six seven"}]
    report = rft.contamination_check([item], tests)
    assert report["max_jaccard"] == 0.0
    assert not report["fail_closed"]


def _logits(seed: int, shape=(1, 4, 7)):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def test_sft_loss_matches_cross_entropy():
    logits = _logits(1, (2, 3, 7))
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    loss, ce = rft_objective.sft_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 7), targets.reshape(-1))
    assert torch.allclose(loss, expected)
    assert torch.allclose(ce, expected)


def test_plain_kd_is_ce_plus_kl():
    student = _logits(2)
    base = _logits(3)
    teacher = _logits(4)
    targets = torch.tensor([[0, 1, 2, 3]])
    temperature = 2.0
    kl_weight = 0.5
    loss, ce, kl, diagnostics = rft_objective.distill_loss(
        student, base, teacher, targets, mode="kd",
        temperature=temperature, kl_weight=kl_weight)
    expected_ce = torch.nn.functional.cross_entropy(
        student.reshape(-1, 7), targets.reshape(-1))
    expected_kl = corrective._per_position_kl(student, teacher, temperature).mean()
    expected = expected_ce + kl_weight * temperature ** 2 * expected_kl
    assert torch.allclose(loss, expected)
    assert diagnostics["objective"] == "ce_kl"


def test_kd_correction_zero_preservation_when_student_is_base():
    base = _logits(5)
    teacher = _logits(6)
    targets = torch.tensor([[0, 1, 2, 3]])
    gate = torch.zeros(1, 4)
    _loss, _ce, _kl, diagnostics = rft_objective.distill_loss(
        base, base, teacher, targets, mode="kd_correction",
        temperature=2.0, kl_weight=0.5, weighting="signed",
        preserve_weight=0.5, interference_weight=0.05, gate=gate)
    assert diagnostics["base_preservation_penalty"] == pytest.approx(0.0)
    assert diagnostics["gate_mean_abs"] == pytest.approx(0.0)
    assert diagnostics["interference_penalty"] == pytest.approx(0.0)
    assert diagnostics["weighting"] == "signed"
    assert diagnostics["advantage_weight_min"] > 0.0


def test_kd_correction_uses_all_registered_weightings():
    student = _logits(7)
    base = _logits(8)
    teacher = _logits(9)
    targets = torch.tensor([[0, 1, 2, 3]])
    for weighting in ("kl", "signed", "both"):
        loss, _ce, _kl, diagnostics = rft_objective.distill_loss(
            student, base, teacher, targets, mode="kd_correction",
            temperature=2.0, kl_weight=0.5, weighting=weighting,
            preserve_weight=0.5, interference_weight=0.05,
            gate=torch.full((1, 4), 0.4))
        assert torch.isfinite(loss)
        assert diagnostics["objective"] == "correction"
        assert diagnostics["mean_kl"] >= 0.0


def test_combined_loss_is_the_weighted_sum():
    sft_logits = _logits(10)
    sft_targets = torch.tensor([[0, 1, 2, 3]])
    kd_student = _logits(11)
    kd_base = _logits(12)
    kd_teacher = _logits(13)
    kd_targets = torch.tensor([[3, 2, 1, 0]])
    sft_batch = {"student_logits": sft_logits, "targets": sft_targets}
    kd_batch = {
        "student_logits": kd_student,
        "base_logits": kd_base,
        "teacher_logits": kd_teacher,
        "targets": kd_targets,
    }
    loss, components = rft_objective.combined_loss(
        sft_batch, kd_batch, sft_weight=0.3, kd_weight=0.7)
    sft_expected, _ = rft_objective.sft_loss(sft_logits, sft_targets)
    kd_expected, _, _, _ = rft_objective.distill_loss(
        kd_student, kd_base, kd_teacher, kd_targets, mode="kd")
    expected = 0.3 * sft_expected + 0.7 * kd_expected
    assert loss.item() == pytest.approx(float(expected), rel=1e-5)
    assert components["sft_ce"] >= 0.0
    assert components["kd_kl"] >= 0.0


def test_kd_refuses_correction_terms_under_the_control_mode():
    student = _logits(14)
    base = _logits(15)
    teacher = _logits(16)
    targets = torch.tensor([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        rft_objective.distill_loss(student, base, teacher, targets,
                                   mode="kd", preserve_weight=0.5)


def test_full_backbone_sft_wiring():
    train_module.validate_objective_args(SimpleNamespace(
        objective="sft", arm="full", kl_weight=0.0))
    with pytest.raises(SystemExit):
        train_module.validate_objective_args(SimpleNamespace(
            objective="sft", arm="full", kl_weight=0.5))
    with pytest.raises(SystemExit):
        train_module.validate_objective_args(SimpleNamespace(
            objective="ce_kl", arm="full", kl_weight=0.0))
    with pytest.raises(SystemExit):
        train_module.validate_objective_args(SimpleNamespace(
            objective="correction", arm="full", kl_weight=0.5))


def test_sft_loss_masks_prompt_and_padding_positions():
    logits = _logits(21, (2, 3, 7))
    targets = torch.tensor([[-100, 2, 3], [-100, -100, 5]])
    loss, ce = rft_objective.sft_loss(logits, targets)
    flat_logits = logits.reshape(-1, 7)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != -100
    expected = torch.nn.functional.cross_entropy(
        flat_logits[valid], flat_targets[valid])
    assert torch.allclose(loss, expected)
    assert torch.allclose(ce, expected)
    # the denominator is the supervised count, not the padded token count
    unpadded = torch.nn.functional.cross_entropy(
        flat_logits.reshape(-1, 7), flat_targets, ignore_index=-100,
        reduction="sum") / int(valid.sum())
    assert torch.allclose(loss, unpadded)


def test_sft_loss_all_ignored_is_a_zero_with_a_graph():
    logits = _logits(22, (1, 2, 7)).requires_grad_(True)
    targets = torch.full((1, 2), -100, dtype=torch.long)
    loss, ce = rft_objective.sft_loss(logits, targets)
    assert float(loss.detach()) == 0.0
    assert float(ce.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def test_pack_examples_masks_prompt_and_pads_without_splitting():
    examples = [
        {"id": "a", "prompt_ids": [1, 2], "solution_ids": [3, 4]},
        {"id": "b", "prompt_ids": [5], "solution_ids": [6, 7, 8]},
    ]
    packed = sft_data.pack_examples(examples, seq_len=8, pad_id=0, eos_id=9)
    assert packed.report["rows"] == 2
    assert packed.report["boundary_safe"] is True
    assert packed.report["completion_only_loss"] is True
    assert packed.report["padding_tokens"] == 3 + 3
    assert packed.report["supervised_tokens"] == 3 + 4
    assert packed.input_ids[0].tolist() == [1, 2, 3, 4, 9, 0, 0, 0]
    assert packed.labels[0].tolist() == [-100, -100, 3, 4, 9, -100, -100, -100]
    assert packed.input_ids[1].tolist() == [5, 6, 7, 8, 9, 0, 0, 0]
    assert packed.labels[1].tolist() == [-100, 6, 7, 8, 9, -100, -100, -100]


def test_pack_examples_overflow_fails_closed_and_drop_is_counted():
    examples = [{"id": "long", "prompt_ids": [1], "solution_ids": [2, 3, 4, 5]}]
    with pytest.raises(ValueError):
        sft_data.pack_examples(examples, seq_len=5, pad_id=0, eos_id=9)
    with pytest.raises(ValueError):
        sft_data.pack_examples(examples, seq_len=5, pad_id=0, eos_id=9,
                               overflow="drop")
    good = examples + [{"id": "short", "prompt_ids": [1], "solution_ids": [2]}]
    packed = sft_data.pack_examples(good, seq_len=5, pad_id=0, eos_id=9,
                                    overflow="drop")
    assert packed.report["dropped_overflow"] == 1
    assert packed.report["packed_examples"] == 1


def test_pack_examples_multiple_per_row_marks_itself_unsafe():
    examples = [
        {"id": "a", "prompt_ids": [1], "solution_ids": [2]},
        {"id": "b", "prompt_ids": [3], "solution_ids": [4]},
    ]
    packed = sft_data.pack_examples(examples, seq_len=8, pad_id=0, eos_id=9,
                                    examples_per_row=2)
    assert packed.report["rows"] == 1
    assert packed.report["boundary_safe"] is False
    assert "attention" in packed.report["boundary_safe_note"]


def test_example_batches_shift_labels_and_cycle_rows():
    packed = sft_data.pack_examples(
        [{"id": "a", "prompt_ids": [1, 2], "solution_ids": [3, 4]}],
        seq_len=6, pad_id=0, eos_id=9)
    batches = list(sft_data.example_batches(packed, micro_batch=1,
                                            grad_accum=2, steps=2))
    assert len(batches) == 2
    inputs, targets = batches[0][0]
    assert inputs.tolist() == [[1, 2, 3, 4, 9]]
    assert targets.tolist() == [[-100, 3, 4, 9, -100]]
    # the second optimiser step has cycled back to the only row
    assert batches[1][0][0].tolist() == [[1, 2, 3, 4, 9]]


def _stratified_items():
    items = []
    for source, difficulty, length in (
            ("gsm8k", 0, 10), ("gsm8k", 0, 10),
            ("gsm8k", 2, 100), ("gsm8k", 2, 100),
            ("math", 1, 20), ("math", 1, 20),
            ("math", 5, 200), ("math", 5, 200)):
        items.append({"id": f"{source}-{difficulty}-{length}-{len(items)}",
                      "dataset": source, "difficulty": difficulty,
                      "reasoning_length": length})
    return items


def test_stratified_selection_is_seeded_balanced_and_not_a_prefix():
    items = _stratified_items()
    selected, report = rft.stratified_select(items, 6, seed=451)
    assert len(selected) == 6
    assert report["pool"] == 8 and report["selected"] == 6
    again, _ = rft.stratified_select(items, 6, seed=451)
    assert [item["id"] for item in selected] == [item["id"] for item in again]
    assert {item["dataset"] for item in selected} == {"gsm8k", "math"}
    assert [item["id"] for item in selected] != [item["id"] for item in items[:6]]
    assert all(entry["selected"] >= 1 for entry in report["strata"]
               if entry["pool"] > 0)


def test_select_items_refuses_prefix_start_and_unknown_modes():
    # `prefix` stays available for reproducing earlier runs; no dataset load is
    # needed for the refusal checks.
    with pytest.raises(SystemExit):
        rft.select_items(["gsm8k-train"], 3, 1, "stratified", 451)
    with pytest.raises(SystemExit):
        rft.select_items(["gsm8k-train"], 3, 0, "unknown", 451)
