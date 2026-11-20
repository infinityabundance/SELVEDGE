"""S67: CPU tests for the resolution-instrument arithmetic.

These tests use synthetic rows only. They check the parts the multi-hour GPU
runs depend on and that a GPU hour cannot cheaply re-check: the reference-gold
rule, the disjoint retention partition, and the verdict arithmetic that turns
receipts into the registered win flags.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import s67_fetch_math_test as fetch  # noqa: E402
import s67_retention_audit as retention  # noqa: E402
import s67_verdict as verdict  # noqa: E402


def test_last_boxed_takes_the_final_reference_box():
    solution = ("we get \\boxed{\\frac{1}{2}} as an intermediate step, "
                "so the sum is \\boxed{161}.")
    assert fetch.last_boxed(solution) == "161"
    assert fetch.last_boxed("no boxed answer here") is None


def _write_jsonl_dump(path, rows, metric):
    payload = {"samples": {"leaf": [
        {"doc_id": doc_id, metric: value, "doc_hash": doc_hash}
        for doc_id, value, doc_hash in rows]}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_retention_partition_is_disjoint_and_the_registered_instrument_is_required(
        tmp_path):
    candidate_rows = [(index, 1.0 if index < 6 else 0.0, f"h{index}")
                      for index in range(10)]
    baseline_rows = [(index, 0.0, f"h{index}") for index in range(10)]
    selection_rows = [(index, 1.0, f"h{index}") for index in range(3)]
    pro_candidate = [(index, 1.0 if index < 50 else 0.0, f"p{index}")
                     for index in range(4000)]
    pro_baseline = [(index, 0.0, f"p{index}") for index in range(4000)]

    paths = {}
    for name, rows, metric in (
            ("candidate_mmlu", candidate_rows, "acc"),
            ("baseline_mmlu", baseline_rows, "acc"),
            ("selection_mmlu", selection_rows, "acc"),
            ("candidate_pro", pro_candidate, "exact_match"),
            ("baseline_pro", pro_baseline, "exact_match")):
        path = tmp_path / f"{name}.json"
        _write_jsonl_dump(path, rows, metric)
        paths[name] = str(path)

    out = tmp_path / "audit.json"
    assert retention.main([
        "--candidate-mmlu", paths["candidate_mmlu"],
        "--candidate-mmlu-pro", paths["candidate_pro"],
        "--baseline-mmlu", paths["baseline_mmlu"],
        "--baseline-mmlu-pro", paths["baseline_pro"],
        "--selection-mmlu", paths["selection_mmlu"],
        "--out", str(out),
    ]) == 0
    payload = json.load(open(out, encoding="utf-8"))
    assert payload["mmlu"]["counts"] == {"full": 10, "selection": 3, "unused": 7}
    assert payload["mmlu"]["full"]["items"] == 10
    assert payload["mmlu"]["selection"]["items"] == 3
    assert payload["mmlu"]["unused"]["items"] == 7
    assert payload["mmlu_pro"]["registered_instrument_present"] is True
    assert payload["mmlu_pro"]["sample_size"] == 4000
    # The synthetic candidate improves MMLU and is far better on MMLU-Pro, so
    # the drop bound passes on both.
    assert payload["verdict"]["retention_pass"] is True


def _row(task, correct, items):
    per_problem = []
    for index in range(items):
        flag = index < correct
        per_problem.append({
            "problem_id": f"{task}:{index}",
            "strict_correct": flag,
            "flexible_correct": flag,
            "exact_correct": flag,
        })
    return {
        "task": task, "mode": "non-thinking", "decoding": "greedy",
        "items": items,
        "flexible": {"correct": correct, "items": items,
                     "accuracy": correct / items},
        "wall_seconds": 10.0, "decode_wall_seconds": 9.0,
        "generated_tokens": 1000, "tokens_per_second": 111.1,
        "per_problem": per_problem,
    }


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_verdict_flags_a_resolved_math_effect_and_a_complete_win(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write(calibration, {"selection": {"coefficient": 0.5}})
    soup = tmp_path / "soup.json"
    _write(soup, {"tasks": [
        _row("gsm8k", 3, 4), _row("math500", 3, 4), _row("math_full", 4, 8)]})
    baseline_math = tmp_path / "baseline_math.json"
    _write(baseline_math, {"tasks": [_row("math_full", 0, 8)]})
    post = tmp_path / "post.json"
    _write(post, {"tasks": [_row("gsm8k", 0, 4), _row("math500", 0, 4)]})
    seed0 = tmp_path / "seed0.json"
    _write(seed0, {"tasks": [_row("gsm8k", 0, 4), _row("math500", 0, 4)]})
    audit = tmp_path / "audit.json"
    _write(audit, {"verdict": {"retention_pass": True}})
    out = tmp_path / "verdict.json"

    assert verdict.main([
        "--calibration", str(calibration),
        "--soup-eval", str(soup),
        "--baseline-math", str(baseline_math),
        "--s65-seed0", str(seed0),
        "--post-baseline", str(post),
        "--retention", str(audit),
        "--out", str(out),
    ]) == 0
    payload = json.load(open(out, encoding="utf-8"))
    flags = payload["verdict"]
    assert payload["selection"]["coefficient"] == 0.5
    assert flags["soup_beats_post_trained_math_full"] is True
    assert flags["math_effect_resolved_at_larger_n"] is True
    assert flags["soup_beats_s65_seed0_gsm8k"] is True
    assert flags["complete_registered_win"] is True
    # Wall-clock and throughput are carried for the soup and the baseline.
    assert payload["throughput"]["soup"]["gsm8k"]["wall_seconds"] == 10.0
    assert payload["throughput"]["baseline_math_full"]["items"] == 8


def test_verdict_refuses_a_soup_without_a_math_full_task(tmp_path):
    calibration = tmp_path / "calibration.json"
    _write(calibration, {"selection": {"coefficient": 0.5}})
    soup = tmp_path / "soup.json"
    _write(soup, {"tasks": [_row("gsm8k", 3, 4)]})
    other = tmp_path / "other.json"
    _write(other, {"tasks": [_row("gsm8k", 0, 4)]})
    audit = tmp_path / "audit.json"
    _write(audit, {"verdict": {"retention_pass": True}})

    try:
        verdict.main([
            "--calibration", str(calibration),
            "--soup-eval", str(soup),
            "--baseline-math", str(other),
            "--s65-seed0", str(other),
            "--post-baseline", str(other),
            "--retention", str(audit),
            "--out", str(tmp_path / "verdict.json"),
        ])
    except SystemExit as error:
        assert "missing a task" in str(error)
    else:
        raise AssertionError("a soup receipt without math_full must be refused")
