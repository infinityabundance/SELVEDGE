"""CPU tests for the capability court: levels, arithmetic, controls, anti-gaming.

The model is a `FakeAdapter` that emits the reference solution (or nothing) and
the checker is an injected function, so these tests exercise the harness's own
arithmetic — category balance, paired CIs, failure accounting, parity checks —
without a GPU and without cargo. The real checker runs are the self-test
receipt.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, ORACLE)

from capability import adapters, court, suite  # noqa: E402

SUITE_ROOT = os.path.join(REPO, "fixtures", "capability")


def reference_answer(task) -> str:
    return "\n".join(
        f"```rust path={path}\n{task.read_reference(path).rstrip()}\n```"
        for path in task.reference_files)


def reference_outputs(fail: set | None = None, empty: set | None = None) -> dict:
    fail = fail or set()
    empty = empty or set()
    outputs = {}
    for kind in ("suite", "trio"):
        for task in suite.load_suite(SUITE_ROOT)[kind]:
            outputs[task.id] = "" if task.id in empty else reference_answer(task)
    assert not (fail & empty)
    return outputs


def run(fail: set | None = None, empty: set | None = None, name: str = "arm",
        tools: bool = False, order_sensitive: bool = False, order: str = "sorted",
        few_shot=None, tmp_path=None, preregistration=None):
    adapter = adapters.FakeAdapter(reference_outputs(fail, empty), name=name,
                                   tools=tools, order_sensitive=order_sensitive,
                                   accounting={"task_success": None})
    generation_path = os.path.join(str(tmp_path), f"gen-{name}-{order.replace(':', '')}.json")
    generations = court.generate(adapter, SUITE_ROOT, generation_path, order=order,
                                 few_shot_texts=few_shot,
                                 preregistration=preregistration)
    passes = {task.id for kind in ("suite", "trio")
              for task in suite.load_suite(SUITE_ROOT)[kind]} - set(fail or set())
    passes -= set(empty or set())

    def runner(task, _files):
        return {"status": "pass" if task.id in passes else "test_failed",
                "evidence": {"stderr_sha256": "0" * 64, "stderr_tail": "",
                             "limits": {"address_space_bytes": 1,
                                        "wall_seconds": 1}}}

    result_path = os.path.join(str(tmp_path), f"result-{name}-{order.replace(':', '')}.json")
    return generations, court.check(generations, SUITE_ROOT, result_path, runner=runner,
                                    preregistration=preregistration)


def test_a_full_pass_yields_a_perfect_balanced_composite(tmp_path):
    _generations, result = run(tmp_path=tmp_path)
    scores = result["scores"]
    assert scores["primary_composite"] == pytest.approx(1.0)
    assert scores["counts"] == {"pass": 27}
    for category, entry in scores["by_category"].items():
        assert entry["mean"] == pytest.approx(1.0), category
        assert entry["n"] >= 3
    assert scores["trio_all_surfaces"] is True
    assert result["anti_gaming"]["prompt_hashes_match"] is True
    assert result["anti_gaming"]["checker_visible_to_arm"] is False


def test_a_failed_category_moves_the_composite_by_one_seventh(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    lifetime = {task.id for task in tasks if task.category == "lifetime-borrow"}
    _generations, result = run(fail=lifetime, tmp_path=tmp_path, name="crippled")
    assert result["scores"]["primary_composite"] == pytest.approx(6 / 7)
    assert result["scores"]["by_category"]["lifetime-borrow"]["mean"] == 0.0
    assert result["scores"]["counts"]["test_failed"] == 4


def test_an_empty_answer_is_a_recorded_failure_not_a_crash(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    missing = {tasks[0].id}
    _generations, result = run(empty=missing, tmp_path=tmp_path, name="empty")
    row = next(row for row in result["items"] if row["id"] in missing)
    assert row["status"] == "invalid_solution"
    assert result["scores"]["counts"]["invalid_solution"] == 1


def test_the_category_balanced_composite_beats_a_pooled_average(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    # Fail an entire 3-item category and one item of a 4-item category: pooled
    # success is higher than the category-balanced score, and that is the point.
    fail = {task.id for task in tasks if task.category == "concurrency"}
    fail |= {task.id for task in tasks if task.category == "trait-impl" and True}
    _generations, result = run(fail=fail, tmp_path=tmp_path, name="unbalanced")
    pooled = result["scores"]["pooled"]["mean"]
    balanced = result["scores"]["primary_composite"]
    assert balanced == pytest.approx(5 / 7)
    assert pooled > balanced


def test_compare_meets_the_registered_threshold_by_a_category(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    lifetime = {task.id for task in tasks if task.category == "lifetime-borrow"}
    _arm_generations, arm = run(tmp_path=tmp_path, name="arm")
    _base_generations, baseline = run(fail=lifetime, tmp_path=tmp_path, name="base")
    comparison = court.compare(arm, baseline)
    assert comparison["primary"]["threshold_met"] is True
    assert comparison["primary"]["difference"]["relative_difference"] == pytest.approx(1 / 6)
    assert comparison["primary"]["difference"]["ci_low"] > 0


def test_compare_refuses_a_single_item_difference_as_a_win(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    one = {next(task.id for task in tasks if task.category == "lifetime-borrow")}
    _arm_generations, arm = run(tmp_path=tmp_path, name="arm1")
    _base_generations, baseline = run(fail=one, tmp_path=tmp_path, name="base1")
    comparison = court.compare(arm, baseline)
    assert comparison["primary"]["threshold_met"] is False
    assert comparison["primary"]["difference"]["relative_difference"] < court.PRIMARY_THRESHOLD
    assert "no win is claimed" in comparison["verdict"]


def test_compare_enforces_tool_parity(tmp_path):
    with_tools = adapters.FakeAdapter(reference_outputs(), name="tools", tools=True,
                                      accounting={"task_success": None})
    generations = court.generate(with_tools, SUITE_ROOT,
                                 os.path.join(str(tmp_path), "gen-tools.json"))
    files = {task.id: "pass" for kind in ("suite", "trio")
             for task in suite.load_suite(SUITE_ROOT)[kind]}
    _gen, tool_result = run(tmp_path=tmp_path, name="toolrun", tools=True)
    comparison_arm = dict(tool_result)
    comparison_arm["generations"] = dict(tool_result["generations"])
    _gen2, base_result = run(tmp_path=tmp_path, name="notools")
    with pytest.raises(court.CourtError, match="tool parity"):
        court.compare(comparison_arm, base_result)
    assert generations["adapter"]["tools"] is True
    assert files


def test_compare_enforces_feedback_parity(tmp_path):
    _gen, arm = run(tmp_path=tmp_path, name="fba")
    _gen, base = run(tmp_path=tmp_path, name="fbb")
    tampered = json.loads(json.dumps(base))
    tampered["generations"]["decoding"]["feedback"] = "tests"
    with pytest.raises(court.CourtError, match="feedback parity"):
        court.compare(arm, tampered)


def test_prompt_order_invariance_catches_an_order_sensitive_arm(tmp_path):
    _gen_a, result_a = run(tmp_path=tmp_path, name="rev", order="sorted")
    _gen_b, result_b = run(tmp_path=tmp_path, name="rev", order="rotate:5",
                           order_sensitive=True)
    verdict = court.prompt_order_invariance(result_a, result_b)
    assert verdict["invariant"] is False
    assert verdict["differences"]


def test_few_shot_hygiene_rejects_a_leaked_item(tmp_path):
    task = suite.load_suite(SUITE_ROOT)["suite"][0]
    leak = task.metadata["prompt_intro"]
    hygiene = court.few_shot_disjoint([leak], suite.load_suite(SUITE_ROOT))
    assert hygiene["offending_pairs"] > 0
    with pytest.raises(court.CourtError, match="few-shot"):
        run(tmp_path=tmp_path, name="leak", few_shot=[leak])


def test_the_control_matrix_rejects_missing_and_unexplained_fields():
    with pytest.raises(court.CourtError, match="no reason"):
        court.control_row("frozen student", {})
    complete = {field: 1 for field in court.CONTROL_FIELDS}
    row = court.control_row("frozen student", complete)
    assert set(row["fields"]) == set(court.CONTROL_FIELDS)
    unexplained = json.loads(json.dumps(row))
    unexplained["fields"]["training_tokens"] = {"value": None, "reason": None}
    with pytest.raises(court.CourtError, match="reason"):
        court.validate_control_row(unexplained)
    broken = json.loads(json.dumps(row))
    del broken["fields"]["task_success"]
    with pytest.raises(court.CourtError, match="ten fields"):
        court.validate_control_row(broken)


def test_a_planned_row_records_the_reason_instead_of_a_number():
    reasons = {field: "not run in this phase" for field in court.CONTROL_FIELDS}
    row = court.control_row("equally trained dense adapter", {}, status="planned",
                            null_reasons=reasons)
    assert row["status"] == "planned"
    assert all(entry["value"] is None and entry["reason"] for entry in row["fields"].values())
    table = court.render_control_table([row])
    assert "not run in this phase" in table


def test_complementarity_bounds_a_selector(tmp_path):
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    arm_a_fail = {task.id for index, task in enumerate(tasks) if index % 2 == 1}
    arm_b_fail = {task.id for index, task in enumerate(tasks) if index % 2 == 0}
    _gen, arm_a = run(fail=arm_a_fail, tmp_path=tmp_path, name="A")
    _gen, arm_b = run(fail=arm_b_fail, tmp_path=tmp_path, name="B")
    payload = court.complementarity(arm_a, arm_b)
    assert payload["oracle_union"] == pytest.approx(1.0)
    assert payload["headroom_over_best_single"] > 0
    assert payload["deployable"] is False


def test_capacity_scaling_requires_a_slope_below_one():
    sublinear = [
        {"total_accessible_capacity": 1e9, "active_unique_params": 1e9,
         "cold_bytes_per_token": 1e6},
        {"total_accessible_capacity": 4e9, "active_unique_params": 1e9,
         "cold_bytes_per_token": 2e6},
        {"total_accessible_capacity": 16e9, "active_unique_params": 1e9,
         "cold_bytes_per_token": 4e6},
    ]
    payload = court.capacity_scaling(sublinear, resamples=200)
    assert payload["slope"] == pytest.approx(0.5)
    assert payload["sublinear"] is True
    linear = [dict(row, cold_bytes_per_token=row["total_accessible_capacity"] / 1000)
              for row in sublinear]
    payload = court.capacity_scaling(linear, resamples=200)
    assert payload["slope"] == pytest.approx(1.0)
    assert payload["sublinear"] is False
    with pytest.raises(court.CourtError, match="fixed-active-compute"):
        court.capacity_scaling([dict(sublinear[0], active_unique_params=1e9),
                                dict(sublinear[1], active_unique_params=3e9)])


def test_pareto_dominance_and_front():
    good = {"task_success": 0.6, "cold_bytes_per_token": 10, "p95_ms": 5,
            "residency_bytes": 100}
    bad = {"task_success": 0.5, "cold_bytes_per_token": 20, "p95_ms": 6,
           "residency_bytes": 200}
    assert court.pareto_dominates(good, bad)
    assert not court.pareto_dominates(bad, good)
    front = court.pareto_front([good, bad])
    assert front == [good]


def test_no_hidden_teacher_is_unverified_when_undeclared():
    verdict = court.no_hidden_teacher({"accounting": {}, "null_reasons": {}}, None)
    assert verdict["status"] == "unverified"
    declared = {"accounting": {"teacher_calls_at_inference": 0}, "null_reasons": {}}
    assert court.no_hidden_teacher(declared, None)["status"] == "unverified"
    assert court.no_hidden_teacher(declared, "results/replay.json")["satisfied"] is True
    assert court.no_hidden_teacher(
        {"accounting": {"teacher_calls_at_inference": 3}, "null_reasons": {}},
        "results/replay.json")["status"] == "failed"


def test_a_prompt_hash_mismatch_refuses_the_result(tmp_path):
    generations, _result = run(tmp_path=tmp_path, name="hash")
    generations["items"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(court.CourtError, match="prompt hash"):
        court.check(generations, SUITE_ROOT, os.path.join(str(tmp_path), "bad.json"),
                    runner=lambda task, files: {"status": "pass"})


def test_a_suite_hash_mismatch_refuses_the_result(tmp_path):
    generations, _result = run(tmp_path=tmp_path, name="suitehash")
    generations["suite"]["suite_sha256"] = "0" * 64
    with pytest.raises(court.CourtError, match="different suite"):
        court.check(generations, SUITE_ROOT, os.path.join(str(tmp_path), "bad2.json"),
                    runner=lambda task, files: {"status": "pass"})


def test_the_t2_adapter_is_declared_without_loading_a_model():
    adapter = adapters.T2ArmAdapter(
        "H", "results/artifacts/train/t2/H/step-001024.pt",
        "results/train/admission-configs/t2/H.json",
        "results/train/admission/t2/H.json", "/hf-cache")
    assert adapter.name == "t2-H"
    assert adapter.kind == "t2-arm"
    assert adapter.tools is False


def test_pre_registration_binds_the_suite_and_the_threshold(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    payload = court.register(path, SUITE_ROOT)
    assert payload["primary"]["relative_threshold"] == 0.10
    assert payload["primary"]["bootstrap"]["seed"] == 20261115
    assert payload["trio"]["requires_all_surfaces"] is True
    assert payload["planned_controls"]["3B-8B quantised local baseline"]["status"] == "planned"
    manifest = suite.build_manifest(SUITE_ROOT)
    assert court.verify_preregistration(payload, manifest) == []
    with pytest.raises(court.CourtError, match="append-only"):
        court.register(path, SUITE_ROOT)
    tampered = json.loads(json.dumps(payload))
    tampered["primary"]["relative_threshold"] = 0.01
    problems = court.verify_preregistration(tampered, manifest)
    assert any("threshold" in problem for problem in problems)


def test_check_refuses_a_run_against_a_stale_preregistration(tmp_path):
    task_suite = suite.build_manifest(SUITE_ROOT)
    path = os.path.join(str(tmp_path), "prereg.json")
    payload = court.register(path, SUITE_ROOT)
    payload["suite_sha256"] = "0" * 64
    court.write_json_atomic(path, payload)
    generations, _result = run(tmp_path=tmp_path, name="stale")
    with pytest.raises(court.CourtError, match="does not match this suite"):
        court.check(generations, SUITE_ROOT, os.path.join(str(tmp_path), "stale.json"),
                    preregistration=path,
                    runner=lambda task, files: {"status": "pass"})
    assert task_suite["suite_sha256"]
