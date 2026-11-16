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
    assert result["anti_gaming"]["tool_mode"] == "no-tools"
    assert result["anti_gaming"]["offline_evidence"] == "no-default-route"


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


def test_compare_is_conjunctive_a_category_regression_kills_an_aggregate_win(tmp_path):
    """The verdict must require BOTH conditions.

    The arm wins 12.5% relative on the composite (above the 10% threshold, with a
    CI above zero) by fixing a whole category, but it pays for that by failing one
    of four `trait-impl` items: a 0.25 absolute regression, five times the
    registered 0.05 limit. The old code computed this regression and then ignored
    it, so the arm was reported as a win.
    """
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    whole_category = {task.id for task in tasks if task.category == "lifetime-borrow"}
    one_concurrency = next(task.id for task in tasks
                           if task.category == "concurrency")
    one_item = next(task.id for task in tasks
                    if task.category == "trait-impl")
    _arm_generations, arm = run(fail={one_item}, tmp_path=tmp_path, name="tradeoff")
    _base_generations, baseline = run(fail=whole_category | {one_concurrency},
                                      tmp_path=tmp_path, name="narrow")
    comparison = court.compare(arm, baseline)
    primary = comparison["primary"]
    assert primary["improvement_met"] is True
    assert primary["difference"]["relative_difference"] > court.PRIMARY_THRESHOLD
    assert primary["difference"]["ci_low"] > 0
    assert "trait-impl" in primary["category_regressions_beyond_margin"]
    assert primary["category_regressions_beyond_margin"]["trait-impl"] < (
        -court.SECONDARY_MAX_CATEGORY_REGRESSION)
    assert primary["threshold_met"] is False
    assert "regress beyond" in comparison["verdict"]


def test_a_registered_comparison_uses_the_registered_threshold(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    court.register(path, SUITE_ROOT)
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    lifetime = {task.id for task in tasks if task.category == "lifetime-borrow"}
    _gen, arm = run(tmp_path=tmp_path, name="reg-arm", preregistration=path)
    _gen, baseline = run(fail=lifetime, tmp_path=tmp_path, name="reg-base",
                         preregistration=path)
    assert arm["preregistration"]["sha256"] == court.sha256_file(path)
    assert arm["preregistration"]["primary_relative_threshold"] == 0.10
    comparison = court.compare(arm, baseline)
    assert comparison["primary"]["threshold_relative"] == 0.10
    assert comparison["primary"]["threshold_source"] == "pre-registration receipt"
    assert comparison["primary"]["threshold_met"] is True


def test_compare_refuses_a_threshold_supplied_after_the_results(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    court.register(path, SUITE_ROOT)
    _gen, arm = run(tmp_path=tmp_path, name="thr-arm", preregistration=path)
    _gen, baseline = run(tmp_path=tmp_path, name="thr-base", preregistration=path)
    with pytest.raises(court.CourtError, match="may not be re-thresholded"):
        court.compare(arm, baseline, threshold=0.01)
    # The registered value itself is accepted; it changes nothing.
    assert court.compare(arm, baseline, threshold=0.10)["primary"][
        "threshold_relative"] == 0.10


def test_compare_refuses_a_tampered_registered_threshold(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    court.register(path, SUITE_ROOT)
    _gen, arm = run(tmp_path=tmp_path, name="tamper-arm", preregistration=path)
    _gen, baseline = run(tmp_path=tmp_path, name="tamper-base", preregistration=path)
    tampered = json.loads(json.dumps(arm))
    baseline_tampered = json.loads(json.dumps(baseline))
    for receipt in (tampered, baseline_tampered):
        receipt["preregistration"]["primary_relative_threshold"] = 0.01
    with pytest.raises(court.CourtError, match="tampered|superseded"):
        court.compare(tampered, baseline_tampered)


def test_compare_refuses_to_mix_a_registered_and_an_unregistered_result(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    court.register(path, SUITE_ROOT)
    _gen, arm = run(tmp_path=tmp_path, name="mixed-arm", preregistration=path)
    _gen, baseline = run(tmp_path=tmp_path, name="mixed-base")
    with pytest.raises(court.CourtError, match="one side of the comparison is registered"):
        court.compare(arm, baseline)


def test_a_registered_check_requires_a_recorded_digest(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    court.register(path, SUITE_ROOT)
    generations, _result = run(tmp_path=tmp_path, name="unbound")
    assert generations["preregistration"]["sha256"] is None
    with pytest.raises(court.CourtError, match="records no pre-registration digest"):
        court.check(generations, SUITE_ROOT,
                    os.path.join(str(tmp_path), "unbound-result.json"),
                    preregistration=path,
                    runner=lambda task, files: {"status": "pass"})


def test_a_registered_check_refuses_a_contract_that_changed_after_generation(tmp_path):
    path = os.path.join(str(tmp_path), "prereg.json")
    payload = court.register(path, SUITE_ROOT)
    generations, _result = run(tmp_path=tmp_path, name="drift", preregistration=path)
    payload["registered_utc"] = "2026-11-16T00:00:00Z"
    court.write_json_atomic(path, payload)
    with pytest.raises(court.CourtError, match="changed after generation"):
        court.check(generations, SUITE_ROOT,
                    os.path.join(str(tmp_path), "drift-result.json"),
                    preregistration=path,
                    runner=lambda task, files: {"status": "pass"})


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


def test_a_disjoint_few_shot_example_is_scored_with_the_same_prompt(tmp_path):
    example = "Example: prefer clear names and short functions."
    generations, result = run(tmp_path=tmp_path, name="fewshot",
                              few_shot=[example])
    assert result["anti_gaming"]["prompt_hashes_match"] is True
    assert generations["decoding"]["few_shot_texts"] == [example]
    assert result["scores"]["primary_composite"] == pytest.approx(1.0)


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


def test_level3_never_invents_responsiveness(tmp_path):
    _generations, result = run(tmp_path=tmp_path, name="level3")
    incomplete = court.level3(result, None)
    assert incomplete["complete"] is False
    assert incomplete["p95_ms"]["value"] is None
    assert incomplete["p95_ms"]["reason"]
    assert incomplete["verified_success"] == pytest.approx(1.0)
    complete = court.level3(result, {"p95_ms": 12.5,
                                     "source": "results/runtime/court-f.json"})
    assert complete["complete"] is True
    assert complete["p95_ms"]["value"] == 12.5
    with pytest.raises(court.CourtError, match="p95_ms and a source"):
        court.level3(result, {"p95_ms": 12.5})
    with pytest.raises(court.CourtError, match="p95_ms and a source"):
        court.level3(result, {"source": "x"})


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


# ---------------------------------------------------------------------------
# T2 checkpoint loading: no unchecked `strict=False`


class _Param:
    def __init__(self, *shape):
        self.shape = shape


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class _FakeArm:
    """The two attributes `_verify_checkpoint` reads, in the real shapes."""

    def __init__(self, params):
        self.trainable_names = list(params)
        self.model = _FakeModel(params)


def _checkpoint_fixture(tmp_path, arm="H", params=None, extra=None, receipt=None,
                        mask_order="legacy", step=100):
    """A real checkpoint file, config, admission record and training receipt."""
    params = params or {"model.layers.0.bank.w": (4, 4),
                        "model.layers.0.router.w": (8, 4)}
    checkpoint = tmp_path / "step-000100.pt"
    checkpoint.write_bytes(b"weights-of-arm-" + arm.encode())
    config = tmp_path / "config.json"
    config.write_text("{\"total_accessible\": 1}", encoding="utf-8")
    admission_path = tmp_path / "admission.json"
    admission = {"schema": "selvedge/admission-record/v1", "arm": arm,
                 "admitted": True, "active_unique_params": 774543360,
                 "config_sha256": adapters.sha256_file(str(config))}
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    receipt_path = tmp_path / "run.json"
    payload = {
        "arm": arm,
        "config": {"path": str(config), "sha256": adapters.sha256_file(str(config))},
        "admission": {"path": str(admission_path),
                      "active_unique_params": 774543360},
        "checkpoint": {"name": "step-000100.pt", "path": str(checkpoint),
                       "sha256": adapters.sha256_file(str(checkpoint))},
    }
    if mask_order is not None:
        payload["dispatch_mask_order"] = mask_order
    payload.update(receipt or {})
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    adapter = adapters.T2ArmAdapter(
        arm, str(checkpoint), str(config), str(admission_path), "/hf-cache",
        training_receipt=str(receipt_path), mask_order="legacy")
    adapter._admission = admission
    state = {"step": step, "arm": arm,
             "model": {name: _Param(*shape)
                       for name, shape in (extra if extra is not None else params).items()}}
    arm_model = _FakeArm({name: _Param(*shape) for name, shape in params.items()})
    return adapter, state, arm_model


def test_a_matching_checkpoint_verifies_against_its_receipt(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path)
    adapter._verify_checkpoint(state, arm)
    verification = adapter._verify_receipt(state)
    assert verification["checked"] is True
    assert verification["step"] == 100
    assert verification["mask_order"] == "legacy"


def test_a_checkpoint_from_another_arm_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, arm="H")
    state["arm"] = "X"
    with pytest.raises(adapters.AdapterError, match="checkpoint arm is 'X'"):
        adapter._verify_checkpoint(state, arm)


def test_a_checkpoint_with_an_unexpected_tensor_is_refused(tmp_path):
    params = {"model.layers.0.bank.w": (4, 4)}
    extra = {"model.layers.0.bank.w": (4, 4), "model.layers.0.smuggled.w": (2, 2)}
    adapter, state, arm = _checkpoint_fixture(tmp_path, params=params, extra=extra)
    with pytest.raises(adapters.AdapterError, match="not parameters of this arm"):
        adapter._verify_checkpoint(state, arm)


def test_a_checkpoint_missing_a_trainable_tensor_is_refused(tmp_path):
    params = {"model.layers.0.bank.w": (4, 4), "model.layers.0.router.w": (8, 4)}
    extra = {"model.layers.0.bank.w": (4, 4)}
    adapter, state, arm = _checkpoint_fixture(tmp_path, params=params, extra=extra)
    with pytest.raises(adapters.AdapterError, match="trainable tensors are missing"):
        adapter._verify_checkpoint(state, arm)


def test_a_checkpoint_with_the_wrong_shape_is_refused(tmp_path):
    params = {"model.layers.0.bank.w": (4, 4)}
    extra = {"model.layers.0.bank.w": (2, 4)}
    adapter, state, arm = _checkpoint_fixture(tmp_path, params=params, extra=extra)
    with pytest.raises(adapters.AdapterError, match="expects"):
        adapter._verify_checkpoint(state, arm)


def test_a_checkpoint_at_the_wrong_step_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, step=99)
    with pytest.raises(adapters.AdapterError, match="receipt names step 100"):
        adapter._verify_receipt(state)


def test_a_checkpoint_with_a_different_config_digest_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(
        tmp_path, receipt={"config": {"sha256": "0" * 64}})
    with pytest.raises(adapters.AdapterError, match="config digest"):
        adapter._verify_receipt(state)


def test_a_routed_arm_without_a_recorded_mask_placement_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    with pytest.raises(adapters.AdapterError, match="no dispatch mask placement"):
        adapter._verify_receipt(state)


def _write_amendment(tmp_path, receipt_path, arm="H", decision="CERTIFIED",
                     inferred="legacy", binding_digest=None, replay_digest=None):
    replay = tmp_path / "replay.json"
    replay.write_text("{\"conclusion\": \"legacy\"}", encoding="utf-8")
    amendment = {
        "schema": "selvedge/t2-provenance-amendment/v1",
        "decision": decision,
        "subject_arms": [arm],
        "inferred_mask_order": inferred,
        "adapter_consumption": {
            "binding_sha256": {
                str(receipt_path): binding_digest
                or adapters.sha256_file(str(receipt_path)),
            }
        },
        "evidence_chain": {
            "replay": {
                "receipt": str(replay),
                "receipt_sha256": replay_digest
                or adapters.sha256_file(str(replay)),
            }
        },
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    return path


def test_a_certified_amendment_supplies_a_missing_mask_placement(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    adapter.provenance_amendment = str(_write_amendment(tmp_path, tmp_path / "run.json"))
    verification = adapter._verify_receipt(state)
    assert verification["checked"] is True
    amendment = verification["provenance_amendment"]
    assert amendment["used"] is True
    assert amendment["inferred_mask_order"] == "legacy"
    assert len(amendment["sha256"]) == 64
    assert amendment["replay_receipt_sha256"]


def test_an_amendment_bound_to_another_receipt_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    adapter.provenance_amendment = str(_write_amendment(
        tmp_path, tmp_path / "run.json", binding_digest="0" * 64))
    with pytest.raises(adapters.AdapterError, match="binding"):
        adapter._verify_receipt(state)


def test_an_amendment_for_another_arm_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    adapter.provenance_amendment = str(_write_amendment(
        tmp_path, tmp_path / "run.json", arm="X"))
    with pytest.raises(adapters.AdapterError, match="does not cover arm"):
        adapter._verify_receipt(state)


def test_an_amendment_inferring_a_different_mask_order_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    adapter.provenance_amendment = str(_write_amendment(
        tmp_path, tmp_path / "run.json", inferred="pair"))
    with pytest.raises(adapters.AdapterError, match="amended mask order"):
        adapter._verify_receipt(state)


def test_an_amendment_with_a_stale_replay_digest_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order=None)
    adapter.provenance_amendment = str(_write_amendment(
        tmp_path, tmp_path / "run.json", replay_digest="0" * 64))
    with pytest.raises(adapters.AdapterError, match="replay receipt digest"):
        adapter._verify_receipt(state)


def test_a_dense_arm_does_not_need_a_mask_placement(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, arm="D", mask_order=None)
    assert adapter._verify_receipt(state)["checked"] is True


def test_a_mismatched_mask_placement_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(tmp_path, mask_order="pair")
    with pytest.raises(adapters.AdapterError, match="mask_order='pair'"):
        adapter._verify_receipt(state)


def test_a_checkpoint_digest_that_does_not_match_the_receipt_is_refused(tmp_path):
    adapter, state, arm = _checkpoint_fixture(
        tmp_path, receipt={"checkpoint": {"name": "step-000100.pt",
                                          "sha256": "0" * 64}})
    with pytest.raises(adapters.AdapterError, match="digest does not match"):
        adapter._verify_receipt(state)
