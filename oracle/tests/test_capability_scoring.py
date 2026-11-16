"""CPU tests for likelihood-based candidate selection (S30, instrument v2).

No GPU and no model weights: the scorer's algebra, the ranking and tie rules, the
selection prompt, the leakage guards and the registration check are pure
functions, and the court stages are exercised with an injected scorer and an
injected checker. The real forwards and the real cargo runs are the GPU and rust
court receipts.
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, ORACLE)

from capability import (  # noqa: E402
    court, formats, forensics, scoring, suite,
)

CALIBRATED = os.path.join(REPO, "fixtures", "capability-calibrated")
REGISTRATION = os.path.join(REPO, "docs",
                            "capability-calibration-preregistration-v2.json")


def _tasks():
    return suite.load_suite(CALIBRATED)["suite"]


def _first(fmt):
    for task in _tasks():
        if formats.task_format(task) == fmt:
            return task
    raise AssertionError(f"no {fmt} item in the calibration split")


# ---------------------------------------------------------------------------
# what is scored


def test_a_patch_candidate_is_scored_as_its_content_without_its_label():
    task = _first(formats.FORMAT_PATCH)
    rows = scoring.selection_contents(task)
    spec = formats.format_spec(task)
    assert [row["label"] for row in rows] == [
        candidate["label"] for candidate in spec["candidates"]]
    for row, candidate in zip(rows, spec["candidates"]):
        assert row["content"] == (candidate.get("function")
                                  or candidate.get("content")).strip()
        ## The label is a position, not content: it must not be in the scored text.
        assert not row["content"].startswith(row["label"])


def test_a_diagnostic_option_is_scored_as_id_and_description():
    task = _first(formats.FORMAT_DIAGNOSTIC)
    rows = scoring.selection_contents(task)
    option = formats.format_spec(task)["options"][0]
    assert rows[0]["content"].startswith(f"`{option['id']}`")
    assert option["description"] in rows[0]["content"]


def test_the_content_continuation_separator_is_identical_for_every_candidate():
    assert scoring.content_continuation("x") == "\n\nx"
    assert scoring.content_continuation("  y\n") == "\n\ny"


def test_letter_continuations_are_position_tokens_with_no_item_text():
    assert scoring.letter_continuations(["A", "B", "C"]) == [" A", " B", " C"]


# ---------------------------------------------------------------------------
# the ranking


def test_choose_takes_the_maximum_and_reports_the_margin():
    scores = [{"sum": -10.0, "mean": -2.0},
              {"sum": -6.0, "mean": -1.5},
              {"sum": -8.0, "mean": -1.0}]
    choice = scoring.choose(scores, ["A", "B", "C"], "mean")
    assert choice["choice"] == "C"
    assert choice["runner_up"] == "B"
    assert math.isclose(choice["margin"], 0.5)
    assert not choice["tie"]
    assert [row["label"] for row in choice["ranking"]] == ["C", "B", "A"]


def test_choose_flags_a_tie_and_breaks_it_by_prompt_order():
    scores = [{"sum": -5.0, "mean": -1.0}, {"sum": -5.0, "mean": -1.0}]
    choice = scoring.choose(scores, ["A", "B"], "mean")
    assert choice["choice"] == "A"
    assert choice["tie"]
    assert "never a decision" in choice["tie_rule"]


def test_choose_refuses_mismatched_lengths():
    try:
        scoring.choose([{"sum": 0.0, "mean": 0.0}], ["A", "B"], "mean")
    except scoring.ScoringError as error:
        assert "same length" in str(error)
    else:  # pragma: no cover - the refusal is the assertion
        raise AssertionError("a mismatched ranking was accepted")


# ---------------------------------------------------------------------------
# the selection prompt and the leakage guards


def test_the_selection_prompt_has_no_example_letter_and_says_what_is_scored():
    for fmt in (formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC):
        task = _first(fmt)
        prompt = scoring.selection_prompt(task)
        assert "for example" not in prompt
        assert "highest-scoring" in prompt
        assert "Return the single letter" not in prompt
    ## Every candidate is still present.
        for row in scoring.selection_contents(task):
            assert row["content"] in prompt


def test_the_scored_texts_carry_no_label_and_no_reference_flag():
    task = _first(formats.FORMAT_PATCH)
    seen = []

    def score_fn(prompt, texts):
        seen.append(list(texts))
        return [{"tokens": 1, "sum": float(-len(text)), "mean": -1.0}
                for text in texts]

    scoring.select_item(task, scoring.selection_prompt(task), score_fn)
    content_texts = seen[0]
    rows = scoring.selection_contents(task)
    assert content_texts == [scoring.content_continuation(row["content"])
                             for row in rows]
    ## No content is prefixed with its own label, and the option that happens to
    ## be the reference cannot be told from the text.
    assert not any(text.lstrip().startswith(f"{row['label']} ")
                   for text, row in zip(content_texts, rows))


def test_select_item_records_every_ranking_and_marks_only_the_primary_scored():
    task = _first(formats.FORMAT_DIAGNOSTIC)
    rows = scoring.selection_contents(task)

    def score_fn(prompt, texts):
        ## The first candidate is preferred on content and the last on letters, so
        ## the modes cannot agree by accident.
        return [{"tokens": 1, "sum": float(-index), "mean": float(-index)}
                for index, _text in enumerate(texts)]

    record = scoring.select_item(task, scoring.selection_prompt(task), score_fn)
    assert set(record["rankings"]) == set(scoring.RANKING_MODES)
    assert [mode for mode, ranking in record["rankings"].items() if ranking["scored"]] \
        == [scoring.PRIMARY_MODE]
    assert record["rankings"][scoring.CONTENT_MEAN]["choice"] == rows[0]["label"]
    assert record["reference"] == forensics.reference_label(task)
    assert record["reference_position"] is not None


def test_the_position_audit_reports_where_the_reference_sits():
    tasks = [task for task in _tasks()
             if formats.task_format(task) == formats.FORMAT_PATCH]
    records = []
    for task in tasks:
        rows = scoring.selection_contents(task)
        reference = forensics.reference_label(task)
        records.append({"format": formats.FORMAT_PATCH,
                        "reference_position": [row["label"] for row in rows].index(
                            reference)})
    audit = scoring.position_audit(records)
    entry = audit["by_format"][formats.FORMAT_PATCH]
    assert entry["items"] == len(tasks)
    assert entry["distinct_positions"] > 1
    assert audit["position_free_by_construction"].startswith("the scored continuation")


# ---------------------------------------------------------------------------
# the registration


def _registration():
    with open(REGISTRATION, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _suite_sha():
    return suite.build_manifest(CALIBRATED)["suite_sha256"]


def test_the_registered_v2_contract_matches_this_code():
    registration = _registration()
    problems = scoring.verify_registration(
        registration, _suite_sha(),
        [formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC])
    assert problems == []
    assert registration["instrument"]["version"] == "v2"
    assert registration["schema"] == scoring.SCHEMA_REGISTRATION


def test_a_registration_bound_to_another_suite_or_mode_is_refused():
    registration = _registration()
    assert scoring.verify_registration(
        registration, "0" * 64, [formats.FORMAT_PATCH])
    changed = json.loads(json.dumps(registration))
    changed["instrument"]["primary_ranking_mode"] = scoring.CONTENT_SUM
    assert scoring.verify_registration(
        changed, _suite_sha(), [formats.FORMAT_PATCH])
    changed = json.loads(json.dumps(registration))
    changed["entropy_screen"]["min_answer_entropy_bits"] = 0.0001
    assert scoring.verify_registration(
        changed, _suite_sha(), [formats.FORMAT_PATCH])


# ---------------------------------------------------------------------------
# the stages


def _reference_scorer(prompt, texts):
    """A scorer whose preference is a marker embedded in the test's contents."""
    return [{"tokens": max(1, len(text.split())),
             "sum": float(10 - len(text)),
             "mean": float(10 - len(text)) / max(1, len(text.split()))}
            for text in texts]


def test_select_writes_a_receipt_the_check_stage_accepts(tmp_path):
    out = tmp_path / "selection.json"
    receipt = court.select(
        _FakeArm(), CALIBRATED, str(out),
        only=["borrow-03-self-referential-window--patch",
              "borrow-03-self-referential-window--diag"],
        score_fn=_reference_scorer, repo=REPO)
    assert receipt["schema"] == court.SCHEMA_GENERATIONS
    assert receipt["stage"] == "select"
    assert receipt["decoding"]["mode"] == "likelihood-selection"
    assert receipt["decoding"]["prompt_variant"] == scoring.PROMPT_VARIANT
    assert receipt["instrument"]["candidate_labels_given_to_the_scorer"] is False
    for item in receipt["items"]:
        selection = item["output"]["extra"]["selection"]
        assert item["output"]["text"] == selection["primary"]["choice"]
        assert selection["primary_mode"] == scoring.PRIMARY_MODE
        assert selection["candidates"]
    ## The receipt is a generations receipt: the check stage recomputes the
    ## variant prompt hash and judges the chosen letter.
    for task in _tasks():
        if task.id not in {item["id"] for item in receipt["items"]}:
            continue
        expected = forensics.sha256_text(scoring.selection_prompt(task))
        recorded = next(item["prompt_sha256"] for item in receipt["items"]
                        if item["id"] == task.id)
        assert recorded == expected


def test_check_recomputes_the_selection_prompt_hash_and_passes_the_ranking_through(
        tmp_path):
    receipt = court.select(
        _FakeArm(), CALIBRATED, str(tmp_path / "selection.json"),
        score_fn=_reference_scorer, repo=REPO)
    result = court.check(receipt, CALIBRATED, str(tmp_path / "result.json"),
                         runner=_reference_runner, min_category_items=0,
                         categories=None,
                         formats_filter=[formats.FORMAT_PATCH,
                                         formats.FORMAT_DIAGNOSTIC])
    assert result["anti_gaming"]["prompt_hashes_match"] is True
    assert result["generations"]["stage"] == "select"
    assert result["instrument"]["name"] == "likelihood-selection"
    assert result["suite"]["format_filter"] == sorted([formats.FORMAT_PATCH,
                                                       formats.FORMAT_DIAGNOSTIC])
    assert all(row["selection"] for row in result["items"])
    assert len(result["items"]) == 37


def test_check_accepts_the_v2_instrument_registration(tmp_path):
    receipt = court.select(
        _FakeArm(), CALIBRATED, str(tmp_path / "selection.json"),
        score_fn=_reference_scorer, preregistration=REGISTRATION, repo=REPO)
    assert receipt["preregistration"]["sha256"]
    result = court.check(
        receipt, CALIBRATED, str(tmp_path / "result.json"),
        preregistration=REGISTRATION, runner=_reference_runner,
        min_category_items=0,
        formats_filter=[formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC])
    assert result["anti_gaming"]["prompt_hashes_match"] is True
    assert result["preregistration"]["sha256"] is not None


def test_a_v2_registration_bound_to_another_suite_is_refused():
    changed = json.loads(json.dumps(_registration()))
    changed["suite_sha256"] = "0" * 64
    manifest = suite.build_manifest(CALIBRATED)
    assert court.verify_preregistration(
        changed, manifest, [formats.FORMAT_PATCH])


class _FakeArm:
    name = "fake-arm"
    kind = "fake"
    tools = False
    accounting = {}
    null_reasons = {}
    provenance = {}

    def metadata(self):
        return {"name": self.name, "kind": self.kind, "tools": False,
                "accounting": {}, "null_reasons": {}, "provenance": {}}


def _reference_runner(task, files):
    reference = task.reference_solution()
    keys = [path for path in files if not path.startswith("tests/")]
    ok = all(files[path].strip() == reference.get(path, "").strip() for path in keys)
    return {"status": "pass" if ok else "test_failed"}


def test_a_selection_run_records_the_margin_as_a_diagnostic_not_as_a_score(tmp_path):
    receipt = court.select(
        _FakeArm(), CALIBRATED, str(tmp_path / "selection.json"),
        only=["borrow-03-self-referential-window--patch"],
        score_fn=_reference_scorer, repo=REPO)
    selection = receipt["items"][0]["output"]["extra"]["selection"]
    assert selection["primary"]["margin"] is not None
    assert receipt["instrument"]["margin_is_diagnostic"] is True
    assert receipt["instrument"]["primary_metric"] == formats.METRIC_EXECUTABLE
