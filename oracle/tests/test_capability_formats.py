"""CPU tests for the redesigned capability formats (S28).

No GPU, no model inference and no cargo: the checkers are injected, so these
tests exercise prompt rendering, answer parsing, body splicing, choice
resolution, diagnostic consequences, the metric separation and the calibration
arithmetic. The real checker runs are the self-test and instrument-smoke
receipts.
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

from capability import adapters, calibrate, court, derive, diagnose, formats, suite  # noqa: E402

SUITE_ROOT = os.path.join(REPO, "fixtures", "capability")

TEMPLATE = """//! A tiny crate.

pub fn add_one(value: u8) -> u8 {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn adds_one() {
        assert_eq!(add_one(1), 2);
    }
}
"""

PATCH_TEMPLATE = """//! A tiny crate.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn adds_one() {
        assert_eq!(add_one(1), 2);
    }
}
"""

REFERENCE = """//! A tiny crate.

pub fn add_one(value: u8) -> u8 {
    value + 1
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn adds_one() {
        assert_eq!(add_one(1), 2);
    }
}
"""

CARGO = """[package]
name = "tiny_fixture"
version = "0.1.0"
edition = "2021"

[workspace]

[lib]
path = "src/lib.rs"
"""

COMPILE_INITIAL = """pub fn label(prefix: &str) -> &str {
    let text = format!("{prefix}:ready");
    &text
}
"""

COMPILE_REFERENCE = """pub fn label(prefix: &str) -> String {
    format!("{prefix}:ready")
}
"""


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def make_task(root, *, fmt, spec, initial, reference, hidden, task_id="tiny-item",
              failure=None, category="lifetime-borrow", intro="Solve the tiny task.",
              observed="an observed failure", extra=None):
    base = os.path.join(str(root), task_id)
    write(os.path.join(base, "initial", "Cargo.toml"), CARGO)
    write(os.path.join(base, "initial", "src", "lib.rs"), initial)
    write(os.path.join(base, "reference", "src", "lib.rs"), reference)
    write(os.path.join(base, "hidden", "tests", "check.rs"), hidden)
    metadata = {
        "schema": "selvedge/capability-task/v1",
        "id": task_id,
        "kind": "suite",
        "category": category,
        "capability": "test fixture",
        "license": "MIT OR Apache-2.0",
        "provenance": "test fixture",
        "entry": "src/lib.rs",
        "checker": {"kind": "cargo_test"},
        "difficulty": "easy",
        "format": fmt,
        "format_spec": spec,
        "extraction_policy": "tolerant",
        "initial_failure": failure or {"kind": "test_failure",
                                       "must_contain": ["not yet implemented"]},
        "observed_failure": observed,
        "prompt_intro": intro,
    }
    metadata.update(extra or {})
    with open(os.path.join(base, "task.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return suite.load_task(base)


BODY_SPEC = {"kind": formats.FORMAT_BODY, "template": "src/lib.rs",
             "function": "add_one"}
PATCH_SPEC = {
    "kind": formats.FORMAT_PATCH,
    "template": "src/lib.rs",
    "function": "add_one",
    "candidates": [
        {"label": "A",
         "function": "pub fn add_one(value: u8) -> u8 {\n    value.saturating_add(1)\n}\n",
         "reference": False},
        {"label": "B",
         "function": "pub fn add_one(value: u8) -> u8 {\n    value + 1\n}\n",
         "reference": True},
        {"label": "C",
         "function": "pub fn add_one(value: u8) -> u8 {\n    value.wrapping_add(2)\n}\n",
         "reference": False},
    ],
}
DIAGNOSTIC_SPEC = {
    "kind": formats.FORMAT_DIAGNOSTIC,
    "options": [
        {"id": "E0308", "description": "mismatched types",
         "consequence": {"kind": "unchanged"}},
        {"id": "E0515", "description": "cannot return a value referencing local data",
         "consequence": {"kind": "reference"}},
    ],
}


def body_task(tmp_path, **kwargs):
    return make_task(tmp_path, fmt=formats.FORMAT_BODY, spec=BODY_SPEC,
                     initial=TEMPLATE, reference=REFERENCE,
                     hidden="use tiny_fixture::add_one;\n\n#[test]\nfn hidden_add() {\n"
                            "    assert_eq!(add_one(41), 42);\n}\n", **kwargs)


def patch_task(tmp_path, **kwargs):
    return make_task(tmp_path, fmt=formats.FORMAT_PATCH, spec=PATCH_SPEC,
                     initial=PATCH_TEMPLATE, reference=REFERENCE,
                     hidden="use tiny_fixture::add_one;\n\n#[test]\nfn hidden_add() {\n"
                            "    assert_eq!(add_one(41), 42);\n}\n", **kwargs)


def diagnostic_task(tmp_path, **kwargs):
    return make_task(tmp_path, fmt=formats.FORMAT_DIAGNOSTIC, spec=DIAGNOSTIC_SPEC,
                     initial=COMPILE_INITIAL, reference=COMPILE_REFERENCE,
                     hidden="use tiny_fixture::label;\n\n#[test]\nfn label_works() {\n"
                            "    assert_eq!(label(\"x\"), \"x:ready\");\n}\n",
                     failure={"kind": "compile_error", "must_contain": ["E0515"]},
                     **kwargs)


def execute(task, parsed):
    """Run the real judge with a scripted checker: the crate must contain the fix."""
    def runner(_task, files):
        content = files[task.entry]
        if task_format_is(task, formats.FORMAT_DIAGNOSTIC):
            ok = content == COMPILE_REFERENCE
        elif task_format_is(task, formats.FORMAT_PATCH):
            ok = "value + 1" in content
        else:
            ok = "value + 1" in content
        return {"status": "pass" if ok else "test_failed"}
    return formats.judge(task, parsed, runner)


def task_format_is(task, fmt):
    return formats.task_format(task) == fmt


# ---------------------------------------------------------------------------
# extraction: the stored T2 answer shapes


def test_tolerant_extraction_reads_a_path_written_inside_the_fence():
    """The M arm wrote `path=src/lib.rs` as the first line of the block."""
    text = "```rust\npath=src/lib.rs\n```rust\npub fn add_one(v: u8) -> u8 { v + 1 }\n```"
    strict = __import__("capability.extract", fromlist=["extract"]).extract_files(
        text, "src/lib.rs", {"src/lib.rs"}, "locked")
    assert not strict.ok or strict.mode == "fenced_entry"
    from capability import extract

    tolerant = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked",
                                     tolerant=True)
    assert tolerant.ok
    assert tolerant.mode == "fenced_paths"
    assert "path=src/lib.rs" not in tolerant.files["src/lib.rs"]
    assert "fn add_one" in tolerant.files["src/lib.rs"]
    assert any("duplicated fence opener" in note for note in tolerant.notes)


def test_strict_extraction_rejects_a_duplicate_path_and_tolerant_takes_the_last():
    from capability import extract

    text = ("```rust path=src/lib.rs\nfn a() {}\n```\n"
            "```rust path=src/lib.rs\nfn b() {}\n```\n")
    strict = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked")
    assert not strict.ok
    assert any("duplicate" in problem for problem in strict.rejected)
    tolerant = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked",
                                     tolerant=True)
    assert tolerant.ok
    assert "fn b" in tolerant.files["src/lib.rs"]
    assert any("last block wins" in note for note in tolerant.notes)


def test_the_registered_suite_still_uses_the_strict_rule():
    """The prompt hashes and the extraction rule of v4 are not changed."""
    tasks = suite.load_suite(SUITE_ROOT)["suite"]
    assert all(formats.extraction_policy(task) == "strict" for task in tasks)
    assert all(formats.task_format(task) == formats.FORMAT_FILE for task in tasks)


# ---------------------------------------------------------------------------
# body completion


def test_a_body_completion_prompt_shows_the_signature_and_the_visible_test(tmp_path):
    task = body_task(tmp_path)
    prompt = suite.render_prompt(task)
    assert "pub fn add_one(value: u8) -> u8 {" in prompt
    assert "todo!()" in prompt
    assert "## Visible failing test" in prompt
    assert "adds_one" in prompt
    assert "hidden_add" not in prompt
    assert "Return only the body" in prompt


def test_body_completion_accepts_the_reference_body(tmp_path):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(
        task, "```rust\n    value + 1\n```")
    assert parsed.ok and parsed.kind == "body"
    outcome = execute(task, parsed)
    assert outcome["status"] == "pass"


def test_body_completion_keeps_only_the_body_of_a_full_function(tmp_path):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(
        task, "```rust\npub fn add_one(value: u8) -> u8 {\n    value + 1\n}\n```")
    assert parsed.ok
    assert parsed.payload["body"].strip() == "value + 1"


def test_a_stub_body_is_recorded_as_stub_returned(tmp_path):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(task, "```rust\ntodo!()\n```")
    assert not parsed.ok
    assert parsed.screen == formats.SCREEN_STUB


def test_an_empty_answer_is_screen_empty(tmp_path):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(task, "   \n")
    assert not parsed.ok and parsed.screen == formats.SCREEN_EMPTY


def test_a_refusal_without_code_is_screen_refusal(tmp_path):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(task, "I cannot solve this task.")
    assert not parsed.ok and parsed.screen == formats.SCREEN_REFUSAL


# ---------------------------------------------------------------------------
# patch selection


def test_patch_selection_reads_a_letter_and_applies_that_candidate(tmp_path):
    task = patch_task(tmp_path)
    parsed = formats.parse_answer(task, "Candidate B — it adds one.\n")
    assert parsed.ok and parsed.payload["label"] == "B"
    assert execute(task, parsed)["status"] == "pass"


def test_patch_selection_of_a_wrong_candidate_fails_the_check(tmp_path):
    task = patch_task(tmp_path)
    parsed = formats.parse_answer(task, "A")
    assert parsed.ok and parsed.payload["label"] == "A"
    assert execute(task, parsed)["status"] == "test_failed"


def test_patch_selection_prompt_shows_every_candidate_and_no_answer_key(tmp_path):
    task = patch_task(tmp_path)
    prompt = suite.render_prompt(task)
    for label in ("A", "B", "C"):
        assert f"### Candidate {label}" in prompt
    assert "reference" not in prompt.lower()
    assert "correct" not in prompt.lower()


def test_a_patch_answer_without_a_readable_letter_is_unparseable(tmp_path):
    task = patch_task(tmp_path)
    parsed = formats.parse_answer(task, "I think the second one looks fine.")
    assert not parsed.ok
    assert parsed.screen == formats.SCREEN_UNPARSEABLE_CHOICE


# ---------------------------------------------------------------------------
# diagnostic identification


def test_the_diagnostic_consequence_verifies_the_right_identification(tmp_path):
    task = diagnostic_task(tmp_path)
    parsed = formats.parse_answer(task, "The error is E0515.")
    assert parsed.ok and parsed.payload["id"] == "E0515"
    outcome = execute(task, parsed)
    assert outcome["diagnostic"]["verified"] is True
    assert outcome["status"] == "pass"


def test_a_wrong_diagnostic_is_not_verified_even_if_it_names_a_real_error(tmp_path):
    task = diagnostic_task(tmp_path)
    parsed = formats.parse_answer(task, "E0308")
    assert parsed.ok and parsed.payload["id"] == "E0308"
    outcome = execute(task, parsed)
    assert outcome["diagnostic"]["verified"] is False
    assert outcome["status"] != "pass"


def test_a_diagnostic_answer_can_be_given_by_option_letter(tmp_path):
    task = diagnostic_task(tmp_path)
    prompt = suite.render_prompt(task)
    letter = next(line.split(".")[0] for line in prompt.splitlines()
                  if "E0515" in line and line.split(".")[0] in ("A", "B"))
    parsed = formats.parse_answer(task, letter)
    assert parsed.ok and parsed.payload["id"] == "E0515"


# ---------------------------------------------------------------------------
# the metric separation


def mini_suite(tmp_path, diagnostic=True):
    root = os.path.join(str(tmp_path), "mini-suite")
    tasks_root = os.path.join(root, "tasks")
    os.makedirs(os.path.join(root, "trio"), exist_ok=True)
    os.makedirs(os.path.join(root, "probes"), exist_ok=True)
    make_task(tasks_root, fmt=formats.FORMAT_BODY, spec=BODY_SPEC,
              initial=TEMPLATE, reference=REFERENCE,
              hidden="#[test]\nfn t() { assert_eq!(tiny_fixture::add_one(1), 2); }\n",
              task_id="exec-item")
    if diagnostic:
        make_task(tasks_root, fmt=formats.FORMAT_DIAGNOSTIC, spec=DIAGNOSTIC_SPEC,
                  initial=COMPILE_INITIAL, reference=COMPILE_REFERENCE,
                  hidden="#[test]\nfn t() { assert_eq!(tiny_fixture::label(\"x\"), "
                         "\"x:ready\"); }\n",
                  failure={"kind": "compile_error", "must_contain": ["E0515"]},
                  task_id="diag-item")
    return root


def run_mini(tmp_path, pass_items, *, partial=None):
    root = mini_suite(tmp_path)
    task_ids = ["exec-item", "diag-item"]
    outputs = {
        "exec-item": "```rust\nvalue + 1\n```",
        "diag-item": "E0515",
    }
    adapter = adapters.FakeAdapter(outputs, name="scripted",
                                   accounting={"task_success": None})
    generations = court.generate(adapter, root, os.path.join(str(tmp_path), "gen.json"))

    def runner(task, _files):
        status = "pass" if task.id in pass_items else "test_failed"
        return {"status": status}

    def partial_runner(_task, _files):
        return {"status": "test_failed",
                "counts": {"passed": partial if partial is not None else 1,
                           "failed": 1}}

    return court.check(generations, root, os.path.join(str(tmp_path), "result.json"),
                       runner=runner, partial_credit=partial is not None,
                       partial_runner=partial_runner, min_category_items=1)


def test_the_diagnostic_metric_is_never_pooled_with_executable_success(tmp_path):
    result = run_mini(tmp_path, pass_items={"diag-item"})
    scores = result["scores"]
    assert scores["primary_composite"] == pytest.approx(0.0)
    assert scores["verified_diagnostic"]["by_category"]["lifetime-borrow"]["mean"] == 1.0
    assert scores["verified_diagnostic"]["pooled_with_primary"] is False
    diag_row = next(row for row in result["items"] if row["id"] == "diag-item")
    exec_row = next(row for row in result["items"] if row["id"] == "exec-item")
    assert diag_row["metric"] == formats.METRIC_DIAGNOSTIC
    assert exec_row["metric"] == formats.METRIC_EXECUTABLE
    assert diag_row["value"] == 1.0
    assert exec_row["value"] == 0.0


def test_partial_credit_is_secondary_and_never_changes_the_composite(tmp_path):
    plain = run_mini(tmp_path, pass_items={"exec-item"})
    with_credit = run_mini(tmp_path, pass_items={"exec-item"}, partial=2)
    assert plain["scores"]["primary_composite"] == pytest.approx(1.0)
    assert with_credit["scores"]["primary_composite"] == pytest.approx(1.0)
    row = next(row for row in with_credit["items"] if row["id"] == "exec-item")
    assert row["partial_credit"]["primary"] is False
    assert row["partial_credit"]["fraction"] == pytest.approx(2 / 3)


def test_the_check_receipt_records_a_taxonomy_class_for_every_item(tmp_path):
    result = run_mini(tmp_path, pass_items={"exec-item"})
    classes = {row["id"]: row["failure_class"] for row in result["items"]}
    assert classes["exec-item"] == "pass"
    assert classes["diag-item"] == "tests_failed"
    assert set(result["scores"]["failure_classes"]) <= set(court.STATUS_CLASS.values()) \
        | {"pass", "no_code_block", "empty_answer"}


# ---------------------------------------------------------------------------
# failure taxonomy


@pytest.mark.parametrize("text,status,rejected,expected", [
    ("", "invalid_solution", ["no code block and no recognisable Rust body"],
     "empty_answer"),
    ("I cannot help with that.", "invalid_solution",
     ["no code block and no recognisable Rust body"], "refusal"),
    ("```rust\nfn a() {}\n", "invalid_solution",
     ["no code block and no recognisable Rust body"], "truncated"),
    ("prose only", "invalid_solution",
     ["no code block and no recognisable Rust body"], "no_code_block"),
    ("```rust path=tests/check.rs\nfn a() {}\n```", "invalid_solution",
     ["path is harness-owned: 'tests/check.rs'"], "wrong_file"),
    ("```rust path=src/lib.rs\nfn a() {}\n```\n```rust path=src/lib.rs\nfn b() {}\n```",
     "invalid_solution", ["duplicate path 'src/lib.rs'"], "duplicate_path_rejected"),
])
def test_each_invalid_failure_class_is_reproduced_from_stored_text(text, status,
                                                                  rejected, expected):
    assert diagnose._classify_invalid(text, rejected, None, 3072) == expected


def test_degenerate_repetition_is_its_own_class():
    text = "The tests must be run with cargo test.\n" * 12
    assert diagnose._classify_invalid(text, ["no code block"], None, 3072) == \
        "degenerate_repetition"


@pytest.mark.parametrize("status,expected", [
    ("pass", "pass"),
    ("test_failed", "tests_failed"),
    ("compile_error", "non_compiling"),
    ("timeout", "timeout"),
    ("oom", "oom"),
])
def test_each_checker_failure_class_maps_from_its_status(tmp_path, status, expected):
    task = body_task(tmp_path)
    parsed = formats.parse_answer(task, "```rust\nvalue + 1\n```")
    assert court.failure_class_for(task, parsed, {"status": status}) == expected


def test_a_taxonomy_receipt_classifies_stored_receipts_without_inference(tmp_path):
    root = mini_suite(tmp_path)
    outputs = {"exec-item": "", "diag-item": "garbage"}
    adapter = adapters.FakeAdapter(outputs, name="scripted",
                                   accounting={"task_success": None})
    directory = os.path.join(str(tmp_path), "arm")
    os.makedirs(directory, exist_ok=True)
    generations = court.generate(adapter, root,
                                os.path.join(directory, "generations.json"))
    court.check(generations, root, os.path.join(directory, "result.json"),
                runner=lambda task, files: {"status": "pass"},
                min_category_items=1)
    receipt = diagnose.analyse({"scripted": directory}, root)
    assert receipt["scope"].startswith("stored receipts only")
    assert receipt["summary"]["items"] == 2
    classes = receipt["summary"]["classes"]
    assert classes.get("empty_answer") == 1
    assert classes.get("unparseable_choice") == 1
    text = diagnose.render_report(receipt)
    assert "Capability failure taxonomy" in text
    path = os.path.join(str(tmp_path), "report.md")
    diagnose.write(path, text)
    assert diagnose.check_report(path, receipt)[0] is True
    diagnose.write(path, text + "tampered\n")
    assert diagnose.check_report(path, receipt)[0] is False


# ---------------------------------------------------------------------------
# derivation


def test_the_calibration_split_is_disjoint_from_scored_and_sealed():
    partition = derive.partition_sources(SUITE_ROOT, 3)
    calibration = {task.id for tasks in partition["calibration"].values()
                   for task in tasks}
    scored = {task.id for tasks in partition["scored"].values() for task in tasks}
    assert not calibration & scored
    assert len(calibration) == 24
    sealed = {task.id for task in suite.load_collection(
        os.path.join(SUITE_ROOT, "sealed"))}
    assert not (calibration | scored) & sealed


def test_deriving_the_same_split_twice_gives_the_same_tree(tmp_path):
    first = os.path.join(str(tmp_path), "first")
    second = os.path.join(str(tmp_path), "second")
    one = derive.derive(SUITE_ROOT, first, calibration_per_category=1)
    two = derive.derive(SUITE_ROOT, second, calibration_per_category=1)
    assert one["items"] == two["items"]
    assert one["manifest_sha256"] == two["manifest_sha256"]


def test_derived_items_validate_and_carry_their_format(tmp_path):
    root = os.path.join(str(tmp_path), "derived")
    payload = derive.derive(SUITE_ROOT, root, calibration_per_category=1)
    loaded = suite.load_suite(root)
    by_id = {task.id: task for task in loaded["suite"]}
    assert payload["counts"]["items"] == len(by_id)
    assert any(formats.task_format(task) == formats.FORMAT_BODY
               for task in by_id.values())
    assert any(formats.task_format(task) == formats.FORMAT_PATCH
               for task in by_id.values())
    assert any(formats.task_format(task) == formats.FORMAT_DIAGNOSTIC
               for task in by_id.values())
    assert any(formats.task_format(task) == formats.FORMAT_FILE
               for task in by_id.values())
    for task in by_id.values():
        assert task.metadata["derived_from"]["source_id"] in {
            entry["source"] for entry in payload["items"]}
        assert formats.validate_declaration(task) == []
    ok, message = derive.check_derivation(root)
    assert ok, message


def test_a_derived_body_task_shows_a_visible_test_and_hides_the_hidden_one(tmp_path):
    root = os.path.join(str(tmp_path), "derived")
    payload = derive.derive(SUITE_ROOT, root, calibration_per_category=1)
    body_id = next(entry["id"] for entry in payload["items"]
                   if entry["format"] == formats.FORMAT_BODY)
    task = suite.load_task(os.path.join(root, "tasks", body_id))
    prompt = suite.render_prompt(task)
    hidden = task.read_hidden("tests/check.rs")
    names = [line.split("fn ")[1].split("(")[0]
             for line in hidden.splitlines() if line.strip().startswith("fn ")]
    assert names, hidden
    assert "## Visible failing test" in prompt
    assert "todo!()" in prompt


def test_a_patch_item_names_the_reference_candidate_in_its_declaration(tmp_path):
    root = os.path.join(str(tmp_path), "derived")
    payload = derive.derive(SUITE_ROOT, root, calibration_per_category=1)
    patch_id = next(entry["id"] for entry in payload["items"]
                    if entry["format"] == formats.FORMAT_PATCH)
    task = suite.load_task(os.path.join(root, "tasks", patch_id))
    marked = [candidate for candidate in formats.format_spec(task)["candidates"]
              if candidate.get("reference")]
    assert len(marked) == 1


# ---------------------------------------------------------------------------
# calibration


def registration_payload():
    return {
        "schema": "selvedge/capability-calibration-preregistration/v1",
        "window": dict(calibrate.WINDOW),
        "primary": {"metric": formats.METRIC_EXECUTABLE},
        "secondary": {"metric": "fraction_of_hidden_tests_passed"},
        "path": "docs/capability-calibration-preregistration-v1.json",
        "sha256": "0" * 64,
    }


def synthetic_results(patterns, *, suite_sha="a" * 64, partial=None):
    """patterns: arm -> list of statuses, aligned with item ids."""
    ids = [f"item-{index}" for index in range(len(next(iter(patterns.values()))))]
    results = {}
    for arm, statuses in patterns.items():
        items = []
        for item_id, status in zip(ids, statuses):
            row = {
                "id": item_id,
                "category": "lifetime-borrow",
                "format": formats.FORMAT_BODY,
                "tier": "easy",
                "metric": formats.METRIC_EXECUTABLE,
                "status": status,
                "value": 1.0 if status == "pass" else 0.0,
            }
            if partial and arm in partial:
                row["partial_credit"] = {"available": True,
                                         "fraction": partial[arm]}
            items.append(row)
        results[arm] = {"schema": "selvedge/capability-result/v1",
                        "suite": {"suite_sha256": suite_sha},
                        "items": items}
    return results


def five_arms(statuses_per_item):
    """statuses_per_item: list of statuses per item; returns 5 arm patterns."""
    patterns = {f"arm{index}": [] for index in range(5)}
    for index, passes in enumerate(statuses_per_item):
        for arm_index, arm in enumerate(sorted(patterns)):
            patterns[arm].append("pass" if arm_index < passes else "test_failed")
    return patterns


def test_the_acceptance_window_retires_a_zero_pass_item():
    results = synthetic_results(five_arms([0, 2]))
    records = calibrate.item_records(calibrate.per_arm_rows(results))
    by_id = {record["id"]: record for record in records}
    assert by_id["item-0"]["accepted"] is False
    assert any("zero pass rate" in reason for reason in by_id["item-0"]["reasons"])


def test_the_acceptance_window_retires_a_saturated_item():
    results = synthetic_results(five_arms([5, 2]))
    by_id = {record["id"]: record
             for record in calibrate.item_records(calibrate.per_arm_rows(results))}
    assert by_id["item-0"]["accepted"] is False
    assert any("saturated" in reason for reason in by_id["item-0"]["reasons"])


def test_an_item_needs_discrimination_not_only_a_mid_pass_rate():
    # Two items with complementary arm splits: every arm passes exactly one of
    # them, so both have a mid pass rate, both arm totals are equal (no
    # correlation is definable) and there is no partial credit to measure. A
    # mid pass rate alone must not be enough to keep an item.
    patterns = {
        "a": ["pass", "test_failed"],
        "b": ["pass", "test_failed"],
        "c": ["pass", "test_failed"],
        "d": ["test_failed", "pass"],
        "e": ["test_failed", "pass"],
    }
    results = synthetic_results(patterns)
    records = calibrate.item_records(calibrate.per_arm_rows(results))
    assert sorted(round(record["pass_rate"], 3) for record in records) == [0.4, 0.6]
    assert all(record["accepted"] is False for record in records)
    assert any("does not discriminate" in reason
               for record in records for reason in record["reasons"])


def test_partial_credit_spread_can_establish_discrimination():
    results = synthetic_results(five_arms([2, 2]), partial={"arm0": 1.0, "arm1": 0.8,
                                                            "arm2": 0.5, "arm3": 0.2,
                                                            "arm4": 0.0})
    records = calibrate.item_records(calibrate.per_arm_rows(results))
    assert records[0]["partial_credit_spread"] >= 0.15
    assert records[0]["accepted"] is True


def test_a_cell_with_two_of_three_items_retired_is_reformatted():
    def record(item_id, accepted):
        return {"id": item_id, "category": "c", "format": formats.FORMAT_BODY,
                "tier": "easy", "metric": formats.METRIC_EXECUTABLE,
                "pass_rate": 0.6 if accepted else 0.0, "accepted": accepted,
                "reasons": [] if accepted else ["zero pass rate"]}

    cells = calibrate.cell_table([record("i0", True), record("i1", False),
                                 record("i2", False)])
    assert cells[0]["retired"] == 2
    assert cells[0]["decision"] == "reformatted"
    assert cells[0]["scored"] is False
    one_retired = calibrate.cell_table([record("i0", True), record("i1", True),
                                       record("i2", False)])
    assert one_retired[0]["decision"] == "scored"
    assert one_retired[0]["accepted"] == 2


def test_the_ladder_reports_a_non_monotone_design_as_a_violation():
    records = [
        {"id": "b", "category": "c", "format": formats.FORMAT_BODY,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.8, "accepted": True,
         "reasons": []},
        {"id": "p", "category": "c", "format": formats.FORMAT_PATCH,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.2, "accepted": True,
         "reasons": []},
        {"id": "f", "category": "c", "format": formats.FORMAT_FILE,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.6, "accepted": True,
         "reasons": []},
        {"id": "f2", "category": "c", "format": formats.FORMAT_FILE,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.6, "accepted": True,
         "reasons": []},
        {"id": "b2", "category": "c", "format": formats.FORMAT_BODY,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.8, "accepted": True,
         "reasons": []},
        {"id": "p2", "category": "c", "format": formats.FORMAT_PATCH,
         "metric": formats.METRIC_EXECUTABLE, "pass_rate": 0.2, "accepted": True,
         "reasons": []},
    ]
    ladder = calibrate.ladder_check(calibrate.cell_table(records))
    assert ladder["monotone"] is False
    assert ladder["violations"] == ["c"]


def test_the_suite_version_is_refused_when_the_ladder_is_not_monotone():
    calibration = {
        "summary": {"categories_scorable": True, "reformatted_categories": []},
        "ladder": {"violations": ["c"]},
        "items": [], "cells": [], "power": {}, "window": {}, "suite": {},
        "calibration_receipt": "receipt.json",
    }
    with pytest.raises(calibrate.CalibrationError, match="not .*monotone"):
        calibrate.build_suite_version(calibration, "0" * 40)


def test_required_n_scales_with_variance_and_the_measured_baseline():
    low = calibrate.required_n(0.5, 0.10, 0.25, 8)
    high = calibrate.required_n(0.5, 0.10, 0.50, 8)
    shallow = calibrate.required_n(0.2, 0.10, 0.25, 8)
    assert high["n_per_category"] == pytest.approx(2 * low["n_per_category"])
    assert shallow["n_per_category"] == pytest.approx(6.25 * low["n_per_category"])
    assert low["absolute_effect"] == pytest.approx(0.05)


def test_calibration_refuses_a_registration_window_it_does_not_apply():
    registration = registration_payload()
    registration["window"]["min_partial_credit_spread"] = 0.05
    with pytest.raises(calibrate.CalibrationError, match="min_partial_credit_spread"):
        calibrate.calibrate(SUITE_ROOT, synthetic_results(five_arms([2])),
                            registration, "receipt.json")


def test_calibration_refuses_arms_scored_on_different_suites():
    results = synthetic_results(five_arms([2]))
    results["arm0"]["suite"]["suite_sha256"] = "b" * 64
    with pytest.raises(calibrate.CalibrationError, match="different suites"):
        calibrate.calibrate(SUITE_ROOT, results, registration_payload(), "receipt.json")


def test_a_calibration_receipt_recomputes_and_can_emit_a_version(tmp_path):
    results = synthetic_results(five_arms([2, 2, 2]), partial={
        "arm0": 1.0, "arm1": 0.9, "arm2": 0.6, "arm3": 0.3, "arm4": 0.1})
    payload = calibrate.calibrate(SUITE_ROOT, results, registration_payload(),
                                  "receipt.json")
    path = os.path.join(str(tmp_path), "calibration.json")
    calibrate._write_json(path, payload)
    ok, message = calibrate.recheck_calibration(path)
    assert ok, message
    with pytest.raises(calibrate.CalibrationError, match="incomplete"):
        # one format only: the body/patch/file ladder is incomplete, so the
        # version is refused rather than emitted with a missing tier.
        calibrate.build_suite_version(payload, "0" * 40)


def ladder_results():
    """A complete, monotone ladder: body easier than patch easier than file."""
    arms = ("a", "b", "c", "d", "e")
    plan = [(formats.FORMAT_BODY, 4), (formats.FORMAT_BODY, 3),
            (formats.FORMAT_PATCH, 3), (formats.FORMAT_PATCH, 2),
            (formats.FORMAT_FILE, 2), (formats.FORMAT_FILE, 1)]
    results = {arm: {"schema": "selvedge/capability-result/v1",
                     "suite": {"suite_sha256": "a" * 64},
                     "items": []} for arm in arms}
    for index, (fmt, passes) in enumerate(plan):
        for position, arm in enumerate(arms):
            status = "pass" if position < passes else "test_failed"
            results[arm]["items"].append({
                "id": f"item-{index}", "category": "lifetime-borrow",
                "format": fmt, "tier": formats.FORMAT_TIERS[fmt],
                "metric": formats.METRIC_EXECUTABLE, "status": status,
                "value": 1.0 if status == "pass" else 0.0,
                "partial_credit": {"available": True,
                                   "fraction": 1.0 if status == "pass" else 0.0}})
    return results


def test_a_complete_monotone_ladder_emits_the_calibrated_suite_version(tmp_path):
    payload = calibrate.calibrate(SUITE_ROOT, ladder_results(), registration_payload(),
                                  "receipt.json")
    assert payload["ladder"]["monotone"] is True
    assert payload["summary"]["categories_scorable"] is True
    version = calibrate.build_suite_version(payload, "0" * 40)
    assert version["schema"] == calibrate.SCHEMA_VERSION
    assert version["metrics"]["primary"] == formats.METRIC_EXECUTABLE
    assert version["metrics"]["secondary"] == "fraction_of_hidden_tests_passed"
    assert version["counts"]["scored"] + version["counts"]["retired"] == 6
    path = os.path.join(str(tmp_path), "version.json")
    calibrate._write_json(path, version)
    receipt_path = os.path.join(str(tmp_path), "calibration.json")
    calibrate._write_json(receipt_path, payload)
    ok, message = calibrate.check_suite_version(path, calibrate.load_json(receipt_path))
    assert ok, message
