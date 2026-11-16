"""CPU tests for the fresh suite's structure, prompts and extraction rules.

These run in the python court (no cargo) and do not execute checkers; the real
checker runs are `scripts/capability_selftest.py`, whose receipt is committed.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, ORACLE)

from capability import extract, suite  # noqa: E402

SUITE_ROOT = os.path.join(REPO, "fixtures", "capability")


def collections():
    return suite.load_suite(SUITE_ROOT)


def test_the_suite_has_the_registered_shape():
    loaded = collections()
    assert len(loaded["suite"]) == 144
    assert len(loaded["trio"]) == 9
    assert len(loaded["probes"]) == 6
    categories = {task.category for task in loaded["suite"]}
    assert categories == {
        "lifetime-borrow", "failing-test", "api-migration", "dependency-compat",
        "concurrency", "trait-impl", "repo-scale", "semantic-repair"}
    for task in loaded["suite"] + loaded["trio"]:
        assert task.metadata["license"] == "MIT OR Apache-2.0"
        assert task.metadata["initial_failure"]["must_contain"]
        assert task.metadata.get("mutation"), task.id
        assert task.metadata.get("difficulty") in ("easy", "medium", "hard")


def test_every_task_has_a_failing_state_a_hidden_checker_and_a_reference():
    for task in collections()["suite"] + collections()["trio"]:
        assert task.entry in task.initial_files
        assert any(path.startswith("tests/") for path in task.hidden_files)
        assert task.reference_files
        reference = task.reference_solution()
        assert reference, task.id
        for path in reference:
            assert not path.startswith("tests/"), task.id


def test_the_manifest_matches_the_tree():
    ok, message = suite.check_manifest(SUITE_ROOT)
    assert ok, message
    manifest = suite.build_manifest(SUITE_ROOT)
    assert manifest["counts"]["total"] == 159
    assert manifest["counts"]["suite"] == 144
    assert manifest["counts"]["trio"] == 9
    assert manifest["counts"]["probes"] == 6
    assert len(manifest["suite_sha256"]) == 64
    hashes = [entry["tree_sha256"] for entry in manifest["tasks"]]
    assert len(hashes) == len(set(hashes))


def test_a_manifest_edit_is_detected(tmp_path):
    copy = tmp_path / "capability"
    shutil.copytree(SUITE_ROOT, copy)
    manifest = suite.build_manifest(str(copy))
    manifest["suite_sha256"] = "0" * 64
    suite.write_manifest(str(copy), manifest)
    ok, message = suite.check_manifest(str(copy))
    assert not ok
    assert "stale" in message


def test_the_prompt_does_not_contain_the_hidden_tests():
    import re

    for task in collections()["suite"] + collections()["trio"]:
        prompt = suite.render_prompt(task)
        assert task.initial_files[0] in prompt
        for relative in task.hidden_files:
            hidden = task.read_hidden(relative)
            for name in re.findall(r"#\[test\]\s*\n\s*fn\s+([a-z0-9_]+)\s*\(", hidden):
                assert not re.search(rf"fn\s+{re.escape(name)}\s*\(", prompt), \
                    (task.id, name)
        assert "path=src/lib.rs" in prompt


def test_the_prompt_is_deterministic():
    task = collections()["suite"][0]
    assert suite.render_prompt(task) == suite.render_prompt(task)
    assert suite.prompt_sha256(task) == suite.prompt_sha256(task)


def test_per_category_counts_are_even_enough_to_balance():
    manifest = suite.build_manifest(SUITE_ROOT)
    counts = manifest["counts"]["by_category"]
    assert set(counts) == {
        "lifetime-borrow", "failing-test", "api-migration", "dependency-compat",
        "concurrency", "trait-impl", "repo-scale", "semantic-repair", "trio",
        "probe"}
    for name in ("lifetime-borrow", "failing-test", "api-migration",
                 "dependency-compat", "concurrency", "trait-impl", "repo-scale",
                 "semantic-repair"):
        assert counts[name] == 18, name
    assert counts["trio"] == 9
    assert counts["probe"] == 6


def test_the_trio_annotates_the_shared_operation():
    trio = collections()["trio"]
    assert len(trio) == 9
    surfaces = {task.metadata["shared_operation"]["surface"] for task in trio}
    assert surfaces == {"rust", "numeric-graph", "logic"}
    operations = {task.metadata["shared_operation"]["name"] for task in trio}
    assert operations == {
        "dependency-closure minimal repair",
        "invariant preservation",
        "quantifier/scope resolution",
    }
    for surface in surfaces:
        assert len([task for task in trio
                    if task.metadata["shared_operation"]["surface"] == surface]) == 3
    for task in trio:
        operation = task.metadata["shared_operation"]
        assert operation["transfer_rule"]
        assert operation["audit"]
        items = task.metadata["operation_items"]
        assert len(items) >= 4
        assert all(item.get("operation") for item in items)
        hidden = "\n".join(task.read_hidden(path) for path in task.hidden_files)
        for item in items:
            assert f"fn {item['test']}" in hidden, (task.id, item["test"])


# ---------------------------------------------------------------------------
# extraction


def test_extracts_named_fenced_files():
    text = "intro\n```rust path=src/lib.rs\npub fn a() {}\n```\n"
    result = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked")
    assert result.ok
    assert result.files == {"src/lib.rs": "pub fn a() {}\n"}
    assert result.mode == "fenced_paths"


def test_extracts_a_single_unnamed_fence_as_the_entry_file():
    text = "```rust\npub fn b() -> u8 { 1 }\n```"
    result = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked")
    assert result.ok
    assert result.mode == "fenced_entry"


def test_full_text_fallback_requires_rust():
    result = extract.extract_files("pub fn c() {}", "src/lib.rs", {"src/lib.rs"}, "locked")
    assert result.ok and result.mode == "full_text"
    prose = extract.extract_files("I cannot help with that.", "src/lib.rs",
                                  {"src/lib.rs"}, "locked")
    assert not prose.ok and prose.mode == "invalid"


def test_the_harness_owned_paths_are_rejected():
    for path in ("tests/check.rs", "build.rs", ".cargo/config.toml", "../escape.rs",
                 "/etc/passwd"):
        text = f"```rust path={path}\nfn x() {{}}\n```"
        result = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked")
        assert not result.ok, path
    locked = extract.extract_files("```rust path=Cargo.toml\n[package]\n```",
                                   "src/lib.rs", {"src/lib.rs"}, "locked")
    assert not locked.ok
    arm = extract.extract_files("```rust path=Cargo.toml\n[package]\n```",
                                "src/lib.rs", {"src/lib.rs"}, "arm")
    assert arm.ok


def test_a_new_src_file_is_allowed_but_a_new_top_level_file_is_not():
    new_source = extract.extract_files("```rust path=src/extra.rs\nfn y() {}\n```",
                                       "src/lib.rs", {"src/lib.rs"}, "locked")
    assert new_source.ok
    new_top = extract.extract_files("```rust path=notes.txt\nhello\n```",
                                    "src/lib.rs", {"src/lib.rs"}, "locked")
    assert not new_top.ok


def test_duplicate_paths_are_rejected():
    text = ("```rust path=src/lib.rs\nfn a() {}\n```\n"
            "```rust path=src/lib.rs\nfn b() {}\n```\n")
    result = extract.extract_files(text, "src/lib.rs", {"src/lib.rs"}, "locked")
    assert not result.ok
    assert any("duplicate" in problem for problem in result.rejected)


def test_a_locked_manifest_cannot_be_checked_in(tmp_path):
    from capability import checkers

    task = next(task for task in collections()["suite"]
                if task.manifest_policy == "locked")
    assert "Cargo.toml" not in task.allowed_paths()
    _workdir, problems = checkers.prepare(
        task, {"src/lib.rs": "pub fn a() {}", "Cargo.toml": "[package]\n"},
        str(tmp_path))
    assert problems
    assert any("manifest" in problem for problem in problems)


def test_an_arm_managed_manifest_is_writable():
    task = next(task for task in collections()["suite"]
                if task.manifest_policy == "arm")
    assert "Cargo.toml" in task.allowed_paths()
    result = extract.extract_files("```rust path=Cargo.toml\n[package]\n```",
                                   task.entry, task.allowed_paths(), "arm")
    assert result.ok
