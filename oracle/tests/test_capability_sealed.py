"""CPU tests for the sealed holdback's custody and scoring gate (P14.7).

These run in the python court: the refusal path must not need cargo, because the
refusal is the property being tested. The scoring path itself is exercised by
`scripts/capability_sealed.py --score` in the rust court.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, ORACLE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from capability import sealed, suite  # noqa: E402

SUITE_ROOT = os.path.join(REPO, "fixtures", "capability")


def registration() -> dict:
    return sealed.load_registration(SUITE_ROOT)


def test_the_sealed_set_has_the_registered_shape():
    items = sealed.load_items(SUITE_ROOT)
    assert len(items) == 24
    by_category = {}
    for task in items:
        by_category[task.category] = by_category.get(task.category, 0) + 1
    assert set(by_category) == {
        "lifetime-borrow", "failing-test", "api-migration", "dependency-compat",
        "concurrency", "trait-impl", "repo-scale", "semantic-repair"}
    assert set(by_category.values()) == {3}
    suite_ids = {task.id for task in suite.load_suite(SUITE_ROOT)["suite"]}
    assert not suite_ids & {task.id for task in items}
    for task in items:
        assert task.metadata["license"] == "MIT OR Apache-2.0"
        assert task.metadata["initial_failure"]["must_contain"]
        assert task.metadata.get("mutation")


def test_the_registration_matches_the_tree():
    ok, message = sealed.check_registration(SUITE_ROOT)
    assert ok, message
    payload = registration()
    assert payload["counts"]["total"] == 24
    assert len(payload["sealed_sha256"]) == 64
    assert len({entry["tree_sha256"] for entry in payload["items"]}) == 24
    assert payload["freeze_date"] == payload["frozen_utc"][:10]


def test_the_registration_records_the_policy_and_the_unlock_procedure():
    payload = registration()
    assert "may not be used for tuning" in payload["rule"]
    assert payload["unlock"]["one_shot"] is True
    assert payload["unlock"]["may_not_be_used_for_tuning"] is True
    assert payload["unlock"]["flag"] == sealed.UNLOCK_FLAG
    assert payload["unlock"]["sealed_sha256"] == payload["sealed_sha256"]
    assert payload["sealed_sha256"] in payload["unlock_procedure"]
    assert "--unlock-sealed" in payload["unlock_procedure"]
    assert "--sealed-sha256" in payload["unlock_procedure"]


def test_the_registration_is_append_only():
    payload = registration()
    with pytest.raises(sealed.SealedError):
        sealed.write_registration(SUITE_ROOT, payload)


def test_authorize_refuses_without_the_flag():
    with pytest.raises(sealed.SealedError) as caught:
        sealed.authorize(registration(), False, registration()["sealed_sha256"])
    assert "--unlock-sealed" in str(caught.value)


def test_authorize_refuses_without_the_hash():
    with pytest.raises(sealed.SealedError):
        sealed.authorize(registration(), True, None)


def test_authorize_refuses_a_wrong_hash():
    with pytest.raises(sealed.SealedError):
        sealed.authorize(registration(), True, "0" * 64)


def test_authorize_accepts_the_flag_and_the_registered_hash():
    payload = registration()
    record = sealed.authorize(payload, True, payload["sealed_sha256"],
                              now="2026-11-16T00:00:00Z")
    assert record["unlocked"] is True
    assert record["unlocked_utc"] == "2026-11-16T00:00:00Z"
    assert record["flag"] == sealed.UNLOCK_FLAG
    assert record["sealed_sha256"] == payload["sealed_sha256"]
    assert record["one_shot"] is True


def test_the_unlock_record_is_written_before_scoring(tmp_path):
    payload = registration()
    record = sealed.authorize(payload, True, payload["sealed_sha256"])
    path = sealed.write_unlock_record(str(tmp_path / "unlock.json"), record)
    with open(path, "r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["unlocked"] is True
    assert written["unlocked_utc"]
    assert written["schema"] == sealed.SCHEMA_UNLOCK


def test_tampering_with_a_sealed_item_is_detected(tmp_path):
    copy = tmp_path / "capability"
    shutil.copytree(SUITE_ROOT, copy)
    item = sealed.load_items(str(copy))[0]
    target = os.path.join(item.root, "initial", "src", "lib.rs")
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n// tampered\n")
    ok, message = sealed.check_registration(str(copy))
    assert not ok
    assert "differ from the freeze" in message or "aggregate" in message


def test_an_extra_sealed_item_is_detected(tmp_path):
    copy = tmp_path / "capability"
    shutil.copytree(SUITE_ROOT, copy)
    extra = os.path.join(copy, "sealed", "sealed-extra")
    shutil.copytree(sealed.load_items(str(copy))[0].root, extra)
    metadata = os.path.join(extra, "task.json")
    with open(metadata, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["id"] = "sealed-extra"
    with open(metadata, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    ok, message = sealed.check_registration(str(copy))
    assert not ok
    assert "not recorded" in message


# ---------------------------------------------------------------------------
# the scoring script refuses without the flag and the hash


def _script():
    import capability_sealed

    return capability_sealed


def test_the_score_command_refuses_without_the_flag(tmp_path, capsys):
    out = tmp_path / "sealed-score.json"
    code = _script().main(["--score", "--out", str(out)])
    assert code == 3
    assert not out.exists()
    assert "refused" in capsys.readouterr().err


def test_the_score_command_refuses_a_wrong_hash(tmp_path, capsys):
    out = tmp_path / "sealed-score.json"
    code = _script().main(["--score", "--unlock-sealed",
                           "--sealed-sha256", "0" * 64, "--out", str(out)])
    assert code == 3
    assert not out.exists()
    assert "does not match" in capsys.readouterr().err


def test_the_score_command_refuses_a_missing_hash(tmp_path, capsys):
    out = tmp_path / "sealed-score.json"
    code = _script().main(["--score", "--unlock-sealed", "--out", str(out)])
    assert code == 3
    assert not out.exists()
    assert "sealed-sha256" in capsys.readouterr().err


def test_the_verify_command_accepts_the_frozen_tree(capsys):
    assert _script().main(["--verify"]) == 0
    assert "intact" in capsys.readouterr().out
