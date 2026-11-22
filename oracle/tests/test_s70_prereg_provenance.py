"""S70: the retrospective label on a pre-registration is checked, not asserted.

`scripts/prereg_provenance.py` fails when a registration entered the repository
after the first receipt that names it. The S68 depth-ramp registration is such a
case (first add 2026-11-21T03:12:54Z, first named receipt 03:12:24Z). Deleting
the evidence would hide the defect and weakening the guard would let a tuned
registration through, so the file carries a *retrospective* label instead. The
court must then:

* accept the label only when it quotes the observed ordering, matches git and
  the receipts as they are now, and proves from the verdict receipt that no
  result was adopted;
* fail closed when the retrospective registration did produce an adopted
  result, or when the label cannot prove non-adoption (a missing indicator
  field is not proof);
* fail closed when a compliant registration carries the label, because that is
  itself a misstatement.

The tests drive `check_registration` with the git-derived and receipt-derived
lookups monkeypatched, so the decision logic is exercised without depending on
the real history. One test then checks the real S68 label against the real
history and the real verdict receipt.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prereg_provenance as prereg  # noqa: E402

S68 = "docs/s68-depth-ramp-preregistration-v1.json"
ADD_COMMIT = "b23c395ce5f01bc6e60d0d91fde4b35d1bed41eb"
ADD_UTC = "2026-11-21T03:12:54+00:00"
SCORE_STAMP = "20261121T031224Z"
SCORE_RECEIPT = "results/stage2/s68-ramp/build.json"


def _write(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _registration(retrospective=None, declared="2026-11-21T03:10:10Z"):
    document = {
        "schema": "selvedge/synthetic-preregistration/v1",
        "identifier": "SYNTHETIC",
        "registered_utc": declared,
        "status": "registered",
    }
    if retrospective is not None:
        document["retrospective"] = retrospective
    return document


def _retro_block(**overrides):
    block = {
        "is_retrospective": True,
        "reason": (
            f"first add {ADD_UTC} but the first named receipt {SCORE_STAMP} "
            "came earlier"
        ),
        "observed": {
            "first_add_commit": ADD_COMMIT,
            "first_add_committed_utc": ADD_UTC,
            "first_scoring_receipt": SCORE_RECEIPT,
            "first_scoring_receipt_stamp": SCORE_STAMP,
        },
        "evidence_paths": [SCORE_RECEIPT, "results/synthetic/verdict.json"],
        "content_sha256_before_label": "0" * 64,
        "substantive_content_changed": False,
        "disposition": {
            "verdict_receipts": ["results/synthetic/verdict.json"],
            "adoption_indicators": [
                {"receipt": "results/synthetic/verdict.json",
                 "field": "verdict.beats_champion_gsm8k"},
                {"receipt": "results/synthetic/verdict.json",
                 "field": "verdict.beats_champion_math500"},
            ],
            "adopted_if": "any indicator is true",
            "selection_changed": False,
        },
    }
    block.update(overrides)
    return block


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A synthetic tree with the git and receipt lookups pinned."""
    _write(str(tmp_path / SCORE_RECEIPT), {"stamp": SCORE_STAMP, "kind": "build"})
    _write(str(tmp_path / "results/synthetic/verdict.json"),
           {"verdict": {"beats_champion_gsm8k": False,
                        "beats_champion_math500": False}})
    state = {
        "add": (ADD_COMMIT, ADD_UTC),
        "score": {
            "receipt": SCORE_RECEIPT,
            "stamp": SCORE_STAMP,
            "started_utc": None,
            "when": prereg.parse_stamp(SCORE_STAMP),
        },
    }
    monkeypatch.setattr(prereg, "is_tracked", lambda repo_path, rel: True)
    monkeypatch.setattr(prereg, "first_add",
                        lambda repo_path, rel: state["add"])
    monkeypatch.setattr(prereg, "find_scoring",
                        lambda repo_path, rel, digest, extra: state["score"])
    return tmp_path, state


def _check(tmp_path, document):
    rel = "docs/synthetic-preregistration-v1.json"
    _write(str(tmp_path / rel), document)
    return prereg.check_registration(str(tmp_path), rel, [])


def test_retrospective_registration_that_changed_no_selection_passes(repo):
    tmp_path, _ = repo
    result = _check(tmp_path, _registration(_retro_block()))
    assert result["ok"], result["checks"]
    assert result["retrospective"]["labelled"] is True
    assert result["retrospective"]["ok"] is True
    assert result["retrospective"]["adopted"] == []
    checks = {item["check"]: item["ok"] for item in result["checks"]}
    assert checks["retrospective_not_adopted"] is True
    # The ordering defect is still visible in the report: the check passes only
    # because of the label, and the detail says so.
    assert checks["committed_before_first_score"] is True
    detail = next(item["detail"] for item in result["checks"]
                  if item["check"] == "committed_before_first_score")
    assert "retrospectively labelled and not adopted" in detail


def test_retrospective_registration_that_was_adopted_fails_closed(repo):
    tmp_path, _ = repo
    _write(str(tmp_path / "results/synthetic/verdict.json"),
           {"verdict": {"beats_champion_gsm8k": True,
                        "beats_champion_math500": False}})
    result = _check(tmp_path, _registration(_retro_block()))
    assert result["ok"] is False
    assert result["retrospective"]["ok"] is False
    assert result["retrospective"]["adopted"] == [
        "verdict.beats_champion_gsm8k=True in results/synthetic/verdict.json"
    ]
    assert "produced an adopted result" in result["retrospective"]["detail"]


def test_retrospective_registration_with_a_missing_indicator_fails_closed(repo):
    tmp_path, _ = repo
    # A missing field must not be read as "false": non-adoption has to be
    # proven from the receipt.
    _write(str(tmp_path / "results/synthetic/verdict.json"),
           {"verdict": {"beats_champion_gsm8k": False}})
    result = _check(tmp_path, _registration(_retro_block()))
    assert result["ok"] is False
    assert any("is absent from" in problem
               for problem in result["retrospective"]["problems"])


def test_retrospective_reason_must_quote_the_ordering(repo):
    tmp_path, _ = repo
    block = _retro_block(reason="it was registered late, sorry")
    result = _check(tmp_path, _registration(block))
    assert result["ok"] is False
    assert any("does not quote the first add instant" in problem
               for problem in result["retrospective"]["problems"])
    assert any("does not quote the first score instant" in problem
               for problem in result["retrospective"]["problems"])


def test_retrospective_label_must_match_the_observed_history(repo):
    tmp_path, _ = repo
    block = _retro_block()
    block["observed"] = dict(block["observed"],
                             first_add_committed_utc="2026-11-21T00:00:00+00:00")
    result = _check(tmp_path, _registration(block))
    assert result["ok"] is False
    assert any("observed.first_add_committed_utc" in problem
               for problem in result["retrospective"]["problems"])


def test_retrospective_label_on_a_compliant_registration_fails_closed(repo):
    tmp_path, state = repo
    # The registration is added before its first receipt: the ordering is
    # correct, so the label would be a misstatement.
    state["add"] = (ADD_COMMIT, "2026-11-21T00:00:00+00:00")
    result = _check(tmp_path, _registration(_retro_block()))
    assert result["ok"] is False
    assert any("committed before its first score" in problem
               for problem in result["retrospective"]["problems"])


def test_a_plain_ordering_defect_still_fails(repo):
    tmp_path, _ = repo
    result = _check(tmp_path, _registration())
    assert result["ok"] is False
    checks = {item["check"]: item["ok"] for item in result["checks"]}
    assert checks["committed_before_first_score"] is False
    assert "retrospective_not_adopted" not in checks


def test_the_s68_label_matches_the_real_history_and_verdict():
    """The real file's label is not a free-text excuse."""
    document = prereg.load_receipt(os.path.join(ROOT, S68))
    block = document["retrospective"]
    assert block["is_retrospective"] is True
    add = prereg.first_add(ROOT, S68)
    assert add is not None
    assert block["observed"]["first_add_commit"] == add[0]
    assert prereg.parse_iso(block["observed"]["first_add_committed_utc"]) == \
        prereg.parse_iso(add[1])
    # The first named receipt is the builder receipt, and the label says so.
    receipt = prereg.load_receipt(os.path.join(ROOT, SCORE_RECEIPT))
    assert receipt["preregistration"]["path"] == S68
    assert block["observed"]["first_scoring_receipt_stamp"] == receipt["stamp"]
    assert receipt["created_utc"] < add[1].replace("+00:00", "Z")
    # Every adoption indicator is present and false in the verdict receipt, so
    # the ramp changed no champion selection.
    verdict = prereg.load_receipt(
        os.path.join(ROOT, "results/stage2/s68-ramp/verdict.json"))
    for indicator in block["disposition"]["adoption_indicators"]:
        value, present = prereg.resolve_field(verdict, indicator["field"])
        assert present, indicator["field"]
        assert value is False, indicator["field"]
    assert block["disposition"]["selection_changed"] is False
