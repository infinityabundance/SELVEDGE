"""S51: eligibility of invalidated results in the status machinery.

The S50 invalidation receipt keeps every affected receipt verbatim but says no
number from those runs may be used as a bar, baseline or selection input. These
tests pin the status generator's three-way classification and the court's
refusal to accept an invalidated artefact cited as current.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load():
    path = os.path.join(ROOT, "scripts", "status_consistency.py")
    spec = importlib.util.spec_from_file_location("status_consistency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status = _load()

INVALIDATED = "results/stage2/eval/full-20m-math"
INVALIDATED_S49 = "results/stage2/eval/full-20m-math-s49"
ELIGIBLE = "results/stage2/eval/s50-bar-20m-lr1e-5"


def _facts() -> dict:
    return {
        "pilot_status": "done",
        "training_arms": [],
        "verdict_present": True,
        "capability_present": True,
        "arms": [],
        "evaluated_arms": [],
        "invalidations": {
            **status._empty_invalidations(),
            "paths": {INVALIDATED, INVALIDATED_S49},
            "evaluations": {INVALIDATED, INVALIDATED_S49},
            "masters": ["results/stage2/s50-invalidation-20261117T205855Z.json"],
        },
    }


def test_classification_is_three_way():
    invalidations = _facts()["invalidations"]
    assert status.classify_result(INVALIDATED, True, invalidations) == "invalidated"
    # a result file below an invalidated eval directory inherits the marker
    assert status.classify_result(
        INVALIDATED + "/m20/eval.json", True, invalidations) == "invalidated"
    assert status.classify_result(
        ELIGIBLE, True, invalidations, current_bar=ELIGIBLE) == "currently_eligible"
    assert status.classify_result(
        ELIGIBLE, True, invalidations) == "historically_scored"
    assert status.classify_result("/tmp/unscored", False, invalidations) == "unscored"


def test_load_invalidations_reads_the_master_record_and_markers(tmp_path):
    root = tmp_path
    master_rel = "results/stage2/s50-invalidation-20260101T000000Z.json"
    master = {
        "schema": "selvedge/s50-invalidation/v1",
        "kind": "s50-invalidation",
        "affected": {
            "runs": [{"path": "results/stage2/runs/a", "invalidated": True}],
            "evaluations": [{"path": INVALIDATED, "invalidated": True}],
            "selections": [{"path": "results/stage2/sel.json", "invalidated": True}],
        },
        "disposition": "retained, not a bar",
        "superseded_by": {"run": "results/stage2/runs/b"},
    }
    master_path = root / master_rel
    master_path.parent.mkdir(parents=True)
    master_path.write_text(json.dumps(master))
    marker_dir = root / "results/stage2/runs/a"
    marker_dir.mkdir(parents=True)
    (marker_dir / "INVALIDATED.json").write_text(
        json.dumps({"schema": "selvedge/s50-invalidation-marker/v1",
                    "invalidated": True}))
    loaded = status.load_invalidations(str(root))
    assert INVALIDATED in loaded["paths"]
    assert "results/stage2/runs/a" in loaded["runs"]
    assert "results/stage2/sel.json" in loaded["selections"]
    assert loaded["masters"] == [master_rel]
    assert loaded["superseded_by"]["run"] == "results/stage2/runs/b"


def test_invalidated_bar_cited_as_current_fails():
    rules = status.contradiction_rules(_facts())
    for line in (
        f"The current bar is `{INVALIDATED}`.",
        f"`{INVALIDATED}` is the current baseline.",
        f"Bar: {INVALIDATED}",
        f"The baseline is `full-20m-math-s49`.",
    ):
        problems = status.find_contradictions("results/REPORT.md", line, rules)
        assert problems, line
        assert "invalidated-not-current" in problems[0]


def test_historically_scored_label_passes():
    rules = status.contradiction_rules(_facts())
    text = (
        f"- historically_scored: `{INVALIDATED}` (stamp 20261117T120059Z); "
        "invalidated and retained as evidence only.\n"
        f"- The S49 bar (`{INVALIDATED_S49}`) is invalidated and is not the "
        "current bar."
    )
    assert status.find_contradictions("results/REPORT.md", text, rules) == []


def test_eligible_result_passes():
    rules = status.contradiction_rules(_facts())
    text = f"currently_eligible: `{ELIGIBLE}` (no invalidation receipt names it)."
    assert status.find_contradictions("results/REPORT.md", text, rules) == []


def test_eligibility_line_names_invalidated_scores_and_master_record():
    line = status._eligibility_line(_facts(), eligible_paths=[ELIGIBLE])
    assert "`currently_eligible`" in line
    assert INVALIDATED in line
    assert "historically_scored" in line
    assert "master record" in line
    assert "may not be cited as the current bar" in line


def test_no_invalidation_means_no_invalidated_rule():
    facts = {**_facts(), "invalidations": status._empty_invalidations()}
    rule_names = {rule["name"] for rule in status.contradiction_rules(facts)}
    assert "invalidated-not-current" not in rule_names
