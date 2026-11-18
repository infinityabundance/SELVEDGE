"""S56 — status discovery must find the milestone score receipts.

The S51 status block looked for one `eval.json` at the root of the run's eval
directory. The milestone driver writes `m5/eval.json`, `m10/eval.json` and
`m20/eval.json`, so the block claimed "no anchored score exists yet" while three
scored receipts sat on disk. These tests pin the corrected behaviour:

* every complete, stamped, step-matching milestone is discovered and eligible;
* the endpoint is the highest eligible milestone, not a hard-coded path;
* an invalidated milestone is excluded (never the endpoint, never eligible);
* a receipt whose checkpoint step the launch did not record is not a score;
* a document that says the bar is unscored while an eligible milestone exists
  fails closed with the `anchored-score-exists` contradiction.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import status_consistency as sc  # noqa: E402

STEPS = (5, 10, 20)


def _write(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _milestone_receipt(step: int, stamp: str) -> dict:
    return {
        "schema": "selvedge/stage2-arm-eval/v1",
        "complete": True,
        "stamp": stamp,
        "revision": "a" * 40,
        "arm_report": {"checkpoint_step": step,
                       "checkpoint": f"results/stage2/runs/s50-bar-20m-lr1e-5/ckpt-{step:06d}.pt"},
        "tasks": [
            {"task": "gsm8k", "all_metrics": {"exact_match,flexible-extract": 0.4}},
            {"task": "minerva_math500", "all_metrics": {"exact_match,none": 0.1}},
            {"task": "mmlu", "all_metrics": {"acc,none": 0.5}},
            {"task": "mmlu_pro", "all_metrics": {"exact_match,custom-extract": 0.2}},
        ],
    }


def _repo(tmp_path) -> str:
    """A disposable repository with the current bar's receipts and documents."""
    root = str(tmp_path)
    _write(os.path.join(root, "results/t2/pilot-state.json"), {
        "schema": "selvedge/t2-pilot-state/v1",
        "status": "done",
        "started_utc": "2026-01-01T00:00:00Z",
        "finished_utc": "2026-01-02T00:00:00Z",
        "stamp": "20260102T000000Z",
        "arms": {"D": {"status": "evaluated"}},
    })
    _write(os.path.join(root, sc.CURRENT_BAR_LAUNCH), {
        "schema": "selvedge/s50-bar-launch/v1",
        "created_utc": "2026-01-03T00:00:00Z",
        "run_dir": sc.CURRENT_BAR_REL,
        "checkpoints": {"milestones": list(STEPS)},
        "stream": {"sha256": "b" * 64},
    })
    _write(os.path.join(root, sc.CURRENT_BAR_MANIFEST), {
        "schema": "selvedge/stage2-train-config/v1",
        "steps": 20,
        "tokens": 200,
        "tokens_per_step": 10,
        "stream_sha256": "b" * 64,
    })
    with open(os.path.join(root, sc.CURRENT_BAR_LOG), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 19}) + "\n")
        handle.write(json.dumps({"step": 20}) + "\n")
    for step in STEPS:
        _write(os.path.join(root, sc.CURRENT_BAR_EVAL_ROOT, f"m{step}", "eval.json"),
               _milestone_receipt(step, f"20260104T0000{step:02d}Z"))
    for doc in ("README.md", "results/REPORT.md"):
        path = os.path.join(root, doc)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if doc == "README.md":
            sentinels = ("readme-status",)
        else:
            sentinels = ("report-status", "report-t2-arm-state")
        with open(path, "w", encoding="utf-8") as handle:
            for sentinel in sentinels:
                handle.write(f"<!-- BEGIN GENERATED: {sentinel} -->\n")
                handle.write("stale\n")
                handle.write(f"<!-- END GENERATED: {sentinel} -->\n")
            handle.write("\nProse.\n")
    return root


def test_all_milestones_are_discovered_and_eligible(tmp_path):
    root = _repo(tmp_path)
    facts = sc.load_facts(root)
    ws19 = facts["ws19"]
    assert ws19["state"] == "scored"
    assert [item["label"] for item in ws19["milestones"]] == ["m5", "m10", "m20"]
    assert all(item["state"] == "eligible" for item in ws19["milestones"])
    # The endpoint is the highest eligible milestone, by its own step.
    assert ws19["endpoint"]["label"] == "m20"
    assert ws19["eval_path"] == "results/stage2/eval/s50-bar-20m-lr1e-5/m20/eval.json"
    assert ws19["score_state"] == "currently_eligible"
    assert ws19["milestones"][0]["tokens"] == 50  # step * tokens_per_step


def test_readme_names_the_milestones_and_not_an_absent_score(tmp_path):
    root = _repo(tmp_path)
    facts = sc.load_facts(root)
    text = sc.render_readme_status(facts)
    assert "currently_eligible" in text
    assert "`m20`" in text
    assert "`m5`" in text and "`m10`" in text
    assert "no anchored score exists" not in text
    assert "`results/stage2/eval/s50-bar-20m-lr1e-5/m20/eval.json`" in text


def test_generated_blocks_check_clean(tmp_path):
    root = _repo(tmp_path)
    assert sc.write(root) == 0
    assert sc.check(root) == 0


def test_invalidated_milestone_is_excluded_from_the_endpoint(tmp_path):
    root = _repo(tmp_path)
    _write(os.path.join(root, sc.CURRENT_BAR_EVAL_ROOT, "m20", "INVALIDATED.json"),
           {"invalidated": True, "reason": "test fixture"})
    facts = sc.load_facts(root)
    ws19 = facts["ws19"]
    states = {item["label"]: item["state"] for item in ws19["milestones"]}
    assert states["m20"] == "invalidated"
    assert states["m10"] == "eligible"
    assert ws19["endpoint"]["label"] == "m10"
    assert ws19["eval_path"].endswith("m10/eval.json")
    assert "m20/eval.json" not in sc._eligible_paths(ws19)
    assert any(path.endswith("m10/eval.json") for path in sc._eligible_paths(ws19))


def test_unrecorded_checkpoint_step_is_not_a_score(tmp_path):
    root = _repo(tmp_path)
    # A stray directory with a receipt for a step the launch never recorded.
    _write(os.path.join(root, sc.CURRENT_BAR_EVAL_ROOT, "m30", "eval.json"),
           _milestone_receipt(30, "20260104T000030Z"))
    ws19 = sc.load_facts(root)["ws19"]
    states = {item["label"]: item["state"] for item in ws19["milestones"]}
    assert states["m30"] == "incomplete"
    assert ws19["endpoint"]["label"] == "m20"


def test_incomplete_milestone_is_not_eligible(tmp_path):
    root = _repo(tmp_path)
    payload = _milestone_receipt(10, "20260104T000010Z")
    payload["complete"] = False
    _write(os.path.join(root, sc.CURRENT_BAR_EVAL_ROOT, "m10", "eval.json"), payload)
    ws19 = sc.load_facts(root)["ws19"]
    states = {item["label"]: item["state"] for item in ws19["milestones"]}
    assert states["m10"] == "incomplete"
    assert [item["label"] for item in ws19["eligible_milestones"]] == ["m5", "m20"]


def test_scored_milestone_but_status_says_none_fails(tmp_path):
    """The regression: a scored milestone exists and a document says it does not."""
    root = _repo(tmp_path)
    facts = sc.load_facts(root)
    assert facts["ws19"]["eligible_milestones"]
    rules = sc.contradiction_rules(facts)
    assert "anchored-score-exists" in {rule["name"] for rule in rules}
    problems = sc.find_contradictions(
        "README.md",
        "Prose.\n\nNo anchored score exists yet; no milestone is scored.\n",
        rules,
    )
    assert problems, "a scored milestone must make 'no anchored score' a contradiction"
    assert any("anchored-score-exists" in problem for problem in problems)


def test_in_flight_claim_contradicts_a_scored_bar(tmp_path):
    root = _repo(tmp_path)
    facts = sc.load_facts(root)
    problems = sc.find_contradictions(
        "README.md",
        "Prose.\n\nThe corrected bar `s50-bar-20m-lr1e-5` is in flight.\n",
        sc.contradiction_rules(facts),
    )
    assert problems and any("anchored-score-exists" in problem for problem in problems)
