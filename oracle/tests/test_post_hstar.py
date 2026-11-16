"""CPU tests for the post-`H*` pipeline (`scripts/post_hstar.py`).

The pipeline is exercised end to end against a miniature repository and a
scripted court (`oracle/tests/fake_court.py`). The scripted court executes the
two CPU readers for real and writes the receipts the next stage verifies, so a
test failure means the wiring is wrong, not that a stub agreed with itself.

Covered, each named:
  * a full run in stage order, with every stage's memory limit recorded;
  * resumability after a mid-stage failure (nothing before it re-runs);
  * preflight refusal when the GPU is busy, when `MemAvailable` cannot cover the
    largest stage, and when a `selvedge-` container is alive;
  * verify refusal for a missing checkpoint and for a missing training receipt
    with no pending court state;
  * the verdict stage stopping on a missing receipt (exit 3, no verdict written);
  * `--dry-run` launching nothing.

These run in the python court, so the court launcher used here is a fixture by
design; the production launcher (`scripts/court`) is the only path that can touch
Docker.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import fake_court  # noqa: E402
import post_hstar  # noqa: E402


def dump(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def calls(root: str) -> list:
    path = os.path.join(root, ".fake-court/calls.jsonl")
    if not os.path.isfile(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def state(root: str) -> dict:
    return json.load(open(os.path.join(root, "results/t2/post-hstar-state.json"),
                          encoding="utf-8"))


def run(root: str, environment, *extra: str) -> int:
    argv = ["--repo-root", root, "--court", os.path.join(HERE, "fake_court.py"),
            "--court-interpreter", sys.executable,
            "--runner", "direct", "--host-probe",
            os.path.join(root, "host-probe.json"), *extra]
    return post_hstar.main(argv)


@pytest.fixture()
def environment(tmp_path, monkeypatch):
    root = str(tmp_path)
    built = fake_court.build_repo(root)
    monkeypatch.setenv("FAKE_COURT_ROOT", root)
    return built


def test_pipeline_runs_every_stage_in_order_and_records_limits(environment):
    root = environment["root"]
    assert run(root, environment, "--resume") == 0
    recorded = state(root)
    assert recorded["status"] == "done"
    for stage in post_hstar.STAGES:
        assert recorded["stages"][stage]["status"] == "done", stage
    tokens = [entry["token"] for entry in calls(root)]
    assert tokens[0] == "eval:H*"
    assert tokens[1] == "instrument:"
    assert tokens[2:12] == ["generate:D", "check:D", "generate:M", "check:M",
                            "generate:X", "check:X", "generate:H", "check:H",
                            "generate:H*", "check:H*"]
    assert tokens[12:] == ["eval:X", "eval:H", "verdict:", "compare:"]
    for entry in calls(root):
        limit = int(str(entry["SELVEDGE_MEM"]).rstrip("g"))
        assert limit >= 4, entry
        if entry["kind"] in ("eval", "generate", "check", "instrument"):
            assert limit >= 12, entry
        if entry["kind"] == "generate" and entry["arm"] in ("H", "H*"):
            assert limit == 24, entry
    assert os.path.isfile(os.path.join(root, "results/t2-verdict.json"))
    assert os.path.isfile(os.path.join(root, "results/T2-VERDICT.md"))
    comparison = json.load(open(os.path.join(
        root, "results/capability/runs/t2-comparison.json"), encoding="utf-8"))
    assert comparison["gate"]["passes"] is True
    verdict = json.load(open(os.path.join(root, "results/t2-verdict.json"),
                             encoding="utf-8"))
    assert verdict["outcome"] == "hybrid_supported"
    assert recorded["stages"]["capability"]["arms"]["H*"]["host_memory_limit_gib"] >= 24
    # Every completed stage carries the required provenance fields.
    for stage in post_hstar.STAGES:
        entry = recorded["stages"][stage]
        assert entry["revision"]
        if stage != "preflight":
            assert entry["host_memory_limit_gib"] > 0
            assert entry["sizing"]


def test_pipeline_resumes_after_a_mid_stage_failure_without_repeating_work(
        environment):
    root = environment["root"]
    os.makedirs(os.path.join(root, ".fake-court"), exist_ok=True)
    with open(os.path.join(root, ".fake-court/fail-once"), "w",
              encoding="utf-8") as handle:
        handle.write("eval:H*")
    assert run(root, environment, "--resume") == 9
    failed = state(root)
    assert failed["status"] == "failed"
    assert failed["failure"]["stage"] == "eval"
    assert "exit 9" in failed["failure"]["reason"]
    assert failed["stages"]["verify"]["status"] == "done"
    assert not os.path.isfile(os.path.join(root, "results/t2-verdict.json"))
    calls_after_failure = len(calls(root))
    assert calls_after_failure == 1

    assert run(root, environment, "--resume") == 0
    resumed = state(root)
    assert resumed["status"] == "done"
    assert resumed["failures"], "the previous failure must be archived, not deleted"
    assert resumed["failures"][0]["stage"] == "eval"
    tokens = [entry["token"] for entry in calls(root)]
    assert tokens.count("eval:H*") == 2
    assert tokens.count("generate:X") == 1
    assert tokens.count("check:X") == 1
    assert resumed["stages"]["verify"]["started_utc"] == \
        failed["stages"]["verify"]["started_utc"]


def test_preflight_refuses_when_a_heavy_gpu_process_is_running(environment, capsys):
    root = environment["root"]
    dump(os.path.join(root, "host-probe.json"), {
        "memory": {"available_gib": 90.0},
        "gpu": {"available": True, "used_mib": 9000, "total_mib": 16376,
                "compute_apps": [{"pid": "1", "name": "python3",
                                  "used_mib": 9000}]},
        "containers": [],
    })
    assert run(root, environment, "--resume") == 3
    assert calls(root) == []
    assert "preflight refused" in state(root)["failure"]["reason"]
    assert not os.path.isfile(os.path.join(root, "results/t2-verdict.json"))


def test_preflight_refuses_when_memavailable_cannot_cover_the_largest_stage(
        environment):
    root = environment["root"]
    dump(os.path.join(root, "host-probe.json"), {
        "memory": {"available_gib": 10.0},
        "gpu": {"available": True, "used_mib": 0, "total_mib": 16376,
                "compute_apps": []},
        "containers": [],
    })
    assert run(root, environment, "--resume") == 3
    assert calls(root) == []
    receipt = [name for name in os.listdir(os.path.join(root, "results/t2"))
               if name.startswith("post-hstar-preflight-")]
    assert receipt, "the arithmetic must be written even when it refuses"
    payload = json.load(open(os.path.join(root, "results/t2", receipt[0]),
                             encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["arithmetic"]["fits"] is False


def test_preflight_refuses_when_a_selvedge_container_is_running(environment):
    root = environment["root"]
    dump(os.path.join(root, "host-probe.json"), {
        "memory": {"available_gib": 90.0},
        "gpu": {"available": True, "used_mib": 0, "total_mib": 16376,
                "compute_apps": []},
        "containers": ["selvedge-oracle-court-20261116T050158Z-739076"],
    })
    assert run(root, environment, "--resume") == 3
    assert calls(root) == []
    assert "selvedge containers are running" in state(root)["failure"]["reason"]


def test_verify_refuses_a_missing_checkpoint(environment):
    root = environment["root"]
    os.unlink(os.path.join(root, "results/artifacts/train/t2/Hstar/"
                                 "step-001024.pt"))
    assert run(root, environment, "--resume", "--stage", "verify") == 1
    assert calls(root) == []
    assert "does not exist" in state(root)["failure"]["reason"]


def test_verify_refuses_a_missing_training_receipt_without_pending_state(
        environment):
    root = environment["root"]
    os.unlink(os.path.join(root, "results/t2/H*/run.json"))
    dump(os.path.join(root, "results/t2/pilot-state.json"), {"arms": {}})
    assert run(root, environment, "--resume", "--stage", "verify") == 1
    assert "no pending court state" in state(root)["failure"]["reason"]


def test_verdict_stage_stops_when_a_required_receipt_is_missing(environment):
    root = environment["root"]
    os.unlink(os.path.join(root, "results/eval/t2-D.json"))
    # The H* evaluation normally precedes the verdict; run it first so the only
    # defect is the missing D receipt.
    assert run(root, environment, "--resume", "--stage", "eval") == 0
    assert run(root, environment, "--resume", "--stage", "verdict") == 3
    entry = state(root)["stages"]["verdict"]
    assert entry["status"] == "failed"
    assert entry["failure"]["exit_code"] == 3
    assert not os.path.isfile(os.path.join(root, "results/t2-verdict.json"))


def test_capability_stage_stops_when_a_generate_fails(environment):
    root = environment["root"]
    os.makedirs(os.path.join(root, ".fake-court"), exist_ok=True)
    with open(os.path.join(root, ".fake-court/fail-once"), "w",
              encoding="utf-8") as handle:
        handle.write("generate:X")
    assert run(root, environment, "--resume", "--stage", "capability") == 9
    entry = state(root)["stages"]["capability"]
    assert entry["status"] == "failed"
    assert "exit 9" in entry["failure"]["reason"]
    assert [record["token"] for record in calls(root)].count("generate:X") == 1
    assert not os.path.isfile(os.path.join(
        root, "results/capability/runs/t2-X/result.json"))


def test_dry_run_launches_nothing(environment):
    root = environment["root"]
    assert run(root, environment, "--resume", "--dry-run") == 0
    assert calls(root) == []
    dry = json.load(open(os.path.join(
        root, "results/t2/post-hstar-state.dry-run.json"), encoding="utf-8"))
    assert dry["status"] == "done"
    assert not os.path.isfile(os.path.join(
        root, "results/t2/post-hstar-state.json")), \
        "a dry run must not leave state a real resume would skip"


def test_every_container_stage_uses_the_image_interpreter(environment):
    """A host absolute interpreter path does not exist inside the image.

    The first real run of this pipeline failed the instrument stage with exit
    127 because the argv carried the host's `/usr/bin/python3`. The scripted
    court never noticed, since it reads the script name and ignores the
    interpreter, so the regression is asserted here on the recorded argv.
    """
    root = environment["root"]
    assert run(root, environment, "--resume") == 0
    recorded = calls(root)
    assert recorded
    for entry in recorded:
        argv = entry["argv"]
        inner = argv[argv.index("--") + 1:]
        assert inner, entry
        assert inner[0] == "python3", entry
        assert not inner[0].startswith("/"), entry


def test_memory_sizing_is_derived_from_a_measured_peak(environment):
    root = environment["root"]
    sizing = post_hstar.size_stage(root, "capability-generate", "H*")
    assert sizing["arithmetic"]["measured_peak_gib"] == 10.81
    assert sizing["arithmetic"]["required_gib"] == pytest.approx(16.215)
    assert sizing["limit_gib"] == 24
    small = post_hstar.size_stage(root, "capability-generate", "D")
    assert small["limit_gib"] == 16
    assert small["arithmetic"]["floor_gib"] == 16


def test_the_systemd_unit_receives_the_limit_in_its_environment(tmp_path):
    """`systemd-run` does not inherit the caller's environment.

    Without --setenv the court fell back to its own 8g default while the state
    recorded 16g/24g, so the declared limit was never applied (S27).
    """
    runner = post_hstar.Runner(str(tmp_path), "court", mode="systemd",
                               log_path=str(tmp_path / "log"))
    record = runner.run(["python3", "x.py"], mem_gib=24, unit="u", dry_run=True)
    wrapped = record["wrapped"]
    assert "--setenv=SELVEDGE_MEM=24g" in wrapped, wrapped
    assert "--setenv=SELVEDGE_TMP=8g" in wrapped, wrapped
    assert "--property=MemoryMax=24G" in wrapped, wrapped


def test_a_mismatched_applied_limit_fails_closed(tmp_path):
    log = tmp_path / "log"
    log.write_text("court: oracle image=x memory=8g cpus=8 pids=1024 tmp=4g\n",
                   encoding="utf-8")
    runner = post_hstar.Runner(str(tmp_path), "court", mode="systemd",
                               log_path=str(log))
    with pytest.raises(post_hstar.StageFailure) as failure:
        runner._verify_applied_memory({"mem_gib": 16}, 0)
    assert "applied 8 GiB" in failure.value.reason


def test_a_matching_applied_limit_is_recorded(tmp_path):
    log = tmp_path / "log"
    log.write_text("court: oracle image=x memory=16g cpus=8\n", encoding="utf-8")
    runner = post_hstar.Runner(str(tmp_path), "court", mode="systemd",
                               log_path=str(log))
    record = {"mem_gib": 16}
    runner._verify_applied_memory(record, 0)
    assert record["applied_memory_gib"] == 16


def test_a_systemd_stage_that_reports_no_limit_fails_closed(tmp_path):
    log = tmp_path / "log"
    log.write_text("nothing about limits here\n", encoding="utf-8")
    runner = post_hstar.Runner(str(tmp_path), "court", mode="systemd",
                               log_path=str(log))
    with pytest.raises(post_hstar.StageFailure) as failure:
        runner._verify_applied_memory({"mem_gib": 16}, 0)
    assert "no applied memory limit" in failure.value.reason
