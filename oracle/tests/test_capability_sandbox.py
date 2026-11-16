"""CPU tests for the bounded sandbox.

Runs `python3` children inside the python court, so it can verify that the
limits are applied in the child (the child prints its own `RLIMIT_AS`,
`RLIMIT_CPU`, `RLIMIT_FSIZE` and `RLIMIT_NOFILE`), that a timeout is killed and
recorded, that the environment is built rather than inherited, and that the
output digests are stable. The real Rust OOM abort is recorded in the
self-test receipt's `probe-oom` item; here the classification function is tested
directly for that case.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from capability import limits, sandbox  # noqa: E402


def test_the_court_environment_is_detected():
    assert sandbox.in_court()
    assert sandbox.network_isolation_evidence() == "no-default-route"


def test_classify_separates_every_outcome():
    assert sandbox.classify(0, False, "") == sandbox.STATUS_PASS
    assert sandbox.classify(None, True, "") == sandbox.STATUS_TIMEOUT
    assert sandbox.classify(-6, False, "memory allocation of 8388608 bytes failed") \
        == sandbox.STATUS_OOM
    assert sandbox.classify(-9, False, "memory allocation of 1 bytes failed") \
        == sandbox.STATUS_OOM
    assert sandbox.classify(101, False, "error[E0308]: mismatched types") \
        == sandbox.STATUS_COMPILE_ERROR
    assert sandbox.classify(101, False, "test result: FAILED. 1 failed") \
        == sandbox.STATUS_TEST_FAILED
    assert sandbox.classify(101, False, "thread 'x' panicked at src/lib.rs:1:1") \
        == sandbox.STATUS_TEST_FAILED
    assert sandbox.classify(2, False, "some other failure") == sandbox.STATUS_NONZERO
    # A solution that merely prints "out of memory" without dying is not OOM.
    assert sandbox.classify(1, False, "out of memory") == sandbox.STATUS_NONZERO


def test_the_child_sees_the_declared_limits(tmp_path):
    probe = limits.Limits(wall_seconds=10.0, cpu_seconds=20,
                          address_space_bytes=768 * 1024 * 1024,
                          file_bytes=4 * 1024 * 1024, open_files=37)
    program = (
        "import json, resource\n"
        "print(json.dumps({\n"
        "  'as': resource.getrlimit(resource.RLIMIT_AS)[0],\n"
        "  'cpu': resource.getrlimit(resource.RLIMIT_CPU)[0],\n"
        "  'fsize': resource.getrlimit(resource.RLIMIT_FSIZE)[0],\n"
        "  'nofile': resource.getrlimit(resource.RLIMIT_NOFILE)[0],\n"
        "  'tmpdir': __import__('os').environ.get('TMPDIR'),\n"
        "  'home': __import__('os').environ.get('HOME'),\n"
        "  'offline': __import__('os').environ.get('CARGO_NET_OFFLINE'),\n"
        "}))\n"
    )
    result = sandbox.run_bounded([sys.executable, "-c", program], str(tmp_path),
                                 probe, toolchain_floor=False)
    assert result.status == sandbox.STATUS_PASS
    child = result.stdout_tail
    assert str(probe.address_space_bytes) in child
    assert str(probe.cpu_seconds) in child
    assert str(probe.file_bytes) in child
    assert "37" in child
    assert str(tmp_path) in child
    assert '"offline": "true"' in child
    assert result.network_isolation == "no-default-route"
    assert result.workdir_bytes > 0
    assert result.limits["address_space_bytes"] == probe.address_space_bytes
    assert len(result.env_sha256) == 64


def test_the_environment_is_built_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("SELVEDGE_TEST_LEAK", "please-do-not-leak")
    result = sandbox.run_bounded(
        [sys.executable, "-c",
         "import os; print('SELVEDGE_TEST_LEAK' in os.environ)"],
        str(tmp_path), limits.RUN_ONLY_LIMITS, toolchain_floor=False)
    assert result.stdout_tail.strip() == "False"


def test_a_busy_child_is_killed_and_recorded_as_a_timeout(tmp_path):
    probe = limits.Limits(wall_seconds=1.0, cpu_seconds=26,
                          address_space_bytes=512 * 1024 * 1024,
                          file_bytes=4 * 1024 * 1024, open_files=64)
    result = sandbox.run_bounded(
        [sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path),
        probe, toolchain_floor=False)
    assert result.status == sandbox.STATUS_TIMEOUT
    assert result.timed_out is True
    assert result.signal == 9
    assert 900 <= result.duration_ms < 10_000


def test_an_allocation_past_the_bound_is_stopped(tmp_path):
    probe = limits.Limits(wall_seconds=10.0, cpu_seconds=26,
                          address_space_bytes=192 * 1024 * 1024,
                          file_bytes=4 * 1024 * 1024, open_files=64)
    program = (
        "blocks = []\n"
        "while True:\n"
        "    blocks.append(bytearray(8 * 1024 * 1024))\n")
    result = sandbox.run_bounded([sys.executable, "-c", program], str(tmp_path),
                                 probe, toolchain_floor=False)
    assert result.status in sandbox.FAIL_STATUSES
    assert result.timed_out is False
    assert result.duration_ms < 10_000
    assert result.limits["address_space_bytes"] == probe.address_space_bytes


def test_the_output_digest_is_stable(tmp_path):
    program = "print('deterministic output')"
    first = sandbox.run_bounded([sys.executable, "-c", program], str(tmp_path),
                                limits.RUN_ONLY_LIMITS, toolchain_floor=False)
    second = sandbox.run_bounded([sys.executable, "-c", program], str(tmp_path),
                                 limits.RUN_ONLY_LIMITS, toolchain_floor=False)
    assert first.stdout_sha256 == second.stdout_sha256
    assert first.stdout_tail == second.stdout_tail


def test_limits_fail_closed_on_missing_or_unsafe_values():
    with pytest.raises(limits.LimitError):
        limits.from_mapping({"wall_seconds": 0}, "test")
    with pytest.raises(limits.LimitError):
        limits.from_mapping({"wall_seconds": 10, "cpu_seconds": 5}, "test")
    with pytest.raises(limits.LimitError):
        limits.from_mapping({"address_space_bytes": 64 * 1024 * 1024}, "test")
    with pytest.raises(limits.LimitError):
        limits.from_mapping({"bogus": 1}, "test")
    # A run-only envelope may sit below the toolchain floor, because no compiler
    # runs in it.
    run_only = limits.run_only_from_mapping({"address_space_bytes": 64 * 1024 * 1024},
                                            "test")
    assert run_only.address_space_bytes == 64 * 1024 * 1024


def test_the_sandbox_refuses_to_run_outside_a_court(tmp_path, monkeypatch):
    monkeypatch.delenv("SELVEDGE_COURT_IMAGE_ID", raising=False)
    with pytest.raises(sandbox.SandboxError, match="outside a SELVEDGE court"):
        sandbox.run_bounded([sys.executable, "-c", "print(1)"], str(tmp_path),
                            limits.RUN_ONLY_LIMITS, toolchain_floor=False)


def test_a_default_route_refuses_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "network_isolation_evidence",
                        lambda: "default-route-present")
    with pytest.raises(sandbox.SandboxError, match="network isolation"):
        sandbox.run_bounded([sys.executable, "-c", "print(1)"], str(tmp_path),
                            limits.RUN_ONLY_LIMITS, toolchain_floor=False)
