#!/usr/bin/env python3
"""Bounded subprocess execution for the capability court.

Every checker invocation goes through `run_bounded`, which enforces:

* a hard wall-clock timeout, with the whole process group killed on expiry
  (so a `cargo` that spawned `rustc` cannot outlive the run);
* `RLIMIT_AS` (virtual address space), `RLIMIT_CPU`, `RLIMIT_FSIZE`,
  `RLIMIT_NOFILE` and no core dumps, applied in the child before `exec`;
* bounded output: stdout/stderr go to files inside the workdir, which is bounded
  by `RLIMIT_FSIZE`; only a tail is kept in memory, and the digest covers the
  whole file;
* a deterministic environment built from scratch (never inherited), with
  `TMPDIR` inside the workdir so scratch is bounded;
* no inherited file descriptors (`close_fds`, `start_new_session`), stdin from
  `/dev/null`;
* a *measured* network-isolation statement: the container's route table is read
  and recorded, and a default route refuses the run unless explicitly allowed.

A timeout, an allocation failure and a non-zero exit are three different
outcomes; they are never conflated with "the test passed".
"""

from __future__ import annotations

import hashlib
import os
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field

from .limits import Limits

#: Outcome vocabulary. The court never invents a status outside this set.
STATUS_PASS = "pass"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_TEST_FAILED = "test_failed"
STATUS_TIMEOUT = "timeout"
STATUS_OOM = "oom"
STATUS_NONZERO = "nonzero_exit"
STATUS_INVALID = "invalid_solution"

PASS_STATUSES = frozenset({STATUS_PASS})
FAIL_STATUSES = frozenset({
    STATUS_COMPILE_ERROR, STATUS_TEST_FAILED, STATUS_TIMEOUT, STATUS_OOM,
    STATUS_NONZERO, STATUS_INVALID,
})


class SandboxError(RuntimeError):
    """The sandbox could not be established; no measurement is valid."""


@dataclass
class RunResult:
    status: str
    exit_code: int | None
    signal: int | None
    timed_out: bool
    duration_ms: int
    stdout_sha256: str
    stderr_sha256: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_tail: str
    stderr_tail: str
    limits: dict
    env_sha256: str
    network_isolation: str
    cgroup_oom_kill_delta: int | None
    cgroup_oom_delta: int | None
    cgroup_memory_peak_bytes: int | None
    workdir_bytes: int
    command: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "limits": self.limits,
            "env_sha256": self.env_sha256,
            "network_isolation": self.network_isolation,
            "cgroup_oom_kill_delta": self.cgroup_oom_kill_delta,
            "cgroup_oom_delta": self.cgroup_oom_delta,
            "cgroup_memory_peak_bytes": self.cgroup_memory_peak_bytes,
            "workdir_bytes": self.workdir_bytes,
            "command": self.command,
        }


# ---------------------------------------------------------------------------
# environment and isolation evidence


def deterministic_env(workdir: str, extra: dict | None = None) -> dict:
    """A complete environment, built rather than inherited.

    `HOME`, `TMPDIR` and `CARGO_HOME` live inside the workdir, so nothing a
    solution does can write outside the bounded tree. `RUSTUP_HOME` is the only
    inherited path, because the toolchain itself is installed there.
    """
    tmp = os.path.join(workdir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    environment = {
        "PATH": "/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": tmp,
        "TMP": tmp,
        "TEMP": tmp,
        "CARGO_HOME": os.path.join(workdir, ".cargo"),
        "CARGO_TARGET_DIR": os.path.join(workdir, "target"),
        "CARGO_NET_OFFLINE": "true",
        "CARGO_INCREMENTAL": "0",
        "CARGO_TERM_COLOR": "never",
        "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", "/usr/local/rustup"),
        "RUST_BACKTRACE": "0",
        "LC_ALL": "C",
        "LANG": "C",
        "TERM": "dumb",
        "TZ": "UTC",
        "SOURCE_DATE_EPOCH": "1700000000",
        "PYTHONHASHSEED": "0",
    }
    if extra:
        environment.update(extra)
    return environment


def env_sha256(environment: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(environment):
        digest.update(f"{key}={environment[key]}\n".encode("utf-8"))
    return digest.hexdigest()


def network_isolation_evidence() -> str:
    """Record whether the process namespace has a default route.

    The court container is launched with `--network none`; this is the measured
    corroboration, not a copy of the flag. `no-default-route` is the value the
    harness accepts; `default-route-present` fails closed.
    """
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()[1:]
    except OSError:
        return "route-table-unreadable"
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "00000000":
            return "default-route-present"
    return "no-default-route"


def in_court() -> bool:
    """True inside a SELVEDGE court container (docker-court sets this pin)."""
    return bool(os.environ.get("SELVEDGE_COURT_IMAGE_ID"))


# ---------------------------------------------------------------------------
# cgroup accounting


def _read_int(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_oom_kills() -> int | None:
    events = _cgroup_oom_events()
    return None if events is None else events[1]


def _cgroup_oom_events() -> tuple[int, int] | None:
    """(oom, oom_kill) counters from cgroup v2 `memory.events`."""
    path = "/sys/fs/cgroup/memory.events"
    counters = {"oom": None, "oom_kill": None}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2 and parts[0] in counters:
                    counters[parts[0]] = int(parts[1])
    except OSError:
        return None
    if None in counters.values():
        return None
    return (counters["oom"], counters["oom_kill"])


def _cgroup_peak() -> int | None:
    return _read_int("/sys/fs/cgroup/memory.peak")


# ---------------------------------------------------------------------------
# hashing helpers


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return "", 0
    return digest.hexdigest(), size


def _tail(path: str, limit: int) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def tree_bytes(root: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            try:
                total += os.lstat(path).st_size
            except OSError:
                continue
    return total


# ---------------------------------------------------------------------------
# the runner


def _apply_limits(limits: Limits):
    def apply() -> None:
        pair = (limits.address_space_bytes, limits.address_space_bytes)
        resource.setrlimit(resource.RLIMIT_AS, pair)
        resource.setrlimit(resource.RLIMIT_CPU,
                           (limits.cpu_seconds, limits.cpu_seconds + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def classify(returncode: int | None, timed_out: bool, stderr_text: str) -> str:
    """Map a process outcome to the court's status vocabulary.

    Allocation failure is detected from the allocator's own message *and* an
    abort/kill signal, so a solution that merely prints "out of memory" is not
    misclassified.
    """
    if timed_out:
        return STATUS_TIMEOUT
    if returncode == 0:
        return STATUS_PASS
    lowered = stderr_text.lower()
    allocation = ("memory allocation of" in lowered
                  or "out of memory" in lowered
                  or "cannot allocate memory" in lowered)
    if allocation and (returncode in (-6, -9, 134, 137) or returncode is None):
        return STATUS_OOM
    if ("test result: failed" in lowered or "panicked at" in lowered
            or "error: test failed" in lowered):
        return STATUS_TEST_FAILED
    if "could not compile" in lowered or "error[e" in lowered:
        return STATUS_COMPILE_ERROR
    if lowered.startswith("error"):
        return STATUS_COMPILE_ERROR
    return STATUS_NONZERO


def run_bounded(argv: list, cwd: str, limits: Limits, drop_privileges: bool = True,
                allow_default_route: bool = False,
                env_extra: dict | None = None,
                toolchain_floor: bool = True) -> RunResult:
    """Run `argv` in `cwd` under `limits` and return the recorded outcome."""
    limits.validate("run_bounded", toolchain_floor=toolchain_floor)
    if drop_privileges and not in_court():
        raise SandboxError(
            "refusing to run outside a SELVEDGE court container; "
            "set SELVEDGE_COURT_IMAGE_ID only by using scripts/court")
    isolation = network_isolation_evidence()
    if isolation != "no-default-route" and not allow_default_route:
        raise SandboxError(
            f"network isolation is not proven ({isolation}); refusing to run")

    os.makedirs(cwd, exist_ok=True)
    environment = deterministic_env(cwd, env_extra)
    stdout_path = os.path.join(cwd, ".stdout")
    stderr_path = os.path.join(cwd, ".stderr")
    for path in (stdout_path, stderr_path):
        with open(path, "wb"):
            pass

    oom_before = _cgroup_oom_events()
    peak_before = _cgroup_peak()
    began = time.monotonic_ns()
    timed_out = False
    returncode: int | None = None
    signum: int | None = None

    with open(stdout_path, "wb") as out_handle, open(stderr_path, "wb") as err_handle:
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment,
            stdin=subprocess.DEVNULL, stdout=out_handle, stderr=err_handle,
            close_fds=True, start_new_session=True,
            preexec_fn=_apply_limits(limits),
        )
        try:
            returncode = process.wait(timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                process.kill()
            returncode = process.wait()
    duration_ms = (time.monotonic_ns() - began) // 1_000_000
    if returncode is not None and returncode < 0:
        signum = -returncode

    stdout_digest, stdout_size = _sha256_file(stdout_path)
    stderr_digest, stderr_size = _sha256_file(stderr_path)
    stderr_tail = _tail(stderr_path, limits.output_bytes)
    stdout_tail = _tail(stdout_path, limits.output_bytes)
    status = classify(returncode, timed_out, stderr_tail)

    oom_after = _cgroup_oom_events()
    peak_after = _cgroup_peak()
    oom_delta = None
    oom_kill_delta = None
    if oom_after is not None and oom_before is not None:
        oom_delta = oom_after[0] - oom_before[0]
        oom_kill_delta = oom_after[1] - oom_before[1]
    peak = None
    if peak_after is not None:
        peak = peak_after
        if peak_before is not None and peak_after < peak_before:
            peak = peak_after  # the counter was reset; report the current value

    # The two files are evidence only; keeping them would make every workdir
    # grow with compiler output. The digests and tails are what is recorded.
    workdir_bytes = tree_bytes(cwd)
    for path in (stdout_path, stderr_path):
        try:
            os.unlink(path)
        except OSError:
            pass

    return RunResult(
        status=status,
        exit_code=returncode,
        signal=signum,
        timed_out=timed_out,
        duration_ms=duration_ms,
        stdout_sha256=stdout_digest,
        stderr_sha256=stderr_digest,
        stdout_bytes=stdout_size,
        stderr_bytes=stderr_size,
        stdout_tail=stdout_tail[-4000:],
        stderr_tail=stderr_tail[-8000:],
        limits=limits.as_dict(),
        env_sha256=env_sha256(environment),
        network_isolation=isolation,
        cgroup_oom_kill_delta=oom_kill_delta,
        cgroup_oom_delta=oom_delta,
        cgroup_memory_peak_bytes=peak,
        workdir_bytes=workdir_bytes,
        command=[os.path.basename(argv[0])] + [str(arg) for arg in argv[1:]],
    )


def copy_tree(source: str, destination: str) -> None:
    """Copy a fixture directory into a fresh workdir."""
    if not os.path.isdir(source):
        raise SandboxError(f"fixture directory {source} is missing")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
