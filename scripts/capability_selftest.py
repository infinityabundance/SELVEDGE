#!/usr/bin/env python3
"""Self-test the fresh suite's executable checkers, and write the evidence.

For every suite task and every trio task this run proves:

* the failing state fails, with the diagnostic or assertion the task declares;
* the reference solution passes;
* a deliberately wrong solution (declared per task) fails.

For the five sandbox probes it proves the checker can record a pass, an
assertion failure, a compile error, a timeout and an allocation failure — each
as an outcome, none as a hang.

Everything runs inside the rust court (`cargo` and a standard-library Python in
one sandbox). The receipt is the evidence the docs quote:

    SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
        python3 scripts/capability_selftest.py \
        --out results/capability/selftest-$(date -u +%Y%m%dT%H%M%SZ).json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO, "oracle")
sys.path.insert(0, ORACLE)

from capability import checkers, sandbox, suite  # noqa: E402
from capability import limits as limit_module  # noqa: E402

SCHEMA = "selvedge/capability-selftest/v1"
DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
DEFAULT_WORKROOT = "/work/target/capability-work"
PROBE_STATUS = {
    "pass": sandbox.STATUS_PASS,
    "fail": sandbox.STATUS_TEST_FAILED,
    "compile_error": sandbox.STATUS_COMPILE_ERROR,
    "timeout": sandbox.STATUS_TIMEOUT,
    "oom": sandbox.STATUS_OOM,
}
INITIAL_STATUS = {
    "compile_error": sandbox.STATUS_COMPILE_ERROR,
    "test_failure": sandbox.STATUS_TEST_FAILED,
    "hidden_test_failure": sandbox.STATUS_TEST_FAILED,
    "timeout": sandbox.STATUS_TIMEOUT,
    "oom": sandbox.STATUS_OOM,
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    value = os.environ.get("SELVEDGE_REVISION", "")
    if value:
        return value
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json_atomic(path: str, payload: dict) -> None:
    absolute = os.path.join(REPO, path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(absolute) or ".", exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, absolute)


def summarise(run: dict) -> dict:
    evidence = run.get("evidence") or {}
    return {
        "status": run.get("status"),
        "problems": run.get("problems"),
        "exit_code": evidence.get("exit_code"),
        "signal": evidence.get("signal"),
        "duration_ms": evidence.get("duration_ms"),
        "stderr_sha256": evidence.get("stderr_sha256"),
        "stderr_tail": (evidence.get("stderr_tail") or "")[-600:],
        "address_space_bytes": (evidence.get("limits") or {}).get("address_space_bytes"),
        "wall_seconds": (evidence.get("limits") or {}).get("wall_seconds"),
        "cgroup_oom_kill_delta": evidence.get("cgroup_oom_kill_delta"),
        "cgroup_oom_delta": evidence.get("cgroup_oom_delta"),
        "cgroup_memory_peak_bytes": evidence.get("cgroup_memory_peak_bytes"),
        "workdir_bytes": evidence.get("workdir_bytes"),
        "phase": evidence.get("phase"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workroot", default=DEFAULT_WORKROOT)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--mem-budget", type=int, default=4096)
    parser.add_argument("--filter", default="",
                        help="only check tasks whose id contains this text")
    parser.add_argument("--category", default="",
                        help="only check tasks in this capability category")
    args = parser.parse_args(argv)

    court_mem = os.environ.get("SELVEDGE_COURT_MEM_BYTES")
    if not court_mem or int(court_mem) < args.mem_budget * 1024 * 1024:
        raise SystemExit(
            "capability_selftest: the court memory pin is missing or smaller than the "
            "declared budget; refusing to run (AGENTS.md directive 2)")

    manifest_ok, manifest_message = suite.check_manifest(args.suite)
    loaded = suite.load_suite(args.suite)
    counts = {
        "suite": len(loaded["suite"]),
        "trio": len(loaded["trio"]),
        "probes": len(loaded["probes"]),
        "total": sum(len(items) for items in loaded.values()),
        "by_category": {},
    }
    for kind in ("suite", "trio", "probes"):
        for task in loaded[kind]:
            counts["by_category"][task.category] = (
                counts["by_category"].get(task.category, 0) + 1)
            difficulty = task.metadata.get("difficulty", "medium")
            counts.setdefault("by_difficulty", {})
            counts["by_difficulty"][difficulty] = (
                counts["by_difficulty"].get(difficulty, 0) + 1)

    os.makedirs(args.workroot, exist_ok=True)
    workroot_budget = limit_module.DEFAULT_WORKDIR_BUDGET_BYTES
    max_workdir = 0
    items = []
    failures = []
    initial_ok = reference_ok = mutation_ok = probes_ok = 0
    naive_ok = naive_declared = 0
    selected = [task for task in loaded["suite"] + loaded["trio"]
                if (not args.filter or args.filter in task.id)
                and (not args.category or task.category == args.category)]

    for task in selected:
        initial = checkers.run_initial(task, args.workroot, keep=args.keep)
        expected_kind = task.metadata["initial_failure"]["kind"]
        initial_matches = (initial["occurred_as_expected"]
                           and initial["status"] == INITIAL_STATUS[expected_kind])
        initial_ok += 1 if initial_matches else 0
        if not initial_matches:
            failures.append(f"{task.id}: the failing state did not fail as declared "
                            f"(status {initial['status']}, expected {expected_kind}, "
                            f"missing {initial['missing_evidence']})")

        reference = checkers.run_check(task, task.reference_solution(), args.workroot,
                                       keep=args.keep)
        reference_passed = reference["status"] == sandbox.STATUS_PASS
        reference_ok += 1 if reference_passed else 0
        if not reference_passed:
            failures.append(f"{task.id}: the reference solution did not pass "
                            f"({reference['status']}: "
                            f"{(reference.get('evidence') or {}).get('stderr_tail', '')[-300:]})")

        build = checkers.run_build(task, task.reference_solution(), args.workroot,
                                   keep=args.keep)
        if build["status"] != sandbox.STATUS_PASS:
            failures.append(f"{task.id}: the reference did not build for the build-time "
                            f"measurement ({build['status']})")

        mutated, mutation = checkers.mutate_reference(task)
        wrong = checkers.run_check(task, mutated, args.workroot, keep=args.keep)
        mutation_failed = wrong["status"] != sandbox.STATUS_PASS
        mutation_ok += 1 if mutation_failed else 0
        if not mutation_failed:
            failures.append(f"{task.id}: the declared mutation passed the checker; "
                            f"the tests do not discriminate {mutation}")

        naive_record = None
        if task.metadata.get("naive"):
            naive_declared += 1
            shallow, naive_spec = checkers.naive_solution(task)
            shallow_build = checkers.run_build(task, shallow, args.workroot,
                                               keep=args.keep)
            shallow_run = checkers.run_check(task, shallow, args.workroot,
                                             keep=args.keep)
            naive_record = {
                "declaration": naive_spec,
                "build_status": shallow_build.get("status"),
                "check_status": shallow_run["status"],
                "rejected": shallow_run["status"] != sandbox.STATUS_PASS,
                "compiles": shallow_build.get("status") == sandbox.STATUS_PASS,
                "evidence": summarise(shallow_run),
            }
            naive_rejected = naive_record["rejected"] and naive_record["compiles"]
            naive_ok += 1 if naive_rejected else 0
            if not naive_rejected:
                failures.append(
                    f"{task.id}: the declared naive fix was not rejected after "
                    f"compiling (build {shallow_build.get('status')}, check "
                    f"{shallow_run['status']})")

        max_workdir = max(max_workdir, initial["evidence"]["workdir_bytes"] or 0)
        max_workdir = max(max_workdir, reference["evidence"]["workdir_bytes"] or 0)
        max_workdir = max(max_workdir, wrong["evidence"]["workdir_bytes"] or 0)
        max_workdir = max(max_workdir, build.get("workdir_bytes") or 0)
        if max_workdir > workroot_budget:
            failures.append(f"workdir budget exceeded: {max_workdir} > {workroot_budget}")
            workroot_budget = max_workdir * 2

        items.append({
            "task": task.id,
            "kind": task.kind,
            "category": task.category,
            "tree_sha256": task.tree_sha256,
            "checker_kind": task.checker_kind,
            "limits": task.limits.as_dict(),
            "initial": {
                "expected_kind": expected_kind,
                "status": initial["status"],
                "duration_ms": initial["duration_ms"],
                "must_contain": initial["must_contain"],
                "evidence": summarise(initial),
            },
            "reference": summarise(reference),
            "build": summarise({"evidence": build}),
            "mutation": dict(mutation, status=wrong["status"],
                             evidence=summarise(wrong)),
            "naive": naive_record,
            "passed": bool(initial_matches and reference_passed and mutation_failed
                           and (naive_record is None or naive_record["rejected"])),
        })
        print(f"capability_selftest: {task.id}: initial={initial['status']} "
              f"reference={reference['status']} mutation={wrong['status']}", flush=True)

    for task in loaded["probes"]:
        run = checkers.run_check(task, checkers.initial_solution(task), args.workroot,
                                 keep=args.keep)
        expected = PROBE_STATUS[task.metadata["probe_expectation"]]
        matches = run["status"] == expected
        forged = task.metadata.get("probe_stdout_contains")
        if forged:
            output = (run.get("evidence") or {}).get("stdout_tail") or ""
            matches = matches and forged in output
            if forged not in output:
                failures.append(f"{task.id}: the forged stdout marker {forged!r} was not "
                                f"observed, so the probe did not exercise the path")
        probes_ok += 1 if matches else 0
        if not matches:
            failures.append(f"{task.id}: expected {expected}, got {run['status']}")
        max_workdir = max(max_workdir, run["evidence"]["workdir_bytes"] or 0)
        items.append({
            "task": task.id,
            "kind": "probe",
            "category": "probe",
            "tree_sha256": task.tree_sha256,
            "checker_kind": task.checker_kind,
            "probe_expectation": task.metadata["probe_expectation"],
            "probe_stdout_contains": task.metadata.get("probe_stdout_contains"),
            "status": run["status"],
            "evidence": summarise(run),
            "passed": matches,
        })
        print(f"capability_selftest: {task.id}: {run['status']} "
              f"(expected {task.metadata['probe_expectation']})", flush=True)

    verdict = "pass" if not failures and initial_ok and reference_ok and mutation_ok else "fail"
    build_by_category: dict[str, int] = {}
    for item in items:
        duration = (item.get("build") or {}).get("duration_ms")
        if duration:
            category = item.get("category") or "unknown"
            build_by_category[category] = max(build_by_category.get(category, 0), duration)
    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": revision(),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "suite": {
            "root": "fixtures/capability",
            "suite_sha256": suite.build_manifest(args.suite)["suite_sha256"],
            "counts": counts,
            "manifest_matches": manifest_ok,
            "manifest_message": manifest_message,
            "partial": bool(args.filter or args.category),
            "filter": args.filter,
            "category_filter": args.category,
        },
        "checker": {
            "tasks_checked": len(items) - len(loaded["probes"]),
            "tasks_selected": len(selected),
            "initial_as_expected": initial_ok,
            "reference_passed": reference_ok,
            "mutation_failed": mutation_ok,
            "naive_declared": naive_declared,
            "naive_rejected": naive_ok,
            "probes_checked": len(loaded["probes"]),
            "probes_as_expected": probes_ok,
            "failures": failures,
        },
        "sandbox": {
            "court_image_id": os.environ.get("SELVEDGE_COURT_IMAGE_ID"),
            "court_memory_bytes": int(court_mem),
            "network_isolation": sandbox.network_isolation_evidence(),
            "default_task_limits": limit_module.Limits(
                limit_module.DEFAULT_TIMEOUT_SECONDS,
                limit_module.DEFAULT_CPU_SECONDS,
                limit_module.DEFAULT_ADDRESS_SPACE_BYTES,
                limit_module.DEFAULT_FILE_BYTES,
                limit_module.DEFAULT_OPEN_FILES).as_dict(),
            "run_only_limits": limit_module.RUN_ONLY_LIMITS.as_dict(),
            "toolchain_min_address_space_bytes": limit_module.TOOLCHAIN_MIN_ADDRESS_SPACE,
            "workdir_budget_bytes": limit_module.DEFAULT_WORKDIR_BUDGET_BYTES,
            "max_workdir_bytes": max_workdir,
            "max_check_ms": max(
                (item.get("reference", {}).get("duration_ms") or 0) for item in items),
            "max_build_ms": max(
                (item.get("build", {}).get("duration_ms") or 0) for item in items),
        },
        "items": items,
        "derived": {
            "suite_tasks": counts["suite"],
            "trio_tasks": counts["trio"],
            "probe_tasks": counts["probes"],
            "tasks_checked": len(items) - len(loaded["probes"]),
            "naive_declared": naive_declared,
            "naive_rejected": naive_ok,
            "checker_timeout_seconds": limit_module.DEFAULT_TIMEOUT_SECONDS,
            "checker_cpu_seconds": limit_module.DEFAULT_CPU_SECONDS,
            "checker_address_space_gib": (
                limit_module.DEFAULT_ADDRESS_SPACE_BYTES / (1024 ** 3)),
            "run_only_timeout_seconds": limit_module.RUN_ONLY_LIMITS.wall_seconds,
            "run_only_address_space_mib": (
                limit_module.RUN_ONLY_LIMITS.address_space_bytes / (1024 ** 2)),
            "max_check_ms": max(
                (item.get("reference", {}).get("duration_ms") or 0) for item in items),
            "max_build_ms": max(
                (item.get("build", {}).get("duration_ms") or 0) for item in items),
            "max_build_ms_by_category": dict(sorted(build_by_category.items())),
            "max_workdir_mib": max_workdir / (1024 ** 2),
            "court_memory_gib": int(court_mem) / (1024 ** 3),
            "workdir_budget_mib": workroot_budget / (1024 ** 2),
            "suite_sha256": suite.build_manifest(args.suite)["suite_sha256"],
        },
        "verdict": verdict,
    }
    write_json_atomic(args.out, receipt)
    print(f"capability_selftest: {verdict}; {initial_ok} initial, {reference_ok} "
          f"reference, {mutation_ok} mutation, {naive_ok}/{naive_declared} naive, "
          f"{probes_ok} probes; receipt {args.out}")
    if not manifest_ok:
        print(f"capability_selftest: {manifest_message}", file=sys.stderr)
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
