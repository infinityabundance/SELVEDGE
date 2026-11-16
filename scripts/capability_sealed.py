#!/usr/bin/env python3
"""Freeze, verify and (once) score the sealed holdback (P14.7).

    python3 scripts/capability_sealed.py --freeze
    python3 scripts/capability_sealed.py --verify
    python3 scripts/capability_sealed.py --score --unlock-sealed \
        --sealed-sha256 <registered hash> --out results/capability/sealed-<stamp>.json

`--freeze` writes `fixtures/capability/SEALED.json` and refuses to overwrite it.
`--verify` recomputes every item hash and the sealed aggregate and fails closed
on drift. `--score` refuses unless the unlock flag and the registered hash are
supplied together; on acceptance it writes an unlock record before checking any
item, then verifies every sealed item's declared failing state, reference and
mutation. The evaluator may run `--score` in the rust court; the refusal path
needs no cargo.
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

from capability import checkers, sandbox, sealed, suite  # noqa: E402

SCHEMA_RECEIPT = "selvedge/capability-sealed-score/v1"
DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
DEFAULT_WORKROOT = "/work/target/capability-sealed-work"
DEFAULT_UNLOCK_DIR = os.path.join(REPO, "results", "capability")
INITIAL_STATUS = {
    "compile_error": sandbox.STATUS_COMPILE_ERROR,
    "test_failure": sandbox.STATUS_TEST_FAILED,
    "hidden_test_failure": sandbox.STATUS_TEST_FAILED,
    "timeout": sandbox.STATUS_TIMEOUT,
    "oom": sandbox.STATUS_OOM,
}


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


def write_json(path: str, payload: dict) -> str:
    absolute = path if os.path.isabs(path) else os.path.join(REPO, path)
    os.makedirs(os.path.dirname(absolute) or ".", exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, absolute)
    return absolute


def score(suite_root: str, unlock_record: dict, registration: dict, out: str,
          workroot: str, keep: bool) -> int:
    items = sealed.load_items(suite_root)
    os.makedirs(workroot, exist_ok=True)
    rows = []
    failures = []
    initial_ok = reference_ok = mutation_ok = 0
    for task in items:
        expected_kind = task.metadata["initial_failure"]["kind"]
        initial = checkers.run_initial(task, workroot, keep=keep)
        initial_matches = (initial["occurred_as_expected"]
                           and initial["status"] == INITIAL_STATUS[expected_kind])
        initial_ok += 1 if initial_matches else 0
        if not initial_matches:
            failures.append(f"{task.id}: the failing state did not fail as declared")

        reference = checkers.run_check(task, task.reference_solution(), workroot, keep=keep)
        reference_passed = reference["status"] == sandbox.STATUS_PASS
        reference_ok += 1 if reference_passed else 0
        if not reference_passed:
            failures.append(f"{task.id}: the reference did not pass ({reference['status']})")

        mutated, mutation = checkers.mutate_reference(task)
        wrong = checkers.run_check(task, mutated, workroot, keep=keep)
        mutation_failed = wrong["status"] != sandbox.STATUS_PASS
        mutation_ok += 1 if mutation_failed else 0
        if not mutation_failed:
            failures.append(f"{task.id}: the declared mutation passed the checker")

        rows.append({
            "task": task.id,
            "category": task.category,
            "tree_sha256": task.tree_sha256,
            "initial": {"expected_kind": expected_kind, "status": initial["status"]},
            "reference": {"status": reference["status"]},
            "mutation": dict(mutation, status=wrong["status"]),
            "passed": bool(initial_matches and reference_passed and mutation_failed),
        })
        print(f"capability_sealed: {task.id}: initial={initial['status']} "
              f"reference={reference['status']} mutation={wrong['status']}", flush=True)

    verdict = "pass" if not failures else "fail"
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "stamp": stamp_now(),
        "revision": revision(),
        "suite_root": "fixtures/capability",
        "sealed_sha256": registration["sealed_sha256"],
        "unlock": unlock_record,
        "items": rows,
        "checker": {
            "tasks_checked": len(rows),
            "initial_as_expected": initial_ok,
            "reference_passed": reference_ok,
            "mutation_failed": mutation_ok,
            "failures": failures,
        },
        "note": (
            "This is an instrument self-check of the sealed set, not a model "
            "evaluation. A model result is published once per revision and carries "
            "this unlock record."),
        "verdict": verdict,
    }
    path = write_json(out, receipt)
    print(f"capability_sealed: {verdict}; {initial_ok} initial, {reference_ok} "
          f"reference, {mutation_ok} mutation; receipt {os.path.relpath(path, REPO)}")
    return 0 if verdict == "pass" else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--unlock-sealed", dest="unlock_sealed", action="store_true")
    parser.add_argument("--sealed-sha256", dest="sealed_sha256", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--unlock-record", default=None)
    parser.add_argument("--workroot", default=DEFAULT_WORKROOT)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="allow a new freeze file over an existing one")
    args = parser.parse_args(argv)

    if not (args.freeze or args.verify or args.score):
        args.verify = True

    if args.freeze:
        manifest = suite.build_manifest(args.suite)
        payload = sealed.build_registration(args.suite,
                                            suite_sha256=manifest["suite_sha256"])
        try:
            path = sealed.write_registration(args.suite, payload,
                                             refuse_overwrite=not args.force)
        except sealed.SealedError as exc:
            print(f"capability_sealed: {exc}", file=sys.stderr)
            return 3
        print(f"capability_sealed: froze {payload['counts']['total']} items; "
              f"sealed {payload['sealed_sha256']}; registration "
              f"{os.path.relpath(path, REPO)}")
        return 0

    if args.verify:
        ok, message = sealed.check_registration(args.suite)
        if not ok:
            print(f"capability_sealed: {message}", file=sys.stderr)
            return 3
        registration = sealed.load_registration(args.suite)
        print(f"capability_sealed: sealed set intact; {registration['counts']['total']} "
              f"items, sealed {registration['sealed_sha256']}, frozen "
              f"{registration['frozen_utc']}")
        if not args.score:
            return 0

    registration = sealed.load_registration(args.suite)
    try:
        record = sealed.authorize(registration, args.unlock_sealed, args.sealed_sha256)
    except sealed.SealedError as exc:
        print(f"capability_sealed: refused: {exc}", file=sys.stderr)
        return 3

    if not args.out:
        print("capability_sealed: --score needs --out", file=sys.stderr)
        return 64
    unlock_path = args.unlock_record or os.path.join(
        DEFAULT_UNLOCK_DIR, f"sealed-unlock-{stamp_now()}.json")
    sealed.write_unlock_record(unlock_path, record)
    print(f"capability_sealed: unlocked {registration['sealed_sha256']} at "
          f"{record['unlocked_utc']}; record {os.path.relpath(unlock_path, REPO)}")
    return score(args.suite, record, registration, args.out, args.workroot, args.keep)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
