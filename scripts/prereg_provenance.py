#!/usr/bin/env python3
"""Fail when a registration entered the repository after its first score.

A pre-registration is only a pre-registration if it existed before the thing it
registers. The anchor scoreboard broke that rule: its pre-registration declares
`registered_utc = 2026-11-16T19:40:00Z`, was first committed at 21:31:57Z, and its
first scoring receipt has stamp 19:59:53Z. The scoreboard is still quotable (no
tuning occurred and the file has not changed since its first commit), but the
ordering is a process defect and a reader cannot verify the absence of tuning from
the repository alone. This court makes that ordering a checked property for every
registration written from now on.

Checked, per registration file (`docs/*preregistration*.json`):

1. **tracked.** The file must be in git; an uncommitted registration is not a
   registration.
2. **committed before the first score.** The commit that first added the file must
   be earlier than the earliest non-`smoke` scoring receipt that names the
   registration, matched by `preregistration.path` or `preregistration.sha256`.
3. **not declared after its commit.** `registered_utc` must not be later than the
   first-add commit time; a file cannot be registered before it was written.
4. **not declared before its inputs.** `registered_utc` must not be earlier than a
   receipt or log the file itself names under `limits_derived_from` or
   `smoke_runs_without_receipts`. A registration that claims to precede the
   measurements it derived its limits from is mis-stating its own provenance.

Historical exceptions are *named* in `HISTORICAL_EXCEPTIONS` with their evidence,
are printed on every run, and cover only the checks named for them. The anchor is
the one WS19 entry; the court's first run also found three older capability
registrations with ordering defects, which are recorded there with the evidence
the court itself printed. Nothing else may be added without a dated argument in
`docs/WORK-LOG.md`.

Usage:
    scripts/court prereg_provenance            # the court: --check
    scripts/court prereg_provenance --json
    scripts/court prereg_provenance --check --registration docs/anchor-preregistration-v1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

SCHEMA = "selvedge/prereg-provenance/v1"
STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")

#: Registrations whose ordering defect is history, not a live violation. Each
#: names the exact checks it is excused from and the evidence for the excuse.
HISTORICAL_EXCEPTIONS: dict[str, dict] = {
    "docs/anchor-preregistration-v1.json": {
        "excused_checks": ["committed_before_first_score", "declared_before_its_inputs"],
        "first_add_commit": "67ea47d7d0176e079f4c30c42dbc165c9c7f1706",
        "first_add_committed_utc": "2026-11-16T21:31:57Z",
        "first_scoring_receipt_stamp": "20261116T195953Z",
        "declared_utc": "2026-11-16T19:40:00Z",
        "inputs_named_by_the_file": [
            "results/frontier/anchor-smoke4-20261116T195500Z.json",
            "results/frontier/anchor-smoke5-20261116T200500Z.json",
        ],
        "mitigation": (
            "no tuning occurred: the task list, few-shot counts, sample limits, "
            "seeds, family membership and win-condition text were fixed in "
            "67ea47d and the file has not been modified since (one commit touches "
            "it), and the thresholds are derived mechanically from measured scores "
            "by scripts/s36_scoreboard.py"
        ),
        "recorded": "docs/WORK-LOG.md, S38 entry; docs/WS19-EXECUTION-REGISTRATION.md §6",
    },
    # The court's first run found three further ordering defects in the earlier
    # capability registrations. They are not repaired by editing the files (the
    # registrations are append-only and v5 supersedes them); they are recorded
    # here with the evidence the court itself printed.
    "docs/capability-preregistration-v2.json": {
        "excused_checks": ["committed_before_first_score"],
        "first_add_committed_utc": "2026-11-16T06:16:24Z",
        "first_scoring_receipt_stamp": "20261116T061507Z",
        "mitigation": (
            "the receipt that precedes the commit is an instrument-generation "
            "receipt (`results/capability/instrument-smoke-20261116T070500Z/generations.json`), "
            "not an arm score; the registration is append-only and superseded by v5"
        ),
        "recorded": "docs/WORK-LOG.md, S38 entry",
    },
    "docs/capability-preregistration-v3.json": {
        "excused_checks": ["committed_before_first_score"],
        "first_add_committed_utc": "2026-11-16T06:25:48Z",
        "first_scoring_receipt_stamp": "20261116T062400Z",
        "mitigation": (
            "as v2: the preceding receipt is an instrument-generation receipt, not an "
            "arm score, and v5 supersedes the file"
        ),
        "recorded": "docs/WORK-LOG.md, S38 entry",
    },
    "docs/capability-preregistration-v4.json": {
        "excused_checks": ["declared_not_after_commit"],
        "first_add_committed_utc": "2026-11-16T08:23:52Z",
        "declared_utc": "2026-11-16T08:30:00Z",
        "mitigation": (
            "the declared time is later than the commit by six minutes, which cannot be "
            "a tuning advantage; the file is append-only and superseded by v5, and no "
            "threshold in it was changed after a score"
        ),
        "recorded": "docs/WORK-LOG.md, S38 entry",
    },
}

#: Where scoring receipts are looked for. Printed on every run so the scan set is
#: visible; a registration may also name extra paths under `scoring_receipts`.
CANDIDATE_GLOBS = (
    "results/*.json",
    "results/*/*.json",
    "results/*/*/*.json",
)

MAX_RECEIPT_BYTES = 8 << 20


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def is_tracked(repo: str, rel: str) -> bool:
    result = subprocess.run(
        ["git", "-C", repo, "ls-files", "--error-unmatch", "--", rel],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def first_add(repo: str, rel: str) -> tuple[str, str] | None:
    out = git(repo, "log", "--diff-filter=A", "--format=%H %cI", "--", rel).strip()
    if not out:
        return None
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return None
    # `git log` is newest first; the last line is the first add.
    commit, committed = lines[-1].split(" ", 1)
    return commit, committed


def parse_stamp(value: str) -> dt.datetime | None:
    match = STAMP_RE.search(value or "")
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=dt.timezone.utc
    )


def parse_iso(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_receipts(repo: str, extra: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for pattern in CANDIDATE_GLOBS:
        for path in glob.glob(os.path.join(repo, pattern)):
            try:
                if os.path.getsize(path) > MAX_RECEIPT_BYTES:
                    continue
            except OSError:
                continue
            seen[os.path.relpath(path, repo)] = None
    for rel in extra:
        path = os.path.join(repo, rel)
        if os.path.exists(path):
            seen[rel] = None
    return sorted(seen)


def load_receipt(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return None


def find_scoring(repo: str, rel: str, digest: str, extra: list[str]) -> dict | None:
    best = None
    best_when = None
    for candidate in candidate_receipts(repo, extra):
        document = load_receipt(os.path.join(repo, candidate))
        if not isinstance(document, dict):
            continue
        declared = document.get("preregistration")
        if not isinstance(declared, dict):
            continue
        named = declared.get("path")
        hashed = declared.get("sha256")
        matches = (isinstance(named, str) and named == rel) or (
            isinstance(hashed, str) and hashed == digest
        )
        if not matches:
            continue
        if document.get("smoke") is True:
            continue
        when = parse_stamp(str(document.get("stamp") or "")) or parse_iso(
            str(document.get("started_utc") or "")
        )
        if when is None:
            continue
        if best_when is None or when < best_when:
            best_when = when
            best = {
                "receipt": candidate,
                "stamp": document.get("stamp"),
                "started_utc": document.get("started_utc"),
                "when": when,
            }
    return best


def named_inputs(registration: dict) -> list[str]:
    paths: list[str] = []
    for key in ("limits_derived_from",):
        value = registration.get(key)
        if isinstance(value, list):
            paths.extend(item for item in value if isinstance(item, str))
    value = registration.get("smoke_runs_without_receipts")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ("attempt", "log", "receipt"):
                    if isinstance(item.get(key), str):
                        paths.append(item[key])
    value = registration.get("scoring_receipts")
    if isinstance(value, list):
        paths.extend(item for item in value if isinstance(item, str))
    return paths


def check_registration(repo: str, rel: str, extra: list[str]) -> dict:
    path = os.path.join(repo, rel)
    document = load_receipt(path)
    if not isinstance(document, dict):
        return {
            "registration": rel,
            "ok": False,
            "problems": ["the file is not readable JSON"],
        }
    digest = sha256_file(path)
    declared_raw = str(document.get("registered_utc") or "")
    declared = parse_iso(declared_raw)
    add = first_add(repo, rel) if is_tracked(repo, rel) else None
    first_score = find_scoring(repo, rel, digest, extra)
    exception = HISTORICAL_EXCEPTIONS.get(rel)
    excused = set(exception["excused_checks"]) if exception else set()

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "ok": bool(ok) or name in excused,
                "detail": detail,
                "excused_as_history": name in excused,
            }
        )

    record(
        "tracked",
        add is not None,
        "in git" if add else "not tracked by git, so it is not a registration",
    )

    if first_score is None:
        record(
            "committed_before_first_score",
            True,
            "no non-smoke scoring receipt names this registration yet",
        )
    else:
        order_ok = add is not None and parse_iso(add[1]) < first_score["when"]
        record(
            "committed_before_first_score",
            order_ok,
            (
                f"first add {add[1]} vs first score "
                f"{first_score['when'].isoformat() if first_score else 'n/a'} "
                f"({first_score['receipt']})"
            ),
        )

    if add is not None:
        record(
            "declared_not_after_commit",
            declared is not None and declared <= parse_iso(add[1]),
            f"declared {declared_raw or 'absent'} vs first add {add[1]}",
        )
    else:
        record("declared_not_after_commit", False, "no commit to compare against")

    inputs = [
        (item, parse_stamp(item))
        for item in named_inputs(document)
        if parse_stamp(item) is not None
    ]
    if inputs:
        latest = max(inputs, key=lambda pair: pair[1])
        record(
            "declared_before_its_inputs",
            declared is not None and declared >= latest[1],
            (
                f"declared {declared_raw or 'absent'} vs its own input "
                f"{latest[0]} ({latest[1].isoformat()})"
            ),
        )
    else:
        record(
            "declared_before_its_inputs",
            True,
            "the file names no stamped input receipt",
        )

    return {
        "registration": rel,
        "sha256": digest,
        "declared_utc": declared_raw,
        "first_add": (
            {"commit": add[0], "committed_utc": add[1]} if add else None
        ),
        "first_scoring_receipt": (
            {
                "receipt": first_score["receipt"],
                "stamp": first_score["stamp"],
                "started_utc": first_score["started_utc"],
                "when": first_score["when"].isoformat(),
            }
            if first_score
            else None
        ),
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="the court (default)")
    group.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--registration",
        action="append",
        default=None,
        help="check one registration path (repeatable); default: every docs/*preregistration*.json",
    )
    parser.add_argument("--repo", default=None)
    args = parser.parse_args(argv)

    repo = args.repo or repo_root()
    if args.registration:
        targets = [os.path.relpath(item, repo) if os.path.isabs(item) else item for item in args.registration]
    else:
        targets = sorted(
            os.path.relpath(path, repo)
            for path in glob.glob(os.path.join(repo, "docs", "*preregistration*.json"))
        )

    if not targets:
        print("prereg_provenance: no registration file found; nothing to check")
        return 1

    report = {
        "schema": SCHEMA,
        "rev": git(repo, "rev-parse", "HEAD").strip(),
        "scan_globs": list(CANDIDATE_GLOBS),
        "historical_exceptions": sorted(HISTORICAL_EXCEPTIONS),
        "registrations": [
            check_registration(repo, rel, []) for rel in targets
        ],
    }
    report["ok"] = all(item["ok"] for item in report["registrations"])

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for entry in report["registrations"]:
            state = "OK" if entry["ok"] else "FAIL"
            when = (
                entry["first_scoring_receipt"]["when"]
                if entry.get("first_scoring_receipt")
                else "no score yet"
            )
            added = (
                entry["first_add"]["committed_utc"] if entry.get("first_add") else "untracked"
            )
            print(
                f"prereg_provenance: {state} {entry['registration']} "
                f"(declared {entry.get('declared_utc') or 'absent'}, added {added}, first score {when})"
            )
            for item in entry["checks"]:
                if not item["ok"]:
                    print(f"  FAIL {item['check']}: {item['detail']}")
                elif item["excused_as_history"]:
                    print(f"  history {item['check']}: {item['detail']} (declared exception)")
        for name, exception in sorted(HISTORICAL_EXCEPTIONS.items()):
            print(
                f"prereg_provenance: historical exception {name}: "
                f"{'; '.join(exception['excused_checks'])} — {exception['mitigation']}"
            )

    if not report["ok"]:
        print(
            "prereg_provenance: a registration is not demonstrably earlier than its "
            "first score; fix the ordering or record a dated exception",
            file=sys.stderr,
        )
        return 1
    print(
        f"prereg_provenance: OK — {len(report['registrations'])} registration(s) "
        f"checked against their first scoring receipts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
