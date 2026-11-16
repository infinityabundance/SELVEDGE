#!/usr/bin/env python3
"""Row-level consistency check for docs/PRIOR-ART-AND-NOVELTY.md.

`scripts/prior_art_fetch.py` fetches sources; it does not audit the ledger. This
script does. It parses the numbered rows of the verification tables (§12 named
works and §12.3 supplementary works), extracts each row's declared status, and
fails closed if:

  * a §12 row has no recognised status;
  * a row's status cell still says "not located"/"title-only"/"identity-only"
    while the status claims VERIFIED (i.e. a correction was half-applied);
  * the tally disagrees with EXPECTED (§1 prose must match the rows).

Usage: python3 scripts/prior_art_audit.py [--doc PATH]
Exit 0 on success, 1 on drift. No network. Bounded; no writes.
"""
from __future__ import annotations

import argparse
import re
import sys

EXPECTED_TABLE12 = {"VERIFIED": 12, "PARTIAL": 2, "UNSUPPORTED": 6}
EXPECTED_SUPPLEMENTARY = {"VERIFIED": 1}
STATUSES = ("VERIFIED", "PARTIAL", "UNSUPPORTED")

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|(.+)$")
STATUS_RE = re.compile(r"\\*\\*(VERIFIED|PARTIAL|UNSUPPORTED)\\*\\*")
NEGATIVE_IN_STATUS = re.compile(r"not located|title-only|identity-only", re.I)
SUPERSEDED_OK = re.compile(r"supersede", re.I)


def parse_rows(lines: list[str], start_marker: str, stop_marker: str) -> list[tuple[int, str]]:
    """Return (row number, whole row text) for numbered rows between markers."""
    active = False
    rows: list[tuple[int, str]] = []
    for line in lines:
        if start_marker in line:
            active = True
            continue
        if active and stop_marker in line:
            break
        if not active:
            continue
        m = ROW_RE.match(line)
        if m:
            rows.append((int(m.group(1)), line))
    return rows


def status_of(row: str) -> str | None:
    # The status cell is column 4 for §12.1 rows and column 5 for §12.2 rows
    # (which add a "nearest prior art" column). Searching the whole row is
    # unsafe because evidence text quotes old verdicts, so check both cells.
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    for idx in (3, 4):
        if idx >= len(cells):
            continue
        m = STATUS_RE.search(cells[idx])
        if m:
            return m.group(1)
    return None


def tally(rows: list[tuple[int, str]]) -> dict[str, int]:
    counts = {s: 0 for s in STATUSES}
    problems: list[str] = []
    for num, row in rows:
        st = status_of(row)
        if st is None:
            problems.append(f"row {num}: no recognised status in column 4")
            continue
        counts[st] += 1
        if st == "VERIFIED" and NEGATIVE_IN_STATUS.search(row) and not SUPERSEDED_OK.search(row):
            problems.append(
                f"row {num}: status VERIFIED but row still asserts a negative "
                "verdict without marking it superseded"
            )
    return counts, problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", default="docs/PRIOR-ART-AND-NOVELTY.md")
    args = ap.parse_args(argv)

    with open(args.doc, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    table12 = parse_rows(lines, "## 12. Verification table", "### 12.3")
    supp = parse_rows(lines, "### 12.3", "## 13.")
    if not table12:
        print(f"FAIL: no §12 rows parsed from {args.doc}", file=sys.stderr)
        return 1

    c12, p12 = tally(table12)
    cs, ps = tally(supp)
    problems = p12 + ps

    ok = True
    if c12 != EXPECTED_TABLE12:
        ok = False
        problems.append(f"§12 tally {c12} != expected {EXPECTED_TABLE12}")
    nonzero_supp = {k: v for k, v in cs.items() if v}
    if nonzero_supp != EXPECTED_SUPPLEMENTARY:
        ok = False
        problems.append(
            f"§12.3 tally {nonzero_supp} != expected {EXPECTED_SUPPLEMENTARY}"
        )

    print(f"§12 named rows: {len(table12)} -> {c12}")
    print(f"§12.3 supplementary rows: {len(supp)} -> {cs}")
    if problems:
        for p in problems:
            print(f"DRIFT: {p}", file=sys.stderr)
        return 1
    print("OK: ledger row tally and status/evidence consistency match.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
