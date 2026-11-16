#!/usr/bin/env python3
"""Near-duplicate screen for the capability suite and the sealed holdback.

Three views of an item, because a copy can hide in any of them:

* **diagnostic** -- the declared failing diagnostic plus the expected evidence;
* **code** -- the non-test source of the initial and reference states, with
  comments and whitespace removed;
* **assertion** -- the hidden tests' assertion lines, with literals normalised.

Each view is compared with a set-Jaccard over shingles (4-gram tokens for code,
2-gram tokens for diagnostics, 3-gram tokens for assertions). A pair is
**rejected** when any of the registered rules below fires; the run fails closed
on a rejection. The thresholds were fixed before this screen was run on the
expansion and are recorded in the receipt.

    python3 scripts/capability_similarity.py --out results/capability/similarity-<stamp>.json
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO, "oracle")
sys.path.insert(0, ORACLE)

from capability import authoring, sealed, suite  # noqa: E402

SCHEMA = "selvedge/capability-similarity/v1"

#: Registered reject rules, fixed before the screen ran on the expansion.
RULES = {
    "code_reject": 0.60,
    "diagnostic_reject": 0.75,
    "assertion_reject": 0.60,
    "cross_view_code_floor": 0.30,
    "report_code": 0.40,
}

CODE_WIDTH = 4
DIAGNOSTIC_WIDTH = 2
ASSERTION_WIDTH = 3


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


def _tokens(text: str) -> list:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\sA-Za-z0-9_]", text)


def shingles(tokens: list, width: int) -> set:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + width])
            for index in range(len(tokens) - width + 1)}


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def strip_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        marker = line.find("//")
        if marker >= 0:
            line = line[:marker]
        lines.append(line)
    return "\n".join(lines)


def diagnostic_text(task) -> str:
    failure = task.metadata.get("initial_failure") or {}
    observed = task.metadata.get("observed_failure") or ""
    observed = re.sub(r"-->.*", "", observed)
    observed = re.sub(r"[0-9]+", "N", observed)
    return " ".join([
        failure.get("kind", ""),
        " ".join(failure.get("must_contain") or []),
        observed,
    ])


def code_text(task) -> str:
    parts = []
    for relative in task.initial_files:
        if relative.endswith(".rs") and not relative.startswith("tests/"):
            parts.append(strip_comments(task.read_initial(relative)))
    for relative in task.reference_files:
        if relative.endswith(".rs") and not relative.startswith("tests/"):
            parts.append(strip_comments(task.read_reference(relative)))
    return "\n".join(parts)


def assertion_text(task) -> str:
    lines = []
    for relative in task.hidden_files:
        for line in task.read_hidden(relative).splitlines():
            if "assert" in line:
                line = re.sub(r'"[^"]*"', '"S"', line)
                line = re.sub(r"\b[0-9]+\b", "N", line)
                lines.append(line.strip())
    return "\n".join(lines)


def views(task) -> dict:
    return {
        "diagnostic": shingles(_tokens(diagnostic_text(task)), DIAGNOSTIC_WIDTH),
        "code": shingles(_tokens(code_text(task)), CODE_WIDTH),
        "assertion": shingles(_tokens(assertion_text(task)), ASSERTION_WIDTH),
    }


def rule_fired(scores: dict) -> str | None:
    if scores["code"] >= RULES["code_reject"]:
        return "code_reject"
    if (scores["diagnostic"] >= RULES["diagnostic_reject"]
            and scores["code"] >= RULES["cross_view_code_floor"]):
        return "diagnostic_reject"
    if (scores["assertion"] >= RULES["assertion_reject"]
            and scores["code"] >= RULES["cross_view_code_floor"]):
        return "assertion_reject"
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures", "capability"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    collections = suite.load_suite(args.suite)
    sealed_items = sealed.load_items(args.suite)
    tasks = collections["suite"] + collections["trio"] + sealed_items
    # Rejection is scoped to the items authored in this expansion: a pair of
    # already-registered historical items cannot be replaced here without
    # changing the registered suite, so it is reported and left for a separate
    # supersession. Every pair with at least one new item is subject to the rules.
    new_ids = ({task["id"] for task in authoring.batch3.TASKS}
               | {task["id"] for task in authoring.sealed.SEALED_TASKS})
    task_views = {task.id: views(task) for task in tasks}
    kinds = {}
    for task in tasks:
        if task.id in {item.id for item in sealed_items}:
            kinds[task.id] = "sealed"
        elif task.kind == "trio":
            kinds[task.id] = "trio"
        else:
            kinds[task.id] = "suite"

    rejected = []
    reported = []
    maxima = {"diagnostic": 0.0, "code": 0.0, "assertion": 0.0}
    pair_count = 0
    for left, right in itertools.combinations(tasks, 2):
        pair_count += 1
        scores = {
            view: jaccard(task_views[left.id][view], task_views[right.id][view])
            for view in ("diagnostic", "code", "assertion")
        }
        for view, value in scores.items():
            maxima[view] = max(maxima[view], value)
        entry = {
            "left": left.id,
            "right": right.id,
            "left_kind": kinds[left.id],
            "right_kind": kinds[right.id],
            "left_category": left.category,
            "right_category": right.category,
            "scores": {key: round(value, 6) for key, value in scores.items()},
        }
        fired = rule_fired(scores)
        involves_new = left.id in new_ids or right.id in new_ids
        if fired and involves_new:
            entry["rule"] = fired
            rejected.append(entry)
        elif fired or scores["code"] >= RULES["report_code"]:
            entry["rule"] = fired or "reported_code"
            entry["scope"] = "historical" if not involves_new else "reported"
            reported.append(entry)

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": revision(),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "items": len(tasks),
        "pairs_compared": pair_count,
        "counts": {
            "pairs_compared": pair_count,
            "rejected_pairs": len(rejected),
            "reported_pairs": len(reported),
            "historical_pairs": sum(1 for entry in reported
                                    if entry.get("scope") == "historical"),
        },
        "rules": RULES,
        "widths": {
            "code": CODE_WIDTH,
            "diagnostic": DIAGNOSTIC_WIDTH,
            "assertion": ASSERTION_WIDTH,
        },
        "metric": "set Jaccard over token shingles per view",
        "maxima": {key: round(value, 6) for key, value in maxima.items()},
        "rejected_pairs": rejected,
        "reported_pairs": reported,
        "replaced": [],
        "rejection_scope": (
            "pairs with at least one item authored in this expansion; historical "
            "pairs above a threshold are reported, not rejected, because replacing "
            "them would change the already-registered suite"),
        "verdict": "pass" if not rejected else "reject",
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(out + ".tmp", out)
    print(json.dumps({
        "verdict": payload["verdict"],
        "items": len(tasks),
        "pairs_compared": pair_count,
        "counts": {
            "pairs_compared": pair_count,
            "rejected_pairs": len(rejected),
            "reported_pairs": len(reported),
            "historical_pairs": sum(1 for entry in reported
                                    if entry.get("scope") == "historical"),
        },
        "rejected": len(rejected),
        "reported": len(reported),
        "maxima": payload["maxima"],
        "out": os.path.relpath(out, REPO),
    }, indent=2, sort_keys=True))
    if rejected:
        for entry in rejected:
            print(f"capability_similarity: {entry['rule']}: {entry['left']} vs "
                  f"{entry['right']} {entry['scores']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
