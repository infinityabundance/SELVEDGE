#!/usr/bin/env python3
"""Survey the 144 source tasks, per file, for S31 item capacity.

Read-only; no model, no cargo. For every source it lists every Rust file, the
functions that file defines, which of them differ between the initial and
reference states, whether the file's code differs at all, and whether the
manifest differs. That is the item budget a derivation can draw on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import derive, formats, suite  # noqa: E402


def definitions(text: str) -> dict:
    result = {}
    for span in formats._function_spans(text):
        try:
            result[span["name"]] = formats.function_definition(text, span["name"])
        except formats.FormatError:
            pass
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures", "capability"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    collections = suite.load_suite(args.suite)
    rows = []
    for task in sorted(collections["suite"], key=lambda item: item.id):
        files = []
        for relative in sorted(task.reference_files):
            if not relative.endswith(".rs"):
                continue
            reference = task.read_reference(relative)
            definitions_reference = definitions(reference)
            initial = task.read_initial(relative) if relative in task.initial_files else ""
            definitions_initial = definitions(initial) if initial else {}
            differing = sorted(
                name for name, body in definitions_reference.items()
                if name not in definitions_initial
                or formats.normalise_body(definitions_initial[name])
                != formats.normalise_body(body))
            code_differs = derive._code_only(initial) != derive._code_only(reference)
            files.append({
                "path": relative,
                "in_initial": relative in task.initial_files,
                "functions": sorted(definitions_reference),
                "function_count": len(definitions_reference),
                "differing": differing,
                "code_differs": code_differs,
            })
        manifest_differs = False
        if "Cargo.toml" in task.reference_files and "Cargo.toml" in task.initial_files:
            manifest_differs = derive._code_only(task.read_initial("Cargo.toml")) != \
                derive._code_only(task.read_reference("Cargo.toml"))
        failure = task.metadata.get("initial_failure") or {}
        code = next((needle for needle in failure.get("must_contain", [])
                     if formats.ERROR_CODE.fullmatch(needle.strip())), None)
        hidden_tests = sum(task.read_hidden(relative).count("#[test]")
                           for relative in task.hidden_files)
        rows.append({
            "id": task.id,
            "category": task.category,
            "difficulty": task.metadata.get("difficulty"),
            "files": files,
            "manifest_differs": manifest_differs,
            "failure_kind": failure.get("kind"),
            "error_code": code,
            "hidden_tests": hidden_tests,
            "has_mutation": isinstance(task.metadata.get("mutation"), dict),
            "has_naive": isinstance(task.metadata.get("naive"), dict),
        })

    by_category: dict[str, dict] = {}
    for row in rows:
        entry = by_category.setdefault(row["category"], {
            "sources": 0, "function_patch_capacity": 0, "file_patch_capacity": 0,
            "manifest_patch_capacity": 0, "diagnostic_capacity": 0,
            "hidden_tests": 0, "multi_file": 0})
        entry["sources"] += 1
        entry["hidden_tests"] += row["hidden_tests"]
        if len({item["path"] for item in row["files"]}) > 1:
            entry["multi_file"] += 1
        for item in row["files"]:
            if not item["in_initial"]:
                continue
            entry["function_patch_capacity"] += len(item["functions"])
            if item["code_differs"]:
                entry["file_patch_capacity"] += 1
        if row["manifest_differs"]:
            entry["manifest_patch_capacity"] += 1
        if row["error_code"]:
            entry["diagnostic_capacity"] += 1
    for entry in by_category.values():
        entry["total_capacity"] = (entry["function_patch_capacity"]
                                   + entry["file_patch_capacity"]
                                   + entry["manifest_patch_capacity"])

    payload = {"schema": "selvedge/s31-population-probe/v2",
               "sources": len(rows), "by_category": by_category, "items": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(by_category, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
