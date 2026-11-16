#!/usr/bin/env python3
"""Materialise, check and manifest the fresh capability suite.

The authoring declarations in `oracle/capability/authoring/` are the reviewable
origin; `fixtures/capability/` is the tracked tree every arm and checker reads.
This script is the only writer of that tree.

    python3 scripts/capability_authoring.py --materialize --manifest
    python3 scripts/capability_authoring.py --check
    python3 scripts/capability_authoring.py --manifest

`--check` fails closed if the tree and the declarations disagree (a changed
fixture, a missing reference, an extra file), and `--manifest` regenerates
`MANIFEST.json` from the materialised tree so the hashes describe what a run
actually reads. Run it inside the rust court:

    scripts/court rust -- python3 scripts/capability_authoring.py --materialize --manifest
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(REPO, "oracle")
sys.path.insert(0, ORACLE)

from capability import authoring, suite  # noqa: E402

DEFAULT_ROOT = os.path.join(REPO, "fixtures", "capability")

COLLECTIONS = (
    ("tasks", authoring.SUITE_TASKS),
    ("trio", authoring.TRIO_TASKS),
    ("probes", authoring.PROBE_TASKS),
    ("sealed", authoring.SEALED_TASKS),
)

#: Keys a declaration may carry. Anything else is a typo that would be silently
#: ignored, so it is rejected instead.
DECLARATION_KEYS = {
    "id", "kind", "category", "capability", "difficulty", "prompt_intro",
    "observed_failure", "files", "hidden", "reference", "readonly", "manifest",
    "manifest_policy", "limits", "run_limits", "initial_failure", "checker",
    "mutation", "naive", "shared_operation", "operation_items", "probe_expectation",
    "probe_stdout_contains",
}


def expected_files(declaration: dict) -> dict:
    """Every file the materialised task must contain, as {relative: content}."""
    files: dict[str, str] = {}
    for relative, content in declaration["files"].items():
        files[f"initial/{relative}"] = content
    for relative, content in declaration["hidden"].items():
        files[f"hidden/{relative}"] = content
    for relative, content in declaration["reference"].items():
        files[f"reference/{relative}"] = content

    metadata = {
        "schema": suite.SCHEMA,
        "id": declaration["id"],
        "kind": declaration.get("kind", "suite"),
        "category": declaration["category"],
        "difficulty": declaration.get("difficulty", "medium"),
        "capability": declaration["capability"],
        "license": authoring.LICENSE,
        "provenance": authoring.PROVENANCE,
        "entry": declaration.get("entry", "src/lib.rs"),
        "manifest_policy": declaration.get("manifest_policy", "locked"),
        "checker": declaration.get("checker", {"kind": "cargo_test"}),
    }
    if "prompt_intro" in declaration:
        metadata["prompt_intro"] = declaration["prompt_intro"]
    if "observed_failure" in declaration:
        metadata["observed_failure"] = declaration["observed_failure"]
    if "readonly" in declaration:
        metadata["readonly"] = declaration["readonly"]
    if "limits" in declaration:
        metadata["limits"] = declaration["limits"]
    if "run_limits" in declaration:
        metadata["run_limits"] = declaration["run_limits"]
    if "initial_failure" in declaration:
        metadata["initial_failure"] = declaration["initial_failure"]
    if "mutation" in declaration:
        metadata["mutation"] = declaration["mutation"]
    if "naive" in declaration:
        metadata["naive"] = declaration["naive"]
    if "shared_operation" in declaration:
        metadata["shared_operation"] = declaration["shared_operation"]
    if "operation_items" in declaration:
        metadata["operation_items"] = declaration["operation_items"]
    if "probe_expectation" in declaration:
        metadata["probe_expectation"] = declaration["probe_expectation"]
    if "probe_stdout_contains" in declaration:
        metadata["probe_stdout_contains"] = declaration["probe_stdout_contains"]

    manifest = declaration.get("manifest")
    if manifest is None and "Cargo.toml" not in declaration["files"]:
        manifest = authoring.crate_manifest(authoring.crate_name(declaration["id"]))
    if manifest is not None:
        files["initial/Cargo.toml"] = manifest

    files["task.json"] = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    return files


def validate_declarations() -> None:
    seen: set[str] = set()
    for _collection, declarations in COLLECTIONS:
        for declaration in declarations:
            unknown = set(declaration) - DECLARATION_KEYS
            if unknown:
                raise SystemExit(
                    f"{declaration.get('id', '<no id>')}: unknown declaration keys "
                    f"{sorted(unknown)}")
            task_id = declaration["id"]
            if task_id in seen:
                raise SystemExit(f"duplicate task id {task_id!r}")
            seen.add(task_id)
            if not declaration.get("mutation") and declaration.get(
                    "kind", "suite") != "probe":
                raise SystemExit(f"{task_id}: a declared mutation is required")
            for relative in declaration["files"]:
                if relative.startswith(("/", "..")):
                    raise SystemExit(f"{task_id}: unsafe file {relative!r}")


def _disk_files(root: str) -> dict:
    found = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "r", encoding="utf-8") as handle:
                found[relative] = handle.read()
    return found


def materialize(root: str) -> list[str]:
    validate_declarations()
    # The authoring source owns tasks/, trio/ and probes/. The root README is
    # tracked documentation, not a generated task, so regeneration preserves it.
    readme = os.path.join(root, "README.md")
    readme_content = None
    if os.path.isfile(readme):
        with open(readme, "r", encoding="utf-8") as handle:
            readme_content = handle.read()
    if os.path.isdir(root):
        shutil.rmtree(root)
    if readme_content is not None:
        os.makedirs(root, exist_ok=True)
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(readme_content)
    written = []
    for collection, declarations in COLLECTIONS:
        for declaration in declarations:
            task_root = os.path.join(root, collection, declaration["id"])
            for relative, content in expected_files(declaration).items():
                path = os.path.join(task_root, relative)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                written.append(os.path.relpath(path, REPO))
    return written


def check(root: str) -> list[str]:
    validate_declarations()
    problems = []
    expected: dict[str, str] = {}
    for collection, declarations in COLLECTIONS:
        for declaration in declarations:
            task_root = os.path.join(root, collection, declaration["id"])
            for relative, content in expected_files(declaration).items():
                expected[os.path.join(task_root, relative)] = content
    for path, content in expected.items():
        if not os.path.isfile(path):
            problems.append(f"missing {os.path.relpath(path, REPO)}")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            if handle.read() != content:
                problems.append(f"differs from the authoring source: {os.path.relpath(path, REPO)}")
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if relative == "MANIFEST.json" or relative == "README.md":
                continue
            if path not in expected:
                problems.append(f"unexpected file {os.path.relpath(path, REPO)}")
    if os.path.isdir(root):
        for collection in ("tasks", "trio", "probes", "sealed"):
            collection_root = os.path.join(root, collection)
            if not os.path.isdir(collection_root):
                problems.append(f"missing collection {collection}/")
                continue
            for name in sorted(os.listdir(collection_root)):
                path = os.path.join(collection_root, name)
                if os.path.isdir(path) and not any(
                        os.path.commonpath([path, expected_path]) == path
                        for expected_path in expected):
                    problems.append(f"unexpected task directory {os.path.relpath(path, REPO)}")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args(argv)
    if not (args.materialize or args.check or args.manifest):
        parser.error("choose at least one of --materialize, --check, --manifest")

    if args.materialize:
        written = materialize(args.root)
        print(f"capability_authoring: materialised {len(written)} files under "
              f"{os.path.relpath(args.root, REPO)}")
    if args.check:
        problems = check(args.root)
        if problems:
            for problem in problems:
                print(f"capability_authoring: {problem}", file=sys.stderr)
            return 3
        print("capability_authoring: the tree matches the authoring source")
    if args.manifest:
        payload = suite.build_manifest(args.root)
        suite.write_manifest(args.root, payload)
        print(f"capability_authoring: wrote {os.path.relpath(suite.manifest_path(args.root), REPO)}"
              f" ({payload['counts']['total']} tasks, suite {payload['suite_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
