#!/usr/bin/env python3
"""Workdir construction and the executable checker.

A check is four steps, all recorded:

1. copy the task's `initial/` into a fresh workdir under the suite's bounded
   work root;
2. apply the arm's extracted files, rejecting paths the harness owns;
3. restore harness-owned paths (the vendored dependency, the manifest when it is
   locked) and overwrite `tests/` with the hidden tests;
4. run `cargo test --offline` under `sandbox.run_bounded`.

The reference solution must pass, the failing state must fail with the recorded
evidence, and a mutated reference must fail; `scripts/capability_selftest.py`
checks all three for every task before the suite can be used.

`compile_then_run` exists for the sandbox probes: it compiles under the normal
envelope and then runs the *prebuilt* test binary under a tighter one, which is
how a timeout and an allocation failure are demonstrated without letting the
compiler itself trip the limit.
"""

from __future__ import annotations

import glob
import os
import tomllib

from . import sandbox
from .extract import path_problem


class CheckerError(RuntimeError):
    """The checker could not be prepared; no measurement is valid."""


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _copy_initial(task, workdir: str) -> None:
    for relative in task.initial_files:
        destination = os.path.join(workdir, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(task.initial_path(relative), "rb") as source, \
                open(destination, "wb") as target:
            target.write(source.read())


def validate_manifest_text(text: str, workdir: str, task_id: str) -> list[str]:
    """Reject a manifest that could fake a pass.

    The arm may edit its own manifest only when the task says so, and then only
    dependency paths/versions/features. Build scripts, patches, extra test or
    binary targets, profiles and dev-dependencies are all rejected, because each
    is a way to change what "the tests pass" means.
    """
    problems: list[str] = []
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [f"{task_id}: Cargo.toml is not valid TOML: {exc}"]
    allowed_tables = {"package", "lib", "dependencies", "workspace"}
    extra = set(parsed) - allowed_tables
    if extra:
        problems.append(f"{task_id}: manifest tables {sorted(extra)} are not allowed")
    package = parsed.get("package") or {}
    if "build" in package:
        problems.append(f"{task_id}: a build script is not allowed")
    if "edition" in package and package["edition"] not in ("2021", "2024"):
        problems.append(f"{task_id}: edition {package['edition']!r} is not supported")
    dependencies = parsed.get("dependencies") or {}
    if not isinstance(dependencies, dict):
        problems.append(f"{task_id}: [dependencies] must be a table")
        return problems
    root = os.path.abspath(workdir)
    for name, spec in dependencies.items():
        if not isinstance(spec, dict):
            problems.append(f"{task_id}: dependency {name!r} must be a table with a path")
            continue
        unknown = set(spec) - {"path", "version", "features", "default-features", "package"}
        if unknown:
            problems.append(f"{task_id}: dependency {name!r} has unsupported keys {sorted(unknown)}")
        path = spec.get("path")
        if not path:
            problems.append(f"{task_id}: dependency {name!r} must be a path dependency")
            continue
        resolved = os.path.abspath(os.path.join(root, path))
        if not (resolved == root or resolved.startswith(root + os.sep)):
            problems.append(f"{task_id}: dependency path {path!r} escapes the crate")
        elif not os.path.isdir(resolved):
            problems.append(f"{task_id}: dependency path {path!r} does not exist")
        features = spec.get("features", [])
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            problems.append(f"{task_id}: dependency {name!r} features must be a list of strings")
    return problems


def prepare(task, solution: dict, workroot: str,
            allow_empty: bool = False) -> tuple[str, list[str]]:
    """Build the workdir; return (path, problems).

    `solution` is already-extracted {relative path: content}; extraction happens
    before this call so the raw answer is never re-fenced or re-parsed.
    """
    workdir = os.path.join(workroot, task.id)
    sandbox.remove_tree(workdir)
    os.makedirs(workdir, exist_ok=True)
    _copy_initial(task, workdir)

    problems = []
    for relative in solution:
        problem = path_problem(relative, task.allowed_paths(), task.manifest_policy)
        if problem:
            problems.append(problem)
    if problems:
        return workdir, problems
    if not solution and not allow_empty:
        return workdir, ["no files were supplied"]

    for relative, content in sorted(solution.items()):
        _write(os.path.join(workdir, relative), content)

    if task.manifest_policy == "arm":
        manifest_path = os.path.join(workdir, "Cargo.toml")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            problems.extend(validate_manifest_text(handle.read(), workdir, task.id))
    else:
        # The manifest is locked for this task; restore it from the fixture.
        for relative in task.initial_files:
            if relative == "Cargo.toml":
                with open(task.initial_path(relative), "rb") as source, \
                        open(os.path.join(workdir, relative), "wb") as target:
                    target.write(source.read())

    # Read-only trees (a vendored dependency) are harness-owned in every task:
    # a migration is a change to the caller, never to the library being migrated
    # to.
    readonly = list(task.metadata.get("readonly", []))
    if readonly:
        for relative in task.initial_files:
            if any(relative == entry.rstrip("/") or relative.startswith(entry)
                   for entry in readonly):
                with open(task.initial_path(relative), "rb") as source, \
                        open(os.path.join(workdir, relative), "wb") as target:
                    target.write(source.read())

    # The harness owns tests: whatever the arm wrote there is discarded.
    sandbox.remove_tree(os.path.join(workdir, "tests"))
    for relative in task.hidden_files:
        destination = os.path.join(workdir, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(task.hidden_path(relative), "r", encoding="utf-8") as source:
            content = source.read()
        _write(destination, content)
    return workdir, problems


def _test_args(task) -> list:
    args = task.metadata["checker"].get("test_args", ["--offline", "--quiet"])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise CheckerError(f"{task.id}: checker.test_args must be a list of strings")
    return args


def initial_solution(task) -> dict:
    """The failing state expressed as an arm answer, for probe runs."""
    return {relative: task.read_initial(relative)
            for relative in task.initial_files if relative != "Cargo.toml"}


def run_check(task, solution: dict, workroot: str, keep: bool = False) -> dict:
    """Run the executable checker for one task and one solution."""
    workdir, problems = prepare(task, solution, workroot)
    if problems:
        return {
            "task": task.id,
            "category": task.category,
            "status": sandbox.STATUS_INVALID,
            "problems": problems,
            "checker": task.checker_kind,
        }
    if task.checker_kind == "compile_then_run":
        result = _compile_then_run(task, workdir)
    else:
        run = sandbox.run_bounded(["cargo", "test"] + _test_args(task), workdir,
                                  task.limits)
        result = run.as_dict()
    if not keep:
        sandbox.remove_tree(workdir)
    return {
        "task": task.id,
        "category": task.category,
        "status": result["status"],
        "checker": task.checker_kind,
        "evidence": result,
    }


def _crate_name(workdir: str) -> str:
    manifest = os.path.join(workdir, "Cargo.toml")
    with open(manifest, "rb") as handle:
        data = tomllib.load(handle)
    name = (data.get("package") or {}).get("name")
    if not name:
        raise CheckerError(f"{manifest}: no [package] name")
    return name


def _find_test_binary(workdir: str) -> str:
    """The crate's unit-test binary, not an integration-test artifact.

    `cargo test --no-run` produces one executable per test target: the lib unit
    tests (`<crate>-<hash>`) and each file in `tests/` (`check-<hash>`). The
    harness restores an empty `tests/check.rs` for the probes, so picking the
    newest executable could pick the empty integration test and record a pass
    without running the probe's own test. Select by the crate name instead.
    """
    crate = _crate_name(workdir).replace("-", "_")
    pattern = os.path.join(workdir, "target", "debug", "deps", "*")
    candidates = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if not name.startswith(crate + "-"):
            continue
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            continue
        if path.endswith((".d", ".rlib", ".rmeta", ".so")):
            continue
        candidates.append(path)
    if not candidates:
        raise CheckerError(f"no unit-test binary for crate {crate!r} under {pattern}")
    return max(candidates, key=lambda path: os.path.getmtime(path))


def _compile_then_run(task, workdir: str) -> dict:
    if task.run_limits is None:
        raise CheckerError(f"{task.id}: compile_then_run needs run_limits")
    compile_run = sandbox.run_bounded(
        ["cargo", "test", "--no-run", "--offline", "--quiet"], workdir, task.limits)
    compiled = compile_run.as_dict()
    compiled["phase"] = "compile"
    if compile_run.status != sandbox.STATUS_PASS:
        return compiled
    binary = _find_test_binary(workdir)
    run = sandbox.run_bounded([binary, "--test-threads=1"], workdir, task.run_limits,
                              toolchain_floor=False)
    executed = run.as_dict()
    executed["phase"] = "run"
    executed["compile"] = {
        "status": compile_run.status,
        "duration_ms": compile_run.duration_ms,
        "stderr_sha256": compile_run.stderr_sha256,
    }
    return executed


# ---------------------------------------------------------------------------
# initial-state evidence and the mutated reference


def run_initial(task, workroot: str, keep: bool = False) -> dict:
    """Run the failing state and return its evidence."""
    expected = task.metadata["initial_failure"]
    workdir, problems = prepare(task, {}, workroot, allow_empty=True)
    if problems:
        raise CheckerError(f"{task.id}: could not prepare the initial state: {problems}")
    if expected["kind"] == "compile_error":
        run = sandbox.run_bounded(["cargo", "check", "--offline", "--quiet", "--lib"],
                                  workdir, task.limits)
    elif expected["kind"] == "test_failure":
        run = sandbox.run_bounded(["cargo", "test", "--offline", "--quiet"],
                                  workdir, task.limits)
    elif expected["kind"] == "timeout":
        run = sandbox.run_bounded(["cargo", "test", "--offline", "--quiet"],
                                  workdir, task.limits)
    else:
        run = sandbox.run_bounded(["cargo", "test", "--offline", "--quiet"],
                                  workdir, task.limits)
    text = run.stdout_tail + run.stderr_tail
    missing = [needle for needle in expected["must_contain"] if needle not in text]
    payload = {
        "task": task.id,
        "expected_kind": expected["kind"],
        "status": run.status,
        "duration_ms": run.duration_ms,
        "must_contain": expected["must_contain"],
        "missing_evidence": missing,
        "stderr_tail": run.stderr_tail[-2000:],
        "stdout_tail": run.stdout_tail[-1000:],
        "occurred_as_expected": run.status != sandbox.STATUS_PASS and not missing,
        "evidence": run.as_dict(),
    }
    if not keep:
        sandbox.remove_tree(workdir)
    return payload


def mutate_reference(task) -> tuple[dict, dict]:
    """A deliberately wrong solution derived from the reference.

    The generic mutation changes the first numeric literal in the first
    reference file that has one; `find`/`replace` in the task metadata overrides
    it. The self-test requires the mutation to fail the checker, so a task whose
    tests do not discriminate is reported rather than accepted.
    """
    solution = task.reference_solution()
    override = task.metadata.get("mutation")
    if isinstance(override, dict):
        path = override["file"]
        find = override["find"]
        replace = override["replace"]
        if path not in solution:
            raise CheckerError(f"{task.id}: mutation file {path!r} is not in the reference")
        if find not in solution[path]:
            raise CheckerError(f"{task.id}: mutation find {find!r} is absent")
        solution[path] = solution[path].replace(find, replace, 1)
        return solution, {"kind": "declared", "file": path, "find": find,
                          "replace": replace}

    for path in sorted(solution):
        if not path.endswith(".rs"):
            continue
        text = solution[path]
        for index, character in enumerate(text):
            if character.isdigit():
                replacement = "0" if character != "0" else "1"
                mutated = text[:index] + replacement + text[index + 1:]
                if mutated != text:
                    solution[path] = mutated
                    return solution, {"kind": "generic_numeric", "file": path,
                                      "from": character, "to": replacement}
                break
    raise CheckerError(f"{task.id}: no generic mutation applies; add a declared mutation")
