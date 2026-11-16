#!/usr/bin/env python3
"""Loading, validating, rendering and hashing the fresh capability suite.

The suite is a directory tree, not a Python data structure, so an arm's prompt,
its failing state, the hidden checker and the reference solution are all
inspectable files with hashes:

    fixtures/capability/tasks/<id>/
        task.json            metadata, limits, failure expectation, prompt intro
        initial/             the failing state handed to the arm
        hidden/              tests the harness owns and restores
        reference/           one reference solution (must pass the checker)

Every item is original to this repository and licensed MIT OR Apache-2.0, the
same terms as the rest of the tree (`LICENSE`). The manifest in
`fixtures/capability/MANIFEST.json` is generated from these files; drift between
the manifest and the tree is a failed check, not a warning.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import limits as limit_module
from . import formats as format_module

SCHEMA = "selvedge/capability-task/v1"
SUITE_SCHEMA = "selvedge/capability-manifest/v1"
LICENSE = "MIT OR Apache-2.0"
CATEGORIES = (
    "lifetime-borrow",
    "failing-test",
    "api-migration",
    "dependency-compat",
    "concurrency",
    "trait-impl",
    "repo-scale",
    "semantic-repair",
    "trio",
    "probe",
)
KINDS = ("suite", "trio", "probe")
CHECKER_KINDS = ("cargo_test", "compile_then_run")


class SuiteError(ValueError):
    """The suite on disk is not a valid measurement instrument."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: str) -> list[str]:
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            found.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return sorted(found)


def tree_digest(root: str) -> str:
    """A digest over (relative path, content) pairs, order-independent."""
    digest = hashlib.sha256()
    for relative in _relative_files(root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(os.path.join(root, relative)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class Task:
    root: str
    metadata: dict
    limits: limit_module.Limits
    run_limits: limit_module.Limits | None = None
    initial_files: list = field(default_factory=list)
    hidden_files: list = field(default_factory=list)
    reference_files: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.metadata["id"]

    @property
    def kind(self) -> str:
        return self.metadata["kind"]

    @property
    def category(self) -> str:
        return self.metadata["category"]

    @property
    def entry(self) -> str:
        return self.metadata["entry"]

    @property
    def manifest_policy(self) -> str:
        return self.metadata.get("manifest_policy", "locked")

    @property
    def checker_kind(self) -> str:
        return self.metadata["checker"]["kind"]

    @property
    def tree_sha256(self) -> str:
        return tree_digest(self.root)

    def initial_path(self, relative: str) -> str:
        return os.path.join(self.root, "initial", relative)

    def reference_path(self, relative: str) -> str:
        return os.path.join(self.root, "reference", relative)

    def hidden_path(self, relative: str) -> str:
        return os.path.join(self.root, "hidden", relative)

    def allowed_paths(self) -> set[str]:
        """Files an arm may return: everything in the failing state, plus new
        `src/**.rs` files. A locked manifest is not writable."""
        return {relative for relative in self.initial_files
                if not (relative == "Cargo.toml" and self.manifest_policy == "locked")}

    def read_initial(self, relative: str) -> str:
        with open(self.initial_path(relative), "r", encoding="utf-8") as handle:
            return handle.read()

    def read_hidden(self, relative: str) -> str:
        with open(self.hidden_path(relative), "r", encoding="utf-8") as handle:
            return handle.read()

    def read_reference(self, relative: str) -> str:
        with open(self.reference_path(relative), "r", encoding="utf-8") as handle:
            return handle.read()

    def reference_solution(self) -> dict:
        return {relative: self.read_reference(relative)
                for relative in self.reference_files}

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "category": self.category,
            "difficulty": self.metadata.get("difficulty"),
            "format": format_module.task_format(self),
            "tier": format_module.tier_for(self),
            "metric": format_module.metric_for(self),
            "extraction_policy": format_module.extraction_policy(self),
            "capability": self.metadata["capability"],
            "license": self.metadata["license"],
            "tree_sha256": self.tree_sha256,
            "initial_files": self.initial_files,
            "hidden_files": self.hidden_files,
            "reference_files": self.reference_files,
            "checker": {
                "kind": self.checker_kind,
                "timeout_seconds": self.limits.wall_seconds,
                "address_space_bytes": self.limits.address_space_bytes,
                "cpu_seconds": self.limits.cpu_seconds,
                "file_bytes": self.limits.file_bytes,
                "open_files": self.limits.open_files,
                "manifest_policy": self.manifest_policy,
            },
            "run_limits": (self.run_limits.as_dict() if self.run_limits else None),
            "initial_failure": self.metadata.get("initial_failure"),
            "probe_expectation": self.metadata.get("probe_expectation"),
        }


def _require(data: dict, key: str, where: str):
    if key not in data:
        raise SuiteError(f"{where}: missing required key {key!r}")
    return data[key]


def load_task(root: str) -> Task:
    where = os.path.relpath(root)
    metadata_path = os.path.join(root, "task.json")
    if not os.path.isfile(metadata_path):
        raise SuiteError(f"{where}: task.json is missing")
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except ValueError as exc:
        raise SuiteError(f"{where}: task.json is not valid JSON: {exc}")

    if metadata.get("schema") != SCHEMA:
        raise SuiteError(f"{where}: schema must be {SCHEMA!r}")
    required = ["id", "kind", "category", "capability", "license", "provenance",
                "entry", "checker"]
    if metadata.get("kind") != "probe":
        required.extend(["initial_failure", "prompt_intro"])
    for key in required:
        _require(metadata, key, where)
    if metadata["kind"] not in KINDS:
        raise SuiteError(f"{where}: kind {metadata['kind']!r} is not one of {KINDS}")
    if metadata["category"] not in CATEGORIES:
        raise SuiteError(f"{where}: unknown category {metadata['category']!r}")
    if metadata["license"] != LICENSE:
        raise SuiteError(f"{where}: license must be {LICENSE!r}")
    if os.path.basename(root.rstrip("/")) != metadata["id"]:
        raise SuiteError(f"{where}: directory name and id differ")

    checker = metadata["checker"]
    if not isinstance(checker, dict) or checker.get("kind") not in CHECKER_KINDS:
        raise SuiteError(f"{where}: checker.kind must be one of {CHECKER_KINDS}")
    if metadata["kind"] == "probe":
        if "probe_expectation" not in metadata:
            raise SuiteError(f"{where}: a probe must declare probe_expectation")
        if metadata["probe_expectation"] not in (
                "pass", "fail", "timeout", "oom", "compile_error"):
            raise SuiteError(f"{where}: unknown probe_expectation")
    if metadata["kind"] == "trio" and "shared_operation" not in metadata:
        raise SuiteError(f"{where}: a trio task must declare shared_operation")

    failure = metadata.get("initial_failure")
    if failure is not None:
        if not isinstance(failure, dict):
            raise SuiteError(f"{where}: initial_failure must be an object")
        if failure.get("kind") not in ("compile_error", "test_failure", "timeout", "oom",
                                       "hidden_test_failure"):
            raise SuiteError(f"{where}: initial_failure.kind is invalid")
        must = failure.get("must_contain")
        if not isinstance(must, list) or not must or not all(isinstance(m, str) and m
                                                             for m in must):
            raise SuiteError(
                f"{where}: initial_failure.must_contain must be non-empty strings")

    if metadata.get("manifest_policy", "locked") not in ("locked", "arm"):
        raise SuiteError(f"{where}: manifest_policy must be 'locked' or 'arm'")
    if "difficulty" in metadata and metadata["difficulty"] not in (
            "easy", "medium", "hard"):
        raise SuiteError(f"{where}: difficulty must be easy, medium or hard")
    if "format" in metadata and metadata["format"] not in format_module.FORMATS:
        raise SuiteError(f"{where}: format {metadata['format']!r} is not one of "
                         f"{format_module.FORMATS}")

    limits = limit_module.from_mapping(metadata.get("limits"), f"{where}: limits")
    run_limits = None
    if "run_limits" in metadata:
        run_limits = limit_module.run_only_from_mapping(
            metadata["run_limits"], f"{where}: run_limits")

    initial_root = os.path.join(root, "initial")
    hidden_root = os.path.join(root, "hidden")
    reference_root = os.path.join(root, "reference")
    for directory in (initial_root, hidden_root, reference_root):
        if not os.path.isdir(directory):
            raise SuiteError(f"{where}: {os.path.basename(directory)}/ is missing")
    initial_files = _relative_files(initial_root)
    hidden_files = _relative_files(hidden_root)
    reference_files = _relative_files(reference_root)
    if not initial_files:
        raise SuiteError(f"{where}: initial/ is empty")
    if not reference_files:
        raise SuiteError(f"{where}: reference/ is empty")
    if metadata["entry"] not in initial_files:
        raise SuiteError(f"{where}: entry {metadata['entry']!r} is not in initial/")
    if not hidden_files:
        raise SuiteError(f"{where}: hidden/ is empty")
    for relative in hidden_files:
        if not relative.startswith("tests/"):
            raise SuiteError(f"{where}: hidden file {relative!r} is not under tests/")
    for relative in reference_files:
        if relative.startswith("tests/"):
            raise SuiteError(f"{where}: reference may not contain tests/: {relative!r}")
        if relative not in initial_files and not (
                relative.startswith("src/") and relative.endswith(".rs")):
            raise SuiteError(
                f"{where}: reference file {relative!r} is neither in initial/ nor a new src file")

    task = Task(root=root, metadata=metadata, limits=limits, run_limits=run_limits,
                initial_files=initial_files, hidden_files=hidden_files,
                reference_files=reference_files)
    format_problems = format_module.validate_declaration(task)
    if format_problems:
        raise SuiteError("; ".join(format_problems))
    return task


def load_collection(directory: str) -> list[Task]:
    if not os.path.isdir(directory):
        raise SuiteError(f"collection directory {directory} is missing")
    tasks = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            tasks.append(load_task(path))
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise SuiteError(f"duplicate task ids in {directory}")
    return tasks


def load_suite(root: str) -> dict:
    """Load tasks/, trio/ and probes/ under a suite root."""
    return {
        "suite": load_collection(os.path.join(root, "tasks")),
        "trio": load_collection(os.path.join(root, "trio")),
        "probes": load_collection(os.path.join(root, "probes")),
    }


# ---------------------------------------------------------------------------
# prompt rendering


def render_prompt(task: Task) -> str:
    """The exact prompt handed to every arm. Deterministic in the fixture."""
    if format_module.task_format(task) != format_module.FORMAT_FILE:
        return format_module.render_prompt(task)
    parts = [task.metadata["prompt_intro"].strip(), "", "## Current files", ""]
    for relative in task.initial_files:
        parts.append(f"```text path={relative}")
        parts.append(task.read_initial(relative).rstrip("\n"))
        parts.append("```")
        parts.append("")
    observed = task.metadata.get("observed_failure")
    if observed:
        parts.extend(["## Observed failure", "", "```text", observed.strip(), "```", ""])
    parts.extend([
        "## Answer format",
        "",
        "Return the complete replacement for every file you change. Each file must",
        "be a fenced block whose opening line names its path, for example:",
        "",
        "```rust path=src/lib.rs",
        "...replacement file contents...",
        "```",
        "",
        "The harness discards any write under `tests/` and restores its own tests.",
        "The crate must build and its tests must pass with `cargo test --offline`.",
        "",
    ])
    return "\n".join(parts)


def prompt_sha256(task: Task) -> str:
    return sha256_bytes(render_prompt(task).encode("utf-8"))


# ---------------------------------------------------------------------------
# manifest


def build_manifest(root: str) -> dict:
    collections = load_suite(root)
    entries = []
    by_category: dict[str, int] = {}
    for kind in ("suite", "trio", "probes"):
        for task in collections[kind]:
            entries.append(task.summary())
            by_category[task.category] = by_category.get(task.category, 0) + 1
    entries.sort(key=lambda item: (item["kind"], item["id"]))
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(entry["id"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(entry["tree_sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema": SUITE_SCHEMA,
        "root": os.path.abspath(root),
        "license": LICENSE,
        "counts": {
            "suite": len(collections["suite"]),
            "trio": len(collections["trio"]),
            "probes": len(collections["probes"]),
            "total": len(entries),
            "by_category": dict(sorted(by_category.items())),
        },
        "suite_sha256": aggregate.hexdigest(),
        "tasks": entries,
    }


def manifest_path(root: str) -> str:
    return os.path.join(root, "MANIFEST.json")


def write_manifest(root: str, payload: dict) -> None:
    path = manifest_path(root)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def check_manifest(root: str) -> tuple[bool, str]:
    path = manifest_path(root)
    if not os.path.isfile(path):
        return False, "MANIFEST.json is missing; run scripts/capability_authoring.py --manifest"
    with open(path, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    current = build_manifest(root)
    # The absolute root is host-specific; compare the content-bearing fields.
    for key in ("schema", "license", "counts", "suite_sha256", "tasks"):
        if recorded.get(key) != current[key]:
            return False, f"MANIFEST.json is stale at {key!r}; regenerate it"
    return True, "manifest matches the tree"
