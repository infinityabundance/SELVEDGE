"""Authoring source for the fresh capability suite (P12.9, P11.21).

The tracked fixture tree under `fixtures/capability/` is materialised from these
declarations by `scripts/capability_authoring.py`, and `--check` fails if the
tree and this file disagree. The declarations are the reviewable origin of every
task: its failing state, its hidden checker, its reference solution, the
deliberately-wrong mutation the self-test must reject, and the bounds it runs
under.

All text is original to this repository (MIT OR Apache-2.0). No prompt, test or
solution is copied from HumanEval, GSM8K, TruthfulQA or a public Rust benchmark;
`scripts/capability_contamination.py` checks that claim against every registered
evaluation source and fails closed on overlap.
"""

from __future__ import annotations

from . import (batch3, concurrency, concurrency2, dependency, dependency2, expanded,
               failing, failing2, lifetime, lifetime2, migration, migration2,
               probes, repo_scale, repo_scale2, semantic, traits, traits2, trio,
               trio2)

#: The first expansion pass produced sixteen near-duplicates of existing
#: capabilities. `authoring/expanded.py` replaces them with a different
#: diagnostic, code path and reference; the superseded declarations remain in
#: their original modules for the review record but are excluded here and are
#: therefore never materialised.
_SUPERSEDED = {
    "borrow-05-return-normalized",
    "borrow-07-extend-while-iterating",
    "borrow-09-owned-excerpt",
    "failing-07-case-insensitive-word",
    "migrate-08-builder-result",
    "deps-04-path-version-requirement",
    "deps-05-enable-feature",
    "deps-06-edition-reserved-keyword",
    "concurrency-04-deterministic-lost-update",
    "traits-06-missing-generic-bound",
    "traits-07-associated-const-object-safety",
    "traits-08-conflicting-from",
    "semantic-01-inclusive-range",
    "semantic-05-saturating-multiply",
    "repo-06-error-conversion",
    "repo-09-trait-method-propagation",
}

_BASE_TASKS = (
    lifetime.TASKS
    + failing.TASKS
    + migration.TASKS
    + dependency.TASKS
    + concurrency.TASKS
    + traits.TASKS
    + repo_scale.TASKS
    + lifetime2.TASKS
    + failing2.TASKS
    + migration2.TASKS
    + dependency2.TASKS
    + concurrency2.TASKS
    + traits2.TASKS
    + repo_scale2.TASKS
    + semantic.TASKS
    + batch3.TASKS
)
SUITE_TASKS = tuple(task for task in _BASE_TASKS if task["id"] not in _SUPERSEDED)
SUITE_TASKS = SUITE_TASKS + tuple(expanded.TASKS)
TRIO_TASKS = trio.TASKS + trio2.TASKS
PROBE_TASKS = probes.TASKS

LICENSE = "MIT OR Apache-2.0"
PROVENANCE = (
    "original; authored for SELVEDGE P12.9/P11.21 (S14), no public benchmark text"
)


def crate_manifest(name: str, edition: str = "2021", dependencies: dict | None = None) -> str:
    """The default locked manifest for a single-crate task."""
    lines = [
        "[package]",
        f'name = "{name}"',
        'version = "0.1.0"',
        f'edition = "{edition}"',
        "",
        "[workspace]",
        "",
        "[lib]",
        'path = "src/lib.rs"',
    ]
    if dependencies:
        lines.extend(["", "[dependencies]"])
        for dependency_name in sorted(dependencies):
            lines.append(f"{dependency_name} = {dependencies[dependency_name]}")
    return "\n".join(lines) + "\n"


def crate_name(task_id: str) -> str:
    return task_id.replace("-", "_")
