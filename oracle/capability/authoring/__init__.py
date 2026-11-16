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

from . import (concurrency, dependency, failing, lifetime, migration, probes,
               repo_scale, traits, trio)

SUITE_TASKS = (
    lifetime.TASKS
    + failing.TASKS
    + migration.TASKS
    + dependency.TASKS
    + concurrency.TASKS
    + traits.TASKS
    + repo_scale.TASKS
)
TRIO_TASKS = trio.TASKS
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
