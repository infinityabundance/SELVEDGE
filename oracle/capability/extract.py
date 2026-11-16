#!/usr/bin/env python3
"""Code extraction from an arm's answer.

The harness owns extraction so that every arm — an HF checkpoint, the Rust
runtime, a future module set — is read by the same rule. The rule is fixed
before any run and is part of the prompt contract:

1. every fenced block whose info string names `path=<relative path>` is a file
   replacement;
2. if no block names a path, exactly one fenced block is taken as the whole
   content of the task's entry file;
3. if there is no fence and the answer contains Rust syntax, the whole answer is
   the entry file (full-text mode);
4. anything else is an invalid answer, recorded as a failure, never repaired by
   guessing.

A path that escapes the crate (`..`, absolute), targets a harness-owned file
(`tests/`, `build.rs`, `.cargo/`) or duplicates another path is rejected, and the
rejection is recorded. This is the code-level half of "no access to the
checker": an answer cannot overwrite the tests that score it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FENCE = re.compile(r"```[ \t]*([^\n`]*)\n(.*?)```", re.DOTALL)

#: Paths the harness owns. An arm may not write them.
FORBIDDEN_PREFIXES = ("tests/", ".cargo/", "target/")
FORBIDDEN_NAMES = ("build.rs",)
ALLOWED_TOP_FILES = ("Cargo.toml",)


@dataclass
class Extraction:
    files: dict = field(default_factory=dict)
    mode: str = "none"
    rejected: list = field(default_factory=list)
    fence_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.rejected and bool(self.files)

    def as_dict(self) -> dict:
        return {
            "files": sorted(self.files),
            "mode": self.mode,
            "rejected": list(self.rejected),
            "fence_count": self.fence_count,
            "ok": self.ok,
        }


def _parse_info(info: str) -> tuple[str, str | None]:
    """Return (language, path) from a fence info string.

    Accepts ```rust path=src/lib.rs, ```path=src/lib.rs, and quoted paths.
    """
    language = ""
    path = None
    for token in info.strip().split():
        if token.startswith("path="):
            value = token[len("path="):]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            path = value
        elif not language and "=" not in token:
            language = token
    return language, path


def path_problem(path: str, allowed: set[str], manifest_policy: str) -> str | None:
    if not path:
        return "empty path"
    if path.startswith("/") or path.startswith("\\"):
        return f"absolute path {path!r}"
    normalised = path.replace("\\", "/")
    if ".." in normalised.split("/"):
        return f"path escapes the crate: {path!r}"
    if normalised.startswith(FORBIDDEN_PREFIXES):
        return f"path is harness-owned: {path!r}"
    if normalised in FORBIDDEN_NAMES:
        return f"path is harness-owned: {path!r}"
    if "/" not in normalised and normalised not in ALLOWED_TOP_FILES:
        return f"top-level file is not allowed: {path!r}"
    if normalised == "Cargo.toml":
        if manifest_policy != "arm":
            return "the crate manifest is locked for this task"
        return None
    if normalised not in allowed:
        # New source files are allowed; new non-Rust files are not.
        if normalised.startswith("src/") and normalised.endswith(".rs"):
            return None
        return f"file is not part of the task's writable set: {path!r}"
    return None


def extract_files(text: str, entry: str, allowed_paths: set[str],
                  manifest_policy: str = "locked") -> Extraction:
    """Extract the answer's files under the registered rule."""
    result = Extraction()
    if text is None:
        text = ""
    blocks = FENCE.findall(text)
    result.fence_count = len(blocks)

    named: list[tuple[str, str]] = []
    unnamed: list[str] = []
    for info, body in blocks:
        _language, path = _parse_info(info)
        if path:
            named.append((path, body))
        else:
            unnamed.append(body)

    if named:
        result.mode = "fenced_paths"
        seen: set[str] = set()
        for path, body in named:
            problem = path_problem(path, allowed_paths, manifest_policy)
            if problem:
                result.rejected.append(problem)
                continue
            if path in seen:
                result.rejected.append(f"duplicate path {path!r}")
                continue
            seen.add(path)
            result.files[path] = body
        return result

    if len(unnamed) == 1 and entry:
        result.mode = "fenced_entry"
        problem = path_problem(entry, allowed_paths, manifest_policy)
        if problem:
            result.rejected.append(problem)
        else:
            result.files[entry] = unnamed[0]
        return result

    stripped = text.strip()
    if entry and stripped and ("fn " in stripped or "pub " in stripped
                               or "struct " in stripped or "impl " in stripped):
        result.mode = "full_text"
        problem = path_problem(entry, allowed_paths, manifest_policy)
        if problem:
            result.rejected.append(problem)
        else:
            result.files[entry] = stripped + "\n"
        return result

    result.mode = "invalid"
    result.rejected.append("no code block and no recognisable Rust body")
    return result
