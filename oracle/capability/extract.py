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

**Tolerant mode (S28, calibrated formats only).** The strict rule above is the
registered rule for the v4 suite and it is *not* changed: every historical
receipt stays readable under exactly the rule that produced it. The stored T2
generation receipts showed two answer shapes the strict rule discarded without
ever reaching the checker — a `path=src/lib.rs` line written *inside* the fence
instead of in its info string, and the same file emitted several times. The
calibrated suite sets `extraction_policy: "tolerant"`, which:

* reads a first line of the form `path=<file>`, ```` ```rust path=<file> ``` ````
  or `// src/<file>.rs` as the block's path,
* takes the last of several blocks for the same path (recording the replacement
  in `notes`) instead of rejecting the whole answer,
* picks the most Rust-like block when several unnamed fences exist,
* still rejects every path the strict rule rejects, and records the rejection.

Tolerance never invents code: whatever it recovers is compiled and tested, so a
recovered block that is wrong is a recorded failure like any other.
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
    policy: str = "strict"
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected and bool(self.files)

    def as_dict(self) -> dict:
        return {
            "files": sorted(self.files),
            "mode": self.mode,
            "rejected": list(self.rejected),
            "fence_count": self.fence_count,
            "policy": self.policy,
            "notes": list(self.notes),
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


def looks_like_rust(text: str) -> bool:
    return any(token in text for token in ("fn ", "pub ", "struct ", "impl ",
                                          "enum ", "use ", "const ", "mod "))


#: A path written as the first line *inside* a fence, the observed T2 shape.
INNER_PATH = re.compile(
    r"^\s*(?:```[a-zA-Z0-9_-]*\s*)?(?:path\s*[:=]\s*|//\s*)?"
    r"(?P<path>(?:src|tests)/[A-Za-z0-9_./-]+\.rs)\s*$")
INNER_PATH_KEY = re.compile(r"^\s*path\s*=\s*(?P<path>\S+)\s*$")


def _split_inner_path(body: str) -> tuple[str | None, str, list[str]]:
    """A first line that names a path inside the fence is the block's path.

    Returns (path or None, cleaned body, leaked-fence notes). The T2 arms wrote
    ```` ```rust\npath=src/lib.rs\n```rust ... ````; the second ```` ```rust ````
    line is a duplicated opener and is dropped with a note, never silently kept
    as Rust source (it would not compile anyway, but the note is the evidence).
    """
    lines = body.splitlines()
    notes: list[str] = []
    index = 0
    # At most two header lines: a bare fence opener and a path line, in either
    # order.
    path = None
    while index < len(lines) and index < 3:
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        match = INNER_PATH_KEY.match(line)
        if not match:
            match = INNER_PATH.match(line)
        if match and path is None:
            path = match.group("path")
            if line.strip().startswith("```"):
                notes.append("dropped a duplicated fence opener inside the block")
            index += 1
            continue
        if line.strip().startswith("```"):
            notes.append("dropped a duplicated fence opener inside the block")
            index += 1
            continue
        break
    return path, "\n".join(lines[index:]), notes


def _rust_score(body: str) -> int:
    return sum(body.count(token) for token in ("fn ", "impl ", "struct ",
                                               "pub ", "let ", "use "))


def fenced_blocks(text: str, tolerant: bool = False) -> list[tuple]:
    """(info, body, terminated, notes) for every fenced block.

    The strict path is the registered `FENCE` regex, unchanged. The tolerant path
    is a line scanner: it survives the observed T2 shape where the writer emits
    ```` ```rust ```` again inside what should be the same block, and it records
    whether a block was ever closed.
    """
    if not tolerant:
        return [(info, body, True, []) for info, body in FENCE.findall(text)]
    blocks: list[tuple] = []
    lines = text.splitlines()
    info = None
    body: list[str] = []
    notes: list[str] = []
    for line in lines:
        stripped = line.strip()
        if info is None:
            if stripped.startswith("```"):
                info = stripped[3:].strip()
                body = []
                notes = []
            continue
        if stripped.startswith("```"):
            joined = "\n".join(body)
            _inner, cleaned, _notes = _split_inner_path(joined)
            if stripped[3:].strip() and not cleaned.strip():
                # A fence opener written *inside* a block that holds only a path
                # header. Keep reading the same block.
                notes.append("dropped a duplicated fence opener inside the block")
                continue
            blocks.append((info, joined, True, notes))
            info = None
            body = []
            notes = []
            continue
        body.append(line)
    if info is not None:
        blocks.append((info, "\n".join(body), False, notes))
    return blocks


def extract_files(text: str, entry: str, allowed_paths: set[str],
                  manifest_policy: str = "locked",
                  tolerant: bool = False) -> Extraction:
    """Extract the answer's files under the registered rule.

    `tolerant` selects the calibrated-suite policy described in the module
    docstring. The default is the strict registered rule.
    """
    result = Extraction(policy="tolerant" if tolerant else "strict")
    if text is None:
        text = ""
    blocks = fenced_blocks(text, tolerant=tolerant)
    result.fence_count = len(blocks)

    named: list[tuple[str, str]] = []
    unnamed: list[str] = []
    for info, body, terminated, notes in blocks:
        _language, path = _parse_info(info)
        if tolerant:
            result.notes.extend(notes)
            inner, body, inner_notes = _split_inner_path(body)
            result.notes.extend(inner_notes)
            if inner and not path:
                path = inner
            if not terminated:
                result.notes.append("a fenced block was never closed")
        if path:
            named.append((path, body))
        elif body.strip():
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
                if not tolerant:
                    result.rejected.append(f"duplicate path {path!r}")
                    continue
                result.notes.append(f"duplicate path {path!r}; the last block wins")
            seen.add(path)
            result.files[path] = body
        return result

    if tolerant and len(unnamed) > 1 and entry:
        # Several unnamed fences: the model repeated itself or explained first.
        # Take the most Rust-like block (ties: the last), which is the only
        # deterministic rule that does not depend on prose order.
        best = max(range(len(unnamed)), key=lambda i: (_rust_score(unnamed[i]), i))
        chosen = unnamed[best]
        if looks_like_rust(chosen):
            result.mode = "fenced_entry"
            result.notes.append(
                f"{len(unnamed)} unnamed blocks; took block {best + 1} as the entry file")
            problem = path_problem(entry, allowed_paths, manifest_policy)
            if problem:
                result.rejected.append(problem)
            else:
                result.files[entry] = chosen
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
    if entry and stripped and looks_like_rust(stripped):
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
