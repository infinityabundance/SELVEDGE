#!/usr/bin/env python3
"""The S31 selection-item population: instrument v3's item bank.

The S30 pilot measured the instrument (likelihood selection over candidate
contents) on the 24-source calibration split: 37 selection items, a measured
baseline of 0.478 and a paired variance of 0.087, which put the registered
power requirement at 38 executable items **per category**. At 3 per category the
comparison could not decide a 10% relative effect (§18.6), so the honest next
step was task population rather than arms (§18.8).

This module builds that population from the registered 144-task source suite. It
does not author new tasks and it does not touch the sealed holdback: it derives
more selection items **per source**, each with its own verified answer space:

* ``patch-function`` — the target function is removed from a solved crate and
  the arm chooses its definition among 4-6 candidates. One item per function of
  each Rust file, so a multi-function crate yields several items (they share the
  crate and the visible test; that clustering is measured, not hidden).
* ``patch-file`` — the target file is removed back to its initial state and the
  arm chooses the whole replacement file. One item per file whose code differs.
* ``patch-manifest`` — the manifest is the question (dependency-compat); the
  source code is handed over solved.
* ``diagnostic-identification`` — unchanged from S28; reported on its own
  metric and never pooled with executable success.

Every item carries 4-6 candidates: exactly one correct definition and 3-5
distractors drawn from the declared taxonomy (initial state, declared mutation,
declared naive fix, single-edit revert, operator inversion, synthesized
mutation). Nothing here *asserts* that a distractor fails; `verify-formats`
compiles every candidate and the receipt records the outcome, so a distractor
that passes is a failed item, not a silent one.

Three properties are enforced mechanically rather than promised:

* **the reference position is balanced** across items of a category (a
  deterministic rotation, histogram recorded), so the item bank does not teach a
  position prior;
* **the item population is deduplicated on its prompt** (8-word shingles), and
  the maximum pairwise similarity is reported;
* **difficulty is assigned from a measured quantity** — the number of changed
  lines between the initial and reference state for the target — with the basis
  recorded per item, so the distribution is auditable rather than asserted.

The derivation is deterministic: no timestamps enter a task file, so a rebuild
from the same source tree is byte-identical and `population-check` proves it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil

from . import derive, formats, suite as suite_module

SCHEMA = "selvedge/capability-population/v1"
SCHEMA_CHECK = "selvedge/capability-population-check/v1"
POPULATION_SEED = "selvedge-population-v1"

FAMILY_FUNCTION = "patch-function"
FAMILY_FILE = "patch-file"
FAMILY_MANIFEST = "patch-manifest"
FAMILY_DIAGNOSTIC = "diagnostic-identification"
FAMILIES = (FAMILY_FUNCTION, FAMILY_FILE, FAMILY_MANIFEST, FAMILY_DIAGNOSTIC)

TAG = {
    FAMILY_FUNCTION: "fn",
    FAMILY_FILE: "file",
    FAMILY_MANIFEST: "manifest",
    FAMILY_DIAGNOSTIC: "diag",
}

#: A source may not contribute an unbounded number of items: the mission's
#: candidate rule is one correct definition plus 3-5 plausible distractors.
MAX_CANDIDATES = 6
MIN_CANDIDATES = 4
MAX_DISTRACTORS = 5

#: Candidate kinds, in the priority order they are admitted. The order is part of
#: the derivation contract: a rebuild admits the same distractors in the same
#: order, and the receipt reports the kinds it used.
KIND_REFERENCE = "reference"
KIND_INITIAL = "initial"
KIND_MUTATION = "declared-mutation"
KIND_NAIVE = "declared-naive"
KIND_REVERT = "single-edit-revert"
KIND_INVERSION = "operator-inversion"
KIND_SYNTHETIC = "synthesized-mutation"
KIND_ORDER = (KIND_MUTATION, KIND_NAIVE, KIND_INITIAL, KIND_REVERT,
              KIND_INVERSION, KIND_SYNTHETIC)

#: Operator inversions: each is (find, replace, description). They are applied to
#: the reference definition one at a time; the checker decides whether the result
#: fails, so an inversion that is semantically neutral is pruned, not shipped.
SYNTH_SWAPS = (
    (">=", ">", "the `>=` comparison narrowed to `>`"),
    ("<=", "<", "the `<=` comparison narrowed to `<`"),
    ("==", "!=", "the equality test inverted to `!=`"),
    ("!=", "==", "the inequality test inverted to `==`"),
    (" && ", " || ", "the conjunction loosened to a disjunction"),
    (" || ", " && ", "the disjunction tightened to a conjunction"),
    (" + 1", " + 2", "the increment widened by one"),
    (" - 1", " - 2", "the decrement widened by one"),
    ("..=", "..", "the inclusive range made exclusive"),
    (".rev()", "", "the reversal dropped"),
    (".len()", ".len() - 1", "the length made off by one"),
    ("min(", "max(", "the minimum replaced by a maximum"),
    ("max(", "min(", "the maximum replaced by a minimum"),
    (" += ", " -= ", "the accumulation reversed"),
    ("&mut ", "&", "the mutable borrow downgraded"),
)

#: Regex mutations, applied one at a time to the reference text. These cover the
#: idioms the corpus actually uses (literals, booleans, comparisons, iterator
#: adapters, single-line guards) and each keeps the text shaped like Rust. A
#: pattern whose replacement would produce equivalent code is caught by the
#: checker (the candidate passes) and pruned, never shipped.
SYNTH_PATTERNS = (
    (re.compile(r"\b0\b"), "1", "the literal 0 widened to 1"),
    (re.compile(r"\b1\b"), "2", "the literal 1 widened to 2"),
    (re.compile(r"\b2\b"), "3", "the literal 2 widened to 3"),
    (re.compile(r"\btrue\b"), "false", "`true` replaced by `false`"),
    (re.compile(r"\bfalse\b"), "true", "`false` replaced by `true`"),
    (re.compile(r"\.is_some\(\)"), ".is_none()", "`is_some` inverted to `is_none`"),
    (re.compile(r"\.is_none\(\)"), ".is_some()", "`is_none` inverted to `is_some`"),
    (re.compile(r"\.is_empty\(\)"), ".is_empty() == false",
     "the emptiness test negated"),
    (re.compile(r"\.iter\(\)"), ".iter().rev()", "the iteration reversed"),
    (re.compile(r"\.chars\(\)"), ".chars().rev()", "the character order reversed"),
    (re.compile(r"\.starts_with\("), ".ends_with(", "`starts_with` replaced by `ends_with`"),
    (re.compile(r"\.ends_with\("), ".starts_with(", "`ends_with` replaced by `starts_with`"),
    (re.compile(r"\.map\("), ".filter(", "the map replaced by a filter"),
    (re.compile(r"\.filter\("), ".map(", "the filter replaced by a map"),
    (re.compile(r"\.find\("), ".position(", "`find` replaced by `position`"),
    (re.compile(r"\.take\("), ".skip(", "`take` replaced by `skip`"),
    (re.compile(r"\.skip\("), ".take(", "`skip` replaced by `take`"),
    (re.compile(r"\.next\(\)"), ".last()", "`next` replaced by `last`"),
    (re.compile(r"\.sum::<" ), ".max::<", "the sum replaced by a maximum"),
    (re.compile(r"\.min\("), ".max(", "the minimum replaced by a maximum"),
    (re.compile(r"\.max\("), ".min(", "the maximum replaced by a minimum"),
    (re.compile(r"\.as_ref\(\)"), ".as_mut()", "`as_ref` replaced by `as_mut`"),
    (re.compile(r"\.as_deref\(\)"), ".as_ref()", "`as_deref` replaced by `as_ref`"),
    (re.compile(r"\.unwrap\(\)"), ".unwrap_or_default()",
     "`unwrap` replaced by `unwrap_or_default`"),
    (re.compile(r"\.cloned\(\)"), ".copied()", "`cloned` replaced by `copied`"),
    (re.compile(r"\.to_vec\(\)"), ".to_owned()", "`to_vec` replaced by `to_owned`"),
    (re.compile(r"\.to_string\(\)"), ".as_str()",
     "`to_string` replaced by `as_str`"),
    (re.compile(r"\.clone\(\)"), "", "the clone dropped"),
    (re.compile(r"\?(?=[;,)\s])"), "", "the `?` propagation dropped"),
    (re.compile(r"(\b[a-z_][a-z0-9_]*)\(([a-z_][a-z0-9_]*), ([a-z_][a-z0-9_]*)\)"),
     r"\1(\3, \2)", "two call arguments swapped"),
    (re.compile(r"^(\s*)if (?![(!])(.+) \{$", re.MULTILINE), r"\1if !(\2) {",
     "the guard negated"),
    (re.compile(r"^(\s*)while (.+) \{$", re.MULTILINE), r"\1while !(\2) {",
     "the loop guard negated"),
)

#: Line deletions that keep the braces balanced: a `let` binding or a one-line
#: `return` removed from the body. A dropped guard is the wrong answer an arm
#: actually produces, not a syntax error.
SYNTH_DELETION = re.compile(r"^(\s*)(let .*;|return .*;)\s*$")

#: Difficulty thresholds in changed lines between the initial and reference
#: state of the target (easy = a single-line fix, medium = multi-line, hard =
#: non-local). The basis is recorded per item so the label is auditable.
EASY_MAX_LINES = 2
MEDIUM_MAX_LINES = 10

#: A scored sequence above the instrument's exactness bound is refused by the
#: scorer (§18.4). This is a conservative character screen applied before an item
#: is admitted, so the refusal happens at build time rather than mid-run.
MAX_SCORED_TOKENS = 4096
CHARS_PER_TOKEN_FLOOR = 2.0
DEFAULT_MAX_SCORED_CHARS = int(MAX_SCORED_TOKENS * CHARS_PER_TOKEN_FLOOR)

#: Prompt near-duplicate screen: 8-word shingles over the rendered selection
#: prompt plus every candidate, Jaccard. Declared before the population is built.
PROMPT_WIDTH = 8
PROMPT_NEAR_DUPLICATE = 0.90


class PopulationError(RuntimeError):
    """The population cannot be derived as declared."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "x"


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _changed_lines(left: str, right: str) -> int:
    """Line-level edits (insertions + deletions) between two texts."""
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    matcher = difflib.SequenceMatcher(None, left_lines, right_lines)
    total = 0
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        total += (a_end - a_start) + (b_end - b_start)
    return total


def _difficulty_from_lines(changed: int) -> str:
    if changed <= EASY_MAX_LINES:
        return "easy"
    if changed <= MEDIUM_MAX_LINES:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Rust structure helpers


def _target_spans(text: str) -> list[dict]:
    """Functions outside the crate's own `#[cfg(test)]` module."""
    cutoff = text.find("#[cfg(test)]")
    spans = [span for span in formats._function_spans(text)
             if span["name"] != "main"]
    if cutoff >= 0:
        spans = [span for span in spans if span["start"] < cutoff]
    return spans


def _definition_at(text: str, span: dict) -> str:
    start = text.rfind("\n", 0, span["start"]) + 1
    return text[start:span["body_end"] + 1]


def _name_counts(text: str) -> dict:
    counts: dict[str, int] = {}
    for span in formats._function_spans(text):
        counts[span["name"]] = counts.get(span["name"], 0) + 1
    return counts


def _enclosing_impl(reference: str, span: dict) -> dict:
    """Whether the span sits inside an `impl` block, and whether it is a trait impl.

    Used only to predict which rustc code a missing definition produces, and the
    prediction is then *measured*: `infer_failures` compiles the initial state and
    replaces the prediction with the code the compiler actually reported.
    """
    mask = formats._code_mask(reference)
    depth = 0
    for index in range(span["start"]):
        if not mask[index]:
            continue
        if reference[index] == "{":
            depth += 1
        elif reference[index] == "}":
            depth -= 1
    if depth == 0:
        return {"in_impl": False, "trait_impl": False}
    depth = 0
    header_start = None
    for index in range(span["start"] - 1, -1, -1):
        if not mask[index]:
            continue
        if reference[index] == "}":
            depth += 1
        elif reference[index] == "{":
            depth -= 1
            if depth == 0:
                header_end = index
                start = reference.rfind("impl", 0, header_end)
                if start >= 0:
                    header_start = start
                break
    if header_start is None:
        return {"in_impl": True, "trait_impl": False}
    header = reference[header_start:header_end]
    return {"in_impl": True, "trait_impl": " for " in header}


def _predicted_missing_code(reference: str, span: dict) -> str:
    placement = _enclosing_impl(reference, span)
    if not placement["in_impl"]:
        return "E0425"
    return "E0046" if placement["trait_impl"] else "E0599"


# ---------------------------------------------------------------------------
# candidate construction


def _synthetic_variants(definition: str, function: str, limit: int) -> list[tuple]:
    """One-at-a-time textual mutations of a definition, each still a definition."""
    variants: list[tuple] = []
    seen: set = set()

    def admit(candidate: str, description: str) -> None:
        if candidate.strip() == definition.strip():
            return
        key = formats.normalise_body(candidate)
        if not key or key in seen:
            return
        try:
            formats.function_body(candidate, function)
        except formats.FormatError:
            return
        seen.add(key)
        variants.append((candidate, f"{description} ({KIND_SYNTHETIC})",
                         KIND_SYNTHETIC))

    for find, replace, description in SYNTH_SWAPS:
        if find in definition:
            admit(definition.replace(find, replace, 1), description)
        if len(variants) >= limit:
            return variants
    for pattern, replacement, description in SYNTH_PATTERNS:
        candidate = pattern.sub(replacement, definition, count=1)
        if candidate != definition:
            admit(candidate, description)
        if len(variants) >= limit:
            return variants
    lines = definition.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not SYNTH_DELETION.match(line.rstrip("\n")):
            continue
        admit("".join(lines[:index] + lines[index + 1:]),
              "a body statement dropped")
        if len(variants) >= limit:
            return variants
    return variants


def _sibling_body_variants(definition: str, reference: str, name: str,
                           limit: int) -> list[tuple]:
    """Definitions whose body is taken from a sibling function of the crate.

    The last generic fallback, for a short accessor that offers no textual handle:
    the wrong answer an arm reaches for when it confuses two similar helpers. The
    candidate keeps this function's signature and takes another function's body, so
    the shape is right and the behaviour is not.
    """
    own = formats._function_spans(definition)
    if not own:
        return []
    span = own[0]
    head = definition[:span["body_start"]]
    tail = definition[span["body_end"]:]
    variants: list[tuple] = []
    seen: set = set()
    for other in formats._function_spans(reference):
        if other["name"] == name:
            continue
        body = reference[other["body_start"]:other["body_end"]]
        if not body.strip():
            continue
        candidate = head + body + tail
        key = formats.normalise_body(candidate)
        if key in seen or key == formats.normalise_body(definition):
            continue
        seen.add(key)
        variants.append((candidate,
                        f"the body of the sibling function `{other['name']}` "
                        f"({KIND_SYNTHETIC})", KIND_SYNTHETIC))
        if len(variants) >= limit:
            break
    return variants


def _function_candidates(source, entry: str, reference: str, initial: str,
                         span: dict) -> dict:
    """(reference definition, prioritized distractors) for one function.

    A distractor is admitted only when it differs from the reference definition
    and from every other candidate after whitespace normalisation; whether it
    *fails* is decided later by the checker, never here.
    """
    name = span["name"]
    definition = _definition_at(reference, span).strip()
    if not definition:
        return {"error": "the function definition is empty"}
    counts = _name_counts(reference)
    if counts.get(name, 0) != 1:
        return {"error": (f"the name {name!r} is defined {counts.get(name)} times in "
                          "the reference; splicing one leaves the other resolving")}
    mutation = source.metadata.get("mutation")
    naive = source.metadata.get("naive")
    initial_definition = None
    if initial and counts.get(name, 0) == 1 and _name_counts(initial).get(name, 0) == 1:
        try:
            initial_definition = formats.function_definition(initial, name).strip()
        except formats.FormatError:
            initial_definition = None

    seen = {formats.normalise_body(definition)}
    admitted: list[tuple] = []

    def admit(text: str, provenance: str, kind: str) -> None:
        text = text.strip()
        if not text:
            return
        key = formats.normalise_body(text)
        if not key or key in seen:
            return
        try:
            formats.function_body(text, name)
        except formats.FormatError:
            return
        seen.add(key)
        admitted.append((text, provenance, kind))

    if isinstance(mutation, dict) and mutation.get("file") == entry \
            and mutation.get("find") and mutation["find"] in definition:
        admit(definition.replace(mutation["find"], mutation["replace"], 1),
              "the source's declared mutation (declared wrong)", KIND_MUTATION)
    if isinstance(naive, dict) and naive.get("file") == entry \
            and naive.get("find") and naive["find"] in definition:
        admit(definition.replace(naive["find"], naive["replace"], 1),
              "the source's declared naive fix (declared wrong)", KIND_NAIVE)
    if initial_definition is not None and \
            formats.normalise_body(initial_definition) != formats.normalise_body(
                definition):
        admit(initial_definition,
              "the source's initial definition (declared wrong)", KIND_INITIAL)
        reverted = derive._revert_regions(definition, initial_definition, limit=1)
        if reverted:
            admit(reverted[0],
                  "the reference with its first edit reverted (declared wrong)",
                  KIND_REVERT)
    for variant, description, kind in _synthetic_variants(
            definition, name, MAX_DISTRACTORS - len(admitted)):
        admit(variant, description, kind)
    if len(admitted) < MAX_DISTRACTORS:
        for variant, description, kind in _sibling_body_variants(
                definition, reference, name, MAX_DISTRACTORS - len(admitted)):
            admit(variant, description, kind)
    return {"definition": definition, "distractors": admitted,
            "initial_definition": initial_definition}


def _file_candidates(source, entry: str, reference: str, initial: str) -> dict:
    """(reference file, prioritized distractors) for a whole-file patch item."""
    if derive._code_only(initial) == derive._code_only(reference):
        return {"error": "the initial and reference files carry the same code"}
    mutation = source.metadata.get("mutation")
    seen = {formats.normalise_body(reference)}
    admitted: list[tuple] = []

    def admit(text: str, provenance: str, kind: str) -> None:
        text = text.strip()
        key = formats.normalise_body(text)
        if not key or key in seen or len(text.splitlines()) < 3:
            return
        seen.add(key)
        admitted.append((text + "\n", provenance, kind))

    if isinstance(mutation, dict) and mutation.get("file") == entry \
            and mutation.get("find") and mutation["find"] in reference:
        admit(reference.replace(mutation["find"], mutation["replace"], 1),
              "the source's declared mutation (declared wrong)", KIND_MUTATION)
    admit(initial, "the source's initial file (declared wrong)", KIND_INITIAL)
    reverted = derive._revert_regions(reference, initial, limit=3)
    for index, candidate in enumerate(reverted):
        admit(candidate,
              "the reference with its first edit reverted (declared wrong)"
              if index == 0 else "the reference with another edit reverted "
                                 "(declared wrong)",
              KIND_REVERT)
    if len(admitted) < MAX_DISTRACTORS:
        for find, replace, description in SYNTH_SWAPS:
            if find not in reference:
                continue
            admit(reference.replace(find, replace, 1),
                  f"{description} ({KIND_INVERSION})", KIND_INVERSION)
            if len(admitted) >= MAX_DISTRACTORS:
                break
    if len(admitted) < MAX_DISTRACTORS:
        for pattern, replacement, description in SYNTH_PATTERNS:
            candidate = pattern.sub(replacement, reference, count=1)
            if candidate != reference:
                admit(candidate, f"{description} ({KIND_SYNTHETIC})",
                      KIND_SYNTHETIC)
            if len(admitted) >= MAX_DISTRACTORS:
                break
    if len(admitted) < MAX_DISTRACTORS:
        lines = reference.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not SYNTH_DELETION.match(line.rstrip("\n")):
                continue
            admit("".join(lines[:index] + lines[index + 1:]),
                  f"a statement dropped from the reference file "
                  f"({KIND_SYNTHETIC})", KIND_SYNTHETIC)
            if len(admitted) >= MAX_DISTRACTORS:
                break
    return {"definition": reference.strip() + "\n", "distractors": admitted}


def _entries(source) -> list[str]:
    """Rust files present in both the initial and reference states."""
    return sorted(path for path in source.reference_files
                  if path.endswith(".rs") and path in source.initial_files)


def _visible_module(source) -> tuple[str, str] | None:
    """The source's hidden test file as an inline `mod visible` for the template.

The S28 helper (`derive._visible_test`) extracts the first test and its free
    function helpers and strips the crate imports. That is right for a single-file
    crate and wrong for two cases the population actually hits:

    * a multi-file crate whose visible test calls a symbol that lives in another
      module — `use super::*` does not bring it into scope, so the reference state
      stops compiling (measured: six items in the first verification pass);
    * a hidden test that declares an `impl` inside a test — the helper extractor
      lifts the trait methods out of the impl block, which is not valid Rust
      (measured: the `traits-04` items).

    The whole hidden file is therefore copied, with the crate's own name rewritten
to `crate::`, which resolves from any module of the crate. Nothing is extracted,
    so nothing can be extracted wrongly.
    """
    if "tests/check.rs" not in source.hidden_files:
        return None
    text = source.read_hidden("tests/check.rs")
    if not text.strip():
        return None
    crate = derive._crate_name(source)
    rewritten = text
    if crate:
        rewritten = re.sub(rf"\b{re.escape(crate)}::", "crate::", rewritten)
    body = "\n".join(("    " + line) if line.strip() else line
                     for line in rewritten.rstrip().splitlines())
    wrapped = "#[cfg(test)]\nmod visible {\n" + body + "\n}\n"
    return wrapped, "the source's hidden test file, shown as an inline visible test"


# ---------------------------------------------------------------------------
# failure inference and reference check (need cargo; run in the rust court)


def _infer_failure(task, workroot: str) -> dict:
    """Compile and test the item's initial state; name the failure it produces.

    A predicted rustc code is not evidence, so the code is measured. A state that
    compiles and passes its tests proves nothing and is dropped; a state that
    compiles but fails a hidden test is a `hidden_test_failure`, which is the
    declared kind for the semantic-repair sources.
    """
    from . import checkers, sandbox

    prepared, problems = checkers.prepare(task, {}, workroot, allow_empty=True)
    if problems:
        return {"kind": None, "code": None,
                "reason": f"the initial state could not be prepared: {problems}"}
    check = sandbox.run_bounded(
        ["cargo", "check", "--offline", "--quiet", "--all-targets"], prepared,
        task.limits)
    text = check.stdout_tail + check.stderr_tail
    if check.status == sandbox.STATUS_PASS:
        test = sandbox.run_bounded(["cargo", "test", "--offline", "--quiet"],
                                   prepared, task.limits)
        sandbox.remove_tree(prepared)
        if "test result: FAILED" in (test.stdout_tail + test.stderr_tail):
            return {"kind": "hidden_test_failure", "code": None,
                    "must_contain": ["test result: FAILED"],
                    "status": test.status}
        return {"kind": None, "code": None,
                "reason": ("the initial state compiles and its tests pass: reverting "
                           "this target breaks nothing the crate or its tests use"),
                "status": check.status}
    sandbox.remove_tree(prepared)
    match = formats.ERROR_CODE.search(text)
    if match is None:
        return {"kind": None, "code": None,
                "reason": (f"the failure carries no rustc error code "
                           f"({check.status}): {text.strip()[-300:]}"),
                "status": check.status}
    return {"kind": "compile_error", "code": match.group(0),
            "must_contain": [match.group(0)], "status": check.status}


def _reference_passes(task, workroot: str) -> dict:
    """Compile and test the item's reference state; it must pass."""
    from . import checkers

    outcome = checkers.run_check(task, task.reference_solution(), workroot)
    status = outcome.get("status")
    evidence = outcome.get("evidence") or {}
    return {"status": status, "passes": status == "pass",
            "stderr_tail": (evidence.get("stderr_tail") or "")[-400:]}


# ---------------------------------------------------------------------------
# item construction


def _item_payload(source, *, family: str, item_id: str, category: str,
                  entry: str, spec: dict, intro: str, failure: dict,
                  difficulty: str, difficulty_basis: str, fix_lines: int,
                  source_split: str, target: str | None, visible: str | None,
                  candidates: list[dict], candidate_key: str,
                  extra: dict | None = None) -> dict:
    final = []
    for index, candidate in enumerate(candidates):
        rendered = {"label": chr(ord("A") + index),
                    "reference": bool(candidate["reference"]),
                    "provenance": candidate["provenance"],
                    "kind": candidate["kind"]}
        rendered[candidate_key] = candidate["text"]
        final.append(rendered)
    spec = dict(spec)
    spec["candidates"] = final
    return {
        "source_id": source.id,
        "source_tree_sha256": source.tree_sha256,
        "source_difficulty": source.metadata.get("difficulty"),
        "source_capability": source.metadata["capability"],
        "checker_kind": source.checker_kind,
        "manifest_policy": source.manifest_policy,
        "id": item_id,
        "family": family,
        "category": category,
        "entry": entry,
        "spec": spec,
        "intro": intro,
        "failure": failure,
        "difficulty": difficulty,
        "difficulty_basis": difficulty_basis,
        "fix_lines": fix_lines,
        "source_split": source_split,
        "target": target,
        "visible": visible,
        "candidates": final,
        "extra": extra or {},
    }


def _function_item(source, *, entry: str, reference: str, initial: str, span: dict,
                   ordinal: int, source_split: str) -> dict:
    name = span["name"]
    built = _function_candidates(source, entry, reference, initial, span)
    if "error" in built:
        return {"skipped": f"{source.id} {entry} {name}: {built['error']}"}
    distractors = built["distractors"]
    if len(distractors) < MIN_CANDIDATES - 1:
        return {"skipped": (f"{source.id} {entry} {name}: only {len(distractors)} "
                            f"distinct distractors; {MIN_CANDIDATES - 1} are required")}
    visible = _visible_module(source) or derive._visible_test(source, entry)
    if visible is None:
        return {"skipped": f"{source.id} {entry} {name}: no visible test"}
    block, visible_provenance = visible
    marker_count = reference.count(formats.FUNCTION_MARKER)
    if marker_count:
        return {"skipped": f"{source.id} {entry} {name}: marker collision"}
    start = reference.rfind("\n", 0, span["start"]) + 1
    template = (reference[:start] + formats.FUNCTION_MARKER
                + reference[span["body_end"] + 1:]).rstrip() + "\n\n" + block.strip() + "\n"
    definition = built["definition"]

    initial_definition = built["initial_definition"]
    if initial_definition is not None and \
            derive._code_only(initial_definition) != derive._code_only(definition):
        fix_lines = _changed_lines(initial_definition, definition)
        basis = "source-fix"
        difficulty = _difficulty_from_lines(fix_lines)
    else:
        nearest = min((_changed_lines(text, definition) for text, _p, _k in distractors),
                      default=0)
        fix_lines = nearest
        basis = "nearest-distractor"
        # A near-miss distractor is the hard case: the discrimination is one line.
        difficulty = ("hard" if nearest <= 1 else
                      "medium" if nearest <= 3 else "easy")

    item_id = f"{source.id}--{TAG[FAMILY_FUNCTION]}--{_slug(entry)}--{_slug(name)}"
    if ordinal:
        item_id += f"-{ordinal + 1}"
    candidates = [{"text": definition, "reference": True, "kind": KIND_REFERENCE,
                   "provenance": "reference"}] + [
        {"text": text, "reference": False, "kind": kind, "provenance": provenance}
        for text, provenance, kind in distractors[:MAX_DISTRACTORS]]
    spec = {
        "kind": formats.FORMAT_PATCH,
        "template": entry,
        "function": name,
        "candidates": candidates,
        "visible_test": visible_provenance,
    }
    intro = (f"Choose the one candidate definition for `{name}` that satisfies the "
             "visible test and the hidden tests. The others compile-or-not and fail "
             "at least one test.")
    payload = _item_payload(
        source, family=FAMILY_FUNCTION, item_id=item_id, category=source.category,
        entry=entry, spec=spec, intro=intro,
        failure={"kind": "compile_error", "must_contain": ["E0425"],
                 "command": "all-targets", "inferred": True},
        difficulty=difficulty, difficulty_basis=basis, fix_lines=fix_lines,
        source_split=source_split, target=name, visible=visible_provenance,
        candidates=candidates, candidate_key="function",
        extra={"mutation": {"file": entry, "find": definition,
                            "replace": distractors[0][0]}})
    payload["copy"] = {
        "initial": {entry: template, **derive._solved_files(source, entry)},
        "reference": {entry: formats.splice_function(template, definition)},
        "hidden": {path: source.read_hidden(path) for path in source.hidden_files},
    }
    return payload


def _file_item(source, *, entry: str, reference: str, initial: str,
               source_split: str) -> dict:
    built = _file_candidates(source, entry, reference, initial)
    if "error" in built:
        return {"skipped": f"{source.id} {entry}: {built['error']}"}
    distractors = built["distractors"]
    if len(distractors) < MIN_CANDIDATES - 1:
        return {"skipped": (f"{source.id} {entry}: only {len(distractors)} distinct "
                            f"file distractors; {MIN_CANDIDATES - 1} are required")}
    item_id = f"{source.id}--{TAG[FAMILY_FILE]}--{_slug(entry)}"
    candidates = [{"text": built["definition"], "reference": True,
                   "kind": KIND_REFERENCE, "provenance": "reference"}] + [
        {"text": text, "reference": False, "kind": kind, "provenance": provenance}
        for text, provenance, kind in distractors[:MAX_DISTRACTORS]]
    spec = {"kind": formats.FORMAT_PATCH, "target": "file", "template": entry,
            "candidates": candidates}
    intro = ("Choose the one replacement file that makes the crate build and its tests "
             "pass. Exactly one candidate does.")
    differing = []
    reference_spans = {span["name"]: _definition_at(reference, span) for span in
                       formats._function_spans(reference)}
    initial_spans = {span["name"]: _definition_at(initial, span) for span in
                     formats._function_spans(initial)}
    for name, text in reference_spans.items():
        if name not in initial_spans or \
                formats.normalise_body(initial_spans[name]) != formats.normalise_body(text):
            differing.append(name)
    fix_lines = _changed_lines(initial, reference)
    basis = "source-fix"
    difficulty = ("hard" if len(differing) > 1 or fix_lines > MEDIUM_MAX_LINES
                  else "medium")
    payload = _item_payload(
        source, family=FAMILY_FILE, item_id=item_id, category=source.category,
        entry=entry, spec=spec, intro=intro,
        failure={"kind": "compile_error", "must_contain": ["E0425"],
                 "command": "all-targets", "inferred": True},
        difficulty=difficulty, difficulty_basis=basis, fix_lines=fix_lines,
        source_split=source_split, target=entry, visible=None, candidates=candidates,
        candidate_key="content",
        extra={"mutation": {"file": entry, "find": reference.strip(),
                            "replace": distractors[0][0].strip()},
               "differing_functions": len(differing)})
    solved = derive._solved_files(source, entry)
    payload["copy"] = {
        "initial": {entry: initial, **solved},
        # The reference holds the target file only: `prepare` already copies the
        # solved initial tree, and a locked manifest may not be part of an answer.
        "reference": {entry: reference},
        "hidden": {path: source.read_hidden(path) for path in source.hidden_files},
    }
    return payload


def _write_item(root: str, payload: dict) -> str:
    """Write one item's tree; return its id."""
    from . import suite as suite_module

    item_root = os.path.join(root, "tasks", payload["id"])
    copy = payload["copy"]
    for relative, content in copy["initial"].items():
        _write(os.path.join(item_root, "initial", relative), content)
    for relative, content in copy["reference"].items():
        _write(os.path.join(item_root, "reference", relative), content)
    for relative, content in copy["hidden"].items():
        _write(os.path.join(item_root, "hidden", relative), content)
    metadata = {
        "schema": suite_module.SCHEMA,
        "id": payload["id"],
        "kind": "suite",
        "category": payload["category"],
        "capability": (f"{payload['source_capability']} "
                       f"(format: {formats.FORMAT_PATCH}; family: {payload['family']})"),
        "license": suite_module.LICENSE,
        "provenance": (
            f"derived from fixtures/capability/tasks/{payload['source_id']} "
            f"(tree {payload['source_tree_sha256']}) by "
            "oracle/capability/population.py; the source's hidden tests, reference "
            "code and licence, re-presented as a selection item (S31 population v1)"),
        "entry": payload["entry"],
        "checker": {"kind": payload["checker_kind"]},
        "difficulty": payload["difficulty"],
        "format": formats.FORMAT_PATCH,
        "format_spec": payload["spec"],
        "manifest_policy": payload["manifest_policy"],
        "extraction_policy": "tolerant",
        "initial_failure": payload["failure"],
        "prompt_intro": payload["intro"],
        "derived_from": {
            "source_id": payload["source_id"],
            "source_tree_sha256": payload["source_tree_sha256"],
            "source_difficulty": payload["source_difficulty"],
            "source_split": payload["source_split"],
            "populator": "oracle/capability/population.py",
            "population": "v1",
            "family": payload["family"],
            "target": payload["target"],
            "difficulty_basis": payload["difficulty_basis"],
            "fix_lines": payload["fix_lines"],
            "candidate_kinds": [candidate["kind"] for candidate in payload["candidates"]],
        },
    }
    metadata.update(payload["extra"])
    _write_json(os.path.join(item_root, "task.json"), metadata)
    return payload["id"]


def _adopt(root: str, built_root: str, *, item_id: str, family: str,
           source_split: str, difficulty: str, difficulty_basis: str,
           fix_lines: int, target: str | None) -> dict:
    """Move a derive.py-built item to its population id and annotate it."""
    os.makedirs(os.path.join(root, "tasks"), exist_ok=True)
    destination = os.path.join(root, "tasks", item_id)
    if os.path.isdir(built_root):
        shutil.move(built_root, destination)
    path = os.path.join(destination, "task.json")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["id"] = item_id
    metadata["difficulty"] = difficulty
    derived = metadata.setdefault("derived_from", {})
    derived["family"] = family
    derived["source_split"] = source_split
    derived["target"] = target
    derived["difficulty_basis"] = difficulty_basis
    derived["fix_lines"] = fix_lines
    derived["populator"] = "oracle/capability/population.py"
    derived["population"] = "v1"
    _write_json(path, metadata)
    return metadata


# ---------------------------------------------------------------------------
# position balancing


def _balance(root: str, item_ids_by_family: dict) -> dict:
    """Rotate every item's candidate/option order so positions are balanced.

    The rotation is deterministic: items of a category are ordered by id and the
    reference (or the registered option) takes position `index % n`. Rotation
    keeps the distractors' relative order, so nothing but the position changes.
    """
    histogram: dict = {}
    for group, ids in item_ids_by_family.items():
        by_category: dict[str, list] = {}
        for item_id in ids:
            path = os.path.join(root, "tasks", item_id, "task.json")
            with open(path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            by_category.setdefault(metadata["category"], []).append((item_id, path))
        for category in sorted(by_category):
            entries = sorted(by_category[category])
            counter = 0
            bucket = histogram.setdefault(group, {}).setdefault(
                category, {"items": 0, "positions": {}, "candidates": {}})
            for item_id, path in entries:
                with open(path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                spec = metadata["format_spec"]
                if "candidates" in spec:
                    candidates = spec["candidates"]
                    reference_index = next(
                        index for index, candidate in enumerate(candidates)
                        if candidate.get("reference"))
                    position = counter % len(candidates)
                    shift = (reference_index - position) % len(candidates)
                    rotated = candidates[shift:] + candidates[:shift]
                    for index, candidate in enumerate(rotated):
                        candidate["label"] = chr(ord("A") + index)
                    spec["candidates"] = rotated
                else:
                    options = spec["options"]
                    reference_index = next(
                        index for index, option in enumerate(options)
                        if option["consequence"]["kind"] == "reference")
                    position = counter % len(options)
                    shift = (reference_index - position) % len(options)
                    spec["options"] = options[shift:] + options[:shift]
                _write_json(path, metadata)
                bucket["items"] += 1
                bucket["positions"][str(position)] = bucket["positions"].get(
                    str(position), 0) + 1
                bucket["candidates"][str(len(spec.get("candidates")
                                             or spec.get("options")))] = (
                    bucket["candidates"].get(
                        str(len(spec.get("candidates") or spec.get("options"))), 0) + 1)
                counter += 1
    for group, categories in histogram.items():
        for category, bucket in categories.items():
            counts = list(bucket["positions"].values())
            bucket["max_share"] = round(max(counts) / max(1, bucket["items"]), 6)
            bucket["distinct_positions"] = len(bucket["positions"])
    return histogram


# ---------------------------------------------------------------------------
# derivation


def _finalise(item_path: str, item_id: str, workroot: str) -> tuple[bool, str | None, dict | None]:
    """Measure the item's own failure and prove its reference passes.

    Two compiles per item, at build time: the initial state (which must fail, and
    whose failure is *named* from what the compiler said, not predicted) and the
    reference state (which must pass). An item that fails either screen is dropped
    here with its reason recorded, so `verify-formats` is a confirmation rather than
    a discovery.
    """
    from . import suite as suite_module

    task = suite_module.load_task(item_path)
    failure = _infer_failure(task, workroot)
    if failure.get("kind") is None:
        return False, f"{item_id}: {failure.get('reason')}", None
    reference = _reference_passes(task, workroot)
    if not reference["passes"]:
        return False, (f"{item_id}: the reference state does not pass "
                       f"({reference['status']}): "
                       f"{reference['stderr_tail'].strip()[:240]}"), None
    metadata_path = os.path.join(item_path, "task.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    block = {"kind": failure["kind"],
             "must_contain": list(failure["must_contain"]),
             "inferred_from": ("the compiler's own output on this item's initial "
                               "state, measured by population.derive")}
    if failure["kind"] == "compile_error":
        block["command"] = "all-targets"
    metadata["initial_failure"] = block
    _write_json(metadata_path, metadata)
    return True, None, failure


def derive_population(suite_root: str, out_root: str, *, force: bool = False,
                      max_scored_chars: int = DEFAULT_MAX_SCORED_CHARS,
                      workroot: str = "/work/target/capability-population",
                      infer_failures: bool = True,
                      only_sources: list | None = None) -> dict:
    from . import suite as suite_module

    if os.path.isdir(out_root) and os.listdir(out_root) and not force:
        raise PopulationError(f"{out_root} is not empty; pass --force to rebuild")
    if os.path.isdir(out_root) and force:
        shutil.rmtree(out_root)
    os.makedirs(os.path.join(out_root, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "trio"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "probes"), exist_ok=True)

    collections = suite_module.load_suite(suite_root)
    sources = sorted(collections["suite"], key=lambda task: task.id)
    if not sources:
        raise PopulationError("the source suite has no suite tasks")
    if only_sources:
        wanted = set(only_sources)
        sources = [source for source in sources if source.id in wanted]
        missing = wanted - {source.id for source in sources}
        if missing:
            raise PopulationError(f"unknown sources: {sorted(missing)}")
    partition = derive.partition_sources(suite_root, 3)
    calibration_ids = {task.id
                       for tasks in partition[derive.SPLIT_CALIBRATION].values()
                       for task in tasks}
    sealed_root = os.path.join(suite_root, "sealed")
    sealed_ids = set()
    if os.path.isdir(sealed_root):
        sealed_ids = {task.id for task in suite_module.load_collection(sealed_root)}

    items: list[dict] = []
    skipped: list[str] = []
    ids_by_family: dict = {FAMILY_FUNCTION: [], FAMILY_FILE: [],
                           FAMILY_MANIFEST: [], FAMILY_DIAGNOSTIC: []}
    inferred: list[dict] = []
    too_long: list[dict] = []

    for source in sources:
        if source.id in sealed_ids:
            raise PopulationError(f"the source {source.id} is sealed")
        source_split = ("calibration" if source.id in calibration_ids else "scored")
        base = {"source_split": source_split}
        entries = _entries(source)
        for entry in entries:
            reference = source.read_reference(entry)
            initial = source.read_initial(entry)
            spans = _target_spans(reference)
            counts = _name_counts(reference)
            ordinal_seen = 0
            for span in spans:
                name = span["name"]
                if counts.get(name, 0) != 1:
                    skipped.append(f"{source.id} {entry} {name}: duplicate function name")
                    continue
                payload = _function_item(
                    source, entry=entry, reference=reference, initial=initial,
                    span=span, ordinal=ordinal_seen, source_split=source_split)
                if "skipped" in payload:
                    skipped.append(payload["skipped"])
                    continue
                payload.update(base)
                item_id = payload["id"]
                item_path = os.path.join(out_root, "tasks", item_id)
                _write_item(out_root, payload)
                if infer_failures:
                    ok, reason, failure = _finalise(item_path, item_id, workroot)
                    if not ok:
                        skipped.append(reason)
                        shutil.rmtree(item_path)
                        continue
                    inferred.append({"id": item_id, **failure})
                items.append({"id": item_id, "family": FAMILY_FUNCTION,
                              "category": source.category, "source": source.id,
                              "entry": entry, "target": name,
                              "difficulty": payload["difficulty"],
                              "difficulty_basis": payload["difficulty_basis"],
                              "fix_lines": payload["fix_lines"],
                              "source_split": source_split,
                              "candidate_kinds": [c["kind"] for c in
                                                  payload["candidates"]]})
                ids_by_family[FAMILY_FUNCTION].append(item_id)
                ordinal_seen += 1
            file_payload = _file_item(source, entry=entry, reference=reference,
                                      initial=initial, source_split=source_split)
            if "skipped" in file_payload:
                skipped.append(file_payload["skipped"])
            else:
                file_payload.update(base)
                file_id = file_payload["id"]
                file_path = os.path.join(out_root, "tasks", file_id)
                _write_item(out_root, file_payload)
                if infer_failures:
                    ok, reason, failure = _finalise(file_path, file_id, workroot)
                    if not ok:
                        skipped.append(reason)
                        shutil.rmtree(file_path)
                        continue
                    inferred.append({"id": file_id, **failure})
                items.append({"id": file_id, "family": FAMILY_FILE,
                              "category": source.category, "source": source.id,
                              "entry": entry, "target": entry,
                              "difficulty": file_payload["difficulty"],
                              "difficulty_basis": file_payload["difficulty_basis"],
                              "fix_lines": file_payload["fix_lines"],
                              "source_split": source_split,
                              "candidate_kinds": [c["kind"] for c in
                                                  file_payload["candidates"]]})
                ids_by_family[FAMILY_FILE].append(file_id)

        if "Cargo.toml" in source.reference_files and \
                "Cargo.toml" in source.initial_files and \
                source.manifest_policy == "arm" and \
                derive._code_only(source.read_initial("Cargo.toml")) != \
                derive._code_only(source.read_reference("Cargo.toml")):
            manifest_root = os.path.join(out_root, "tasks",
                                         f"{source.id}--{TAG[FAMILY_MANIFEST]}")
            built = derive._derive_manifest_patch(source, manifest_root)
            if built is None:
                skipped.append(f"{source.id}: no manifest patch could be derived")
            else:
                manifest_lines = _changed_lines(source.read_initial("Cargo.toml"),
                                                source.read_reference("Cargo.toml"))
                difficulty = ("easy" if manifest_lines <= EASY_MAX_LINES else "medium")
                item_id = f"{source.id}--{TAG[FAMILY_MANIFEST]}"
                _adopt(out_root, manifest_root, item_id=item_id,
                       family=FAMILY_MANIFEST, source_split=source_split,
                       difficulty=difficulty, difficulty_basis="source-fix",
                       fix_lines=manifest_lines, target="Cargo.toml")
                items.append({"id": item_id, "family": FAMILY_MANIFEST,
                              "category": source.category, "source": source.id,
                              "entry": "Cargo.toml", "target": "Cargo.toml",
                              "difficulty": difficulty,
                              "difficulty_basis": "source-fix",
                              "fix_lines": manifest_lines,
                              "source_split": source_split,
                              "candidate_kinds": ["reference"]})
                ids_by_family[FAMILY_MANIFEST].append(item_id)

        diagnostic_root = os.path.join(out_root, "tasks",
                                       f"{source.id}--{TAG[FAMILY_DIAGNOSTIC]}")
        built = derive._derive_diagnostic(source, diagnostic_root)
        if built is None:
            pass
        else:
            item_id = f"{source.id}--{TAG[FAMILY_DIAGNOSTIC]}"
            difficulty = source.metadata.get("difficulty", "medium")
            _adopt(out_root, diagnostic_root, item_id=item_id,
                   family=FAMILY_DIAGNOSTIC, source_split=source_split,
                   difficulty=difficulty, difficulty_basis="source-declared",
                   fix_lines=0, target=built["correct_id"])
            items.append({"id": item_id, "family": FAMILY_DIAGNOSTIC,
                          "category": source.category, "source": source.id,
                          "entry": source.entry, "target": built["correct_id"],
                          "difficulty": difficulty,
                          "difficulty_basis": "source-declared",
                          "fix_lines": 0, "source_split": source_split,
                          "candidate_kinds": ["reference"]})
            ids_by_family[FAMILY_DIAGNOSTIC].append(item_id)

    histogram = _balance(out_root, {
        "patch": ids_by_family[FAMILY_FUNCTION] + ids_by_family[FAMILY_FILE]
                 + ids_by_family[FAMILY_MANIFEST],
        "diagnostic": ids_by_family[FAMILY_DIAGNOSTIC],
    })

    manifest = suite_module.build_manifest(out_root)
    scored_chars = _scored_size(out_root, too_long, max_scored_chars)
    if too_long:
        for entry in too_long:
            path = os.path.join(out_root, "tasks", entry["id"])
            if os.path.isdir(path):
                shutil.rmtree(path)
        items = [item for item in items
                 if item["id"] not in {entry["id"] for entry in too_long}]
        for family in ids_by_family:
            ids_by_family[family] = [item_id for item_id in ids_by_family[family]
                                     if item_id not in {entry["id"]
                                                        for entry in too_long}]
        manifest = suite_module.build_manifest(out_root)

    suite_module.write_manifest(out_root, manifest)
    actual = [_describe(task) for task in suite_module.load_collection(
        os.path.join(out_root, "tasks"))]
    payload = {
        "schema": SCHEMA,
        "stamp": derive._stamp_now(),
        "created_utc": derive._utc_now(),
        "revision": derive._revision(),
        "command": " ".join(os.sys.argv) if os.sys else None,
        "seed": POPULATION_SEED,
        "source_suite": {
            "root": os.path.abspath(suite_root),
            "suite_sha256": suite_module.build_manifest(suite_root)["suite_sha256"],
            "sources": len(sources),
        },
        "split_rule": {
            "calibration": ("the S28 rule, sha256(seed:calibration:task_id) ascending, "
                            "first 3 per category; retained as an item attribute, not "
                            "as a holdout"),
            "sealed_holdback": "never read; the population derives from tasks/ only",
        },
        "rules": {
            "candidates": f"{MIN_CANDIDATES}-{MAX_CANDIDATES} per item, exactly one "
                          "reference",
            "distractor_kinds": list(KIND_ORDER),
            "difficulty": {
                "easy": f"<= {EASY_MAX_LINES} changed lines",
                "medium": f"<= {MEDIUM_MAX_LINES} changed lines (or a whole-file item "
                          "with one differing function)",
                "hard": f"> {MEDIUM_MAX_LINES} changed lines or a non-local fix",
                "nearest_distractor_basis": ("an item whose target does not differ "
                                             "between the initial and reference state "
                                             "is labelled by the distance to its "
                                             "nearest distractor: 1 line hard, 2-3 "
                                             "medium, else easy"),
            },
            "positions": "reference position = item index mod candidates, per category",
            "prompt_near_duplicate": {
                "metric": f"set Jaccard over {PROMPT_WIDTH}-word shingles",
                "reject_at": PROMPT_NEAR_DUPLICATE,
            },
            "max_scored_chars": max_scored_chars,
            "max_scored_tokens": MAX_SCORED_TOKENS,
        },
        "counts": _counts(actual),
        "positions": histogram,
        "scored_size": scored_chars,
        "items": sorted(actual, key=lambda item: item["id"]),
        "skipped": sorted(skipped),
        "failure_inference": inferred,
        "dropped_too_long": too_long,
        "manifest_sha256": manifest["suite_sha256"],
        "verified": False,
        "note": ("no candidate is claimed to fail here: verify-formats compiles every "
                 "candidate and the verification receipt is the evidence"),
    }
    _write_json(os.path.join(out_root, "POPULATION.json"), payload)
    _write(os.path.join(out_root, "README.md"), README)
    return payload


def _describe(task) -> dict:
    derived = task.metadata.get("derived_from") or {}
    spec = task.metadata.get("format_spec") or {}
    candidates = spec.get("candidates") or spec.get("options") or []
    return {
        "id": task.id,
        "family": derived.get("family"),
        "category": task.category,
        "format": formats.task_format(task),
        "source": derived.get("source_id"),
        "source_split": derived.get("source_split"),
        "target": derived.get("target"),
        "difficulty": task.metadata.get("difficulty"),
        "difficulty_basis": derived.get("difficulty_basis"),
        "fix_lines": derived.get("fix_lines"),
        "candidates": len(candidates),
        "candidate_kinds": derived.get("candidate_kinds"),
        "tree_sha256": task.tree_sha256,
    }


def _positions(root: str) -> dict:
    """Reference-position and candidate-count histograms, read from the tree."""
    from . import suite as suite_module

    histogram: dict = {}
    for task in suite_module.load_collection(os.path.join(root, "tasks")):
        spec = task.metadata.get("format_spec") or {}
        family = (task.metadata.get("derived_from") or {}).get("family")
        group = "diagnostic" if family == FAMILY_DIAGNOSTIC else "patch"
        bucket = histogram.setdefault(group, {}).setdefault(
            task.category, {"items": 0, "positions": {}, "candidates": {}})
        if "candidates" in spec:
            candidates = spec["candidates"]
            position = next(index for index, candidate in enumerate(candidates)
                            if candidate.get("reference"))
            count = len(candidates)
        else:
            options = spec["options"]
            position = next(index for index, option in enumerate(options)
                            if option["consequence"]["kind"] == "reference")
            count = len(options)
        bucket["items"] += 1
        bucket["positions"][str(position)] = bucket["positions"].get(str(position), 0) + 1
        bucket["candidates"][str(count)] = bucket["candidates"].get(str(count), 0) + 1
    for categories in histogram.values():
        for bucket in categories.values():
            counts = list(bucket["positions"].values())
            bucket["max_share"] = round(max(counts) / max(1, bucket["items"]), 6)
            bucket["distinct_positions"] = len(bucket["positions"])
    return histogram


def seal(out_root: str, *, rebalance: bool = False,
         verification: str | None = None) -> dict:
    """Recompute the population receipt from the tree, optionally re-balancing.

    `verify-formats --prune` rewrites an item's candidate list when a distractor
    turns out to pass, which reassigns labels and can move the reference. The
    receipt is therefore recomputed from the tree after verification, and the
    positions are re-balanced on request so the registered histogram describes
    the population that is actually run.
    """
    from . import suite as suite_module

    path = os.path.join(out_root, "POPULATION.json")
    if not os.path.isfile(path):
        raise PopulationError(f"{path} is missing; derive the population first")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tasks = suite_module.load_collection(os.path.join(out_root, "tasks"))
    if rebalance:
        by_family: dict = {FAMILY_DIAGNOSTIC: [], "patch": []}
        for task in tasks:
            family = (task.metadata.get("derived_from") or {}).get("family")
            group = FAMILY_DIAGNOSTIC if family == FAMILY_DIAGNOSTIC else "patch"
            by_family[group].append(task.id)
        _balance(out_root, by_family)
        tasks = suite_module.load_collection(os.path.join(out_root, "tasks"))
    items = [_describe(task) for task in tasks]
    too_long: list = []
    scored_chars = _scored_size(out_root, too_long)
    if too_long:
        raise PopulationError(
            "the tree carries items above the scored-size bound: "
            f"{[entry['id'] for entry in too_long][:3]}")
    manifest = suite_module.build_manifest(out_root)
    suite_module.write_manifest(out_root, manifest)
    payload["items"] = sorted(items, key=lambda item: item["id"])
    payload["counts"] = _counts(payload["items"])
    payload["positions"] = _positions(out_root)
    payload["scored_size"] = scored_chars
    payload["manifest_sha256"] = manifest["suite_sha256"]
    payload["sealed_utc"] = derive._utc_now()
    payload["rebalanced"] = bool(rebalance)
    if verification:
        with open(verification, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        payload["verification"] = {
            "path": verification,
            "sha256": sha256_file(verification),
            "schema": receipt.get("schema"),
            "verdict": receipt.get("verdict"),
            "tasks": receipt.get("tasks"),
            "passed": receipt.get("passed"),
            "dropped": receipt.get("dropped"),
            "pruned_candidates": sum(
                len((item.get("candidates") or {}).get("pruned") or [])
                for item in receipt.get("items", []))
                                  + sum(
                len((item.get("diagnostic") or {}).get("pruned") or [])
                for item in receipt.get("items", [])),
            "distractor_status": _distractor_status(receipt),
        }
        payload["verified"] = receipt.get("verdict") == "pass"
    _write_json(path, payload)
    return payload


def _distractor_status(receipt: dict) -> dict:
    """How each verified distractor failed: a compile error or a failing test.

    Both are admissible (the prompt says "compile-or-not and fail at least one
    test"), and the split matters: an item whose distractors *all* fail to compile
    tests spelling rather than reasoning, so the share is reported per format.
    """
    counts: dict = {}
    for item in receipt.get("items", []):
        if not item.get("passed"):
            continue
        expected = set((item.get("candidates") or {}).get("expected") or [])
        expected |= set((item.get("diagnostic") or {}).get("expected") or [])
        statuses = dict((item.get("candidates") or {}).get("statuses") or {})
        statuses.update((item.get("diagnostic") or {}).get("statuses") or {})
        for label, status in statuses.items():
            if label in expected:
                continue
            key = str(status)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# checks


def _scored_size(out_root: str, too_long: list,
                 max_scored_chars: int = DEFAULT_MAX_SCORED_CHARS) -> dict:
    """The largest rendered selection prompt plus its largest candidate."""
    from . import scoring, suite as suite_module

    maximum = {"chars": 0, "id": None, "prompt_chars": 0, "candidate_chars": 0}
    collections = suite_module.load_suite(out_root)
    for task in collections["suite"]:
        if formats.task_format(task) not in (formats.FORMAT_PATCH,
                                             formats.FORMAT_DIAGNOSTIC):
            continue
        prompt = scoring.selection_prompt(task)
        contents = scoring.selection_contents(task)
        largest = max((len(row["content"]) for row in contents), default=0)
        total = len(prompt) + largest
        if total > maximum["chars"]:
            maximum = {"chars": total, "id": task.id,
                       "prompt_chars": len(prompt),
                       "candidate_chars": largest}
        if total > max_scored_chars:
            too_long.append({"id": task.id, "scored_chars": total})
    return maximum


def _counts(items: list) -> dict:
    by_category: dict = {}
    by_family: dict = {}
    by_difficulty: dict = {}
    by_category_difficulty: dict = {}
    by_candidates: dict = {}
    by_split: dict = {}
    for item in items:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
        by_family[item["family"]] = by_family.get(item["family"], 0) + 1
        by_difficulty[item["difficulty"]] = by_difficulty.get(item["difficulty"], 0) + 1
        bucket = by_category_difficulty.setdefault(item["category"], {})
        bucket[item["difficulty"]] = bucket.get(item["difficulty"], 0) + 1
        by_candidates[str(item["candidates"])] = by_candidates.get(
            str(item["candidates"]), 0) + 1
        by_split[item["source_split"]] = by_split.get(item["source_split"], 0) + 1
    executable = [item for item in items
                  if item["format"] == formats.FORMAT_PATCH]
    per_category_executable = {}
    for item in executable:
        per_category_executable[item["category"]] = (
            per_category_executable.get(item["category"], 0) + 1)
    return {
        "items": len(items),
        "executable_items": len(executable),
        "diagnostic_items": len(items) - len(executable),
        "by_category": dict(sorted(by_category.items())),
        "per_category_executable": dict(sorted(per_category_executable.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_category_difficulty": {key: dict(sorted(value.items()))
                                   for key, value in sorted(
                                       by_category_difficulty.items())},
        "by_candidates": dict(sorted(by_candidates.items())),
        "by_source_split": dict(sorted(by_split.items())),
    }


# ---------------------------------------------------------------------------
# checks


def check(out_root: str) -> tuple[bool, str, dict]:
    """The tree still matches POPULATION.json and the manifest."""
    path = os.path.join(out_root, "POPULATION.json")
    if not os.path.isfile(path):
        return False, "POPULATION.json is missing", {}
    with open(path, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    ok, message = suite_module.check_manifest(out_root)
    if not ok:
        return False, message, recorded
    manifest = suite_module.build_manifest(out_root)
    if recorded.get("manifest_sha256") != manifest["suite_sha256"]:
        return False, "the population tree differs from its receipt", recorded
    present = {task.id: task for task in suite_module.load_collection(
        os.path.join(out_root, "tasks"))}
    recorded_ids = {item["id"] for item in recorded.get("items", [])}
    if recorded_ids != set(present):
        missing = sorted(recorded_ids - set(present))
        extra = sorted(set(present) - recorded_ids)
        return False, (f"the receipt and the tree disagree: absent {missing[:5]}, "
                       f"unrecorded {extra[:5]}"), recorded
    for item in recorded["items"]:
        if present[item["id"]].tree_sha256 != item["tree_sha256"]:
            return False, f"{item['id']}: the tree does not match its recorded digest", recorded
    if recorded.get("counts") != _counts([_describe(task) for task in present.values()]):
        return False, "the receipt's counts are stale", recorded
    return True, "the population tree matches its receipt", recorded


# ---------------------------------------------------------------------------
# prompt near-duplicate screen


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\sA-Za-z0-9_]", text)


def _shingles(tokens: list, width: int) -> set:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + width])
            for index in range(len(tokens) - width + 1)}


def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def screen(out_root: str, *, threshold: float = PROMPT_NEAR_DUPLICATE) -> dict:
    """Pairwise prompt similarity inside the population.

    The source-level screen (`scripts/capability_similarity.py`) compares source
    tasks on crate code, diagnostics and assertions. Inside a population built
    from the same sources, items of one source necessarily share its crate; that
    is clustering, and it is measured by the analysis, not screened away. What
    this screen rejects is a *duplicate item*: two items whose rendered prompts
    are the same question, which cannot carry independent information.
    """
    from . import scoring, suite as suite_module

    collections = suite_module.load_suite(out_root)
    tasks = [task for task in collections["suite"]
             if formats.task_format(task) in (formats.FORMAT_PATCH,
                                              formats.FORMAT_DIAGNOSTIC)]
    shingles = {}
    for task in tasks:
        prompt = scoring.selection_prompt(task)
        contents = "\n".join(row["content"] for row in
                             scoring.selection_contents(task))
        shingles[task.id] = _shingles(_words(prompt + "\n" + contents), PROMPT_WIDTH)
    rejected = []
    reported = []
    maximum = 0.0
    maximum_pair = None
    counts = {"pairs": 0}
    ids = sorted(shingles)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            counts["pairs"] += 1
            score = _jaccard(shingles[left], shingles[right])
            if score > maximum:
                maximum = score
                maximum_pair = [left, right]
            if score >= threshold:
                rejected.append({"left": left, "right": right,
                                 "similarity": round(score, 6)})
            elif score >= threshold - 0.05:
                reported.append({"left": left, "right": right,
                                 "similarity": round(score, 6)})
    return {
        "schema": "selvedge/capability-population-screen/v1",
        "items": len(tasks),
        "pairs": counts["pairs"],
        "width": PROMPT_WIDTH,
        "metric": "set Jaccard over word shingles of the rendered selection prompt "
                  "and its candidate contents",
        "threshold": threshold,
        "maximum_similarity": round(maximum, 6),
        "maximum_pair": maximum_pair,
        "rejected": rejected,
        "near_threshold": reported,
        "verdict": "pass" if not rejected else "reject",
    }


README = """# The S31 selection-item population (generated) — do not edit by hand

This tree is **derived** from `fixtures/capability/tasks` by
`oracle/capability/population.py`. Every item names its source task, that
source's tree hash and the S28 split it belongs to in `task.json`
(`derived_from`), and `POPULATION.json` binds the whole tree to the source suite
hash and records every item's own tree digest.

Four families, all of them selection items:

- `--fn--<file>--<function>` — one function is removed from a solved crate; the
  arm chooses among 4-6 candidate definitions.
- `--file--<file>` — one file is reverted to its initial state; the arm chooses
  among 4-6 candidate whole files.
- `--manifest` — the manifest is the question (dependency-compat), the code is
  solved.
- `--diag` — diagnostic-identification, unchanged from S28, reported on its own
  metric and never pooled with executable success.

Every candidate is downloaded into a crate and compiled by
`verify-formats`; exactly one must pass and every other must fail. The
verification receipt is the evidence, not this file.

The sealed holdback (`fixtures/capability/sealed`) is never read here. The
calibration/scored split from S28 is retained as an attribute of each item so a
reader can still separate the two; it is **not** a holdout for this population.

Verification, with the real checkers (CPU):

```bash
SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \\
scripts/court rust -- python3 scripts/capability_court.py verify-formats \\
  --root fixtures/capability-population --prune \\
  --out results/capability/derivation-verify-population.json
```
"""
