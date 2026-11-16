#!/usr/bin/env python3
"""Derive the calibrated task formats from the registered suite (S28).

The v4 suite asks one question: emit complete replacement files for a crate.
Every stored T2 answer failed that question, and §"diagnosis" in
`docs/CAPABILITY-COURT.md` records why — the format, not only the model, was out
of reach. This module derives three easier formats **from the same source
tasks**, so a calibrated instrument measures the capability the source task was
authored for rather than the arm's ability to retype a crate:

* ``body-completion``  — from the source reference, the target function's body is
  replaced by ``todo!()``; the signature and a visible failing test are shown.
* ``patch-selection``  — the same template plus 3–5 candidate bodies: the
  reference body, the source's *initial* (failing) body, the source's declared
  mutation, and the declared naive fix when the source has one. Exactly one must
  pass; that is verified with `cargo`, not asserted.
* ``file-replacement`` — the source task itself, copied under a new id. This is
  the hardest tier and the only multi-file format.
* ``diagnostic-identification`` — only for sources whose initial state is a
  compile error with an `E####` code. The arm names the diagnostic; the harness
  applies that option's registered consequence and compiles it. Reported as
  ``verified_diagnostic``, never pooled with executable success.

Splits. Sources are partitioned by a deterministic hash of
``(seed, split, task id)``: the calibration split is disjoint from the scored
split by construction, and the sealed holdback lives in a separate collection
that is never read here. A derivation records the source tree hash for every
item, so a fixture change is a failed check rather than a silent drift.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess

from . import formats
from . import suite as suite_module

SCHEMA = "selvedge/capability-derivation/v1"
SELECTION_SEED = "selvedge-calibration-v1"
SPLIT_CALIBRATION = "calibration"
SPLIT_SCORED = "scored"
SPLITS = (SPLIT_CALIBRATION, SPLIT_SCORED)
TAG = {
    formats.FORMAT_BODY: "body",
    formats.FORMAT_PATCH: "patch",
    formats.FORMAT_FILE: "full",
    formats.FORMAT_DIAGNOSTIC: "diag",
}

#: Descriptions for the diagnostic options. They are the compiler's own short
#: descriptions; a wrong option is wrong because its registered repair does not
#: resolve the failure, not because an author labelled it.
DIAGNOSTIC_DESCRIPTIONS = {
    "E0106": "missing lifetime specifier",
    "E0277": "the trait bound is not satisfied",
    "E0308": "mismatched types",
    "E0382": "use of moved value",
    "E0499": "cannot borrow as mutable more than once at a time",
    "E0502": "cannot borrow as immutable while it is also borrowed as mutable",
    "E0505": "cannot move out of a borrowed value",
    "E0507": "cannot move out of a borrowed context",
    "E0515": "cannot return a value referencing local data",
    "E0596": "cannot borrow as mutable because it is not declared mutable",
    "E0597": "borrowed value does not live long enough",
    "E0716": "temporary value dropped while borrowed",
}
DIAGNOSTIC_POOL = tuple(sorted(DIAGNOSTIC_DESCRIPTIONS))
DISTRACTORS = 3

#: Operator inversions used only when a source offers fewer than three candidate
#: bodies. Every candidate is verified: one must pass and the rest must fail, or
#: the item is dropped.
OPERATOR_SWAPS = (
    (">=", ">"),
    ("<=", "<"),
    ("==", "!="),
    ("!=", "=="),
    (" + ", " - "),
    (" - ", " + "),
    (" && ", " || "),
    (" || ", " && "),
)


class DerivationError(RuntimeError):
    """The calibrated split cannot be derived as registered."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stamp_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _revision() -> str:
    value = os.environ.get("SELVEDGE_REVISION", "")
    if value:
        return value
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# split selection


def partition_sources(suite_root: str, calibration_per_category: int) -> dict:
    """Deterministic, disjoint calibration / scored source selection.

    Ranking is by `sha256(seed:split:id)`, so the split is reproducible from the
    suite alone and cannot be tuned after seeing a result.
    """
    collections = suite_module.load_suite(suite_root)
    by_category: dict[str, list] = {}
    for task in collections["suite"]:
        by_category.setdefault(task.category, []).append(task)
    calibration: dict[str, list] = {}
    scored: dict[str, list] = {}
    for category in sorted(by_category):
        tasks = by_category[category]
        ranked_calibration = sorted(
            tasks, key=lambda task: _sha(f"{SELECTION_SEED}:{SPLIT_CALIBRATION}:{task.id}"))
        chosen = ranked_calibration[:calibration_per_category]
        chosen_ids = {task.id for task in chosen}
        calibration[category] = chosen
        remaining = [task for task in tasks if task.id not in chosen_ids]
        scored[category] = sorted(
            remaining, key=lambda task: _sha(f"{SELECTION_SEED}:{SPLIT_SCORED}:{task.id}"))
    calibration_ids = {task.id for tasks in calibration.values() for task in tasks}
    scored_ids = {task.id for tasks in scored.values() for task in tasks}
    if calibration_ids & scored_ids:
        raise DerivationError("the calibration and scored splits overlap")
    sealed_root = os.path.join(suite_root, "sealed")
    if os.path.isdir(sealed_root):
        sealed_ids = {task.id for task in suite_module.load_collection(sealed_root)}
        overlap = (calibration_ids | scored_ids) & sealed_ids
        if overlap:
            raise DerivationError(f"a split item is also sealed: {sorted(overlap)}")
    return {"calibration": calibration, "scored": scored}


# ---------------------------------------------------------------------------
# visible test helpers


def _test_function_text(text: str, index_into: int) -> str | None:
    """The `#[test] fn ...` block that starts at or after `index_into`."""
    marker = text.find("#[test]", index_into)
    if marker < 0:
        return None
    spans = formats._function_spans(text[marker:])
    if not spans:
        return None
    span = spans[0]
    line_start = text.rfind("\n", 0, marker) + 1
    end = marker + span["body_end"] + 1
    return text[line_start:end].strip()


def _crate_name(source) -> str:
    match = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"',
                      source.read_initial("Cargo.toml"))
    return match.group(1).replace("-", "_") if match else ""


def _hidden_visible_test(source) -> tuple[str, str] | None:
    """A visible test taken from the hidden set, with its in-file helpers.

    The source's own inline test is *not* used: in the semantic-repair sources
    that test encodes the old, wrong behaviour and passes in the failing state,
    so carrying it into the derived reference would make the reference fail. A
    hidden test only uses the public API and already passes against the
    reference, which is the property the visible test must have.
    """
    text = source.read_hidden("tests/check.rs")
    spans = formats._function_spans(text)
    if not spans:
        return None
    tests = []
    for match in re.finditer(r"(?m)^\s*#\[test\]", text):
        found = formats._function_spans(text[match.start():])
        if found:
            tests.append(found[0]["name"])
    if not tests:
        return None
    chosen = next(span for span in spans if span["name"] == tests[0])
    helpers = []
    for span in spans:
        if span["name"] in tests:
            continue
        start = text.rfind("\n", 0, span["start"]) + 1
        helpers.append(text[start:span["body_end"] + 1].strip())
    test_text = _test_function_text(text, 0)
    if not test_text:
        return None
    imports = [line for line in text.splitlines() if line.strip().startswith("use ")]
    crate = _crate_name(source)
    imports = [line for line in imports
               if not re.match(rf"^\s*use\s+{re.escape(crate)}::", line)]
    body = "\n\n".join(imports + helpers + [test_text])
    wrapped = ("#[cfg(test)]\nmod visible {\n    use super::*;\n\n"
               + "\n".join("    " + line for line in body.splitlines())
               + "\n}\n")
    return wrapped, "the first hidden test and its helpers, shown as the visible test"


def _visible_test(source, entry: str) -> tuple[str, str] | None:
    """(visible test text, provenance) for a derived body template."""
    derived = _hidden_visible_test(source)
    if derived is not None:
        return derived
    block = ""
    if entry in source.initial_files:
        block = formats.visible_test_block(source.read_initial(entry))
    if block:
        return block, "the source's inline #[cfg(test)] module"
    hidden = source.read_hidden("tests/check.rs")
    test = _test_function_text(hidden, 0)
    if not test:
        return None
    test = "\n".join(line for line in test.splitlines()
                     if not line.strip().startswith("use "))
    wrapped = ("#[cfg(test)]\nmod visible {\n    use super::*;\n\n"
               + "\n".join("    " + line for line in test.splitlines())
               + "\n}\n")
    return wrapped, "the first hidden test, shown as the visible test"


def _target_function(reference: str) -> str:
    spans = formats._function_spans(reference)
    if not spans:
        raise formats.FormatError("the reference defines no function with a body")
    for span in spans:
        signature = reference[span["start"]:span["signature_end"]]
        if "pub" in signature:
            return span["name"]
    return spans[0]["name"]


def _differing_function(source, entry: str, reference: str) -> str | None:
    """The function the source actually changed, if there is exactly one.

    The first `pub fn` is often a helper that is identical in the failing state
    and the reference; completing *that* function would be a trivial item. The
    target is the function whose definition differs, which is the capability the
    source was authored for.
    """
    if entry not in source.initial_files:
        return None
    initial = source.read_initial(entry)
    counts: dict[str, int] = {}
    for span in formats._function_spans(reference):
        counts[span["name"]] = counts.get(span["name"], 0) + 1
    differing = []
    for span in formats._function_spans(reference):
        name = span["name"]
        if counts[name] > 1:
            # An inherent method and a trait method can share a name; replacing
            # one leaves the other resolving, so the item would not discriminate.
            continue
        try:
            before = formats.function_definition(initial, name)
            after = formats.function_definition(reference, name)
        except formats.FormatError:
            continue
        if formats.normalise_body(before) == formats.normalise_body(after):
            continue
        if _code_only(before) == _code_only(after):
            # A comment-only difference is not a code capability.
            continue
        signature = reference[span["start"]:span["signature_end"]].lstrip()
        mask = formats._code_mask(reference)
        depth = 0
        for index in range(span["start"]):
            if mask[index] and reference[index] == "{":
                depth += 1
            elif mask[index] and reference[index] == "}":
                depth -= 1
        free = depth == 0 and (signature.startswith("pub fn")
                               or signature.startswith("fn "))
        differing.append((0 if free else 1, name))
    differing.sort()
    return differing[0][1] if differing else None


def _target_function_for(source, entry: str, reference: str) -> str:
    return _differing_function(source, entry, reference) or _target_function(reference)


def _target_file(source) -> str | None:
    """The file a body/patch item completes.

    Usually the source entry. For the multi-file sources the entry is unchanged
    and the answer lives in a *new* module, so the target is the first new src
    file that defines a function. The derived task then declares that path as its
    entry, and the rest of the crate is handed over solved.
    """
    if source.entry in source.reference_files:
        return source.entry
    candidates = [path for path in source.reference_files
                  if path.startswith("src/") and path.endswith(".rs")
                  and formats._function_spans(source.read_reference(path))]
    return candidates[0] if candidates else None


def _body_template(source, entry: str, reference: str) -> tuple[str, str, str, str]:
    """(template, function name, reference body, visible provenance)."""
    target = _target_function_for(source, entry, reference)
    reference_body = formats.function_body(reference, target)
    spans = formats._function_spans(reference)
    span = next(span for span in spans if span["name"] == target)
    template = (reference[:span["body_start"]] + "\n    " + formats.BODY_MARKER + "\n"
                + reference[span["body_end"]:])
    visible = _visible_test(source, entry)
    if visible is None:
        raise formats.FormatError("the source has no test to show as the visible test")
    block, provenance = visible
    template = template.rstrip() + "\n\n" + block.strip() + "\n"
    return template, target, reference_body, provenance


# ---------------------------------------------------------------------------
# derivation


def _missing_symbol_error(reference: str, name: str) -> str:
    """The rustc code that a missing definition produces in its context.

    A missing free function is E0425; a missing inherent method is E0599.
    """
    mask = formats._code_mask(reference)
    for span in formats._function_spans(reference):
        if span["name"] != name:
            continue
        depth = 0
        for index in range(span["start"]):
            if mask[index] and reference[index] == "{":
                depth += 1
            elif mask[index] and reference[index] == "}":
                depth -= 1
        return "E0425" if depth == 0 else "E0599"
    return "E0425"


def _declared_mutation(reference: str, initial: str, entry: str) -> dict | None:
    """A declared mutation derived from the source's own change.

    The self-test's generic numeric mutation does not apply to every derived
    item; a declared one does, and it is the source's own wrong answer reversed.
    """
    ref_lines = reference.splitlines(keepends=True)
    init_lines = initial.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, ref_lines, init_lines)
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag != "replace":
            continue
        find = "".join(ref_lines[a_start:a_end])
        replace = "".join(init_lines[b_start:b_end])
        if find.strip() and find in reference and len(find) > 0:
            return {"file": entry, "find": find, "replace": replace}
        return None
    return None


def _mutation_for(source, entry: str, reference: str, initial: str) -> dict | None:
    override = source.metadata.get("mutation")
    if (isinstance(override, dict) and override.get("file") == entry
            and override.get("find") and override["find"] in reference):
        return override
    return _declared_mutation(reference, initial, entry)


def _metadata(source, fmt: str, spec: dict, intro: str, failure: dict,
              observed: str | None, entry: str | None = None,
              extra: dict | None = None) -> dict:
    payload = {
        "schema": suite_module.SCHEMA,
        "id": f"{source.id}--{TAG[fmt]}",
        "kind": "suite",
        "category": source.category,
        "capability": f"{source.metadata['capability']} (format: {fmt})",
        "license": suite_module.LICENSE,
        "provenance": (
            f"derived from fixtures/capability/tasks/{source.id} "
            f"(tree {source.tree_sha256}) by oracle/capability/derive.py; the "
            "hidden tests, reference code and licence are the source's, the "
            "format is new (S28)"),
        "entry": entry or source.entry,
        "checker": {"kind": source.checker_kind},
        "difficulty": formats.FORMAT_TIERS[fmt] if fmt != formats.FORMAT_DIAGNOSTIC
        else source.metadata.get("difficulty", "medium"),
        "format": fmt,
        "format_spec": spec,
        "manifest_policy": source.manifest_policy,
        "extraction_policy": "tolerant",
        "initial_failure": failure,
        "prompt_intro": intro,
        "derived_from": {
            "source_id": source.id,
            "source_tree_sha256": source.tree_sha256,
            "source_difficulty": source.metadata.get("difficulty"),
            "format": fmt,
        },
    }
    payload.update({key: value for key, value in (extra or {}).items()
                    if value is not None})
    return payload


def _solved_files(source, entry: str) -> dict:
    """Every initial file except the target, with the reference content where the
    source has one. The derived task's failing state is "one body missing, the
    rest of the crate solved", so a body answer cannot be blamed on other files
    the arm was never asked about.
    """
    files = {}
    for path in source.initial_files:
        if path == entry:
            continue
        if path in source.reference_files:
            # Including Cargo.toml: a body/patch item isolates the code question,
            # so the manifest must be the solved one or the crate cannot build for
            # a reason the arm was never asked about.
            files[path] = source.read_reference(path)
        else:
            files[path] = source.read_initial(path)
    for path in source.reference_files:
        if path == entry or path in files:
            continue
        files[path] = source.read_reference(path)
    return files


def _copy_inputs(source, root: str, *, initial: dict, reference: dict,
                 hidden: dict) -> None:
    for relative, content in initial.items():
        _write(os.path.join(root, "initial", relative), content)
    for relative, content in reference.items():
        _write(os.path.join(root, "reference", relative), content)
    for relative, content in hidden.items():
        _write(os.path.join(root, "hidden", relative), content)


def _derive_body(source, root: str) -> dict | None:
    entry = _target_file(source)
    if entry is None:
        return {"skipped": f"{source.id}: no reference source file to complete"}
    reference = source.read_reference(entry)
    try:
        template, target, reference_body, visible_provenance = _body_template(
            source, entry, reference)
    except formats.FormatError as exc:
        return {"skipped": f"{source.id}: {exc}"}
    spec = {
        "kind": formats.FORMAT_BODY,
        "template": entry,
        "function": target,
        "visible_test": visible_provenance,
    }
    completed = formats.splice_body(template, reference_body)
    intro = (f"Only the body of `{target}` is missing. Complete it so that the "
             "visible test and the hidden tests pass. Keep the signature exactly as "
             "it is; return the body only.")
    metadata = _metadata(
        source, formats.FORMAT_BODY, spec, intro,
        {"kind": "test_failure", "must_contain": ["not yet implemented"]},
        "the visible test panics at `todo!()`:\nthread panicked: not yet implemented",
        entry=entry,
        extra={"mutation": _mutation_for(source, entry, reference,
                                         source.read_initial(entry))}
        if entry in source.initial_files else {})
    _copy_inputs(source, root,
                 initial={entry: template, **_solved_files(source, entry)},
                 reference={entry: completed},
                 hidden={path: source.read_hidden(path) for path in source.hidden_files})
    _write_json(os.path.join(root, "task.json"), metadata)
    return {"metadata": metadata, "reference_body": reference_body}


def _patch_template(source, entry: str, reference: str) -> tuple[str, str, str]:
    """(template, target name, visible provenance)."""
    target = _target_function_for(source, entry, reference)
    spans = formats._function_spans(reference)
    span = next(span for span in spans if span["name"] == target)
    line_start = reference.rfind("\n", 0, span["start"]) + 1
    template = (reference[:line_start] + formats.FUNCTION_MARKER
                + reference[span["body_end"] + 1:])
    visible = _visible_test(source, entry)
    if visible is None:
        raise formats.FormatError("the source has no test to show as the visible test")
    block, provenance = visible
    template = template.rstrip() + "\n\n" + block.strip() + "\n"
    return template, target, provenance


def _revert_regions(reference: str, initial: str, limit: int = 3) -> list[str]:
    """One variant per differing region, each reverted to the initial text."""
    ref_lines = reference.splitlines(keepends=True)
    init_lines = initial.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, ref_lines, init_lines)
    variants = []
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        reverted = ref_lines[:a_start] + init_lines[b_start:b_end] + ref_lines[a_end:]
        candidate = "".join(reverted)
        if candidate.strip() != reference.strip():
            variants.append(candidate)
        if len(variants) >= limit:
            break
    return variants


def _first_difference_revert(reference: str, initial: str) -> str | None:
    variants = _revert_regions(reference, initial, limit=1)
    return variants[0] if variants else None


def _manifest_variants(reference: str, initial: str) -> list[tuple[str, str]]:
    """Plausible wrong manifests beside the reference, all verified later."""
    variants: list[tuple[str, str]] = [
        (initial, "the source's initial manifest (wrong)")]
    if "default-features = false" in reference:
        variants.append((reference.replace("default-features = false",
                                           "default-features = true", 1),
                         "default features re-enabled (wrong)"))
    for line in reference.splitlines():
        if line.strip().startswith("features = [") and line in reference:
            variants.append((reference.replace(line + "\n", "", 1),
                             "the dependency features removed (wrong)"))
            inner = line[line.index("[") + 1:line.rindex("]")].strip()
            variants.append((reference.replace(line, line[:line.index("[")]
                                               + "[]" + line[line.rindex("]") + 1:], 1),
                             f"every dependency feature dropped (wrong; was [{inner}])"))
            break
    if "version.workspace = true" in initial and "version = \"" in reference:
        line = next(line for line in reference.splitlines()
                    if line.strip().startswith("version = \""))
        variants.append((reference.replace(line, "version = { workspace = true }", 1),
                         "the table form of workspace inheritance (wrong here)"))
    if "edition = \"2021\"" in reference:
        variants.append((reference.replace("edition = \"2021\"", "edition = \"2018\"", 1),
                         "the edition lowered instead of the dependency fixed (wrong)"))
    if "[features]" in reference:
        variants.append((reference.replace("[features]", "[features-unused]", 1),
                         "the feature table renamed (wrong)"))
    reverted = _first_difference_revert(reference, initial)
    if reverted:
        variants.append((reverted, "the first manifest edit reverted (wrong)"))
    for candidate in _revert_regions(reference, initial)[1:]:
        variants.append((candidate, "another manifest edit reverted (wrong)"))
    in_dependencies = False
    for line in reference.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_dependencies = stripped.startswith("[dependencies]")
            continue
        if in_dependencies and stripped and "=" in stripped \
                and not stripped.startswith("#"):
            variants.append((reference.replace(line + "\n", "", 1),
                             "a dependency removed (wrong)"))
            break
    return variants


def _derive_manifest_patch(source, root: str) -> dict | None:
    if "Cargo.toml" not in source.reference_files:
        return None
    initial_manifest = source.read_initial("Cargo.toml")
    reference_manifest = source.read_reference("Cargo.toml")
    if formats.normalise_body(initial_manifest) == formats.normalise_body(
            reference_manifest):
        return None
    candidates: list[tuple[str, str]] = [(reference_manifest, "reference")]
    candidates.extend(_manifest_variants(reference_manifest, initial_manifest))
    override = source.metadata.get("mutation")
    if isinstance(override, dict) and override.get("file") == "Cargo.toml":
        candidates.append((reference_manifest.replace(
            override["find"], override["replace"], 1),
            "the source's declared manifest mutation (wrong)"))
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for content, provenance in candidates:
        key = formats.normalise_body(content)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((content, provenance))
    if len(unique) < 3:
        return None
    ordered = sorted(unique, key=lambda item: _sha(formats.normalise_body(item[0])))
    spec_candidates = [{"label": chr(ord("A") + index),
                        "content": content.strip() + "\n",
                        "reference": provenance == "reference",
                        "provenance": provenance}
                       for index, (content, provenance) in enumerate(ordered)]
    spec = {"kind": formats.FORMAT_PATCH, "target": "file",
            "template": "Cargo.toml", "candidates": spec_candidates}
    intro = ("Choose the one replacement manifest that makes the crate build and its "
             "tests pass. The source code is already correct; only the manifest is "
             "missing the capability the code needs.")
    metadata = _metadata(
        source, formats.FORMAT_PATCH, spec, intro,
        dict(source.metadata["initial_failure"]), source.metadata.get("observed_failure"),
        extra={"mutation": _declared_mutation(reference_manifest, initial_manifest,
                                              "Cargo.toml")}
        or {})
    solved = _solved_files(source, "Cargo.toml")
    reference_files = {path: source.read_reference(path)
                       for path in source.reference_files if path != "Cargo.toml"}
    _copy_inputs(source, root,
                 initial={"Cargo.toml": initial_manifest, **solved},
                 reference={"Cargo.toml": reference_manifest, **reference_files},
                 hidden={path: source.read_hidden(path) for path in source.hidden_files})
    _write_json(os.path.join(root, "task.json"), metadata)
    return {"metadata": metadata, "target": "manifest"}


def _code_only(text: str) -> str:
    """Text with comments and blank lines removed, for a difference test."""
    without_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    without_line = re.sub(r"//[^\n]*", "", without_block)
    return "\n".join(line.strip() for line in without_line.splitlines()
                     if line.strip())


def _derive_file_patch(source, root: str, entry: str) -> dict | None:
    """Whole-file patch selection, for a source whose fix is not in a function."""
    initial = source.read_initial(entry)
    reference = source.read_reference(entry)
    if _code_only(initial) == _code_only(reference):
        return None
    candidates: list[tuple[str, str]] = [(reference, "reference")]
    candidates.append((initial, "the source's initial file (wrong)"))
    override = source.metadata.get("mutation")
    if isinstance(override, dict) and override.get("file") == entry:
        candidates.append((reference.replace(override["find"], override["replace"], 1),
                           "the source's declared mutation (wrong)"))
    reverted = _first_difference_revert(reference, initial)
    if reverted:
        candidates.append((reverted, "the source's first edit reverted (wrong)"))
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for content, provenance in candidates:
        key = formats.normalise_body(content)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((content, provenance))
    if len(unique) < 3:
        return None
    ordered = sorted(unique, key=lambda item: _sha(formats.normalise_body(item[0])))
    spec_candidates = [{"label": chr(ord("A") + index),
                        "content": content.strip() + "\n",
                        "reference": provenance == "reference",
                        "provenance": provenance}
                       for index, (content, provenance) in enumerate(ordered)]
    spec = {"kind": formats.FORMAT_PATCH, "target": "file",
            "template": entry, "candidates": spec_candidates}
    intro = ("Choose the one replacement file that makes the crate build and its tests "
             "pass. Exactly one candidate does.")
    metadata = _metadata(
        source, formats.FORMAT_PATCH, spec, intro,
        dict(source.metadata["initial_failure"]), source.metadata.get("observed_failure"),
        entry=entry,
        extra={"mutation": _declared_mutation(reference, initial, entry)})
    _copy_inputs(source, root,
                 initial={entry: initial, **_solved_files(source, entry)},
                 reference={entry: reference},
                 hidden={path: source.read_hidden(path) for path in source.hidden_files})
    _write_json(os.path.join(root, "task.json"), metadata)
    return {"metadata": metadata, "target": "file"}


def _derive_patch(source, root: str) -> dict | None:
    entry = _target_file(source)
    manifest = None
    if "Cargo.toml" in source.reference_files and (
            _code_only(source.read_initial("Cargo.toml"))
            != _code_only(source.read_reference("Cargo.toml"))):
        code_differs = entry is not None and (
            _differing_function(source, entry, source.read_reference(entry)) is not None)
        if not code_differs:
            manifest = _derive_manifest_patch(source, root)
            if manifest is not None:
                return manifest
    if entry is None:
        return {"skipped": f"{source.id}: no reference source file to complete"}
    reference = source.read_reference(entry)
    if _differing_function(source, entry, reference) is None:
        file_patch = _derive_file_patch(source, root, entry)
        if file_patch is not None:
            return file_patch
        return {"skipped": (f"{source.id}: the reference differs from the initial "
                            "state in no single function or file the patch format "
                            "can express")}
    try:
        template, target, visible_provenance = _patch_template(source, entry, reference)
    except formats.FormatError as exc:
        return {"skipped": f"{source.id}: {exc}"}
    definitions: list[tuple[str, str]] = [
        (formats.function_definition(reference, target), "reference")]
    if entry in source.initial_files:
        initial_text = source.read_initial(entry)
        try:
            definitions.append((formats.function_definition(initial_text, target),
                                "source initial (failing) definition"))
        except formats.FormatError:
            pass
    override = source.metadata.get("mutation")
    if isinstance(override, dict) and override.get("file") == entry:
        mutated = reference.replace(override["find"], override["replace"], 1)
        try:
            definitions.append((formats.function_definition(mutated, target),
                                "the source's declared mutation (wrong)"))
        except formats.FormatError:
            pass
    naive = source.metadata.get("naive")
    if isinstance(naive, dict) and naive.get("file") == entry:
        shallow = reference.replace(naive["find"], naive["replace"], 1)
        try:
            definitions.append((formats.function_definition(shallow, target),
                                "the source's declared naive fix (wrong)"))
        except formats.FormatError:
            pass
    if entry in source.initial_files:
        reverted = _first_difference_revert(reference, source.read_initial(entry))
        if reverted:
            try:
                definitions.append((formats.function_definition(reverted, target),
                                    "the source's first edit reverted (wrong)"))
            except formats.FormatError:
                pass
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for definition, provenance in definitions:
        key = formats.normalise_body(definition)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((definition, provenance))
    for find, replace in OPERATOR_SWAPS:
        if len(unique) >= 3:
            break
        for base, provenance in list(unique):
            if find not in base:
                continue
            variant = base.replace(find, replace, 1)
            key = formats.normalise_body(variant)
            if key and key not in seen:
                seen.add(key)
                unique.append((variant,
                               f"{provenance} with {find!r} -> {replace!r} (wrong)"))
                break
    if len(unique) < 3:
        return {"skipped": (f"{source.id}: only {len(unique)} distinct candidate "
                            "definitions; patch-selection needs at least 3")}
    ordered = sorted(unique, key=lambda item: _sha(formats.normalise_body(item[0])))
    spec_candidates = []
    for index, (definition, provenance) in enumerate(ordered):
        spec_candidates.append({
            "label": chr(ord("A") + index),
            "function": definition.strip() + "\n",
            "reference": provenance == "reference",
            "provenance": provenance,
        })
    spec = {
        "kind": formats.FORMAT_PATCH,
        "template": entry,
        "function": target,
        "candidates": spec_candidates,
        "visible_test": visible_provenance,
    }
    intro = (f"Choose the one candidate definition for `{target}` that satisfies the "
             "visible test and the hidden tests. The others compile-or-not and fail "
             "at least one test.")
    reference_candidate = next(c for c in spec_candidates if c["reference"])
    wrong_candidate = next(c for c in spec_candidates if not c["reference"])
    missing_error = _missing_symbol_error(reference, target)
    metadata = _metadata(
        source, formats.FORMAT_PATCH, spec, intro,
        {"kind": "compile_error", "must_contain": [missing_error],
         "command": "all-targets"},
        "the target function is missing; the visible test does not compile",
        entry=entry,
        extra={"mutation": {"file": entry,
                            "find": reference_candidate["function"].strip(),
                            "replace": wrong_candidate["function"].strip()}})
    reference_index = next(i for i, c in enumerate(spec_candidates) if c["reference"])
    completed = formats.splice_function(template,
                                        spec_candidates[reference_index]["function"])
    _copy_inputs(source, root,
                 initial={entry: template, **_solved_files(source, entry)},
                 reference={entry: completed},
                 hidden={path: source.read_hidden(path) for path in source.hidden_files})
    _write_json(os.path.join(root, "task.json"), metadata)
    return {"metadata": metadata,
            "reference_label": spec_candidates[reference_index]["label"],
            "candidates": [c["label"] for c in spec_candidates]}


def _diagnostic_options(source) -> list[dict] | None:
    failure = source.metadata.get("initial_failure") or {}
    if failure.get("kind") != "compile_error":
        return None
    code = next((needle for needle in failure.get("must_contain", [])
                 if formats.ERROR_CODE.fullmatch(needle.strip())), None)
    if code is None:
        return None
    distractors = sorted(
        (candidate for candidate in DIAGNOSTIC_POOL if candidate != code),
        key=lambda candidate: _sha(f"{source.id}:{candidate}"))[:DISTRACTORS]
    consequence_kind = "mutation" if isinstance(source.metadata.get("mutation"), dict) \
        else "unchanged"
    options = [{"id": code,
                "description": DIAGNOSTIC_DESCRIPTIONS.get(code, "the reported error"),
                "consequence": {"kind": "reference"}}]
    for candidate in distractors:
        options.append({
            "id": candidate,
            "description": DIAGNOSTIC_DESCRIPTIONS.get(candidate, "another compiler error"),
            "consequence": {"kind": consequence_kind},
        })
    options.sort(key=lambda option: _sha(f"{source.id}:{option['id']}"))
    return options


def _derive_diagnostic(source, root: str) -> dict | None:
    options = _diagnostic_options(source)
    if options is None:
        return None
    spec = {"kind": formats.FORMAT_DIAGNOSTIC, "options": options}
    intro = ("This crate does not compile. Identify which compiler error explains the "
             "failure. The harness applies the registered repair for the option you "
             "choose and compiles it: a diagnosis whose repair does not resolve the "
             "failure is not verified.")
    metadata = _metadata(
        source, formats.FORMAT_DIAGNOSTIC, spec, intro,
        {"kind": "compile_error",
         "must_contain": list(source.metadata["initial_failure"]["must_contain"])},
        source.metadata.get("observed_failure"),
        extra={"mutation": _mutation_for(
            source, source.entry,
            source.read_reference(source.entry)
            if source.entry in source.reference_files else "",
            source.read_initial(source.entry))})
    _copy_inputs(source, root,
                 initial={path: source.read_initial(path) for path in source.initial_files},
                 reference={path: source.read_reference(path)
                            for path in source.reference_files},
                 hidden={path: source.read_hidden(path) for path in source.hidden_files})
    _write_json(os.path.join(root, "task.json"), metadata)
    return {"metadata": metadata,
            "correct_id": next(option["id"] for option in options
                               if option["consequence"]["kind"] == "reference")}


def _derive_full(source, root: str) -> dict:
    shutil.copytree(source.root, root)
    path = os.path.join(root, "task.json")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    original = source.metadata
    metadata["id"] = f"{source.id}--{TAG[formats.FORMAT_FILE]}"
    metadata["difficulty"] = "hard"
    metadata["format"] = formats.FORMAT_FILE
    metadata["extraction_policy"] = "tolerant"
    metadata["derived_from"] = {
        "source_id": source.id,
        "source_tree_sha256": source.tree_sha256,
        "source_difficulty": original.get("difficulty"),
        "format": formats.FORMAT_FILE,
    }
    metadata["provenance"] = (
        f"derived from fixtures/capability/tasks/{source.id} "
        f"(tree {source.tree_sha256}) by oracle/capability/derive.py: the source "
        "task unchanged, copied under a new id as the hardest tier (S28)")
    _write_json(path, metadata)
    return {"metadata": metadata}


# ---------------------------------------------------------------------------
# collection


README = """# Calibration split (S28) — generated, do not edit by hand

This directory is **derived** from `fixtures/capability/tasks` by
`oracle/capability/derive.py`. Every item names its source task and that source's
tree hash in `task.json` (`derived_from`), and `DERIVATION.json` binds the whole
tree to the source suite hash. A hand edit is a failed `derive-formats --check`.

- `tasks/` — the calibration items, in four formats (see
  `docs/CAPABILITY-COURT.md` §13);
- `trio/`, `probes/` — empty; this collection has no trio or probes;
- `DERIVATION.json` — the derivation receipt: the split rule, the source ids, the
  per-format counts, the items that could not be derived and why, and the
  manifest hash;
- `MANIFEST.json` — the capability-suite manifest, with `format`, `tier`,
  `metric` and `extraction_policy` per item;
- `VERSION.json` — **absent until the calibration pass runs.** It is the
  calibrated suite version: the items that survived the registered acceptance
  window. Its absence is the honest statement that this split is a hypothesis.

Verification, with the real checkers (CPU):

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \\
scripts/court rust -- python3 scripts/capability_court.py verify-formats \\
  --root fixtures/capability-calibrated --prune \\
  --out results/capability/derivation-verify-calibration.json
```

`--prune` removes a patch candidate or diagnostic option that the checker shows
*also* passes, and drops an item left with too few choices; the removals are in
the verification receipt. Re-running `derive-formats --force` recreates the
pre-prune candidates, so the order is derive → verify.

This split is **not** the scored suite, provides **no** capability claim, and is
disjoint from the scored split and from the sealed holdback by construction.
"""


def derive(suite_root: str, out_root: str, split: str = SPLIT_CALIBRATION,
           calibration_per_category: int = 3,
           force: bool = False) -> dict:
    if split not in SPLITS:
        raise DerivationError(f"split must be one of {SPLITS}")
    if os.path.isdir(out_root) and os.listdir(out_root) and not force:
        raise DerivationError(f"{out_root} is not empty; pass --force to rebuild")
    if os.path.isdir(out_root) and force:
        shutil.rmtree(out_root)
    partition = partition_sources(suite_root, calibration_per_category)
    sources = [task for tasks in partition[split].values() for task in tasks]
    os.makedirs(os.path.join(out_root, "trio"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "probes"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "tasks"), exist_ok=True)

    items: list[dict] = []
    skipped: list[str] = []
    for source in sorted(sources, key=lambda task: task.id):
        derived_paths = {
            formats.FORMAT_BODY: os.path.join(out_root, "tasks",
                                              f"{source.id}--{TAG[formats.FORMAT_BODY]}"),
            formats.FORMAT_PATCH: os.path.join(out_root, "tasks",
                                               f"{source.id}--{TAG[formats.FORMAT_PATCH]}"),
            formats.FORMAT_FILE: os.path.join(out_root, "tasks",
                                              f"{source.id}--{TAG[formats.FORMAT_FILE]}"),
            formats.FORMAT_DIAGNOSTIC: os.path.join(
                out_root, "tasks", f"{source.id}--{TAG[formats.FORMAT_DIAGNOSTIC]}"),
        }
        for fmt, builder in ((formats.FORMAT_BODY, _derive_body),
                             (formats.FORMAT_PATCH, _derive_patch),
                             (formats.FORMAT_DIAGNOSTIC, _derive_diagnostic)):
            result = builder(source, derived_paths[fmt])
            if result is None or "skipped" in result:
                if result and "skipped" in result:
                    skipped.append(result["skipped"])
                continue
            items.append({"id": result["metadata"]["id"], "format": fmt,
                          "category": source.category, "source": source.id,
                          "metric": formats.metric_for_source(fmt)})
        _derive_full(source, derived_paths[formats.FORMAT_FILE])
        items.append({"id": f"{source.id}--{TAG[formats.FORMAT_FILE]}",
                      "format": formats.FORMAT_FILE, "category": source.category,
                      "source": source.id,
                      "metric": formats.metric_for_source(formats.FORMAT_FILE)})

    payload = {
        "schema": SCHEMA,
        "stamp": _stamp_now(),
        "created_utc": _utc_now(),
        "revision": _revision(),
        "command": " ".join(os.sys.argv) if os.sys else None,
        "split": split,
        "selection": {
            "rule": "sha256(seed:split:task_id) ascending, first N per category",
            "seed": SELECTION_SEED,
            "calibration_per_category": calibration_per_category,
            "sources": len(sources),
            "source_ids": [task.id for task in sorted(sources, key=lambda t: t.id)],
            "disjoint_from": ("the scored split (the remaining sources) and the sealed "
                              "holdback (a separate collection, never read here)"),
        },
        "source_suite": {
            "root": os.path.abspath(suite_root),
            "suite_sha256": suite_module.build_manifest(suite_root)["suite_sha256"],
        },
        "counts": _counts(items),
        "items": sorted(items, key=lambda item: item["id"]),
        "skipped": sorted(skipped),
        "verified": False,
    }
    _write_json(os.path.join(out_root, "DERIVATION.json"), payload)
    manifest = suite_module.build_manifest(out_root)
    suite_module.write_manifest(out_root, manifest)
    payload["manifest_sha256"] = manifest["suite_sha256"]
    _write_json(os.path.join(out_root, "DERIVATION.json"), payload)
    _write(os.path.join(out_root, "README.md"), README)
    return payload


def _counts(items: list) -> dict:
    by_format: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_metric: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for item in items:
        by_format[item["format"]] = by_format.get(item["format"], 0) + 1
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
        by_metric[item["metric"]] = by_metric.get(item["metric"], 0) + 1
        tier = formats.FORMAT_TIERS[item["format"]]
        by_tier[tier] = by_tier.get(tier, 0) + 1
    return {"items": len(items), "by_format": by_format, "by_category": by_category,
            "by_metric": by_metric, "by_tier": by_tier}


def _rebuild_manifest(out_root: str) -> str:
    """Rebuild MANIFEST.json and refresh the derivation receipt's binding."""
    manifest = suite_module.build_manifest(out_root)
    suite_module.write_manifest(out_root, manifest)
    path = os.path.join(out_root, "DERIVATION.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        present = set(manifest_ids(out_root))
        payload["items"] = [item for item in payload["items"]
                            if item["id"] in present]
        payload["counts"] = _counts(payload["items"])
        payload["manifest_sha256"] = manifest["suite_sha256"]
        _write_json(path, payload)
    return manifest["suite_sha256"]


def manifest_ids(out_root: str) -> list:
    tasks_root = os.path.join(out_root, "tasks")
    return sorted(name for name in os.listdir(tasks_root)
                  if os.path.isdir(os.path.join(tasks_root, name)))


def _prune_task(task, keep_labels: list, keep_ids: list) -> None:
    """Rewrite a task.json keeping only the declared candidates/options."""
    path = os.path.join(task.root, "task.json")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    spec = metadata["format_spec"]
    if "candidates" in spec:
        kept = [candidate for candidate in spec["candidates"]
                if candidate["label"] in keep_labels]
        for index, candidate in enumerate(kept):
            candidate["label"] = chr(ord("A") + index)
        spec["candidates"] = kept
    if "options" in spec:
        spec["options"] = [option for option in spec["options"]
                           if option["id"] in keep_ids]
    _write_json(path, metadata)


def verify(out_root: str, workroot: str, only: str | None = None,
           keep: bool = False, prune: bool = False) -> dict:
    """Compile the derived suite and prove every declared outcome.

    Per task: the failing state fails as declared, the reference passes, and a
    mutated reference fails. Per patch item: exactly one candidate passes, and it
    is the candidate marked `reference`. Per diagnostic item: exactly one
    option's registered consequence makes the checker pass.
    """
    from . import checkers
    from . import sandbox

    collections = suite_module.load_suite(out_root)
    tasks = [task for task in collections["suite"]
             if not only or only in task.id]
    items = []
    failures = []
    dropped: list = []
    changed = False
    for task in tasks:
        fmt = formats.task_format(task)
        record = {"id": task.id, "format": fmt}
        initial = checkers.run_initial(task, workroot, keep=keep)
        record["initial"] = {
            "expected": task.metadata["initial_failure"]["kind"],
            "status": initial["status"],
            "as_expected": initial["occurred_as_expected"],
            "missing_evidence": initial["missing_evidence"],
        }
        if not initial["occurred_as_expected"]:
            failures.append(f"{task.id}: the failing state did not fail as declared "
                            f"({initial['status']}, missing {initial['missing_evidence']})")
        reference = checkers.run_check(task, task.reference_solution(), workroot,
                                       keep=keep)
        record["reference"] = reference["status"]
        if reference["status"] != sandbox.STATUS_PASS:
            failures.append(f"{task.id}: the reference did not pass "
                            f"({reference['status']}: "
                            f"{(reference.get('evidence') or {}).get('stderr_tail', '')[-300:]})")
        try:
            mutated, mutation = checkers.mutate_reference(task)
        except checkers.CheckerError as exc:
            if fmt == formats.FORMAT_DIAGNOSTIC:
                # A diagnostic item's discrimination evidence is its option
                # consequences, proven below; a text mutation adds nothing there.
                record["mutation"] = {"skipped": str(exc)}
            else:
                record["mutation"] = {"error": str(exc)}
                failures.append(f"{task.id}: {exc}")
        else:
            wrong = checkers.run_check(task, mutated, workroot, keep=keep)
            record["mutation"] = {"spec": mutation, "status": wrong["status"],
                                  "rejected": wrong["status"] != sandbox.STATUS_PASS}
            if wrong["status"] == sandbox.STATUS_PASS:
                failures.append(f"{task.id}: the mutated reference passed the checker")

        def runner(item, files):
            return checkers.run_check(item, files, workroot, keep=keep)

        if fmt == formats.FORMAT_PATCH:
            spec = formats.format_spec(task)
            passing = []
            statuses = {}
            for index, candidate in enumerate(spec["candidates"]):
                parsed = formats.ParsedAnswer(
                    ok=True, kind="choice", mode="choice_letter",
                    payload={"label": candidate["label"], "index": index,
                             "id": None, "option": candidate})
                outcome = formats.judge(task, parsed, runner)
                statuses[candidate["label"]] = outcome.get("status")
                if outcome["status"] == sandbox.STATUS_PASS:
                    passing.append(candidate["label"])
            expected = [candidate["label"] for candidate in spec["candidates"]
                        if candidate.get("reference")]
            record["candidates"] = {
                "labels": [candidate["label"] for candidate in spec["candidates"]],
                "passing": passing,
                "expected": expected,
                "statuses": statuses,
                "target": spec.get("target", "function"),
            }
            if passing != expected:
                if prune and all(label in record["candidates"]["labels"]
                                 for label in expected):
                    keep_labels = [label for label in record["candidates"]["labels"]
                                   if label in expected or label not in passing]
                    if len(keep_labels) >= 3:
                        _prune_task(task, keep_labels, [])
                        changed = True
                        record["candidates"]["pruned"] = sorted(
                            set(passing) - set(expected))
                        passing = expected
                    else:
                        record["candidates"]["dropped"] = True
                        shutil.rmtree(task.root)
                        dropped.append(task.id)
                        changed = True
                if passing != expected:
                    failures.append(f"{task.id}: candidates that pass are {passing}, the "
                                    f"declared reference is {expected}")
        if fmt == formats.FORMAT_DIAGNOSTIC:
            spec = formats.format_spec(task)
            passing = []
            statuses = {}
            for index, option in enumerate(spec["options"]):
                parsed = formats.ParsedAnswer(
                    ok=True, kind="choice", mode="choice_letter",
                    payload={"label": chr(ord("A") + index), "index": index,
                             "id": option["id"], "option": option})
                outcome = formats.judge(task, parsed, runner)
                statuses[option["id"]] = outcome.get("status")
                if outcome["status"] == sandbox.STATUS_PASS:
                    passing.append(option["id"])
            expected = [option["id"] for option in spec["options"]
                        if option["consequence"]["kind"] == "reference"]
            record["diagnostic"] = {"passing": passing, "expected": expected,
                                    "statuses": statuses}
            if passing != expected:
                if prune and all(identifier in [option["id"] for option in spec["options"]]
                                 for identifier in expected):
                    keep_ids = [identifier for identifier in
                                [option["id"] for option in spec["options"]]
                                if identifier in expected or identifier not in passing]
                    if len(keep_ids) >= 2:
                        _prune_task(task, [], keep_ids)
                        changed = True
                        record["diagnostic"]["pruned"] = sorted(
                            set(passing) - set(expected))
                        passing = expected
                    else:
                        record["diagnostic"]["dropped"] = True
                        shutil.rmtree(task.root)
                        dropped.append(task.id)
                        changed = True
                if passing != expected:
                    failures.append(f"{task.id}: diagnostic options that resolve the "
                                    f"failure are {passing}, the registered repair is "
                                    f"{expected}")
        record["passed"] = not any(failure.startswith(task.id + ":")
                                   for failure in failures)
        items.append(record)
        print(f"derive.verify: {task.id}: initial={record['initial']['status']} "
              f"reference={record['reference']}", flush=True)
    failures = [failure for failure in failures
                if not any(failure.startswith(identifier + ":") for identifier in dropped)]
    if changed:
        _rebuild_manifest(out_root)
    return {
        "schema": "selvedge/capability-derivation-verification/v1",
        "suite": {"root": os.path.abspath(out_root),
                  "suite_sha256": suite_module.build_manifest(out_root)["suite_sha256"]},
        "tasks": len(items),
        "passed": sum(1 for item in items if item["passed"]),
        "pruned": prune,
        "dropped": sorted(dropped),
        "failures": failures,
        "items": items,
        "verdict": "pass" if items and not failures else "fail",
    }


def check_derivation(out_root: str) -> tuple[bool, str]:
    """The tree still matches DERIVATION.json and the manifest."""
    path = os.path.join(out_root, "DERIVATION.json")
    if not os.path.isfile(path):
        return False, "DERIVATION.json is missing"
    with open(path, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)
    ok, message = suite_module.check_manifest(out_root)
    if not ok:
        return False, message
    manifest = suite_module.build_manifest(out_root)
    if recorded.get("manifest_sha256") != manifest["suite_sha256"]:
        return False, "the derived tree differs from the derivation receipt"
    if recorded["counts"] != _counts(recorded["items"]):
        return False, "the derivation receipt's counts are stale"
    return True, "the derived tree matches its receipt"
