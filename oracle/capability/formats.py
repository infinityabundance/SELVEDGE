#!/usr/bin/env python3
"""Task formats: prompt, answer parsing and executable judging (S28).

The v4 suite has one format. The stored T2 receipts showed what that costs: a
0.6B arm asked for *complete replacement files of multi-file crates* produced
almost no compiling answer, so the instrument measured harness mismatch rather
than capability. This module adds three more formats and keeps every one of them
executable:

* ``file-replacement``     — the v4 format, unchanged; the hard tier.
* ``body-completion``      — one function body, signature and a visible failing
                             test shown; the harness splices the body into the
                             template and runs the crate's tests.
* ``patch-selection``      — pick one of a small candidate set; the harness
                             applies the chosen body and runs the tests. Exactly
                             one candidate is verified to pass at authoring time.
* ``diagnostic-identification`` — name the compiler error / constraint that
                             explains the failure. This answer is *not*
                             executable, so the harness applies the registered
                             **consequence** of the named diagnostic and compiles
                             and runs it. The outcome is reported as
                             ``verified_diagnostic`` and is never pooled with
                             executable task success.

Two registered rules hold across formats:

1. the harness never guesses. Anything it cannot parse is a recorded failure
   with a taxonomy class, never a repaired answer;
2. every claim about behaviour is decided by `cargo`, not by comparing text to a
   stored answer. The patch format's *correct* candidate is the one that passes,
   not the label an author wrote.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

FORMAT_FILE = "file-replacement"
FORMAT_BODY = "body-completion"
FORMAT_PATCH = "patch-selection"
FORMAT_DIAGNOSTIC = "diagnostic-identification"
FORMATS = (FORMAT_FILE, FORMAT_BODY, FORMAT_PATCH, FORMAT_DIAGNOSTIC)
FORMAT_TIERS = {
    FORMAT_BODY: "easy",
    FORMAT_PATCH: "medium",
    FORMAT_FILE: "hard",
    FORMAT_DIAGNOSTIC: "diagnostic",
}

#: Executable formats contribute to the primary composite.
EXECUTABLE_FORMATS = (FORMAT_FILE, FORMAT_BODY, FORMAT_PATCH)
#: The non-executable format, reported on its own metric.
DIAGNOSTIC_FORMATS = (FORMAT_DIAGNOSTIC,)

METRIC_EXECUTABLE = "verified_task_success"
METRIC_DIAGNOSTIC = "verified_diagnostic"

#: The marker a body-completion template carries where the arm's body goes.
BODY_MARKER = "todo!()"
#: The marker a patch-selection template carries where the chosen function goes.
FUNCTION_MARKER = "// TODO(FUNCTION)"

#: Sentinel for `render_prompt`'s example letter. The default must stay distinct
#: from every real argument (including None = "show no example") so the
#: registered v1 prompts are reproduced byte for byte.
EXAMPLE_DEFAULT = object()

#: Taxonomy classes a *parsed* answer can be screened into before any checker
#: runs. Checker outcomes are added by `diagnose`, never here.
SCREEN_EMPTY = "empty_answer"
SCREEN_REFUSAL = "refusal"
SCREEN_NO_CODE = "no_code_block"
SCREEN_TRUNCATED = "truncated"
SCREEN_DEGENERATE = "degenerate_repetition"
SCREEN_WRONG_FILE = "wrong_file"
SCREEN_UNPARSEABLE_CHOICE = "unparseable_choice"
SCREEN_STUB = "stub_returned"
SCREEN_INVALID = "invalid_answer"


class FormatError(ValueError):
    """A task's format declaration or an answer cannot be judged as declared."""


# ---------------------------------------------------------------------------
# format declaration helpers


def task_format(task) -> str:
    value = task.metadata.get("format", FORMAT_FILE)
    if value not in FORMATS:
        raise FormatError(f"{task.id}: unknown format {value!r}")
    return value


def format_spec(task) -> dict:
    spec = task.metadata.get("format_spec") or {}
    if not isinstance(spec, dict):
        raise FormatError(f"{task.id}: format_spec must be an object")
    return spec


def metric_for_source(fmt: str) -> str:
    """The metric a format is scored on. Only the diagnostic is non-executable."""
    return METRIC_DIAGNOSTIC if fmt in DIAGNOSTIC_FORMATS else METRIC_EXECUTABLE


def metric_for(task) -> str:
    return metric_for_source(task_format(task))


def tier_for(task) -> str:
    return FORMAT_TIERS[task_format(task)]


def extraction_policy(task) -> str:
    policy = task.metadata.get("extraction_policy", "strict")
    if policy not in ("strict", "tolerant"):
        raise FormatError(f"{task.id}: extraction_policy must be strict or tolerant")
    return policy


def validate_declaration(task) -> list[str]:
    """Problems with the task's format declaration; empty means valid."""
    fmt = task_format(task)
    spec = format_spec(task)
    problems: list[str] = []
    if fmt == FORMAT_FILE:
        return problems
    template = spec.get("template")
    if fmt in (FORMAT_BODY, FORMAT_PATCH, FORMAT_FILE):
        if not template:
            problems.append(f"{task.id}: format_spec.template is required")
        elif template not in task.initial_files:
            problems.append(f"{task.id}: template {template!r} is not in initial/")
    if fmt in (FORMAT_BODY, FORMAT_PATCH):
        function = spec.get("function")
        if fmt == FORMAT_BODY and not (function and isinstance(function, str)):
            problems.append(f"{task.id}: format_spec.function is required")
        elif fmt == FORMAT_BODY and not problems:
            try:
                body = function_body(task.read_initial(template), function)
            except FormatError as exc:
                problems.append(f"{task.id}: {exc}")
            else:
                if not body.strip():
                    problems.append(f"{task.id}: function {function!r} has an empty body")
        if template and not problems:
            marker = spec.get("marker") or (
                FUNCTION_MARKER if fmt == FORMAT_PATCH else BODY_MARKER)
            if spec.get("target", "function") != "file" and \
                    task.read_initial(template).count(marker) != 1:
                problems.append(
                    f"{task.id}: the template must contain exactly one {marker!r} marker")
    if fmt == FORMAT_PATCH:
        target_kind = spec.get("target", "function")
        if target_kind not in ("function", "file"):
            problems.append(f"{task.id}: patch target must be function or file")
        candidates = spec.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 3:
            problems.append(f"{task.id}: patch-selection needs at least 3 candidates")
        else:
            labels = [c.get("label") for c in candidates]
            key = "function" if target_kind == "function" else "content"
            definitions = [c.get(key) for c in candidates]
            if not all(isinstance(label, str) and label for label in labels):
                problems.append(f"{task.id}: every candidate needs a label")
            if len(set(labels)) != len(labels):
                problems.append(f"{task.id}: candidate labels are not unique")
            if not all(isinstance(item, str) and item.strip() for item in definitions):
                problems.append(f"{task.id}: every candidate needs a {key}")
            if len({item.strip() for item in definitions}) != len(definitions):
                problems.append(f"{task.id}: duplicate candidate {key}s")
            expected = [chr(ord("A") + index) for index in range(len(candidates))]
            if labels != expected:
                problems.append(
                    f"{task.id}: candidate labels must be {expected} in order")
            function = spec.get("function")
            if target_kind == "function" and not (function and isinstance(function, str)):
                problems.append(f"{task.id}: format_spec.function is required")
            if target_kind == "function":
                for candidate in candidates:
                    definition = candidate.get("function") or ""
                    if function and isinstance(definition, str) and definition.strip():
                        try:
                            function_body(definition, function)
                        except FormatError as exc:
                            problems.append(
                                f"{task.id}: candidate {candidate.get('label')!r} does "
                                f"not define {function!r}: {exc}")
    if fmt == FORMAT_DIAGNOSTIC:
        options = spec.get("options")
        if not isinstance(options, list) or len(options) < 2:
            problems.append(f"{task.id}: diagnostic-identification needs 2+ options")
        else:
            identifiers = [option.get("id") for option in options]
            if not all(isinstance(value, str) and value for value in identifiers):
                problems.append(f"{task.id}: every diagnostic option needs an id")
            if len(set(identifiers)) != len(identifiers):
                problems.append(f"{task.id}: diagnostic option ids are not unique")
            for option in options:
                consequence = option.get("consequence")
                if not isinstance(consequence, dict) or consequence.get("kind") not in (
                        "reference", "mutation", "unchanged", "body"):
                    problems.append(
                        f"{task.id}: option {option.get('id')!r} has no registered "
                        "consequence")
    return problems


# ---------------------------------------------------------------------------
# a small Rust scanner: function bodies without a parser dependency


def _skip_string(text: str, index: int) -> int:
    """Return the index just past a string/char literal starting at `index`."""
    if text.startswith('r#', index) or text.startswith('r"', index):
        match = re.match(r'r(#*)".*?"\1', text[index:], re.DOTALL)
        if match:
            return index + match.end()
        return len(text)
    if text[index] == '"':
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == '"':
                return index + 1
            index += 1
        return index
    # char literal or lifetime
    if index + 2 < len(text) and text[index + 1] != "\\" and text[index + 2] == "'":
        return index + 3
    if index + 3 < len(text) and text[index + 1] == "\\" and text[index + 3] == "'":
        return index + 4
    return index + 1


def _code_mask(text: str) -> list[bool]:
    """True where a character is *code* (not a comment or a literal)."""
    mask = [True] * len(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                mask[position] = False
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
            for position in range(index, end):
                mask[position] = False
            index = end
            continue
        character = text[index]
        if character == '"' or character == "'":
            end = _skip_string(text, index)
            for position in range(index, min(end, len(text))):
                mask[position] = False
            index = max(end, index + 1)
            continue
        index += 1
    return mask


FN = re.compile(
    r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:const[ \t]+|async[ \t]+|unsafe[ \t]+|"
    r"extern[ \t]+\"[^\"]*\"[ \t]+)*fn[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def _function_spans(text: str) -> list[dict]:
    """Every function in `text` with the offsets of its body braces."""
    mask = _code_mask(text)
    spans = []
    for match in FN.finditer(text):
        opening = None
        for index in range(match.end(), len(text)):
            if not mask[index]:
                continue
            if text[index] == "{":
                opening = index
                break
            if text[index] == ";":
                break
        if opening is None:
            continue
        depth = 0
        closing = None
        for index in range(opening, len(text)):
            if not mask[index]:
                continue
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            continue
        spans.append({
            "name": match.group("name"),
            "start": match.start(),
            "signature_end": opening,
            "body_start": opening + 1,
            "body_end": closing,
        })
    return spans


def function_body(text: str, name: str | None = None) -> str:
    """The inner body of a function, by name or the first function."""
    spans = _function_spans(text)
    if not spans:
        raise FormatError("no function with a body was found")
    if name is not None:
        matching = [span for span in spans if span["name"] == name]
        if not matching:
            raise FormatError(f"no function named {name!r} was found")
        span = matching[0]
    else:
        span = spans[0]
    return text[span["body_start"]:span["body_end"]]


def function_signature(text: str, name: str) -> str:
    for span in _function_spans(text):
        if span["name"] == name:
            return text[span["start"]:span["signature_end"]].strip()
    raise FormatError(f"no function named {name!r} was found")


def visible_test_block(text: str) -> str:
    """The `#[cfg(test)] mod ...` block in the template, if there is one."""
    match = re.search(r"(?m)^[ \t]*#\[cfg\(test\)\]", text)
    if not match:
        return ""
    spans = _function_spans(text)
    opening = None
    mask = _code_mask(text)
    for index in range(match.start(), len(text)):
        if mask[index] and text[index] == "{":
            opening = index
            break
        if mask[index] and text[index] == ";":
            return ""
    if opening is None:
        return ""
    depth = 0
    for index in range(opening, len(text)):
        if not mask[index]:
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():index + 1].strip()
    return ""


def splice_body(template: str, body: str) -> str:
    """Put `body` where the template's marker is. Exactly one marker."""
    count = template.count(BODY_MARKER)
    if count != 1:
        raise FormatError(
            f"the template must contain exactly one {BODY_MARKER} marker, found {count}")
    return template.replace(BODY_MARKER, body, 1)


def splice_function(template: str, definition: str) -> str:
    """Put a whole function definition where a patch template's marker is."""
    count = template.count(FUNCTION_MARKER)
    if count != 1:
        raise FormatError(
            f"the template must contain exactly one {FUNCTION_MARKER} marker, "
            f"found {count}")
    return template.replace(FUNCTION_MARKER, definition.strip(), 1)


def function_definition(text: str, name: str) -> str:
    """The full definition (signature, body and attributes-free text) of a function."""
    for span in _function_spans(text):
        if span["name"] == name:
            start = text.rfind("\n", 0, span["start"]) + 1
            return text[start:span["body_end"] + 1].strip()
    raise FormatError(f"no function named {name!r} was found")


def normalise_body(body: str) -> str:
    """Whitespace-insensitive form used only to match a choice to a candidate."""
    return "\n".join(line.strip() for line in body.strip().splitlines() if line.strip())


# ---------------------------------------------------------------------------
# prompt rendering


def _body_prompt_intro(task, spec, template: str, signature: str,
                       visible: str) -> list[str]:
    parts = [task.metadata["prompt_intro"].strip(), "",
             "## Function to complete", "",
             f"`{spec['template']}` already contains this signature. Only the body",
             "is missing.", "",
             "```rust", signature + " {", "    " + BODY_MARKER, "}", "```", ""]
    if visible:
        parts.extend(["## Visible failing test", "", "```rust", visible, "```", ""])
    observed = task.metadata.get("observed_failure")
    if observed:
        parts.extend(["## Observed failure", "", "```text", observed.strip(), "```", ""])
    return parts


def render_prompt(task, example_letter=EXAMPLE_DEFAULT) -> str:
    """The exact prompt for a non-legacy format. Deterministic in the fixture.

    `example_letter` controls the concrete letter the answer-format section shows.
    The default reproduces the registered v1 instrument byte for byte (``C`` for
    patch-selection, ``B`` for diagnostic-identification); ``None`` omits the
    example letter, which is the S30 likelihood-selection variant (a model that
    copies the example token cannot be asked with one).
    """
    fmt = task_format(task)
    spec = format_spec(task)
    if fmt == FORMAT_FILE:
        raise FormatError("file-replacement is rendered by suite.render_prompt")
    observed = task.metadata.get("observed_failure")
    if example_letter is EXAMPLE_DEFAULT:
        example_letter = {FORMAT_PATCH: "C", FORMAT_DIAGNOSTIC: "B"}.get(fmt)

    if fmt == FORMAT_BODY:
        template = task.read_initial(spec["template"])
        signature = function_signature(template, spec["function"])
        visible = visible_test_block(template)
        parts = _body_prompt_intro(task, spec, template, signature, visible)
        parts.extend([
            "## Answer format", "",
            "Return only the body of",
            f"`{spec['function']}`: the statements and the final expression, with",
            "no signature, no surrounding braces and no other text. Put it in a",
            "single ```rust block.",
            "",
        ])
        return "\n".join(parts)

    if fmt == FORMAT_PATCH:
        template = task.read_initial(spec["template"])
        visible = visible_test_block(template)
        target_kind = spec.get("target", "function")
        parts = [task.metadata["prompt_intro"].strip(), "",
                 "## Current file" if target_kind == "file"
                 else "## Current file (the target function is missing)", "",
                 f"```text path={spec['template']}", template.rstrip("\n"), "```", ""]
        if visible:
            parts.extend(["## Visible failing test", "", "```rust", visible, "```", ""])
        if observed:
            parts.extend(["## Observed failure", "", "```text", observed.strip(),
                          "```", ""])
        parts.extend(["## Candidate replacements", "",
                      "Exactly one of these replacements makes the crate build and",
                      "its tests pass. Each is shown in full.", ""])
        for candidate in spec["candidates"]:
            body = candidate.get("function") or candidate.get("content") or ""
            parts.extend([f"### Candidate {candidate['label']}", "", "```rust",
                          body.strip(), "```", ""])
        parts.extend([
            "## Answer format", "",
            ("Return the single letter of the candidate you choose, for example\n"
             f"`{example_letter}`. Do not repeat the replacement.")
            if example_letter is not None else
            ("Return the single letter of the candidate you choose. Do not repeat\n"
             "the replacement."),
            "",
        ])
        return "\n".join(parts)

    # diagnostic-identification
    parts = [task.metadata["prompt_intro"].strip(), "", "## Current files", ""]
    for relative in task.initial_files:
        parts.extend([f"```text path={relative}",
                      task.read_initial(relative).rstrip("\n"), "```", ""])
    if observed:
        parts.extend(["## Observed failure", "", "```text", observed.strip(), "```", ""])
    parts.extend(["## Diagnostic options", ""])
    for index, option in enumerate(spec["options"]):
        label = chr(ord("A") + index)
        description = option.get("description") or ""
        parts.append(f"{label}. `{option['id']}` — {description}".rstrip())
    parts.extend([
        "", "## Answer format", "",
        (f"Return the option letter (for example `{example_letter}`). The harness "
         "will apply that"
         if example_letter is not None else
         "Return the option letter. The harness will apply that"),
        "diagnosis's registered repair and compile it, so a label whose repair does",
        "not resolve the failure is not verified.",
        "",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# answer parsing


@dataclass
class ParsedAnswer:
    ok: bool = False
    kind: str = "invalid"
    payload: dict = field(default_factory=dict)
    mode: str = "none"
    problems: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    screen: str = SCREEN_INVALID
    flags: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "mode": self.mode,
            "screen": self.screen,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "flags": dict(self.flags),
        }


REFUSAL = re.compile(
    r"\b(i cannot|i can't|i can not|i'm sorry|i am sorry|as an ai|unable to help|"
    r"cannot help with|i won't|i will not)\b", re.IGNORECASE)
CHOICE = re.compile(
    r"(?i)\b(?:candidate|option|answer|choice)\s*(?:is|:)?\s*[\[(]?\s*([A-E])\b")
LEADING_CHOICE = re.compile(r"^\s*(?:[#>*_-]+\s*)*[\[(]?\s*([A-E])\s*[\])]?(?:[.:\s]|$)")
ERROR_CODE = re.compile(r"\bE\d{4}\b")


def text_flags(text: str, tokens_out=None, max_new_tokens=None) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = max(Counter(lines).values()) if lines else 0
    fences = text.count("```")
    return {
        "characters": len(text),
        "lines": len(lines),
        "fences": fences,
        "unterminated_fence": fences % 2 == 1,
        "repeated_line_max": repeated,
        "degenerate_repetition": repeated >= 8,
        "brace_counts_equal": text.count("{") == text.count("}"),
        "tokens_out": tokens_out,
        "length_capped": bool(tokens_out and max_new_tokens
                              and tokens_out >= max_new_tokens),
    }


def screen_text(text: str, flags: dict) -> str:
    """The taxonomy class of an answer that produced no usable code."""
    if not text or not text.strip():
        return SCREEN_EMPTY
    if flags.get("degenerate_repetition"):
        return SCREEN_DEGENERATE
    if flags.get("unterminated_fence") or flags.get("length_capped"):
        return SCREEN_TRUNCATED
    if REFUSAL.search(text[:400]):
        return SCREEN_REFUSAL
    return SCREEN_NO_CODE


def _fenced_candidates(text: str, entry: str) -> list[tuple[str | None, str]]:
    """(declared path or None, body) for every fence, tolerant about headers."""
    from . import extract as extract_module

    found = []
    for info, body, _terminated, _notes in extract_module.fenced_blocks(
            text, tolerant=True):
        _language, path = extract_module._parse_info(info)
        inner, body, _notes = extract_module._split_inner_path(body)
        found.append((path or inner, body))
    return found


def _extract_candidate(text: str, template: str, entry: str) -> str | None:
    """The most plausible single code payload from an answer."""
    blocks = _fenced_candidates(text, entry)
    if not blocks:
        stripped = text.strip()
        return stripped or None
    named = [body for path, body in blocks if path in (entry, template)]
    pool = named if named else [body for _path, body in blocks]
    pool = [body for body in pool if body.strip()]
    if not pool:
        return None
    from . import extract as extract_module

    best = max(range(len(pool)), key=lambda index: (extract_module._rust_score(pool[index]),
                                                    index))
    return pool[best]


def _strip_braces(body: str) -> str:
    stripped = body.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1]
        if inner.count("{") == inner.count("}"):
            return inner.strip()
    return stripped


def is_stub(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return True
    without_comments = re.sub(r"//[^\n]*", "", stripped).strip()
    return without_comments in ("", BODY_MARKER, "unimplemented!()", "unreachable!()")


def parse_answer(task, text: str, tokens_out=None,
                 max_new_tokens=None) -> ParsedAnswer:
    """Parse one answer under its task's format. Never repairs, only records."""
    fmt = task_format(task)
    spec = format_spec(task)
    policy = extraction_policy(task)
    tolerant = policy == "tolerant"
    text = text or ""
    flags = text_flags(text, tokens_out, max_new_tokens)
    parsed = ParsedAnswer(flags=flags)

    if not text.strip():
        parsed.screen = SCREEN_EMPTY
        parsed.problems.append("the answer is empty")
        return parsed

    if fmt == FORMAT_FILE:
        from . import extract as extract_module

        extraction = extract_module.extract_files(
            text, task.entry, task.allowed_paths(), task.manifest_policy,
            tolerant=tolerant)
        parsed.notes.extend(extraction.notes)
        if not extraction.ok:
            parsed.problems.extend(extraction.rejected)
            parsed.screen = SCREEN_WRONG_FILE if extraction.rejected else SCREEN_NO_CODE
            if any("no code block" in problem for problem in extraction.rejected):
                parsed.screen = screen_text(text, flags)
            return parsed
        parsed.ok = True
        parsed.kind = "files"
        parsed.payload = {"files": extraction.files}
        parsed.mode = extraction.mode
        return parsed

    if fmt == FORMAT_DIAGNOSTIC:
        return _parse_choice(task, text, flags, spec, identifier=True)

    if fmt == FORMAT_PATCH:
        return _parse_choice(task, text, flags, spec, identifier=False)

    # body-completion
    candidate = _extract_candidate(text, spec["template"], task.entry)
    from . import extract as extract_module

    if candidate is None or ("```" not in text
                             and not extract_module.looks_like_rust(candidate)):
        parsed.screen = screen_text(text, flags)
        parsed.problems.append("no code block and no recognisable body")
        return parsed
    parsed.mode = "fenced_body" if "```" in text else "full_text"
    body = candidate
    spans = _function_spans(candidate)
    target = spec["function"]
    if spans:
        named = [span for span in spans if span["name"] == target]
        if named:
            body = candidate[named[0]["body_start"]:named[0]["body_end"]]
        elif len(spans) == 1:
            body = candidate[spans[0]["body_start"]:spans[0]["body_end"]]
            parsed.notes.append(
                f"the answer defined {spans[0]['name']!r}, not {target!r}; used its body")
        else:
            parsed.problems.append(
                f"the answer defines {len(spans)} functions and none is {target!r}")
            parsed.screen = SCREEN_INVALID
            return parsed
    body = _strip_braces(body)
    if is_stub(body):
        parsed.screen = SCREEN_STUB
        parsed.problems.append("the answer is a stub, not a body")
        return parsed
    if not flags["brace_counts_equal"] and flags.get("length_capped"):
        # Still judge it: an unbalanced body will fail to compile, and the
        # receipt records both facts. It is not silently dropped.
        parsed.notes.append("the answer's braces are unbalanced and it hit the token cap")
    parsed.ok = True
    parsed.kind = "body"
    parsed.payload = {"body": body}
    return parsed


def _parse_choice(task, text: str, flags: dict, spec: dict,
                  identifier: bool) -> ParsedAnswer:
    parsed = ParsedAnswer(flags=flags)
    options = spec.get("candidates") if not identifier else spec.get("options")
    by_letter = {chr(ord("A") + index): option for index, option in enumerate(options)}
    letter = None
    match = LEADING_CHOICE.match(text)
    if match:
        letter = match.group(1).upper()
    else:
        match = CHOICE.search(text)
        if match:
            letter = match.group(1).upper()
    chosen_id = None
    if identifier:
        code = ERROR_CODE.search(text)
        if code:
            chosen_id = code.group(0)
        if chosen_id and chosen_id not in {option["id"] for option in options}:
            chosen_id = None
    if letter is None and chosen_id is not None:
        for candidate_letter, option in by_letter.items():
            if option["id"] == chosen_id:
                letter = candidate_letter
                break
    if letter is None and not identifier:
        # A definition that matches one candidate exactly is a valid answer too.
        candidate = _extract_candidate(text, task.metadata.get("entry", ""),
                                       task.entry)
        if candidate:
            wanted = normalise_body(_strip_braces(candidate))
            for candidate_letter, option in by_letter.items():
                definition = (option.get("function") or option.get("content")
                              or option.get("body") or "")
                if normalise_body(definition) == wanted:
                    letter = candidate_letter
                    parsed.notes.append(
                        f"matched the answer definition to candidate {candidate_letter}")
                    break
    if letter is None or letter not in by_letter:
        parsed.screen = screen_text(text, flags) if not flags.get(
            "degenerate_repetition") else SCREEN_DEGENERATE
        if parsed.screen == SCREEN_NO_CODE:
            parsed.screen = SCREEN_UNPARSEABLE_CHOICE
        parsed.problems.append("no candidate letter could be read from the answer")
        return parsed
    option = by_letter[letter]
    if identifier and chosen_id and chosen_id != option["id"]:
        parsed.notes.append(
            f"the answer named {chosen_id} but chose option {letter} ({option['id']}); "
            "the explicit choice is used")
    parsed.ok = True
    parsed.kind = "choice"
    parsed.payload = {"label": letter,
                      "index": ord(letter) - ord("A"),
                      "id": option.get("id"),
                      "option": option}
    parsed.mode = "choice_letter"
    return parsed


# ---------------------------------------------------------------------------
# judging: every behaviour claim goes through cargo


def _diagnostic_files(task, option: dict, workroot_note=None) -> tuple[dict, dict]:
    """The files produced by applying a diagnostic option's consequence."""
    from . import checkers

    consequence = option.get("consequence") or {}
    kind = consequence.get("kind")
    if kind == "reference":
        return task.reference_solution(), {"kind": kind,
                                           "source": "reference solution"}
    if kind == "unchanged":
        return checkers.initial_solution(task), {"kind": kind,
                                                 "source": "unrepaired initial state"}
    if kind == "mutation":
        from . import checkers

        try:
            files, spec = checkers.mutate_reference(task)
        except checkers.CheckerError:
            # No mutation can be derived for this fixture. A wrong diagnostic's
            # consequence is then the unrepaired state, which cannot resolve the
            # failure; the verifier still proves exactly one option resolves it.
            return (checkers.initial_solution(task),
                    {"kind": "unchanged", "requested": "mutation",
                     "reason": "no mutation could be derived for this fixture"})
        return files, {"kind": kind, "mutation": spec}
    if kind == "body":
        template = task.read_initial(consequence["template"])
        body = consequence["body"]
        return ({consequence["template"]: splice_body(template, body)},
                {"kind": kind, "template": consequence["template"]})
    raise FormatError(f"{task.id}: unknown consequence kind {kind!r}")


def judge(task, parsed: ParsedAnswer, runner) -> dict:
    """Apply the parsed answer and run the checker. Returns one outcome.

    `solution` carries the files that were compiled. The court keeps it for the
    partial-credit run and drops it from the receipt (it is the fixture's code,
    not a new measurement).
    """
    fmt = task_format(task)
    spec = format_spec(task)
    if not parsed.ok:
        return {"status": "invalid_solution", "problems": parsed.problems,
                "format": fmt, "metric": metric_for(task)}
    if fmt == FORMAT_FILE:
        files = dict(parsed.payload["files"])
        outcome = runner(task, files)
        return dict(outcome, files=sorted(files), solution=files, format=fmt,
                    metric=metric_for(task))
    if fmt == FORMAT_BODY:
        template = task.read_initial(spec["template"])
        files = {spec["template"]: splice_body(template, parsed.payload["body"])}
        outcome = runner(task, files)
        return dict(outcome, files=sorted(files), solution=files, format=fmt,
                    metric=metric_for(task))
    if fmt == FORMAT_PATCH:
        option = spec["candidates"][parsed.payload["index"]]
        template = task.read_initial(spec["template"])
        if spec.get("target", "function") == "file":
            files = {spec["template"]: option["content"]}
        else:
            files = {spec["template"]: splice_function(template, option["function"])}
        outcome = runner(task, files)
        return dict(outcome, files=sorted(files), solution=files, format=fmt,
                    metric=metric_for(task),
                    choice={"label": option["label"],
                            "reference_matches": option.get("reference", False)})
    # diagnostic
    option = parsed.payload["option"]
    files, consequence = _diagnostic_files(task, option)
    outcome = runner(task, files)
    verified = outcome.get("status") == "pass"
    return dict(outcome, files=sorted(files), solution=files, format=fmt,
                metric=metric_for(task),
                diagnostic={"id": option["id"], "label": parsed.payload["label"],
                            "verified": verified,
                            "consequence": consequence})
