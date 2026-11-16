#!/usr/bin/env python3
"""Prompt forensics and the response-entropy screen (S30).

The S29 calibration pass emitted no suite version because every arm's answer was
the same letter on every item (``C`` for patch-selection, ``B`` for
diagnostic-identification). A constant answer cannot discriminate, so the pass
was uninformative. This module is the diagnostic instrument for that failure:

* it renders a prompt exactly and audits it — are the candidates present, are
  their labels unique, does the instruction suggest a letter, which candidate is
  the reference and at which position;
* it measures the letter distribution the stored generations actually produced
  and separates "the model cannot answer" from "the model repeats the example
  letter in the instruction";
* it screens any format's answers for near-zero entropy, which is the structural
  statement that the format cannot discriminate for that arm.

Nothing here scores an item or changes a registered metric. It produces
evidence, and the entropy screen marks a format non-discriminating so that it is
excluded rather than scored (see `docs/CAPABILITY-COURT.md` §18).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from . import formats

#: The prompt width the generation adapter tokenises to. `greedy_generate`
#: truncates at this many tokens with the tokenizer's default truncation side
#: (right), so a longer prompt loses its *tail* — which is where the candidates
#: and the answer-format instruction live. Measured against this number, not
#: assumed equal to the model's 131072-token context.
ADAPTER_MAX_PROMPT_TOKENS = 4096

#: Registered minimum Shannon entropy (bits) of a selection format's answers
#: across items. A constant answer has entropy 0; a format whose answers carry no
#: item-to-item information cannot discriminate and is excluded, not scored.
MIN_ANSWER_ENTROPY_BITS = 0.10

#: The most frequent answer may not cover more than this share of items. This is
#: the check that catches a near-constant answer whose entropy clears the minimum
#: only because of a handful of stragglers.
MAX_MODAL_ANSWER_SHARE = 0.90

LETTER_LINE = re.compile(r"^([A-E])[.:)]?$")
EXAMPLE_LETTER = re.compile(r"for example\s+`([A-E])`")
LETTER = re.compile(r"^\s*([A-E])\b")
#: A projection used only by the audit and the screen: a stand-alone letter line
#: (optionally in backticks) or a one-letter fence. It is never used to score an
#: item; the registered parser decides scores.
PROJECTED_LETTER_LINE = re.compile(r"^[A-E]$")
PROJECTED_FENCED_LETTER = re.compile(r"```[A-Za-z]*\s*\n\s*([A-E])\s*\n\s*```")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_labels(task) -> list[str]:
    """The labels the prompt shows, in prompt order."""
    fmt = formats.task_format(task)
    spec = formats.format_spec(task)
    if fmt == formats.FORMAT_PATCH:
        return [str(candidate["label"]) for candidate in spec["candidates"]]
    if fmt == formats.FORMAT_DIAGNOSTIC:
        return [chr(ord("A") + index) for index in range(len(spec["options"]))]
    raise ValueError(f"{task.id}: {fmt} has no candidate labels")


def reference_label(task) -> str | None:
    """The label whose content the checker verified as correct, or None."""
    fmt = formats.task_format(task)
    spec = formats.format_spec(task)
    if fmt == formats.FORMAT_PATCH:
        for candidate in spec["candidates"]:
            if candidate.get("reference"):
                return str(candidate["label"])
        return None
    if fmt == formats.FORMAT_DIAGNOSTIC:
        for index, option in enumerate(spec["options"]):
            if (option.get("consequence") or {}).get("kind") == "reference":
                return chr(ord("A") + index)
        return None
    return None


def candidate_contents(task) -> list[dict]:
    """The exact text scored for each candidate, independent of its label.

    For patch-selection the content is the replacement definition (or the whole
    file when the target is a file); for diagnostic-identification it is the
    option's rendered line without its positional label. The label is never part
    of the scored content, so a ranking cannot be driven by the display order.
    """
    fmt = formats.task_format(task)
    spec = formats.format_spec(task)
    rows = []
    if fmt == formats.FORMAT_PATCH:
        for candidate in spec["candidates"]:
            content = candidate.get("function") or candidate.get("content") or ""
            rows.append({"label": str(candidate["label"]),
                         "content": content.strip(),
                         "reference": bool(candidate.get("reference"))})
        return rows
    if fmt == formats.FORMAT_DIAGNOSTIC:
        for index, option in enumerate(spec["options"]):
            description = option.get("description") or ""
            content = f"`{option['id']}` — {description}".rstrip()
            rows.append({"label": chr(ord("A") + index),
                         "content": content,
                         "reference": (option.get("consequence") or {}).get("kind")
                         == "reference"})
        return rows
    raise ValueError(f"{task.id}: {fmt} has no scored candidates")


def prompt_audit(task, prompt: str) -> dict:
    """What the rendered prompt contains, checked rather than assumed."""
    fmt = formats.task_format(task)
    spec = formats.format_spec(task)
    labels = candidate_labels(task)
    contents = candidate_contents(task)
    reference = reference_label(task)
    example = EXAMPLE_LETTER.search(prompt)
    problems: list[str] = []
    per_candidate = []
    for row in contents:
        if fmt == formats.FORMAT_PATCH:
            heading = f"### Candidate {row['label']}"
            count = prompt.count(heading)
        else:
            heading = f"{row['label']}. `"
            count = prompt.count(heading)
        present = row["content"] in prompt
        if count != 1:
            problems.append(f"candidate {row['label']}: heading appears {count} times")
        if not present:
            problems.append(f"candidate {row['label']}: content is not in the prompt")
        per_candidate.append({
            "label": row["label"],
            "heading_occurrences": count,
            "content_present": present,
            "content_sha256": sha256_text(row["content"]),
            "content_characters": len(row["content"]),
            "reference": row["reference"],
        })
    seen = Counter(label for row in per_candidate for label in [row["label"]])
    duplicates = sorted(label for label, count in seen.items() if count > 1)
    if duplicates:
        problems.append(f"duplicate candidate labels: {duplicates}")
    positions = {row["label"]: index for index, row in enumerate(contents)}
    example_letter = example.group(1) if example else None
    ## The answer-format section: everything after the last "## Answer format".
    if "## Answer format" in prompt:
        instruction = prompt.rsplit("## Answer format", 1)[1].strip()
    else:
        instruction = ""
    letters_in_instruction = sorted(set(re.findall(r"\b[A-E]\b", instruction)))
    return {
        "id": task.id,
        "format": fmt,
        "prompt_sha256": sha256_text(prompt),
        "prompt_characters": len(prompt),
        "prompt_lines": prompt.count("\n") + 1,
        "labels": labels,
        "reference": reference,
        "reference_position": positions.get(reference) if reference else None,
        "per_candidate": per_candidate,
        "example_letter": example_letter,
        "letters_in_answer_instruction": letters_in_instruction,
        "answer_instruction": instruction,
        "instruction_bias": {
            "names_an_example_letter": example_letter is not None,
            "example_letter": example_letter,
            "example_equals_reference": (
                example_letter is not None and example_letter == reference),
            "why": ("a concrete letter in the instruction is a token the model can "
                    "copy; if every arm emits it, the answer is the instruction's, "
                    "not the item's"),
        },
        "problems": problems,
        "ok": not problems,
    }


def tokenizer_audit(task, prompt: str, tokenize, limit: int = ADAPTER_MAX_PROMPT_TOKENS,
                    decode=None) -> dict:
    """How the adapter's tokenisation sees the prompt, including truncation.

    `tokenize` maps text to a list of token ids with the adapter's settings
    (`add_special_tokens` default, no padding). `decode` is optional and lets the
    receipt show the exact text the model receives after truncation.
    """
    ids = list(tokenize(prompt))
    truncated = len(ids) > limit
    kept = ids[:limit]
    record = {
        "id": task.id,
        "prompt_tokens": len(ids),
        "adapter_max_prompt_tokens": limit,
        "truncated_by_adapter": truncated,
        "tokens_kept": len(kept),
        "tokens_dropped": max(0, len(ids) - limit),
        "special_token_prefix": ids[:5],
    }
    if decode is not None:
        record["as_the_model_sees_it"] = decode(kept)
    if truncated:
        ## The tail holds the candidates and the instruction, so truncation is a
        ## prompt defect, not a cosmetic one.
        record["tail_dropped"] = True
        record["hidden_by_truncation"] = (
            "the prompt's tail — candidate list and the answer-format instruction — "
            "is cut, so the model is asked a question it never sees")
    return record


def shannon_entropy(counts: Counter, total: int | None = None) -> float:
    """Shannon entropy in bits of a count distribution."""
    total = total if total is not None else sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def projected_letters(text: str) -> list[str]:
    """Stand-alone letters in an answer, tolerant of fences (projection only)."""
    letters = []
    for line in (text or "").splitlines():
        stripped = line.strip().strip("`").strip()
        if PROJECTED_LETTER_LINE.match(stripped):
            letters.append(stripped)
    letters.extend(PROJECTED_FENCED_LETTER.findall(text or ""))
    return letters


def answer_key(task, text: str) -> str:
    """The comparable answer for the entropy screen.

    A selection format's answer is the letter it chooses, read with the same
    projection the S29 analysis used (a stand-alone letter line or a one-letter
    fence). A free-form format's answer is its whitespace-normalised text, so the
    degenerate "same fragment on every item" failure is a single key.
    """
    fmt = formats.task_format(task)
    if fmt in (formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC):
        match = LETTER.match(text or "")
        if match:
            return match.group(1)
        projected = sorted(set(projected_letters(text or "")))
        if len(projected) == 1:
            return projected[0]
        if len(projected) > 1:
            return "ambiguous:" + "".join(projected)
        return "<none>"
    return " ".join((text or "").split())


def format_answer_screen(tasks: list, outputs: dict, fmt: str,
                         min_bits: float = MIN_ANSWER_ENTROPY_BITS,
                         max_modal_share: float = MAX_MODAL_ANSWER_SHARE) -> dict:
    """Is `fmt`'s answer distribution non-degenerate for this arm?

    `outputs` maps item id to the stored answer text. The screen is per arm and
    per format: a near-zero entropy means the arm's answers carry no item-to-item
    information, so the format cannot discriminate for that arm and its items are
    excluded rather than scored.
    """
    keys: Counter = Counter()
    per_item = {}
    for task in tasks:
        if formats.task_format(task) != fmt:
            continue
        key = answer_key(task, outputs.get(task.id, ""))
        keys[key] += 1
        per_item[task.id] = key
    total = sum(keys.values())
    entropy = shannon_entropy(keys, total)
    modal_key, modal_count = (keys.most_common(1)[0] if keys else ("<none>", 0))
    modal_share = (modal_count / total) if total else 0.0
    discriminating = total > 0 and entropy >= min_bits and modal_share <= max_modal_share
    return {
        "format": fmt,
        "items": total,
        "distinct_answers": len(keys),
        "entropy_bits": round(entropy, 6),
        "modal_answer": modal_key,
        "modal_share": round(modal_share, 6),
        "min_entropy_bits": min_bits,
        "max_modal_share": max_modal_share,
        "discriminating": discriminating,
        "screen": "scored" if discriminating else "excluded_non_discriminating",
        "per_item": per_item,
        "counts": dict(sorted(keys.items())),
    }


def answer_collapse(tasks: list, outputs: dict, fmt: str) -> dict:
    """The letter distribution and what it says about the instruction.

    Separates three hypotheses that a zero pass rate cannot distinguish:

    * the model answers the item (letters vary, some are right);
    * the model copies the instruction's example letter (one letter, equal to the
      example, on every item);
    * the model answers nothing (no letter, or a constant that is not the
      example).
    """
    by_letter: Counter = Counter()
    by_item = {}
    correct = 0
    example_matches = 0
    example_letters = set()
    reference_letters: Counter = Counter()
    for task in tasks:
        if formats.task_format(task) != fmt:
            continue
        text = outputs.get(task.id, "") or ""
        reference = reference_label(task)
        if reference:
            reference_letters[reference] += 1
        key = answer_key(task, text)
        by_item[task.id] = {"answer": key, "reference": reference,
                            "correct": key == reference}
        if len(key) == 1 and key in "ABCDE":
            by_letter[key] += 1
        if key == reference:
            correct += 1
    # The example letter is a property of the prompt, so read it back from the
    # rendered prompt the generation used instead of assuming it.
    from . import suite as suite_module
    for task in tasks:
        if formats.task_format(task) != fmt:
            continue
        prompt = suite_module.render_prompt(task)
        match = EXAMPLE_LETTER.search(prompt)
        if match:
            example_letters.add(match.group(1))
    constant = len(by_letter) == 1
    constant_letter = next(iter(by_letter)) if constant else None
    return {
        "format": fmt,
        "items": len(by_item),
        "answers_with_a_letter": sum(by_letter.values()),
        "by_letter": dict(sorted(by_letter.items())),
        "distinct_letters": len(by_letter),
        "constant": constant,
        "constant_letter": constant_letter,
        "correct": correct,
        "reference_letters": dict(sorted(reference_letters.items())),
        "example_letters_in_prompts": sorted(example_letters),
        "constant_equals_example": bool(
            constant_letter and example_letters == {constant_letter}),
        "per_item": by_item,
    }


def collapse_diagnosis(patch: dict, diagnostic: dict) -> dict:
    """The named conclusion the two collapse reports support."""
    evidence = []
    for name, report in (("patch", patch), ("diagnostic", diagnostic)):
        if report["items"] and report["constant"]:
            evidence.append(
                f"{name}: all {report['answers_with_a_letter']} recoverable answers "
                f"are {report['constant_letter']!r} across {report['items']} items")
    copies_example = any(report["constant_equals_example"]
                         for report in (patch, diagnostic))
    if copies_example and evidence:
        verdict = "response_collapse_copies_the_instruction_example"
        statement = ("the constant answer is the letter the answer-format section "
                     "shows as an example; the model repeats the prompt instead of "
                     "answering it")
    elif evidence:
        verdict = "response_collapse_not_explained_by_the_example"
        statement = ("the answers are constant but not equal to the instruction's "
                     "example letter; the cause is not the example token")
    else:
        verdict = "no_collapse_detected"
        statement = "the answers vary across items"
    return {"verdict": verdict, "statement": statement, "evidence": evidence}
