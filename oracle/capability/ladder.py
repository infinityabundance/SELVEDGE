#!/usr/bin/env python3
"""The prompt sanity ladder (S30).

The S29 calibration pass collapsed to one letter per format. A floor can be a
capability fact or an instrument defect, and this ladder separates the two with
direct measurements on the same checkpoint:

1. a trivial three-option MCQ whose answer sits at ``A``, then ``B``, then ``C`` —
   can the arm emit a varying letter at all?
2. the same trivial MCQ with the instruction's example letter set to the correct
   one and to a wrong one — is the emitted letter the example token?
3. a real patch item under the registered prompt (the control), with the example
   letter removed, and with an explicit "one character only" instruction;
4. the same patch item with its candidates permuted three ways, so the correct
   *content* sits at each position in turn;
5. the patch item with two few-shot examples of the answer format.

Every condition is a prompt plus the letter the arm emitted, and the receipt
records the mapping from the emitted letter back to the candidate content. The
ladder scores nothing: it is a diagnostic, and it never changes a registered
metric.
"""

from __future__ import annotations

import copy
import hashlib

from . import formats

#: The ladder's decoding: one letter needs a handful of tokens, and a short cap
#: keeps a degenerate arm from spending 3072 tokens per probe.
LADDER_MAX_NEW_TOKENS = 32

TRIVIAL = {
    "question": "What is 2 + 3?",
    "options": ["4", "5", "6"],
    "correct": "5",
}

FEW_SHOT = [
    ("What is 1 + 1?\n\nA. 1\nB. 2\nC. 3\n\n## Answer format\n\n"
     "Return the single letter of the correct option. Do not repeat it.",
     "B"),
    ("What colour is a cloudless noon sky?\n\nA. blue\nB. red\nC. green\n\n"
     "## Answer format\n\nReturn the single letter of the correct option. "
     "Do not repeat it.",
     "A"),
]


def mcq_prompt(position: int, example_letter: str | None = None) -> tuple[str, str]:
    """A trivial MCQ with the correct option at `position`; (prompt, correct letter)."""
    options = list(TRIVIAL["options"])
    options.insert(position, options.pop(options.index(TRIVIAL["correct"])))
    lines = [TRIVIAL["question"], ""]
    for index, option in enumerate(options):
        lines.append(f"{chr(ord('A') + index)}. {option}")
    lines.extend(["", "## Answer format", ""])
    if example_letter is None:
        lines.append("Return the single letter of the correct option. Do not repeat it.")
    else:
        lines.append(
            "Return the single letter of the correct option, for example "
            f"`{example_letter}`. Do not repeat it.")
    correct = chr(ord("A") + options.index(TRIVIAL["correct"]))
    return "\n".join(lines), correct


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_prompt(task, order: list[int] | None = None,
                 example_letter=formats.EXAMPLE_DEFAULT) -> tuple[str, dict]:
    """A patch item's prompt with an explicit candidate order; (prompt, mapping).

    `order` is a permutation of the fixture's candidate indices. Labels are
    reassigned by *position*, so the label always says where the candidate is
    shown and never which one it is. The mapping records new label -> original
    index and whether that candidate is the reference.
    """
    spec = task.metadata["format_spec"]
    original = spec["candidates"]
    indices = list(order) if order is not None else list(range(len(original)))
    permuted = [dict(original[index], label=chr(ord("A") + position))
                for position, index in enumerate(indices)]
    task_copy = copy.copy(task)
    task_copy.metadata = dict(task.metadata)
    task_copy.metadata["format_spec"] = dict(spec, candidates=permuted)
    prompt = formats.render_prompt(task_copy, example_letter=example_letter)
    mapping = {
        chr(ord("A") + position): {
            "original_index": index,
            "reference": bool(original[index].get("reference")),
            "content_sha256": _sha256(
                (original[index].get("function") or original[index].get("content")
                 or "").strip()),
        }
        for position, index in enumerate(indices)
    }
    return prompt, mapping


def permutations(count: int) -> list[list[int]]:
    """The rotations of `count` positions — a deterministic, label-diverse set."""
    return [list(range(count))[shift:] + list(range(count))[:shift]
            for shift in range(count)]


def patch_conditions(task) -> list[dict]:
    """The ladder's patch conditions for one item: control, variants, permutations."""
    reference = None
    for index, candidate in enumerate(task.metadata["format_spec"]["candidates"]):
        if candidate.get("reference"):
            reference = index
    conditions = []

    prompt, mapping = patch_prompt(task)  # the registered default shows `C`
    conditions.append({
        "name": "control_registered_prompt",
        "kind": "patch",
        "prompt": prompt,
        "mapping": mapping,
        "example_letter": "C",
        "note": "the registered v1 prompt, which shows `C` as the example",
    })
    prompt, mapping = patch_prompt(task, example_letter=None)
    conditions.append({
        "name": "no_example_letter",
        "kind": "patch",
        "prompt": prompt,
        "mapping": mapping,
        "example_letter": None,
        "note": "the same item with the example letter removed",
    })
    base, mapping = patch_prompt(task, example_letter=None)
    explicit = base.replace(
        "Return the single letter of the candidate you choose. Do not repeat\n"
        "the replacement.",
        "Reply with exactly one character: A, B, or C. Nothing else.")
    conditions.append({
        "name": "explicit_single_letter",
        "kind": "patch",
        "prompt": explicit,
        "mapping": mapping,
        "example_letter": None,
        "note": "no example letter plus an explicit one-character instruction",
    })
    for shift, order in enumerate(permutations(len(mapping))):
        prompt, mapping = patch_prompt(task, order=order, example_letter=None)
        conditions.append({
            "name": f"permutation_{shift}",
            "kind": "patch",
            "prompt": prompt,
            "mapping": mapping,
            "example_letter": None,
            "note": ("the same candidates, rotated so the reference content is at "
                     f"position {shift}"),
        })
    few_shot_base, few_shot_mapping = patch_prompt(task, example_letter=None)
    conditions.append({
        "name": "two_few_shot_examples",
        "kind": "patch",
        "prompt": few_shot_base,
        "mapping": few_shot_mapping,
        "example_letter": None,
        "note": "two answer-format examples appended before the item",
        "few_shot": FEW_SHOT,
    })
    return conditions


def choose_letter(task, text: str) -> dict:
    """The registered parser's reading of a ladder answer, plus its projection.

    `formats.parse_answer` is used unchanged so the ladder reports the same
    decision the instrument would make; the projection is recorded beside it so a
    parser that reads nothing is distinguishable from an answer that says nothing.
    """
    from . import forensics

    parsed = formats.parse_answer(task, text, tokens_out=None, max_new_tokens=None)
    payload = {"parsed_ok": parsed.ok, "kind": parsed.kind, "screen": parsed.screen,
               "problems": list(parsed.problems)}
    choice = None
    if parsed.kind == "choice":
        choice = parsed.payload.get("label") or parsed.payload.get("id")
    payload["parsed_choice"] = choice
    projected = sorted(set(forensics.projected_letters(text or "")))
    payload["projected_letters"] = projected
    payload["letter"] = choice or (projected[0] if len(projected) == 1 else None)
    payload["ambiguous"] = len(projected) > 1
    return payload


def judge_condition(condition: dict, generation: dict) -> dict:
    """One ladder row: what was asked, what came back, what it means."""
    text = generation.get("text") or ""
    record = {
        "name": condition["name"],
        "kind": condition["kind"],
        "note": condition["note"],
        "example_letter": condition["example_letter"],
        "finish_reason": generation.get("finish_reason"),
        "tokens_out": generation.get("tokens_out"),
        "text_head": text[:300],
        "text_characters": len(text),
    }
    if condition["kind"] == "mcq":
        record["correct_letter"] = condition["correct"]
        record["letter"] = _first_letter(text)
        record["correct"] = record["letter"] == condition["correct"]
    else:
        mapping = condition["mapping"]
        record["reference_letter"] = next(
            (label for label, entry in mapping.items() if entry["reference"]), None)
        record["reference_letter_from_mapping"] = record["reference_letter"]
        parsed = None
        if condition.get("task") is not None:
            parsed = choose_letter(condition["task"], text)
        record.update({
            "mapping": mapping,
            "letter": (parsed or {}).get("letter") or _first_letter(text),
            "parsed": parsed,
        })
        letter = record["letter"]
        record["chosen_is_reference"] = bool(
            letter and mapping.get(letter, {}).get("reference"))
    if record.get("letter") is not None:
        record["equals_example"] = record["letter"] == condition["example_letter"]
    return record


def _first_letter(text: str) -> str | None:
    for line in (text or "").splitlines():
        stripped = line.strip().strip("`").strip()
        if len(stripped) == 1 and stripped in "ABCDE":
            return stripped
    match = formats.LEADING_CHOICE.match(text or "")
    if match:
        return match.group(1).upper()
    match = formats.CHOICE.search(text or "")
    return match.group(1).upper() if match else None


def ladder_verdict(rows: list[dict]) -> dict:
    """What the ladder's rows jointly establish, stated in falsifiable terms."""

    def base(name: str) -> str:
        return name.split(":")[-1]

    plain = [row for row in rows
             if row["kind"] == "mcq" and base(row["name"]).startswith("mcq_answer_at")]
    examples = {base(row["name"]): row for row in rows
                if row["kind"] == "mcq" and base(row["name"]).startswith("mcq_example")}
    wrong_example = examples.get("mcq_example_wrong")
    parrots = None
    if wrong_example and wrong_example.get("letter"):
        ## The discriminating condition: the example letter is A and the correct
        ## option is C. An arm that emits A copied the instruction; an arm that
        ## emits C answered the question.
        parrots = (wrong_example["letter"] == wrong_example.get("example_letter")
                   and wrong_example["letter"] != wrong_example.get("correct_letter"))

    per_item: dict = {}
    for row in rows:
        if row["kind"] != "patch":
            continue
        item_id, condition = row["name"].split(":", 1)
        per_item.setdefault(item_id, {})[condition] = row
    items = {}
    for item_id, conditions in sorted(per_item.items()):
        permutations_rows = [conditions[name] for name in sorted(conditions)
                             if name.startswith("permutation_")]
        letters = [row.get("letter") for row in permutations_rows]
        contents = [row["mapping"].get(row["letter"], {}).get("original_index")
                    for row in permutations_rows]
        control = conditions.get("control_registered_prompt") or {}
        no_example = conditions.get("no_example_letter") or {}
        explicit = conditions.get("explicit_single_letter") or {}
        items[item_id] = {
            "control_letter": control.get("letter"),
            "control_equals_example": control.get("equals_example"),
            "control_is_reference": control.get("chosen_is_reference"),
            "no_example_letter": no_example.get("letter"),
            "no_example_is_reference": no_example.get("chosen_is_reference"),
            "explicit_letter": explicit.get("letter"),
            "explicit_is_reference": explicit.get("chosen_is_reference"),
            "permutation_letters": letters,
            "permutation_chosen_original_indices": contents,
            "permutation_reference_letters": [row.get("reference_letter")
                                              for row in permutations_rows],
            "permutation_letter_stable": (
                None not in letters and len(set(letters)) == 1),
            "permutation_content_stable": (
                None not in contents and len(set(contents)) == 1),
        }

    plain_correct = sum(1 for row in plain if row.get("correct"))
    return {
        "mcq_plain_letters": [row.get("letter") for row in plain],
        "mcq_plain_correct": plain_correct,
        "mcq_plain_items": len(plain),
        "mcq_example_conditions": {
            name: {"letter": row.get("letter"),
                   "example": row.get("example_letter"),
                   "correct_letter": row.get("correct_letter"),
                   "correct": row.get("correct")}
            for name, row in examples.items()},
        "parrots_the_example": parrots,
        "patch": items,
        "interpretation": _interpretation(parrots, items, plain_correct, len(plain)),
    }


def _interpretation(parrots: bool | None, items: dict, plain_correct: int,
                    plain_items: int) -> str:
    copied = [item_id for item_id, entry in items.items()
              if entry["control_equals_example"]]
    no_content = [item_id for item_id, entry in items.items()
                  if not entry["permutation_content_stable"]]
    if parrots and copied and len(no_content) == len(items):
        return ("the arm emits the instruction's example letter when one is shown "
                "and a fixed position prior when none is; its choice does not "
                "follow the candidate content under permutation. The collapse is "
                "a prompt-conditioning defect, not evidence about the tasks")
    if parrots:
        return ("the arm copies the instruction's example letter in the MCQ control; "
                "see the per-item rows for whether its choice follows the content")
    if plain_items and plain_correct == plain_items:
        return ("the arm answers the trivial MCQ correctly at every position and does "
                "not copy the example letter; the collapse has a different cause")
    return ("no single mechanism: the ladder's conditions disagree, and each row is "
            "recorded rather than summarised away")
