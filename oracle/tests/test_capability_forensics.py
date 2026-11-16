"""CPU tests for the S30 prompt forensics and the sanity ladder.

No GPU and no model weights: the prompt audit, the truncation audit (with a
scripted tokenizer), the answer-collapse report, the entropy screen and the
ladder's prompt construction are pure functions. The model runs that use them are
the GPU receipts under results/capability/.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
REPO = os.path.dirname(ORACLE)
sys.path.insert(0, ORACLE)

from capability import forensics, formats, ladder, suite  # noqa: E402

CALIBRATED = os.path.join(REPO, "fixtures", "capability-calibrated")


def _collections():
    return suite.load_suite(CALIBRATED)


def _patch_task():
    for task in _collections()["suite"]:
        if task.metadata.get("format") == formats.FORMAT_PATCH:
            return task
    raise AssertionError("the calibration split has no patch item")


def _diagnostic_task():
    for task in _collections()["suite"]:
        if task.metadata.get("format") == formats.FORMAT_DIAGNOSTIC:
            return task
    raise AssertionError("the calibration split has no diagnostic item")


# ---------------------------------------------------------------------------
# the prompt audit


def test_a_patch_prompt_carries_every_candidate_once_under_a_unique_label():
    task = _patch_task()
    prompt = suite.render_prompt(task)
    audit = forensics.prompt_audit(task, prompt)
    assert audit["ok"], audit["problems"]
    assert audit["labels"] == ["A", "B", "C"]
    for row in audit["per_candidate"]:
        assert row["heading_occurrences"] == 1
        assert row["content_present"]
    assert audit["reference"] in audit["labels"]
    assert audit["reference_position"] == int(ord(audit["reference"]) - ord("A"))


def test_the_registered_prompt_names_an_example_letter_and_the_audit_says_which():
    task = _patch_task()
    audit = forensics.prompt_audit(task, suite.render_prompt(task))
    assert audit["example_letter"] == "C"
    assert audit["instruction_bias"]["names_an_example_letter"]
    assert audit["letters_in_answer_instruction"] == ["C"]
    diagnostic = _diagnostic_task()
    audit = forensics.prompt_audit(diagnostic, suite.render_prompt(diagnostic))
    assert audit["example_letter"] == "B"


def test_a_duplicated_candidate_label_is_a_recorded_problem():
    task = _patch_task()
    prompt = suite.render_prompt(task)
    ## Break the prompt the way a renderer bug would: the same label twice.
    broken = prompt.replace("### Candidate C", "### Candidate B")
    audit = forensics.prompt_audit(task, broken)
    assert not audit["ok"]
    assert any("heading appears" in problem for problem in audit["problems"])


def test_removing_the_example_letter_changes_only_the_instruction():
    task = _patch_task()
    default = suite.render_prompt(task)
    without = formats.render_prompt(task, example_letter=None)
    assert "for example" in default
    assert "for example" not in without
    ## Everything before the answer-format section is untouched.
    assert default.rsplit("## Answer format", 1)[0] == without.rsplit(
        "## Answer format", 1)[0]
    audit = forensics.prompt_audit(task, without)
    assert audit["example_letter"] is None
    assert audit["ok"]


def test_the_default_prompt_is_unchanged_by_the_new_parameter():
    ## The registered v1 instrument renders byte for byte: the parameter's
    ## default is a sentinel, never a value a caller could pass by accident.
    task = _diagnostic_task()
    assert suite.render_prompt(task) == formats.render_prompt(
        task, example_letter="B")
    assert formats.EXAMPLE_DEFAULT is not None
    assert formats.EXAMPLE_DEFAULT is not formats.EXAMPLE_DEFAULT.__class__


# ---------------------------------------------------------------------------
# the tokenizer audit


def test_the_truncation_audit_names_the_dropped_tail():
    task = _patch_task()
    prompt = suite.render_prompt(task)
    words = prompt.split()

    def tokenize(text):
        return list(range(len(text.split())))

    record = forensics.tokenizer_audit(task, prompt, tokenize, limit=10,
                                       decode=lambda ids: " ".join(words[:len(ids)]))
    assert record["truncated_by_adapter"]
    assert record["tokens_dropped"] == len(words) - 10
    assert "tail" in record["hidden_by_truncation"]
    assert record["as_the_model_sees_it"] == " ".join(words[:10])


def test_the_calibration_prompts_fit_the_adapter_limit():
    ## Measured, not assumed: the fixture's prompts are short enough that the
    ## 4096-token truncation keeps the candidates and the instruction.
    for task in _collections()["suite"]:
        record = forensics.tokenizer_audit(
            task, suite.render_prompt(task), lambda text: list(range(len(text))),
            limit=4096)
        assert not record["truncated_by_adapter"]


# ---------------------------------------------------------------------------
# the collapse report and the answer projection


def test_a_fenced_letter_is_projected_and_a_constant_is_named():
    task = _patch_task()
    outputs = {task.id: "```\nC\n```\n```\nC\n```"}
    collapse = forensics.answer_collapse([task], outputs, formats.FORMAT_PATCH)
    assert collapse["by_letter"] == {"C": 1}
    assert collapse["constant"]
    assert collapse["constant_letter"] == "C"


def test_an_ambiguous_answer_is_not_read_as_a_choice():
    task = _patch_task()
    ## No leading letter, and two stand-alone letter lines: the projection cannot
    ## name one candidate, so it is not read as a choice.
    outputs = {task.id: "the answer is A or B\nA\nB\nC"}
    collapse = forensics.answer_collapse([task], outputs, formats.FORMAT_PATCH)
    assert collapse["by_letter"] == {}
    assert not collapse["constant"]


def test_the_collapse_diagnosis_names_the_instruction_example():
    patch = {"items": 3, "answers_with_a_letter": 3, "constant": True,
             "constant_letter": "C", "constant_equals_example": True}
    diagnostic = {"items": 2, "answers_with_a_letter": 2, "constant": True,
                  "constant_letter": "B", "constant_equals_example": True}
    diagnosis = forensics.collapse_diagnosis(patch, diagnostic)
    assert diagnosis["verdict"] == "response_collapse_copies_the_instruction_example"
    assert len(diagnosis["evidence"]) == 2


# ---------------------------------------------------------------------------
# the response-entropy screen


def _outputs(format_name, texts):
    tasks = [task for task in _collections()["suite"]
             if formats.task_format(task) == format_name]
    return tasks, {task.id: text for task, text in zip(tasks, texts)}


def test_a_constant_selection_answer_is_excluded_by_the_entropy_screen():
    tasks = [task for task in _collections()["suite"]
             if formats.task_format(task) == formats.FORMAT_PATCH]
    outputs = {task.id: "C" for task in tasks}
    screen = forensics.format_answer_screen(tasks, outputs, formats.FORMAT_PATCH)
    assert screen["items"] == len(tasks)
    assert screen["entropy_bits"] == 0.0
    assert not screen["discriminating"]
    assert screen["screen"] == "excluded_non_discriminating"


def test_varying_selection_answers_pass_the_entropy_screen():
    tasks = [task for task in _collections()["suite"]
             if formats.task_format(task) == formats.FORMAT_PATCH]
    letters = ["A", "B", "C"]
    outputs = {task.id: letters[index % 3] for index, task in enumerate(tasks)}
    screen = forensics.format_answer_screen(tasks, outputs, formats.FORMAT_PATCH)
    assert screen["discriminating"]
    assert screen["entropy_bits"] > forensics.MIN_ANSWER_ENTROPY_BITS
    assert screen["screen"] == "scored"


def test_a_degenerate_free_form_answer_is_one_key():
    tasks = [task for task in _collections()["suite"]
             if formats.task_format(task) == formats.FORMAT_BODY]
    outputs = {task.id: "```\n\n## Example" for task in tasks}
    screen = forensics.format_answer_screen(tasks, outputs, formats.FORMAT_BODY)
    assert screen["distinct_answers"] == 1
    assert screen["modal_share"] == 1.0
    assert not screen["discriminating"]


def test_the_modal_share_catches_a_near_constant_answer():
    tasks = [task for task in _collections()["suite"]
             if formats.task_format(task) == formats.FORMAT_BODY]
    outputs = {task.id: "the same fragment" for task in tasks}
    outputs[tasks[0].id] = "a different fragment"
    screen = forensics.format_answer_screen(tasks, outputs, formats.FORMAT_BODY)
    assert screen["distinct_answers"] == 2
    assert screen["modal_share"] > forensics.MAX_MODAL_ANSWER_SHARE
    assert not screen["discriminating"]


# ---------------------------------------------------------------------------
# the ladder's prompts


def test_the_trivial_mcq_places_the_answer_at_each_position():
    for position, expected in enumerate("ABC"):
        prompt, correct = ladder.mcq_prompt(position)
        assert correct == expected
        assert f"{expected}. 5" in prompt
        assert "for example" not in prompt


def test_the_mcq_example_letter_is_the_condition_being_tested():
    prompt, correct = ladder.mcq_prompt(2, example_letter="A")
    assert "for example `A`" in prompt
    assert correct == "C"


def test_a_permutation_relabels_by_position_and_keeps_the_reference():
    task = _patch_task()
    spec = task.metadata["format_spec"]
    reference_index = next(index for index, candidate in enumerate(spec["candidates"])
                           if candidate.get("reference"))
    base_prompt, base_mapping = ladder.patch_prompt(task, example_letter=None)
    assert next(label for label, row in base_mapping.items()
                if row["reference"]) == chr(ord("A") + reference_index)
    for shift, order in enumerate(ladder.permutations(len(spec["candidates"]))):
        prompt, mapping = ladder.patch_prompt(task, order=order, example_letter=None)
        assert [mapping[label]["original_index"] for label in sorted(mapping)] == order
        moved = next(label for label, row in mapping.items() if row["reference"])
        assert mapping[moved]["original_index"] == reference_index
        assert f"### Candidate {moved}" in prompt
        assert "for example" not in prompt
        ## The prompt's candidate blocks are the ones the mapping claims.
        content = spec["candidates"][reference_index].get("function") or ""
        assert content.strip() in prompt


def test_the_ladder_maps_an_emitted_letter_back_to_the_candidate_content():
    task = _patch_task()
    conditions = {condition["name"]: condition
                  for condition in ladder.patch_conditions(task)}
    condition = dict(conditions["permutation_1"], task=task)
    row = ladder.judge_condition(condition, {"text": "```\nA\n```",
                                             "finish_reason": "stop",
                                             "tokens_out": 3})
    assert row["letter"] == "A"
    assert row["mapping"]["A"]["original_index"] == 1
    assert row["chosen_is_reference"] is True


def test_the_ladder_verdict_calls_a_copied_example_what_it_is():
    rows = [
        {"name": "mcq_answer_at_A", "kind": "mcq", "letter": "A",
         "correct": True, "correct_letter": "A", "example_letter": None},
        {"name": "mcq_example_wrong", "kind": "mcq", "letter": "A", "correct": False,
         "correct_letter": "C", "example_letter": "A"},
    ]
    mapping = {"A": {"reference": False, "original_index": 0},
               "B": {"reference": True, "original_index": 1},
               "C": {"reference": False, "original_index": 2}}
    for name, letter in (("control_registered_prompt", "C"),
                         ("no_example_letter", "A"),
                         ("explicit_single_letter", "A"),
                         ("permutation_0", "A"), ("permutation_1", "A"),
                         ("permutation_2", "A")):
        if name.startswith("permutation_"):
            ## The same letter under three rotations points at three different
            ## candidates, which is exactly what makes it position-driven.
            shift = int(name.rsplit("_", 1)[1])
            order = list(range(3))[shift:] + list(range(3))[:shift]
            row_mapping = {chr(ord("A") + position): {
                "original_index": index, "reference": index == 1}
                for position, index in enumerate(order)}
        else:
            row_mapping = mapping
        rows.append({"name": f"item:{name}", "kind": "patch", "letter": letter,
                     "mapping": row_mapping, "reference_letter": "B",
                     "chosen_is_reference": letter == "B",
                     "equals_example": letter == "C",
                     "example_letter": "C" if name.startswith("control") else None})
    verdict = ladder.ladder_verdict(rows)
    assert verdict["parrots_the_example"] is True
    assert verdict["patch"]["item"]["permutation_letter_stable"] is True
    assert verdict["patch"]["item"]["permutation_content_stable"] is False
    assert "prompt-conditioning defect" in verdict["interpretation"]
