#!/usr/bin/env python3
"""Failure taxonomy of the stored T2 capability receipts (S28).

Every classification here is computed from files already in the repository: the
five `results/capability/runs/t2-*/generations.json` receipts and the `result.json`
receipts produced by the CPU checker stage. **No model forward pass runs.** The
optional `--recheck` pass compiles stored answer text with `cargo` — a CPU
re-judgement, labelled as such, never a new generation.

Classes (one primary class per arm/item; flags carry the rest):

=====================  =====================================================
class                  meaning
=====================  =====================================================
pass                   the checker passed
tests_failed           compiled, hidden test failed
non_compiling          reached the checker, did not compile
timeout / oom          the checker bound fired
nonzero_exit           the checker exited non-zero without a class
empty_answer           no characters
refusal                an explicit refusal, no code
degenerate_repetition  one line repeated eight or more times
truncated              unterminated fence, or the token cap with no closed answer
wrong_file             only harness-owned or unwritable paths were returned
duplicate_path_rejected  the strict rule rejected a repeated file block
no_code_block          prose or a non-Rust block, nothing extractable
=====================  =====================================================

Flags: `truncated`, `degenerate_repetition`, `empty`, `wrong_file`, `stub`,
`unchanged_initial`, `unbalanced_braces`.
"""

from __future__ import annotations

import json
import os
import statistics

from . import formats
from . import suite as suite_module

SCHEMA = "selvedge/capability-taxonomy/v1"

CLASSES = (
    "pass",
    "tests_failed",
    "non_compiling",
    "timeout",
    "oom",
    "nonzero_exit",
    "empty_answer",
    "refusal",
    "degenerate_repetition",
    "truncated",
    "wrong_file",
    "duplicate_path_rejected",
    "no_code_block",
    "stub_returned",
    "unparseable_choice",
    "other",
)

STATUS_TO_CLASS = {
    "pass": "pass",
    "test_failed": "tests_failed",
    "compile_error": "non_compiling",
    "timeout": "timeout",
    "oom": "oom",
    "nonzero_exit": "nonzero_exit",
}


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _classify_invalid(text: str, rejected: list, tokens_out, max_new_tokens) -> str:
    flags = formats.text_flags(text, tokens_out, max_new_tokens)
    if any("duplicate" in problem for problem in rejected):
        return "duplicate_path_rejected"
    if flags["degenerate_repetition"]:
        return "degenerate_repetition"
    if not text or not text.strip():
        return "empty_answer"
    if formats.REFUSAL.search(text[:400]):
        return "refusal"
    if any(("harness-owned" in problem or "writable set" in problem
            or "not allowed" in problem) for problem in rejected):
        return "wrong_file"
    if flags["unterminated_fence"] or flags["length_capped"]:
        return "truncated"
    return "no_code_block"


def _initial_shape(task) -> dict:
    entry = task.entry
    return {
        "initial_files": len(task.initial_files),
        "source_files": len([path for path in task.initial_files
                             if path.startswith("src/")]),
        "entry_bytes": len(task.read_initial(entry)) if entry in task.initial_files else 0,
        "shape": ("multi-file" if len(task.initial_files) >= 4 else "single-entry"),
    }


def collect(arm_dirs: dict, suite_root: str) -> dict:
    collections = suite_module.load_suite(suite_root)
    tasks = {task.id: task for kind in ("suite", "trio") for task in collections[kind]}
    rows = []
    for arm in sorted(arm_dirs):
        directory = arm_dirs[arm]
        generations = _load(os.path.join(directory, "generations.json"))
        result = _load(os.path.join(directory, "result.json"))
        max_new = (generations.get("decoding") or {}).get("max_new_tokens")
        outputs = {item["id"]: item for item in generations.get("items", [])}
        for row in result.get("items", []):
            task = tasks[row["id"]]
            output = outputs[row["id"]]["output"]
            text = output.get("text") or ""
            tokens_out = output.get("tokens_out")
            flags = formats.text_flags(text, tokens_out, max_new)
            rejected = list((row.get("extraction") or {}).get("rejected") or [])
            status = row.get("status")
            rejection_reason = None
            if rejected:
                joined = " ".join(rejected)
                if "duplicate" in joined:
                    rejection_reason = "duplicate_path"
                elif ("harness-owned" in joined or "writable set" in joined
                      or "manifest is locked" in joined):
                    rejection_reason = "wrong_file"
                elif "no code block" in joined:
                    rejection_reason = "no_code_block"
                else:
                    rejection_reason = rejected[0]
            if row.get("failure_class"):
                # A receipt written after S28 carries its own class; recomputation
                # is for the historical receipts that predate the field.
                primary = row["failure_class"]
            elif status == "invalid_solution":
                primary = _classify_invalid(text, rejected, tokens_out, max_new)
            else:
                primary = STATUS_TO_CLASS.get(status, "other")
            row_flags = []
            if flags["length_capped"]:
                row_flags.append("length_capped")
            if flags["unterminated_fence"]:
                row_flags.append("unterminated_fence")
            if flags["degenerate_repetition"]:
                row_flags.append("degenerate_repetition")
            if primary == "empty_answer":
                row_flags.append("empty")
            if primary == "wrong_file":
                row_flags.append("wrong_file")
            from . import extract as extract_module

            tolerant = extract_module.extract_files(
                text, task.entry, task.allowed_paths(), task.manifest_policy,
                tolerant=True)
            if tolerant.ok:
                for path, content in tolerant.files.items():
                    if path in task.initial_files and \
                            content.strip() == task.read_initial(path).strip():
                        row_flags.append("unchanged_initial")
                        break
            strict_ok = bool((row.get("extraction") or {}).get("ok"))
            rows.append({
                "arm": arm,
                "id": task.id,
                "category": task.category,
                "difficulty": task.metadata.get("difficulty"),
                "format": formats.task_format(task),
                "metric": formats.metric_for(task),
                **_initial_shape(task),
                "status": status,
                "class": primary,
                "flags": sorted(set(row_flags)),
                "extraction_mode": (row.get("extraction") or {}).get("mode"),
                "rejection_reason": rejection_reason,
                "truncated_unterminated_fence": bool(
                    flags["unterminated_fence"] and primary == "truncated"),
                "truncated_cap_only": bool(
                    flags["length_capped"] and not flags["unterminated_fence"]
                    and primary == "truncated"),
                "strict_extracted": strict_ok,
                "tolerant_extracted": tolerant.ok,
                "tolerant_mode": tolerant.mode,
                "tolerant_recovered": bool(tolerant.ok and not strict_ok),
                "answer_chars": len(text),
                "tokens_out": tokens_out,
                "reference_entry_bytes": (
                    len(task.read_reference(task.entry))
                    if task.entry in task.reference_files else None),
            })
    return {"tasks": tasks, "rows": rows}


def summarise(rows: list) -> dict:
    total = len(rows)

    def count(key, value):
        return sum(1 for row in rows if row.get(key) == value)

    by_arm = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "items": len(arm_rows),
            "classes": {name: sum(1 for row in arm_rows if row["class"] == name)
                        for name in CLASSES
                        if any(row["class"] == name for row in arm_rows)},
            "passed": sum(1 for row in arm_rows if row["class"] == "pass"),
            "reached_checker": sum(1 for row in arm_rows if row["class"] in
                                   ("pass", "tests_failed", "non_compiling",
                                    "timeout", "oom", "nonzero_exit")),
            "compiled": sum(1 for row in arm_rows if row["class"] in
                            ("pass", "tests_failed")),
            "strict_extracted": sum(1 for row in arm_rows if row["strict_extracted"]),
            "tolerant_extracted": sum(1 for row in arm_rows
                                      if row["tolerant_extracted"]),
            "tolerant_recovered": sum(1 for row in arm_rows
                                      if row["tolerant_recovered"]),
            "truncated": sum(1 for row in arm_rows if "truncated" in row["flags"]),
            "degenerate": sum(1 for row in arm_rows
                              if "degenerate_repetition" in row["flags"]),
            "median_answer_chars": statistics.median(
                [row["answer_chars"] for row in arm_rows]),
        }

    by_class = {name: sum(1 for row in rows if row["class"] == name)
                for name in CLASSES}
    by_mode = {}
    for mode in sorted({row["extraction_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["extraction_mode"] == mode]
        by_mode[mode] = {
            "items": len(mode_rows),
            "reached_checker": sum(1 for row in mode_rows if row["class"] in
                                   ("pass", "tests_failed", "non_compiling",
                                    "timeout", "oom", "nonzero_exit")),
            "compiled": sum(1 for row in mode_rows if row["class"] in
                            ("pass", "tests_failed")),
            "passed": sum(1 for row in mode_rows if row["class"] == "pass"),
        }
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        task_ids = sorted({row["id"] for row in category_rows})
        compiled_tasks = {row["id"] for row in category_rows
                          if row["class"] in ("pass", "tests_failed")}
        extracted_tasks = {row["id"] for row in category_rows if row["strict_extracted"]}
        by_category[category] = {
            "items": len(category_rows),
            "tasks": len(task_ids),
            "extractable_tasks": len(extracted_tasks),
            "compilable_tasks": len(compiled_tasks),
            "terminal_tasks": len(task_ids) - len(compiled_tasks),
            "passed": sum(1 for row in category_rows if row["class"] == "pass"),
        }
    by_shape = {}
    for shape in sorted({row["shape"] for row in rows}):
        shape_rows = [row for row in rows if row["shape"] == shape]
        task_ids = sorted({row["id"] for row in shape_rows})
        compiled_tasks = {row["id"] for row in shape_rows
                          if row["class"] in ("pass", "tests_failed")}
        extracted_tasks = {row["id"] for row in shape_rows
                           if row["strict_extracted"]}
        by_shape[shape] = {
            "items": len(shape_rows),
            "tasks": len(task_ids),
            "extractable_tasks": len(extracted_tasks),
            "compilable_tasks": len(compiled_tasks),
            "terminal_tasks": len(task_ids) - len(compiled_tasks),
            "passed": sum(1 for row in shape_rows if row["class"] == "pass"),
        }
    by_difficulty = {}
    for difficulty in sorted({str(row["difficulty"]) for row in rows}):
        difficulty_rows = [row for row in rows if str(row["difficulty"]) == difficulty]
        by_difficulty[difficulty] = {
            "items": len(difficulty_rows),
            "extractable": sum(1 for row in difficulty_rows
                               if row["strict_extracted"]),
            "compiled": sum(1 for row in difficulty_rows if row["class"] in
                            ("pass", "tests_failed")),
            "passed": sum(1 for row in difficulty_rows if row["class"] == "pass"),
        }
    recovery = {
        "strict_invalid": sum(1 for row in rows if not row["strict_extracted"]),
        "tolerant_extractable": sum(1 for row in rows if row["tolerant_extracted"]),
        "recovered": sum(1 for row in rows if row["tolerant_recovered"]),
        "still_invalid": sum(1 for row in rows if not row["tolerant_extracted"]),
    }
    by_rejection: dict[str, int] = {}
    for row in rows:
        reason = row.get("rejection_reason")
        if reason:
            by_rejection[reason] = by_rejection.get(reason, 0) + 1
    truncation = {
        "unterminated_fence": sum(1 for row in rows
                                  if row.get("truncated_unterminated_fence")),
        "cap_only": sum(1 for row in rows if row.get("truncated_cap_only")),
        "statement": ("the stored receipts record the batch's decoded width, not the "
                      "item's; `cap_only` is therefore corroborating evidence, while "
                      "an unterminated fence is direct evidence in the answer"),
    }
    return {
        "items": total,
        "classes": by_class,
        "by_arm": by_arm,
        "by_extraction_mode": by_mode,
        "by_category": by_category,
        "by_shape": by_shape,
        "by_difficulty": by_difficulty,
        "by_rejection_reason": dict(sorted(by_rejection.items())),
        "truncation": truncation,
        "tolerant_recovery": recovery,
    }


def recheck(rows: list, tasks: dict, arm_dirs: dict, workroot: str,
            limit: int) -> dict:
    """Compile stored answers that the strict rule discarded but tolerance keeps."""
    from . import checkers

    candidates = [row for row in rows if row["tolerant_recovered"]][:limit]
    outcomes = []
    classes = {}
    for row in candidates:
        task = tasks[row["id"]]
        generations = _load(os.path.join(arm_dirs[row["arm"]], "generations.json"))
        text = next(item["output"]["text"] for item in generations["items"]
                    if item["id"] == row["id"])
        from . import extract as extract_module

        extraction = extract_module.extract_files(
            text, task.entry, task.allowed_paths(), task.manifest_policy,
            tolerant=True)
        if not extraction.ok:
            outcomes.append({"arm": row["arm"], "id": row["id"],
                             "status": "not_extracted"})
            continue
        outcome = checkers.run_check(task, extraction.files, workroot)
        status = outcome.get("status")
        classes[status] = classes.get(status, 0) + 1
        outcomes.append({"arm": row["arm"], "id": row["id"], "status": status,
                         "mode": extraction.mode})
    return {"requested_limit": limit,
            "candidates": len(rows) if not limit else min(limit, sum(
                1 for row in rows if row["tolerant_recovered"])),
            "rechecked": len(outcomes),
            "statuses": classes,
            "recovered_passing": classes.get("pass", 0),
            "recovered_compiling": classes.get("pass", 0) + classes.get("test_failed", 0),
            "items": outcomes,
            "statement": ("a CPU re-judgement of stored answer text under the tolerant "
                          "policy; no model forward pass ran")}


_ARM_DIRS: dict = {}

def analyse(arm_dirs: dict, suite_root: str, recheck_limit: int = 0,
            workroot: str | None = None) -> dict:
    collected = collect(arm_dirs, suite_root)
    rows = collected["rows"]
    summary = summarise(rows)
    payload = {
        "schema": SCHEMA,
        "arms": {arm: arm_dirs[arm] for arm in sorted(arm_dirs)},
        "suite": {"root": os.path.abspath(suite_root),
                  "suite_sha256": suite_module.build_manifest(suite_root)["suite_sha256"]},
        "scope": "stored receipts only; no model inference",
        "classes": list(CLASSES),
        "summary": summary,
        "items": rows,
        "tolerant_recheck": None,
    }
    if recheck_limit:
        if not workroot:
            raise ValueError("--recheck needs a workroot")
        payload["tolerant_recheck"] = recheck(
            rows, collected["tasks"], arm_dirs, workroot, recheck_limit)
        payload["tolerant_recheck"]["candidates_total"] = summary[
            "tolerant_recovery"]["recovered"]
    return payload


# ---------------------------------------------------------------------------
# markdown report


def _table(headers: list, rows: list) -> list:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def render_report(receipt: dict) -> str:
    summary = receipt["summary"]
    lines = [
        "# Capability failure taxonomy — the stored T2 receipts",
        "",
        f"Receipt stamp `{receipt.get('stamp', 'unknown')}`, revision "
        f"`{receipt.get('revision', 'unknown')}`. "
        "Generated by `oracle/capability/diagnose.py` from the stored generation and "
        "check receipts; **no model forward pass ran for this report**. `--check` "
        "regenerates it byte for byte.",
        "",
        f"Arms: {', '.join(receipt['arms'])}. Arm/item outcomes: "
        f"{summary['items']}.",
        "",
        "## Primary failure class per arm",
        "",
    ]
    class_names = [name for name in receipt["classes"]
                   if any(name in entry["classes"] for entry in
                          summary["by_arm"].values())]
    headers = ["arm", "items"] + class_names + ["extracted", "tolerant", "recovered"]
    rows = []
    for arm, entry in summary["by_arm"].items():
        rows.append([arm, entry["items"]]
                    + [entry["classes"].get(name, 0) for name in class_names]
                    + [entry["strict_extracted"], entry["tolerant_extracted"],
                       entry["tolerant_recovered"]])
    lines.extend(_table(headers, rows))
    lines.extend(["", "## Per extraction mode", ""])
    lines.extend(_table(
        ["mode", "items", "reached checker", "compiled", "passed", "compile rate"],
        [[mode, entry["items"], entry["reached_checker"], entry["compiled"],
          entry["passed"], _rate(entry["compiled"], entry["items"])]
         for mode, entry in summary["by_extraction_mode"].items()]))
    lines.extend(["", "## Strict extraction rejection reasons", ""])
    lines.extend(_table(
        ["reason", "items"],
        [[reason, count] for reason, count in
         summary["by_rejection_reason"].items()]))
    lines.extend(["", "## Truncation evidence", ""])
    statement = summary["truncation"]["statement"]
    lines.append(f"Unterminated fence in the answer: {summary['truncation']['unterminated_fence']}. "
                 f"Token cap reached without an unterminated fence: "
                 f"{summary['truncation']['cap_only']}. "
                 + statement[0].upper() + statement[1:] + ".")
    lines.extend(["", "## Per task category", ""])
    lines.extend(_table(
        ["category", "tasks", "extractable", "compilable", "terminal", "passed"],
        [[category, entry["tasks"], entry["extractable_tasks"],
          entry["compilable_tasks"], entry["terminal_tasks"], entry["passed"]]
         for category, entry in summary["by_category"].items()]))
    lines.extend(["", "## Per task shape", ""])
    lines.extend(_table(
        ["shape", "tasks", "extractable", "compilable", "terminal", "passed"],
        [[shape, entry["tasks"], entry["extractable_tasks"],
          entry["compilable_tasks"], entry["terminal_tasks"], entry["passed"]]
         for shape, entry in summary["by_shape"].items()]))
    lines.extend(["", "## Per author difficulty label", ""])
    lines.extend(_table(
        ["difficulty", "items", "extracted", "compiled", "passed"],
        [[difficulty, entry["items"], entry["extractable"], entry["compiled"],
          entry["passed"]]
         for difficulty, entry in summary["by_difficulty"].items()]))
    recovery = summary["tolerant_recovery"]
    lines.extend([
        "", "## Tolerant-extraction recovery (diagnosis only)", "",
        f"Strict-invalid answers: {recovery['strict_invalid']}. Extractable under the "
        f"tolerant policy: {recovery['tolerant_extractable']}. Recovered by tolerance "
        f"alone: {recovery['recovered']}. Still invalid: {recovery['still_invalid']}.",
    ])
    check = receipt.get("tolerant_recheck")
    if check:
        lines.extend([
            "",
            f"CPU re-judgement of up to {check['requested_limit']} recovered answers "
            f"({check['rechecked']} run; no forwards): "
            + ", ".join(f"{status}: {count}"
                        for status, count in sorted(check["statuses"].items())),
        ])
    lines.append("")
    return "\n".join(lines)


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def check_report(path: str, receipt: dict) -> tuple[bool, str]:
    if not os.path.isfile(path):
        return False, f"{path} is missing; regenerate it"
    with open(path, "r", encoding="utf-8") as handle:
        current = handle.read()
    if current != render_report(receipt):
        return False, f"{path} does not match the receipt"
    return True, f"{path} matches the receipt"
