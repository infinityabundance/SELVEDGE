#!/usr/bin/env python3
"""Generate the current-status blocks, and refuse a document that contradicts them.

Status is a claim about receipts, so it is derived from them rather than typed.
This script owns three multi-line generated regions:

    README.md            readme-status
    results/REPORT.md    report-status, report-t2-arm-state

They are declared in `docs/perf-numbers.json` under `external_regions`, so the
perf-numbers court still refuses a generated region nobody owns, and this script
is the only writer.

Two failure modes are checked, and both fail closed:

1. **Stale content.** The block body must equal what the receipts render now. A
   hand-edited verdict, a run that finished since the last write, or a receipt
   stamp that moved makes the block stale.
2. **A contradicting sentence in the prose.** The block is not the only place a
   document can claim a status. If the receipts say the pilot is `done` with
   every arm `evaluated`, a paragraph that says a run *is training*, that there
   *is no verdict*, that *no arm has run the capability suite*, or that *four of
   the five* arms are yet to evaluate is a contradiction, and the court names
   the file, line and rule.

The claim patterns are deliberately narrow and present-tense: history is allowed
to say what was true (with its receipt stamp), and only a claim about *now* is
checked. Nothing in this script reads the network or starts a run.

Usage:
    scripts/court status_consistency                 # --check, the court
    python3 scripts/status_consistency.py --check
    python3 scripts/status_consistency.py --write    # regenerate the regions
    python3 scripts/status_consistency.py --list
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import perf_numbers as pn  # noqa: E402  (sentinel parsing is shared, never copied)

PILOT_REL = "results/t2/pilot-state.json"
VERDICT_REL = "results/t2-verdict.json"
CAPABILITY_REL = "results/capability/runs/t2-comparison.json"
WS19_RUN_REL = "results/stage2/runs/full-20m-math"
WS19_LAUNCH = WS19_RUN_REL + "/launch.json"
WS19_MANIFEST = WS19_RUN_REL + "/manifest.json"
WS19_LOG = WS19_RUN_REL + "/log.jsonl"
WS19_EVAL = "results/stage2/eval/full-20m-math/eval.json"
WS19_SELECTED = "results/stage2/runs/full-20m-math-s49-selected.json"
WS19_AMENDED_EVAL = "results/stage2/eval/full-20m-math-s49/m20/eval.json"
# S51: the live bar is the corrected-optimiser rerun. It is named by its own
# launch receipt, so the status block does not have to infer it from an
# invalidated selection marker.
CURRENT_BAR_REL = "results/stage2/runs/s50-bar-20m-lr1e-5"
CURRENT_BAR_LAUNCH = CURRENT_BAR_REL + "/launch.json"
CURRENT_BAR_MANIFEST = CURRENT_BAR_REL + "/manifest.json"
CURRENT_BAR_LOG = CURRENT_BAR_REL + "/log.jsonl"
CURRENT_BAR_EVAL_ROOT = "results/stage2/eval/s50-bar-20m-lr1e-5"
#: The path the first S51 status block assumed: one `eval.json` at the root of
#: the eval directory. The milestone driver writes `m5/eval.json`,
#: `m10/eval.json` and `m20/eval.json` instead, so this path is only the
#: fallback for a run that predates milestones. Discovery, not this constant,
#: decides whether a score exists.
CURRENT_BAR_EVAL = CURRENT_BAR_EVAL_ROOT + "/eval.json"
INVALIDATION_GLOB = "results/stage2/*invalidation*.json"

#: Milestone directories are `<eval-root>/m<k>` where `k` is the checkpoint's
#: ordinal in the run's own recorded milestone list.
MILESTONE_DIR_RE = re.compile(r"^m(\d+)$")
#: The metric each milestone scored, in preference order. The driver emits
#: lm-evaluation-harness metric names verbatim; a task with none of these keys
#: contributes no number rather than a guessed one.
MILESTONE_SCORE_KEYS = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
    # The registered primary is math_verify; exact_match is the secondary
    # fallback for a receipt that does not declare one. The order was reversed
    # until S60, which reported 8.8% where the primary is 18.6%.
    "minerva_math500": ("math_verify,none", "exact_match,none"),
    "mmlu": ("acc,none",),
    "mmlu_pro": ("exact_match,custom-extract", "acc,none"),
}

REGIONS = (
    ("readme-status", "README.md"),
    ("report-status", "results/REPORT.md"),
    ("report-t2-arm-state", "results/REPORT.md"),
)

TRAINING_STATES = {"training", "running", "launched", "queued"}
EVALUATED_STATES = {"evaluated"}
#: Result eligibility is a property of receipts, not of prose. A result is
#: `currently_eligible` only when its own anchored score exists and no
#: invalidation receipt names it; an invalidated score is retained as
#: `historically_scored` evidence and may never be cited as the current bar.
RESULT_STATES = ("currently_eligible", "historically_scored", "invalidated")


# ---------------------------------------------------------------------------
# receipts


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _last_json_line(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        value = json.loads(last)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _normalise_path(path: str) -> str:
    return str(path).rstrip("/")


def _empty_invalidations() -> dict:
    return {
        "paths": set(),
        "runs": set(),
        "evaluations": set(),
        "selections": set(),
        "masters": [],
        "disposition": None,
        "superseded_by": {},
    }


def load_invalidations(root: str) -> dict:
    """Collect every artefact an invalidation receipt marks as unusable.

    The master records under `results/stage2/*invalidation*.json` name their
    affected runs, evaluations and selections; the per-directory
    `INVALIDATED.json` markers are read too, so a directory is classified even
    if only the marker survived. This function never deletes or edits a receipt;
    it only derives eligibility.
    """
    result = _empty_invalidations()
    for path in sorted(glob.glob(os.path.join(root, INVALIDATION_GLOB))):
        record = read_json(path)
        if not isinstance(record, dict):
            continue
        schema = str(record.get("schema", ""))
        if not (schema.endswith("/invalidation/v1")
                or record.get("kind") == "s50-invalidation"):
            continue
        result["masters"].append(os.path.relpath(path, root))
        result["disposition"] = result["disposition"] or record.get("disposition")
        for key, value in (record.get("superseded_by") or {}).items():
            result["superseded_by"][key] = value
        affected = record.get("affected") or {}
        for category, entries in affected.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if not (entry.get("invalidated")
                        or entry.get("invalidated_as_evidence")):
                    continue
                rel = entry.get("path")
                if not isinstance(rel, str) or not rel:
                    continue
                rel = _normalise_path(rel)
                result["paths"].add(rel)
                bucket = {
                    "runs": "runs", "evaluations": "evaluations",
                    "selections": "selections",
                }.get(category)
                if bucket:
                    result[bucket].add(rel)
    # A marker is authoritative for the directory it sits in, whether or not the
    # master record is present (the marker and the master are written together,
    # but a shallow checkout may have only one).
    for pattern in ("results/stage2/runs/*/INVALIDATED.json",
                    "results/stage2/eval/*/INVALIDATED.json",
                    # A milestone receipt lives one level below the eval
                    # directory (`eval/<run>/m20/eval.json`); a marker there is
                    # authoritative for that milestone only.
                    "results/stage2/eval/*/*/INVALIDATED.json"):
        for marker_path in glob.glob(os.path.join(root, pattern)):
            marker = read_json(marker_path)
            if not isinstance(marker, dict) or not marker.get("invalidated"):
                continue
            directory = _normalise_path(
                os.path.dirname(os.path.relpath(marker_path, root)))
            result["paths"].add(directory)
            if "/runs/" in f"/{directory}/":
                result["runs"].add(directory)
            elif "/eval/" in f"/{directory}/":
                result["evaluations"].add(directory)
    return result


def _is_invalidated(path: str, invalidations: dict) -> bool:
    """True when the path, or any directory above it, is invalidated.

    The master record names run/eval directories while the result is usually a
    file inside one (`eval.json`); a marker in the directory invalidates the
    files below it.
    """
    normalised = _normalise_path(path)
    if not normalised:
        return False
    for invalid in invalidations["paths"]:
        invalid = _normalise_path(invalid)
        if normalised == invalid or normalised.startswith(invalid + "/"):
            return True
    return False


def classify_result(path: str, scored: bool, invalidations: dict,
                    current_bar: str | None = None) -> str:
    """Classify one result artefact for the status machinery.

    * `invalidated` — an invalidation receipt names it; it may only be cited as
      `historically_scored` evidence, never as a bar, baseline or eligible
      result;
    * `currently_eligible` — it has its own anchored score and is the live bar;
    * `historically_scored` — it has a score but is not the live bar (superseded
      without being invalidated);
    * `unscored` — no anchored score exists yet.
    """
    normalised = _normalise_path(path)
    if _is_invalidated(normalised, invalidations):
        return "invalidated"
    if scored and current_bar and normalised == _normalise_path(current_bar):
        return "currently_eligible"
    if scored:
        return "historically_scored"
    return "unscored"


def _milestone_metric(receipt: dict, task: str) -> dict:
    """The task's declared primary metric, its value, and its secondary metrics.

    The eval receipt names the metric it registered as primary
    (`primary_metric`). A preference list of lm-evaluation-harness metric names
    is only the fallback for a receipt that does not declare one. Reading the
    preference list first is what made the status report 8.8% (`exact_match`)
    for MATH-500 where the receipt's primary is `math_verify` at 18.6%.
    """
    for entry in receipt.get("tasks") or []:
        if not isinstance(entry, dict) or entry.get("task") != task:
            continue
        metrics = entry.get("all_metrics") or {}
        declared = entry.get("primary_metric")
        primary_name = None
        primary = None
        if isinstance(declared, str) and isinstance(metrics.get(declared),
                                                    (int, float)):
            primary_name = declared
            primary = float(metrics[declared])
        if primary is None:
            for key in MILESTONE_SCORE_KEYS.get(task, ()):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    primary_name = key
                    primary = float(value)
                    break
        secondary = {
            key: float(metrics[key])
            for key in MILESTONE_SCORE_KEYS.get(task, ())
            if key != primary_name and isinstance(metrics.get(key), (int, float))
        }
        return {
            "task": task,
            "primary_metric": primary_name,
            "primary": primary,
            "secondary": secondary,
            "declared_primary_metric": declared,
        }
    return {"task": task, "primary_metric": None, "primary": None,
            "secondary": {}, "declared_primary_metric": None}


def _milestone_score(receipt: dict, task: str) -> float | None:
    """The task's primary metric value from a milestone eval receipt."""
    return _milestone_metric(receipt, task)["primary"]


def discover_milestones(root: str, eval_root: str, invalidations: dict,
                        tokens_per_step: int | None = None) -> list[dict]:
    """Find, classify and order a run's milestone score receipts.

    Eligibility is derived, not assumed:

    * the directory must be `<eval-root>/m<k>` and contain `eval.json`;
    * the receipt must be `complete` and carry a `stamp` and a `revision`;
    * it must name a checkpoint step, and that step must be one the run's
      launch receipt recorded (when the launch lists milestones);
    * no invalidation receipt may name the directory or the file.

    An invalidated milestone is returned with `state = invalidated` so the
    document can name it as history; it is never the endpoint. An incomplete
    receipt is returned with `state = incomplete` and is never eligible.
    """
    base = os.path.join(root, eval_root)
    if not os.path.isdir(base):
        return []
    launch = read_json(os.path.join(root, CURRENT_BAR_LAUNCH)) if (
        _normalise_path(eval_root) == _normalise_path(CURRENT_BAR_EVAL_ROOT)
    ) else None
    recorded = {
        int(step) for step in ((launch or {}).get("checkpoints") or {}).get(
            "milestones") or []
        if isinstance(step, int) or str(step).isdigit()
    }
    milestones: list[dict] = []
    for directory in sorted(glob.glob(os.path.join(base, "m*"))):
        if not os.path.isdir(directory):
            continue
        label = os.path.basename(directory)
        if not MILESTONE_DIR_RE.match(label):
            continue
        rel = _normalise_path(os.path.relpath(directory, root))
        path = f"{rel}/eval.json"
        receipt = read_json(os.path.join(root, path))
        if not isinstance(receipt, dict):
            continue
        report = receipt.get("arm_report") or {}
        raw_step = report.get("checkpoint_step")
        step = int(raw_step) if isinstance(raw_step, (int, str)) and str(
            raw_step).isdigit() else None
        tokens = report.get("checkpoint_tokens")
        if not isinstance(tokens, int) and step is not None and tokens_per_step:
            tokens = step * int(tokens_per_step)
        invalidated = _is_invalidated(path, invalidations)
        complete = receipt.get("complete") is True
        anchored = bool(receipt.get("stamp")) and bool(receipt.get("revision"))
        step_matches = (step is not None and (not recorded or step in recorded))
        eligible = complete and anchored and step_matches and not invalidated
        if invalidated:
            state = "invalidated"
        elif eligible:
            state = "eligible"
        else:
            state = "incomplete"
        milestones.append({
            "label": label,
            "path": path,
            "stamp": receipt.get("stamp"),
            "revision": (receipt.get("revision") or "")[:12] or None,
            "step": step,
            "tokens": tokens if isinstance(tokens, int) else None,
            "state": state,
            "scores": {task: _milestone_score(receipt, task)
                       for task in MILESTONE_SCORE_KEYS},
            "metrics": {task: _milestone_metric(receipt, task)
                        for task in MILESTONE_SCORE_KEYS},
        })
    milestones.sort(key=lambda item: (
        item["step"] if item["step"] is not None else -1, item["label"]))
    return milestones


def load_ws19(root: str, invalidations: dict | None = None) -> dict | None:
    """Derive the WS19 stage-2 bar's state without reading a live step count.

    The block must stay stable while the bar runs, so it publishes the launch
    stamp, unit, budget and stream hash and a state word, not the current step.
    A step number would make the generated block stale after every line the run
    writes, which would make the court fail for a reason that is not a status
    contradiction. The state moves to `complete` when the recorded step reaches
    the manifest's last step, and to `scored` when an anchored receipt exists.

    When the S49 selection marker exists, the block describes the amended bar
    (the stability rerun), not the lr 1e-4 run that diverged; the diverged run's
    receipt is named as retained history on its own line. S51 additionally names
    the corrected-optimiser rerun by its own launch receipt when that exists,
    and classifies every anchored score against the invalidation receipts.
    """
    invalidations = invalidations or _empty_invalidations()
    current_launch = read_json(os.path.join(root, CURRENT_BAR_LAUNCH))
    milestone_aware = current_launch is not None
    if current_launch is not None:
        run_dir = CURRENT_BAR_REL
        launch = current_launch
        manifest = read_json(os.path.join(root, CURRENT_BAR_MANIFEST))
        log_tail = _last_json_line(os.path.join(root, CURRENT_BAR_LOG))
        amended = False
        selected: dict = {}
        eval_root = CURRENT_BAR_EVAL_ROOT
        fallback_eval = CURRENT_BAR_EVAL
    else:
        marker = read_json(os.path.join(root, WS19_SELECTED))
        selected = (marker or {}).get("selected") or {}
        amended = bool(selected.get("run_dir"))
        if amended:
            run_dir = selected["run_dir"]
            launch = read_json(os.path.join(root, run_dir, "launch.json"))
            manifest = read_json(os.path.join(root, run_dir, "manifest.json"))
            log_tail = _last_json_line(os.path.join(root, run_dir, "log.jsonl"))
            fallback_eval = WS19_AMENDED_EVAL
        else:
            run_dir = WS19_RUN_REL
            launch = read_json(os.path.join(root, WS19_LAUNCH))
            manifest = read_json(os.path.join(root, WS19_MANIFEST))
            log_tail = _last_json_line(os.path.join(root, WS19_LOG))
            fallback_eval = WS19_EVAL
        eval_root = os.path.dirname(fallback_eval)
    if launch is None and manifest is None:
        return None
    steps = int((manifest or {}).get("steps") or 0)
    last_step = int((log_tail or {}).get("step") or -1)
    tokens_per_step = (manifest or {}).get("tokens_per_step")
    milestones = (discover_milestones(root, eval_root, invalidations,
                                      tokens_per_step)
                  if milestone_aware else [])
    eligible = [item for item in milestones if item["state"] == "eligible"]
    endpoint = eligible[-1] if eligible else None
    eval_path = endpoint["path"] if endpoint is not None else fallback_eval
    receipt = read_json(os.path.join(root, eval_path))
    invalidated = _is_invalidated(eval_path, invalidations)
    if endpoint is not None:
        state = "scored"
    elif milestones and all(item["state"] == "invalidated"
                            for item in milestones):
        state = "invalidated"
    elif receipt is not None and not invalidated:
        state = "scored"
    elif steps and last_step >= steps - 1:
        state = "complete"
    elif log_tail is not None:
        state = "running"
    else:
        state = "launched"
    return {
        "state": state,
        "launch": launch,
        "manifest": manifest,
        "eval_present": receipt is not None,
        "eval_stamp": (receipt or {}).get("stamp"),
        "amended": amended,
        "selected": selected,
        "run_dir": run_dir,
        "eval_path": eval_path,
        "invalidated": invalidated,
        "milestones": milestones,
        "eligible_milestones": eligible,
        "endpoint": endpoint,
        "score_state": classify_result(
            eval_path, receipt is not None and not invalidated, invalidations,
            current_bar=eval_path if not invalidated else None),
        "diverged_receipt": read_json(os.path.join(root, WS19_EVAL)),
        "diverged_invalidated": (
            _is_invalidated(WS19_EVAL, invalidations)),
        "amended_invalidated": (
            _is_invalidated(WS19_AMENDED_EVAL, invalidations)),
    }


def load_facts(root: str) -> dict:
    pilot = read_json(os.path.join(root, PILOT_REL))
    verdict = read_json(os.path.join(root, VERDICT_REL))
    capability = read_json(os.path.join(root, CAPABILITY_REL))
    if verdict is not None and verdict.get("schema") != "selvedge/t2-verdict/v1":
        raise pn.Failure(f"{VERDICT_REL}: unexpected schema {verdict.get('schema')!r}")
    if capability is not None and not str(capability.get("schema", "")).startswith(
        "selvedge/capability-arm-comparison/"
    ):
        raise pn.Failure(
            f"{CAPABILITY_REL}: unexpected schema {capability.get('schema')!r}"
        )
    arms = sorted((pilot or {}).get("arms", {}))
    states = {arm: (pilot["arms"][arm].get("status") or "unknown") for arm in arms}
    training = [arm for arm in arms if states[arm] in TRAINING_STATES]
    evaluated = [arm for arm in arms if states[arm] in EVALUATED_STATES]
    invalidations = load_invalidations(root)
    return {
        "pilot": pilot,
        "verdict": verdict,
        "capability": capability,
        "invalidations": invalidations,
        "ws19": load_ws19(root, invalidations),
        "arms": arms,
        "states": states,
        "training_arms": training,
        "evaluated_arms": evaluated,
        "pilot_status": (pilot or {}).get("status"),
        "verdict_present": verdict is not None,
        "capability_present": capability is not None,
    }


# ---------------------------------------------------------------------------
# rendering


def _stamp_of(receipt: dict, rel: str) -> str:
    stamp = receipt.get("stamp") or receipt.get("updated_utc")
    if not isinstance(stamp, str) or not stamp:
        raise pn.Failure(f"{rel}: receipt has no stamp/updated_utc to publish")
    return stamp


def _revision_of(receipt: dict, rel: str) -> str:
    revision = receipt.get("revision")
    if not isinstance(revision, str) or not revision:
        raise pn.Failure(f"{rel}: receipt has no revision to publish")
    return revision[:12]


def _eligibility_line(facts: dict,
                      eligible_paths: Sequence[str] = ()) -> str:
    """Render the eligibility of scored results from the invalidation receipts."""
    invalidations = facts.get("invalidations") or _empty_invalidations()
    masters = invalidations.get("masters") or []
    evaluations = sorted(invalidations.get("evaluations") or [])
    eligible = (", ".join(f"`{path}`" for path in sorted(eligible_paths))
                if eligible_paths else "none")
    if invalidations["paths"]:
        invalid_text = (
            f"{len(invalidations['paths'])} artefact(s) are **`invalidated`**"
            + (f", including the anchored scores "
               f"{', '.join(f'`{path}`' for path in evaluations)}"
               if evaluations else "")
            + "; they are retained as `historically_scored` evidence only and "
              "may not be cited as the current bar, baseline or eligible result"
            + (f" (master record "
               f"{', '.join(f'`{master}`' for master in masters)})"
               if masters else "")
            + ".")
    else:
        invalid_text = "no result is invalidated at this revision."
    return f"- **Result eligibility:** `currently_eligible`: {eligible}; {invalid_text}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _milestone_summary(milestones: Sequence[dict]) -> str:
    """`m5 task `metric` 50.0% [secondary ...] ...` for the status block.

    The metric name is printed beside every number, and a metric the receipt
    also carries but did not declare primary is labelled `secondary`. A reader
    can therefore tell 18.6% (`math_verify,none`) from 8.8%
    (`exact_match,none`) without opening the receipt.
    """
    parts = []
    for item in milestones:
        metrics = item.get("metrics") or {}
        scores = item.get("scores") or {}
        details = []
        for task in ("gsm8k", "minerva_math500", "mmlu", "mmlu_pro"):
            metric = metrics.get(task) or {}
            value = metric.get("primary", scores.get(task))
            if value is None:
                continue
            name = metric.get("primary_metric")
            detail = (f"{task} `{name}` {_percent(value)}" if name
                      else f"{task} {_percent(value)}")
            secondary = metric.get("secondary") or {}
            if secondary:
                detail += " [secondary " + ", ".join(
                    f"`{key}` {_percent(secondary[key])}"
                    for key in sorted(secondary)) + "]"
            details.append(detail)
        step = f"step `{item['step']}`" if item.get("step") is not None else "step ?"
        tokens = (f" / `{item['tokens']}` tokens"
                  if item.get("tokens") is not None else "")
        parts.append(
            f"`{item['label']}` ({step}{tokens}, stamp `{item.get('stamp')}`"
            + (", " + ", ".join(details) if details else "") + ")")
    return "; ".join(parts)


def _milestone_explanation(ws19: dict) -> str | None:
    """The milestone sentence for the live bar, or None when there are none."""
    milestones = ws19.get("milestones") or []
    if not milestones:
        return None
    eligible = ws19.get("eligible_milestones") or []
    endpoint = ws19.get("endpoint")
    if eligible:
        return (
            "- **WS19 stage-2b milestone receipts:** "
            + _milestone_summary(eligible)
            + f". The scored endpoint is `{endpoint['label']}`"
            + (f" at step `{endpoint['step']}`" if endpoint["step"] is not None else "")
            + (f" / `{endpoint['tokens']}` tokens"
               if endpoint["tokens"] is not None else "")
            + f" (`{endpoint['path']}`); no eligible milestone is marked invalidated."
        )
    invalidated = [item for item in milestones if item["state"] == "invalidated"]
    if invalidated:
        return (
            "- **WS19 stage-2b milestone receipts are history:** "
            + _milestone_summary(invalidated)
            + "; an invalidation receipt names them, so none is the current bar."
        )
    return (
        "- **WS19 stage-2b milestone receipts are incomplete:** "
        + _milestone_summary(milestones)
        + "; an incomplete receipt is not a score."
    )


def _eligible_paths(ws19: dict | None) -> list[str]:
    """Every path the status may call `currently_eligible`."""
    if ws19 is None or ws19.get("score_state") != "currently_eligible":
        return []
    milestones = ws19.get("eligible_milestones") or []
    return ([item["path"] for item in milestones] if milestones
            else [ws19["eval_path"]])


def render_readme_status(facts: dict) -> str:
    pilot = facts["pilot"]
    if pilot is None:
        return (
            "No T2 pilot receipt exists at this revision "
            f"(`{PILOT_REL}` is absent), so no status is claimed."
        )
    lines = [
        "Generated by `scripts/status_consistency.py` from the receipts named on",
        "each line; the `status_consistency` court re-derives this block and fails",
        "closed if a sentence in this file contradicts it. Nothing here is typed by",
        "hand.",
        "",
        f"- **T2 pilot `{facts['pilot_status']}`** — started "
        f"`{pilot.get('started_utc')}`, finished `{pilot.get('finished_utc')}`, "
        f"receipt stamp `{_stamp_of(pilot, PILOT_REL)}` (`{PILOT_REL}`).",
        f"- **Arms evaluated: {len(facts['evaluated_arms'])} of {len(facts['arms'])}** "
        f"({', '.join(f'`{arm}`' for arm in facts['arms'])}); "
        f"**{len(facts['training_arms'])} training**, "
        f"**{len(facts['arms']) - len(facts['evaluated_arms']) - len(facts['training_arms'])} "
        "neither trained-and-scored nor training**.",
    ]
    terminal = (facts["pilot_status"] in {"done", "failed"}
                and not facts["training_arms"]
                and len(facts["evaluated_arms"]) == len(facts["arms"]))
    lines.append(
        "- **T2 branch: complete** — the pilot is terminal and every arm is "
        "evaluated; this is the closed branch, not the live work."
        if terminal else
        "- **T2 branch: not complete** — the pilot or an arm is still open."
    )
    ws19 = facts.get("ws19")
    if ws19 is None:
        lines.append(
            "- **WS19 stage-2b: no run receipt** — "
            f"`{WS19_LAUNCH}` is absent at this revision, so no bar run is claimed."
        )
    else:
        launch = ws19["launch"] or {}
        manifest = ws19["manifest"] or {}
        stream = launch.get("stream") or {}
        sha = stream.get("sha256") or (manifest.get("stream_sha256") or "")
        unit = launch.get("run_dir", ws19.get("run_dir", WS19_RUN_REL))
        if ws19["state"] == "invalidated":
            lines.append(
                "- **WS19 stage-2b: `invalidated`** — the only anchored score at "
                f"`{ws19['eval_path']}` (stamp `{ws19['eval_stamp']}`) is "
                "invalidated by the S50 optimiser defect; it is "
                "`historically_scored` evidence, not an eligible bar or baseline."
            )
        elif ws19["state"] == "scored":
            if ws19.get("endpoint") is not None:
                lines.append(
                    "- **WS19 stage-2b: `currently_eligible`** — the 20M "
                    f"math-weighted full-distillation bar (`{unit}`) was launched "
                    f"`{launch.get('created_utc')}` at "
                    f"`{manifest.get('tokens')}` tokens / `{manifest.get('steps')}` "
                    f"steps on stream `{sha[:16]}…`; the scored endpoint is "
                    f"`{ws19['endpoint']['label']}` and every eligible milestone "
                    "receipt is listed on the next line."
                )
            else:
                lines.append(
                    f"- **WS19 stage-2b: `currently_eligible`** — the 20M math-weighted "
                    f"full-distillation bar (`{unit}`) was launched "
                    f"`{launch.get('created_utc')}` at "
                    f"`{manifest.get('tokens')}` tokens / `{manifest.get('steps')}` "
                    f"steps on stream `{sha[:16]}…`; anchored receipt stamp "
                    f"`{ws19['eval_stamp']}` (`{ws19['eval_path']}`)."
                )
        else:
            lines.append(
                f"- **WS19 stage-2b: `{ws19['state']}`** — the 20M math-weighted "
                f"full-distillation bar (`{unit}`) was launched "
                f"`{launch.get('created_utc')}` at "
                f"`{manifest.get('tokens')}` tokens / `{manifest.get('steps')}` "
                f"steps on stream `{sha[:16]}…`; no anchored score exists yet."
            )
        milestone_line = _milestone_explanation(ws19)
        if milestone_line:
            lines.append(milestone_line)
        if ws19["amended"] and ws19["amended_invalidated"]:
            recipe = (ws19["selected"] or {}).get("lr")
            lines.append(
                "- **The S49 stability-amended rerun is retained history** — "
                f"lr `{recipe}` (`{ws19['selected']['run_dir']}`) and its score "
                f"(`{WS19_AMENDED_EVAL}`) are `invalidated` by the S50 optimiser "
                "defect; neither is a bar or baseline."
            )
        elif ws19["amended"]:
            recipe = (ws19["selected"] or {}).get("lr")
            lines.append(
                "- **The bar is the stability-amended rerun** — "
                f"lr `{recipe}`, selected by the registered S49b escalation "
                f"(`{ws19['selected']['run_dir']}`). The lr 1e-4 run is retained "
                f"as a recorded divergence (`{WS19_EVAL}`); it is not the arm the "
                "stage-2 win conditions compare against."
            )
    lines.append(_eligibility_line(facts, eligible_paths=_eligible_paths(ws19)))
    verdict = facts["verdict"]
    if verdict is None:
        lines.append(
            "- **No T2 verdict:** "
            f"`{VERDICT_REL}` does not exist at this revision, so none is claimed."
        )
    else:
        kill = verdict.get("kill_conditions") or {}
        kill_text = ", ".join(
            f"{name} {kill[name]['verdict']}" for name in sorted(kill) if isinstance(kill[name], dict)
        )
        lines.append(
            f"- **Registered T2 outcome: `{verdict['outcome']}`** — K15–K19: "
            f"{kill_text}; verdict receipt stamp `{verdict.get('stamp')}`, revision "
            f"`{_revision_of(verdict, VERDICT_REL)}…` (`{VERDICT_REL}`)."
        )
    capability = facts["capability"]
    if capability is None:
        lines.append(
            "- **No verified-task comparison:** "
            f"`{CAPABILITY_REL}` does not exist at this revision."
        )
    else:
        lines.append(
            f"- **Verified-task comparison: `{capability['outcome']}`** — receipt "
            f"stamp `{capability.get('stamp')}`, revision "
            f"`{_revision_of(capability, CAPABILITY_REL)}…` (`{CAPABILITY_REL}`). A "
            "floor on the instrument is not a ranking of arms; the diagnosis is in "
            "`docs/CAPABILITY-COURT.md` §12."
        )
    return "\n".join(lines)


def render_report_status(facts: dict) -> str:
    pilot = facts["pilot"]
    if pilot is None:
        return (
            "Status: **no T2 pilot receipt at this revision.** Nothing below is a "
            "status claim; the M1/M2 measurements stand on their own receipts."
        )
    verdict = facts["verdict"]
    capability = facts["capability"]
    tail = (
        " The frontier program below (`docs/PROGRAM-PLAN.md` §24, WS19) is "
        "registered; the WS19 stage state above is derived from its launch and "
        "score receipts, and the two committed win conditions are stated per "
        "stage, not only its risks."
    )
    lines = [
        "Status: **the T2 pilot is "
        f"`{facts['pilot_status']}` — {len(facts['evaluated_arms'])} of "
        f"{len(facts['arms'])} arms trained and evaluated, {len(facts['training_arms'])} "
        "training.** M1 rejected static whole-group omission and M2 rejected the "
        "inference-only, post-hoc per-token selection branch; those negatives are "
        "unchanged and are not softened here (`results/M1-FRONTIER.md`, "
        "`results/M2-SELECTORS.md`, `docs/M2-CORRECTION.md`).",
        "",
        "Generated by `scripts/status_consistency.py` from `results/t2/pilot-state.json` "
        f"(stamp `{_stamp_of(pilot, PILOT_REL)}`)"
        + (
            f", `results/t2-verdict.json` (stamp `{verdict.get('stamp')}`)"
            if verdict
            else ""
        )
        + (
            f", `{CAPABILITY_REL}` (stamp `{capability.get('stamp')}`)"
            if capability
            else ""
        )
        + ". The `status_consistency` court re-derives this block and refuses a "
        "document that contradicts it.",
    ]
    if verdict:
        lines.append(
            f"- The registered T2 outcome is **`{verdict['outcome']}`**. The decision "
            "rule was applied mechanically to all five summaries; `results/T2-VERDICT.md` "
            "is rendered from the same receipt."
        )
    if capability:
        lines.append(
            f"- The verified-task comparison outcome is **`{capability['outcome']}`**. "
            "A floor effect cannot rank arms; it is reported as an instrument limit, "
            "not as a statement that any mechanism is impossible."
        )
    ws19 = facts.get("ws19")
    if ws19 is not None:
        lines.append(
            "- WS19 stage-2b is **`"
            + (ws19["score_state"] if ws19["eval_present"] else ws19["state"])
            + f"`** (`{ws19['run_dir']}`); every scored result is classified by "
            "`scripts/status_consistency.py` against the invalidation receipts."
        )
        lines.append(_eligibility_line(facts,
                                     eligible_paths=_eligible_paths(ws19)))
        milestone_line = _milestone_explanation(ws19)
        if milestone_line:
            lines.append(milestone_line)
    else:
        lines.append(_eligibility_line(facts))
    lines.append(tail)
    return "\n".join(lines)


def render_report_arm_state(facts: dict) -> str:
    pilot = facts["pilot"]
    if pilot is None:
        return "No T2 pilot receipt exists at this revision, so there is no per-arm state."
    lines = [
        "| arm | state | training receipt | evaluation receipt | started (UTC) | finished (UTC) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in facts["arms"]:
        record = pilot["arms"][arm]
        lines.append(
            f"| `{arm}` | **{facts['states'][arm]}** | `{record.get('receipt', '—')}` | "
            f"`{record.get('eval_summary') or record.get('eval') or '—'}` | "
            f"{record.get('started_utc') or '—'} | {record.get('finished_utc') or '—'} |"
        )
    lines.append("")
    lines.append(
        "Generated from `results/t2/pilot-state.json` "
        f"(stamp `{_stamp_of(pilot, PILOT_REL)}`); the state column is the receipt's own "
        "`status` field, not an interpretation."
    )
    return "\n".join(lines)


RENDERERS = {
    "readme-status": render_readme_status,
    "report-status": render_report_status,
    "report-t2-arm-state": render_report_arm_state,
}


# ---------------------------------------------------------------------------
# contradiction rules


def _strip_regions(text: str, where: str) -> list[tuple[int, str]]:
    """Prose lines, 1-based, with every generated region body removed."""
    lines, regions = pn.scan_regions(text, where)
    hidden = set()
    for region in regions:
        hidden.update(range(region.begin, region.end + 1))
    return [(index + 1, line) for index, line in enumerate(lines) if index not in hidden]


def contradiction_rules(facts: dict) -> list[dict]:
    rules: list[dict] = []
    terminal = facts["pilot_status"] in {"done", "failed"} and not facts["training_arms"]
    if terminal:
        rules.append(
            {
                "name": "no-run-is-training",
                "why": "the pilot receipt is terminal and no arm's state is training",
                "pattern": re.compile(
                    r"\b(is|are)\s+(currently\s+)?(training|running)\b"
                    r"|\btraining\s+unattended\b|\bowns?\s+the\s+device\b"
                    r"|\boccupies\s+the\s+GPU\b",
                    re.IGNORECASE,
                ),
            }
        )
    if facts["verdict_present"]:
        rules.append(
            {
                "name": "verdict-exists",
                "why": "results/t2-verdict.json exists and states an outcome",
                "pattern": re.compile(
                    r"\bno\s+t2\s+verdict\b|\bthere\s+is\s+no\s+verdict\b"
                    r"|\bbefore\s+a\s+verdict\b|\bnot\s+yet\s+been\s+decided\b"
                    r"|\bpilot\s+is\s+unfinished\b|\brequires\s+all\s+five\s+summaries\b"
                    r"|\bis\s+not\s+finished\b",
                    re.IGNORECASE,
                ),
            }
        )
    if facts["capability_present"]:
        rules.append(
            {
                "name": "capability-suite-has-run",
                "why": "the five arms were scored on the verified-task suite",
                "pattern": re.compile(
                    r"\bno\s+arm\s+has\s+run\s+the\s+(rust\s+)?(capability|verified[- ]task)s?\s+"
                    r"(suite|court)\b"
                    r"|\bno\s+arm\s+has\s+been\s+evaluated\s+on\s+the\s+fresh\s+rust\b",
                    re.IGNORECASE,
                ),
            }
        )
    if facts["arms"] and len(facts["evaluated_arms"]) == len(facts["arms"]):
        rules.append(
            {
                "name": "all-arms-evaluated",
                "why": "every arm's state is evaluated",
                "pattern": re.compile(
                    r"\bfour\s+of\s+the\s+five\b|\b4\s+of\s+the\s+5\b", re.IGNORECASE
                ),
            }
        )
        for arm in facts["arms"]:
            rules.append(
                {
                    "name": f"arm-{arm}-evaluated",
                    "why": f"arm {arm}'s state is evaluated",
                    "pattern": re.compile(
                        rf"\b{re.escape(arm)}\b[^\n.]{{0,80}}\bnot\s+(yet\s+)?(been\s+)?evaluated\b"
                        r"|\bnot\s+(yet\s+)?(been\s+)?evaluated\b[^\n.]{0,80}\b"
                        + re.escape(arm)
                        + r"\b",
                        re.IGNORECASE,
                    ),
                }
            )
    invalidations = facts.get("invalidations") or _empty_invalidations()
    if invalidations["paths"]:
        # Any prose line that names an invalidated artefact and also calls
        # something the current bar/baseline is a contradiction unless it is
        # explicitly marking the artefact as history. The generated regions are
        # stripped before this runs, so the block may name the paths and state
        # that they are invalidated without tripping its own rule.
        names = set()
        for path in invalidations["paths"]:
            rel = _normalise_path(path)
            names.add(rel)
            base = os.path.basename(rel)
            if base:
                names.add(base)
        alternatives = "|".join(
            re.escape(name) for name in sorted(names, key=len, reverse=True))
        rules.append(
            {
                "name": "invalidated-not-current",
                "why": (
                    "an invalidation receipt marks this artefact as usable only "
                    "as historical evidence, never as the current bar or baseline"),
                "pattern": re.compile(
                    rf"({alternatives})[^\n]{{0,120}}\b(is|remains|serves\s+as)\s+"
                    r"(the\s+)?(current\s+)?(bar|baseline)\b"
                    rf"|\b(current|eligible)\b[^\n]{{0,80}}\b(bar|baseline)\b"
                    rf"[^\n]{{0,120}}({alternatives})"
                    rf"|\b(bar|baseline)\b\s*[:=]\s*[^\n]{{0,80}}({alternatives})"
                    rf"|\b(bar|baseline)\s+is\s+[^\n]{{0,80}}({alternatives})\b"
                    rf"|({alternatives})[^\n]{{0,40}}\b(bar|baseline)\b"
                    rf"|({alternatives})[^\n]{{0,60}}\b(scored|score|eligible)\b"
                    rf"|({alternatives})[^\n]{{0,80}}\bcurrent\s+(bar|baseline)\b",
                    re.IGNORECASE,
                ),
                "exempt": re.compile(
                    r"\b(not|no\s+longer|never|invalidated|historical|history|"
                    r"retained|superseded|may\s+not|cannot|isn't|is\s+not|"
                    r"are\s+not)\b",
                    re.IGNORECASE,
                ),
            }
        )
    ws19 = facts.get("ws19") or {}
    if ws19.get("eligible_milestones"):
        run_name = os.path.basename(
            _normalise_path(ws19.get("run_dir") or CURRENT_BAR_REL))
        rules.append(
            {
                "name": "anchored-score-exists",
                "why": (
                    "eligible milestone receipts exist, so a claim that the bar "
                    "is unscored or still in flight contradicts them"),
                "pattern": re.compile(
                    r"\bno\s+anchored\s+score\s+exists\b"
                    rf"|{re.escape(run_name)}[^\n]{{0,160}}\b(no\s+score|unscored|"
                    r"not\s+(yet\s+)?scored|is\s+(currently\s+)?(still\s+)?in\s+flight|"
                    r"is\s+(currently\s+)?(still\s+)?training)"
                    r"|\b(no\s+score|unscored|is\s+(currently\s+)?(still\s+)?in\s+flight|"
                    r"is\s+(currently\s+)?(still\s+)?training)\b[^\n]{{0,160}}"
                    + re.escape(run_name),
                    re.IGNORECASE,
                ),
            }
        )
    return rules


def find_contradictions(doc: str, text: str, rules: list[dict]) -> list[str]:
    problems: list[str] = []
    for number, line in _strip_regions(text, doc):
        for rule in rules:
            if rule["pattern"].search(line):
                exempt = rule.get("exempt")
                if exempt is not None and exempt.search(line):
                    continue
                problems.append(
                    f"{doc}:{number}: status contradiction ({rule['name']}): {line.strip()!r} "
                    f"— {rule['why']}"
                )
    return problems


# ---------------------------------------------------------------------------
# check / write


def expected_blocks(root: str) -> dict[tuple[str, str], str]:
    facts = load_facts(root)
    return {
        (doc, sentinel): RENDERERS[sentinel](facts) for sentinel, doc in REGIONS
    }


def check(root: str) -> int:
    problems: list[str] = []
    expected = expected_blocks(root)
    facts = load_facts(root)
    rules = contradiction_rules(facts)
    docs = sorted({doc for _sentinel, doc in REGIONS})
    for doc in docs:
        path = os.path.join(root, doc)
        if not os.path.exists(path):
            problems.append(f"{doc}: document is missing")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        lines, regions = pn.scan_regions(text, doc)
        by_id = {region.id: region for region in regions}
        for sentinel, region_doc in REGIONS:
            if region_doc != doc:
                continue
            if sentinel not in by_id:
                problems.append(
                    f"{doc}: owned region {sentinel!r} is missing; the status block "
                    "cannot be checked when it is not printed"
                )
                continue
            body = pn.region_body(lines, by_id[sentinel])
            want = expected[(doc, sentinel)]
            if body != want:
                problems.append(
                    f"{doc}:{by_id[sentinel].begin + 1}: stale status block {sentinel!r} "
                    "(the receipts render a different body; run --write)"
                )
        problems.extend(find_contradictions(doc, text, rules))
    if problems:
        for problem in problems:
            print(f"status_consistency: {problem}", file=sys.stderr)
        print(
            f"status_consistency: {len(problems)} problem(s); the status blocks are "
            "derived from receipts, so fix the document, not the receipt",
            file=sys.stderr,
        )
        return 1
    print(
        "status_consistency: OK — "
        f"{len(REGIONS)} block(s) match the receipts and {len(rules)} contradiction "
        "rule(s) found no claim in README/REPORT that the receipts contradict"
    )
    return 0


def write(root: str) -> int:
    expected = expected_blocks(root)
    changed: list[str] = []
    by_doc: dict[str, list[tuple[str, str]]] = {}
    for (doc, sentinel), body in expected.items():
        by_doc.setdefault(doc, []).append((sentinel, body))
    for doc, entries in sorted(by_doc.items()):
        path = os.path.join(root, doc)
        with open(path, "r", encoding="utf-8") as handle:
            original = handle.read()
        lines, regions = pn.scan_regions(original, doc)
        by_id = {region.id: region for region in regions}
        missing = [sentinel for sentinel, _body in entries if sentinel not in by_id]
        if missing:
            raise pn.Failure(
                f"{doc}: owned region(s) {', '.join(missing)} are not in the document; "
                "the sentinels must exist before they can be generated"
            )
        working = list(lines)
        for sentinel, body in sorted(
            entries, key=lambda item: by_id[item[0]].begin, reverse=True
        ):
            region = by_id[sentinel]
            if pn.region_body(working, region) != body:
                changed.append(f"{doc}:{sentinel}")
            working = pn.replace_region(working, region, body)
        new_text = "\n".join(working) + ("\n" if original.endswith("\n") else "")
        if new_text != original:
            directory = os.path.dirname(path) or "."
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, prefix=".status-", delete=False
            )
            try:
                handle.write(new_text)
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(handle.name, path)
            except BaseException:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
                raise
    if changed:
        print(f"status_consistency: updated {len(changed)} block(s): {', '.join(changed)}")
    else:
        print(f"status_consistency: {len(expected)} block(s) already current")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify, never mutate")
    group.add_argument("--write", action="store_true", help="regenerate the status blocks")
    group.add_argument("--list", action="store_true", help="print block -> document")
    parser.add_argument("--repo", default=pn.repo_root())
    args = parser.parse_args(argv)
    root = os.path.abspath(args.repo)
    try:
        if args.check:
            return check(root)
        if args.write:
            return write(root)
        for sentinel, doc in REGIONS:
            print(f"{sentinel:<22} {doc}")
        return 0
    except pn.Failure as exc:
        print(f"status_consistency: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
