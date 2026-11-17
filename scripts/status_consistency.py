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
import json
import os
import re
import sys
import tempfile

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

REGIONS = (
    ("readme-status", "README.md"),
    ("report-status", "results/REPORT.md"),
    ("report-t2-arm-state", "results/REPORT.md"),
)

TRAINING_STATES = {"training", "running", "launched", "queued"}
EVALUATED_STATES = {"evaluated"}


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


def load_ws19(root: str) -> dict | None:
    """Derive the WS19 stage-2 bar's state without reading a live step count.

    The block must stay stable while the bar runs, so it publishes the launch
    stamp, unit, budget and stream hash and a state word, not the current step.
    A step number would make the generated block stale after every line the run
    writes, which would make the court fail for a reason that is not a status
    contradiction. The state moves to `complete` when the recorded step reaches
    the manifest's last step, and to `scored` when an anchored receipt exists.

    When the S49 selection marker exists, the block describes the amended bar
    (the stability rerun), not the lr 1e-4 run that diverged; the diverged run's
    receipt is named as retained history on its own line.
    """
    marker = read_json(os.path.join(root, WS19_SELECTED))
    selected = (marker or {}).get("selected") or {}
    amended = bool(selected.get("run_dir"))
    if amended:
        run_dir = selected["run_dir"]
        launch = read_json(os.path.join(root, run_dir, "launch.json"))
        manifest = read_json(os.path.join(root, run_dir, "manifest.json"))
        log_tail = _last_json_line(os.path.join(root, run_dir, "log.jsonl"))
        receipt = read_json(os.path.join(root, WS19_AMENDED_EVAL))
    else:
        launch = read_json(os.path.join(root, WS19_LAUNCH))
        manifest = read_json(os.path.join(root, WS19_MANIFEST))
        log_tail = _last_json_line(os.path.join(root, WS19_LOG))
        receipt = read_json(os.path.join(root, WS19_EVAL))
    if launch is None and manifest is None:
        return None
    steps = int((manifest or {}).get("steps") or 0)
    last_step = int((log_tail or {}).get("step") or -1)
    if receipt is not None:
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
        "eval_path": WS19_AMENDED_EVAL if amended else WS19_EVAL,
        "diverged_receipt": read_json(os.path.join(root, WS19_EVAL)),
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
    return {
        "pilot": pilot,
        "verdict": verdict,
        "capability": capability,
        "ws19": load_ws19(root),
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
        lines.append(
            f"- **WS19 stage-2b: `{ws19['state']}`** — the 20M math-weighted "
            f"full-distillation bar (`{launch.get('unit', WS19_RUN_REL)}`) was "
            f"launched `{launch.get('created_utc')}` at "
            f"`{manifest.get('tokens')}` tokens / `{manifest.get('steps')}` "
            f"steps on stream `{sha[:16]}…`; "
            + (f"anchored receipt stamp `{ws19['eval_stamp']}` "
               f"(`{ws19['eval_path']}`)." if ws19["eval_present"] else
               "no anchored score exists yet.")
        )
        if ws19["amended"]:
            recipe = (ws19["selected"] or {}).get("lr")
            lines.append(
                "- **The bar is the stability-amended rerun** — "
                f"lr `{recipe}`, selected by the registered S49b escalation "
                f"(`{ws19['selected']['run_dir']}`). The lr 1e-4 run is retained "
                f"as a recorded divergence (`{WS19_EVAL}`); it is not the arm the "
                "stage-2 win conditions compare against."
            )
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
        "registered; its stages are not started, and the two committed win "
        "conditions are stated per stage, not only its risks."
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
    return rules


def find_contradictions(doc: str, text: str, rules: list[dict]) -> list[str]:
    problems: list[str] = []
    for number, line in _strip_regions(text, doc):
        for rule in rules:
            if rule["pattern"].search(line):
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
