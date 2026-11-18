#!/usr/bin/env python3
"""S54: render the stage-1 post-trained frontier table from its receipts.

The stage-1 receipts are `selvedge/s54-stage1-eval/v1` files written by
`scripts/s54_post_trained_eval.py`. This script owns the
`report-s54-stage1` region in `results/REPORT.md`; it is declared under
`external_regions` in `docs/perf-numbers.json`, so the perf-numbers court
refuses an undeclared region and this script refuses a missing receipt.

Usage:
    scripts/court python -- python3 scripts/s54_stage1_report.py --write
    scripts/court python -- python3 scripts/s54_stage1_report.py --check
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = "results/REPORT.md"
SENTINEL = "report-s54-stage1"
BEGIN = "<!-- BEGIN GENERATED: %s -->" % SENTINEL
END = "<!-- END GENERATED: %s -->" % SENTINEL
TASKS = ("gsm8k", "math500")
MODES = ("non-thinking", "thinking")
DECODINGS = ("greedy", "maj16")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _pct(value, places: int = 2) -> str:
    return "n/a" if value is None else f"{value * 100:.{places}f}"


def _cell(entry: dict) -> str:
    accuracy = entry.get("accuracy") or {}
    return (f"{_pct(accuracy.get('accuracy'))} ± "
            f"{_pct(accuracy.get('stderr'))}")


def render(receipts: list[dict], gap_rows: list[dict] | None = None) -> str:
    lines = [
        "### Post-trained frontier: greedy and maj@16 on the registered protocol",
        "",
        "Zero-shot CoT with a final `\\boxed{}`, each checkpoint's own chat "
        "template, greedy and majority-of-16 (temperature 0.8, top_p 0.95), "
        "EOS-only stopping.",
        "",
        "| model | mode | decoding | task | problems | accuracy | strict | flexible | exact | answered | boxed | truncated | tok/s | mean new tokens | peak VRAM GiB | receipt |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for receipt in receipts:
        model = receipt.get("model") or {}
        model_id = model.get("id") or "unknown"
        for task in receipt.get("tasks") or []:
            row = (f"| `{model_id}` | {task['mode']} | {task['decoding']} | "
                   f"{task['task']} | {task['items']} | {_cell(task)} | "
                   f"{_pct(task['strict']['accuracy'])} | "
                   f"{_pct(task['flexible']['accuracy'])} | "
                   f"{_pct(task['exact']['accuracy'])} | "
                   f"{task['answered_samples']}/{task['total_samples']} | "
                   f"{_pct(task['format_correct_fraction'])} | "
                   f"{_pct(task['truncated_fraction'])} | "
                   f"{task['tokens_per_second']:.1f} | "
                   f"{task['mean_new_tokens']:.1f} | "
                   f"{task['peak_vram_gib']:.2f} | "
                   f"`{os.path.relpath(receipt['_path'], ROOT)}` |")
            lines.append(row)
    truncated = []
    for receipt in receipts:
        model_id = (receipt.get("model") or {}).get("id")
        for task in receipt.get("tasks") or []:
            if (task.get("truncated_fraction") or 0.0) >= 1.0:
                truncated.append(f"`{model_id}` {task['task']} "
                                 f"{task['mode']} {task['decoding']}")
    if truncated:
        lines += [
            "",
            ("**Truncation evidence, not a score:** every generation in "
             + ", ".join(truncated) +
             " reached the registered token cap without EOS. The thinking "
             "mode is therefore not scoreable at that cap on this harness; "
             "the non-thinking rows are the usable measurement."),
        ]
    if gap_rows:
        lines += [
            "",
            "#### Greedy-to-maj@16 headroom",
            "",
            "The gap is the free-compute headroom for stage 3. The maj@16 runs "
            "are subsets; the interval is paired at the problem level and "
            "resamples distinct problems, not continuations. A subset gap is "
            "not a full-task score.",
            "",
            "| model | task | mode | greedy | maj@16 | gap (pp) | prompt-level paired 95% CI | problems |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
        for entry in gap_rows:
            gap = entry.get("gap") or {}
            ci = gap.get("ci95") or [None, None]
            lines.append(
                f"| `{entry.get('model_id')}` | {entry.get('task')} | "
                f"{entry.get('mode')} | {_pct(entry.get('greedy_accuracy'))} | "
                f"{_pct(entry.get('maj16_accuracy'))} | "
                f"{_pct(gap.get('delta'))} | "
                f"[{_pct(ci[0])}, {_pct(ci[1])}] | "
                f"{entry.get('common_problems')} |")
    lines += [
        "",
        "`strict` is the last balanced `\\boxed{}` only; `flexible` adds the "
        "registered fallbacks; `exact` compares the flexible extraction with "
        "`math_verify`. The exact basis is named in each receipt. Every cell is "
        "emitted by `scripts/s54_stage1_report.py` from the receipts named in "
        "the table; nothing is typed by hand.",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="results/stage1/s54-stage1-*.json")
    parser.add_argument("--gap-glob", default="results/stage1/s54-gaps-*.json")
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    receipts = []
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        if payload.get("schema") != "selvedge/s54-stage1-eval/v1":
            continue
        if not payload.get("complete"):
            continue
        payload["_path"] = path
        receipts.append(payload)
    if not receipts:
        raise SystemExit(f"s54-stage1-report: no complete receipts under {args.glob}")
    gap_rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, args.gap_glob))):
        payload = json.load(open(path, encoding="utf-8"))
        if payload.get("schema") != "selvedge/s54-stage1-gap/v1":
            continue
        gap_rows.extend(payload.get("rows") or [])
    body = render(receipts, gap_rows)
    doc = os.path.join(ROOT, args.report)
    text = open(doc, encoding="utf-8").read()
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"s54-stage1-report: {args.report} has no {SENTINEL} region")
    current = text[start + len(BEGIN):end].strip("\n")
    if args.check:
        if current != body:
            raise SystemExit("s54-stage1-report: report region is stale")
        print("s54-stage1-report: region is current")
        return 0
    updated = text[:start + len(BEGIN)] + "\n" + body + "\n" + text[end:]
    if updated != text:
        with open(doc, "w", encoding="utf-8") as handle:
            handle.write(updated)
        print(f"s54-stage1-report: updated {args.report}")
    else:
        print("s54-stage1-report: already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
