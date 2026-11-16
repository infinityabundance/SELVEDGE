#!/usr/bin/env python3
"""Aggregate one calibration pass into the numbers the report publishes.

    python3 scripts/capability_calibration_analysis.py \
      --calibration results/capability/calibration-<stamp>.json \
      --result D=... --result M=... --result X=... --result H=... --result 'H*=...' \
      --generations D=... --generations M=... --generations X=... \
      --generations H=... --generations 'H*=...' \
      --out results/capability/calibration-analysis-<stamp>.json

The calibration receipt already holds the per-item window, cells, ladder and
power arithmetic. This script adds what a reader needs to judge *why* the window
closed: pass counts per arm and per format, the status mix, and the shape of the
stored answers.

`answer_shape` is a projection, not the registered parser. It scans the stored
generation text for a standalone candidate letter and reports how often one
letter is recoverable, how often several are (ambiguous, never resolved by
guessing), and how often the recovered letter is the reference. The registered
run in `check` is unaffected; this block exists to distinguish "the arm answered
a different way" from "the arm did not answer".
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import formats, suite as suite_module  # noqa: E402

SCHEMA = "selvedge/capability-calibration-analysis/v1"
LETTER_LINE = re.compile(r"^[A-E]$")
FENCED_LETTER = re.compile(r"```[A-Za-z]*\s*\n\s*([A-E])\s*\n\s*```")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pairs(values: list[str], flag: str) -> dict:
    result = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path:
            raise SystemExit(f"{flag} must be NAME=PATH, got {value!r}")
        result[name] = path
    return result


def standalone_letters(text: str) -> list[str]:
    letters = []
    for line in text.splitlines():
        stripped = line.strip().strip("`").strip()
        if LETTER_LINE.match(stripped):
            letters.append(stripped)
    letters.extend(FENCED_LETTER.findall(text))
    return letters


def answer_shape(arms: list[str], generations: dict, tasks: dict,
                 fmt: str) -> dict:
    by_arm = {}
    totals = {"recoverable": 0, "unambiguous": 0, "ambiguous": 0, "none": 0,
              "correct": 0, "items": 0}
    reference_letters: dict = {}
    for arm in arms:
        outputs = {item["id"]: item["output"]["text"]
                   for item in generations[arm]["items"]}
        counts = {"recoverable": 0, "unambiguous": 0, "ambiguous": 0, "none": 0,
                  "correct": 0, "items": 0}
        chosen: dict = {}
        for task in tasks:
            if formats.task_format(task) != fmt:
                continue
            spec = formats.format_spec(task)
            options = spec["candidates"] if fmt == formats.FORMAT_PATCH else spec["options"]
            letters = [chr(ord("A") + index) for index in range(len(options))]
            reference = None
            for index, option in enumerate(options):
                if fmt == formats.FORMAT_PATCH:
                    if option.get("reference"):
                        reference = letters[index]
                elif (option.get("consequence") or {}).get("kind") == "reference":
                    reference = letters[index]
            if reference is None:
                raise SystemExit(f"{task.id}: no reference option")
            if arm == arms[0]:
                reference_letters[reference] = reference_letters.get(reference, 0) + 1
            found = sorted({letter for letter in standalone_letters(outputs[task.id])
                            if letter in letters})
            counts["items"] += 1
            totals["items"] += 1
            if not found:
                counts["none"] += 1
                totals["none"] += 1
                continue
            counts["recoverable"] += 1
            totals["recoverable"] += 1
            if len(found) > 1:
                counts["ambiguous"] += 1
                totals["ambiguous"] += 1
                continue
            counts["unambiguous"] += 1
            totals["unambiguous"] += 1
            chosen[found[0]] = chosen.get(found[0], 0) + 1
            if found[0] == reference:
                counts["correct"] += 1
                totals["correct"] += 1
        by_arm[arm] = dict(counts, chosen_letters=chosen,
                           constant=len(chosen) == 1,
                           constant_letter=(next(iter(chosen)) if len(chosen) == 1 else None))
    constant_arms = [arm for arm, entry in by_arm.items() if entry["constant"]]
    return {
        "format": fmt,
        "method": ("standalone candidate letter in the stored generation "
                   "(a line or a one-letter fence); projection, not the "
                   "registered parser, and never used to score an item"),
        "by_arm": by_arm,
        "totals": totals,
        "reference_letters": dict(sorted(reference_letters.items())),
        "arms_with_a_constant_letter": constant_arms,
        "constant_letter_count": len(constant_arms),
    }


def resource_summary(generations: dict, smoke: dict | None) -> dict:
    """Measured peaks from the generation receipts, in GiB.

    The numbers a container limit is sized from have to be readable in the same
    place as the run they sized, so the report cites this block instead of a
    converted copy of the raw bytes.
    """
    def one(receipt: dict) -> dict:
        resources = receipt.get("resources") or {}
        vram = resources.get("vram") or {}
        return {
            "process_maxrss_gib": round(
                resources.get("process_maxrss_bytes", 0) / (1024 ** 3), 3),
            "cgroup_memory_peak_gib": round(
                resources.get("cgroup_memory_peak_bytes", 0) / (1024 ** 3), 3),
            "vram_max_memory_allocated_gib": vram.get("max_memory_allocated_gib"),
            "vram_max_memory_reserved_gib": vram.get("max_memory_reserved_gib"),
            "items": len(receipt.get("items") or []),
        }

    by_arm = {arm: one(receipt) for arm, receipt in sorted(generations.items())}
    summary = {"by_arm": by_arm,
               "max_process_maxrss_gib": max(
                   (entry["process_maxrss_gib"] for entry in by_arm.values()),
                   default=0.0),
               "max_cgroup_memory_peak_gib": max(
                   (entry["cgroup_memory_peak_gib"] for entry in by_arm.values()),
                   default=0.0),
               "max_vram_allocated_gib": max(
                   (entry["vram_max_memory_allocated_gib"] or 0.0
                    for entry in by_arm.values()), default=0.0)}
    if smoke:
        summary["smoke"] = one(smoke)
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--generations", action="append", default=[])
    parser.add_argument("--smoke", default=None,
                        help="one-arm smoke generations receipt, if there was one")
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures", "capability-calibrated"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result_paths = pairs(args.result, "--result")
    generation_paths = pairs(args.generations, "--generations")
    if set(result_paths) != set(generation_paths):
        raise SystemExit("--result and --generations must name the same arms")
    calibration = load(args.calibration)
    arms = sorted(result_paths)
    results = {arm: load(path) for arm, path in result_paths.items()}
    generations = {arm: load(path) for arm, path in generation_paths.items()}
    tasks = suite_module.load_suite(args.suite)["suite"]

    per_arm = {}
    per_format: dict = {}
    for arm in arms:
        items = results[arm]["items"]
        statuses: dict = {}
        for item in items:
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
            bucket = per_format.setdefault(item["format"], {
                "items": 0, "executable_items": 0, "executable_passes": 0,
                "diagnostic_items": 0, "diagnostic_passes": 0, "by_arm": {}})
            bucket["items"] += 1
            bucket["by_arm"].setdefault(arm, {
                "items": 0, "executable_items": 0, "executable_passes": 0,
                "diagnostic_items": 0, "diagnostic_passes": 0})
            bucket["by_arm"][arm]["items"] += 1
            if item["metric"] == formats.METRIC_EXECUTABLE:
                bucket["executable_items"] += 1
                bucket["by_arm"][arm]["executable_items"] += 1
                if item["value"] == 1.0:
                    bucket["executable_passes"] += 1
                    bucket["by_arm"][arm]["executable_passes"] += 1
            else:
                bucket["diagnostic_items"] += 1
                bucket["by_arm"][arm]["diagnostic_items"] += 1
                if item["value"] == 1.0:
                    bucket["diagnostic_passes"] += 1
                    bucket["by_arm"][arm]["diagnostic_passes"] += 1
        executable = [item for item in items
                      if item["metric"] == formats.METRIC_EXECUTABLE]
        diagnostic = [item for item in items
                      if item["metric"] == formats.METRIC_DIAGNOSTIC]
        per_arm[arm] = {
            "items": len(items),
            "executable_items": len(executable),
            "executable_passes": sum(1 for item in executable if item["value"] == 1.0),
            "diagnostic_items": len(diagnostic),
            "diagnostic_passes": sum(1 for item in diagnostic if item["value"] == 1.0),
            "diagnostic_passing_ids": sorted(item["id"] for item in diagnostic
                                             if item["value"] == 1.0),
            "statuses": dict(sorted(statuses.items())),
        }

    records = calibration["items"]
    executable_records = [record for record in records
                          if record["metric"] == formats.METRIC_EXECUTABLE]
    window = {
        "accepted": sum(1 for record in executable_records if record["accepted"]),
        "retired": sum(1 for record in executable_records if not record["accepted"]),
        "retired_zero_pass": sum(1 for record in executable_records
                                 if not record["accepted"]
                                 and record["passes"] == 0),
        "retired_non_discriminating": sum(1 for record in executable_records
                                          if not record["accepted"]
                                          and record["passes"] > 0),
        "saturated": sum(1 for record in executable_records
                         if record["passes"] == record["arms_scored"]),
        "partial_credit_spread_available": sum(
            1 for record in executable_records
            if record["partial_credit_spread"] is not None),
    }

    categories = sorted({record["category"] for record in executable_records})
    power = calibration["power"]
    z_total = power["required_n"]["planning"]["z_total"]
    available = max(power.get("items_per_category_available", {}).values() or [0])
    mde = {}
    for name, variance in (("planning", power["planning_variance"]),
                           ("conservative", power["conservative_variance"])):
        effect = z_total * (variance / (len(categories) * available)) ** 0.5
        mde[name] = {
            "variance": variance,
            "categories": len(categories),
            "available_items_per_category": available,
            "absolute_effect": effect,
            "relative_to_measured_baseline": (
                effect / power["measured_baseline_pass_rate"]
                if power["measured_baseline_pass_rate"] else None),
            "formula": ("z_total * sqrt(Var(d) / (categories * n_available)); "
                        "the inverse of the registered required-n formula"),
        }

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "revision": revision(),
        "command": " ".join(sys.argv),
        "inputs": {
            "calibration": {"path": args.calibration,
                            "sha256": sha256_file(args.calibration),
                            "stamp": calibration.get("stamp")},
            "results": {arm: {"path": path, "sha256": sha256_file(path)}
                        for arm, path in sorted(result_paths.items())},
            "generations": {arm: {"path": path, "sha256": sha256_file(path)}
                            for arm, path in sorted(generation_paths.items())},
        },
        "arms": arms,
        "per_arm": per_arm,
        "per_format": dict(sorted(per_format.items())),
        "item_window": window,
        "cells": {
            "total": len(calibration["cells"]),
            "scored": sum(1 for cell in calibration["cells"] if cell["scored"]),
            "reformatted": sum(1 for cell in calibration["cells"]
                               if not cell["scored"]),
        },
        "ladder": calibration["ladder"],
        "power": {
            "measured_baseline_pass_rate": power["measured_baseline_pass_rate"],
            "planning_variance": power["planning_variance"],
            "conservative_variance": power["conservative_variance"],
            "required_n_planning": power["required_n"]["planning"]["n_per_category_ceil"],
            "required_n_conservative": power["required_n"]["conservative"]["n_per_category_ceil"],
            "items_per_category_available": power.get("items_per_category_available"),
            "shortfall": power.get("shortfall"),
            "minimum_detectable_effect": mde,
        },
        "answer_shape": {
            "patch": answer_shape(arms, generations, tasks, formats.FORMAT_PATCH),
            "diagnostic": answer_shape(arms, generations, tasks,
                                       formats.FORMAT_DIAGNOSTIC),
        },
        "resources": resource_summary(
            generations, load(args.smoke) if args.smoke else None),
        "suite_version": calibration.get("suite_version"),
    }
    shapes = payload["answer_shape"]
    power_block = payload["power"]
    payload["rendered"] = {
        **{f"arm_{arm}": (f"{entry['executable_passes']} of "
                           f"{entry['executable_items']} executable passes; "
                           f"{entry['diagnostic_passes']} of "
                           f"{entry['diagnostic_items']} verified diagnostics")
           for arm, entry in per_arm.items()},
        **{f"format_{fmt}": (f"{entry['executable_passes']} of "
                              f"{entry['executable_items']} verified executable passes"
                              if entry["executable_items"]
                              else f"{entry['diagnostic_passes']} of "
                                   f"{entry['diagnostic_items']} verified diagnostics")
           for fmt, entry in per_format.items()},
        "patch_answer_shape": (
            f"{shapes['patch']['totals']['recoverable']} of "
            f"{shapes['patch']['totals']['items']} answers hold a standalone "
            f"letter, {shapes['patch']['totals']['ambiguous']} of them ambiguous, "
            f"and {shapes['patch']['totals']['correct']} name the reference"),
        "diagnostic_answer_shape": (
            f"{shapes['diagnostic']['totals']['recoverable']} of "
            f"{shapes['diagnostic']['totals']['items']} answers hold a standalone "
            f"letter, {shapes['diagnostic']['totals']['ambiguous']} of them "
            f"ambiguous, and {shapes['diagnostic']['totals']['correct']} name "
            "the reference"),
        "patch_reference_letters": ", ".join(
            f"{letter} {count}" for letter, count in
            shapes['patch']['reference_letters'].items()),
        "diagnostic_reference_letters": ", ".join(
            f"{letter} {count}" for letter, count in
            shapes['diagnostic']['reference_letters'].items()),
        "patch_constant_arms": str(shapes['patch']['constant_letter_count']),
        "diagnostic_constant_arms": str(
            shapes['diagnostic']['constant_letter_count']),
        "window": (f"{window['accepted']} accepted, {window['retired']} retired "
                   f"({window['retired_zero_pass']} zero pass, "
                   f"{window['retired_non_discriminating']} does not discriminate), "
                   f"{window['saturated']} saturated, "
                   f"{window['partial_credit_spread_available']} with a partial-credit "
                   "spread"),
        "cells": (f"{payload['cells']['scored']} scored, "
                  f"{payload['cells']['reformatted']} reformatted of "
                  f"{payload['cells']['total']}"),
        "ladder": ("monotone" if calibration["ladder"]["monotone"] else
                   "not monotone: " + ", ".join(calibration["ladder"]["violations"])),
        "baseline": f"{power_block['measured_baseline_pass_rate']:.6f}",
        "planning_variance": f"{power_block['planning_variance']:.6f}",
        "conservative_variance": f"{power_block['conservative_variance']:.6f}",
        "required_n_planning": f"{power_block['required_n_planning']:,}",
        "required_n_conservative": f"{power_block['required_n_conservative']:,}",
        "available_items": str(max(
            (power_block.get("items_per_category_available") or {0: 0}).values())),
        "mde_planning": (f"{mde['planning']['absolute_effect']:.4f} absolute, "
                         f"{mde['planning']['relative_to_measured_baseline']:.2f}x baseline"),
        "mde_conservative": (
            f"{mde['conservative']['absolute_effect']:.4f} absolute, "
            f"{mde['conservative']['relative_to_measured_baseline']:.2f}x baseline"),
    }
    smoke_entry = payload["resources"].get("smoke") or {}
    payload["rendered"]["smoke_peak"] = (
        f"{smoke_entry.get('process_maxrss_gib', 0.0):.3f} GiB process peak, "
        f"{smoke_entry.get('cgroup_memory_peak_gib', 0.0):.3f} GiB cgroup peak, "
        f"{smoke_entry.get('vram_max_memory_allocated_gib') or 0.0:.3f} GiB VRAM "
        "allocated")
    payload["rendered"]["arm_peak"] = (
        f"{payload['resources']['max_process_maxrss_gib']:.3f} GiB process peak, "
        f"{payload['resources']['max_cgroup_memory_peak_gib']:.3f} GiB cgroup peak, "
        f"{payload['resources']['max_vram_allocated_gib']:.3f} GiB VRAM allocated")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"capability_calibration_analysis: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
