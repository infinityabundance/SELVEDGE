#!/usr/bin/env python3
"""The S30 likelihood-selection pilot: D and X on the selection formats.

Inputs are the two selection receipts and their checked results. The output is one
receipt that answers the questions the registration asked, in its own terms:

* per-format accuracy for each arm, from the executable checker (patch items) and
  from verified diagnostics (diagnostic items), never from the ranking margin;
* the across-arm spread and the item-level cross-tab, which says whether any cell
  is non-zero and non-saturated;
* the response-entropy screen per arm and format: a format whose answers are
  near-constant is marked non-discriminating and excluded, not scored;
* the measured baseline and paired-difference variance, and the required n for 80%
  power at a 10% relative effect — or the explicit statement that the comparison
  cannot be powered at the available n.

It never pools the diagnostic metric with executable success, never opens the
sealed holdback, and never reports an arm comparison as a capability result.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import calibrate, formats, forensics, suite as suite_module  # noqa: E402

SCHEMA = "selvedge/capability-pilot/v1"
RANKING_MODES = ("content-mean", "content-sum", "letter-mean", "letter-sum")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pairs(values: list, flag: str) -> dict:
    named = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        named[name] = path
    return named


def revision() -> str:
    """The commit this receipt was produced at (perf-number blocks cite it)."""
    import subprocess

    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resource_summary(selections: dict) -> dict:
    """Measured peaks, in GiB, from each selection receipt.

    A container limit is sized from these, so the number a limit is derived from
    must be readable in the receipt the run wrote.
    """
    summary = {}
    for arm, receipt in sorted(selections.items()):
        resources = receipt.get("resources") or {}
        vram = resources.get("vram") or {}
        summary[arm] = {
            "process_maxrss_gib": round(
                resources.get("process_maxrss_bytes", 0) / (1024 ** 3), 3),
            "cgroup_memory_peak_gib": round(
                resources.get("cgroup_memory_peak_bytes", 0) / (1024 ** 3), 3),
            "vram_max_memory_allocated_gib": vram.get("max_memory_allocated_gib"),
            "vram_max_memory_reserved_gib": vram.get("max_memory_reserved_gib"),
            "items": len(receipt.get("items") or []),
        }
    return summary


def exact_sign_test(discordant: int, one_way: int) -> float:
    """Two-sided exact binomial p for `one_way` of `discordant` discordant pairs."""
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(one_way, discordant + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def per_format(result: dict) -> dict:
    """Per-format items and passes, executable and diagnostic kept apart."""
    table: dict = {}
    for row in result["items"]:
        entry = table.setdefault(row["format"], {
            "items": 0, "executable_items": 0, "executable_passes": 0,
            "diagnostic_items": 0, "diagnostic_passes": 0,
            "pass_ids": [], "executable_ids": []})
        entry["items"] += 1
        if row["metric"] == formats.METRIC_EXECUTABLE:
            entry["executable_items"] += 1
            entry["executable_ids"].append(row["id"])
            if row["status"] == "pass":
                entry["executable_passes"] += 1
                entry["pass_ids"].append(row["id"])
        else:
            entry["diagnostic_items"] += 1
            if row["status"] == "pass":
                entry["diagnostic_passes"] += 1
                entry["pass_ids"].append(row["id"])
    for entry in table.values():
        primary = entry["executable_items"] or entry["diagnostic_items"]
        passes = (entry["executable_passes"] if entry["executable_items"]
                  else entry["diagnostic_passes"])
        entry["accuracy"] = passes / primary if primary else None
        entry["metric"] = (formats.METRIC_EXECUTABLE if entry["executable_items"]
                           else formats.METRIC_DIAGNOSTIC)
    return table


def ranking_agreement(selection: dict) -> dict:
    """How often each ranking mode chose the reference candidate.

    This is a *ranking* score, not an item score: it says whether a mode would
    have chosen the candidate the checker verified, and it is reported as a
    diagnostic beside the compiled outcome.
    """
    table = {}
    for mode in RANKING_MODES:
        correct = 0
        total = 0
        by_format: dict = {}
        for item in selection["items"]:
            entry = item["output"]["extra"]["selection"]
            ranking = entry["rankings"][mode]
            hit = ranking["choice"] == entry["reference"]
            total += 1
            correct += 1 if hit else 0
            bucket = by_format.setdefault(item["format"], [0, 0])
            bucket[0] += 1
            bucket[1] += 1 if hit else 0
        table[mode] = {
            "correct": correct,
            "items": total,
            "accuracy": correct / total if total else None,
            "scored": mode == (selection["instrument"].get("primary_mode")
                               or (selection.get("decoding") or {}).get(
                                   "primary_mode")),
            "by_format": {fmt: {"items": values[0], "correct": values[1],
                                "accuracy": values[1] / values[0]}
                          for fmt, values in sorted(by_format.items())},
        }
    return table


def margin_summary(result: dict) -> dict:
    """The primary ranking margins per format, from the checked rows."""
    table: dict = {}
    for row in result["items"]:
        selection = row.get("selection") or {}
        primary = selection.get("primary") or {}
        if primary.get("margin") is None:
            continue
        table.setdefault(row["format"], []).append(primary["margin"])
    return {fmt: {"items": len(values),
                  "median": statistics.median(values),
                  "min": min(values), "max": max(values),
                  "mean": statistics.fmean(values)}
            for fmt, values in sorted(table.items())}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--selection", action="append", default=[],
                        metavar="ARM=PATH")
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures",
                                                        "capability-calibrated"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    result_paths = pairs(args.result, "--result")
    selection_paths = pairs(args.selection, "--selection")
    if set(result_paths) != set(selection_paths):
        raise SystemExit("--result and --selection must name the same arms")
    arms = sorted(result_paths)
    results = {arm: load(path) for arm, path in result_paths.items()}
    selections = {arm: load(path) for arm, path in selection_paths.items()}
    tasks = suite_module.load_suite(args.suite)["suite"]

    per_arm = {arm: {"per_format": per_format(results[arm]),
                     "ranking_agreement": ranking_agreement(selections[arm]),
                     "margins": margin_summary(results[arm]),
                     "diagnostic_passes": sum(
                         1 for row in results[arm]["items"]
                         if row["metric"] == formats.METRIC_DIAGNOSTIC
                         and row["status"] == "pass"),
                     "diagnostic_items": sum(
                         1 for row in results[arm]["items"]
                         if row["metric"] == formats.METRIC_DIAGNOSTIC)}
               for arm in arms}

    ## Per-item executable outcomes: the only input to the paired comparison.
    item_ids = sorted({row["id"] for row in results[arms[0]]["items"]
                       if row["metric"] == formats.METRIC_EXECUTABLE})
    status = {arm: {row["id"]: row["status"]
                    for row in results[arm]["items"]
                    if row["metric"] == formats.METRIC_EXECUTABLE}
              for arm in arms}
    both_pass = [item_id for item_id in item_ids
                 if all(status[arm][item_id] == "pass" for arm in arms)]
    neither = [item_id for item_id in item_ids
               if all(status[arm][item_id] != "pass" for arm in arms)]
    one_way = {}
    for arm in arms:
        others = [other for other in arms if other != arm]
        one_way[arm] = [item_id for item_id in item_ids
                        if status[arm][item_id] == "pass"
                        and all(status[other][item_id] != "pass"
                                for other in others)]
    discordant = sum(len(value) for value in one_way.values())
    passed = {arm: sum(1 for item_id in item_ids if status[arm][item_id] == "pass")
              for arm in arms}
    baseline = sum(passed.values()) / (len(arms) * len(item_ids)) if item_ids else 0.0
    differences = [((1.0 if status[arms[0]][item_id] == "pass" else 0.0)
                    - (1.0 if status[arms[1]][item_id] == "pass" else 0.0))
                   for item_id in item_ids] if len(arms) >= 2 else []
    variance = statistics.pvariance(differences) if len(differences) >= 2 else None

    categories = len({row["category"] for row in results[arms[0]]["items"]
                      if row["metric"] == formats.METRIC_EXECUTABLE})
    power = None
    if baseline > 0 and variance is not None:
        try:
            power = calibrate.required_n(baseline, 0.10, variance, categories)
        except calibrate.CalibrationError:
            power = None
    available = {}
    for row in results[arms[0]]["items"]:
        if row["metric"] == formats.METRIC_EXECUTABLE:
            available[row["category"]] = available.get(row["category"], 0) + 1
    n_available = max(available.values()) if available else 0
    mde = None
    if power and n_available:
        z_total = power["z_total"]
        effect = z_total * (variance / (categories * n_available)) ** 0.5
        mde = {"absolute_effect": effect,
               "relative_to_baseline": effect / baseline if baseline else None,
               "formula": "z_total * sqrt(Var(d) / (categories * n_available))"}

    ## The entropy screen: answers as the instrument produced them.
    screen = {}
    for arm in arms:
        answers = {item["id"]: item["output"]["text"]
                   for item in selections[arm]["items"]}
        screen[arm] = {
            fmt: forensics.format_answer_screen(tasks, answers, fmt)
            for fmt in (formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC)}
        for fmt, entry in screen[arm].items():
            entry.pop("per_item", None)

    excluded = sorted({(arm, fmt) for arm in screen for fmt, entry in screen[arm].items()
                       if not entry["discriminating"]})
    cells = {}
    for fmt in (formats.FORMAT_PATCH, formats.FORMAT_DIAGNOSTIC):
        values = [per_arm[arm]["per_format"][fmt]["accuracy"] for arm in arms
                  if fmt in per_arm[arm]["per_format"]]
        cells[fmt] = {
            "per_arm_accuracy": {arm: per_arm[arm]["per_format"][fmt]["accuracy"]
                                 for arm in arms
                                 if fmt in per_arm[arm]["per_format"]},
            "accuracy_spread": (max(values) - min(values)) if len(values) >= 2 else None,
            "non_zero": any(value and value > 0 for value in values),
            "non_saturated": any(value is not None and value < 1.0 for value in values),
            "excluded_by_screen": any(fmt == excluded_fmt for _arm, excluded_fmt
                                      in excluded),
        }
    informative = [fmt for fmt, cell in cells.items()
                   if cell["non_zero"] and cell["non_saturated"]
                   and not cell["excluded_by_screen"]]

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "revision": revision(),
        "command": " ".join(sys.argv),
        "suite": {"root": os.path.abspath(args.suite),
                  "suite_sha256": suite_module.build_manifest(args.suite)["suite_sha256"]},
        "inputs": {
            "results": {arm: {"path": path, "sha256": sha256_file(path)}
                        for arm, path in sorted(result_paths.items())},
            "selections": {arm: {"path": path, "sha256": sha256_file(path),
                                 "stamp": selections[arm].get("stamp"),
                                 "revision": selections[arm].get("revision")}
                           for arm, path in sorted(selection_paths.items())},
        },
        "arms": arms,
        "instrument": {
            "name": "likelihood-selection",
            "primary_metric": formats.METRIC_EXECUTABLE,
            "diagnostic_metric": formats.METRIC_DIAGNOSTIC,
            "primary_mode": selections[arms[0]]["decoding"].get("primary_mode"),
            "pooled_with_primary": False,
        },
        "per_arm": per_arm,
        "item_outcomes": {
            "executable_items": len(item_ids),
            "passes": passed,
            "both_pass": len(both_pass),
            "neither_pass": len(neither),
            "one_arm_only": {arm: len(ids) for arm, ids in one_way.items()},
            "one_arm_only_ids": one_way,
            "discordant_pairs": discordant,
            "sign_test_p": exact_sign_test(
                discordant, len(one_way[arms[0]]) if len(arms) >= 2 else 0),
        },
        "power": {
            "baseline": baseline,
            "variance_of_paired_difference": variance,
            "categories": categories,
            "available_items_per_category": available,
            "n_available": n_available,
            "required_n": power,
            "minimum_detectable_effect": mde,
            "shortfall": (None if not power
                          else power["n_per_category_ceil"] > n_available),
        },
        "entropy_screen": {
            "min_answer_entropy_bits": forensics.MIN_ANSWER_ENTROPY_BITS,
            "max_modal_answer_share": forensics.MAX_MODAL_ANSWER_SHARE,
            "by_arm": screen,
            "excluded": [{"arm": arm, "format": fmt} for arm, fmt in excluded],
        },
        "cells": cells,
        "informative_formats": informative,
        "resources": resource_summary(selections),
        "statement": (
            "a pilot of the instrument, not a capability comparison: per-format "
            "accuracy from the executable checker, the entropy screen, and the "
            "measured power inputs. The five-arm comparison is not run here."),
    }
    payload["rendered"] = {
        **{f"arm_{arm}_patch": (
            f"{per_arm[arm]['per_format'][formats.FORMAT_PATCH]['executable_passes']} "
            f"of {per_arm[arm]['per_format'][formats.FORMAT_PATCH]['executable_items']}"
            f" patch items pass the executable checker "
            f"({per_arm[arm]['per_format'][formats.FORMAT_PATCH]['accuracy']:.4f}); "
            f"{per_arm[arm]['per_format'][formats.FORMAT_DIAGNOSTIC]['diagnostic_passes']}"
            f" of {per_arm[arm]['per_format'][formats.FORMAT_DIAGNOSTIC]['diagnostic_items']}"
            f" diagnostics verified "
            f"({per_arm[arm]['per_format'][formats.FORMAT_DIAGNOSTIC]['accuracy']:.4f})")
            for arm in arms},
        "item_outcomes": (
            f"{len(item_ids)} executable items: {len(both_pass)} passed by both arms, "
            f"{len(neither)} by neither, and {discordant} by exactly one"),
        "baseline": f"{baseline:.6f}",
        "variance": ("unavailable" if variance is None else f"{variance:.6f}"),
        "required_n": ("unavailable" if not power else
                       f"{power['n_per_category_ceil']:,} per category"),
        "n_available": str(n_available),
        "mde": ("unavailable" if not mde else
                f"{mde['absolute_effect']:.4f} absolute, "
                f"{mde['relative_to_baseline']:.2f}x baseline"),
        "screen": ("; ".join(
            f"{arm}/{fmt} entropy {screen[arm][fmt]['entropy_bits']:.3f} bits "
            f"({screen[arm][fmt]['screen']})"
            for arm in arms for fmt in screen[arm])),
        "informative_formats": ", ".join(informative) if informative else "none",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"capability_pilot_analysis: wrote {args.out}")
    for arm in arms:
        print(f"  {arm}: {payload['rendered'][f'arm_{arm}_patch']}")
    print(f"  {payload['rendered']['item_outcomes']}")
    print(f"  baseline {payload['rendered']['baseline']}; variance "
          f"{payload['rendered']['variance']}; required n "
          f"{payload['rendered']['required_n']}; MDE {payload['rendered']['mde']}")
    print(f"  informative formats: {payload['rendered']['informative_formats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
