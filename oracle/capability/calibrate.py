#!/usr/bin/env python3
"""Calibration statistics for the redesigned capability instrument (S28).

The v4 power analysis assumed a 0.5 baseline. The measured baseline was 0.0 on
every arm, so the assumption was not merely optimistic — it described an
instrument that did not exist. Calibration replaces the assumption with two
measured quantities per item: **how often it is solved** and **whether it
separates the arms**.

The registered window (see `docs/capability-calibration-preregistration-v1.json`):

* an item is *accepted* when at least one arm passes it and at least one arm
  fails it (never zero, never saturated) **and** it discriminates — either the
  partial-credit spread across arms is at least the registered minimum, or the
  item–total correlation is at least the registered minimum;
* an item that fails the window is retired with a named reason;
* a cell (category × format) keeps at most the registered fraction retired; a
  cell with fewer than the registered minimum accepted items is *reformatted*,
  not scored;
* a category is scorable only when every one of its executable cells is scored.

Nothing here decides a capability claim. It decides which items may carry one.
The primary metric stays executable success; the partial-credit fraction is
secondary, is recorded per item, and is never substituted for it.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import statistics

from . import formats
from . import power as power_module
from . import suite as suite_module

SCHEMA = "selvedge/capability-calibration/v1"
SCHEMA_VERSION = "selvedge/capability-suite-version/v1"

#: The registered window. `docs/capability-calibration-preregistration-v1.json`
#: carries the same numbers; a mismatch is refused, never silently used.
WINDOW = {
    "arms_expected": 5,
    "min_pass_rate": 0.10,
    "max_pass_rate": 0.90,
    "binary_equivalence": ("with A arms and binary outcomes, 0.10 <= pass rate <= "
                           "0.90 is the same statement as 1 <= passes <= A-1"),
    "min_partial_credit_spread": 0.15,
    "min_item_total_correlation": 0.30,
    "min_items_per_cell": 2,
    "max_cell_retirement_fraction": 1.0 / 3.0,
    "primary_metric": formats.METRIC_EXECUTABLE,
    "secondary_metric": "fraction_of_hidden_tests_passed",
    "secondary_statement": (
        "partial credit is secondary, is reported per item, and never enters the "
        "primary composite or a decision rule"),
    "power": {
        "target_power": 0.80,
        "level": 0.95,
        "relative_effect": 0.10,
        "variance_source": ("the pooled variance of per-item paired differences "
                            "over the calibrated arm pairs"),
        "planning_variance": "median of the pairwise variances",
        "conservative_variance": "maximum of the pairwise variances",
    },
}


class CalibrationError(RuntimeError):
    """The calibration inputs cannot support the registered decision."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def verify_registration(registration: dict) -> list:
    """Problems between the registration file and the constants applied here."""
    problems = []
    if registration.get("schema") != "selvedge/capability-calibration-preregistration/v1":
        problems.append("the calibration pre-registration has the wrong schema")
    window = registration.get("window") or {}
    for key, value in WINDOW.items():
        if key in ("binary_equivalence", "secondary_statement"):
            continue
        if window.get(key) != value:
            problems.append(f"registered window {key!r} is {window.get(key)!r}, "
                            f"the harness applies {value!r}")
    power = window.get("power") or {}
    for key, value in WINDOW["power"].items():
        if power.get(key) != value:
            problems.append(f"registered power field {key!r} is {power.get(key)!r}, "
                            f"the harness applies {value!r}")
    if registration.get("primary", {}).get("metric") != WINDOW["primary_metric"]:
        problems.append("the registration does not name executable success as primary")
    if registration.get("secondary", {}).get("metric") != WINDOW["secondary_metric"]:
        problems.append("the registration does not name the partial-credit metric")
    return problems


def _point_biserial(values: list, totals: list) -> float | None:
    """Pearson correlation across arms between an item and the arm totals."""
    if len(values) < 3 or len(values) != len(totals):
        return None
    mean_x = statistics.fmean(values)
    mean_y = statistics.fmean(totals)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, totals))
    variance_x = sum((x - mean_x) ** 2 for x in values)
    variance_y = sum((y - mean_y) ** 2 for y in totals)
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def per_arm_rows(results: dict) -> dict:
    """arm -> {item id: row}, validated to be the same suite."""
    hashes = {}
    rows = {}
    for arm, payload in results.items():
        if payload.get("schema") != "selvedge/capability-result/v1":
            raise CalibrationError(f"arm {arm}: not a capability result receipt")
        hashes[arm] = (payload.get("suite") or {}).get("suite_sha256")
        rows[arm] = {row["id"]: row for row in payload.get("items", [])}
    if len(set(hashes.values())) != 1:
        raise CalibrationError(f"the arms were scored on different suites: {hashes}")
    ids = {arm: set(items) for arm, items in rows.items()}
    if len({frozenset(value) for value in ids.values()}) != 1:
        raise CalibrationError("the arms were scored on different item sets")
    return rows


def item_records(rows: dict, ceiling: int | None = None) -> list:
    """One record per item: pass rate, spread, correlation, acceptance."""
    arms = sorted(rows)
    first = rows[arms[0]]
    totals = {arm: sum(row.get("value") or 0.0 for row in rows[arm].values())
              for arm in arms}
    records = []
    for item_id in sorted(first):
        template = first[item_id]
        metric = template.get("metric") or formats.METRIC_EXECUTABLE
        values = []
        partials = []
        statuses = {}
        for arm in arms:
            row = rows[arm][item_id]
            values.append(1.0 if row.get("status") == "pass" else 0.0)
            statuses[arm] = row.get("status")
            credit = row.get("partial_credit") or {}
            if credit.get("available"):
                partials.append(float(credit.get("fraction") or 0.0))
        spread = statistics.pstdev(partials) if len(partials) >= 2 else None
        without_item = {arm: totals[arm] - (1.0 if statuses[arm] == "pass" else 0.0)
                        for arm in arms}
        correlation = _point_biserial(
            values, [without_item[arm] for arm in arms])
        passes = int(sum(values))
        record = {
            "id": item_id,
            "category": template.get("category"),
            "format": template.get("format", formats.FORMAT_FILE),
            "tier": template.get("tier"),
            "metric": metric,
            "arms": statuses,
            "passes": passes,
            "arms_scored": len(arms),
            "pass_rate": passes / len(arms),
            "partial_credit_available": len(partials),
            "partial_credit_spread": spread,
            "item_total_correlation": correlation,
            "accepted": False,
            "reasons": [],
        }
        if metric == formats.METRIC_DIAGNOSTIC:
            record["reasons"].append(
                "diagnostic metric: reported separately, not part of the executable "
                "calibration window")
            records.append(record)
            continue
        reasons = []
        if passes == 0:
            reasons.append("zero pass rate: no arm solved the item")
        elif passes == len(arms):
            reasons.append("saturated: every arm solved the item")
        elif not (WINDOW["min_pass_rate"] <= record["pass_rate"]
                  <= WINDOW["max_pass_rate"]):
            reasons.append(
                f"pass rate {record['pass_rate']:.2f} is outside the registered window")
        discriminating = (
            (spread is not None and spread >= WINDOW["min_partial_credit_spread"])
            or (correlation is not None
                and correlation >= WINDOW["min_item_total_correlation"]))
        if not reasons and not discriminating:
            reasons.append(
                "does not discriminate: partial-credit spread and item-total "
                "correlation are both below the registered minima")
        record["accepted"] = not reasons
        record["reasons"] = reasons
        records.append(record)
    return records


def cell_table(records: list) -> list:
    """Category x format cells with the registered retire/reformat decision."""
    cells: dict[tuple, list] = {}
    for record in records:
        if record["metric"] != formats.METRIC_EXECUTABLE:
            continue
        cells.setdefault((record["category"], record["format"]), []).append(record)
    table = []
    for (category, fmt), items in sorted(cells.items()):
        accepted = [item for item in items if item["accepted"]]
        retired_fraction = ((len(items) - len(accepted)) / len(items))
        scored = (len(accepted) >= WINDOW["min_items_per_cell"]
                  and retired_fraction <= WINDOW["max_cell_retirement_fraction"]
                  + 1e-9)
        table.append({
            "category": category,
            "format": fmt,
            "tier": formats.FORMAT_TIERS.get(fmt),
            "items": len(items),
            "accepted": len(accepted),
            "retired": len(items) - len(accepted),
            "retired_fraction": retired_fraction,
            "mean_pass_rate": statistics.fmean([item["pass_rate"] for item in items]),
            "mean_accepted_pass_rate": (statistics.fmean(
                [item["pass_rate"] for item in accepted]) if accepted else None),
            "scored": scored,
            "decision": "scored" if scored else "reformatted",
        })
    return table


def ladder_check(cells: list) -> dict:
    """Is the easy/medium/hard ladder monotone in measured difficulty?"""
    order = (formats.FORMAT_BODY, formats.FORMAT_PATCH, formats.FORMAT_FILE)
    by_category: dict[str, dict[str, float]] = {}
    for cell in cells:
        if cell["format"] in order:
            by_category.setdefault(cell["category"], {})[cell["format"]] = (
                1.0 - cell["mean_pass_rate"])
    rows = []
    violations = []
    for category in sorted(by_category):
        difficulty = by_category[category]
        row = {"category": category, "difficulty": difficulty}
        if len(difficulty) == len(order):
            monotone = (difficulty[formats.FORMAT_BODY]
                        <= difficulty[formats.FORMAT_PATCH]
                        <= difficulty[formats.FORMAT_FILE])
            row["monotone"] = monotone
            if not monotone:
                violations.append(category)
        else:
            row["monotone"] = None
        rows.append(row)
    return {
        "expected_order": list(order),
        "measure": "1 - mean pass rate per cell",
        "rows": rows,
        "violations": violations,
        "monotone": not violations,
        "statement": ("the ladder is a hypothesis; a violation is reported and the "
                      "tier labels are not re-used as a difficulty claim"),
    }


def paired_variances(rows: dict) -> list:
    """Variance of per-item paired differences for every arm pair."""
    arms = sorted(rows)
    item_ids = sorted(rows[arms[0]])
    variances = []
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            differences = []
            for item_id in item_ids:
                if (rows[left][item_id].get("metric")
                        or formats.METRIC_EXECUTABLE) != formats.METRIC_EXECUTABLE:
                    continue
                differences.append((1.0 if rows[left][item_id].get("status") == "pass"
                                    else 0.0)
                                   - (1.0 if rows[right][item_id].get("status") == "pass"
                                      else 0.0))
            if len(differences) >= 2 and statistics.pvariance(differences) > 0:
                variances.append({"arm_a": left, "arm_b": right,
                                  "items": len(differences),
                                  "variance": statistics.pvariance(differences)})
    return variances


def required_n(baseline: float, relative_effect: float, variance: float,
               categories: int, power: float = 0.80,
               level: float = 0.95) -> dict:
    """Items per category for `power` at a relative effect, paired design.

    Var(composite) = Var(d) / (categories * n), so
    n = z_total^2 * Var(d) / (categories * (baseline * relative_effect)^2).
    """
    if baseline <= 0:
        raise CalibrationError("a zero baseline makes the relative effect undefined")
    absolute = baseline * relative_effect
    z_total = power_module.z_value(1.0 - (1.0 - level) / 2.0) + power_module.z_value(power)
    n = (z_total ** 2) * variance / (categories * absolute ** 2)
    return {
        "baseline": baseline,
        "relative_effect": relative_effect,
        "absolute_effect": absolute,
        "variance_of_paired_difference": variance,
        "categories": categories,
        "power": power,
        "level": level,
        "z_total": z_total,
        "n_per_category": n,
        "n_per_category_ceil": int(math.ceil(n)),
        "formula": ("n = (z_{1-alpha/2} + z_power)^2 * Var(d) / "
                    "(categories * (baseline * relative_effect)^2)"),
        "approximation": "normal approximation to the paired composite mean",
    }


def calibrate(suite_root: str, results: dict, registration: dict,
              calibration_receipt: str) -> dict:
    problems = verify_registration(registration)
    if problems:
        raise CalibrationError("; ".join(problems))
    rows = per_arm_rows(results)
    if len(rows) != WINDOW["arms_expected"]:
        raise CalibrationError(
            f"calibration is registered on {WINDOW['arms_expected']} arms; got "
            f"{len(rows)} ({sorted(rows)})")
    records = item_records(rows)
    cells = cell_table(records)
    ladder = ladder_check(cells)
    variances = paired_variances(rows)
    executable = [record for record in records
                  if record["metric"] == formats.METRIC_EXECUTABLE]
    baseline = (statistics.fmean([record["pass_rate"] for record in executable])
                if executable else 0.0)
    categories = len({record["category"] for record in executable if record["accepted"]})
    power_block = {
        "available": bool(variances) and baseline > 0,
        "measured_baseline_pass_rate": baseline,
        "pairwise_variances": variances,
        "required_n": None,
        "statement": (
            "power is computed from the measured baseline and the measured paired "
            "variance; until this calibration pass exists there is no number to "
            "report, and an assumed 0.5 baseline is not used"),
    }
    if variances and baseline > 0:
        measured = sorted(entry["variance"] for entry in variances)
        planning = measured[len(measured) // 2]
        conservative = measured[-1]
        power_block["planning_variance"] = planning
        power_block["conservative_variance"] = conservative
        power_block["required_n"] = {
            "planning": required_n(baseline, WINDOW["power"]["relative_effect"],
                                   planning, max(1, categories)),
            "conservative": required_n(baseline, WINDOW["power"]["relative_effect"],
                                       conservative, max(1, categories)),
        }
        per_category_items = {}
        for record in executable:
            if record["accepted"]:
                per_category_items[record["category"]] = (
                    per_category_items.get(record["category"], 0) + 1)
        power_block["items_per_category_available"] = per_category_items
        required = power_block["required_n"]["conservative"]["n_per_category_ceil"]
        power_block["power_met"] = bool(per_category_items) and all(
            count >= required for count in per_category_items.values())
        power_block["shortfall"] = {
            category: required - count for category, count in per_category_items.items()
            if count < required}
    else:
        power_block["power_met"] = None
        power_block["shortfall"] = None
        if baseline <= 0:
            power_block["reason"] = "the measured baseline is zero on every arm"
        elif not variances:
            power_block["reason"] = (
                "no arm pair has a non-zero paired-difference variance, so the "
                "effect size the instrument can resolve is not estimable")

    scored_categories = sorted({cell["category"] for cell in cells if cell["scored"]})
    reformatted = sorted({cell["category"] for cell in cells if not cell["scored"]})
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "suite": {
            "root": os.path.abspath(suite_root),
            "suite_sha256": suite_module.build_manifest(suite_root)["suite_sha256"],
        },
        "registration": {
            "path": registration.get("path"),
            "sha256": registration.get("sha256"),
            "window": WINDOW,
        },
        "arms": sorted(rows),
        "window": WINDOW,
        "items": records,
        "cells": cells,
        "ladder": ladder,
        "power": power_block,
        "summary": {
            "items": len(records),
            "executable_items": len(executable),
            "diagnostic_items": len(records) - len(executable),
            "accepted": sum(1 for record in executable if record["accepted"]),
            "retired": sum(1 for record in executable if not record["accepted"]),
            "scored_categories": scored_categories,
            "reformatted_categories": reformatted,
            "categories_scorable": (not reformatted) and bool(scored_categories),
        },
        "calibration_receipt": calibration_receipt,
    }


def check_suite_version(version_path: str, calibration: dict) -> tuple[bool, str]:
    """The emitted version is exactly the decision the receipt records."""
    if not os.path.isfile(version_path):
        return False, f"{version_path} is missing"
    recorded = load_json(version_path)
    expected = build_suite_version(calibration, recorded.get("revision") or "")
    for key in ("items", "counts", "cells", "ladder", "power", "window",
                "metrics"):
        if recorded.get(key) != expected[key]:
            return False, f"{version_path} is stale at {key!r}"
    if recorded.get("calibration_suite_sha256") != expected["calibration_suite_sha256"]:
        return False, f"{version_path} binds a different calibration suite"
    return True, f"{version_path} matches the calibration receipt"


def recheck_calibration(path: str) -> tuple[bool, str]:
    """Recompute the cells, ladder and summary from the stored item records."""
    if not os.path.isfile(path):
        return False, f"{path} is missing"
    payload = load_json(path)
    if payload.get("schema") != SCHEMA:
        return False, f"{path}: wrong schema"
    records = payload.get("items") or []
    cells = cell_table(records)
    if cells != payload.get("cells"):
        return False, f"{path}: the cell table does not match the item records"
    ladder = ladder_check(cells)
    if ladder != payload.get("ladder"):
        return False, f"{path}: the ladder does not match the cell table"
    executable = [record for record in records
                  if record["metric"] == formats.METRIC_EXECUTABLE]
    summary = payload.get("summary") or {}
    if summary.get("accepted") != sum(1 for record in executable
                                      if record["accepted"]):
        return False, f"{path}: the accepted count is stale"
    if summary.get("retired") != sum(1 for record in executable
                                     if not record["accepted"]):
        return False, f"{path}: the retired count is stale"
    return True, f"{path}: cells, ladder and counts recompute"


def build_suite_version(calibration: dict, revision: str) -> dict:
    """The calibrated suite version: which items may carry a claim."""
    summary = calibration["summary"]
    if not summary["categories_scorable"]:
        raise CalibrationError(
            "cannot emit a calibrated suite version: categories "
            f"{summary['reformatted_categories']} have a reformatted cell; redesign "
            "that format before scoring")
    if calibration["ladder"]["violations"]:
        raise CalibrationError(
            "cannot emit a calibrated suite version: the measured ladder is not "
            f"monotone for {calibration['ladder']['violations']}; re-tier the "
            "formats or report the ladder as unordered")
    tiers = {}
    for cell in calibration["cells"]:
        tiers.setdefault(cell["category"], set()).add(cell["format"])
    incomplete = {
        category: sorted(set(formats.EXECUTABLE_FORMATS) - present)
        for category, present in tiers.items()
        if set(formats.EXECUTABLE_FORMATS) - present
    }
    if incomplete or not tiers:
        raise CalibrationError(
            "cannot emit a calibrated suite version: the executable ladder is "
            f"incomplete for {sorted(incomplete) or ['every category']}")
    accepted = [record for record in calibration["items"]
                if record["metric"] == formats.METRIC_EXECUTABLE and record["accepted"]]
    retired = [record for record in calibration["items"]
               if record["metric"] == formats.METRIC_EXECUTABLE and not record["accepted"]]
    diagnostic = [record for record in calibration["items"]
                  if record["metric"] == formats.METRIC_DIAGNOSTIC]
    return {
        "schema": SCHEMA_VERSION,
        "version": 1,
        "created_utc": utc_now(),
        "revision": revision,
        "calibration_suite_sha256": calibration["suite"]["suite_sha256"],
        "calibration_receipt": calibration["calibration_receipt"],
        "window": calibration["window"],
        "items": {
            "scored": sorted(record["id"] for record in accepted),
            "retired": sorted(record["id"] for record in retired),
            "diagnostic": sorted(record["id"] for record in diagnostic),
        },
        "counts": {
            "scored": len(accepted),
            "retired": len(retired),
            "diagnostic": len(diagnostic),
        },
        "cells": calibration["cells"],
        "ladder": calibration["ladder"],
        "power": calibration["power"],
        "metrics": {
            "primary": formats.METRIC_EXECUTABLE,
            "secondary": WINDOW["secondary_metric"],
            "diagnostic": formats.METRIC_DIAGNOSTIC,
            "pooling": ("the diagnostic metric is never pooled with executable "
                        "success and partial credit is never a decision input"),
        },
        "statement": (
            "This version lists the calibration items that survived the registered "
            "window. It authorises the scored split derived from the remaining "
            "source tasks; it is not a capability result."),
    }
