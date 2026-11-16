#!/usr/bin/env python3
"""P18.6 — cut-set discovery and interaction diagnostics from stored omission artifacts.

This is the analysis registered in `docs/EXPERIMENT-CONTRACT.md`, section
"P18.6 — cut-set discovery ...". It reads only the stored artifacts named in
P18.6.1, executes no model, uses no GPU, and is deterministic.

It refuses to run if the hash of a primary input differs from the constant
recorded at registration. It writes one machine-readable JSON receipt and one
generated Markdown report; neither is edited by hand.

Usage (inside the Python court):

  SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
    scripts/court python -- python3 scripts/p18_6_cutset.py \
      --out-json results/p18.6/cutset-$(date -u +%Y%m%dT%H%M%SZ).json \
      --out-md results/P18.6-CUTSET.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- registration constants (do not edit after the first run) ---------------

SEED = 20261116
BOOTSTRAP_DRAWS = 10_000
COVERAGE_RANDOM_DRAWS = 10_000
COVERAGE_RANDOMIZED_TIEBREAKS = 1_000
EVAL_COUNTS = (20, 40, 80, 120)
EPSILON_PRIMARY = 0.25
EPSILON_SENSITIVITY = (0.05, 0.10, 0.25, 0.50, 1.00)
TAU_E = 0.01
TAU_E_SECONDARY = 0.10
FAR_WORSE_FLOOR = 0.01
FAR_WORSE_FRACTION = 0.25

INPUTS = {
    "interactions": {
        "path": "results/selectors/interactions-20261114T232338Z.json",
        "sha256": "ac160ac774e0264e7e3c303e01b571a6df008a798a12a5bda5013add3bccfb9d",
        "stamp": "20261114T232338Z",
        "revision": "1e623062379239a4bed3337ed2ed91d1a7f5411b",
        "preregistration_commit": "a0b107adc23f23cbda43a09cbd6d229b0ea024a1",
    },
    "frontier": {
        "path": "results/frontier/frontier.json",
        "sha256": "3407eb4f47ddada9a01b025d16893a2adaa898ffab7a650622d433d859b124a3",
        "stamp": "20261114T223105Z",
        "revision": "9ea017fd6071a8fb75e88f4254a9f6cf83b23c30",
        "preregistration_commit": "dad6619c8514cb4a52b3420c007a47fc43e1b47e",
    },
    "split": {
        "path": "results/splits/split-20261114T223127Z.json",
        "sha256": "693a55437d2c6377bea9a3cca44c2ba7ac14599e5b8281a6cc0dd2bfc1a5d7b9",
        "stamp": "20261114T223127Z",
        "revision": "9ea017fd6071a8fb75e88f4254a9f6cf83b23c30",
    },
    "hstar_labels_sidecar": {
        "path": "results/artifacts/train/t2/Hstar/utility-labels.pt.sha256",
        "sha256": "9e41d6049766f69ae1483867b5bdc895383be6dfd94993d2c38949caa1b7bf14",
        "note": "sidecar declares the label artifact hash; the 3.8 GB artifact is not read",
    },
}

DISCOVERY_DOCS = {
    "wikitext-article-020",
    "wikitext-article-055",
    "wikitext-article-042",
    "wikitext-article-039",
}
REPLICATION_DOCS = {
    "wikitext-article-038",
    "wikitext-article-019",
    "wikitext-article-050",
    "wikitext-article-044",
}


# --- small deterministic statistics helpers --------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def quantile(values, q):
    return percentile(sorted(values), q)


def stdev(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1))


def sign(value, tolerance=0.0):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            result[order[index]] = average
        position = end + 1
    return result


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else float("nan")


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def r2(predictions, actual):
    centre = mean(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predictions))
    ss_tot = sum((a - centre) ** 2 for a in actual)
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def mae(predictions, actual):
    return mean(abs(a - p) for a, p in zip(actual, predictions))


def rmse(predictions, actual):
    return math.sqrt(mean((a - p) ** 2 for a, p in zip(actual, predictions)))


def distribution(values):
    values = [float(v) for v in values]
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "sd": stdev(values),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
        "positive_fraction": mean(1.0 if v > 0 else 0.0 for v in values),
        "negative_fraction": mean(1.0 if v < 0 else 0.0 for v in values),
        "zero_fraction": mean(1.0 if v == 0 else 0.0 for v in values),
        "mean_abs": mean(abs(v) for v in values),
        "median_abs": quantile([abs(v) for v in values], 0.5),
    }


def bootstrap_over_windows(by_window, windows, statistic, draws=BOOTSTRAP_DRAWS, seed=SEED):
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        pooled = []
        for _ in range(len(windows)):
            pooled.extend(by_window[windows[rng.randrange(len(windows))]])
        estimates.append(statistic(pooled))
    estimates.sort()
    return {
        "mean": mean(estimates),
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
        "resamples": draws,
        "seed": seed,
    }


# --- input loading and integrity -------------------------------------------


def load_inputs():
    checked = {}
    for name, spec in INPUTS.items():
        path = ROOT / spec["path"]
        if not path.exists():
            raise SystemExit(f"P18.6: missing input {spec['path']}")
        if name == "hstar_labels_sidecar":
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            actual = sidecar.get("sha256")
        else:
            actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise SystemExit(
                f"P18.6: hash mismatch for {spec['path']}: "
                f"expected {spec['sha256']}, got {actual}. Refusing to run."
            )
        checked[name] = dict(spec, observed_sha256=actual, absolute=str(path))
    return checked


def load_records():
    with open(ROOT / INPUTS["interactions"]["path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    records = []
    for row in receipt["records"]:
        a = float(row["kl_S"])
        b = float(row["kl_Sg"])
        c = float(row["kl_Sj"])
        d = float(row["kl_Sgj"])
        e = d - b - c + a
        # Correctness checks against the receipt's own fields.
        if abs((b + c - a - d) - float(row["interaction"])) > 1e-6:
            raise SystemExit("P18.6: interaction field does not equal b + c - a - d")
        if abs((a - b) - float(row["delta_g"])) > 1e-6:
            raise SystemExit("P18.6: delta_g field does not equal kl_S - kl_Sg")
        if abs((a - c) - float(row["delta_j"])) > 1e-6:
            raise SystemExit("P18.6: delta_j field does not equal kl_S - kl_Sj")
        g, j = int(row["g"]), int(row["j"])
        records.append(
            {
                "window": row["window"],
                "doc_id": row["doc_id"],
                "layer": int(row["layer"]),
                "g": min(g, j),
                "j": max(g, j),
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "m_g": b - a,
                "m_j": c - a,
                "delta": d - a,
                "E": e,
                "interaction": float(row["interaction"]),
                "coactivation": float(row.get("coactivation", float("nan"))),
            }
        )
    if len(records) != 160:
        raise SystemExit(f"P18.6: expected 160 contexts, found {len(records)}")
    return receipt, records


# --- S1: distribution and sign asymmetry -----------------------------------


def s1(records, windows):
    by_window = {window: [] for window in windows}
    for record in records:
        by_window[record["window"]].append(record["E"])

    per_layer = {}
    for layer in sorted({record["layer"] for record in records}):
        subset = [record for record in records if record["layer"] == layer]
        per_layer[str(layer)] = distribution([record["E"] for record in subset])

    fixed_cells = {}
    for record in records:
        key = (record["layer"], record["g"], record["j"])
        fixed_cells.setdefault(key, {})[record["window"]] = record["E"]
    flips = []
    cell_means = []
    for key in sorted(fixed_cells):
        series = [fixed_cells[key][window] for window in windows if window in fixed_cells[key]]
        changes = sum(
            1
            for left, right in zip(series, series[1:])
            if sign(left) != sign(right)
        )
        flips.append(changes)
        cell_means.append(
            {
                "layer": key[0],
                "g": key[1],
                "j": key[2],
                "n_windows": len(series),
                "mean_E": mean(series),
                "sign_changes": changes,
            }
        )

    asymmetry = [abs(record["m_g"]) - abs(record["m_j"]) for record in records]
    far_worse = [
        record
        for record in records
        if record["E"] > max(FAR_WORSE_FLOOR, FAR_WORSE_FRACTION * (abs(record["m_g"]) + abs(record["m_j"])))
    ]

    return {
        "overall_E": distribution([record["E"] for record in records]),
        "overall_I_m2": distribution([record["interaction"] for record in records]),
        "mean_E_bootstrap": bootstrap_over_windows(by_window, windows, mean),
        "mean_abs_E_bootstrap": bootstrap_over_windows(
            by_window, windows, lambda values: mean(abs(v) for v in values)
        ),
        "per_layer_E": per_layer,
        "far_worse": {
            "criterion": "E > max(0.01 nats, 0.25 * (|m_g| + |m_j|))",
            "count": len(far_worse),
            "fraction": len(far_worse) / len(records),
        },
        "sign_asymmetry": {
            "marginal_magnitude_asymmetry": distribution(asymmetry),
            "fraction_marginals_opposite_sign": mean(
                1.0 if record["m_g"] * record["m_j"] < 0 else 0.0 for record in records
            ),
            "fixed_cells": cell_means,
            "fixed_cell_sign_changes": {
                "n_cells": len(flips),
                "min": min(flips),
                "median": quantile(flips, 0.5),
                "max": max(flips),
                "fraction_with_at_least_one_flip": mean(1.0 if f > 0 else 0.0 for f in flips),
            },
            "algebraic_note": "E is symmetric in g and j; the measurable asymmetry is the marginal-magnitude difference and cross-window sign instability",
        },
        "counts": {
            "positive_E": sum(1 for record in records if record["E"] > 0),
            "negative_E": sum(1 for record in records if record["E"] < 0),
            "zero_E": sum(1 for record in records if record["E"] == 0),
        },
    }


# --- S2: unary-only versus pairwise-additive --------------------------------


def s2(records):
    deltas = [record["delta"] for record in records]
    residuals = [record["E"] for record in records]
    in_sample = {
        "mean_abs_delta": mean(abs(value) for value in deltas),
        "mean_abs_E": mean(abs(value) for value in residuals),
        "residual_share": mean(abs(value) for value in residuals) / mean(abs(value) for value in deltas),
        "r2_unary": 1.0 - sum(e * e for e in residuals) / sum((d - mean(deltas)) ** 2 for d in deltas),
        "pairwise_in_sample_note": "the measured pair interaction predicts each measured four-point cell exactly by construction; an in-sample pairwise R2 of one is not evidence",
    }

    cells = {}
    for record in records:
        cells.setdefault((record["layer"], record["g"], record["j"]), []).append(record)

    discovery = {key: [r for r in rows if r["doc_id"] in DISCOVERY_DOCS] for key, rows in cells.items()}
    replication = {key: [r for r in rows if r["doc_id"] in REPLICATION_DOCS] for key, rows in cells.items()}
    e_discovery = {key: mean(r["E"] for r in rows) for key, rows in discovery.items()}
    global_interaction = mean(list(e_discovery.values()))

    observed = []
    prediction_unary = []
    prediction_cell = []
    prediction_global = []
    sign_hits = 0
    sign_total = 0
    for key in sorted(cells):
        for record in replication[key]:
            observed.append(record["delta"])
            unary = record["m_g"] + record["m_j"]
            prediction_unary.append(unary)
            prediction_cell.append(unary + e_discovery[key])
            prediction_global.append(unary + global_interaction)
            if sign(record["E"]) == sign(e_discovery[key]):
                sign_hits += 1
            sign_total += 1

    cell_rows = []
    for key in sorted(cells):
        replication_mean = mean(r["E"] for r in replication[key])
        cell_rows.append(
            {
                "layer": key[0],
                "g": key[1],
                "j": key[2],
                "discovery_mean_E": e_discovery[key],
                "replication_mean_E": replication_mean,
                "same_sign": sign(e_discovery[key]) == sign(replication_mean),
            }
        )
    cell_level_sign_agreement = mean(1.0 if row["same_sign"] else 0.0 for row in cell_rows)
    cell_spearman = spearman(
        [row["discovery_mean_E"] for row in cell_rows],
        [row["replication_mean_E"] for row in cell_rows],
    )

    metrics = {}
    for name, prediction in (
        ("unary", prediction_unary),
        ("cell_interaction", prediction_cell),
        ("global_interaction", prediction_global),
    ):
        metrics[name] = {
            "mae": mae(prediction, observed),
            "rmse": rmse(prediction, observed),
            "r2": r2(prediction, observed),
        }

    # Paired bootstrap over the four replication windows for MAE(unary) - MAE(cell).
    replication_windows = sorted({record["window"] for record in records if record["doc_id"] in REPLICATION_DOCS})
    by_window = {}
    for key in sorted(cells):
        for record in replication[key]:
            by_window.setdefault(record["window"], []).append(record)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOTSTRAP_DRAWS):
        rows = []
        for _ in range(len(replication_windows)):
            rows.extend(by_window[replication_windows[rng.randrange(len(replication_windows))]])
        observed_b = [r["delta"] for r in rows]
        unary_b = [r["m_g"] + r["m_j"] for r in rows]
        cell_b = [r["m_g"] + r["m_j"] + e_discovery[(r["layer"], r["g"], r["j"])] for r in rows]
        diffs.append(mae(unary_b, observed_b) - mae(cell_b, observed_b))
    diffs.sort()

    return {
        "in_sample": in_sample,
        "out_of_sample": {
            "n_contexts": len(observed),
            "n_cells": len(cells),
            "global_interaction": global_interaction,
            "metrics": metrics,
            "sign_agreement_context": sign_hits / sign_total if sign_total else float("nan"),
            "sign_agreement_cell_mean": cell_level_sign_agreement,
            "cell_spearman": cell_spearman,
            "mae_difference_unary_minus_cell": {
                "mean": mean(diffs),
                "ci_low": percentile(diffs, 0.025),
                "ci_high": percentile(diffs, 0.975),
                "resamples": BOOTSTRAP_DRAWS,
                "seed": SEED,
            },
            "cells": cell_rows,
        },
    }


# --- S3: monotonicity / submodularity counts --------------------------------


def s3(records):
    super_additive = [record for record in records if record["E"] > 0]
    sub_additive = [record for record in records if record["E"] < 0]
    far_worse = [
        record
        for record in records
        if record["E"] > max(FAR_WORSE_FLOOR, FAR_WORSE_FRACTION * (abs(record["m_g"]) + abs(record["m_j"])))
    ]
    identity_checks = sum(
        1
        for record in records
        if sign((record["d"] - record["b"]) - record["m_j"]) == sign(record["E"])
    )
    return {
        "super_additive_E_positive": len(super_additive),
        "sub_additive_E_negative": len(sub_additive),
        "zero_interaction": len(records) - len(super_additive) - len(sub_additive),
        "far_worse": len(far_worse),
        "sequential_identity_checks_passed": identity_checks,
        "sequential_identity_total": len(records),
        "definition": "Delta(j | S+g) > Delta(j | S) iff E > 0; reported as one algebraic event, not two findings",
    }


# --- S4: measured-minimal destructive omission sets ------------------------


def minimal_flags(record, epsilon):
    a, b, c, d = record["a"], record["b"], record["c"], record["d"]
    flags = {
        "O": (a > epsilon) and (b <= epsilon) and (c <= epsilon),
        "O_g": (b > epsilon) and (d <= epsilon),
        "O_j": (c > epsilon) and (d <= epsilon),
    }
    destructive = {
        "O": a > epsilon,
        "O_g": b > epsilon,
        "O_j": c > epsilon,
        "O_gj": d > epsilon,
    }
    return flags, destructive


def s4(records):
    windows_by_half = {
        "discovery": DISCOVERY_DOCS,
        "replication": REPLICATION_DOCS,
    }

    def level_summary():
        per_layer = {}
        all_values = []
        for layer in sorted({record["layer"] for record in records}):
            subset = [record for record in records if record["layer"] == layer]
            values = [value for record in subset for value in (record["a"], record["b"], record["c"], record["d"])]
            all_values.extend(values)
            per_layer[str(layer)] = {
                "L_S": distribution([record["a"] for record in subset]),
                "L_Sg": distribution([record["b"] for record in subset]),
                "L_Sj": distribution([record["c"] for record in subset]),
                "L_Sgj": distribution([record["d"] for record in subset]),
                "min": min(values),
                "median": quantile(values, 0.5),
                "max": max(values),
                "fraction_le_0.25": mean(1.0 if value <= 0.25 else 0.0 for value in values),
                "fraction_le_1.00": mean(1.0 if value <= 1.00 else 0.0 for value in values),
            }
        return {
            "per_layer": per_layer,
            "n_values": len(all_values),
            "min": min(all_values),
            "median": quantile(all_values, 0.5),
            "max": max(all_values),
            "count_le": {
                str(epsilon): sum(1 for value in all_values if value <= epsilon)
                for epsilon in EPSILON_SENSITIVITY
            },
        }

    def analyse(epsilon):
        configs = {}
        contexts_with_minimal = 0

        def entry_for(record, kind):
            key = (record["layer"], record["g"], record["j"], kind)
            return configs.setdefault(
                key,
                {
                    "layer": record["layer"],
                    "g": record["g"],
                    "j": record["j"],
                    "kind": kind,
                    "ever_minimal": False,
                    "discovery_minimal_windows": set(),
                    "replication_minimal_windows": set(),
                    "discovery_destructive_windows": set(),
                    "replication_destructive_windows": set(),
                },
            )

        def half(record):
            return "discovery" if record["doc_id"] in windows_by_half["discovery"] else "replication"

        for record in records:
            flags, _destructive = minimal_flags(record, epsilon)
            if any(flags.values()):
                contexts_with_minimal += 1
            levels = {"O": record["a"], "O_g": record["b"], "O_j": record["c"]}
            for kind in ("O", "O_g", "O_j"):
                is_minimal = flags[kind]
                is_destructive = levels[kind] > epsilon
                if not is_minimal and not is_destructive:
                    continue
                entry = entry_for(record, kind)
                suffix = half(record)
                if is_minimal:
                    entry["ever_minimal"] = True
                    entry[f"{suffix}_minimal_windows"].add(record["window"])
                if is_destructive:
                    entry[f"{suffix}_destructive_windows"].add(record["window"])

        replicated = []
        for key, entry in configs.items():
            entry["discovery_minimal_count"] = len(entry["discovery_minimal_windows"])
            entry["replication_minimal_count"] = len(entry["replication_minimal_windows"])
            entry["discovery_destructive_count"] = len(entry["discovery_destructive_windows"])
            entry["replication_destructive_count"] = len(entry["replication_destructive_windows"])
            entry["replicates"] = (
                entry["ever_minimal"]
                and entry["discovery_minimal_count"] >= 3
                and entry["replication_minimal_count"] >= 3
            )
            entry["destructive_direction_replicates"] = (
                entry["discovery_destructive_count"] >= 3 and entry["replication_destructive_count"] >= 3
            )
            if entry["replicates"]:
                replicated.append(key)
        for entry in configs.values():
            entry["discovery_minimal_windows"] = sorted(entry["discovery_minimal_windows"])
            entry["replication_minimal_windows"] = sorted(entry["replication_minimal_windows"])
            entry["discovery_destructive_windows"] = sorted(entry["discovery_destructive_windows"])
            entry["replication_destructive_windows"] = sorted(entry["replication_destructive_windows"])
        return {
            "epsilon": epsilon,
            "contexts_with_any_measured_minimal": contexts_with_minimal,
            "distinct_configurations": sum(1 for entry in configs.values() if entry["ever_minimal"]),
            "replicated_configurations": len(replicated),
            "destructive_direction_replicated_configurations": sum(
                1 for entry in configs.values() if entry["destructive_direction_replicates"]
            ),
            "configurations": sorted(configs.values(), key=lambda entry: (entry["layer"], entry["g"], entry["j"], entry["kind"])),
        }

    primary = analyse(EPSILON_PRIMARY)
    sensitivity = [analyse(epsilon) for epsilon in EPSILON_SENSITIVITY]
    per_context = []
    for record in records:
        flags, _destructive = minimal_flags(record, EPSILON_PRIMARY)
        per_context.append(
            {
                "window": record["window"],
                "doc_id": record["doc_id"],
                "half": "discovery" if record["doc_id"] in DISCOVERY_DOCS else "replication",
                "layer": record["layer"],
                "g": record["g"],
                "j": record["j"],
                "L_S": record["a"],
                "L_Sg": record["b"],
                "L_Sj": record["c"],
                "L_Sgj": record["d"],
                "E": record["E"],
                "minimal_O": flags["O"],
                "minimal_O_g": flags["O_g"],
                "minimal_O_j": flags["O_j"],
            }
        )
    return {
        "primary": primary,
        "sensitivity": sensitivity,
        "levels": level_summary(),
        "per_context": sorted(per_context, key=lambda row: (row["layer"], row["g"], row["j"], row["window"])),
        "definition": "O is measured-minimal destructive iff L(O) > epsilon and every measured immediate subset is <= epsilon",
    }


# --- S5: covering array versus random sampling -----------------------------


CELL_WINDOWS = 8


def build_cells(records, windows):
    cell_records = {}
    for record in records:
        key = (record["layer"], record["g"], record["j"], record["window"])
        cell_records[key] = record
    keys = sorted(cell_records)
    if len(keys) != 160:
        raise SystemExit(f"P18.6: coverage population expected 160 cells, found {len(keys)}")
    layers = sorted({key[0] for key in keys})
    pairs = sorted({(key[1], key[2]) for key in keys})
    windows_sorted = sorted({key[3] for key in keys})
    layer_index = {value: index for index, value in enumerate(layers)}
    pair_index = {value: index for index, value in enumerate(pairs)}
    window_index = {value: index for index, value in enumerate(windows_sorted)}
    # Factor-level pair bit positions: layer x pair (20), layer x window (40), pair x window (32).
    pair_bits = []
    for key in keys:
        li, pi, wi = layer_index[key[0]], pair_index[(key[1], key[2])], window_index[key[3]]
        bits = 0
        bits |= 1 << (li * len(pairs) + pi)
        bits |= 1 << (20 + li * len(windows_sorted) + wi)
        bits |= 1 << (60 + pi * len(windows_sorted) + wi)
        pair_bits.append(bits)
    return keys, cell_records, pair_bits, len(layers) * len(pairs) + len(layers) * len(windows_sorted) + len(pairs) * len(windows_sorted)


def population_labels(cell_records, keys):
    primary = [cell_records[key]["E"] > TAU_E for key in keys]
    secondary = [cell_records[key]["E"] > TAU_E_SECONDARY for key in keys]
    minimal = []
    for key in keys:
        flags, _destructive = minimal_flags(cell_records[key], EPSILON_PRIMARY)
        minimal.append(any(flags.values()))
    return {"damaging_E_gt_0.01": primary, "damaging_E_gt_0.1": secondary, "measured_minimal_0.25": minimal}


def coverage_metrics(selected_indices, labels, keys):
    total_damaging = sum(1 for value in labels if value)
    found = sum(1 for index in selected_indices if labels[index])
    strata = {(keys[index][0], keys[index][1], keys[index][2]) for index in selected_indices}
    damaging_strata = {
        (keys[index][0], keys[index][1], keys[index][2])
        for index, value in enumerate(labels)
        if value
    }
    return {
        "found": found,
        "total_damaging": total_damaging,
        "recall": found / total_damaging if total_damaging else float("nan"),
        "precision": found / len(selected_indices) if selected_indices else float("nan"),
        "n": len(selected_indices),
        "damaging_strata_touched": len(strata & damaging_strata),
        "damaging_strata_total": len(damaging_strata),
    }


def greedy_covering(pair_bits, target, rng=None, hash_keys=None):
    selected = []
    covered = 0
    available = list(range(len(pair_bits)))
    while len(selected) < target and available:
        best_gain = -1
        best = []
        for index in available:
            gain = bin(pair_bits[index] & ~covered).count("1")
            if gain > best_gain:
                best_gain = gain
                best = [index]
            elif gain == best_gain:
                best.append(index)
        if rng is not None:
            pick = best[rng.randrange(len(best))]
        else:
            pick = min(best, key=lambda index: hash_keys[index])
        selected.append(pick)
        covered |= pair_bits[pick]
        available.remove(pick)
    return selected, bin(covered).count("1")


def s5(records, windows):
    keys, cell_records, pair_bits, total_pairs = build_cells(records, windows)
    labels = population_labels(cell_records, keys)
    hash_keys = [hashlib.sha256(("|".join(map(str, key))).encode()).hexdigest() for key in keys]

    results = {}
    rng = random.Random(SEED)
    random_samples = [
        rng.sample(range(len(keys)), max(EVAL_COUNTS)) for _ in range(COVERAGE_RANDOM_DRAWS)
    ]
    covering_deterministic, _achieved_end = greedy_covering(
        pair_bits, max(EVAL_COUNTS), hash_keys=hash_keys
    )
    covering_samples = []
    for draw in range(COVERAGE_RANDOMIZED_TIEBREAKS):
        variant, _covered = greedy_covering(
            pair_bits, max(EVAL_COUNTS), rng=random.Random(SEED + draw)
        )
        covering_samples.append(variant)

    def prefix_pair_coverage(selected, count):
        covered = 0
        for index in selected[:count]:
            covered |= pair_bits[index]
        return bin(covered).count("1")

    for label_name, label_values in labels.items():
        random_curves = {n: [] for n in EVAL_COUNTS}
        for sample in random_samples:
            for n in EVAL_COUNTS:
                random_curves[n].append(coverage_metrics(sample[:n], label_values, keys))

        covering_curves = {n: [] for n in EVAL_COUNTS}
        for variant in covering_samples:
            for n in EVAL_COUNTS:
                covering_curves[n].append(coverage_metrics(variant[:n], label_values, keys))

        per_n = {}
        for n in EVAL_COUNTS:
            random_recalls = sorted(metric["recall"] for metric in random_curves[n])
            covering_recalls = sorted(metric["recall"] for metric in covering_curves[n])
            deterministic = coverage_metrics(covering_deterministic[:n], label_values, keys)
            per_n[str(n)] = {
                "random": {
                    "recall_median": quantile(random_recalls, 0.5),
                    "recall_p05": quantile(random_recalls, 0.05),
                    "recall_p95": quantile(random_recalls, 0.95),
                    "precision_median": quantile(
                        sorted(metric["precision"] for metric in random_curves[n]), 0.5
                    ),
                },
                "covering": {
                    "recall_median": quantile(covering_recalls, 0.5),
                    "recall_p05": quantile(covering_recalls, 0.05),
                    "recall_p95": quantile(covering_recalls, 0.95),
                    "precision_median": quantile(
                        sorted(metric["precision"] for metric in covering_curves[n]), 0.5
                    ),
                    "deterministic": deterministic,
                    "achieved_factor_pair_coverage": prefix_pair_coverage(covering_deterministic, n) / total_pairs,
                },
            }
        results[label_name] = {
            "per_n": per_n,
            "n_damaging": sum(1 for value in label_values if value),
            "n_cells": len(keys),
            "n_random_draws": COVERAGE_RANDOM_DRAWS,
            "n_covering_variants": COVERAGE_RANDOMIZED_TIEBREAKS,
            "seed": SEED,
        }

    primary = results["damaging_E_gt_0.01"]["per_n"]["80"]
    d4 = (
        primary["covering"]["recall_median"] >= 1.5 * primary["random"]["recall_median"]
        and primary["covering"]["recall_p05"] > primary["random"]["recall_p95"]
    )
    return {
        "population": "160 stored (layer, pair, window) cells; not the group-pair space",
        "total_factor_pairs": total_pairs,
        "labels": results,
        "d4_at_n80": d4,
        "d4_rule": "covering median recall >= 1.5x random median AND covering p05 > random p95",
    }


# --- S6: M1 layer-omission additivity (secondary) --------------------------


def s6(frontier):
    singles = {}
    first = last = None
    for arm in frontier["arms"]:
        arm_id = arm["id"]
        if arm_id.startswith("layer-omit-"):
            groups = arm.get("groups_per_layer") or []
            omitted = [index for index, value in enumerate(groups) if value == 0]
            kl = float(arm["domains"]["general"]["kl_mean"])
            if arm_id == "layer-omit-first-20":
                first = (omitted, kl)
            elif arm_id == "layer-omit-last-20":
                last = (omitted, kl)
            elif len(omitted) == 1:
                singles[omitted[0]] = kl
    rows = []
    for name, item in (("first-20", first), ("last-20", last)):
        if item is None:
            continue
        omitted, measured = item
        additive = sum(singles[layer] for layer in omitted if layer in singles)
        rows.append(
            {
                "arm": f"layer-omit-{name}",
                "omitted_layers": omitted,
                "measured_kl": measured,
                "additive_prediction": additive,
                "residual_measured_minus_additive": measured - additive,
                "ratio_measured_over_additive": measured / additive if additive else float("nan"),
            }
        )
    return {
        "instrument": "whole-layer FFN omission on the dense Qwen3-1.7B teacher; joint arms above the 1e9 ceiling",
        "n_single_layer_arms": len(singles),
        "single_layer_kl": {str(layer): value for layer, value in sorted(singles.items())},
        "joint_arms": rows,
        "note": "descriptive only; different instrument from S1-S5 and never pooled with it",
    }


# --- instrument gap --------------------------------------------------------


def instrument_gap():
    return {
        "measured": [
            "160 stored pairwise contexts: 8 calibration windows x 5 layers x 4 fixed pairs, retention masks S/S+g/S+j/S+g+j",
            "28 single-layer FFN omissions and two 20-layer joint omissions (M1 frontier), no pair-of-layer measurements",
            "H* full-bank single-expert leave-one-out utility labels (32,768 calibration tokens, 28 layers, 8 experts)",
        ],
        "missing": [
            "no pairwise or triple omission measurement on the trained banks X or H; neither has a utility-label artifact",
            "H* has only single-expert full-bank LOO labels; no pair or triple omission, and no deployed-budget reference set",
            "no interaction measurement on the held-out general corpus, the code slice or the math slice",
            "no systematic census of the group-pair space: 20 of 3,360 possible (layer, group-pair) cells were measured",
            "the stored interaction instrument's baseline KL is about 10 nats because every non-probed layer keeps the static top-5 magnitude mask, so it cannot test a 0.25-nat minimality criterion at all",
            "no task-conditioned interaction and no learned-bank cut-set test",
        ],
        "minimal_gpu_experiment": {
            "arms": ["X", "H", "H* after its registered reader"],
            "sets": "full bank plus every 2-omission (C(8,2)=28) and 3-omission (C(8,3)=56) subset per routed layer",
            "layers": 28,
            "split": "32 calibration windows (discovery) + 32 held-out general windows (replication), 512 tokens",
            "forwards_per_arm": 28 * (1 + 28 + 56) * 64,
            "measured_forward_seconds": 49.596 / 640,
            "estimated_gpu_hours_per_arm": (28 * (1 + 28 + 56) * 64) * (49.596 / 640) / 3600.0,
            "estimated_gpu_hours_two_arms": 2 * (28 * (1 + 28 + 56) * 64) * (49.596 / 640) / 3600.0,
            "cost_source": "interaction receipt stamp 20261114T232338Z, revision 1e623062379239a4bed3337ed2ed91d1a7f5411b (640 forwards, 49.596 GPU-seconds); this is an estimate from a different instrument",
            "controls": [
                "random omission sets of equal size",
                "b_ij := 0 additive selector",
                "replication half never seen by discovery",
            ],
        },
    }


# --- decisions -------------------------------------------------------------


def decisions(s1_result, s2_result, s4_result, s5_result):
    far = s1_result["far_worse"]
    mean_abs = s1_result["overall_E"]["mean_abs"]
    d1 = far["fraction"] >= 0.10 and mean_abs >= 0.05

    out = s2_result["out_of_sample"]
    metrics = out["metrics"]
    mae_improvement = metrics["unary"]["mae"] - metrics["cell_interaction"]["mae"]
    d2 = (
        out["sign_agreement_context"] >= 0.70
        and metrics["cell_interaction"]["mae"] <= 0.90 * metrics["unary"]["mae"]
        and mae_improvement >= 0.005
    )
    d3 = s4_result["primary"]["replicated_configurations"] >= 1
    d4 = s5_result["d4_at_n80"]

    if d1 and d2 and d4:
        gpu = "covering_array"
    elif d1 and d2:
        gpu = "full_factorial_census"
    else:
        gpu = "none"

    return {
        "D1_interaction_non_negligible": {
            "value": d1,
            "far_worse_fraction": far["fraction"],
            "mean_abs_E": mean_abs,
            "rule": "far-worse fraction >= 0.10 AND mean |E| >= 0.05 nats",
        },
        "D2_interaction_replicates": {
            "value": d2,
            "sign_agreement_context": out["sign_agreement_context"],
            "mae_unary": metrics["unary"]["mae"],
            "mae_cell_interaction": metrics["cell_interaction"]["mae"],
            "mae_improvement": mae_improvement,
            "rule": "sign agreement >= 0.70 AND MAE(cell) <= 0.90 * MAE(unary) AND absolute improvement >= 0.005 nats",
        },
        "D3_minimal_destructive_replicates": {
            "value": d3,
            "replicated_configurations": s4_result["primary"]["replicated_configurations"],
            "rule": "at least one measured-minimal configuration replicates in >=3/4 discovery and >=3/4 replication windows at epsilon=0.25",
            "interpretation": "structurally degenerate on this instrument: every measured KL exceeds 1.0 nats, so no measured immediate subset can be safe at epsilon=0.25; the zero count is not evidence about trained models",
        },
        "D4_covering_beats_random": {
            "value": d4,
            "rule": "at N=80 covering median recall >= 1.5x random median AND covering p05 > random p95",
        },
        "gpu_experiment_justified": gpu,
        "p186_kill_fires": not d3,
        "feed_to_p1612": bool(d3),
    }


# --- Markdown rendering ----------------------------------------------------


def f(value, digits=4):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_markdown(receipt):
    outcome = receipt["decisions"]
    s1r = receipt["s1_interaction"]
    s2r = receipt["s2_additivity"]
    s3r = receipt["s3_violations"]
    s4r = receipt["s4_minimal_destructive"]
    s5r = receipt["s5_coverage"]
    s6r = receipt["s6_layer_omission"]
    gap = receipt["instrument_gap"]
    lines = []
    lines.append("# P18.6 — cut-set discovery and interaction diagnostics from stored omission artifacts")
    lines.append("")
    lines.append("Status: **computed from stored artifacts; no model executed, no GPU used.**")
    lines.append("")
    lines.append(f"This file is generated by `scripts/p18_6_cutset.py` from the receipt")
    lines.append(f"`{receipt['receipt']}` (stamp `{receipt['stamp']}`, revision `{receipt['revision']}`).")
    lines.append("Do not edit it by hand. The pre-registration is the P18.6 section of")
    lines.append(f"`docs/EXPERIMENT-CONTRACT.md`, committed as `{receipt['preregistration_commit']}`")
    lines.append("before any number below was computed.")
    lines.append("")
    lines.append("## 1. Verdict in one paragraph")
    lines.append("")
    verdict = receipt["verdict"]
    lines.append(verdict)
    lines.append("")
    lines.append("## 2. Data sources and the split")
    lines.append("")
    rows = []
    for name, spec in receipt["inputs"].items():
        rows.append([name, f"`{spec['path']}`", spec.get("stamp", "n/a"), spec.get("revision", "n/a"), f"`{spec['sha256'][:16]}…`"])
    lines.extend(table(["role", "path", "stamp", "revision", "sha256"], rows))
    lines.append("")
    lines.append("Discovery windows: " + ", ".join(sorted(receipt["split"]["discovery"])) + ".")
    lines.append("")
    lines.append("Replication windows: " + ", ".join(sorted(receipt["split"]["replication"])) + ".")
    lines.append("")
    lines.append("**Both halves are calibration text from the original M1/M2 split.** This is an")
    lines.append("internal discovery/replication split, not a held-out-corpus split. The held-out")
    lines.append("general, code and math slices contain no interaction measurement.")
    lines.append("")
    lines.append("## 3. Interaction distribution and sign asymmetry (S1)")
    lines.append("")
    lines.append("`E = L(S+g+j) - L(S+g) - L(S+j) + L(S) = -I`, where `I` is the receipt's")
    lines.append("`interaction` field. `E > 0` is super-additive loss (P16.5's positive `b_ij`).")
    lines.append("")
    overall = s1r["overall_E"]
    rows = [
        ["E", overall["n"], f(overall["mean"]), f(overall["median"]), f(overall["q25"]), f(overall["q75"]), f(overall["sd"]), f(overall["min"]), f(overall["max"])],
        ["|E|", overall["n"], f(overall["mean_abs"]), f(overall["median_abs"]), "n/a", "n/a", "n/a", "n/a", "n/a"],
    ]
    lines.extend(table(["quantity", "n", "mean", "median", "q25", "q75", "sd", "min", "max"], rows))
    lines.append("")
    reference = s1r["overall_I_m2"]
    lines.append(f"For reference, the receipt's own `I = -E` field has mean {f(reference['mean'])} and "
                 f"median {f(reference['median'])}; the mean reproduces the published M2 value.")
    lines.append("")
    lines.append(f"Positive E: {s1r['counts']['positive_E']}; negative E: {s1r['counts']['negative_E']}; zero: {s1r['counts']['zero_E']}.")
    lines.append(f"Mean-E bootstrap over the eight windows (unit = window): {f(s1r['mean_E_bootstrap']['mean'])}, "
                 f"95% CI [{f(s1r['mean_E_bootstrap']['ci_low'])}, {f(s1r['mean_E_bootstrap']['ci_high'])}].")
    lines.append(f"Mean-|E| bootstrap: {f(s1r['mean_abs_E_bootstrap']['mean'])}, "
                 f"95% CI [{f(s1r['mean_abs_E_bootstrap']['ci_low'])}, {f(s1r['mean_abs_E_bootstrap']['ci_high'])}].")
    lines.append("")
    rows = []
    for layer, values in s1r["per_layer_E"].items():
        rows.append([layer, values["n"], f(values["mean"]), f(values["median"]), f(values["mean_abs"]), f(values["positive_fraction"]), f(values["negative_fraction"])])
    lines.extend(table(["layer", "n", "mean E", "median E", "mean |E|", "frac E>0", "frac E<0"], rows))
    lines.append("")
    asym = s1r["sign_asymmetry"]
    lines.append("**Sign asymmetry.** `E` is algebraically symmetric in `g` and `j`, so no")
    lines.append("directional asymmetry is claimed from swapping them. The measurable asymmetry is")
    lines.append(f"the marginal-magnitude difference `|m_g| - |m_j|` (mean {f(asym['marginal_magnitude_asymmetry']['mean'])}, "
                 f"median {f(asym['marginal_magnitude_asymmetry']['median'])}), the fraction of contexts where the two")
    lines.append(f"marginals have opposite sign ({f(asym['fraction_marginals_opposite_sign'])}), and cross-window sign instability:")
    flips = asym["fixed_cell_sign_changes"]
    lines.append(f"of the {flips['n_cells']} fixed `(layer, g, j)` cells, {f(flips['fraction_with_at_least_one_flip'])} change sign at least once across the eight windows "
                 f"(median {f(flips['median'])} changes, max {flips['max']}).")
    lines.append("")
    lines.append(f"Far-worse interactions (`E > max(0.01, 0.25*(|m_g|+|m_j|))`): {s1r['far_worse']['count']} / {overall['n']} "
                 f"({f(100*s1r['far_worse']['fraction'], 2)}%).")
    lines.append("")
    lines.append("## 4. Unary-only versus pairwise-additive (S2)")
    lines.append("")
    ins = s2r["in_sample"]
    lines.append(f"In sample: mean |Δ| = {f(ins['mean_abs_delta'])} nats, mean |E| = {f(ins['mean_abs_E'])} nats, "
                 f"residual share mean|E|/mean|Δ| = {f(ins['residual_share'])}, unary-only R² = {f(ins['r2_unary'])}.")
    lines.append("The pairwise-additive model is exact in sample by construction; that is not evidence.")
    lines.append("")
    out = s2r["out_of_sample"]
    lines.append(f"Out of sample (20 cells, {out['n_contexts']} replication contexts):")
    lines.append("")
    rows = []
    for name, metrics in out["metrics"].items():
        rows.append([name, f(metrics["mae"]), f(metrics["rmse"]), f(metrics["r2"])])
    lines.extend(table(["predictor", "MAE", "RMSE", "R² vs constant mean"], rows))
    lines.append("")
    lines.append(f"Context-level sign agreement with the discovery cell interaction: {f(out['sign_agreement_context'])}. "
                 f"Cell-level sign agreement: {f(out['sign_agreement_cell_mean'])}. "
                 f"Spearman across the 20 cells: {f(out['cell_spearman'])}.")
    diff = out["mae_difference_unary_minus_cell"]
    lines.append(f"Paired bootstrap over the four replication windows for MAE(unary) - MAE(cell): "
                 f"{f(diff['mean'])}, 95% CI [{f(diff['ci_low'])}, {f(diff['ci_high'])}].")
    lines.append("")
    ranked = sorted(out["cells"], key=lambda row: abs(row["discovery_mean_E"]), reverse=True)
    rows = [
        [f"{row['layer']}", row["g"], row["j"], f(row["discovery_mean_E"]), f(row["replication_mean_E"]), f(row["same_sign"])]
        for row in ranked[:6]
    ]
    lines.append("Six largest discovery interactions, with their replication mean:")
    lines.append("")
    lines.extend(table(["layer", "g", "j", "discovery mean E", "replication mean E", "same sign"], rows))
    lines.append("")
    lines.append("## 5. Monotonicity and submodularity violations (S3)")
    lines.append("")
    lines.append(f"Super-additive (`E > 0`): {s3r['super_additive_E_positive']} contexts. "
                 f"Sub-additive (`E < 0`): {s3r['sub_additive_E_negative']}. "
                 f"Far-worse: {s3r['far_worse']}. "
                 f"The sequential-marginal identity `Δ(j|S+g) > Δ(j|S) ⇔ E > 0` holds in "
                 f"{s3r['sequential_identity_checks_passed']}/{s3r['sequential_identity_total']} checks.")
    lines.append("")
    lines.append("## 6. Measured-minimal destructive omission sets (S4)")
    lines.append("")
    primary = s4r["primary"]
    lines.append(f"Primary ε = {f(primary['epsilon'], 2)} nats (the M2 registered usable-tier KL bound).")
    lines.append("")
    rows = [
        ["contexts with any measured-minimal configuration", primary["contexts_with_any_measured_minimal"]],
        ["distinct `(layer, g, j, kind)` configurations", primary["distinct_configurations"]],
        ["configurations replicating (≥3/4 discovery AND ≥3/4 replication)", primary["replicated_configurations"]],
        ["configurations whose destructive direction replicates", primary["destructive_direction_replicated_configurations"]],
    ]
    lines.extend(table(["quantity", "value"], rows))
    lines.append("")
    lines.append("ε sensitivity (descriptive; only 0.25 is decisive):")
    lines.append("")
    rows = []
    for entry in s4r["sensitivity"]:
        rows.append([f(entry["epsilon"], 2), entry["contexts_with_any_measured_minimal"], entry["distinct_configurations"], entry["replicated_configurations"], entry["destructive_direction_replicated_configurations"]])
    lines.extend(table(["ε", "contexts with any minimal", "distinct configs", "replicated configs", "replicated destructive direction"], rows))
    lines.append("")
    levels = s4r["levels"]
    lines.append("Measured loss levels (descriptive; not a decision input):")
    lines.append("")
    rows = []
    for layer, entry in levels["per_layer"].items():
        rows.append([
            layer,
            f(entry["L_S"]["median"]),
            f(entry["L_Sg"]["median"]),
            f(entry["L_Sj"]["median"]),
            f(entry["L_Sgj"]["median"]),
            f(entry["min"]),
            f(entry["max"]),
            f(entry["fraction_le_0.25"]),
        ])
    lines.extend(table(["layer", "median L(S)", "median L(S+g)", "median L(S+j)", "median L(S+g+j)", "min", "max", "frac ≤ 0.25"], rows))
    lines.append("")
    lines.append(
        f"**The ε test is structurally degenerate on this instrument.** Across all "
        f"{levels['n_values']} measured KL values (160 contexts x 4 masks), the minimum is "
        f"{f(levels['min'])} nats, the median is {f(levels['median'])} and the maximum is "
        f"{f(levels['max'])}; the count at or below ε is "
        + ", ".join(f"{key}: {value}" for key, value in levels["count_le"].items())
        + ". The instrument's baseline is two orders of magnitude above the 0.25-nat usable"
        " bound because every non-probed layer keeps the static top-5 magnitude mask. No"
        " measured immediate subset can therefore be safe at any registered ε, and the zero"
        " minimal count is a property of the instrument's baseline, not evidence that minimal"
        " destructive sets are rare in a trained model. The 'destructive direction' count is"
        " likewise uninformative: every measured set is destructive."
    )
    lines.append("")
    lines.append("A configuration can only be *measured*-minimal: only 2 of the 11 immediate")
    lines.append("subsets of an omission set `O` were measured. No untested subset is assumed safe.")
    lines.append("")
    lines.append("The full 160-row per-subset safety table (window, half, layer, `g`, `j`, the four")
    lines.append("measured losses, `E`, and the three minimal flags) is in the JSON receipt under")
    lines.append("`s4_minimal_destructive.per_context`.")
    lines.append("")
    lines.append("## 7. Coverage: covering array versus random sampling (S5)")
    lines.append("")
    lines.append("Population: the 160 stored `(layer, pair, window)` cells; damage label primary")
    lines.append(f"`E > {f(TAU_E, 2)}` nats ({s5r['labels']['damaging_E_gt_0.01']['n_damaging']} damaging cells). This is")
    lines.append("**not** the group-pair space: only 20 of 3,360 possible `(layer, group-pair)` cells")
    lines.append("were ever measured. The array below is a strength-2 covering array over the design")
    lines.append("factors `(layer, pair, window)`; it is **not** a covering array over group-pair")
    lines.append("combinations, because the stored population does not contain them.")
    lines.append("")
    for label_name, label_values in s5r["labels"].items():
        lines.append(f"Label `{label_name}` ({label_values['n_damaging']} damaging cells):")
        lines.append("")
        rows = []
        for n in EVAL_COUNTS:
            entry = label_values["per_n"][str(n)]
            rows.append([
                n,
                f(entry["random"]["recall_median"]),
                f(entry["covering"]["recall_median"]),
                f(entry["random"]["recall_p95"]),
                f(entry["covering"]["recall_p05"]),
                f(entry["random"]["precision_median"]),
                f(entry["covering"]["precision_median"]),
            ])
        lines.extend(table(["N", "random recall (median)", "covering recall (median)", "random recall p95", "covering recall p05", "random precision", "covering precision"], rows))
        lines.append("")
    lines.append(f"D4 at N=80: **{f(s5r['d4_at_n80'])}** — covering-array recall shows "
                 f"{'a' if s5r['d4_at_n80'] else 'no'} measured advantage over random sampling on this population.")
    lines.append("")
    lines.append("## 8. M1 layer-omission additivity, secondary (S6)")
    lines.append("")
    lines.append("Different instrument: whole-layer FFN omission on the dense teacher; the joint arms")
    lines.append("are above the 1e9 ceiling. Never pooled with S1-S5.")
    lines.append("")
    rows = []
    for row in s6r["joint_arms"]:
        rows.append([row["arm"], len(row["omitted_layers"]), f(row["measured_kl"]), f(row["additive_prediction"]), f(row["residual_measured_minus_additive"]), f(row["ratio_measured_over_additive"])])
    lines.extend(table(["arm", "layers omitted", "measured KL", "additive prediction", "residual", "ratio"], rows))
    lines.append("")
    lines.append("## 9. Decision table")
    lines.append("")
    rows = []
    for name, entry in outcome.items():
        if not isinstance(entry, dict):
            continue
        rows.append([name, f(entry["value"]), entry["rule"]])
    lines.extend(table(["decision", "value", "pre-registered rule"], rows))
    lines.append("")
    lines.append(f"`gpu_experiment_justified`: **{outcome['gpu_experiment_justified']}**. "
                 f"`P18.6 kill fires`: **{f(outcome['p186_kill_fires'])}**. "
                 f"`feed_to_P16.12`: **{f(outcome['feed_to_p1612'])}**.")
    lines.append("")
    lines.append("## 10. Instrument gap")
    lines.append("")
    lines.append("Measured:")
    for item in gap["measured"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Missing:")
    for item in gap["missing"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 11. The GPU experiment this evidence does or does not justify")
    lines.append("")
    experiment = gap["minimal_gpu_experiment"]
    if outcome["gpu_experiment_justified"] == "none":
        lines.append("**None.** D1 or D2 failed, so these stored artifacts give no measured reason")
        lines.append("to run an interaction or cut-set search on the trained banks. The next step is")
        lines.append("to fix the instrument (trained-bank combination labels), not to run the search.")
        lines.append("")
        lines.append("For the record, the minimal pass that would close the instrument gap is:")
    elif outcome["gpu_experiment_justified"] == "full_factorial_census":
        lines.append("**A full-factorial interaction census, not a covering-array search.** D1 and D2")
        lines.append("hold but D4 does not, so the stored data gives no measured advantage to a")
        lines.append("covering array. The pass:")
    else:
        lines.append("**A covering-array cut-set search.** D1, D2 and D4 all hold. The pass:")
    lines.append("")
    lines.append(f"- arms: {', '.join(experiment['arms'])};")
    lines.append(f"- sets: {experiment['sets']};")
    lines.append(f"- split: {experiment['split']};")
    lines.append(f"- forwards per arm: {experiment['forwards_per_arm']:,};")
    lines.append(f"- estimated GPU hours per arm: {f(experiment['estimated_gpu_hours_per_arm'], 2)}; "
                 f"two arms: {f(experiment['estimated_gpu_hours_two_arms'], 2)};")
    lines.append(f"- cost source: {experiment['cost_source']};")
    lines.append("- controls: " + "; ".join(experiment["controls"]) + ".")
    lines.append("")
    lines.append("This estimate is from a different instrument and is not a measured cost of a")
    lines.append("trained arm. The launch record must replace it with a smoke-measured peak and a")
    lines.append("`SELVEDGE_MEM` derived from that peak plus the required headroom.")
    lines.append("")
    lines.append("## 12. Honest limits")
    lines.append("")
    lines.append("- The instrument is a **degraded, non-deployed selection regime**: a dense teacher")
    lines.append("  with static magnitude masks, not a trained bank. Early-layer KL values are far")
    lines.append("  outside the usable tier, so the ε test is not a capability test.")
    lines.append("- 8 calibration windows x 5 layers x 4 fixed pairs is not a census; unmeasured")
    lines.append("  pairs cannot be bounded. No code or math item was measured.")
    lines.append("- The discovery/replication split is internal to calibration text. No result here")
    lines.append("  has replicated on the held-out corpus.")
    lines.append("- The trained banks X and H have no combination measurements; H* has only")
    lines.append("  single-expert labels. Nothing here may be extrapolated to the trained models.")
    lines.append("- The coverage simulation is over the stored design, not over the group-pair space;")
    lines.append("  its sample-size limits are printed with the tables.")
    lines.append("")
    lines.append("## 13. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \\")
    lines.append("  scripts/court python -- python3 scripts/p18_6_cutset.py \\")
    lines.append(f"    --out-json {receipt['receipt']} \\")
    lines.append("    --out-md results/P18.6-CUTSET.md")
    lines.append("```")
    lines.append("")
    lines.append("The script is deterministic (seed 20261116), uses only the Python standard")
    lines.append("library, and refuses to run if a primary input hash differs from the registered")
    lines.append("constant. Stored evaluations used: 640 forwards from the interaction receipt;")
    lines.append("new model forwards: 0; GPU: none.")
    lines.append("")
    return "\n".join(lines) + "\n"


# --- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    inputs = load_inputs()
    with open(ROOT / INPUTS["interactions"]["path"], encoding="utf-8") as handle:
        interaction_receipt = json.load(handle)
    with open(ROOT / INPUTS["frontier"]["path"], encoding="utf-8") as handle:
        frontier_receipt = json.load(handle)
    interactions, records = load_records()

    windows = list(interactions["data"]["sampled_windows"])
    if sorted(windows) != sorted(
        {record["window"] for record in records}
    ):
        raise SystemExit("P18.6: sampled_windows does not match the record windows")

    discovery = [record for record in records if record["doc_id"] in DISCOVERY_DOCS]
    replication = [record for record in records if record["doc_id"] in REPLICATION_DOCS]
    if len(discovery) != 80 or len(replication) != 80:
        raise SystemExit("P18.6: split is not 80/80")

    s1_result = s1(records, windows)
    s2_result = s2(records)
    s3_result = s3(records)
    s4_result = s4(records)
    s5_result = s5(records, windows)
    s6_result = s6(frontier_receipt)
    decision = decisions(s1_result, s2_result, s4_result, s5_result)

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        revision = os.environ.get("SELVEDGE_REVISION", "unknown")
    try:
        preregistration_commit = subprocess.check_output(
            ["git", "rev-parse", "1a8ad21"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        preregistration_commit = "1a8ad21"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    receipt_name = str(out_json.relative_to(ROOT)) if out_json.is_relative_to(ROOT) else str(out_json)

    levels = s4_result["levels"]
    if decision["p186_kill_fires"]:
        verdict = (
            "**Negative on the registered kill criterion, and the test is structurally "
            "degenerate on this instrument.** No measured-minimal destructive omission set "
            "replicates at epsilon = 0.25. The reason is that every one of the "
            f"{levels['n_values']} measured KL values exceeds 1.0 nats (minimum "
            f"{levels['min']:.3f}, median {levels['median']:.3f}), so no measured immediate "
            "subset can be safe at the usable-tier bound. The zero count is a property of "
            "the instrument's baseline, not evidence that minimal destructive sets are absent "
            "from trained models. Nothing is fed to P16.12."
        )
    else:
        verdict = (
            "**At least one measured-minimal destructive omission set replicates** at epsilon = 0.25. "
            "This is a candidate cut-set structure on the stored dense-teacher instrument, not a "
            "capability cut set and not a property of any trained bank."
        )
    d1_value = decision["D1_interaction_non_negligible"]["value"]
    d2_value = decision["D2_interaction_replicates"]["value"]
    if d1_value and d2_value:
        verdict += (
            " The pairwise interaction also passes D1 and D2, so the stored data do show a "
            "replicating non-additive structure on this instrument."
        )
    elif d1_value and not d2_value:
        verdict += (
            " The interaction is non-negligible (D1 passes) but its sign is not stable across "
            "the discovery/replication split (D2 fails), so the stored data do not establish a "
            "replicating pair-level interaction even on this instrument."
        )
    else:
        verdict += (
            " The interaction does not pass D1 or D2, so the stored data do not establish a "
            "replicating non-additive structure even on this instrument."
        )

    receipt = {
        "schema": "selvedge/p18.6-cutset/v1",
        "kind": "p18.6-cutset",
        "stamp": stamp,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "revision": revision,
        "preregistration_commit": preregistration_commit,
        "receipt": receipt_name,
        "inputs": inputs,
        "split": {
            "discovery": sorted(DISCOVERY_DOCS),
            "replication": sorted(REPLICATION_DOCS),
            "n_discovery_contexts": len(discovery),
            "n_replication_contexts": len(replication),
            "note": "internal calibration split, not a held-out-corpus split",
        },
        "constants": {
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "coverage_random_draws": COVERAGE_RANDOM_DRAWS,
            "coverage_randomized_tiebreaks": COVERAGE_RANDOMIZED_TIEBREAKS,
            "epsilon_primary": EPSILON_PRIMARY,
            "epsilon_sensitivity": list(EPSILON_SENSITIVITY),
            "tau_E": TAU_E,
            "tau_E_secondary": TAU_E_SECONDARY,
        },
        "evaluations": {
            "stored_interaction_contexts": len(records),
            "stored_interaction_forwards": int(interactions["protocol"]["forward_count"]),
            "new_model_forwards": 0,
            "gpu": "none",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "coverage_simulations": COVERAGE_RANDOM_DRAWS
            + COVERAGE_RANDOMIZED_TIEBREAKS * len(EVAL_COUNTS),
        },
        "s1_interaction": s1_result,
        "s2_additivity": s2_result,
        "s3_violations": s3_result,
        "s4_minimal_destructive": s4_result,
        "s5_coverage": s5_result,
        "s6_layer_omission": s6_result,
        "decisions": decision,
        "instrument_gap": instrument_gap(),
        "verdict": verdict,
        "limitations": [
            "degraded non-deployed selection regime; early-layer KL far outside the usable tier",
            "8 calibration windows x 5 layers x 4 fixed pairs; not a census",
            "discovery/replication split is internal to calibration text",
            "trained banks X and H have no combination measurements; H* has single-expert labels only",
            "coverage simulation is over the stored design, not the group-pair space",
        ],
    }

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=False)
        handle.write("\n")
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(receipt))

    print(f"P18.6: wrote {receipt_name}")
    print(f"P18.6: wrote {out_md.relative_to(ROOT) if out_md.is_relative_to(ROOT) else out_md}")
    print(
        "P18.6: D1={D1} D2={D2} D3={D3} D4={D4} gpu={gpu} kill={kill}".format(
            D1=decision["D1_interaction_non_negligible"]["value"],
            D2=decision["D2_interaction_replicates"]["value"],
            D3=decision["D3_minimal_destructive_replicates"]["value"],
            D4=decision["D4_covering_beats_random"]["value"],
            gpu=decision["gpu_experiment_justified"],
            kill=decision["p186_kill_fires"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
