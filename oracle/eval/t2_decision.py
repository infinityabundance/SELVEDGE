#!/usr/bin/env python3
"""Apply the pre-registered T2 decision rule to the five eval receipts.

This is a *reader*: it consumes `oracle/eval/baseline.py` receipts (which carry
per-item and per-document values) and the per-arm run receipts, recomputes the
registered statistics, and applies `docs/EXPERIMENT-CONTRACT.md` T2.9/T2.10
exactly as written. It implements no new threshold and may not change one.

Registered endpoints (contract T2.9):

  * `PC` — primary composite, `(HumanEval pass@1 + GSM8K exact match +
    TruthfulQA MC1) / 3`, higher is better.
  * `P1` — co-primary pooled teacher KL on the fixed mixed interleaving,
    lower is better.

Decision order, fixed before measurement: `PC` is the decisive statistic; if a
comparison's paired 95% PC interval contains zero, that comparison's decisive
statistic becomes `P1`. No other endpoint may be substituted.

Margin rule: relative improvement `(arm - comparator)/comparator` for `PC` and
`(comparator - arm)/comparator` for KL; if the comparator's `PC` is below
`0.02`, the relative margin becomes an absolute `+0.01`.

Kill conditions applied: K15 (H* vs H, >=3%), K16 (H* vs D, >=5%), K17 (M vs
D0, >=5% and >=10% of the D0->D gap), K18 (X vs D, >=3%), K19 (aggregate gate).

Outputs `results/t2/decision-<stamp>.json` (the single source every published
T2 number is generated from) and prints the verdict.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

from metrics import stratified_bootstrap  # noqa: E402

TASKS = ("humaneval", "gsm8k", "truthfulqa")
ENGINES = ("H*", "H", "D", "M", "X", "D0")
KL_REGRESSION_RATIO = 1.02
PC_REGRESSION_ABS = 0.01
PC_RELATIVE_FLOOR = 0.02
PC_ABSOLUTE_MARGIN = 0.01

#: contract T2.10
THRESHOLDS = {"K15": 0.03, "K16": 0.05, "K17": 0.05, "K18": 0.03}
GAP_CLOSED_MIN = 0.10


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# loading


class ArmMetrics:
    def __init__(self, name: str, receipt: dict):
        self.name = name
        self.receipt = receipt
        results = receipt.get("results") or {}
        humaneval = results.get("humaneval") or {}
        gsm8k = results.get("gsm8k") or {}
        truthfulqa = results.get("truthfulqa") or {}
        self.task_items = {
            "humaneval": [row["id"] for row in humaneval.get("items") or []],
            "gsm8k": [row["id"] for row in gsm8k.get("items") or []],
            "truthfulqa": [row["id"] for row in truthfulqa.get("items") or []],
        }
        self.task_values = {
            "humaneval": np.array([1.0 if row["passed"] else 0.0
                                   for row in humaneval.get("items") or []]),
            "gsm8k": np.array([1.0 if row["correct"] else 0.0
                               for row in gsm8k.get("items") or []]),
            "truthfulqa": np.array([1.0 if row["correct"] else 0.0
                                    for row in truthfulqa.get("items") or []]),
        }
        self.task_means = {task: (float(values.mean()) if values.size else None)
                           for task, values in self.task_values.items()}
        present = [value for value in self.task_means.values() if value is not None]
        self.pc = float(np.mean(present)) if present else None
        distributional = results.get("distributional") or {}
        self.domains = {}
        for domain, block in (distributional.get("domains") or {}).items():
            documents = block.get("documents") or []
            self.domains[domain] = {
                "ids": [doc["doc_id"] for doc in documents],
                "kl": np.array([doc["kl"] for doc in documents]),
                "ppl": np.array([doc["ppl"] for doc in documents]),
                "top1": np.array([doc["top1"] for doc in documents]),
                "windows": int(block.get("window_count") or 0),
            }
        self.mixed = distributional.get("mixed") or {}
        # P1 is the pooled KL on the mixed interleaving. The harness defines it as
        # the equal-weight mean over domains; it is recomputed here from the
        # per-document values so the reported endpoint and the paired differences
        # below are the same numbers by construction.
        self.kl_mixed_stored = float((self.mixed.get("kl") or {}).get("mean", np.nan))
        domain_means = [float(block["kl"].mean()) for block in self.domains.values()
                        if block["kl"].size]
        self.kl_mixed = float(np.mean(domain_means)) if domain_means else self.kl_mixed_stored
        self.stamp = receipt.get("stamp")
        self.revision = receipt.get("revision")
        self.seconds = receipt.get("seconds")

    def kl_domain_means(self) -> dict:
        return {domain: float(block["kl"].mean()) for domain, block in self.domains.items()}

    def ppl_domain_means(self) -> dict:
        return {domain: float(block["ppl"].mean()) for domain, block in self.domains.items()}


def composite_bootstrap(task_values: dict, seed: int, resamples: int) -> dict:
    """Bootstrap the PC composite, resampling items within each task."""
    generator = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    used = 0
    for task in TASKS:
        values = task_values.get(task)
        if values is None or values.size == 0:
            continue
        indexes = generator.integers(0, values.size, size=(resamples, values.size))
        draws += values[indexes].mean(axis=1)
        used += 1
    if used == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "resamples": resamples,
                "seed": seed, "n": 0}
    draws /= used
    means = [float(task_values[t].mean()) for t in TASKS
             if task_values.get(t) is not None and task_values[t].size]
    return {
        "mean": float(np.mean(means)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "n": int(sum(task_values[t].size for t in TASKS if task_values.get(t) is not None)),
        "task_means": {task: (float(task_values[task].mean())
                              if task_values.get(task) is not None
                              and task_values[task].size else None) for task in TASKS},
    }


def composite_paired(left: dict, right: dict, seed: int, resamples: int) -> dict:
    """Paired PC difference (left - right) with a percentile CI.

    Items are paired across arms by position within each task (the harness uses
    a fixed item order and the same item set for every arm); resampling draws
    the same indices for both arms, so the difference is paired.
    """
    generator = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=np.float64)
    used = 0
    for task in TASKS:
        a, b = left.get(task), right.get(task)
        if a is None or b is None or a.size == 0 or b.size == 0:
            continue
        if a.size != b.size:
            raise SystemExit(f"paired PC comparison: {task} sizes differ "
                             f"({a.size} vs {b.size})")
        indexes = generator.integers(0, a.size, size=(resamples, a.size))
        draws += (a[indexes] - b[indexes]).mean(axis=1)
        used += 1
    if used == 0:
        raise SystemExit("paired PC comparison: no shared task")
    draws /= used
    means = []
    for task in TASKS:
        a, b = left.get(task), right.get(task)
        if a is not None and b is not None and a.size and b.size == a.size:
            means.append(float(a.mean() - b.mean()))
    return {
        "mean_difference": float(np.mean(means)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "n": int(sum(left[t].size for t in TASKS
                     if left.get(t) is not None and right.get(t) is not None)),
        "statistic": "paired composite bootstrap over items within each task",
    }


def paired_kl_domains(left: ArmMetrics, right: ArmMetrics, seed: int,
                     resamples: int) -> dict:
    """Paired, domain-stratified KL difference (left - right), equal weights."""
    per_domain = {}
    per_domain_mean = {}
    for domain in sorted(set(left.domains) & set(right.domains)):
        left_block, right_block = left.domains[domain], right.domains[domain]
        if left_block["ids"] != right_block["ids"]:
            raise SystemExit(f"paired KL: document order differs for {domain}")
        per_domain[domain] = left_block["kl"] - right_block["kl"]
        per_domain_mean[domain] = float(per_domain[domain].mean())
    if not per_domain:
        raise SystemExit("paired KL: no shared domain")
    pooled = stratified_bootstrap(per_domain, seed=seed, resamples=resamples)
    return {
        "mean_difference": pooled["mean"],
        "ci_low": pooled["ci_low"],
        "ci_high": pooled["ci_high"],
        "per_domain_mean_difference": per_domain_mean,
        "resamples": resamples,
        "seed": seed,
        "n": pooled["n"],
        "statistic": "paired per-document KL, equal weight per domain "
                     "(matches the harness's mixed-interleaving mean)",
    }


# ---------------------------------------------------------------------------
# the rule


def relative(improvement: float, comparator: float) -> float | None:
    if comparator is None or abs(comparator) < 1e-12:
        return None
    return improvement / comparator


def decide(arm: ArmMetrics, comparator: ArmMetrics, threshold: float,
           seed: int, resamples: int) -> dict:
    """One comparison, with the registered statistic order applied."""
    pc_paired = composite_paired(arm.task_values, comparator.task_values, seed, resamples)
    kl_paired = paired_kl_domains(arm, comparator, seed, resamples)

    pc_ci_contains_zero = pc_paired["ci_low"] <= 0.0 <= pc_paired["ci_high"]
    decisive = "P1" if pc_ci_contains_zero else "PC"

    # PC, higher is better.
    pc_improvement = pc_paired["mean_difference"]
    pc_rel = relative(pc_improvement, comparator.pc)
    if comparator.pc is not None and comparator.pc < PC_RELATIVE_FLOOR:
        pc_margin_kind = "absolute"
        pc_margin = PC_ABSOLUTE_MARGIN
        pc_margin_met = pc_improvement >= PC_ABSOLUTE_MARGIN
    else:
        pc_margin_kind = "relative"
        pc_margin = threshold
        pc_margin_met = pc_rel is not None and pc_rel >= threshold
    pc_ci_improves = pc_paired["ci_low"] > 0.0

    # KL, lower is better.
    kl_improvement = -kl_paired["mean_difference"]
    kl_rel = relative(kl_improvement, comparator.kl_mixed)
    kl_margin_met = kl_rel is not None and kl_rel >= threshold
    kl_ci_improves = kl_paired["ci_high"] < 0.0

    # No regression beyond the margin on the other endpoint.
    kl_regression_ok = arm.kl_mixed <= KL_REGRESSION_RATIO * comparator.kl_mixed
    pc_regression_ok = (arm.pc is not None and comparator.pc is not None
                        and arm.pc >= comparator.pc - PC_REGRESSION_ABS)

    if decisive == "PC":
        threshold_met = pc_margin_met and pc_ci_improves
        no_regression = kl_regression_ok
        passes = bool(threshold_met and no_regression)
        applied = (f"{decisive}: relative margin {threshold:.2%} "
                   if pc_margin_kind == "relative"
                   else f"{decisive}: absolute margin +{PC_ABSOLUTE_MARGIN}")
        improvement = pc_improvement
        comparison_ci = pc_paired
    else:
        threshold_met = kl_margin_met and kl_ci_improves
        no_regression = pc_regression_ok
        passes = bool(threshold_met and no_regression)
        applied = f"{decisive}: relative margin {threshold:.2%} (lower KL is better)"
        improvement = kl_improvement
        comparison_ci = kl_paired

    # T2.8 one-seed rule for the decisive pair: the item-level half-width must
    # be exceeded by twice, else "not separable from noise".
    half_width = (comparison_ci["ci_high"] - comparison_ci["ci_low"]) / 2.0
    separable = abs(improvement) > 2.0 * half_width

    return {
        "arm": arm.name,
        "comparator": comparator.name,
        "decisive_statistic": decisive,
        "decisive_reason": (
            "the paired PC interval contains zero, so T2.9's registered fallback "
            "makes P1 decisive" if decisive == "P1"
            else "the paired PC interval excludes zero, so PC is decisive"),
        "pc": {
            "arm": arm.pc,
            "comparator": comparator.pc,
            "difference": pc_paired,
            "relative_improvement": pc_rel,
            "margin_kind": pc_margin_kind,
            "margin": pc_margin,
            "margin_met": bool(pc_margin_met),
            "ci_improves": bool(pc_ci_improves),
            "ci_contains_zero": bool(pc_ci_contains_zero),
        },
        "kl": {
            "arm": arm.kl_mixed,
            "comparator": comparator.kl_mixed,
            "difference": kl_paired,
            "relative_improvement": kl_rel,
            "improvement": kl_improvement,
            "margin": threshold,
            "margin_met": bool(kl_margin_met),
            "ci_improves": bool(kl_ci_improves),
        },
        "no_regression": {
            "kl_within_1.02x": bool(kl_regression_ok),
            "pc_within_0.01_absolute": bool(pc_regression_ok),
            "endpoint_checked": "KL" if decisive == "PC" else "PC",
            "satisfied": bool(no_regression),
        },
        "rule_applied": applied,
        "one_seed_rule": {
            "decisive_pair": arm.name == "H*" and comparator.name == "H",
            "ci_half_width": half_width,
            "twice_half_width": 2.0 * half_width,
            "difference_abs": abs(improvement),
            "separable_from_noise": bool(separable),
            "note": "T2.8: without the replication the H*-H difference must exceed "
                    "twice the item-level CI half-width to count as anything",
        },
        "passes": passes,
        "threshold_met": bool(threshold_met),
    }


def gap_closed(m: ArmMetrics, d0: ArmMetrics, d: ArmMetrics, decisive: str) -> dict:
    """K17 clause 2: how much of the D0 -> D gap the memory arm closes."""
    if decisive == "PC":
        baseline, arm, reference = d0.pc, m.pc, d.pc
        numerator = arm - baseline
        denominator = reference - baseline
        better = "higher PC"
    else:
        baseline, arm, reference = d0.kl_mixed, m.kl_mixed, d.kl_mixed
        numerator = baseline - arm
        denominator = baseline - reference
        better = "lower KL"
    if denominator is None or abs(denominator) < 1e-9:
        return {
            "decisive_statistic": decisive,
            "baseline_d0": baseline,
            "arm": arm,
            "reference_d": reference,
            "numerator": numerator,
            "denominator": denominator,
            "fraction": None,
            "passes": False,
            "reason": f"the D0 -> D gap on {decisive} is zero or undefined, so the "
                      f"fraction cannot be computed ({better})",
        }
    fraction = numerator / denominator
    return {
        "decisive_statistic": decisive,
        "baseline_d0": baseline,
        "arm": arm,
        "reference_d": reference,
        "numerator": numerator,
        "denominator": denominator,
        "fraction": fraction,
        "minimum": GAP_CLOSED_MIN,
        "passes": bool(fraction >= GAP_CLOSED_MIN),
        "reason": f"{better}: (arm - D0) / (D - D0)" if decisive == "PC"
                  else f"{better}: (D0 - arm) / (D0 - D)",
    }


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0", required=True, help="eval receipt of the frozen D0")
    parser.add_argument("--arm", action="append", default=[],
                        help="NAME=path to an eval receipt (repeat for each arm)")
    parser.add_argument("--out-dir", default="results/t2")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--run-receipt-root", default="results/t2",
                        help="where per-arm run.json and eval.json live")
    args = parser.parse_args(argv)

    paths = {}
    for entry in args.arm:
        if "=" not in entry:
            raise SystemExit(f"--arm expects NAME=path, got {entry!r}")
        name, path = entry.split("=", 1)
        paths[name] = path
    missing = [name for name in ("D", "M", "X", "H", "H*") if name not in paths]
    if missing:
        raise SystemExit(f"missing eval receipts for: {', '.join(missing)}")

    def load(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    arms = {name: ArmMetrics(name, load(path)) for name, path in paths.items()}
    d0 = ArmMetrics("D0", load(args.d0))
    arms["D0"] = d0

    order = ["D0", "D", "M", "X", "H", "H*"]
    arm_table = {}
    for name in order:
        arm = arms[name]
        arm_table[name] = {
            "pc": composite_bootstrap(arm.task_values, args.seed, args.resamples),
            "kl_mixed": {
                "mean": arm.kl_mixed,
                "mean_stored_in_receipt": arm.kl_mixed_stored,
                "definition": "equal-weight mean of the per-domain mean per-document "
                              "teacher KL (the harness's mixed interleaving)",
                "bootstrap": arm.mixed.get("kl"),
            },
            "kl_mixed_top1": arm.mixed.get("top1"),
            "per_task": arm.task_means,
            "per_domain": {
                domain: {
                    "kl": float(block["kl"].mean()),
                    "kl_n": int(block["kl"].size),
                    "ppl": float(block["ppl"].mean()),
                    "top1": float(block["top1"].mean()),
                    "documents": int(block["kl"].size),
                    "windows": block["windows"],
                }
                for domain, block in sorted(arm.domains.items())
            },
            "eval_stamp": arm.stamp,
            "eval_revision": arm.revision,
            "eval_seconds": arm.seconds,
            "receipt": paths.get(name, args.d0),
        }

    comparisons = {
        "H*-H": decide(arms["H*"], arms["H"], THRESHOLDS["K15"], args.seed, args.resamples),
        "H*-D": decide(arms["H*"], arms["D"], THRESHOLDS["K16"], args.seed, args.resamples),
        "X-D": decide(arms["X"], arms["D"], THRESHOLDS["K18"], args.seed, args.resamples),
        "M-D0": decide(arms["M"], d0, THRESHOLDS["K17"], args.seed, args.resamples),
        "M-D": decide(arms["M"], arms["D"], THRESHOLDS["K17"], args.seed,
                      args.resamples),
        "H*-D0": decide(arms["H*"], d0, THRESHOLDS["K16"], args.seed, args.resamples),
    }

    k17_gap = gap_closed(arms["M"], d0, arms["D"],
                         comparisons["M-D0"]["decisive_statistic"])
    k17 = {
        "condition": "M vs D0 >= 5% relative with CI excluding zero (clause 1) AND "
                     f"gap_closed >= {GAP_CLOSED_MIN} (clause 2)",
        "clause_1_passes": comparisons["M-D0"]["passes"],
        "clause_2": k17_gap,
        "passes": bool(comparisons["M-D0"]["passes"] and k17_gap["passes"]),
    }
    kill = {
        "K15": {
            "statement": "H* must improve over H on the decisive statistic by >= 3% "
                         "relative with the paired 95% CI excluding zero and no "
                         "regression beyond the margin on the other endpoint",
            "comparison": comparisons["H*-H"],
            "passes": comparisons["H*-H"]["passes"],
        },
        "K16": {
            "statement": "H* must improve over D by >= 5% relative with the paired "
                         "95% CI excluding zero and no regression beyond the margin",
            "comparison": comparisons["H*-D"],
            "passes": comparisons["H*-D"]["passes"],
        },
        "K17": k17,
        "K18": {
            "statement": "X must improve over D by >= 3% relative with the paired "
                         "95% CI excluding zero and no regression beyond the margin",
            "comparison": comparisons["X-D"],
            "passes": comparisons["X-D"]["passes"],
        },
    }
    kill["K19"] = {
        "statement": "scaling the bank, P3.5 phases C/D, or spending WS4/WS5 on the "
                     "conditional-capacity assumption requires both K15 and K16",
        "k15": kill["K15"]["passes"],
        "k16": kill["K16"]["passes"],
        "passes": bool(kill["K15"]["passes"] and kill["K16"]["passes"]),
    }

    # The interpretation, quoted from T2.11, chosen by the observed pattern.
    h_star_vs_h = kill["K15"]["passes"]
    h_star_vs_d = kill["K16"]["passes"]
    if h_star_vs_h and h_star_vs_d:
        outcome = "hybrid_supported"
        interpretation = (
            "K15 and K16 pass: error-directed conditional capacity is supported at "
            "this scale; the bank-scaling ladder (P2.2) and P3.5 phases C/D become "
            "the next experiments.")
    elif h_star_vs_h and not h_star_vs_d:
        outcome = "error_direction_supported_but_not_over_dense"
        interpretation = (
            "K15 passes, K16 fails: error direction is a real routing effect but "
            "conditional capacity does not beat dense capacity of equal active size "
            "at this budget. The bank is not scaled.")
    elif not h_star_vs_h and h_star_vs_d:
        outcome = "inconsistent_k15_fails_k16_passes"
        interpretation = (
            "K16 passes while K15 fails: an inconsistent pattern that cannot be "
            "read as support for error direction. Reported as is; nothing is scaled, "
            "and the anomaly is recorded rather than explained away.")
    else:
        outcome = "error_direction_not_supported"
        interpretation = (
            "K15 fails: error-directed routing is not supported at this scale. "
            "WS3's central bet is recorded as a negative result, no bank scaling "
            "follows, and the mechanism is not carried into WS4/WS5 without a new "
            "pre-registration.")

    training = {}
    for name in ("D", "M", "X", "H", "H*"):
        path = os.path.join(args.run_receipt_root, name, "run.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        measured = receipt.get("measured") or {}
        compute = receipt.get("compute") or {}
        memory = (receipt.get("resource") or {}).get("memory") or {}
        training[name] = {
            "tokens": measured.get("tokens"),
            "steps": (receipt.get("budget") or {}).get("steps"),
            "tokens_per_second": measured.get("tokens_per_second"),
            "vram_peak_gib": measured.get("vram_peak_gib"),
            "wall_seconds": measured.get("wall_seconds"),
            "train_seconds": measured.get("train_seconds"),
            "active_unique_params": (receipt.get("admission") or {}).get(
                "active_unique_params"),
            "total_accessible_params": receipt.get("total_accessible"),
            "trainable_elements": receipt.get("trainable_elements"),
            "declared_flops_trained": compute.get("declared_flops_trained"),
            "declared_total_flops_per_token": compute.get(
                "declared_total_flops_per_token"),
            "checkpoint_bytes": (receipt.get("checkpoint") or {}).get("bytes"),
            "checkpoint_sha256": (receipt.get("checkpoint") or {}).get("sha256"),
            "seed": receipt.get("seed"),
            "revision": receipt.get("revision"),
            "run_receipt": os.path.join(args.run_receipt_root, name, "run.json"),
            "memory_bytes": (memory.get("bytes") or None),
        }

    stamp = stamp_now()
    payload = {
        "schema": "selvedge/t2-decision/v1",
        "stamp": stamp,
        "utc": utc_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "contract": "docs/EXPERIMENT-CONTRACT.md T2.9/T2.10 (with A8, A9)",
        "seed": args.seed,
        "resamples": args.resamples,
        "decision_order": {
            "primary": "PC = (HumanEval pass@1 + GSM8K exact match + TruthfulQA MC1) / 3",
            "co_primary": "P1 = pooled teacher KL on the fixed mixed interleaving",
            "fallback": "if a comparison's paired PC 95% CI contains zero, P1 is "
                        "decisive for that comparison",
            "margin_rule": "relative, or absolute +0.01 when the comparator's PC is "
                           "below 0.02",
            "no_regression": "KL <= 1.02x the comparator's and PC >= comparator - 0.01",
        },
        "arms": arm_table,
        "training": training,
        "comparisons": comparisons,
        "kill_conditions": kill,
        "outcome": outcome,
        "interpretation": interpretation,
        "caveats": [
            "one seed per arm (T2.8): no seed-level CI is claimed; the H*-H "
            "difference must exceed twice the item-level CI half-width",
            "the evaluation reports the compact training target; the full-vocabulary "
            "teacher KL is in the P2.4 instrument receipt (A9), not here",
            "every number in the T2 report is generated from this receipt",
        ],
    }
    out_path = os.path.join(args.out_dir, f"decision-{stamp}.json")
    write_json_atomic(out_path, payload)

    # ---- console verdict -------------------------------------------------
    print(f"T2 decision ({stamp}) — outcome: {outcome}")
    print()
    print(f"{'arm':4} {'PC':>7} {'KL mixed':>9} {'HE':>6} {'GSM8K':>7} {'MC1':>6}")
    for name in order:
        row = arm_table[name]
        pc = row["pc"]
        print(f"{name:4} {pc['mean']:>7.4f} {row['kl_mixed']['mean']:>9.4f} "
              f"{row['per_task']['humaneval']:>6.3f} "
              f"{row['per_task']['gsm8k']:>7.3f} "
              f"{row['per_task']['truthfulqa']:>6.3f}")
    print()
    for label, comparison in comparisons.items():
        pc = comparison["pc"]
        kl = comparison["kl"]
        print(f"{label:6} decisive={comparison['decisive_statistic']:2} "
              f"PC {pc['difference']['mean_difference']:+.4f} "
              f"[{pc['difference']['ci_low']:+.4f},{pc['difference']['ci_high']:+.4f}] "
              f"KL {kl['difference']['mean_difference']:+.4f} "
              f"[{kl['difference']['ci_low']:+.4f},{kl['difference']['ci_high']:+.4f}] "
              f"-> {'PASS' if comparison['passes'] else 'FAIL'}")
    print()
    for name in ("K15", "K16", "K17", "K18", "K19"):
        entry = kill[name]
        print(f"{name}: {'PASS' if entry['passes'] else 'FAIL'}")
    print()
    print(interpretation)
    print()
    print(f"receipt: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
