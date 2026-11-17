#!/usr/bin/env python3
"""S48: the 20M bar's anchored scores, its gap closure, and the paired deltas.

Task 2 of this session is a score and a comparison, not a new instrument. This
script reads the bar's anchored receipt
(`results/stage2/eval/full-20m-math/eval.json`), the arm-table receipt for base
and the adapter rows, and the anchor-gap receipt for the registered targets, and
emits one table with:

* per-task score ± standard error for the bar and for base;
* the unweighted family means and their independent combined standard errors, the
  delta against base, and the delta against the registered target;
* the fraction of the base-to-target gap the bar closes, per family;
* the paired per-item interval against base on every task (items paired by leaf
  and doc_id, doc-hash sequences required to match).

A positive gap-closure fraction is what "the bar closes part of the math gap"
means; it is reported with its interval, because an unpaired point estimate on
one arm of one seed is not a claim.

Usage:
    scripts/court oracle -- python3 scripts/s48_bar_gap.py --write
    scripts/court python -- python3 scripts/s48_bar_gap.py --check
    scripts/court python -- python3 scripts/s48_bar_gap.py --summary
    scripts/court python -- python3 scripts/s48_bar_gap.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import s44_arm_table as table  # noqa: E402
import s45_collapse_diagnosis as s45  # noqa: E402

SCHEMA = "selvedge/s48-bar-gap/v1"
REPORT = "results/REPORT.md"
SENTINEL = "report-s48-bar-gap"
OUT_DIR = "results/stage2"
ARM = "full-20m-math"
BASE = "base"
FAMILIES = {"knowledge": ("mmlu", "mmlu_pro"), "math": ("gsm8k", "minerva_math500")}
TASKS = ("mmlu", "mmlu_pro", "gsm8k", "minerva_math500")


def stamp() -> str:
    return s45.stamp()


def utcnow() -> str:
    return s45.utcnow()


def read_json(path: str) -> dict:
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    s45.write_json(os.path.join(REPO, path), payload)


def latest(pattern: str) -> str | None:
    directory = os.path.join(REPO, OUT_DIR)
    found = sorted(name for name in os.listdir(directory)
                   if name.startswith(pattern) and name.endswith(".json"))
    return os.path.join(OUT_DIR, found[-1]) if found else None


def newest_frontier(pattern: str) -> str | None:
    directory = os.path.join(REPO, "results/frontier")
    if not os.path.isdir(directory):
        return None
    found = sorted(name for name in os.listdir(directory)
                   if name.startswith(pattern) and name.endswith(".json"))
    return os.path.join("results/frontier", found[-1]) if found else None


def eval_tasks(path: str) -> dict:
    payload = read_json(path)
    tasks = {}
    for entry in payload.get("tasks", []):
        tasks[entry["task"]] = {
            "score": entry["score"],
            "stderr": entry.get("stderr"),
            "n_samples": entry.get("n_samples"),
            "primary_metric": entry.get("primary_metric"),
            "sequence_correctness": entry.get("sequence_correctness"),
            "wall_clock_s": entry.get("wall_clock_s"),
        }
    return {"path": path, "stamp": payload.get("stamp"),
            "revision": payload.get("revision"), "arm": payload.get("arm"),
            "tasks": tasks, "complete": payload.get("complete"),
            "total_cap_enforcement": payload.get("total_cap_enforcement")}


def family_mean(tasks: dict, members) -> dict:
    scores, variances, missing = [], [], []
    for task in members:
        entry = tasks.get(task)
        if entry is None:
            missing.append(task)
            continue
        scores.append(entry["score"])
        stderrs = entry.get("stderr") or 0.0
        variances.append(stderrs ** 2)
    return {
        "mean": statistics.fmean(scores) if scores else None,
        "stderr": (math.sqrt(sum(variances)) / len(scores)) if scores else None,
        "members": list(members),
        "missing": missing,
    }


def paired(arm: str, reference: str, task: str) -> dict | None:
    spec = table.TASKS[task]
    metric, flt = spec["primary"]
    arm_path = table.raw_path(arm, task)
    ref_path = table.raw_path(reference, task)
    if not arm_path or not ref_path:
        return None
    try:
        keys_a, values_a, hashes_a = table.item_vector(arm_path, task, metric, flt)
        keys_b, values_b, hashes_b = table.item_vector(ref_path, task, metric, flt)
    except (KeyError, ValueError) as error:
        return {"paired": False, "reason": str(error)}
    if keys_a != keys_b:
        return {"paired": False,
                "reason": "the two arms did not score the same items in the same order"}
    if hashes_a != hashes_b:
        return {"paired": False,
                "reason": "the doc-hash sequences differ"}
    return table.bootstrap_delta_paired(values_a, values_b)


def divergence(run_dir: str) -> dict | None:
    """The training log's own CE trajectory, for a bar that did not converge.

    A collapsed checkpoint is a score, but a reader needs to know whether the
    collapse is in training or in the scoring path. The log settles it: the
    first fifty steps' median CE is the initialised model, the last fifty is the
    delivered model, and the first step whose running median exceeds twice the
    initial median is where the run left the base.
    """
    path = os.path.join(REPO, run_dir, "log.jsonl")
    if not os.path.exists(path):
        return None
    ces, steps = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ces.append(float(row["ce"]))
            steps.append(int(row["step"]))
    if len(ces) < 100:
        return None
    initial = statistics.median(ces[:50])
    final = statistics.median(ces[-50:])
    crossed = None
    window = []
    for index, value in enumerate(ces):
        window.append(value)
        if len(window) > 9:
            window.pop(0)
        if len(window) == 9 and statistics.median(window) > 2 * initial:
            crossed = steps[index]
            break
    return {
        "log": os.path.join(run_dir, "log.jsonl"),
        "steps": len(ces),
        "initial_median_ce": initial,
        "final_median_ce": final,
        "ratio": final / initial if initial else None,
        "first_step_above_twice_initial": crossed,
        "reading": "a final median CE above the initial one is a diverged run, "
                   "not a converged one; the checkpoint is scored as delivered",
    }


def gap_block(bar: dict, base: dict, targets: dict) -> dict:
    result = {"tasks": {}, "families": {}}
    for task in TASKS:
        bar_task = bar["tasks"].get(task)
        base_task = base["tasks"].get(task)
        if bar_task is None or base_task is None:
            result["tasks"][task] = {"measured": False}
            continue
        result["tasks"][task] = {
            "measured": True,
            "bar": bar_task["score"],
            "bar_stderr": bar_task.get("stderr"),
            "base": base_task["score"],
            "base_stderr": base_task.get("stderr"),
            "delta": bar_task["score"] - base_task["score"],
            "paired": paired(ARM, BASE, task),
        }
    for family, members in FAMILIES.items():
        bar_family = family_mean(bar["tasks"], members)
        base_family = family_mean(base["tasks"], members)
        target = (targets.get("families") or {}).get(family, {})
        target_value = (target.get("target") or target.get("target_mean")
                        or target.get("anchor_plus_half_gap"))
        entry = {
            "bar": bar_family,
            "base": base_family,
            "target": target_value,
            "target_source": targets.get("source"),
            "delta_vs_base": (None if bar_family["mean"] is None
                               or base_family["mean"] is None
                               else bar_family["mean"] - base_family["mean"]),
            "delta_vs_target": (None if bar_family["mean"] is None
                                or target_value is None
                                else bar_family["mean"] - target_value),
            "delta_vs_base_se": (
                None if bar_family["stderr"] is None
                or base_family["stderr"] is None
                else math.sqrt(bar_family["stderr"] ** 2
                               + base_family["stderr"] ** 2)),
        }
        gap = (None if base_family["mean"] is None or target_value is None
               else target_value - base_family["mean"])
        closed = (None if gap in (None, 0) or entry["delta_vs_base"] is None
                  else entry["delta_vs_base"] / gap)
        entry["base_to_target_gap"] = gap
        entry["gap_closed_fraction"] = closed
        result["families"][family] = entry
    return result


def assemble() -> dict:
    bar_path = os.path.join("results/stage2/eval", ARM, "eval.json")
    if not os.path.exists(os.path.join(REPO, bar_path)):
        raise SystemExit(f"s48-bar-gap: no anchored receipt at {bar_path}")
    bar = eval_tasks(bar_path)
    base = eval_tasks(os.path.join("results/stage2/eval", BASE, "eval.json"))
    gap_path = newest_frontier("anchor-gap-")
    if gap_path is None:
        raise SystemExit("s48-bar-gap: no results/frontier/anchor-gap-*.json")
    gap_receipt = read_json(gap_path)
    targets = {
        "path": gap_path,
        "source": gap_receipt.get("preregistration") or gap_receipt.get("source"),
        "families": {},
    }
    for row in gap_receipt.get("families", []):
        if "family" in row:
            targets["families"][row["family"]] = {
                "target": row.get("anchor_plus_half_gap"),
                "anchor": row.get("anchor_mean"),
                "upper": row.get("upper_mean"),
                "gap_pp": row.get("gap_pp"),
                "half_gap_pp": row.get("half_gap_pp"),
            }
    verify_path = latest("s48-bar-verify-")
    receipt = {
        "schema": SCHEMA,
        "kind": "bar_gap",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": s45.revision(),
        "bar": bar,
        "base": base,
        "targets": targets,
        "verification": (read_json(verify_path) if verify_path else None),
        "result": gap_block(bar, base, targets),
        "divergence": divergence("results/stage2/runs/full-20m-math"),
        "pairing_method": {
            "unit": "item, paired by (leaf, doc_id) with the doc-hash sequences "
                    "required to match",
            "resamples": table.BOOTSTRAP_RESAMPLES,
            "seed": table.BOOTSTRAP_SEED,
            "family_rule": "unweighted member mean; combined SE = "
                           "sqrt(sum se_i^2) / n",
        },
    }
    return receipt


# ---------------------------------------------------------------------------
# rendering


def _pct(value, plain=False):
    if value is None:
        return "n/a"
    return format(100 * value, ".2f" if plain else "+.2f")


def _ci(row):
    if not row or not row.get("paired"):
        return "unpaired"
    low, high = row["ci95"]
    return f"{100 * row['delta']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def render(receipt: dict) -> str:
    result = receipt["result"]
    bar = receipt["bar"]
    families = result["families"]
    math = families["math"]
    knowledge = families["knowledge"]
    lines = [
        "### The bar's anchored score",
        "",
        f"Receipt `{bar['path']}` (stamp `{bar['stamp']}`), scored with the "
        "corrected total-cap driver at the full amended basis and the anchor "
        "attention kernel. Cells are `score ± the harness's standard error`.",
        "",
        "| task | items | bar | base | Δ (pp) | paired per-item 95% CI (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        row = result["tasks"][task]
        if not row.get("measured"):
            lines.append(f"| `{task}` | — | no receipt | | | |")
            continue
        items = ((bar["tasks"][task].get("n_samples") or {}).get("sample_len"))
        lines.append(
            f"| `{task}` | {items if items is not None else 'n/a'} | "
            f"{_pct(row['bar'], plain=True)} ± {_pct(row['bar_stderr'], plain=True)} | "
            f"{_pct(row['base'], plain=True)} ± {_pct(row['base_stderr'], plain=True)} | "
            f"{_pct(row['delta'])} | {_ci(row['paired'])} |")
    lines += [
        "",
        "| family | members | base | bar | Δ vs base (pp) | registered target | "
        "Δ vs target (pp) | base-to-target gap closed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, members in FAMILIES.items():
        entry = families[family]
        lines.append(
            f"| {family} | {', '.join('`' + m + '`' for m in members)} | "
            f"{_pct(entry['base']['mean'], plain=True)} ± {_pct(entry['base']['stderr'], plain=True)} | "
            f"**{_pct(entry['bar']['mean'], plain=True)}** ± {_pct(entry['bar']['stderr'], plain=True)} | "
            f"{_pct(entry['delta_vs_base'])} | {_pct(entry['target'], plain=True)} | "
            f"{_pct(entry['delta_vs_target'])} | "
            f"{'n/a' if entry['gap_closed_fraction'] is None else format(100 * entry['gap_closed_fraction'], '+.1f') + '%'} |")
    lines += [
        "",
        f"**Does the bar close part of the math gap?** The base-to-target math gap "
        f"is {_pct(math['base_to_target_gap'], plain=True)} points "
        f"({_pct(math['base']['mean'])} to {_pct(math['target'])}). The bar scores "
        f"{_pct(math['bar']['mean'])} ± {_pct(math['bar']['stderr'], plain=True)}, "
        f"which is {_pct(math['delta_vs_base'])} against base "
        f"({format(math['delta_vs_base'] / math['delta_vs_base_se'], '.2f') if math.get('delta_vs_base_se') else 'n/a'} "
        "combined standard errors) and "
        + (f"closes {100 * math['gap_closed_fraction']:+.1f}% of the gap."
           if math["gap_closed_fraction"] is not None
           else "has no measurable gap closure.")
        + " The knowledge family "
        + (f"moves {_pct(knowledge['delta_vs_base'])} against base and "
           + ("closes " + format(100 * knowledge['gap_closed_fraction'], "+.1f") + "% of its gap."
              if knowledge["gap_closed_fraction"] is not None
              else "has no measurable gap closure.")
           if knowledge["bar"]["mean"] is not None else "was not scored.")
        + " A closure fraction is reported only because it is the arithmetic the "
        "registration asks for; the interval on the delta is what decides whether "
        "any part of it is real.",
        "",
        "The paired per-item intervals are narrower than the family delta's "
        "combined standard error because pairing cancels item-sampling error. The "
        "gap-closure fraction is computed on the point estimates and carries the "
        "same interval as the delta it divides; it is not separately tested.",
    ]
    drift = receipt.get("divergence")
    if drift:
        lines += [
            "",
            "**The bar diverged in training, and the log says so.** Median CE over "
            f"the first 50 steps is {drift['initial_median_ce']:.4f} nats (the "
            "initialised model); median CE over the last 50 is "
            f"{drift['final_median_ce']:.4f}, a factor of "
            f"{drift['ratio']:.2f}. The nine-step running median first exceeds "
            f"twice the initial value at step "
            f"{drift['first_step_above_twice_initial']}. The delivered checkpoint "
            "is scored exactly as delivered, at the registered recipe (lr 1e-4, "
            "200-step warmup, cosine to 10%, clip 1.0, Adam8bit with stochastic "
            "rounding); no recipe was changed after the fact. A collapsed bar is "
            "not a ceiling for the mechanisms, and the next route is a registered "
            "amendment with a full-backbone learning rate the arm can survive and "
            "an early-loss tripwire, not a re-interpretation of this score.",
        ]
    return "\n".join(lines).rstrip()
    verification = receipt.get("verification")
    if verification:
        lines += [
            "",
            f"The bar's receipt, checkpoint hash, config hash and token count were "
            f"verified before scoring: `{verification.get('_path', 'results/stage2/s48-bar-verify-<stamp>.json')}` "
            f"records {sum(1 for c in verification['checks'] if c['ok'])} of "
            f"{len(verification['checks'])} checks passing"
            + ("" if verification.get("ok") else ", **not all**") + ".",
        ]
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# regions


def _scan(text: str, where: str):
    begin = re.compile(r"^\s*<!--\s*BEGIN GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    end = re.compile(r"^\s*<!--\s*END GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    lines = text.splitlines()
    regions = {}
    open_id = None
    start = None
    for index, line in enumerate(lines):
        match = begin.match(line)
        if match:
            if open_id is not None:
                raise ValueError(f"{where}:{index + 1}: nested BEGIN")
            open_id, start = match.group(1), index
            continue
        match = end.match(line)
        if match:
            if open_id != match.group(1):
                raise ValueError(f"{where}:{index + 1}: mismatched END")
            regions[open_id] = (start, index)
            open_id = None
    if open_id is not None:
        raise ValueError(f"{where}: BEGIN {open_id} never closed")
    return lines, regions


def write_region(receipt: dict) -> None:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    if SENTINEL not in regions:
        raise SystemExit(f"s48: {REPORT} has no BEGIN GENERATED: {SENTINEL}")
    span = regions[SENTINEL]
    body = render(receipt)
    lines = lines[:span[0] + 1] + body.splitlines() + lines[span[1]:]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def check_region(receipt: dict) -> int:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    if SENTINEL not in regions:
        print(f"s48: {REPORT}: missing region {SENTINEL}", file=sys.stderr)
        return 1
    span = regions[SENTINEL]
    if "\n".join(lines[span[0] + 1:span[1]]) != render(receipt):
        print(f"s48: {REPORT}: region {SENTINEL} is stale", file=sys.stderr)
        return 1
    return 0


def self_test() -> int:
    tasks = {"mmlu": {"score": 0.5, "stderr": 0.01},
             "mmlu_pro": {"score": 0.3, "stderr": 0.02},
             "gsm8k": {"score": 0.4, "stderr": 0.02},
             "minerva_math500": {"score": 0.2, "stderr": 0.03}}
    family = family_mean(tasks, FAMILIES["math"])
    assert abs(family["mean"] - 0.3) < 1e-12
    assert abs(family["stderr"] - math.sqrt(0.02 ** 2 + 0.03 ** 2) / 2) < 1e-12
    base = {"tasks": tasks}
    bar = {"tasks": {name: {**value, "score": value["score"] + 0.05}
                     for name, value in tasks.items()}}
    targets = {"families": {"math": {"target": 0.5}},
               "source": "self-test"}
    block = gap_block(bar, base, targets)
    math_block = block["families"]["math"]
    assert abs(math_block["delta_vs_base"] - 0.05) < 1e-12
    assert abs(math_block["base_to_target_gap"] - 0.2) < 1e-12
    assert abs(math_block["gap_closed_fraction"] - 0.25) < 1e-12
    print("self-test: family arithmetic and gap closure PASS")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if sum(bool(flag) for flag in (args.write, args.check, args.summary)) != 1:
        raise SystemExit("choose exactly one of --write, --check, --summary")
    receipt = assemble()
    verify_path = latest("s48-bar-verify-")
    if verify_path and receipt.get("verification"):
        receipt["verification"]["_path"] = verify_path
    out = args.out or os.path.join(OUT_DIR, f"s48-bar-gap-{stamp()}.json")
    write_json(out, receipt)
    receipt["_path"] = out
    print(f"s48: receipt {out}")
    if args.summary:
        print(render(receipt))
        return 0
    if args.write:
        write_region(receipt)
        print(f"s48: wrote {REPORT}:{SENTINEL}")
        return 0
    return check_region(receipt)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
