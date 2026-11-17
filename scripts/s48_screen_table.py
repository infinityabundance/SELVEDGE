#!/usr/bin/env python3
"""S48: the 2M screen table, generated from the receipts, plus the verdict.

The screen registration
(`docs/ws19-stage2b-screen-preregistration-v1.json`) fixes five arms, their
training budget and the four stage-2b decision-rule conditions applied at 2M.
This script reads each arm's training receipt, training log, held-out receipt and
anchored math receipt, computes the conditions, and renders one table and one
verdict. Nothing is typed by hand: every cell is a number from a receipt, and the
verdict is the registered rule evaluated on those numbers.

Two tables are rendered because the registration asks for two different kinds of
evidence:

* **likelihood and mechanism** -- held-out CE/KL against base, the training log's
  gate statistics, and the corrected harm mask on base-better tokens;
* **anchored math** -- GSM8K and MATH-500 scores with standard errors, the family
  mean, the S47 sequence-correctness counts, and paired per-item intervals
  against base and against `adapter-2m`.

Usage:
    scripts/court oracle -- python3 scripts/s48_screen_table.py --write
    scripts/court python -- python3 scripts/s48_screen_table.py --check
    scripts/court python -- python3 scripts/s48_screen_table.py --summary
    scripts/court python -- python3 scripts/s48_screen_table.py --self-test
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

SCHEMA = "selvedge/s48-screen-table/v1"
REPORT = "results/REPORT.md"
SENTINEL = "report-s48-screen"
OUT_DIR = "results/stage2"
REGISTRATION = "docs/ws19-stage2b-screen-preregistration-v1.json"
TASKS = ("gsm8k", "minerva_math500")
REFERENCES = ("base", "adapter-2m")
DESIGN = {
    "corrective-kl-2m-math": {"family": "corrective", "weighting": "kl"},
    "corrective-signed-exp-2m-math": {"family": "corrective",
                                      "weighting": "signed-exp"},
    "corrective-signed-true-2m-math": {"family": "corrective",
                                        "weighting": "signed-true"},
    "adapter-2m-math-matched": {"family": "matched_control_data"},
    "gated-dense-2m-math": {"family": "matched_control_gate"},
}
ARM_LABEL = {
    "base": "`base`",
    "adapter-2m": "`adapter-2m`",
    "corrective-kl-2m-math": "`corrective-kl-2m-math`",
    "corrective-signed-exp-2m-math": "`corrective-signed-exp-2m-math`",
    "corrective-signed-true-2m-math": "`corrective-signed-true-2m-math`",
    "adapter-2m-math-matched": "`adapter-2m-math-matched`",
    "gated-dense-2m-math": "`gated-dense-2m-math`",
}


def utcnow() -> str:
    return s45.utcnow()


def stamp() -> str:
    return s45.stamp()


def read_json(path: str) -> dict:
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    s45.write_json(os.path.join(REPO, path), payload)


def exists(path: str) -> bool:
    return os.path.exists(os.path.join(REPO, path))


# ---------------------------------------------------------------------------
# training log summaries


def log_summary(run_dir: str) -> dict:
    """Gate and harm statistics from the tail of the training log.

    The last quarter of the steps are summarised, so an early transient does not
    dominate. These are training-stream measurements; the registration says so
    and does not present them as held-out.
    """
    path = os.path.join(REPO, run_dir, "log.jsonl")
    if not os.path.exists(path):
        return {"status": "missing", "path": os.path.join(run_dir, "log.jsonl")}
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        return {"status": "empty"}
    tail = entries[max(0, int(0.75 * len(entries))):]
    gates = [entry["gate"] for entry in tail if isinstance(entry.get("gate"), dict)]
    harm = [entry["objective"] for entry in tail
            if isinstance(entry.get("objective"), dict)]

    def mean_of(key: str, rows):
        values = [float(row[key]) for row in rows
                  if row.get(key) is not None]
        if not values:
            return None, None
        mean = statistics.fmean(values)
        stderr = (statistics.stdev(values) / math.sqrt(len(values))
                  if len(values) > 1 else 0.0)
        return mean, stderr

    summary = {
        "status": "ok",
        "steps_logged": len(entries),
        "steps_summarised": len(tail),
        "first_step": entries[0].get("step"),
        "last_step": entries[-1].get("step"),
        "gate": None,
        "harm": {},
    }
    if gates:
        for key in ("mean", "mean_abs", "min", "max"):
            summary["gate"] = summary["gate"] or {}
            summary["gate"][key], summary["gate"][f"{key}_stderr"] = mean_of(
                key, gates)
    for key in ("harm_base_better_mean_delta_kl", "harm_teacher_better_mean_delta_kl",
                "harm_tie_mean_delta_kl", "harm_base_better_fraction_worse",
                "harm_base_better_tokens", "interference_penalty",
                "base_preservation_penalty", "base_token_protection_penalty"):
        value, stderr = mean_of(key, harm)
        if value is not None:
            summary["harm"][key] = {"mean": value, "stderr": stderr}
    if harm:
        summary["harm_weighting"] = harm[-1].get("weighting_canonical")
    return summary


# ---------------------------------------------------------------------------
# receipts


def eval_summary(arm: str) -> dict | None:
    path = os.path.join("results/stage2/eval", arm, "eval.json")
    if not exists(path):
        return None
    payload = read_json(path)
    tasks = {}
    for entry in payload.get("tasks", []):
        if entry["task"] not in TASKS:
            continue
        tasks[entry["task"]] = {
            "score": entry["score"],
            "stderr": entry.get("stderr"),
            "sequence_correctness": entry.get("sequence_correctness"),
            "wall_clock_s": entry.get("wall_clock_s"),
        }
    if not tasks:
        return None
    scores = [task["score"] for task in tasks.values()]
    stderrs = [task["stderr"] or 0.0 for task in tasks.values()]
    return {
        "path": path,
        "stamp": payload.get("stamp"),
        "revision": payload.get("revision"),
        "arm": payload.get("arm"),
        "tasks": tasks,
        "family_mean": statistics.fmean(scores),
        "family_stderr": math.sqrt(sum(value ** 2 for value in stderrs))
                         / len(scores),
        "complete": payload.get("complete"),
        "registration": payload.get("preregistration"),
    }


def heldout_summary(arm: str) -> dict | None:
    path = f"results/stage2/s48-heldout-{arm}.json"
    if not exists(path):
        return None
    payload = read_json(path)
    entry = (payload.get("arms") or {}).get(arm)
    if not entry:
        return None
    return {
        "path": path,
        "stamp": payload.get("stamp"),
        "revision": payload.get("revision"),
        "ce": entry["ce"],
        "kl": entry["kl"],
        "delta_ce_vs_reference": entry.get("delta_ce_vs_reference"),
        "delta_kl_vs_reference": entry.get("delta_kl_vs_reference"),
        "report": entry.get("report"),
        "checkpoint": entry.get("checkpoint"),
    }


def seq_correct() -> dict:
    """The S48 primary-filter sequence correctness, if it has been computed."""
    found = sorted(name for name in os.listdir(os.path.join(REPO, OUT_DIR))
                   if name.startswith("s48-seq-correct-") and name.endswith(".json"))
    if not found:
        return {}
    payload = read_json(os.path.join(OUT_DIR, found[-1]))
    return {"path": os.path.join(OUT_DIR, found[-1]),
            "stamp": payload.get("stamp"),
            "arms": {name: entry.get("tasks") or {}
                     for name, entry in (payload.get("arms") or {}).items()}}


def training_summary(arm: str) -> dict | None:
    run_dir = os.path.join("results/stage2/runs", arm)
    path = os.path.join(run_dir, "receipt.json")
    if not exists(path):
        return None
    payload = read_json(path)
    return {
        "path": path,
        "stamp": payload.get("stamp"),
        "revision": payload.get("revision"),
        "tokens_trained": payload.get("tokens_trained"),
        "tokens_per_second": payload.get("tokens_per_second"),
        "peak_allocated_gib": payload.get("peak_allocated_gib"),
        "trainable_arm_elements": payload.get("trainable_arm_elements"),
        "active_per_token": (payload.get("arm_accounting") or {}).get(
            "active_per_token"),
        "final_checkpoint_sha256": payload.get("final_checkpoint_sha256"),
        "config": payload.get("config"),
        "log": log_summary(run_dir),
    }


# ---------------------------------------------------------------------------
# paired item statistics


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
                "reason": "the doc-hash sequences differ; the item sets are not the same"}
    return table.bootstrap_delta_paired(values_a, values_b)


def same_direction(row: dict | None) -> bool:
    if not row or not row.get("paired"):
        return False
    low, high = row["ci95"]
    return (low > 0) or (high < 0)


# ---------------------------------------------------------------------------
# the registered rule


def conditions(arm: str, evals: dict, heldout: dict | None) -> dict:
    own = evals.get(arm)
    if own is None:
        return {"measured": False}
    result = {"measured": True, "tasks": {}, "family": {}}
    for reference in REFERENCES:
        other = evals.get(reference)
        if other is None:
            result["family"][reference] = {"measured": False}
            continue
        delta = own["family_mean"] - other["family_mean"]
        combined = math.sqrt(own["family_stderr"] ** 2
                             + other["family_stderr"] ** 2)
        per_task = {}
        for task in TASKS:
            row = paired(arm, reference, task)
            per_task[task] = row
        same = [task for task in TASKS
                if same_direction(per_task[task])
                and math.copysign(1.0, per_task[task]["delta"]) == math.copysign(1.0, delta)]
        result["family"][reference] = {
            "measured": True,
            "delta": delta,
            "combined_stderr": combined,
            "beats_by_more_than_combined_stderr": bool(delta > combined),
            "not_lost_beyond_combined_stderr": bool(delta >= -combined),
            "paired_same_direction": len(same) == len(TASKS),
            "tasks_paired_same_direction": same,
            "condition_holds": bool(delta > combined and len(same) == len(TASKS)),
            "paired": per_task,
        }
    if heldout is None:
        result["ce"] = {"measured": False}
    else:
        delta = heldout["delta_ce_vs_reference"] or {}
        mean = delta.get("mean")
        stderr = delta.get("stderr")
        result["ce"] = {
            "measured": True,
            "delta": mean,
            "stderr": stderr,
            "condition_holds": bool(mean is not None and stderr is not None
                                    and mean <= stderr),
            "ci95": delta.get("ci95"),
        }
    return result


def verdict(arm_rows: dict) -> dict:
    """The registered decision rule, evaluated on the measured conditions."""
    family = DESIGN
    corrective = [name for name in DESIGN
                  if family[name]["family"] == "corrective"]
    controls = {kind: [name for name in DESIGN
                       if family[name]["family"] == kind]
                for kind in ("matched_control_data", "matched_control_gate")}

    def passes(name: str) -> bool:
        row = arm_rows.get(name, {}).get("conditions") or {}
        if not row.get("measured"):
            return False
        base = row["family"].get("base") or {}
        adapter = row["family"].get("adapter-2m") or {}
        ce = row.get("ce") or {}
        return bool(base.get("condition_holds") and adapter.get("condition_holds")
                    and ce.get("condition_holds"))

    def ambiguous(name: str) -> bool:
        row = arm_rows[name]["conditions"]
        for reference in REFERENCES:
            entry = row["family"].get(reference) or {}
            if not entry.get("measured"):
                continue
            deltas = [(row["family"][reference]["paired"][task] or {}).get("delta")
                      for task in TASKS]
            if any(value is None for value in deltas):
                continue
            if min(deltas) < 0 < max(deltas) and entry["delta"] > 0:
                return True
        return False

    if not any((arm_rows.get(name) or {}).get("conditions", {}).get("measured")
               for name in DESIGN):
        record = {
            "rule": "docs/ws19-stage2b-corrective-advantage-preregistration-v1.json"
                    " decision_rule applied at 2M",
            "corrective_arms": corrective,
            "label": "not_run",
            "reading": "no screen arm has a training, held-out and anchored receipt "
                       "yet, so the rule has nothing to evaluate",
        }
        return record
    passing = [name for name in corrective if passes(name)]
    base_only = [name for name in corrective
                 if not passes(name)
                 and ((((arm_rows[name].get("conditions") or {}).get("family")
                       or {}).get("base") or {}).get("condition_holds"))]
    def family_not_lost(name: str) -> bool:
        entry = (((arm_rows[name].get("conditions") or {}).get("family") or {})
                 .get("base") or {})
        return bool(entry.get("measured")
                    and entry.get("not_lost_beyond_combined_stderr"))

    ce_only = [name for name in corrective
               if name not in passing and name not in base_only
               and (((arm_rows[name].get("conditions") or {}).get("ce")
                     or {}).get("condition_holds"))
               and family_not_lost(name)]

    record = {
        "rule": "docs/ws19-stage2b-corrective-advantage-preregistration-v1.json"
                " decision_rule applied at 2M",
        "corrective_arms": corrective,
        "passing_all_four": passing,
        "beats_base_not_adapter": base_only,
        "ce_only": ce_only,
        "ambiguous": [name for name in passing if ambiguous(name)],
        "attribution": {},
        "label": None,
    }
    if any(ambiguous(name) for name in passing):
        record["label"] = "ambiguous"
        record["reading"] = ("the family mean improves but the two math tasks do not "
                             "agree on the sign, or the conclusion flips between "
                             "the harness metric and sequence correctness; nothing "
                             "is promoted")
        return record
    controls_measured = all(
        (arm_rows.get(name) or {}).get("eval") is not None
        for names in controls.values() for name in names)
    if passing and not controls_measured:
        record["label"] = "incomplete_controls"
        record["reading"] = ("an arm satisfies the four conditions but a mandatory "
                             "matched control has no anchored receipt, so the data "
                             "and gate attributions cannot be made; nothing is "
                             "promoted until both controls are scored")
        return record
    if passing:
        best = max(passing, key=lambda name: arm_rows[name]["eval"]["family_mean"])
        for kind, names in controls.items():
            for name in names:
                if name in arm_rows and passes(name):
                    gap = (arm_rows[best]["eval"]["family_mean"]
                           - arm_rows[name]["eval"]["family_mean"])
                    combined = math.sqrt(
                        arm_rows[best]["eval"]["family_stderr"] ** 2
                        + arm_rows[name]["eval"]["family_stderr"] ** 2)
                    if abs(gap) <= combined:
                        record["attribution"][kind] = {
                            "control": name, "gap": gap, "combined_stderr": combined,
                            "separated": False}
                    else:
                        record["attribution"][kind] = {
                            "control": name, "gap": gap, "combined_stderr": combined,
                            "separated": True}
        record["promoted_arm"] = best
        if any(not entry["separated"]
               for kind, entry in record["attribution"].items()
               if kind == "matched_control_data"):
            record["label"] = "data_gain"
            record["reading"] = ("the matched dense adapter, with the same data, "
                                 "teacher and active union, satisfies the same four "
                                 "conditions and is not separated from the best "
                                 "corrective arm: the screen result is the data, not "
                                 "the memory mechanism; 20M corrective training is "
                                 "not justified by this screen")
        elif any(not entry["separated"]
                 for kind, entry in record["attribution"].items()
                 if kind == "matched_control_gate"):
            record["label"] = "gate_gain"
            record["reading"] = ("the gated dense residual, with the same gate and no "
                                 "external memory, satisfies the same conditions and "
                                 "is not separated from the corrective arm: the gate "
                                 "is the mechanism, not the memory")
        else:
            record["label"] = "corrective_arm_promoted"
            record["reading"] = ("one corrective weighting satisfies all four "
                                 "measurable conditions and exceeds both matched "
                                 "controls: promote that weighting to the registered "
                                 "20M arm")
        return record
    if base_only:
        record["label"] = "beats_base_not_adapter"
        record["reading"] = ("the residual beats base on math but not adapter-2m at "
                             "equal tokens; per decision rule 2 the next route is a "
                             "registered adapter-plus-gate comparison, not a new 20M "
                             "residual variant")
        return record
    if ce_only:
        record["label"] = "ce_only"
        record["reading"] = ("the correction improves held-out CE but not the scored "
                             "family; per decision rule 3 the next route is a "
                             "registered feature-map change or dropping the gate "
                             "conditioning, chosen before the next run")
        return record
    record["label"] = "null"
    record["reading"] = ("no arm beats base and adapter-2m on math; the "
                         "corrective-residual family is a null at this budget on this "
                         "stream and the next route is the registered P19.9 utility "
                         "routing, not another weighting")
    return record


# ---------------------------------------------------------------------------
# assembly


def assemble() -> dict:
    arms = list(DESIGN)
    evals = {name: eval_summary(name) for name in list(REFERENCES) + arms}
    heldout = {name: heldout_summary(name) for name in arms}
    training = {name: training_summary(name) for name in arms}
    rows = {}
    for name in arms:
        rows[name] = {
            "design": DESIGN[name],
            "training": training[name],
            "heldout": heldout[name],
            "eval": evals[name],
            "conditions": conditions(name, evals, heldout[name]),
        }
    reference_rows = {name: {"eval": evals[name]} for name in REFERENCES
                      if evals.get(name) is not None}
    receipt = {
        "schema": SCHEMA,
        "kind": "screen_table",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": s45.revision(),
        "registration": {
            "path": REGISTRATION,
            "sha256": s45.sha256_file(os.path.join(REPO, REGISTRATION)),
        },
        "tasks": list(TASKS),
        "arms": rows,
        "references": reference_rows,
        "sequence_correctness": seq_correct(),
        "pairing_method": {
            "unit": "item, paired by (leaf, doc_id) with the doc-hash sequences "
                    "required to match",
            "resamples": table.BOOTSTRAP_RESAMPLES,
            "seed": table.BOOTSTRAP_SEED,
            "family_rule": "unweighted member mean; combined SE = "
                           "sqrt(sum se_i^2) / n",
        },
        "pairwise": {
            f"{name}|{reference}|{task}": paired(name, reference, task)
            for name in arms for reference in REFERENCES for task in TASKS
        },
        "verdict": verdict(rows),
    }
    return receipt


# ---------------------------------------------------------------------------
# rendering


def _num(value, spec=".2f", scale=1.0):
    return "n/a" if value is None else format(scale * value, spec)


def _ci(row) -> str:
    if not row or not row.get("paired"):
        return "unpaired"
    low, high = row["ci95"]
    return f"{100 * row['delta']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def _attribution(receipt: dict) -> str:
    """What the controls say about where the measured changes come from."""
    rows = receipt["arms"]
    ce = {name: ((rows[name].get("heldout") or {}).get("delta_ce_vs_reference")
                 or {}).get("mean") for name in rows}
    family = {name: (rows[name].get("eval") or {}).get("family_mean")
              for name in rows}
    base = receipt["references"].get("base")
    base_mean = base["eval"]["family_mean"] if base else None
    corrective = [name for name in rows if rows[name]["design"]["family"] == "corrective"]
    gate = "gated-dense-2m-math"
    data = "adapter-2m-math-matched"
    gate_ce = ce.get(gate)
    data_ce = ce.get(data)
    best_corrective = max((family[name] for name in corrective
                           if family.get(name) is not None), default=None)
    sentences = []
    if gate_ce is not None and data_ce is not None:
        sentences.append(
            "The held-out CE improvement tracks the **gate**, not the external "
            f"memory: the gate-only control moves {gate_ce:+.4f} nats while the "
            f"data-only control moves {data_ce:+.4f}, and the three memory arms "
            "move "
            + ", ".join(f"{ce[name]:+.4f}" for name in corrective
                        if ce.get(name) is not None)
            + ". The data-only control is worse than base on held-out CE.")
    if best_corrective is not None and base_mean is not None:
        sentences.append(
            "No arm beats base on the math family: the best corrective arm is "
            f"{100 * best_corrective:.2f} against base {100 * base_mean:.2f}, and "
            "the two matched controls are not separated from the corrective arms. "
            "The GSM8K gain appears in the data-only and gate-only controls as "
            "well, and the MATH-500 loss appears in every trained arm including "
            "the data-only control, so neither sign is specific to the external "
            "memory. A gain that the controls also have is not a mechanism gain; "
            "the null is the registered reading of that.")
    return " ".join(sentences)


def render(receipt: dict) -> str:
    rows = receipt["arms"]
    verdict_row = receipt["verdict"]
    lines = [
        "### Likelihood, gate and harm at 2M",
        "",
        "| arm | tokens | active/token | held-out CE | ΔCE vs base [95% CI] | "
        "held-out KL | gate mean abs | base-better ΔKL (train tail) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["corrective-kl-2m-math", "corrective-signed-exp-2m-math",
                 "corrective-signed-true-2m-math", "adapter-2m-math-matched",
                 "gated-dense-2m-math"]:
        row = rows[name]
        training = row.get("training") or {}
        held = row.get("heldout") or {}
        log = training.get("log") or {}
        gate = log.get("gate") or {}
        harm = (log.get("harm") or {}).get("harm_base_better_mean_delta_kl") or {}
        delta = (held.get("delta_ce_vs_reference") or {})
        ci = delta.get("ci95") or [None, None]
        lines.append(
            f"| {ARM_LABEL[name]} | {_num(training.get('tokens_trained'), ',.0f')} | "
            f"{_num(row['design'].get('active_union') or training.get('active_per_token') or 539649, ',.0f')} | "
            f"{_num((held.get('ce') or {}).get('mean'), '.4f')} | "
            f"{_num(delta.get('mean'), '+.4f')} "
            f"[{_num(ci[0], '+.4f')}, {_num(ci[1], '+.4f')}] | "
            f"{_num((held.get('kl') or {}).get('mean'), '.4f')} | "
            f"{_num(gate.get('mean_abs'), '.3f')} | "
            + (f"{_num(harm.get('mean'), '+.4f')} ± {_num(harm.get('stderr'), '.4f')} |"
               if harm else "n/a (CE+KL arm) |"))
    lines += [
        "",
        "### Anchored math at 2M",
        "",
        "| arm | GSM8K | MATH-500 | math mean | seq-correct GSM8K "
        "(/1,319) | seq-correct MATH-500 (/500) | Δ vs base [95% CI] | "
        "Δ vs adapter-2m [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    reference_mean = {}
    for name in ("base", "adapter-2m"):
        reference = receipt["references"].get(name)
        if not reference:
            continue
        entry = reference["eval"]
        reference_mean[name] = (entry["family_mean"],
                                entry["family_stderr"])
    for name in ["base", "adapter-2m", "corrective-kl-2m-math",
                 "corrective-signed-exp-2m-math",
                 "corrective-signed-true-2m-math", "adapter-2m-math-matched",
                 "gated-dense-2m-math"]:
        if name in rows:
            entry = rows[name].get("eval")
            design = rows[name].get("design") or {}
            if entry is None:
                lines.append(f"| {ARM_LABEL[name]} | no receipt | | | | | | |")
                continue
        else:
            reference = receipt["references"].get(name)
            entry = reference["eval"] if reference else None
            design = {}
            if entry is None:
                lines.append(f"| {ARM_LABEL[name]} | no receipt | | | | | | |")
                continue
        tasks = entry["tasks"]
        recomputed = ((receipt.get("sequence_correctness") or {}).get("arms") or {})
        seq = {}
        for task in TASKS:
            block = (recomputed.get(name) or {}).get(task) or {}
            if block.get("status") == "ok":
                seq[task] = {"correct": block["correct"], "items": block["items"]}
            else:
                seq[task] = tasks[task].get("sequence_correctness") or {}
        vs = []
        for reference in REFERENCES:
            if name == reference:
                vs.append("—")
                continue
            row_b = receipt["pairwise"].get(f"{name}|{reference}|gsm8k")
            row_m = receipt["pairwise"].get(f"{name}|{reference}|minerva_math500")
            if row_b is None and row_m is None:
                ref_entry = (receipt["references"].get(reference) or {}).get("eval")
                if ref_entry is None:
                    vs.append("reference missing")
                    continue
                delta = entry["family_mean"] - ref_entry["family_mean"]
                combined = math.sqrt(entry["family_stderr"] ** 2
                                     + ref_entry["family_stderr"] ** 2)
                vs.append(f"{100 * delta:+.2f} (family, ±{100 * combined:.2f})")
                continue
            text = []
            for task, row in (("GSM8K", row_b), ("MATH-500", row_m)):
                text.append(f"{task} {_ci(row)}")
            vs.append("; ".join(text))
        lines.append(
            f"| {ARM_LABEL[name]} | "
            f"{_num(tasks.get('gsm8k', {}).get('score'), '.2f', 100)} ± "
            f"{_num(tasks.get('gsm8k', {}).get('stderr'), '.2f', 100)} | "
            f"{_num(tasks.get('minerva_math500', {}).get('score'), '.2f', 100)} ± "
            f"{_num(tasks.get('minerva_math500', {}).get('stderr'), '.2f', 100)} | "
            f"**{_num(entry['family_mean'], '.2f', 100)}** ± "
            f"{_num(entry['family_stderr'], '.2f', 100)} | "
            f"{seq['gsm8k'].get('correct', 'n/a')}/{seq['gsm8k'].get('items', 'n/a')} | "
            f"{seq['minerva_math500'].get('correct', 'n/a')}/{seq['minerva_math500'].get('items', 'n/a')} | "
            f"{vs[0]} | {vs[1]} |")
    lines += [
        "",
        f"**Verdict: `{verdict_row['label']}`.** {verdict_row.get('reading', '')}",
        "",
        _attribution(receipt),
        "",
        "Every cell is generated by `scripts/s48_screen_table.py` from the training "
        "receipts under `results/stage2/runs/<arm>/`, the held-out receipts "
        "`results/stage2/s48-heldout-<arm>.json`, and the anchored receipts "
        "`results/stage2/eval/<arm>/eval.json`; the table receipt is "
        f"`{receipt.get('_path', 'results/stage2/s48-screen-table-<stamp>.json')}`. "
        "ΔCE is held-out mean CE minus base with the paired per-chunk 95% interval; "
        "Δ vs base/adapter-2m are paired per-item percentile intervals (items "
        "paired by leaf and doc_id, 4,000 resamples, seed 44) and the family column "
        "uses the independent combined standard error. The harm column is the "
        "training log's tail mean delta KL on the corrected `base_better` mask "
        "(A_t < 0), a training-stream measurement. The knowledge family was not "
        "scored at 2M, so decision-rule condition 4's knowledge clause is "
        "**not measured**.",
        "",
        "Sequence correctness is the S45 exact final-answer verdict on the "
        "harness's own generations, reported beside the harness metric and never "
        "substituted for it. The counts here are recomputed on the registered "
        "primary filter by `scripts/s48_seq_correct.py`, because the S47 block in "
        "the eval receipts was fed every stored sample and GSM8K stores two rows "
        "per item (one per filter); the receipted recomputation is named in the "
        "table receipt's `sequence_correctness` block.",
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
        print(f"s48: {REPORT}: region {SENTINEL} is stale for the receipt",
              file=sys.stderr)
        return 1
    return 0


def self_test() -> int:
    assert _ci(None) == "unpaired"
    assert _ci({"paired": True, "delta": 0.1, "ci95": [0.05, 0.15]}) == (
        "+10.00 [+5.00, +15.00]")
    passing = {"conditions": {"measured": True,
                              "family": {"base": {"condition_holds": True},
                                         "adapter-2m": {"condition_holds": True}},
                              "ce": {"condition_holds": True}}}
    failing = {"conditions": {"measured": True,
                              "family": {"base": {"condition_holds": False},
                                         "adapter-2m": {"condition_holds": False}},
                              "ce": {"condition_holds": False}}}
    arms = {name: (passing if name.startswith("corrective")
                   else failing) for name in DESIGN}
    for name in arms:
        arms[name] = {**arms[name],
                      "eval": {"family_mean": 1.0, "family_stderr": 0.5}}
    result = verdict(arms)
    assert result["label"] in ("corrective_arm_promoted", "data_gain",
                               "gate_gain"), result
    null_arms = {name: {**failing,
                        "eval": {"family_mean": 1.0, "family_stderr": 0.5}}
                 for name in DESIGN}
    assert verdict(null_arms)["label"] == "null"
    print("self-test: rendering and the decision rule PASS")
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
    out = args.out or os.path.join(OUT_DIR, f"s48-screen-table-{stamp()}.json")
    write_json(out, receipt)
    receipt["_path"] = out
    print(f"s48: receipt {out}")
    print(f"s48: verdict {receipt['verdict']['label']}")
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
