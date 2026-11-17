#!/usr/bin/env python3
"""S49: the stability sweep's table, its registered decision, and the receipt.

The registration
(`docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json`)
fixes three learning rates, the 2M-token budget, the early-loss tripwire and the
decision rule. This script reads each arm's training log, training receipt (or
abort record) and held-out receipt, evaluates the rule, and writes one decision
receipt. Nothing is typed by hand: every cell is a number from a file, and the
chosen arm is the rule applied to those numbers.

The decision rule, verbatim from the registration: the chosen recipe is the
highest learning rate among the arms that (a) never tripped the tripwire and has
a trailing-50 median CE at 2M below 1.5x its first-100-step median and (b) has a
held-out mean CE at most base + 0.01 nats with the paired per-chunk delta
reported beside it.

Usage:
    scripts/court python -- python3 scripts/s49_stability_table.py --write
    scripts/court python -- python3 scripts/s49_stability_table.py --check
    scripts/court python -- python3 scripts/s49_stability_table.py --summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCHEMA = "selvedge/s49-stability-decision/v1"
REGISTRATION = "docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
REPORT = "results/REPORT.md"
SENTINEL = "report-s49-stability"
OUT_DIR = "results/stage2"
HELDOUT_FLOOR_NATS = 0.01
TRIPWIRE_FACTOR = 1.5
BASELINE_STEPS = 100
WINDOW_STEPS = 50


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_log(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def log_stats(rows: list[dict]) -> dict:
    ce = [float(row["ce"]) for row in rows]
    if not ce:
        return {"steps_logged": 0}
    stats = {
        "steps_logged": len(ce),
        "tokens_seen": int(rows[-1].get("tokens_seen") or 0),
        "median_ce_first_100": None,
        "median_ce_trailing_50": None,
        "trailing_ratio": None,
        "max_trailing_ratio": None,
        "first_step_trailing_over_factor": None,
        "max_ce": max(ce),
        "median_ce_by_1000_step_block": [],
    }
    for start in range(0, len(ce), 1000):
        block = ce[start:start + 1000]
        if block:
            stats["median_ce_by_1000_step_block"].append(
                {"steps": f"{start + 1}-{start + len(block)}",
                 "median": round(statistics.median(block), 6)})
    if len(ce) >= BASELINE_STEPS:
        baseline = statistics.median(ce[:BASELINE_STEPS])
        stats["median_ce_first_100"] = baseline
        if len(ce) >= BASELINE_STEPS + WINDOW_STEPS:
            trailing = statistics.median(ce[-WINDOW_STEPS:])
            stats["median_ce_trailing_50"] = trailing
            stats["trailing_ratio"] = trailing / baseline
            ratios = []
            first_over = None
            for end in range(BASELINE_STEPS + WINDOW_STEPS, len(ce) + 1):
                ratio = statistics.median(ce[end - WINDOW_STEPS:end]) / baseline
                ratios.append(ratio)
                if first_over is None and ratio > TRIPWIRE_FACTOR:
                    first_over = end
            stats["max_trailing_ratio"] = max(ratios) if ratios else None
            stats["first_step_trailing_over_factor"] = first_over
    return stats


def heldout_stats(path: str, arm: str) -> dict | None:
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    entry = (payload.get("arms") or {}).get(arm)
    base = (payload.get("arms") or {}).get("base")
    if not entry or not base:
        return None
    delta = entry.get("delta_ce_vs_reference") or {}
    return {
        "receipt": path,
        "ce_mean": entry["ce"]["mean"],
        "ce_stderr": entry["ce"]["stderr"],
        "kl_mean": entry["kl"]["mean"],
        "base_ce_mean": base["ce"]["mean"],
        "delta_ce": delta.get("mean"),
        "delta_ce_ci95": delta.get("ci95"),
        "chunks": entry.get("chunks"),
    }


def collect(arm_entry: dict) -> dict:
    name = arm_entry["name"]
    run_dir = arm_entry["run_dir"]
    log_path = os.path.join(run_dir, "log.jsonl")
    receipt_path = os.path.join(run_dir, "receipt.json")
    abort_path = os.path.join(run_dir, "abort.json")
    heldout_path = os.path.join(OUT_DIR, f"s49-heldout-{name}.json")

    row = {
        "arm": name,
        "lr": float(arm_entry["lr"]),
        "run_dir": run_dir,
        "log": log_path,
        "training_receipt": receipt_path if os.path.exists(receipt_path) else None,
        "abort": abort_path if os.path.exists(abort_path) else None,
        "heldout": heldout_stats(heldout_path, name),
        "complete": os.path.exists(receipt_path),
        "aborted": os.path.exists(abort_path),
        "tokens_per_second": None,
        "gpu_peak_gib": None,
        "cgroup_peak_gib": None,
        "tripwire": None,
    }
    if os.path.exists(log_path):
        rows = read_log(log_path)
        row["log_stats"] = log_stats(rows)
        if rows:
            row["tokens_per_second"] = round(
                statistics.median([r["tokens_per_second"] for r in rows
                                   if r.get("tokens_per_second")]), 1)
            row["gpu_peak_gib"] = max(r.get("peak_allocated_gib") or 0 for r in rows)
    else:
        row["log_stats"] = {"steps_logged": 0}
    if os.path.exists(receipt_path):
        receipt = load_json(receipt_path)
        memory = receipt.get("container_memory") or {}
        peak = memory.get("memory_peak") or memory.get("peak")
        if isinstance(peak, str) and peak.isdigit():
            peak = int(peak)
        if isinstance(peak, int) and peak > 0:
            row["cgroup_peak_gib"] = round(peak / (1 << 30), 3)
        row["tripwire"] = receipt.get("tripwire")
    if os.path.exists(abort_path):
        abort = load_json(abort_path)
        row["tripwire"] = {
            "factor": abort.get("factor"),
            "baseline_median_ce": abort.get("baseline_median_ce"),
            "trailing_median_ce": abort.get("trailing_median_ce"),
            "ratio": abort.get("ratio"),
            "step": abort.get("step"),
            "tokens_seen": abort.get("tokens_seen"),
            "fired": True,
        }
    return row


def evaluate(row: dict) -> dict:
    stats = row.get("log_stats") or {}
    reasons = []
    stable = False
    if row["aborted"]:
        reasons.append("tripped the early-loss tripwire")
    elif not row["complete"]:
        reasons.append("no training receipt (arm incomplete)")
    else:
        ratio = stats.get("trailing_ratio")
        if ratio is None:
            reasons.append("training log is too short to evaluate the rule")
        elif ratio >= TRIPWIRE_FACTOR:
            reasons.append("trailing-50 median CE is not below 1.5x the first-100 median")
        else:
            stable = True
    heldout = row.get("heldout")
    preserved = False
    if not stable:
        pass
    elif heldout is None:
        reasons.append("no held-out receipt")
    elif heldout.get("delta_ce") is None:
        reasons.append("held-out receipt has no paired delta against base")
    elif heldout["delta_ce"] > HELDOUT_FLOOR_NATS:
        reasons.append("held-out CE is worse than base + 0.01 nats")
    else:
        preserved = True
    return {
        "stable": stable,
        "heldout_preserved": preserved,
        "eligible": bool(stable and preserved),
        "reasons": reasons,
    }


def decide(rows: list[dict]) -> dict:
    for row in rows:
        row["evaluation"] = evaluate(row)
    eligible = [row for row in rows if row["evaluation"]["eligible"]]
    eligible.sort(key=lambda row: row["lr"], reverse=True)
    chosen = eligible[0] if eligible else None
    return {
        "chosen": chosen,
        "eligible": [row["arm"] for row in eligible],
        "second_factor_required": chosen is None,
    }


def render(rows: list[dict], decision: dict) -> str:
    def num(value, digits=4):
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "### Full-backbone stability at 2M: the lr sweep",
        "",
        "| arm | lr | steps | tokens | median CE first 100 | median CE trailing 50 | trailing ratio | max trailing ratio | tripwire | held-out CE | ΔCE vs base [95% CI] | held-out KL | tok/s | GPU peak GiB | cgroup peak GiB | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        stats = row.get("log_stats") or {}
        heldout = row.get("heldout") or {}
        delta = heldout.get("delta_ce")
        ci = heldout.get("delta_ce_ci95")
        delta_text = "n/a" if delta is None else f"{delta:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        tripwire = "tripped at step %s" % row["tripwire"].get("step") if row["aborted"] else "not tripped"
        lines.append(
            f"| `{row['arm']}` | {row['lr']:.0e} | {stats.get('steps_logged', 0)} | "
            f"{stats.get('tokens_seen', 0)} | {num(stats.get('median_ce_first_100'))} | "
            f"{num(stats.get('median_ce_trailing_50'))} | {num(stats.get('trailing_ratio'), 3)} | "
            f"{num(stats.get('max_trailing_ratio'), 3)} | {tripwire} | "
            f"{num(heldout.get('ce_mean'))} | {delta_text} | {num(heldout.get('kl_mean'))} | "
            f"{num(row.get('tokens_per_second'), 1)} | {num(row.get('gpu_peak_gib'), 3)} | "
            f"{num(row.get('cgroup_peak_gib'), 3)} | "
            f"{'yes' if row['evaluation']['eligible'] else 'no'} |"
        )
    chosen = decision["chosen"]
    lines += [
        "",
        "### The registered decision",
        "",
        (
            f"The chosen recipe is `{chosen['arm']}` at lr **{chosen['lr']:.0e}**: it is the "
            "highest learning rate that never tripped the 1.5x tripwire, ended below it, "
            "and held held-out CE within base + 0.01 nats."
            if chosen else
            "No swept learning rate satisfies both registered conditions, so the "
            "second factor (grad-clip 0.5, and warmup 500 at the 10.24M-token budget "
            "where the trainer's warmup clamp does not truncate it) is required before "
            "the bar can be rerun."
        ),
    ]
    if not chosen:
        for row in rows:
            if row["evaluation"]["reasons"]:
                lines.append(
                    f"- `{row['arm']}`: {'; '.join(row['evaluation']['reasons'])}.")
    lines += [
        "",
        "Every cell is generated by `scripts/s49_stability_table.py` from the training "
        "logs and receipts under `results/stage2/runs/` and the held-out receipts "
        "`results/stage2/s49-heldout-<arm>.json`; the decision receipt under "
        "`results/stage2/` is the evidence. The tripwire is the registered rule of "
        "`oracle/frontier/train.py`: median CE over the first 100 optimiser steps "
        "against the median over the trailing 50, evaluated from step 150. The "
        "held-out slice is the registered 3,906-chunk stream, paired per chunk "
        "against base; ΔCE is held-out mean CE minus base with its paired 95% "
        "interval. `eligible` is the registered rule, not a preference.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# report region


BEGIN = "<!-- BEGIN GENERATED: %s -->" % SENTINEL
END = "<!-- END GENERATED: %s -->" % SENTINEL


def read_report(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def region_body(text: str, where: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"s49: {where} has no {BEGIN} / {END} pair")
    return text[start + len(BEGIN):end].strip("\n")


def write_region(path: str, body: str) -> bool:
    text = read_report(path)
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(f"s49: {path} has no {BEGIN} / {END} pair")
    updated = text[:start + len(BEGIN)] + "\n" + body + "\n" + text[end:]
    if updated == text:
        return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True


def build() -> tuple[list[dict], dict]:
    registration = load_json(os.path.join(REPO, REGISTRATION))
    rows = [collect(entry) for entry in registration["stability_sweep"]["arms"]]
    decision = decide(rows)
    return rows, decision


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.write, args.check, args.summary)) != 1:
        raise SystemExit("choose exactly one of --write, --check, --summary")

    rows, decision = build()
    body = render(rows, decision)
    report = os.path.join(REPO, REPORT)

    if args.summary:
        print(body)
        return 0
    if args.check:
        current = region_body(read_report(report), REPORT)
        if current != body:
            print("s49: the stability region is stale; run --write", file=sys.stderr)
            return 1
        print("s49: stability region matches the receipts")
        return 0

    chosen = decision["chosen"]
    receipt = {
        "schema": SCHEMA,
        "kind": "s49-stability-decision",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": {
            "path": REGISTRATION,
            "sha256": __import__("hashlib").sha256(
                open(os.path.join(REPO, REGISTRATION), "rb").read()).hexdigest(),
        },
        "rule": {
            "tripwire_factor": TRIPWIRE_FACTOR,
            "baseline_steps": BASELINE_STEPS,
            "window_steps": WINDOW_STEPS,
            "heldout_floor_nats_vs_base": HELDOUT_FLOOR_NATS,
            "registered_text": registration_text(),
        },
        "arms": rows,
        "eligible": decision["eligible"],
        "chosen": None if chosen is None else {
            "arm": chosen["arm"],
            "lr": chosen["lr"],
            "run_dir": chosen["run_dir"],
            "reason": ("highest lr that never tripped the tripwire, ended with a "
                       "trailing-50 median CE below 1.5x its first-100 median, and "
                       "held held-out CE within base + 0.01 nats"),
        },
        "second_factor_required": decision["second_factor_required"],
    }
    out = os.path.join(REPO, OUT_DIR, f"s49-stability-decision-{receipt['stamp']}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    changed = write_region(report, body)
    print(f"s49: decision receipt {out}")
    print(f"s49: {'wrote' if changed else 'kept'} the stability region in {REPORT}")
    if chosen is None:
        print("s49: no arm is eligible; the second factor is required", file=sys.stderr)
        return 2
    print(f"s49: chosen {chosen['arm']} lr={chosen['lr']}")
    return 0


def registration_text() -> str:
    registration = load_json(os.path.join(REPO, REGISTRATION))
    return registration["stability_sweep"]["decision_rule"]["registered_text"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
