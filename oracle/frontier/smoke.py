#!/usr/bin/env python3
"""Aggregate the stage-2 smoke and register the stage-2 execution plan.

`collect` reads one evaluation receipt per arm plus the training receipts and
emits one flat smoke receipt. It computes deltas in percentage points against
the frozen base, a combined standard error for each pair, and the three signal
questions the mission asks:

* did any arm move on any task beyond its own noise?
* is the memory arm's knowledge delta distinguishable from the training-free
  NGM control and from the parameter-matched dense adapter?
* does any arm sit more than one point below the base on any task?

`plan` writes the execution plan. Every budget number is **derived at run time**
from the measured throughput in the smoke receipt: tokens / (tokens per second
x 3600) is GPU-hours, not a hand-typed estimate. The registered full-budget
token count is 20,000,000 per arm per seed (ten times the smoke), three seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

SMOKE_TOKENS = 2_000_000
FULL_TOKENS_PER_ARM_SEED = 20_000_000
SEEDS = (0, 1, 2)
EVAL_EVERY_TOKENS = 5_000_000
TASKS = ("mmlu_pro", "gsm8k", "mmlu")
KNOWLEDGE = ("mmlu_pro", "mmlu")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(HERE)), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def binomial_stderr(score: float, n: int | None) -> float | None:
    if not n:
        return None
    return math.sqrt(max(0.0, score * (1.0 - score)) / n)


def combined_stderr(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(a * a + b * b)


def score_map(receipt: dict) -> dict:
    out = {}
    for entry in receipt["tasks"]:
        stderr = entry.get("stderr")
        if stderr is None:
            stderr = binomial_stderr(entry["score"], entry.get("n_samples"))
        out[entry["task"]] = {
            "score": entry["score"],
            "stderr": stderr,
            "n_samples": entry.get("n_samples"),
            "wall_seconds": entry.get("wall_seconds"),
            "peak_allocated_gib": entry.get("peak_allocated_gib"),
            "metric": entry["primary_metric"],
            "receipt": receipt,
        }
    return out


def collect(args) -> int:
    base = json.load(open(args.base))
    base_scores = score_map(base)
    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        receipt = json.load(open(path))
        scores = score_map(receipt)
        missing = [task for task in base_scores if task not in scores]
        if missing:
            raise SystemExit(f"collect: {name} misses tasks {missing}")
        tasks = {}
        for task in base_scores:
            delta = (scores[task]["score"] - base_scores[task]["score"]) * 100.0
            se = combined_stderr(scores[task]["stderr"],
                                 base_scores[task]["stderr"])
            tasks[task] = {
                "score": scores[task]["score"],
                "base_score": base_scores[task]["score"],
                "delta_pp": delta,
                "stderr_pp": (None if scores[task]["stderr"] is None
                              else scores[task]["stderr"] * 100.0),
                "combined_stderr_pp": None if se is None else se * 100.0,
                "n_samples": scores[task]["n_samples"],
                "metric": scores[task]["metric"],
            }
        knowledge = [tasks[t]["delta_pp"] for t in KNOWLEDGE if t in tasks]
        arms[name] = {
            "eval_receipt": path,
            "arm_report": receipt.get("arm_report"),
            "tasks": tasks,
            "knowledge_mean_delta_pp": sum(knowledge) / len(knowledge),
            "worst_task_delta_pp": min(t["delta_pp"] for t in tasks.values()),
            "tasks_more_than_one_point_lower": [
                task for task, entry in tasks.items() if entry["delta_pp"] < -1.0],
            "peak_allocated_gib": max(
                (entry.get("peak_allocated_gib") or 0.0)
                for entry in receipt["tasks"]),
            "eval_wall_seconds": receipt.get("wall_seconds"),
        }

    def distinguishable(a: str, b: str, task: str) -> dict:
        if a not in arms or b not in arms:
            return {"available": False}
        ea, eb = arms[a]["tasks"][task], arms[b]["tasks"][task]
        delta = ea["delta_pp"] - eb["delta_pp"]
        se = combined_stderr(ea["stderr_pp"], eb["stderr_pp"])
        return {
            "available": True,
            "delta_pp": delta,
            "combined_stderr_pp": se,
            "beyond_1_stderr": None if se in (None, 0) else abs(delta) / se > 1.0,
        }

    trained = {}
    for spec in args.train or []:
        name, path = spec.split("=", 1)
        receipt = json.load(open(path))
        trained[name] = {
            "receipt": path,
            "tokens": receipt["tokens_trained"],
            "tokens_per_second": receipt["tokens_per_second"],
            "peak_allocated_gib": receipt["peak_allocated_gib"],
            "peak_reserved_gib": receipt["peak_reserved_gib"],
            "wall_seconds": receipt["wall_seconds"],
            "config": receipt["config"],
            "container_memory": receipt.get("container_memory"),
            "arm_accounting": receipt.get("arm_accounting"),
        }

    signal = {}
    for name in arms:
        if name == "base":
            continue
        signal[name] = {
            "knowledge_mean_delta_pp": arms[name]["knowledge_mean_delta_pp"],
            "moved_on_any_task_beyond_1_stderr": any(
                entry["combined_stderr_pp"] not in (None, 0)
                and abs(entry["delta_pp"]) / entry["combined_stderr_pp"] > 1.0
                for entry in arms[name]["tasks"].values()),
            "no_task_more_than_one_point_lower":
                not arms[name]["tasks_more_than_one_point_lower"],
        }
    comparisons = {}
    for task in TASKS:
        comparisons[task] = {
            "memory_vs_ngm": distinguishable("memory-trained", "ngm", task),
            "memory_vs_adapter": distinguishable("memory-trained", "adapter", task),
            "memory_graft_vs_base": distinguishable("memory-graft", "base", task),
        }

    receipt = {
        "schema": "selvedge/stage2-smoke/v1",
        "kind": "collect",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "harness": base.get("harness"),
        "base_receipt": args.base,
        "arms": arms,
        "trained": trained,
        "signal": signal,
        "comparisons": comparisons,
        "limits": {
            "smoke_tokens_per_arm": SMOKE_TOKENS,
            "full_budget_tokens_per_arm_seed": FULL_TOKENS_PER_ARM_SEED,
            "seeds": list(SEEDS),
            "eval_every_tokens": EVAL_EVERY_TOKENS,
        },
        "graft": json.load(open(args.graft)) if args.graft else None,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"stage2-smoke-{receipt['stamp']}.json")
    write_json(out, receipt)
    print(f"collect: {out}")
    for name, entry in arms.items():
        print(f"  {name:<16} knowledge_delta={entry['knowledge_mean_delta_pp']:+.2f}pp "
              f"worst={entry['worst_task_delta_pp']:+.2f}pp")
    for name, entry in trained.items():
        print(f"  {name:<16} {entry['tokens_per_second']:.0f} tok/s "
              f"peak={entry['peak_allocated_gib']:.2f}GiB")
    return 0


def plan(args) -> int:
    smoke = json.load(open(args.smoke))
    trained = smoke["trained"]
    if "memory" not in trained:
        raise SystemExit("plan: smoke receipt has no memory training measurement")
    tps_memory = trained["memory"]["tokens_per_second"]
    tps_adapter = trained.get("adapter", {}).get("tokens_per_second")
    eval_wall = 0.0
    for arm in smoke["arms"].values():
        eval_wall = max(eval_wall, arm.get("eval_wall_seconds") or 0.0)
    print(f"plan: measured memory {tps_memory:.1f} tok/s, eval {eval_wall:.0f}s")

    per_arm_seed = FULL_TOKENS_PER_ARM_SEED
    runs_a = [("memory", seed) for seed in SEEDS] + \
             [("adapter", seed) for seed in SEEDS]
    train_hours_a = sum(per_arm_seed / (
        tps_memory if arm == "memory" else tps_adapter) for arm, _ in runs_a
    ) / 3600.0
    evals_per_run = max(1, FULL_TOKENS_PER_ARM_SEED // EVAL_EVERY_TOKENS)
    eval_hours_a = len(runs_a) * evals_per_run * eval_wall / 3600.0
    runs_b = [("memory-1m", seed) for seed in SEEDS]
    train_hours_b = len(runs_b) * per_arm_seed / tps_memory / 3600.0
    eval_hours_b = len(runs_b) * evals_per_run * eval_wall / 3600.0

    receipt = {
        "schema": "selvedge/stage2-plan/v1",
        "kind": "plan",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "smoke_receipt": args.smoke,
        "smoke_tokens_per_arm": SMOKE_TOKENS,
        "win_condition": {
            "registered": "grafted memory beats the 0.8B-Base score on the "
                          "knowledge family with no task more than 1 point "
                          "lower, and beats both the training-free NGM baseline "
                          "and the parameter-matched dense adapter at equal "
                          "unique active",
            "first_configuration": "65,536 rows, one site after decoder layer 2, "
                                   "8 reads/token, teacher 2B-Base, online KL "
                                   "T=2.0 lambda=0.5, memory-only training",
        },
        "phase_a_development_scale": {
            "rows": 65_536,
            "sites": [2],
            "tokens_per_arm_seed": per_arm_seed,
            "seeds": list(SEEDS),
            "trainable_arms": ["memory", "adapter"],
            "runs": [f"{arm}-seed{seed}" for arm, seed in runs_a],
            "measured_tokens_per_second": {
                "memory": tps_memory,
                "adapter": tps_adapter,
            },
            "train_gpu_hours": train_hours_a,
            "eval_cadence_tokens": EVAL_EVERY_TOKENS,
            "evals_per_run": evals_per_run,
            "measured_eval_wall_seconds": eval_wall,
            "eval_gpu_hours": eval_hours_a,
            "total_gpu_hours": train_hours_a + eval_hours_a,
            "arithmetic": "sum(tokens / measured_tokens_per_second) / 3600; "
                          "eval wall from the smoke receipt",
        },
        "phase_b_scale_question_1m": {
            "rows": 1_048_576,
            "tokens_per_arm_seed": per_arm_seed,
            "seeds": list(SEEDS),
            "arms": ["memory"],
            "train_gpu_hours": train_hours_b,
            "eval_gpu_hours": eval_hours_b,
            "total_gpu_hours": train_hours_b + eval_hours_b,
            "blocker": "dense AdamW state for the 1M-row table is 3x the table "
                       "in float32 (about 3 GiB); the training loop must switch "
                       "to sparse row updates before this phase runs",
        },
        "phase_c_scale_question_8m": {
            "rows": 8_388_608,
            "status": "not runnable on this device",
            "resident_table_gib_bf16": 4.0,
            "why": "training the table needs the table plus optimiser state; "
                   "serving needs the NVMe path of stage 5 (P19.5)",
        },
        "next_route_if_first_configuration_misses": [
            "site placement: repeat phase A at decoder layer 6 and layer 11 "
            "(and the two-site pair), same tokens and seeds",
            "4B teacher: the 4B-Base checkpoint is not cached; the cache "
            "filesystem has room for 9.3 GB plus 1 GiB headroom (recorded in "
            "the graft receipt), so fetch it and repeat the graft",
            "larger table: 1,048,576 rows after the sparse-update change",
            "LoRA on the core: rank 8 on the linear-attention projections, "
            "after the memory-only phase, with the frozen-core control",
        ],
        "kill_rule": "only if the best admission policy loses to both the dense "
                     "adapter and the training-free NGM at all three row scales "
                     "with the placement route measured (PROGRAM-PLAN P19.2)",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"stage2-plan-{receipt['stamp']}.json")
    write_json(out, receipt)
    print(f"plan: {out}")
    print(f"plan: phase A {receipt['phase_a_development_scale']['total_gpu_hours']:.2f} "
          f"GPU-hours, phase B {receipt['phase_b_scale_question_1m']['total_gpu_hours']:.2f}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--base", required=True)
    c.add_argument("--arm", action="append", required=True)
    c.add_argument("--train", action="append")
    c.add_argument("--graft")
    c.add_argument("--out-dir", default="results/stage2")
    p = sub.add_parser("plan")
    p.add_argument("--smoke", required=True)
    p.add_argument("--out-dir", default="results/stage2")
    args = parser.parse_args(argv)
    if args.command == "collect":
        return collect(args)
    return plan(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
