#!/usr/bin/env python3
"""Score a stage-2 arm under the anchor pre-registration (WS19 Job 1).

The anchor scoreboard (`scripts/s36_anchor.py run`) drives the pinned harness
from a model directory. The distilled control is not a directory: it is the
anchor checkpoint plus a residual adapter trained in-process. This driver keeps
the *instrument* identical — the same harness commit, task list, few-shot
counts, sample limits, batch sizes, greedy generation, seeds, bootstrap iters
and extraction filters, all read from `docs/anchor-preregistration-v1.json` —
and swaps only the model object, which is handed to `HFLM(pretrained=<model>)`.

It refuses to run if the harness commit differs from the pre-registration, if a
task is not registered, or if the adapter checkpoint is missing. It writes the
receipt after every task, so an interrupted run says which tasks it finished.

Scored tasks are the four registered family members (MMLU-Pro, MMLU, GSM8K,
MATH-500). `humaneval`, `ifeval`, `arc_challenge` and `triviaqa` are registered
in the anchor scoreboard but belong to neither family the win bar is defined
over; they are not scored here and the receipt says so.

Usage:
    scripts/court --gpu eval -- python3 scripts/s39_control_eval.py \
        --checkpoint results/stage2/runs/control-20m/ckpt-009765.pt \
        --out results/stage2/eval/control-20m/eval.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

import s36_anchor as anchor  # noqa: E402  (shared instrument helpers)

DEFAULT_PREREG = "docs/anchor-preregistration-v1.json"
DEFAULT_AMENDMENT = "docs/ws19-distillation-control-amendment-preregistration-v1.json"
FAMILY_TASKS = ("mmlu_pro", "mmlu", "gsm8k", "minerva_math500")
SCHEMA = "selvedge/stage2-control-eval/v1"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def n_of(entry: dict) -> int | None:
    samples = entry.get("n_samples") or {}
    return samples.get("sample_len") or samples.get("sample_count")


def cgroup_peak() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    return report


def build_model(checkpoint: str, hidden: int, site: int, device: str):
    import torch

    from frontier import arms, memory

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("arm") != "adapter":
        raise SystemExit(f"{checkpoint}: arm is {payload.get('arm')!r}, not 'adapter'")
    model = arms.load_model(anchor_model_dir(), dtype=torch.bfloat16, device=device)
    module = arms.DenseAdapter(hidden=hidden).to(device)
    module.load_state_dict(payload["arm_state"], strict=True)
    handles = arms.install_adapter(model, module, (site,))
    report = module.active_parameter_report()
    report["checkpoint"] = checkpoint
    report["checkpoint_sha256"] = anchor.sha256_file(checkpoint)
    report["checkpoint_step"] = payload.get("step")
    report["checkpoint_tokens_seen"] = payload.get("tokens_seen")
    report["checkpoint_config_sha256"] = payload.get("config_sha256")
    return model, module, handles, report


def anchor_model_dir() -> str:
    from frontier import arms  # noqa: F401  (keeps the import explicit)

    cache = os.environ.get("HF_HOME", "/hf-cache")
    return os.path.join(cache, "models", "Qwen", "Qwen3.5-0.8B-Base",
                        "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


def run(args) -> int:
    import torch
    import lm_eval
    import transformers
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    prereg = anchor.load_json(args.prereg)
    if prereg.get("schema") != anchor.SCHEMA_PREREG:
        raise SystemExit(f"{args.prereg}: unexpected schema")
    prereg_digest = anchor.sha256_file(args.prereg)
    amendment = anchor.load_json(args.amendment)
    amendment_digest = anchor.sha256_file(args.amendment)
    if amendment.get("evaluation", {}).get("instrument_sha256") != prereg_digest:
        raise SystemExit(f"{args.amendment} names a different instrument than {args.prereg}")

    commit, version = anchor.harness_commit()
    if commit != prereg["harness"]["commit"]:
        raise SystemExit(f"harness commit {commit} != pre-registered {prereg['harness']['commit']}")

    declared = {task["name"]: task for task in prereg["tasks"]}
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    for name in tasks:
        if name not in declared:
            raise SystemExit(f"task {name!r} is not in the pre-registration")
        if name not in FAMILY_TASKS:
            raise SystemExit(f"task {name!r} is not a registered family member")

    if not torch.cuda.is_available():
        raise SystemExit("s39_control_eval: CUDA required; run through scripts/court --gpu eval")
    device = "cuda"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * (1 << 30) > free_bytes:
        raise SystemExit(f"vram limit {args.vram_limit_gib} GiB exceeds free "
                         f"{free_bytes / (1 << 30):.2f} GiB")

    model, module, handles, arm_report = build_model(
        args.checkpoint, args.hidden, args.site, device)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=anchor_model_dir(),
              batch_size=1, max_batch_size=None,
              add_bos_token=False, trust_remote_code=False,
              dtype="bfloat16", device=device)

    exceptions = {}
    for name in tasks:
        task = declared[name]
        limit = args.limit if args.limit is not None else task.get("limit")
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        lm.batch_size_per_gpu = int(task.get("batch_size", 1))
        print(f"s39_control_eval: {name} limit={limit} batch={lm.batch_size_per_gpu}", flush=True)
        result = simple_evaluate(
            model=lm,
            tasks=[name],
            num_fewshot=task["num_fewshot"],
            batch_size=task.get("batch_size", "auto"),
            max_batch_size=prereg.get("max_batch_size"),
            device=device,
            limit=limit,
            log_samples=True,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            gen_kwargs=prereg.get("gen_kwargs"),
            confirm_run_unsafe_code=True,
            bootstrap_iters=prereg.get("bootstrap_iters", 100000),
            random_seed=prereg["seed"]["random_seed"],
            numpy_random_seed=prereg["seed"]["numpy"],
            torch_random_seed=prereg["seed"]["torch"],
            fewshot_random_seed=prereg["seed"]["fewshot"],
        )
        if result is None:
            raise SystemExit(f"{name}: the harness returned no results")
        elapsed = time.time() - started
        metrics = result["results"][name]
        metric_key, stderr_key = anchor.resolve_metric(metrics, task["primary_metric"])
        score = metrics[metric_key]
        configs = result.get("configs") or {}
        harness_config = anchor.condense_config(configs.get(name))
        if harness_config is None:
            harness_config = {
                "group": True,
                "subtasks": {key: anchor.condense_config(value)
                             for key, value in configs.items()
                             if key == name or key.startswith(name)},
            }
        peak = torch.cuda.max_memory_allocated()
        if peak / (1 << 30) > args.vram_limit_gib:
            raise RuntimeError(f"{name}: VRAM peak {peak / (1 << 30):.2f} GiB over limit")
        entry = {
            "task": name,
            "family": task["family"],
            "score": score,
            "stderr": metrics.get(stderr_key),
            "primary_metric": metric_key,
            "primary_metric_registered": task["primary_metric"],
            "all_metrics": metrics,
            "n_samples": {
                "sample_len": metrics.get("sample_len"),
                "sample_count": metrics.get("sample_count", {}).get(metric_key)
                if isinstance(metrics.get("sample_count"), dict) else None,
            },
            "pre_registered": {
                "limit": limit,
                "limit_from_preregistration": task.get("limit"),
                "num_fewshot": task["num_fewshot"],
                "batch_size": task.get("batch_size"),
            },
            "harness_config": harness_config,
            "wall_clock_s": round(elapsed, 1),
            "peak_memory_allocated_bytes": int(peak),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cgroup": cgroup_peak(),
            "zero_score": score == 0.0,
        }
        exceptions[name] = entry
        partial = base_receipt(args, prereg, prereg_digest, amendment,
                               amendment_digest, commit, version, arm_report,
                               free_bytes, total_bytes, exceptions)
        anchor.write_json(args.out, partial)
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{partial['stamp']}-{name}.json")
        anchor.write_json(raw_path, result)
        entry["raw_result"] = raw_path
        print(f"s39_control_eval: {name} {metric_key}={score:.6f} "
              f"stderr={entry['stderr']} wall={elapsed:.1f}s "
              f"peak={peak / (1 << 30):.2f}GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    receipt = base_receipt(args, prereg, prereg_digest, amendment,
                           amendment_digest, commit, version, arm_report,
                           free_bytes, total_bytes, exceptions)
    receipt["finished_utc"] = utcnow()
    receipt["complete"] = len(exceptions) == len(tasks)
    receipt["family_means"] = family_means(prereg, exceptions)
    anchor.write_json(args.out, receipt)
    print(f"s39_control_eval: receipt {args.out}", flush=True)
    for row in receipt["family_means"]:
        print(f"s39_control_eval: family {row['family']} mean={row['mean'] * 100:.2f} "
              f"anchor={row['anchor_mean'] * 100:.2f} "
              f"2b={row['upper_mean'] * 100:.2f} "
              f"delta_anchor={row['delta_vs_anchor_pp']:+.2f}pp "
              f"target={row['anchor_plus_half_gap'] * 100:.2f}", flush=True)
    handles and anchor_module_uninstall(handles)
    return 0


def anchor_module_uninstall(handles) -> None:
    from frontier import arms
    arms.uninstall(handles)


def base_receipt(args, prereg, prereg_digest, amendment, amendment_digest,
                 commit, version, arm_report, free_bytes, total_bytes, exceptions) -> dict:
    import torch

    return {
        "schema": SCHEMA,
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": anchor.git_revision(),
        "model": {
            "id": "Qwen/Qwen3.5-0.8B-Base",
            "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "role": "distilled control: frozen core + parameter-matched dense adapter",
            "directory": anchor_model_dir(),
        },
        "preregistration": {"path": args.amendment, "sha256": amendment_digest},
        "harness_preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "harness": {
            "name": prereg["harness"]["name"],
            "commit": commit,
            "version": version,
            "image_ref": os.environ.get("SELVEDGE_COURT_IMAGE", "unknown"),
            "image_id": os.environ.get("SELVEDGE_COURT_IMAGE_ID", "unknown"),
        },
        "court": {
            "name": os.environ.get("SELVEDGE_COURT", "unknown"),
            "memory_limit": os.environ.get("SELVEDGE_COURT_MEM", "unknown"),
            "memory_limit_bytes": os.environ.get("SELVEDGE_COURT_MEM_BYTES", "unknown"),
            "cpus": os.environ.get("SELVEDGE_COURT_CPUS", "unknown"),
            "pids_limit": os.environ.get("SELVEDGE_COURT_PIDS", "unknown"),
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset"),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(total_bytes),
            "free_at_start_bytes": int(free_bytes),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "arm": arm_report,
        "scored_tasks": sorted(exceptions),
        "unscored_registered_tasks": [
            task["name"] for task in prereg["tasks"]
            if task["name"] not in exceptions
        ],
        "unscored_reason": (
            "the four scored tasks are the registered family members that define "
            "the knowledge and math means; the other registered tasks belong to "
            "neither family the win bar is defined over"
        ),
        "prompting": prereg["prompting"],
        "model_object": "pre-initialised transformers model + residual adapter handed to HFLM",
        "argv": sys.argv,
        "tasks": [exceptions[name] for name in sorted(exceptions)],
    }


def family_means(prereg, entries: dict) -> list[dict]:
    reference = load_reference()
    by_task = {}
    for row in reference.get("tasks", []):
        by_task[row["task"]] = row
    rows = []
    for family in prereg["families"]:
        expected = family.get("members") or []
        members = [name for name in expected if name in entries]
        complete = sorted(members) == sorted(expected)
        if not members:
            continue
        scores = [entries[name]["score"] for name in members]
        stderrs = [entries[name]["stderr"] for name in members if entries[name]["stderr"] is not None]
        mean = sum(scores) / len(scores)
        combined = (math.sqrt(sum(value ** 2 for value in stderrs)) / len(stderrs)
                    if stderrs else None)
        anchor_scores = [by_task[name]["anchor_score"] for name in members
                         if name in by_task]
        upper_scores = [by_task[name]["upper_score"] for name in members
                        if name in by_task]
        anchor_mean = sum(anchor_scores) / len(anchor_scores) if anchor_scores else None
        upper_mean = sum(upper_scores) / len(upper_scores) if upper_scores else None
        row = {
            "family": family["name"],
            "members": members,
            "registered_members": expected,
            "complete": complete,
            "aggregation": "unweighted mean of member scores",
            "mean": mean,
            "combined_stderr_independent": combined,
            "combined_stderr_note": (
                "sqrt(sum(task stderr^2)) / n; treats tasks as independent, which "
                "understates a family SE because the same items are shared"
            ),
            "anchor_mean": anchor_mean,
            "upper_mean": upper_mean,
            "delta_vs_anchor_pp": ((mean - anchor_mean) * 100.0
                                   if anchor_mean is not None else None),
            "delta_vs_upper_pp": ((mean - upper_mean) * 100.0
                                  if upper_mean is not None else None),
            "anchor_plus_half_gap": (
                anchor_mean + (upper_mean - anchor_mean) / 2.0
                if anchor_mean is not None and upper_mean is not None else None),
            "reference": "results/frontier/anchor-gap-20261116T225500Z.json",
        }
        row["reaches_half_gap"] = (
            complete and row["anchor_plus_half_gap"] is not None
            and mean >= row["anchor_plus_half_gap"])
        rows.append(row)
    return rows


def load_reference() -> dict:
    path = os.path.join(REPO, "results", "frontier", "anchor-gap-20261116T225500Z.json")
    if not os.path.isfile(path):
        return {}
    return anchor.load_json(path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prereg", default=DEFAULT_PREREG)
    parser.add_argument("--amendment", default=DEFAULT_AMENDMENT)
    parser.add_argument("--tasks", default=",".join(FAMILY_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--site", type=int, default=2)
    parser.add_argument("--vram-limit-gib", type=float, default=11.0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
