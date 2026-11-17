#!/usr/bin/env python3
"""Score one S40 comparison arm under the anchor pre-registration.

This is the anchored-harness half of `scripts/s40_compare.py`. It keeps the
instrument identical to the S36 scoreboard and the Job-1 control: the same
harness commit, task list, few-shot counts, sample limits, batch sizes, greedy
generation, seeds, bootstrap iterations and extraction filters from
`docs/anchor-preregistration-v1.json`. Only the model object differs; it is
handed to `HFLM(pretrained=<model>)`.

Arm construction is shared with the held-out instrument (`s40_compare`), so the
same checkpoint builds the arm in both halves.

Usage:
    SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 scripts/s40_score.py \
        --arm full-distill \
        --full-checkpoint results/stage2/runs/full-20m/ckpt-009765.pt \
        --tasks mmlu_pro,mmlu,gsm8k,minerva_math500 \
        --out results/stage2/eval/full-distill/eval.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

import s36_anchor as anchor  # noqa: E402
import s40_compare as comparison  # noqa: E402

DEFAULT_PREREG = "docs/anchor-preregistration-v1.json"
FAMILY_TASKS = ("mmlu_pro", "mmlu", "gsm8k", "minerva_math500")
SCHEMA = "selvedge/stage2-arm-eval/v1"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cgroup_peak() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    return report


def receipt_base(args, prereg, prereg_digest, commit, version, arm_report,
                 free_bytes, total_bytes, entries) -> dict:
    import torch
    return {
        "schema": SCHEMA,
        "kind": "arm-eval",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": anchor.git_revision(),
        "arm": args.arm,
        "model": {
            "id": "Qwen/Qwen3.5-0.8B-Base",
            "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "directory": args.student_dir,
        },
        "harness_preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "harness": {
            "name": prereg["harness"]["name"],
            "commit": commit,
            "version": version,
            "image_ref": os.environ.get("SELVEDGE_COURT_IMAGE", "unknown"),
            "image_id": os.environ.get("SELVEDGE_COURT_IMAGE_ID", "unknown"),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(total_bytes),
            "free_at_start_bytes": int(free_bytes),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "arm_report": arm_report,
        "scored_tasks": sorted(entries),
        "tasks": [entries[name] for name in sorted(entries)],
        "complete": False,
        "argv": sys.argv,
    }


def run(args) -> int:
    import lm_eval
    import torch
    import transformers
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    prereg = anchor.load_json(args.prereg)
    if prereg.get("schema") != anchor.SCHEMA_PREREG:
        raise SystemExit(f"{args.prereg}: unexpected schema")
    prereg_digest = anchor.sha256_file(args.prereg)
    commit, version = anchor.harness_commit()
    if commit != prereg["harness"]["commit"]:
        raise SystemExit(f"harness commit {commit} != pre-registered "
                         f"{prereg['harness']['commit']}")
    declared = {task["name"]: task for task in prereg["tasks"]}
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    for name in tasks:
        if name not in declared:
            raise SystemExit(f"task {name!r} is not pre-registered")
        if name not in FAMILY_TASKS:
            raise SystemExit(f"task {name!r} is not a registered family member")

    if not torch.cuda.is_available():
        raise SystemExit("s40_score: CUDA required; run through scripts/court --gpu eval")
    device = "cuda"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * (1 << 30) > free_bytes:
        raise SystemExit(f"vram limit {args.vram_limit_gib} GiB exceeds free "
                         f"{free_bytes / (1 << 30):.2f} GiB")

    model = comparison.load_backbone(args, device)
    model.eval()
    model, handles, arm_report = comparison.build_named_arm(
        args.arm, model, args, device)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=args.student_dir,
              batch_size=1, max_batch_size=None,
              add_bos_token=False, trust_remote_code=False,
              dtype="bfloat16", device=device)

    entries = {}
    for name in tasks:
        task = declared[name]
        limit = args.limit if args.limit is not None else task.get("limit")
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        lm.batch_size_per_gpu = int(task.get("batch_size", 1))
        print(f"s40_score: {args.arm} {name} limit={limit} "
              f"batch={lm.batch_size_per_gpu}", flush=True)
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
            raise RuntimeError(f"{name}: VRAM peak {peak / (1 << 30):.2f} GiB "
                               f"over limit {args.vram_limit_gib}")
        entries[name] = {
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
            "cgroup": cgroup_peak(),
        }
        partial = receipt_base(args, prereg, prereg_digest, commit, version,
                               arm_report, free_bytes, total_bytes, entries)
        anchor.write_json(args.out, partial)
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{partial['stamp']}-{name}.json")
        anchor.write_json(raw_path, result)
        print(f"s40_score: {name} {metric_key}={score:.6f} "
              f"stderr={entries[name]['stderr']} wall={elapsed:.1f}s "
              f"peak={peak / (1 << 30):.2f}GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    receipt = receipt_base(args, prereg, prereg_digest, commit, version,
                           arm_report, free_bytes, total_bytes, entries)
    receipt["finished_utc"] = utcnow()
    receipt["complete"] = len(entries) == len(tasks)
    receipt["framework"] = {"lm_eval": getattr(lm_eval, "__version__", "unknown"),
                            "transformers": transformers.__version__}
    anchor.write_json(args.out, receipt)
    print(f"s40_score: receipt {args.out}", flush=True)
    from frontier import arms
    arms.uninstall(handles)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=comparison.ARMS, required=True)
    parser.add_argument("--prereg", default=DEFAULT_PREREG)
    parser.add_argument("--student-dir", default=comparison.DEFAULT_STUDENT)
    parser.add_argument("--teacher-dir", default=comparison.DEFAULT_TEACHER)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--memory-checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--full-checkpoint", default=None)
    parser.add_argument("--heldout-manifest", default=None)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--site", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-chunk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--tasks", default="mmlu_pro,mmlu,gsm8k,minerva_math500")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--harness-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    args = parser.parse_args(argv)
    if args.fake:
        raise SystemExit("s40_score is the GPU harness driver; use "
                         "scripts/s40_compare.py --self-test for the CPU test")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
