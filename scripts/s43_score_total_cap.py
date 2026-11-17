#!/usr/bin/env python3
"""Score one arm under a *total* item cap, for group tasks (S43).

The S41 anchored-subset pass asked for `--limit-map mmlu=2000`. MMLU is a
57-subject group task and lm-evaluation-harness applies `limit` per subtask, so
that argument selected `min(2000, n_subject)` for every subject: the effective
sample was all 14,042 items. The receipts record what happened
(`sample_len = 14042`), and `docs/ws19-anchored-subset-amendment-preregistration-v2.json`
records the defect.

This driver is new code: it does not edit `scripts/s40_score.py`, which may be
executing while this file is written or run. It expresses a total cap in the
only place the harness exposes one: `simple_evaluate(samples=...)`, where each
leaf task receives a list of exact sample indices. The allocation is
deterministic (largest remainder by leaf size, ties broken by leaf name), the
per-leaf allocation is written into the receipt, and the run fails closed if the
effective sample exceeds the requested cap or if a cap is smaller than the
number of leaves.

Usage:
    SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 scripts/s43_score_total_cap.py \
        --arm base --tasks mmlu,mmlu_pro,gsm8k,minerva_math500 \
        --total-cap-map mmlu=2000 \
        --registration docs/ws19-anchored-subset-amendment-preregistration-v2.json \
        --out results/stage2/eval/base-2000mmlu/eval.json

The JSON receipt schema is `selvedge/stage2-arm-eval/v1`, the same as
`s40_score.py`, with one added top-level key `total_cap_enforcement`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

import s36_anchor as anchor  # noqa: E402
import s40_compare as comparison  # noqa: E402
from frontier import correctness as correctness_check  # noqa: E402
from s40_score import FAMILY_TASKS, SCHEMA, cgroup_peak  # noqa: E402


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_caps(text: str | None, declared: dict, tasks: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in declared:
            raise SystemExit(f"--total-cap-map names unregistered task {name!r}")
        if name not in tasks:
            raise SystemExit(f"--total-cap-map names {name!r}, which is not in --tasks")
        try:
            caps[name] = int(raw)
        except ValueError:
            raise SystemExit(f"--total-cap-map {item!r} is not task=<int>")
        if caps[name] <= 0:
            raise SystemExit(f"--total-cap-map {item!r} must be positive")
    return caps


def allocate(cap: int, sizes: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Largest-remainder allocation of `cap` items over leaves of `sizes`."""
    total = sum(size for _, size in sizes)
    if cap >= total:
        return [(name, size) for name, size in sizes]
    if cap < len(sizes):
        raise SystemExit(
            f"total cap {cap} is smaller than the {len(sizes)} leaves; a total "
            "cap cannot be stratified below one item per leaf")
    raw = [(name, cap * size / total) for name, size in sizes]
    base = {name: int(value) for name, value in raw}
    remainder = cap - sum(base.values())
    order = sorted(raw, key=lambda pair: (-(pair[1] - int(pair[1])), pair[0]))
    for name, _ in order[:remainder]:
        base[name] += 1
    # Guarantee one item per leaf; take the surplus back from the largest.
    for name, _ in sorted(sizes, key=lambda pair: pair[1]):
        if base[name] == 0:
            donor = max(base, key=lambda key: (base[key], key))
            base[donor] -= 1
            base[name] = 1
    return [(name, base[name]) for name, _ in sizes]


def leaf_sizes(task_name: str) -> list[tuple[str, int]]:
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    task_dict = manager.load([task_name])
    group = task_dict["groups"].get(task_name)
    if group is None:
        task = task_dict["tasks"].get(task_name)
        if task is None:
            raise SystemExit(f"{task_name}: the harness loaded neither a group nor a task")
        return [(task_name, len(task.eval_docs))]
    leaves = sorted(group.get_all_tasks(), key=lambda task: task.task_name)
    return [(task.task_name, len(task.eval_docs)) for task in leaves]


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
    caps = parse_caps(args.total_cap_map, declared, tasks)

    if not torch.cuda.is_available():
        raise SystemExit("s43_score_total_cap: CUDA required; run with --gpu eval")
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
    enforcement = {}
    for name in tasks:
        task = declared[name]
        cap_report = None
        samples = None
        limit = task.get("limit")
        if name in caps:
            plan = allocate(caps[name], leaf_sizes(name))
            samples = {leaf: list(range(count)) for leaf, count in plan}
            limit = None
            cap_report = {
                "requested_total_cap": caps[name],
                "allocated_total": sum(count for _, count in plan),
                "method": "largest remainder by leaf eval size, ties by leaf name",
                "leaves": [{"task": leaf, "allocated": count} for leaf, count in plan],
            }
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        lm.batch_size_per_gpu = int(task.get("batch_size", 1))
        print(f"s43_score_total_cap: {args.arm} {name} cap={caps.get(name)} "
              f"batch={lm.batch_size_per_gpu}", flush=True)
        result = simple_evaluate(
            model=lm,
            tasks=[name],
            num_fewshot=task["num_fewshot"],
            batch_size=task.get("batch_size", "auto"),
            max_batch_size=prereg.get("max_batch_size"),
            device=device,
            limit=limit,
            samples=samples,
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
        if cap_report is not None:
            effective = result.get("n-samples") or {}
            actual = sum(int((value or {}).get("effective") or 0)
                         for key, value in effective.items()
                         if key == name or key.startswith(name))
            cap_report["effective_total"] = actual
            if actual > caps[name]:
                raise RuntimeError(
                    f"{name}: effective total {actual} exceeds the requested "
                    f"cap {caps[name]}; refusing the receipt")
            cap_report["within_cap"] = True
            enforcement[name] = cap_report
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
                "total_cap": caps.get(name),
                "num_fewshot": task["num_fewshot"],
                "batch_size": task.get("batch_size"),
            },
            "harness_config": harness_config,
            "wall_clock_s": round(elapsed, 1),
            "peak_memory_allocated_bytes": int(peak),
            "cgroup": cgroup_peak(),
            "sequence_correctness": (
                correctness_check.harness_samples(
                    (result.get("samples") or {}).get(name) or [], name)
                if name in correctness_check.VERIFIABLE_TASKS
                and (result.get("samples") or {}).get(name) else None),
        }
        partial = receipt_base(args, prereg, prereg_digest, commit, version,
                               arm_report, free_bytes, total_bytes, entries,
                               enforcement)
        anchor.write_json(args.out, partial)
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{partial['stamp']}-{name}.json")
        anchor.write_json(raw_path, result)
        print(f"s43_score_total_cap: {name} {metric_key}={score:.6f} "
              f"stderr={entries[name]['stderr']} wall={elapsed:.1f}s "
              f"peak={peak / (1 << 30):.2f}GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    receipt = receipt_base(args, prereg, prereg_digest, commit, version,
                           arm_report, free_bytes, total_bytes, entries,
                           enforcement)
    receipt["finished_utc"] = utcnow()
    receipt["complete"] = len(entries) == len(tasks)
    receipt["framework"] = {"lm_eval": getattr(lm_eval, "__version__", "unknown"),
                            "transformers": transformers.__version__}
    receipt["correctness_and_likelihood"] = {
        "heldout": "mean next-token CE and KL(teacher_T || student_T) on the "
                   "held-out slice are reported by the companion held-out receipt",
        "sequence": "the exact final-answer verdict from the S45 extractor on "
                    "this task's own generations (gsm8k and minerva_math500)",
        "why_both": "a lower mean CE can coexist with a lower exact-answer rate; "
                    "neither quantity is substituted for the other",
    }
    anchor.write_json(args.out, receipt)
    print(f"s43_score_total_cap: receipt {args.out}", flush=True)
    from frontier import arms
    arms.uninstall(handles)
    return 0


def receipt_base(args, prereg, prereg_digest, commit, version, arm_report,
                 free_bytes, total_bytes, entries, enforcement) -> dict:
    import torch
    return {
        "schema": SCHEMA,
        "kind": "arm-eval-total-cap",
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
        "preregistration": (
            {"path": args.registration,
             "sha256": anchor.sha256_file(args.registration)}
            if args.registration else None),
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
        "total_cap_enforcement": enforcement,
        "scored_tasks": sorted(entries),
        "tasks": [entries[name] for name in sorted(entries)],
        "complete": False,
        "argv": sys.argv,
    }


def self_test() -> int:
    """The allocation is exact and the group is what the defect claims."""
    sizes = [("mmlu_a", 100), ("mmlu_b", 300), ("mmlu_c", 600)]
    plan = allocate(200, sizes)
    assert sum(count for _, count in plan) == 200, plan
    assert [name for name, _ in plan] == ["mmlu_a", "mmlu_b", "mmlu_c"]
    assert plan == allocate(200, sizes), "allocation is not deterministic"
    assert sum(count for _, count in allocate(1_000, sizes)) == 1_000
    assert sum(count for _, count in allocate(10_000, sizes)) == 1_000
    try:
        allocate(2, sizes)
    except SystemExit:
        pass
    else:
        raise AssertionError("a cap below the leaf count must fail closed")
    leaves = leaf_sizes("mmlu")
    total = sum(size for _, size in leaves)
    if total != 14042 or len(leaves) != 57:
        raise SystemExit(f"self-test: mmlu has {len(leaves)} leaves / {total} items")
    capped = allocate(2000, leaves)
    if sum(count for _, count in capped) != 2000:
        raise SystemExit("self-test: a 2,000-item total cap did not allocate 2,000")
    print(f"self-test: mmlu leaves={len(leaves)} total={total} "
          f"cap2000={sum(count for _, count in capped)} PASS")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="CPU test of the allocation and the group size")
    parser.add_argument("--arm", choices=comparison.SCORED_ARMS, required=False)
    parser.add_argument("--prereg", default="docs/anchor-preregistration-v1.json")
    parser.add_argument("--registration", default=None)
    parser.add_argument("--student-dir", default=comparison.DEFAULT_STUDENT)
    parser.add_argument("--teacher-dir", default=comparison.DEFAULT_TEACHER)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--memory-checkpoint", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--control-checkpoint", default=None)
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
    parser.add_argument("--total-cap-map", default=None,
                        help="comma-separated task=<total items> caps; a group "
                             "task receives a stratified allocation over its "
                             "leaves whose sizes sum to exactly <total items>")
    parser.add_argument("--out", required=False)
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.arm or not args.out:
        raise SystemExit("--arm and --out are required unless --self-test is given")
    args.fake = False
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
