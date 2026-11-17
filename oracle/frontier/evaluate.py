#!/usr/bin/env python3
"""Score one stage-2 arm on the pinned harness (WS19 P19.2 smoke).

The harness is `EleutherAI/lm-evaluation-harness` at the revision pinned by
`docker/Dockerfile.eval`. The model object is built by this script and handed to
`HFLM(pretrained=<model>)`, so every arm runs through the same harness code,
tokenizer, prompt formatting and metric; only the residual module differs.

Tasks and limits used by the smoke (registered in `docs/STAGE2-PLAN.md`):

* `mmlu_pro` — 5-shot, generative, `exact_match,custom-extract`. The harness's
  `limit` is **per subject** for a group task, so `--mmlu-pro-limit 2` queues 28
  requests (14 subjects x 2), not 2 in total.
* `gsm8k` — 5-shot, generative, `exact_match,strict-match`; `limit` is total.
* `mmlu` — 5-shot loglikelihood, `acc`; `limit` per subject.

Batch size is 1 for every arm: this is a delta measurement inside one smoke,
not a reproduction of the S36 absolute scoreboard (which used batch 2/8).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, memory  # noqa: E402
from frontier.stream import sha256_file  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
GIB = float(1 << 30)

TASKS = {
    "mmlu_pro": {"num_fewshot": 5, "primary": "exact_match,custom-extract",
                 "limit_kind": "per_subject"},
    "gsm8k": {"num_fewshot": 5, "primary": "exact_match,strict-match",
              "limit_kind": "total"},
    "mmlu": {"num_fewshot": 5, "primary": "acc", "limit_kind": "per_subject"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def cgroup_memory() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    events = {}
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                events[key] = int(value)
    except (OSError, ValueError):
        events = {}
    report["memory_events"] = events
    return report


def resolve_metric(metrics: dict, primary: str) -> tuple[str, str | None]:
    candidates = [primary, f"{primary},none"]
    for candidate in candidates:
        if candidate in metrics:
            name, _, filt = candidate.partition(",")
            return candidate, (f"{name}_stderr,{filt}" if filt else f"{name}_stderr")
    matching = [key for key in metrics
                if key.split(",")[0] == primary and "stderr" not in key]
    if len(matching) == 1:
        name, _, filt = matching[0].partition(",")
        return matching[0], (f"{name}_stderr,{filt}" if filt else f"{name}_stderr")
    raise SystemExit(f"evaluate: metric {primary!r} absent from {sorted(metrics)}")


def build_model(arm_name: str, args, device: str):
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    report: dict = {"arm": arm_name}
    handles = []
    arm_module = None
    if arm_name == "ngm":
        injector = arms.NGMInjector(hidden=args.hidden)
        injector.bind(model.get_input_embeddings(), memory.TokenContext())
        handles = arms.install_ngm(model, injector, (args.site,))
        arm_module = injector
        report.update(injector.active_parameter_report())
    elif arm_name == "memory":
        if not args.memory_init:
            raise SystemExit("evaluate: --memory-init is required for the memory arm")
        init = torch.load(args.memory_init, map_location="cpu", weights_only=False)
        config = init["config"]
        module = memory.ConditionalMemory(
            hidden=config["hidden"], value_dim=config["value_dim"],
            rows=config["rows"], sites=tuple(config["sites"]),
            conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
            seed=config["seed"])
        state = init["state_dict"]
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu",
                                 weights_only=False)
            state = payload["arm_state"]
            report["checkpoint"] = args.checkpoint
            report["checkpoint_sha256"] = sha256_file(args.checkpoint)
            report["checkpoint_step"] = payload.get("step")
        module.load_state_dict(state, strict=True)
        module = module.to(device)
        handles = arms.install_memory(model, module)
        arm_module = module
        report.update(module.active_parameter_report())
        report["memory_init"] = args.memory_init
        report["memory_init_sha256"] = sha256_file(args.memory_init)
    elif arm_name == "adapter":
        if not args.checkpoint:
            raise SystemExit("evaluate: --checkpoint is required for the adapter arm")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        module = arms.DenseAdapter(hidden=args.hidden).to(device)
        module.load_state_dict(payload["arm_state"], strict=True)
        handles = arms.install_adapter(model, module, (args.site,))
        arm_module = module
        report.update(module.active_parameter_report())
        report["checkpoint"] = args.checkpoint
        report["checkpoint_sha256"] = sha256_file(args.checkpoint)
        report["checkpoint_step"] = payload.get("step")
    return model, arm_module, handles, report


def run(args) -> int:
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    import lm_eval
    import transformers

    started = time.time()
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("evaluate: CUDA required; run through scripts/court --gpu eval")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(f"evaluate: vram limit exceeds free memory "
                         f"({free_bytes / GIB:.2f} GiB)")

    model, arm_module, handles, arm_report = build_model(args.arm, args, device)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=args.student_dir,
              batch_size=args.batch_size, max_batch_size=args.batch_size,
              add_bos_token=False, trust_remote_code=False,
              dtype="bfloat16", device=device)
    task_manager = TaskManager(include_path=args.task_manager_path or None)
    limits = {
        "mmlu_pro": args.mmlu_pro_limit,
        "gsm8k": args.gsm8k_limit,
        "mmlu": args.mmlu_limit,
    }
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    per_task = []
    for name in tasks:
        if name not in TASKS:
            raise SystemExit(f"evaluate: unregistered task {name!r}")
        torch.cuda.reset_peak_memory_stats()
        task_started = time.time()
        print(f"evaluate: {args.arm} {name} limit={limits[name]}", flush=True)
        result = simple_evaluate(
            model=lm,
            tasks=[name],
            num_fewshot=TASKS[name]["num_fewshot"],
            limit=limits[name],
            bootstrap_iters=args.bootstrap_iters,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
            log_samples=False,
            task_manager=task_manager,
            verbosity="WARNING",
        )
        metrics = result["results"][name]
        metric_key, stderr_key = resolve_metric(metrics, TASKS[name]["primary"])
        stderr_raw = metrics.get(stderr_key)
        stderr = stderr_raw if isinstance(stderr_raw, (int, float)) else None
        sample_count = metrics.get("sample_count") or {}
        n_samples = sample_count.get(metric_key, metrics.get("sample_len"))
        peak = torch.cuda.max_memory_allocated() / GIB
        if peak > args.vram_limit_gib:
            raise RuntimeError(f"evaluate: VRAM peak {peak:.2f} over limit")
        entry = {
            "task": name,
            "limit": limits[name],
            "limit_kind": TASKS[name]["limit_kind"],
            "num_fewshot": TASKS[name]["num_fewshot"],
            "primary_metric": metric_key,
            "score": metrics[metric_key],
            "stderr": stderr,
            "stderr_reported": stderr_raw,
            "all_metrics": metrics,
            "n_samples": n_samples,
            "wall_seconds": round(time.time() - task_started, 3),
            "peak_allocated_gib": round(peak, 3),
            "harness_configs": result.get("configs", {}).get(name),
        }
        per_task.append(entry)
        print(f"evaluate: {name} {metric_key}={entry['score']:.6f} "
              f"wall={entry['wall_seconds']:.1f}s peak={peak:.2f}GiB", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    receipt = {
        "schema": "selvedge/stage2-eval/v1",
        "kind": "evaluate",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "arm": args.arm,
        "model": {"dir": args.student_dir},
        "arm_report": arm_report,
        "harness": {
            "name": "EleutherAI/lm-evaluation-harness",
            "version": getattr(lm_eval, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "model_object": "pre-initialised transformers model handed to HFLM",
            "batch_size": args.batch_size,
            "bootstrap_iters": args.bootstrap_iters,
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "dtype": "bfloat16",
            "add_bos_token": False,
            "limits": limits,
            "note": "batch 1 for every arm; deltas are internal to this smoke",
        },
        "tasks": per_task,
        "vram": {
            "free_at_start_gib": free_bytes / GIB,
            "device_total_gib": total_bytes / GIB,
            "limit_gib": args.vram_limit_gib,
            "source": "torch.cuda.max_memory_allocated, reset per task",
        },
        "container_memory": cgroup_memory(),
        "wall_seconds": round(time.time() - started, 3),
        "command": sys.argv,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "eval.json")
    write_json(out, receipt)
    arms.uninstall(handles)
    print(f"evaluate: receipt {out}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["base", "ngm", "memory", "adapter"],
                        required=True)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--site", type=int, default=memory.DEFAULT_SITE)
    parser.add_argument("--hidden", type=int, default=memory.DEFAULT_HIDDEN)
    parser.add_argument("--tasks", default="mmlu_pro,gsm8k")
    parser.add_argument("--mmlu-pro-limit", type=int, default=2)
    parser.add_argument("--gsm8k-limit", type=int, default=100)
    parser.add_argument("--mmlu-limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--task-manager-path", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=14.0)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
