#!/usr/bin/env python3
"""S56: termination rate at a checkpoint, on a fixed greedy sample.

Training steps do not generate text, so a termination rate cannot be read from
the training log. This probe closes that gap: it loads one stage-2 checkpoint on
the base backbone, generates the first N GSM8K and MATH-500 test problems with
the *registered* greedy protocol (temperature 0, cap unchanged, no repetition
penalty), verifies each answer with the same flexible extractor the eval receipt
uses, and writes the termination block.

It is a diagnostic, never a score: the receipt says so, and the limit is small
(default 50 problems per task) so it can run at the registered cadence
(2M / 5M / final tokens) without competing with a scoring job.

Usage (GPU, when the device is free):
    SELVEDGE_MEM=8g SELVEDGE_CPUS=4 scripts/court --gpu eval -- python3 \\
        scripts/s56_termination_probe.py \\
        --run-dir results/stage2/runs/s50-bar-20m-lr1e-5 \\
        --checkpoint results/stage2/runs/s50-bar-20m-lr1e-5/ckpt-004882.pt \\
        --label m10

Usage (CPU, plan only):
    SELVEDGE_MEM=2g scripts/court python -- python3 \\
        scripts/s56_termination_probe.py --run-dir ... --label m10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import termination  # noqa: E402

OUT_ROOT = "results/stage2/termination"
GIB = float(1 << 30)
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_STUDENT_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def cgroup_memory() -> dict:
    values = {}
    for name in ("memory.max", "memory.peak", "memory.swap.peak"):
        path = f"/sys/fs/cgroup/{name}"
        if os.path.exists(path):
            values[name.replace(".", "_")] = open(path, encoding="utf-8").read().strip()
    return values


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="default: <run-dir>/latest.pt")
    parser.add_argument("--label", required=True,
                        help="checkpoint label, e.g. m2, m5, final")
    parser.add_argument("--student-dir", default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--tasks", default="gsm8k,math500")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--vram-limit-gib", type=float, default=12.5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or os.path.join(args.run_dir, "latest.pt")
    out = args.out or os.path.join(OUT_ROOT, os.path.basename(
        os.path.normpath(args.run_dir)), f"{args.label}.json")
    student_dir = args.student_dir or DEFAULT_STUDENT_DIR
    if not os.path.exists(checkpoint):
        raise SystemExit(f"s56-probe: missing checkpoint {checkpoint}")
    if args.dry_run:
        print("s56-probe: plan (no model loaded)")
        print(f"s56-probe: checkpoint {checkpoint}")
        print(f"s56-probe: student {student_dir}")
        print(f"s56-probe: tasks {args.tasks} limit {args.limit} "
              f"cap {args.max_new_tokens} greedy")
        print(f"s56-probe: out {out}")
        print("s56-probe: limits SELVEDGE_MEM=8g SELVEDGE_CPUS=4 "
              "SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g")
        print("s56-probe: command: SELVEDGE_MEM=8g SELVEDGE_CPUS=4 "
              "scripts/court --gpu eval -- python3 scripts/s56_termination_probe.py "
              + " ".join(argv))
        return 0

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("s56-probe: CUDA required; run with --gpu eval")
    import s54_post_trained_eval as s54
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s56-probe: vram limit {args.vram_limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(
        min(0.99, args.vram_limit_gib * GIB / total_bytes))

    # Reuse the scoring instrument's own loader and adapters so the probe is the
    # same model as the score, not a second interpretation of the checkpoint.
    class ProbeArgs:
        fake = False
        student_dir = None
        arm = "full-distill"
        full_checkpoint = None
        hidden = 1024

    probe_args = ProbeArgs()
    probe_args.student_dir = student_dir
    probe_args.full_checkpoint = checkpoint
    import s40_compare as comparison

    started = time.time()
    model = comparison.load_backbone(probe_args, "cuda")
    model, _handles, arm_report = comparison.build_named_arm(
        "full-distill", model, probe_args, "cuda")
    model.eval()
    tokenizer = s54.load_tokenizer(student_dir)

    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    rows = []
    for task in tasks:
        items = s54.load_task(task, args.limit)
        prompts = [s54.render_prompt(tokenizer, item["question"], False)
                   for item in items]
        completions, stats = s54.generate_batch(
            model, tokenizer, prompts, greedy=True,
            max_new_tokens=args.max_new_tokens, device="cuda",
            num_return_sequences=1, seed=None)
        samples = []
        for index, (item, completion) in enumerate(zip(items, completions)):
            flexible = s54.extract_flexible(completion, task)
            correct = (flexible["value"] is not None
                       and s54.rft.answers_equal(flexible["value"], item["gold"]))
            samples.append({
                "problem_id": item["problem_id"],
                "completion": completion,
                "new_tokens": int(stats["new_tokens_per_sequence"][index]),
                "truncated": bool(stats["truncated"][index]),
                "correct": bool(correct),
                "problem_correct": bool(correct),
            })
        summary = termination.summarise(samples)
        rows.append({"task": task, "mode": "non-thinking",
                     "decoding": "greedy", "problems": len(items),
                     "termination": summary})
        print(f"s56-probe: {task} truncated="
              f"{summary['truncation_rate']:.4f} loop="
              f"{summary['loop_classified_fraction']:.4f} "
              f"acc_terminated="
              f"{summary['sample_level']['terminated']['accuracy']} "
              f"acc_truncated="
              f"{summary['sample_level']['truncated']['accuracy']}", flush=True)

    payload = {
        "schema": "selvedge/s56-termination-probe/v1",
        "kind": "termination-probe",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "label": args.label,
        "run_dir": args.run_dir,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": arm_report.get("checkpoint_step"),
        "arm_report": arm_report,
        "model": {"dir": student_dir},
        "engine": {"name": "transformers",
                   "torch_version": torch.__version__,
                   "attention": "sdpa", "dtype": "bfloat16"},
        "protocol": {
            "decoding": "greedy (temperature 0, top_p 1)",
            "max_new_tokens": args.max_new_tokens,
            "stop": "EOS only",
            "repetition_penalty": 1.0,
            "prompt": "the stage-1 non-thinking protocol, unchanged",
            "limit_per_task": args.limit,
            "note": ("diagnostic only: this is not an anchored score and is "
                     "never compared with one"),
        },
        "termination": {
            "thresholds": termination.DEFAULT_THRESHOLDS,
            "tail_chars": termination.TAIL_CHARS,
            "min_chars": termination.MIN_CHARS,
            "calibration": "results/stage1/s56-termination-calibration.json",
        },
        "tasks": rows,
        "cgroup_memory": cgroup_memory(),
        "wall_seconds": round(time.time() - started, 1),
        "command": sys.argv,
    }
    write_json(out, payload)
    print(f"s56-probe: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
