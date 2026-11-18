#!/usr/bin/env python3
"""S50: fixed-batch teacher KL for the revalidation checkpoints.

The training log records KL on a *different* batch at every step, because the
stream is consumed in order and the batches differ in difficulty (the first
batch is easy: CE 0.60, KL 0.23 for every run at every learning rate). A
within-run comparison of those numbers therefore measures the data order, not
the optimiser. This script measures the same fixed held-out batches for the
initial student and for every checkpoint, so the direction of teacher KL is
attributable.

KL is `KL(teacher_T || student_T)` at T = 2.0, averaged over positions, exactly
as `frontier.train.token_loss` computes it; CE is against the next token at
T = 1.0.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import arms, full  # noqa: E402

STUDENT_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
GIB = float(1 << 30)


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cgroup_memory() -> dict:
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


def _load_stream(manifest_path: str) -> np.ndarray:
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    path = manifest["stream_path"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    digest = _sha256(path)
    if digest != manifest["sha256"]:
        raise SystemExit(f"stream sha mismatch: {digest} != {manifest['sha256']}")
    return np.fromfile(path, dtype="<u4")


@torch.no_grad()
def _metrics(student, teacher, inputs, targets, temperature: float,
             chunk: int = 32) -> dict:
    student_logits = student(input_ids=inputs, use_cache=False).logits
    teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
    flat_student = student_logits.reshape(-1, student_logits.shape[-1]).float()
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
    flat_targets = targets.reshape(-1)
    ce_total = 0.0
    kl_total = 0.0
    positions = 0
    for start in range(0, flat_student.shape[0], chunk):
        s = flat_student[start:start + chunk]
        t = flat_teacher[start:start + chunk]
        ce_total += float(F.cross_entropy(s, flat_targets[start:start + chunk],
                                          reduction="sum"))
        log_s = F.log_softmax(s / temperature, dim=-1)
        log_t = F.log_softmax(t / temperature, dim=-1)
        kl_total += float((log_t.exp() * (log_t - log_s)).sum())
        positions += s.shape[0]
    return {"ce": ce_total / positions, "kl": kl_total / positions,
            "positions": positions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-manifest",
                        default=("results/stage2/"
                                 "stage2-math-stream-20261117T071120Z.json"),
                        help="manifest whose fixed batches are measured")
    parser.add_argument("--start-token", type=int, default=0,
                        help="first stream offset of the fixed batch window")
    parser.add_argument("--run-dir",
                        default="results/stage2/runs/s50-revalidate-200-lr1e-4-curve")
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("s50-kl: CUDA required; run through scripts/court --gpu oracle")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = "cuda"
    stream = _load_stream(args.stream_manifest)
    batches = []
    for index in range(args.batches):
        offset = args.start_token + index * args.seq_len
        chunk = torch.from_numpy(stream[offset:offset + args.seq_len + 1].astype(np.int64))
        if chunk.numel() != args.seq_len + 1:
            raise SystemExit("s50-kl: held-out stream too short for the batch count")
        batches.append((chunk[None, :-1].to(device), chunk[None, 1:].to(device)))

    student = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    arm = full.FullBackboneArm(student, freeze_embedding=True)
    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    checkpoints = sorted(glob.glob(os.path.join(args.run_dir, "ckpt-*.pt")))
    points = [("init", None)] + [
        (os.path.basename(path), path) for path in checkpoints]
    results = []
    for label, path in points:
        if path is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            arm.load_state_dict(payload["arm_state"])
        totals = {"ce": 0.0, "kl": 0.0, "positions": 0}
        for inputs, targets in batches:
            metrics = _metrics(student, teacher, inputs, targets,
                               args.temperature)
            for key in totals:
                totals[key] += metrics[key]
        row = {
            "checkpoint": label,
            "sha256": _sha256(path) if path else None,
            "ce": totals["ce"] / len(batches),
            "kl": totals["kl"] / len(batches),
            "batches": len(batches),
            "positions": totals["positions"],
        }
        results.append(row)
        print(f"s50-kl: {label}: ce={row['ce']:.6f} kl={row['kl']:.6f} "
              f"({row['positions']} positions)", flush=True)

    receipt = {
        "schema": "selvedge/s50-fixed-batch-kl/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _git_revision(),
        "heldout_manifest": args.stream_manifest,
        "heldout_sha256": _sha256(os.path.join(
            ROOT, json.load(open(os.path.join(ROOT, args.stream_manifest),
                                 encoding="utf-8"))["stream_path"])),
        "start_token": args.start_token,
        "run_dir": args.run_dir,
        "student_dir": args.student_dir,
        "teacher_dir": args.teacher_dir,
        "temperature": args.temperature,
        "batch_count": args.batches,
        "seq_len": args.seq_len,
        "metric": ("KL(teacher_T || student_T) per position at T=2.0, fixed "
                   "held-out batches; CE at T=1.0"),
        "results": results,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "container_memory": _cgroup_memory(),
        "command": sys.argv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
