#!/usr/bin/env python3
"""Registered teacher+student throughput measurement (WS19 section 4).

One *full* training step at the smoke geometry, repeated with warmup discarded:

    teacher forward  Qwen/Qwen3.5-2B-Base, bf16, no_grad
    student forward  Qwen/Qwen3.5-0.8B-Base, frozen core + arm, bf16 autocast
    student backward gradients for the arm's trainable parameters only
    loss             CE + lambda * T^2 * KL(teacher_T || student_T), T = 2.0
    optimiser step   AdamW over the arm's parameters

The protocol fixed in `docs/WS19-EXECUTION-REGISTRATION.md` section 4 is
implemented literally:

* sequence length 512, micro-batch 1, gradient accumulation 4 (2,048 tokens per
  optimiser step), one site after decoder layer 2, 65,536-row table, 8 reads per
  token;
* at least 20 discarded warmup steps, then at least 50 measured steps; the
  median is reported, never the best;
* the teacher is loaded once, outside the timed region, and runs under no_grad;
* no `torch.compile`;
* per-phase CUDA-event time for teacher forward, student forward, loss,
  backward, optimiser and data transfer, each as a fraction of the step.

`--arm memory` is the registered measurement. `--arm adapter` measures the same
step with the parameter-matched dense control installed instead, which is the
arm the distillation-only control (Job 1) trains; its receipt sizes that run's
budget without borrowing the memory arm's table cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, memory  # noqa: E402
from frontier.stream import load_stream, sha256_file  # noqa: E402
from frontier.train import build_arm, cgroup_memory, next_batch, token_loss  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-2B-Base",
                           "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
GIB = float(1 << 30)
PHASES = ("data_transfer", "teacher_forward", "student_forward", "loss",
          "backward", "optimizer")


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


def config_hash(path: str) -> str | None:
    config = os.path.join(path, "config.json")
    if not os.path.isfile(config):
        return None
    return sha256_file(config)


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        index = min(count - 1, max(0, int(round(fraction * (count - 1)))))
        return ordered[index]

    return {
        "count": count,
        "min": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": at(0.95),
        "max": max(ordered),
    }


class BoundaryEvents:
    """One CUDA event per phase boundary, per micro-batch, reused each step."""

    def __init__(self, grad_accum: int):
        # 4 boundaries inside each micro-batch: after data, after teacher, after
        # student, after loss. The micro-batch's start is the previous
        # micro-batch's backward end (or the step start for the first), and its
        # backward end is recorded separately.
        self.micro = grad_accum
        self.events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                       for _ in range(grad_accum)]
        self.backward_end = [torch.cuda.Event(enable_timing=True)
                             for _ in range(grad_accum)]
        self.optimizer_end = torch.cuda.Event(enable_timing=True)
        self.step_start = torch.cuda.Event(enable_timing=True)

    def begin(self) -> None:
        self.step_start.record()

    def record(self, micro: int, index: int) -> None:
        self.events[micro][index].record()

    def record_backward_end(self, micro: int) -> None:
        self.backward_end[micro].record()

    def record_optimizer_end(self) -> None:
        self.optimizer_end.record()

    def step_breakdown(self) -> dict[str, float]:
        """Seconds per phase for the step just recorded."""
        out = {name: 0.0 for name in PHASES}
        for micro in range(self.micro):
            before_data = self.step_start if micro == 0 else self.backward_end[micro - 1]
            record = self.events[micro]
            out["data_transfer"] += before_data.elapsed_time(record[0]) / 1000.0
            out["teacher_forward"] += record[0].elapsed_time(record[1]) / 1000.0
            out["student_forward"] += record[1].elapsed_time(record[2]) / 1000.0
            out["loss"] += record[2].elapsed_time(record[3]) / 1000.0
            out["backward"] += record[3].elapsed_time(self.backward_end[micro]) / 1000.0
        out["optimizer"] = self.backward_end[self.micro - 1].elapsed_time(
            self.optimizer_end) / 1000.0
        return out


def run(args) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("throughput: CUDA required; run through scripts/court --gpu")
    if args.warmup < 20:
        raise SystemExit("throughput: the registered protocol needs >= 20 warmup steps")
    if args.measure < 50:
        raise SystemExit("throughput: the registered protocol needs >= 50 measured steps")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(f"throughput: vram limit {args.vram_limit_gib} GiB exceeds free "
                         f"{free_bytes / GIB:.2f} GiB")

    manifest = json.load(open(args.stream_manifest))
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit("throughput: stream sha mismatch")
    stream = torch.from_numpy(load_stream(manifest["stream_path"]).astype(np.int64))
    offset = int(manifest["graft_tokens"])
    tokens_per_step = args.seq_len * args.grad_accum
    needed = (args.warmup + args.measure + 2) * tokens_per_step
    if stream.numel() - offset < needed:
        raise SystemExit(f"throughput: stream has {stream.numel() - offset} train tokens, "
                         f"need {needed} for the protocol")

    print(f"throughput: loading student {args.student_dir}", flush=True)
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    frozen = arms.freeze_backbone(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args.memory_init = args.memory_init or None
    arm, handles, arm_report = build_arm(args.arm, model, args, device)
    if args.arm == "memory":
        for parameter in arm.parameters():
            parameter.requires_grad_(True)

    print(f"throughput: loading teacher {args.teacher_dir}", flush=True)
    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
    teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in arm.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    boundaries = BoundaryEvents(args.grad_accum)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    def one_step(offset: int, timing: bool) -> tuple[float, dict[str, float] | None, int]:
        started = time.perf_counter()
        if timing:
            boundaries.begin()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            batch, _ = next_batch(stream, offset + micro * args.seq_len, args.seq_len)
            batch = batch.to(device)
            if timing:
                boundaries.record(micro, 0)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            with torch.no_grad():
                teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
            if timing:
                boundaries.record(micro, 1)
            student = model(input_ids=inputs, use_cache=False).logits
            if timing:
                boundaries.record(micro, 2)
            loss, _ce, _kl = token_loss(student, teacher_logits, targets,
                                        args.temperature, args.kl_weight)
            if timing:
                boundaries.record(micro, 3)  # loss end == backward start
            (loss / args.grad_accum).backward()
            if timing:
                boundaries.record_backward_end(micro)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if timing:
            boundaries.record_optimizer_end()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        breakdown = boundaries.step_breakdown() if timing else None
        return elapsed, breakdown, offset + tokens_per_step

    # Warmup: discarded entirely.
    for _ in range(args.warmup):
        _, _, offset = one_step(offset, timing=False)
    print(f"throughput: {args.warmup} warmup steps discarded", flush=True)

    step_times: list[float] = []
    micro_times: list[float] = []
    phase_rows: list[dict[str, float]] = []
    peak_alloc = 0.0
    peak_reserved = 0.0
    for index in range(args.measure):
        elapsed, breakdown, offset = one_step(offset, timing=True)
        if breakdown is None:
            raise RuntimeError("throughput: measured step lost its breakdown")
        step_times.append(elapsed)
        micro_times.append(elapsed / args.grad_accum)
        phase_rows.append(breakdown)
        allocated = torch.cuda.max_memory_allocated() / GIB
        reserved = torch.cuda.max_memory_reserved() / GIB
        peak_alloc = max(peak_alloc, allocated)
        peak_reserved = max(peak_reserved, reserved)
        if allocated > args.vram_limit_gib:
            raise RuntimeError(f"throughput: VRAM peak {allocated:.2f} GiB over limit "
                               f"{args.vram_limit_gib:.2f} GiB")
        if index % 10 == 0:
            print(f"throughput: measure {index + 1}/{args.measure} "
                  f"{tokens_per_step / elapsed:.0f} tok/s step={elapsed * 1000:.0f}ms "
                  f"peak={allocated:.2f}GiB", flush=True)

    step_stats = quantiles(step_times)
    micro_stats = quantiles(micro_times)
    phase_stats = {}
    for name in PHASES:
        values = [row[name] for row in phase_rows]
        phase_stats[name] = quantiles(values)
    median_step = step_stats["median"]
    median_phase_sum = sum(phase_stats[name]["median"] for name in PHASES)
    for name in PHASES:
        phase_stats[name]["fraction_of_step"] = (
            phase_stats[name]["median"] / median_step if median_step else 0.0)
        phase_stats[name]["fraction_of_phase_sum"] = (
            phase_stats[name]["median"] / median_phase_sum if median_phase_sum else 0.0)

    receipt = {
        "schema": "selvedge/stage2-throughput/v1",
        "kind": "throughput",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "label": args.label,
        "protocol": {
            "registration": "docs/WS19-EXECUTION-REGISTRATION.md section 4",
            "description": "one full teacher+student optimiser step at the smoke geometry",
            "warmup_steps_discarded": args.warmup,
            "measured_steps": args.measure,
            "median_not_best": True,
            "torch_compile": False,
            "teacher_no_grad": True,
        },
        "geometry": {
            "arm": args.arm,
            "seq_len": args.seq_len,
            "micro_batch": 1,
            "grad_accum": args.grad_accum,
            "tokens_per_optimiser_step": tokens_per_step,
            "site": args.site,
            "temperature": args.temperature,
            "kl_weight": args.kl_weight,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "table_rows": (None if args.arm != "memory"
                           else arm_report.get("rows_per_site")),
            "reads_per_token": (None if args.arm != "memory"
                                else arm_report.get("reads_per_token_per_site")),
        },
        "throughput": {
            "optimiser_step_tokens_per_second": tokens_per_step / median_step,
            "micro_batch_tokens_per_second": args.seq_len / micro_stats["median"],
            "tokens_per_optimiser_step": tokens_per_step,
            "optimiser_step_seconds": step_stats,
            "micro_batch_seconds": micro_stats,
        },
        "vram": {
            "peak_allocated_bytes": int(peak_alloc * GIB),
            "peak_allocated_gib": peak_alloc,
            "peak_reserved_bytes": int(peak_reserved * GIB),
            "peak_reserved_gib": peak_reserved,
            "device_total_bytes": total_bytes,
            "device_total_gib": total_bytes / GIB,
            "free_at_start_gib": free_bytes / GIB,
            "limit_gib": args.vram_limit_gib,
            "source": "torch.cuda.max_memory_allocated after the last measured step",
        },
        "phase_breakdown_seconds": phase_stats,
        "phase_method": (
            "CUDA events recorded at phase boundaries on the single default stream; "
            "one stream synchronisation per measured step; fractions are of the "
            "median wall step and of the summed median phase times"
        ),
        "load": {
            "student_dir": args.student_dir,
            "student_config_sha256": config_hash(args.student_dir),
            "teacher_dir": args.teacher_dir,
            "teacher_config_sha256": config_hash(args.teacher_dir),
            "teacher_parameters": teacher_params,
            "frozen_backbone_elements": int(frozen),
            "trainable_arm_elements": int(sum(p.numel() for p in arm.parameters())),
            "arm_report": arm_report,
            "memory_init": args.memory_init,
            "memory_init_sha256": (sha256_file(args.memory_init)
                                   if args.memory_init else None),
            "dtype": "bfloat16",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": True,
        },
        "stream": {
            "manifest": args.stream_manifest,
            "sha256": manifest["sha256"],
            "tokens": int(stream.numel()),
            "graft_tokens": int(manifest["graft_tokens"]),
        },
        "container_memory": cgroup_memory(),
        "command": sys.argv,
    }
    write_json(args.out, receipt)
    arms.uninstall(handles)
    print(f"throughput: {args.arm} {tokens_per_step / median_step:.1f} tok/s "
          f"median_step={median_step * 1000:.1f}ms p95={step_stats['p95'] * 1000:.1f}ms "
          f"peak_alloc={peak_alloc:.2f}GiB peak_reserved={peak_reserved:.2f}GiB", flush=True)
    print(f"throughput: receipt {args.out}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["memory", "adapter"], default="memory")
    parser.add_argument("--stream-manifest", required=True)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--site", type=int, default=memory.DEFAULT_SITE)
    parser.add_argument("--hidden", type=int, default=memory.DEFAULT_HIDDEN)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="measured")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=11.0)
    args = parser.parse_args(argv)
    if args.arm == "memory" and not args.memory_init:
        raise SystemExit("--memory-init is required for --arm memory")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
