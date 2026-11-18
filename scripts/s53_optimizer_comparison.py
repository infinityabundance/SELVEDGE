#!/usr/bin/env python3
"""S53: GPU driver for the model-level optimiser comparison. Queued, not run.

Question: at matched initialisation, matched stream and matched learning rate,
does the custom int8-state optimiser (`optim8bit.Adam8bit`, block 256,
stochastic rounding) track fp32 AdamW on **fixed-batch** teacher KL and held-out
CE over 200 optimiser steps?

The measured quantity is the same fixed set of batches for both arms and at
every evaluation point, so the comparison is not measuring the stream's
difficulty order. The arms run sequentially (one student resident at a time) and
each `make()` restores the exact same initial state captured once.

This script is not launched by CPU work. Its container limit must be derived
from its own smoke run: start with a generous explicit limit, read
`container_memory.memory_peak`, then launch at max(1.5 x peak, peak + 1 GiB).
`docs/S50-GPU-QUEUE.md` carries the queued command and that sizing rule.
"""

from __future__ import annotations

import argparse
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

from frontier import arms, full, optim8bit, optim_compare  # noqa: E402
from frontier import train as train_module  # noqa: E402

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
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_stream(manifest_path: str) -> np.ndarray:
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    path = manifest["stream_path"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if _sha256(path) != manifest["sha256"]:
        raise SystemExit("stream sha mismatch")
    return np.fromfile(path, dtype="<u4")


def _batch(stream: np.ndarray, offset: int, seq_len: int, device: str):
    chunk = torch.from_numpy(stream[offset:offset + seq_len + 1].astype(np.int64))
    if chunk.numel() != seq_len + 1:
        raise SystemExit("stream too short for the requested steps")
    return chunk[None, :-1].to(device), chunk[None, 1:].to(device)


@torch.no_grad()
def _metrics(model, teacher, batches, temperature: float, chunk: int = 32) -> dict:
    totals = {"ce": 0.0, "kl": 0.0, "positions": 0}
    model.eval()
    for inputs, targets in batches:
        student_logits = model(input_ids=inputs, use_cache=False).logits
        teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
        flat_student = student_logits.reshape(-1, student_logits.shape[-1]).float()
        flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
        flat_targets = targets.reshape(-1)
        for start in range(0, flat_student.shape[0], chunk):
            s = flat_student[start:start + chunk]
            t = flat_teacher[start:start + chunk]
            totals["ce"] += float(F.cross_entropy(
                s, flat_targets[start:start + chunk], reduction="sum"))
            log_s = F.log_softmax(s / temperature, dim=-1)
            log_t = F.log_softmax(t / temperature, dim=-1)
            totals["kl"] += float((log_t.exp() * (log_t - log_s)).sum())
            totals["positions"] += s.shape[0]
    positions = max(1, totals["positions"])
    return {"ce": totals["ce"] / positions, "kl": totals["kl"] / positions}


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.int16).numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-manifest",
                        default=("results/stage2/"
                                 "stage2-math-stream-20261117T071120Z.json"))
    parser.add_argument("--start-token", type=int, default=2_000_000,
                        help="first fixed-batch and first training-batch offset")
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batches", type=int, default=8,
                        help="fixed evaluation batches, identical for both arms")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default="custom,adamw",
                        help="comma-separated arms: custom (int8 first moment), "
                             "custom_bf16 (bfloat16 first moment), adamw "
                             "(fp32 reference)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("s53-optim: CUDA required; run through scripts/court --gpu oracle")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = "cuda"
    stream = _load_stream(args.stream_manifest)

    eval_batches = [
        _batch(stream, args.start_token + index * args.seq_len, args.seq_len,
               device)
        for index in range(args.batches)]

    print(f"s53-optim: loading teacher {args.teacher_dir}", flush=True)
    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    # Capture the exact initial trainable state once. Every arm restores it, so
    # a difference cannot come from a different initialisation.
    print(f"s53-optim: capturing matched init from {args.student_dir}", flush=True)
    probe_model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                                  device=device)
    probe_arm = full.FullBackboneArm(probe_model, freeze_embedding=True)
    init_state = probe_arm.state_dict()
    init_digest = _state_digest(init_state)
    trainable_elements = int(probe_arm.trainable_elements)
    del probe_arm, probe_model
    torch.cuda.empty_cache()

    def make(name):
        def build():
            model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                                    device=device)
            arm = full.FullBackboneArm(model, freeze_embedding=True)
            arm.load_state_dict(init_state)
            arm.train()
            trainable = arm.parameters()
            if name in ("custom", "custom_bf16"):
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + 1)
                optimizer = optim8bit.Adam8bit(
                    trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                    block=256, weight_rounding="stochastic",
                    first_moment_storage=(
                        "bfloat16" if name == "custom_bf16" else "int8"),
                    generator=generator)
            else:
                optimizer = torch.optim.AdamW(
                    trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
            torch.manual_seed(args.seed)
            return {"arm": arm, "model": model, "optimizer": optimizer,
                    "trainable": trainable}
        return build

    #: The online loss of each step's own microbatches. The fixed-batch eval
    #: measures repetition on one batch set; this is the objective the optimiser
    #: is actually asked to reduce, recorded on fresh batches every step.
    online: dict[str, list[dict]] = {name: [] for name in args.arms.split(",")}
    current = {"arm": None}

    def step(state, step_index):
        optimizer = state["optimizer"]
        model = state["model"]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(args.grad_accum):
            offset = (args.start_token
                      + (step_index * args.grad_accum + micro) * args.seq_len)
            inputs, targets = _batch(stream, offset, args.seq_len, device)
            student_logits = model(input_ids=inputs, use_cache=False).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
            loss, _ce, _kl = train_module.token_loss(
                student_logits, teacher_logits, targets, args.temperature,
                args.kl_weight)
            (loss / args.grad_accum).backward()
            losses.append({"loss": float(loss.detach()), "ce": float(_ce),
                           "kl": float(_kl)})
        online[current["arm"]].append({
            "step": step_index + 1,
            "loss": sum(item["loss"] for item in losses) / len(losses),
            "ce": sum(item["ce"] for item in losses) / len(losses),
            "kl": sum(item["kl"] for item in losses) / len(losses),
        })
        torch.nn.utils.clip_grad_norm_(state["trainable"], args.grad_clip)
        optimizer.step()
        return state

    def evaluate(state):
        return _metrics(state["model"], teacher, eval_batches, args.temperature)

    def release(state):
        del state["arm"], state["model"], state["optimizer"], state["trainable"]
        torch.cuda.empty_cache()

    def wrapped(label, name):
        def build():
            current["arm"] = label
            return make(name)()
        return build

    builders = {"custom": "custom", "custom_bf16": "custom_bf16",
                "adamw": "adamw"}
    specs = {}
    for label in args.arms.split(","):
        if label not in builders:
            raise SystemExit(f"s53-optim: unknown arm {label!r}")
        specs[label] = optim_compare.ArmSpec(
            make=wrapped(label, builders[label]),
            step=step, release=release)
    if "adamw" not in specs:
        raise SystemExit("s53-optim: the fp32 AdamW reference arm is required")
    torch.cuda.reset_peak_memory_stats()
    comparison = optim_compare.run_comparison(
        specs, evaluate, args.steps, eval_every=args.eval_every)
    summaries = {}
    for label in specs:
        if label == "adamw":
            continue
        summaries[f"{label}_minus_adamw"] = optim_compare.summarise_comparison(
            comparison, reference="adamw", candidate=label, seed=args.seed + 53)

    receipt = {
        "schema": "selvedge/s53-optimiser-comparison-run/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "revision": _git_revision(),
        "stream_manifest": args.stream_manifest,
        "stream_sha256": _sha256(os.path.join(
            ROOT, json.load(open(os.path.join(ROOT, args.stream_manifest),
                                 encoding="utf-8"))["stream_path"])),
        "start_token": args.start_token,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "eval_batches": args.batches,
        "seq_len": args.seq_len,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "temperature": args.temperature,
        "kl_weight": args.kl_weight,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "matched_init_sha256": init_digest,
        "trainable_elements": trainable_elements,
        "arms": {
            label: {
                "optimizer": ("adam8bit block 256 stochastic rounding, int8 "
                              "first moment" if label == "custom" else
                              "adam8bit block 256 stochastic rounding, "
                              "bfloat16 first moment" if label == "custom_bf16"
                              else "torch.optim.AdamW fp32 moments")}
            for label in specs},
        "comparisons": summaries,
        "online_loss": {label: trace for label, trace in online.items() if trace},
        "online_loss_note": (
            "each step's own microbatches at the training offsets, so this trace "
            "is the objective the optimiser is asked to reduce; the fixed-batch "
            "series asks whether the arms agree on one batch set"),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "command": sys.argv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s53-optim: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
