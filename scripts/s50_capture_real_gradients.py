#!/usr/bin/env python3
"""S51: capture real Qwen3.5-0.8B gradients for the first-moment validation.

This is the GPU half of the real-gradient validation. It runs at most two
forward/backward probes on the registered math stream with the pinned student and
online teacher, clips the gradients with the bar's registered `grad_clip=1.0`,
and writes the post-clip gradients of a handful of trainable tensors to a CPU
fixture. The model is **not** updated between probes: this is a gradient
sample, not a training step, and the second probe uses the next stream window so
the two gradients are different batches at the same initial weights.

The CPU harness `scripts/s50_real_gradient_validation.py` then runs 50 optimiser
steps of the shipped int8-first-moment `Adam8bit`, the optional bfloat16 variant
and fp32 AdamW on exactly these tensors.

Default selection (six tensors, ~18M elements, capped to whole rows):

    layers.0.mlp.down_proj.weight
    layers.11.mlp.down_proj.weight
    layers.23.mlp.down_proj.weight
    layers.0.mlp.gate_proj.weight
    layers.11.self_attn.q_proj.weight
    layers.23.self_attn.q_proj.weight

Early/mid/late depth, MLP down/gate and a full-attention q projection: the
tensor kinds that dominate the full arm's element count and that see the
widest gradient dynamic range. `--tensors` overrides the list; a suffix that
matches zero or more than one trainable tensor fails closed.

The fixture is a `.pt` file (gitignored); the tracked evidence is `capture.json`
beside it, with the fixture SHA-256 and the stream hash.

Queued command (do not run while the 20M bar is training):

    SELVEDGE_MEM=20g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \\
      scripts/court --gpu oracle -- python3 scripts/s50_capture_real_gradients.py \\
        --out-dir results/artifacts/stage2/s50-real-gradients \\
        --steps 2 --max-elements-per-tensor 500000

`SELVEDGE_MEM=20g` matches the bar launch: the same student + teacher are
resident, so the measured 8.76 GB cgroup peak plus the 1.5x/+1 GiB floor is
13.1 GB and 20g is the registered value.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import arms, full, train  # noqa: E402
from frontier.stream import load_stream, sha256_file  # noqa: E402

FIXTURE_SCHEMA = "selvedge/s50-real-gradient-fixture/v1"
CAPTURE_SCHEMA = "selvedge/s50-real-gradient-capture/v1"
DEFAULT_STREAM = "results/stage2/stage2-math-stream-20261117T071120Z.json"
STUDENT_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
DEFAULT_TENSORS = (
    "layers.0.mlp.down_proj.weight",
    "layers.11.mlp.down_proj.weight",
    "layers.23.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.11.self_attn.q_proj.weight",
    "layers.23.self_attn.q_proj.weight",
)
GIB = float(1 << 30)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _select_tensors(named, suffixes: tuple[str, ...]) -> dict:
    """Match each suffix to exactly one trainable parameter, fail closed.

    `named` is materialised first: `named_trainable()` yields a generator, and a
    generator would be exhausted by the first suffix, silently matching zero
    tensors for every later suffix (S55 fix; the script had never run).
    """
    named = list(named)
    selected = {}
    for suffix in suffixes:
        matches = [
            (name, parameter) for name, parameter in named
            if name.endswith(suffix)
            # The conditional checkpoint also carries an MTP head and a vision
            # tower with the same layer suffixes; neither is the text tower the
            # bar trains.
            and "mtp." not in name and "visual." not in name
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"s50-capture: suffix {suffix!r} matched {len(matches)} "
                "trainable tensors; expected exactly one")
        name, parameter = matches[0]
        if parameter.ndim < 2:
            raise SystemExit(
                f"s50-capture: {name!r} is not a matrix; the row cap needs ndim>=2")
        selected[name] = parameter
    return selected


def _cap(parameter: torch.Tensor, gradient: torch.Tensor,
         max_elements: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Keep whole leading rows so 256-element blocks stay contiguous."""
    if parameter.numel() <= max_elements:
        return parameter, gradient, {"source_shape": list(parameter.shape),
                                     "captured_shape": list(parameter.shape),
                                     "row_capped": False}
    row_elements = int(parameter[0].numel())
    rows = max(1, max_elements // row_elements)
    if rows >= parameter.shape[0]:
        return parameter, gradient, {"source_shape": list(parameter.shape),
                                     "captured_shape": list(parameter.shape),
                                     "row_capped": False}
    view = parameter.reshape(parameter.shape[0], -1)
    gradient_view = gradient.reshape(gradient.shape[0], -1)
    return (
        view[:rows].clone(),
        gradient_view[:rows].clone(),
        {
            "source_shape": list(parameter.shape),
            "captured_shape": [rows, row_elements],
            "captured_elements": int(rows * row_elements),
            "row_capped": True,
            "rows_kept": rows,
            "rows_total": int(parameter.shape[0]),
            "cap_basis": (
                "whole leading rows only, so every captured 256-element block "
                "is a real contiguous block of the source tensor"),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-manifest", default=DEFAULT_STREAM)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--start-token", type=int, default=0,
                        help="first stream offset (0 = the bar's training start)")
    parser.add_argument("--steps", type=int, default=2, choices=(1, 2))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-elements-per-tensor", type=int, default=500_000)
    parser.add_argument("--tensors", nargs="+", default=list(DEFAULT_TENSORS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=12.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit(
            "s50-capture: CUDA required; run through scripts/court --gpu oracle")
    if args.max_elements_per_tensor < 256:
        raise SystemExit("s50-capture: cap must hold at least one block")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    free_bytes, _total = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s50-capture: vram limit {args.vram_limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")

    manifest = json.load(open(args.stream_manifest, encoding="utf-8"))
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit("s50-capture: stream sha mismatch")
    stream_np = load_stream(manifest["stream_path"])
    stream = torch.from_numpy(stream_np.astype(np.int64))
    offset = int(manifest.get("graft_tokens", 0)) + args.start_token
    needed = offset + args.steps * args.seq_len * args.micro_batch + args.seq_len
    if needed > stream.numel():
        raise SystemExit("s50-capture: stream is too short for the capture")

    device = "cuda"
    print(f"s50-capture: loading student {args.student_dir}", flush=True)
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    arm = full.FullBackboneArm(model, freeze_embedding=True)
    print(f"s50-capture: loading teacher {args.teacher_dir}", flush=True)
    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in arm.parameters()
                 if parameter.requires_grad]
    selected = _select_tensors(arm.named_trainable(), tuple(args.tensors))
    torch.cuda.reset_peak_memory_stats()

    captured: dict[str, list[torch.Tensor]] = {
        name: [] for name in selected}
    probe_metadata = []
    for step in range(args.steps):
        arm.model.zero_grad(set_to_none=True)
        micro_grad_norms = []
        losses = []
        for micro in range(args.micro_batch):
            batch, _ = train.next_batch(
                stream, offset + (step * args.micro_batch + micro) * args.seq_len,
                args.seq_len)
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            student_logits = arm.model(input_ids=inputs, use_cache=False).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids=inputs,
                                         use_cache=False).logits
            loss, ce, kl = train.token_loss(
                student_logits, teacher_logits, targets, args.temperature,
                args.kl_weight)
            (loss / args.micro_batch).backward()
            losses.append({"loss": float(loss.detach()), "ce": float(ce),
                           "kl": float(kl)})
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable, args.grad_clip))
        for name, parameter in selected.items():
            if parameter.grad is None:
                raise SystemExit(f"s50-capture: {name!r} received no gradient")
            captured[name].append(
                parameter.grad.detach().reshape(-1).to(torch.float32).clone())
        probe_metadata.append({
            "probe": step + 1,
            "stream_offset": offset + step * args.micro_batch * args.seq_len,
            "pre_clip_grad_norm": pre_clip_norm,
            "clip": args.grad_clip,
            "micro_batches": args.micro_batch,
            "losses": losses,
        })
        print(f"s50-capture: probe {step + 1}/{args.steps} "
              f"pre_clip_norm={pre_clip_norm:.6f}", flush=True)

    tensors = {}
    tensor_report = []
    for name, parameter in selected.items():
        first_gradient = captured[name][0].reshape(parameter.shape)
        capped_parameter, capped_gradient, cap = _cap(
            parameter.detach(), first_gradient, args.max_elements_per_tensor)
        capped_grads = []
        for gradient in captured[name]:
            gradient = gradient.reshape(parameter.shape)
            rows = cap.get("rows_kept")
            if rows is not None:
                gradient = gradient.reshape(parameter.shape[0], -1)[:rows]
            capped_grads.append(gradient.to(torch.float32).clone())
        tensors[name] = {
            "parameter": capped_parameter.to(torch.float32).clone(),
            "grads": capped_grads,
        }
        tensor_report.append({
            "name": name,
            "elements": int(capped_parameter.numel()),
            "grad_steps": len(capped_grads),
            "grad_first_step_absmax": float(capped_grads[0].abs().max()),
            "grad_first_step_rms": float(capped_grads[0].pow(2).mean().sqrt()),
            "grad_absmax_over_steps": [
                float(gradient.abs().max()) for gradient in capped_grads],
            **cap,
        })

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(args.out_dir, exist_ok=True)
    fixture_path = os.path.join(args.out_dir, "fixture.pt")
    metadata = {
        "schema": CAPTURE_SCHEMA,
        "capture_stamp": stamp,
        "capture_revision": _revision(),
        "stream_manifest": args.stream_manifest,
        "stream_sha256": manifest["sha256"],
        "stream_path": manifest["stream_path"],
        "student_dir": args.student_dir,
        "teacher_dir": args.teacher_dir,
        "start_token": args.start_token,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "steps_captured": args.steps,
        "grad_clip": args.grad_clip,
        "gradient_timing": ("post-clip gradient of the loss the bar optimises "
                            "(CE + kl_weight * T^2 * KL); the model is not "
                            "updated between probes"),
        "parameter_dtype": "bfloat16 in the model, stored float32 in the fixture",
        "gradient_dtype": "float32 (exact widening of the bf16 gradient)",
        "probes": probe_metadata,
        "tensors": tensor_report,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "command": sys.argv,
    }
    torch.save({
        "schema": FIXTURE_SCHEMA,
        "metadata": metadata,
        "tensors": tensors,
    }, fixture_path)

    import hashlib
    digest = hashlib.sha256()
    with open(fixture_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    receipt = {
        **metadata,
        "kind": "s50-real-gradient-capture",
        "fixture_path": fixture_path,
        "fixture_sha256": digest.hexdigest(),
        "fixture_bytes": os.path.getsize(fixture_path),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    with open(os.path.join(args.out_dir, "capture.json"), "w",
              encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s50-capture: wrote {fixture_path} "
          f"({receipt['fixture_bytes']} bytes) and capture.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
