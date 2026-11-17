#!/usr/bin/env python3
"""Stage-2 training loop: frozen anchor, trainable memory, online teacher KL.

The core is frozen. Only the arm module (the memory or the parameter-matched
dense adapter) receives gradients. The objective is

    L = CE(student, next token) + lambda * T^2 * KL(teacher_T || student_T)

where the teacher distribution is computed **online** on the same batch — no
logit cache is built or read. `T` and `lambda` are command-line values written
into the run manifest before the first step; they are not tuned here.

Discipline recorded per run:

* the memory table is **not** in the active union; only the 8 rows read per
  token are. The projections and the convolution are. `active_per_token` comes
  from `memory.MemoryGeometry`, never from a parameter count;
* the data order is the pinned stream in order (no shuffling in the smoke);
  resume continues at the recorded token offset, so a resumed run sees exactly
  the same tokens;
* the checkpoint carries the arm state, optimiser, step, offset, RNG states, the
  stream hash and the config hash. `--resume latest` verifies the hashes before
  loading;
* every step logs loss, CE, KL, learning rate, tokens/s, peak allocated and
  reserved VRAM, and (periodically) the measured distinct-identity count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, memory  # noqa: E402
from frontier.stream import load_stream, sha256_file  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-2B-Base",
                           "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
GIB = float(1 << 30)


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


def config_hash(config: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def cgroup_memory() -> dict:
    """cgroup v2 peak/limit/events, read from inside the court container."""
    root = "/sys/fs/cgroup"
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    events = {}
    try:
        with open(os.path.join(root, "memory.events"), encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                events[key] = int(value)
    except (OSError, ValueError):
        events = {}
    report["memory_events"] = events
    return report


# ---------------------------------------------------------------------------
# objective


def token_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor | None,
               targets: torch.Tensor, temperature: float, kl_weight: float,
               vocab_chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CE plus full-vocabulary online KL, chunked over positions for memory."""
    flat_student = student_logits.reshape(-1, student_logits.shape[-1])
    flat_targets = targets.reshape(-1)
    ce_total = torch.zeros((), device=student_logits.device)
    positions = 0
    for start in range(0, flat_student.shape[0], vocab_chunk):
        s = flat_student[start:start + vocab_chunk].float()
        t = flat_targets[start:start + vocab_chunk]
        ce_total = ce_total + F.cross_entropy(s, t, reduction="sum")
        positions += s.shape[0]
    ce = ce_total / max(1, positions)
    if teacher_logits is None or kl_weight == 0.0:
        zero = torch.zeros((), device=student_logits.device)
        return ce, ce.detach(), zero
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    total = torch.zeros((), device=student_logits.device)
    count = 0
    for start in range(0, flat_student.shape[0], vocab_chunk):
        s = flat_student[start:start + vocab_chunk].float() / temperature
        t = flat_teacher[start:start + vocab_chunk].float() / temperature
        log_s = F.log_softmax(s, dim=-1)
        log_t = F.log_softmax(t, dim=-1)
        total = total + (log_t.exp() * (log_t - log_s)).sum()
        count += s.shape[0]
    kl = total / max(1, count)
    return ce + kl_weight * (temperature ** 2) * kl, ce.detach(), kl.detach()


# ---------------------------------------------------------------------------
# arm construction


def build_arm(kind: str, model, args, device: str):
    if kind == "memory":
        checkpoint = torch.load(args.memory_init, map_location="cpu",
                                weights_only=False)
        config = checkpoint["config"]
        if config["sites"] != [args.site]:
            raise SystemExit(f"memory-init site {config['sites']} != --site {args.site}")
        module = memory.ConditionalMemory(
            hidden=config["hidden"], value_dim=config["value_dim"],
            rows=config["rows"], sites=tuple(config["sites"]),
            conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
            seed=config["seed"])
        # graft_counts is non-persistent and therefore not in the state dict
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        module = module.to(device)
        handles = arms.install_memory(model, module)
        report = module.active_parameter_report()
        report["graft"] = checkpoint["graft"]
        report["init_checkpoint"] = args.memory_init
        report["init_checkpoint_sha256"] = sha256_file(args.memory_init)
        return module, handles, report
    if kind == "adapter":
        module = arms.DenseAdapter(hidden=args.hidden).to(device)
        handles = arms.install_adapter(model, module, (args.site,))
        report = module.active_parameter_report()
        report["matched_memory_active"] = memory.MemoryGeometry(
            hidden=args.hidden).active_per_token_per_site
        report["match_delta"] = (module.active_elements
                                 - report["matched_memory_active"])
        return module, handles, report
    raise SystemExit(f"unknown trainable arm {kind!r}")


# ---------------------------------------------------------------------------
# batch and loop


def next_batch(stream: torch.Tensor, offset: int, seq_len: int) -> tuple[torch.Tensor, int]:
    end = offset + seq_len + 1
    if end > stream.numel():
        raise SystemExit("stream exhausted")
    chunk = stream[offset:end].to(torch.int64)
    return chunk[None, :], offset + seq_len


def run(args) -> int:
    started = time.time()
    if not torch.cuda.is_available():
        raise SystemExit("train: CUDA required; run through scripts/court --gpu oracle")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(f"train: vram limit {args.vram_limit_gib} GiB exceeds free "
                         f"{free_bytes / GIB:.2f} GiB")
    if args.micro_batch * args.seq_len * args.grad_accum > args.tokens:
        raise SystemExit("train: token budget smaller than one optimiser step")

    manifest = json.load(open(args.stream_manifest))
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit("train: stream sha mismatch")
    stream_np = load_stream(manifest["stream_path"])
    stream = torch.from_numpy(stream_np.astype(np.int64))
    train_offset = int(manifest["graft_tokens"])
    train_end = int(stream.numel())
    available = train_end - train_offset
    if args.tokens > available:
        raise SystemExit(f"train: asked {args.tokens} tokens, stream has {available}")

    print(f"train: loading student {args.student_dir}", flush=True)
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    frozen = arms.freeze_backbone(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    arm, handles, arm_report = build_arm(args.arm, model, args, device)
    if args.arm == "memory":
        for parameter in arm.parameters():
            parameter.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()

    teacher = None
    teacher_params = 0
    if args.teacher_dir and args.kl_weight > 0.0:
        print(f"train: loading teacher {args.teacher_dir}", flush=True)
        teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
        teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    trainable = [parameter for parameter in arm.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    steps = args.tokens // tokens_per_step
    warmup = min(args.warmup, max(1, steps // 10))

    config = {
        "schema": "selvedge/stage2-train-config/v1",
        "arm": args.arm,
        "seed": args.seed,
        "tokens": args.tokens,
        "tokens_per_step": tokens_per_step,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "steps": steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup,
        "temperature": args.temperature,
        "kl_weight": args.kl_weight,
        "grad_clip": args.grad_clip,
        "site": args.site,
        "stream_sha256": manifest["sha256"],
        "student_dir": args.student_dir,
        "teacher_dir": args.teacher_dir if teacher is not None else None,
        "teacher_parameters": teacher_params,
        "deterministic_algorithms": True,
        "data_order": "pinned stream order; no shuffling",
        "memory_table_in_active_union": False,
    }
    config["config_sha256"] = config_hash(config)
    os.makedirs(args.run_dir, exist_ok=True)
    write_json(os.path.join(args.run_dir, "manifest.json"), {
        **config, "created_utc": utcnow(), "revision": revision(),
        "frozen_backbone_elements": int(frozen),
        "trainable_arm_elements": int(sum(p.numel() for p in arm.parameters())),
        "arm_accounting": arm_report,
        "gpu_free_gib": free_bytes / GIB,
        "gpu_total_gib": total_bytes / GIB,
        "vram_limit_gib": args.vram_limit_gib,
        "command": sys.argv,
    })

    start_step = 0
    offset = train_offset
    if args.resume:
        path = args.resume
        if path == "latest":
            path = os.path.join(args.run_dir, "latest.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config_sha256"] != config["config_sha256"]:
            raise SystemExit("train: resume config hash mismatch")
        if payload["stream_sha256"] != manifest["sha256"]:
            raise SystemExit("train: resume stream hash mismatch")
        arm.load_state_dict(payload["arm_state"])
        arm.to(device)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        start_step = int(payload["step"])
        offset = int(payload["offset"])
        print(f"train: resumed step={start_step} offset={offset}", flush=True)

    log_path = os.path.join(args.run_dir, "log.jsonl")
    log = open(log_path, "a" if start_step else "w", encoding="utf-8")
    step_time = 0.0
    run_tokens = start_step * tokens_per_step
    global_started = time.time()
    for step in range(start_step, steps):
        lr = args.lr * (min(1.0, (step + 1) / warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        kl_sum = 0.0
        loss_sum = 0.0
        step_started = time.time()
        for micro in range(args.grad_accum):
            batch, _ = next_batch(stream, offset + micro * args.seq_len, args.seq_len)
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            student = model(input_ids=inputs, use_cache=False).logits
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
            loss, ce, kl = token_loss(student, teacher_logits, targets,
                                      args.temperature, args.kl_weight)
            (loss / args.grad_accum).backward()
            loss_sum += float(loss.detach())
            ce_sum += float(ce)
            kl_sum += float(kl)
        offset = offset + tokens_per_step
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        step_time = time.time() - step_started
        run_tokens += tokens_per_step
        peak_alloc = torch.cuda.max_memory_allocated() / GIB
        peak_reserved = torch.cuda.max_memory_reserved() / GIB
        if peak_alloc > args.vram_limit_gib:
            raise RuntimeError(f"train: VRAM peak {peak_alloc:.2f} GiB over limit "
                               f"{args.vram_limit_gib:.2f} GiB")
        entry = {
            "step": step + 1,
            "tokens_seen": run_tokens,
            "loss": loss_sum / args.grad_accum,
            "ce": ce_sum / args.grad_accum,
            "kl": kl_sum / args.grad_accum,
            "lr": lr,
            "step_seconds": round(step_time, 4),
            "tokens_per_second": round(tokens_per_step / step_time, 1),
            "peak_allocated_gib": round(peak_alloc, 3),
            "peak_reserved_gib": round(peak_reserved, 3),
            "active_per_token": arm_report.get("active_per_token"),
        }
        if step % args.collision_every == 0 and args.arm == "memory":
            sample = stream[offset - args.seq_len:offset]
            collision = arm.collision_report(sample[None, :])
            entry["distinct_rows_in_batch"] = collision["distinct_rows_total"]
            entry["distinct_identities_in_batch"] = collision[
                "distinct_identities_total"]
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if step % args.log_every == 0:
            print(f"train: step {step + 1}/{steps} loss={entry['loss']:.4f} "
                  f"ce={entry['ce']:.4f} kl={entry['kl']:.4f} "
                  f"{entry['tokens_per_second']:.0f} tok/s "
                  f"peak={peak_alloc:.2f}GiB", flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == steps:
            payload = {
                "schema": "selvedge/stage2-ckpt/v1",
                "step": step + 1,
                "offset": offset,
                "tokens_seen": run_tokens,
                "config_sha256": config["config_sha256"],
                "stream_sha256": manifest["sha256"],
                "arm": args.arm,
                "arm_state": arm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }
            ckpt = os.path.join(args.run_dir, f"ckpt-{step + 1:06d}.pt")
            torch.save(payload, ckpt)
            torch.save(payload, os.path.join(args.run_dir, "latest.pt"))
            print(f"train: checkpoint {ckpt}", flush=True)

    log.close()
    wall = time.time() - global_started
    receipt = {
        "schema": "selvedge/stage2-train/v1",
        "kind": "train",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "arm": args.arm,
        "config": config,
        "steps_run": steps - start_step,
        "tokens_trained": run_tokens - start_step * tokens_per_step,
        "tokens_seen_total": run_tokens,
        "wall_seconds": round(wall, 3),
        "tokens_per_second": round(
            (run_tokens - start_step * tokens_per_step) / wall, 1),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "vram_limit_gib": args.vram_limit_gib,
        "frozen_backbone_elements": int(frozen),
        "trainable_arm_elements": int(sum(p.numel() for p in arm.parameters())),
        "arm_accounting": arm_report,
        "final_checkpoint": os.path.join(args.run_dir, f"ckpt-{steps:06d}.pt"),
        "final_checkpoint_sha256": sha256_file(
            os.path.join(args.run_dir, f"ckpt-{steps:06d}.pt")),
        "container_memory": cgroup_memory(),
        "log": log_path,
        "command": sys.argv,
    }
    receipt_path = os.path.join(args.run_dir, "receipt.json")
    write_json(receipt_path, receipt)
    arms.uninstall(handles)
    print(f"train: receipt {receipt_path}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["memory", "adapter"], required=True)
    parser.add_argument("--stream-manifest", required=True)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--teacher-dir", default=TEACHER_DIR)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--site", type=int, default=memory.DEFAULT_SITE)
    parser.add_argument("--hidden", type=int, default=memory.DEFAULT_HIDDEN)
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--collision-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=12.0)
    args = parser.parse_args(argv)
    if args.arm == "memory" and not args.memory_init:
        raise SystemExit("--memory-init is required for --arm memory")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
