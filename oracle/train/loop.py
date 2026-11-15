#!/usr/bin/env python3
"""T1 training loop.

One arm, one run. Reads a fixed window order from `train.bin`, the shared
teacher targets, and an admitted arm config. Refuses to start without the Rust
admission record for the config it is running.

Logs one JSONL record per optimizer step (loss, LR, tokens/s, VRAM, routing
utilisation, accumulated FLOPs) and writes crash-safe checkpoints: write to a
temporary file, fsync, rename, then update `latest.json` atomically. A resume
verifies the checkpoint hash and the step-consistent data cursor.

The registered effective batch is 64 x 1024 tokens. `--micro-batch` only changes
how that batch is split across forward/backward calls; it never changes the
token count, the LR schedule or the data order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import utility
from modeling import base_logits_for, build_arm, load_admission, zero_point_report

WINDOW = 1024
RECORD_BYTES = 100
TOP_K = 16


def load_balance_loss(arm_model, coefficient: float = 0.01):
    """T1.6 Switch auxiliary loss over every routed bank.

    `coefficient * m * sum_i f_i * p_i`, summed over layers. Evaluated on the
    current micro-batch's routing. Never used by `H*`, whose router is trained
    only on measured utility labels (T2.5); the caller decides.
    """
    banks = arm_model.banks()
    if not banks:
        return None
    total = None
    for bank in banks:
        if bank.last_probs is None or bank.last_routing is None:
            continue
        probs = bank.last_probs
        tokens = probs.shape[0]
        counts = torch.bincount(bank.last_routing.reshape(-1),
                                minlength=bank.experts).to(probs.dtype)
        fraction = counts / max(1, tokens * bank.k)
        mean_prob = probs.mean(dim=0)
        term = coefficient * bank.experts * (fraction * mean_prob).sum()
        total = term if total is None else total + term
    return total


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TargetReader:
    """Per-position teacher targets from the packed cache."""

    def __init__(self, artifact_dir: str):
        self.manifest = json.load(open(os.path.join(artifact_dir, "teacher-targets.json"),
                                       encoding="utf-8"))
        path = os.path.join(artifact_dir, "teacher-targets.bin")
        self.tokens = self.manifest["tokens"]
        self.raw = np.memmap(path, dtype=np.uint8, mode="r",
                             shape=(self.manifest["bytes"],))

    def read(self, start: int, count: int, device) -> tuple[torch.Tensor, torch.Tensor,
                                                             torch.Tensor]:
        offset = start * RECORD_BYTES
        block = np.asarray(self.raw[offset:offset + count * RECORD_BYTES]).reshape(
            count, RECORD_BYTES)
        bits = block[:, 0:32].copy().view(np.uint16)
        wide = (bits.astype(np.uint32) << 16).view(np.float32).reshape(count, TOP_K)
        ids = block[:, 32:96].copy().view(np.int32).reshape(count, TOP_K)
        lse = block[:, 96:100].copy().view(np.float32).reshape(count)
        return (torch.from_numpy(wide.copy()).to(device),
                torch.from_numpy(ids.copy()).to(device),
                torch.from_numpy(lse.copy()).to(device))


def _lse_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(x.float(), dim=-1)


def chunked_logsumexp(logits: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Vocabulary-chunked log-sum-exp with checkpointed f32 chunks.

    A full f32 copy of the [tokens, vocab] logits is 2x the bf16 logits and is
    the peak-memory term. Each chunk is recomputed in the backward pass instead,
    so only one chunk's f32 temporary is live at a time; the per-chunk results
    are combined exactly (logsumexp of logsumexps).
    """
    parts = []
    for start in range(0, logits.shape[-1], chunk):
        part = logits[:, start:start + chunk]
        parts.append(torch.utils.checkpoint.checkpoint(_lse_f32, part,
                                                       use_reentrant=False))
    return torch.logsumexp(torch.stack(parts, dim=0), dim=0)


def distillation_loss(logits: torch.Tensor, values: torch.Tensor, ids: torch.Tensor,
                      lse: torch.Tensor) -> torch.Tensor:
    """Exact top-k cross-entropy with a tail bucket.

    `logits` is [N, vocab]; student logits for the positions being predicted.
    Teacher top-k logits (`values`), their indices (`ids`) and the teacher
    log-sum-exp (`lse`) are [N, k], [N, k] and [N].
    """
    return distillation_loss_per_position(logits, values, ids, lse).mean()


def distillation_loss_per_position(logits: torch.Tensor, values: torch.Tensor,
                                 ids: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """The T1.4 objective per position, `[N]`.

    The utility labels need the per-position value because a counterfactual is
    a per-token loss delta (T2.5).
    """
    student_lse = chunked_logsumexp(logits)
    student_support = logits.gather(1, ids).float()
    log_student = student_support - student_lse.unsqueeze(-1)
    p_teacher = (values - lse.unsqueeze(-1)).exp()
    p_tail = (1.0 - p_teacher.sum(dim=-1)).clamp_min(0.0)
    support_mass = log_student.exp().sum(dim=-1)
    # The student tail mass is 1 - support_mass. Its absolute resolution is the
    # f32 rounding of a 16-term sum (~1e-7), so the tail is clamped at that
    # resolution; a tail below it is treated as the smallest representable
    # mass instead of producing -inf. See docs/TRAINING.md.
    tail_mass = (1.0 - support_mass).clamp(min=1e-7, max=1.0)
    log_tail = tail_mass.log()
    return -(p_teacher * log_student).sum(dim=-1) - p_tail * log_tail


def lr_at(step: int, total_steps: int) -> float:
    warmup = 100
    peak = 1e-3
    final = 1e-4
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    checksum = sha256_file(tmp)
    os.replace(tmp, path)
    write_json_atomic(path + ".sha256", {"sha256": checksum, "bytes": os.path.getsize(path)})


def load_checkpoint(path: str, device) -> dict:
    sidecar = json.load(open(path + ".sha256", encoding="utf-8"))
    if sha256_file(path) != sidecar["sha256"]:
        raise SystemExit(f"checkpoint {path}: sha256 does not match its sidecar")
    return torch.load(path, map_location=device, weights_only=False)


def run_router_phase(arm_model, targets, stream, device, args, report_path: str) -> dict:
    """T2 phase U: label the bank by leave-one-expert-out, train the routers."""
    banks = arm_model.banks()
    if args.profile != "t2" or args.arm != "H*":
        return {}
    if not banks:
        raise SystemExit("H* router phase needs a specialist bank")
    calibration_tokens = args.calibration_windows * WINDOW
    print(f"router-phase: labelling {len(banks)} layers on {calibration_tokens} tokens "
          f"({1 + banks[0].experts} passes per layer)", flush=True)
    payload = utility.label_utilities(arm_model, targets, stream, device,
                                      calibration_tokens=calibration_tokens)
    labels_path = args.labels_out or os.path.join(args.out, "utility-labels.pt")
    utility.save_labels(labels_path, payload)
    runs = []
    for index, bank in enumerate(banks):
        run = utility.train_router(payload["features"][index],
                                   payload["utilities"][index], bank.router,
                                   steps=args.router_steps, seed=args.seed + index)
        run["layer"] = index
        runs.append(run)
    report = {
        "schema": "selvedge/t2-router-phase/v1",
        "arm": args.arm,
        "labels": labels_path,
        "labels_manifest": payload["manifest"],
        "router_runs": runs,
        "router_steps": args.router_steps,
        "calibration_tokens": calibration_tokens,
        "utility_passes": payload["manifest"]["loo_passes"],
        "note": "the router is frozen after this phase; experts and memory are "
                "trained by distillation through the frozen mask (T2.6)",
    }
    write_json_atomic(report_path, report)
    # Freeze the treatment: no distillation gradient, no load balance, no
    # further router updates.
    arm_model.set_router_grad(False)
    for bank in banks:
        for parameter in bank.router.parameters():
            parameter.requires_grad_(False)
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--profile", choices=["t1", "t2"], default="t1")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="T2 phase W length (H and H* run it identically)")
    parser.add_argument("--calibration-windows", type=int, default=32,
                        help="T2 H* utility calibration windows")
    parser.add_argument("--router-steps", type=int, default=200,
                        help="T2 H* router-head steps on the cached utility labels")
    parser.add_argument("--labels-out", default=None,
                        help="T2 H* utility label checkpoint path")
    parser.add_argument("--load-balance", action="store_true",
                        help="add the T1.6 Switch auxiliary loss (never for H*)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="activation checkpointing for the frozen core; T2 hybrid "
                             "arms need it to fit 16 GiB (recorded in the receipt)")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="windows per forward/backward call")
    parser.add_argument("--accum", type=int, default=32,
                        help="gradient accumulation steps")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--zero-point", action="store_true",
                        help="verify step-0 identity against the frozen base")
    parser.add_argument("--smoke", action="store_true",
                        help="write a smoke receipt shape and stop early checks")
    parser.add_argument("--no-admission-check", action="store_true",
                        help="only for instrument tests; never used in a registered run")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("training requires the GPU court (--gpu)")
    device = torch.device("cuda")

    config, admission = load_admission(args.config, args.admission)
    if config["arm"] != args.arm:
        raise SystemExit(f"config arm {config['arm']} != --arm {args.arm}")
    if not args.no_admission_check and not admission["admitted"]:
        raise SystemExit("arm is not admitted")
    tokens_per_step = args.micro_batch * WINDOW * args.accum
    if tokens_per_step != 64 * WINDOW:
        raise SystemExit(
            f"effective batch {tokens_per_step} tokens != registered 65536; "
            "micro-batch x accum must be 64 windows")

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.jsonl")
    set_seed(args.seed)

    revision = {"0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
                "1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"}
    size = "1.7b" if args.arm in ("A", "A0") else "0.6b"
    directory = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B"}[size]
    model_path = os.path.join(args.hf_root, "models", "Qwen", directory,
                              revision[size])
    stream_index = json.load(open(os.path.join(args.artifact_dir, "train.index.json"),
                                  encoding="utf-8"))
    stream = np.memmap(os.path.join(args.artifact_dir, "train.bin"), dtype="<u4",
                       mode="r", shape=(stream_index["tokens"],))
    targets = TargetReader(args.artifact_dir)
    budget = min(targets.tokens, args.steps * tokens_per_step)
    if args.steps * tokens_per_step > budget:
        raise SystemExit(f"budget {budget} < steps x tokens_per_step")

    base = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False, attn_implementation="sdpa")
    base.to(device)
    base.eval()
    probe_ids = None
    probe_base = None
    if args.zero_point:
        stream_probe = np.asarray(stream[0:2 * WINDOW], dtype=np.int64).reshape(2, WINDOW)
        probe_ids = torch.from_numpy(stream_probe).to(device)
        probe_base = base_logits_for(base, probe_ids)
    arm_model = build_arm(base, args.arm, config, admission, args.seed,
                          profile=args.profile)
    arm_model.model.to(device)
    for parameter in arm_model.trainable_parameters():
        parameter.requires_grad_(True)
    arm_model.model.train()
    if args.grad_checkpoint:
        arm_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        arm_model.model.enable_input_require_grads()
        for parameter in arm_model.trainable_parameters():
            parameter.requires_grad_(True)
    if args.profile == "t2" and args.arm == "H*" and args.warmup_steps > 0:
        # Phase W trains the router with the naive rule; the utility treatment
        # starts after the label step.
        arm_model.set_router_grad(True)
    router_phase = {"done": args.warmup_steps <= 0 or args.profile != "t2"
                    or args.arm != "H*", "report": None}
    if args.zero_point:
        report = zero_point_report(arm_model, probe_base, probe_ids)
        write_json_atomic(os.path.join(args.out, "zero-point.json"), report)
        headline = report.get("max_abs_logit_delta",
                              report.get("max_abs_reconstruction_delta"))
        print(f"train: zero-point max|delta|={headline:.3e} ok={report['ok']} "
              f"kind={report['kind']}")
        if not report["ok"]:
            raise SystemExit("zero-point identity check failed (K12)")
        del probe_base, probe_ids
        probe_base = None
        probe_ids = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainable = arm_model.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.0)
    start_step = 0
    if args.resume:
        state = load_checkpoint(args.resume, device)
        arm_model.model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        set_seed(args.seed + state["step"])
        start_step = state["step"]
        print(f"train: resumed from {args.resume} at step {start_step}")

    meta = {
        "schema": "selvedge/train-run/v1",
        "arm": args.arm,
        "seed": args.seed,
        "started_utc": utc_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "config": args.config,
        "config_sha256": admission["config_sha256"],
        "admission": args.admission,
        "admission_active_unique": admission["active_unique_params"],
        "admitted_selector_unique": admission["selector_unique_params"],
        "admitted_execution_unique": admission["execution_unique_params"],
        "trainable_elements": arm_model.trainable_elements(),
        "stream_sha256": stream_index["stream_sha256"],
        "teacher_manifest": targets.manifest["teacher_revision"],
        "targets_tokens": targets.tokens,
        "steps": args.steps,
        "micro_batch": args.micro_batch,
        "accum": args.accum,
        "tokens_per_step": tokens_per_step,
        "registered_budget_tokens": args.steps * tokens_per_step,
        "window": WINDOW,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "grad_checkpoint": bool(args.grad_checkpoint),
    }
    if args.profile == "t2":
        meta["profile"] = args.profile
        meta["warmup_steps"] = args.warmup_steps
        meta["load_balance"] = bool(args.load_balance)
        meta["extra_ledger"] = arm_model.extra_ledger()
        meta["active_extra_elements"] = arm_model.active_extra_elements()
        meta["declared_active"] = int(admission["active_unique_params"])
        meta["total_accessible"] = int(config["total_accessible"])
        if arm_model.memories():
            from memory import MemoryGeometry

            geometry = arm_model.memories()[0].geometry()
            meta["memory"] = {
                "insertions": len(arm_model.memories()),
                "layers": list(arm_model.memory_layers),
                "active": geometry.active_report(len(arm_model.memories())),
                "bytes": geometry.bytes_report(len(arm_model.memories())),
            }
    write_json_atomic(os.path.join(args.out, "run-meta.json"), meta)

    log = open(log_path, "a", encoding="utf-8")
    started = time.time()
    train_seconds = 0.0
    tokens_done = start_step * tokens_per_step
    scaler_report = {}
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_start = time.time()
        for micro in range(args.accum):
            offset = step * tokens_per_step + micro * args.micro_batch * WINDOW
            ids = np.asarray(stream[offset:offset + args.micro_batch * WINDOW],
                             dtype=np.int64).reshape(args.micro_batch, WINDOW)
            input_ids = torch.from_numpy(ids).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = arm_model.model(input_ids=input_ids, use_cache=False).logits
            flat_logits = logits.reshape(-1, logits.shape[-1])
            count = flat_logits.shape[0]
            values, target_ids, lse = targets.read(offset, count, device)
            loss = distillation_loss(flat_logits, values, target_ids, lse)
            if args.load_balance and args.arm != "H*":
                auxiliary = load_balance_loss(arm_model)
                if auxiliary is not None:
                    loss = loss + auxiliary
            (loss / args.accum).backward()
            step_loss += float(loss.detach()) / args.accum
            del logits, flat_logits, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        tokens_done += tokens_per_step
        elapsed = time.time() - step_start
        train_seconds += elapsed
        record = {
            "step": step + 1,
            "loss": step_loss,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens": tokens_done,
            "tokens_per_second": tokens_per_step / max(elapsed, 1e-9),
            "vram_peak_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "vram_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            "step_seconds": elapsed,
            "trainable_elements": meta["trainable_elements"],
            "admitted_active_unique": admission["active_unique_params"],
            "routing": arm_model.routing_summary() if (step + 1) % 20 == 0 else {},
        }
        log.write(json.dumps(record, sort_keys=True) + "\n")
        log.flush()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"train: step {step + 1}/{args.steps} loss={step_loss:.4f} "
                  f"tok/s={record['tokens_per_second']:.0f} "
                  f"vram={record['vram_peak_gib']:.2f}GiB", flush=True)
        if (step + 1) % args.checkpoint_every == 0 or (step + 1) == args.steps:
            state = {
                "step": step + 1,
                "model": {name: parameter.detach().cpu()
                          for name, parameter in arm_model.model.named_parameters()
                          if parameter.requires_grad},
                "optimizer": optimizer.state_dict(),
                "arm": args.arm,
                "seed": args.seed,
                "tokens": tokens_done,
            }
            path = os.path.join(args.out, f"step-{step + 1:06d}.pt")
            save_checkpoint(path, state)
            write_json_atomic(os.path.join(args.out, "latest.json"), {
                "step": step + 1, "checkpoint": path, "tokens": tokens_done})
            # Retain the latest two checkpoints (the atomic write, not a long
            # history, is the crash-safety mechanism).
            saved = sorted(name for name in os.listdir(args.out)
                           if name.startswith("step-") and name.endswith(".pt"))
            for stale in saved[:-2]:
                os.unlink(os.path.join(args.out, stale))
                sidecar = os.path.join(args.out, stale + ".sha256")
                if os.path.exists(sidecar):
                    os.unlink(sidecar)
        if (not router_phase["done"] and (step + 1) >= args.warmup_steps
                and start_step < args.warmup_steps):
            router_phase["done"] = True
            report = run_router_phase(arm_model, targets, stream, device, args,
                                      os.path.join(args.out, "router-phase.json"))
            router_phase["report"] = report
            arm_model.model.train()
    log.close()
    total = time.time() - started
    summary = {
        "schema": "selvedge/train-summary/v1",
        "arm": args.arm,
        "seed": args.seed,
        "steps": args.steps,
        "tokens": tokens_done,
        "train_seconds": train_seconds,
        "wall_seconds": total,
        "tokens_per_second": (tokens_done - start_step * tokens_per_step)
        / max(train_seconds, 1e-9),
        "vram_peak_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "trainable_elements": meta["trainable_elements"],
        "admitted_active_unique": admission["active_unique_params"],
        "finished_utc": utc_now(),
        "log": log_path,
    }
    if args.profile == "t2":
        summary["profile"] = args.profile
        summary["warmup_steps"] = args.warmup_steps
        summary["router_phase"] = router_phase["report"]
        summary["total_accessible"] = int(config["total_accessible"])
        summary["extra_ledger"] = arm_model.extra_ledger()
    write_json_atomic(os.path.join(args.out, "summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
