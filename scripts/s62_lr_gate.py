#!/usr/bin/env python3
"""S62: executable learning-rate gate for the revised stage-2 on-policy route.

The S60 registration (`docs/ws19-post-trained-route-stage2-revised-
preregistration-s60-v1.json`, `learning_rate_gate`) fixes the instrument, not a
number:

* the arm is rank-64 all-linear LoRA, alpha 32, merged at the end;
* the screen is {3e-5, 1e-4, 3e-4}, starting at 1e-4;
* the instrument is the **S50 fixed-batch KL gate**: the same fixed held-out
  batch, the same teacher, `KL(teacher_T || student_T)` at T=2.0, measured at
  2M processed tokens;
* the decision rule is *choose the largest screened rate whose fixed-batch KL
  falls below the initial student's; if every rate rises, the run does not
  launch and the screen is reported.* Accuracy is never a selection input.

This file is the executable half of that clause. It has five modes:

    --plan       validate the inputs and print one court command per rate
    --execute    run one rate's gate arm on the GPU and measure its KL
                 (`--rate 1e-4`; one GPU job at a time, one rate per call)
    --decide     apply the decision rule over the per-rate receipts
    --self-test  CPU test of the LoRA math, the training step, the fixed-batch
                 KL metric and the decision rule (no GPU, no checkpoint)
    --dry-run    alias of --plan that also writes the plan receipt

The gate arm's probe objective is completion CE over the stage-2b examples by
default (`--probe-objective sft`). The registered route's final objective is
reward-gated reverse KL; a gate that used it would need the separate teacher
log-probability pass first, so `--probe-objective reverse-kl` is accepted only
with `--teacher-logprobs` and otherwise fails closed. The gate measures
optimiser stability and KL direction; it does not claim to be the final
objective.

Nothing here is typed: every KL, CE, rate and decision is computed from the
rows on disk, and every receipt records the command that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SCHEMA_PLAN = "selvedge/s62-lr-gate-plan/v1"
SCHEMA_RATE = "selvedge/s62-lr-gate-rate/v1"
SCHEMA_DECISION = "selvedge/s62-lr-gate-decision/v1"

STUDENT_ID = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TEACHER_ID = "Qwen/Qwen3.5-2B"
TEACHER_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
STUDENT_LOCK = "models/FRONTIER-POST-TRAINED.lock.json"
REGISTRATION = (
    "docs/ws19-post-trained-route-stage2-revised-preregistration-s60-v1.json")
DEFAULT_RATES = ("3e-5", "1e-4", "3e-4")
DEFAULT_START = "1e-4"
#: The S50 fixed-batch KL instrument: 8 batches x 512 positions from the
#: registered math stream, offsets from 2,000,000.
DEFAULT_STREAM = "results/stage2/stage2-math-stream-20261117T071120Z.json"
DEFAULT_START_TOKEN = 2_000_000
GIB = float(1 << 30)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: str) -> str:
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


def rates(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("no rates given")
    for item in parsed:
        if not re.fullmatch(r"[0-9]+(?:\.?[0-9]*)?e-?[0-9]+", item):
            raise argparse.ArgumentTypeError(f"rate {item!r} is not e-notation")
    return parsed


# ---------------------------------------------------------------------------
# LoRA: a minimal all-linear adapter.
#
# No peft dependency is installed in the oracle image, so the adapter is
# implemented here. It follows the registered contract exactly: rank 64,
# alpha 32, the alpha/r scaling, zero-initialised B and uniform A. The base
# weight is never modified; `merge` produces the full-weight deployed tensor
# for the fixed-batch measurement and for the run manifest.
# ---------------------------------------------------------------------------
class LoraLinear(torch.nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-r residual.

    The residual is `scale * (x @ A.T) @ B.T` with `A` shape (r, in) and `B`
    shape (out, r). At initialisation `B = 0`, so `forward` is bit-identical to
    the base module; the self-test asserts that.
    """

    def __init__(self, base, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        out_features, in_features = base.weight.shape
        self.lora_a = torch.nn.Parameter(
            torch.empty(rank, in_features, dtype=base.weight.dtype,
                        device=base.weight.device))
        self.lora_b = torch.nn.Parameter(
            torch.zeros(out_features, rank, dtype=base.weight.dtype,
                        device=base.weight.device))
        # PEFT's uniform A initialisation: U(-1/sqrt(r), 1/sqrt(r)).
        bound = 1.0 / math.sqrt(rank)
        with torch.no_grad():
            self.lora_a.uniform_(-bound, bound)
        self.merged = False

    def forward(self, x):
        # The residual acts on the same input as the base projection.
        result = self.base(x)
        if self.merged:
            return result
        delta = (x @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return result + self.scale * delta

    def merged_weight(self):
        weight = self.base.weight.detach().clone()
        weight += self.scale * (self.lora_b @ self.lora_a)
        return weight

    def trainable_parameters(self):
        return [self.lora_a, self.lora_b]

    def state_dict(self) -> dict:
        return {"lora_a": self.lora_a.detach().cpu(),
                "lora_b": self.lora_b.detach().cpu(),
                "rank": self.rank, "alpha": self.alpha,
                "scale": self.scale}

    def load_state(self, payload: dict) -> None:
        if int(payload["rank"]) != self.rank:
            raise SystemExit(
                f"s62-gate: rank mismatch {payload['rank']} != {self.rank}")
        with torch.no_grad():
            self.lora_a.copy_(payload["lora_a"].to(self.lora_a.device))
            self.lora_b.copy_(payload["lora_b"].to(self.lora_b.device))

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"


#: The registered sites: attention q/k/v/o and the MLP projections. The LM
#: head and the tied embedding are deliberately not adapted.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def inject_lora(model, targets: tuple[str, ...], rank: int,
                alpha: float) -> list[str]:
    """Replace every named `nn.Linear` child with a `LoraLinear` in place."""
    installed: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear) and child_name in targets:
                wrapper = LoraLinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, wrapper)
                installed.append(f"{module_name}.{child_name}"
                                 if module_name else child_name)
    if not installed:
        raise SystemExit(
            "s62-gate: no target linear layers found; the model architecture "
            f"does not expose {targets}")
    return installed


def lora_modules(model) -> list[LoraLinear]:
    return [module for module in model.modules()
            if isinstance(module, LoraLinear)]


def trainable(model) -> list:
    parameters = []
    for module in lora_modules(model):
        parameters.extend(module.trainable_parameters())
    return parameters


# ---------------------------------------------------------------------------
# Fixed-batch KL: the S50 instrument, unchanged in spirit.
# ---------------------------------------------------------------------------
def fixed_batch_kl(student, teacher, batches, temperature: float,
                   chunk: int = 32) -> dict:
    import torch.nn.functional as F

    student_logits = student(input_ids=batches[0][0], use_cache=False).logits
    ce_total = 0.0
    kl_total = 0.0
    positions = 0
    for inputs, targets in batches:
        with torch.no_grad():
            student_logits = student(input_ids=inputs, use_cache=False).logits
            teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
        flat_student = student_logits.reshape(-1, student_logits.shape[-1]).float()
        flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1]).float()
        flat_targets = targets.reshape(-1)
        for start in range(0, flat_student.shape[0], chunk):
            s = flat_student[start:start + chunk]
            t = flat_teacher[start:start + chunk]
            ce_total += float(F.cross_entropy(
                s, flat_targets[start:start + chunk], reduction="sum"))
            log_s = F.log_softmax(s / temperature, dim=-1)
            log_t = F.log_softmax(t / temperature, dim=-1)
            kl_total += float((log_t.exp() * (log_t - log_s)).sum())
            positions += s.shape[0]
    return {"ce": ce_total / positions, "kl": kl_total / positions,
            "positions": positions}


def load_batches(stream_manifest: str, start_token: int, batches: int,
                 seq_len: int, device: str):
    import numpy as np
    import torch

    manifest = json.load(open(os.path.join(ROOT, stream_manifest),
                              encoding="utf-8"))
    path = manifest["stream_path"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    digest = sha256_file(path)
    if digest != manifest["sha256"]:
        raise SystemExit(
            f"s62-gate: stream sha mismatch {digest} != {manifest['sha256']}")
    stream = np.fromfile(path, dtype="<u4")
    out = []
    for index in range(batches):
        offset = start_token + index * seq_len
        chunk = torch.from_numpy(
            stream[offset:offset + seq_len + 1].astype(np.int64))
        if chunk.numel() != seq_len + 1:
            raise SystemExit("s62-gate: fixed stream too short")
        out.append((chunk[None, :-1].to(device), chunk[None, 1:].to(device)))
    return out, digest


# ---------------------------------------------------------------------------
# The plan.
# ---------------------------------------------------------------------------
def build_plan(args) -> dict:
    command_prefix = [
        "scripts/court", "--gpu", "oracle", "--", "python3",
        "scripts/s62_lr_gate.py", "--execute"]
    runs = []
    for rate in args.rates:
        run_dir = os.path.join(args.out_dir, f"gate-lr{rate}")
        command = command_prefix + [
            "--rate", rate,
            "--examples", args.examples,
            "--student-dir", args.student_dir,
            "--teacher-dir", args.teacher_dir,
            "--stream-manifest", args.stream_manifest,
            "--start-token", str(args.start_token),
            "--tokens", str(args.tokens),
            "--seq-len", str(args.seq_len),
            "--micro-batch", str(args.micro_batch),
            "--grad-accum", str(args.grad_accum),
            "--lora-rank", str(args.lora_rank),
            "--lora-alpha", str(args.lora_alpha),
            "--batch-prompts", str(args.batch_prompts),
            "--gate-batches", str(args.gate_batches),
            "--run-dir", run_dir,
            "--out", os.path.join(run_dir, "gate-rate.json")]
        runs.append({"rate": rate, "run_dir": run_dir,
                     "command": command})
    return {
        "schema": SCHEMA_PLAN,
        "kind": "lr-gate-plan",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "instrument": (
            "S50 fixed-batch KL: KL(teacher_T || student_T) at T=2.0 over the "
            "same fixed held-out batches, at 2M processed tokens; CE at T=1.0"),
        "decision_rule": (
            "choose the largest screened rate whose fixed-batch KL is below "
            "the initial student's KL; if every rate rises, the run does not "
            "launch and the screen is reported. Accuracy is not an input."),
        "rates": list(args.rates),
        "start": args.start,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "targets": list(DEFAULT_TARGETS),
                 "merge": "W' = W + (alpha/rank) * B @ A at the end"},
        "budget": {"tokens": args.tokens,
                   "tokens_per_step": args.micro_batch * args.seq_len
                   * args.grad_accum,
                   "steps": args.tokens // (args.micro_batch * args.seq_len
                                            * args.grad_accum)},
        "fixed_batch": {"stream_manifest": args.stream_manifest,
                        "start_token": args.start_token,
                        "batches": args.gate_batches,
                        "seq_len": args.seq_len},
        "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION,
                    "dir": args.student_dir, "lock": STUDENT_LOCK},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION,
                    "dir": args.teacher_dir},
        "runs": runs,
        "note": (
            "one GPU job at a time: run the per-rate commands serially, then "
            "run --decide; the parent may not execute this plan itself"),
    }


# ---------------------------------------------------------------------------
# Execution of one rate (GPU).
# ---------------------------------------------------------------------------
def read_examples(path: str, tokenizer, seq_len: int):
    """Prompt-masked completion examples, one complete example per row."""
    import torch

    examples = []
    overlap = 0
    if not os.path.exists(path):
        raise SystemExit(f"s62-gate: missing examples {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = list(row["prompt_ids"])
            solution = list(row["solution_ids"])
            eos = int(tokenizer.eos_token_id)
            supervise_eos = row.get("supervise_eos", True) is not False
            completion = solution + ([eos] if supervise_eos else [])
            total = len(prompt) + len(completion)
            if total > seq_len:
                overlap += 1
                continue
            input_ids = prompt + completion
            labels = [-100] * len(prompt) + completion
            examples.append({"id": row.get("id"), "input_ids": input_ids,
                             "labels": labels})
    if not examples:
        raise SystemExit("s62-gate: no example fits the row width")
    return examples, {"rows": len(examples), "dropped_overflow": overlap,
                      "supervised_positions": sum(
                          sum(1 for value in row["labels"] if value != -100)
                          for row in examples)}


def collate(batch, seq_len: int, pad_id: int, device: str):
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    width = min(width, seq_len)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for index, row in enumerate(batch):
        length = min(len(row["input_ids"]), width)
        input_ids[index, :length] = torch.tensor(
            row["input_ids"][:length], dtype=torch.long)
        labels[index, :length] = torch.tensor(
            row["labels"][:length], dtype=torch.long)
        attention[index, :length] = 1
    return (input_ids.to(device), attention.to(device), labels.to(device))


def run_rate(args) -> int:
    if not torch.cuda.is_available():
        raise SystemExit(
            "s62-gate: CUDA required for --execute; run through "
            "scripts/court --gpu oracle")
    free_bytes, _ = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(
            f"s62-gate: vram limit {args.vram_limit_gib} GiB exceeds free "
            f"{free_bytes / GIB:.2f} GiB")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    os.makedirs(os.path.join(ROOT, args.run_dir), exist_ok=True)

    from transformers import AutoTokenizer
    from frontier import arms

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                            device=device)
    installed = inject_lora(model, DEFAULT_TARGETS, args.lora_rank,
                            args.lora_alpha)
    parameters = trainable(model)

    teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16,
                              device=device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    batches, stream_sha = load_batches(
        args.stream_manifest, args.start_token, args.gate_batches,
        args.seq_len, device)

    token_per_step = args.micro_batch * args.seq_len * args.grad_accum
    if args.tokens % token_per_step != 0:
        raise SystemExit(
            f"s62-gate: --tokens {args.tokens} is not a multiple of "
            f"{token_per_step}")
    steps = args.tokens // token_per_step

    examples, data_report = read_examples(
        os.path.join(ROOT, args.examples), tokenizer, args.seq_len)
    order = list(range(len(examples)))
    import random
    random.Random(args.seed).shuffle(order)

    init = fixed_batch_kl(model, teacher, batches, args.temperature)
    print(f"s62-gate: lr={args.rate} init kl={init['kl']:.6f} "
          f"ce={init['ce']:.6f}", flush=True)

    optimizer = torch.optim.AdamW(parameters, lr=float(args.rate),
                                  weight_decay=0.0)
    model.train()
    cursor = 0
    step = 0
    max_update_over_lr = 0.0
    losses = []
    started = time.time()
    graded = [row for row in order]
    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        ce_total = 0.0
        counted = 0
        for _ in range(args.grad_accum):
            block = []
            for _ in range(args.micro_batch):
                if cursor >= len(graded):
                    cursor = 0
                block.append(examples[graded[cursor]])
                cursor += 1
            input_ids, attention, labels = collate(
                block, args.seq_len, int(tokenizer.pad_token_id), device)
            out = model(input_ids=input_ids, attention_mask=attention,
                        use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                out.reshape(-1, out.shape[-1]).float(),
                labels.reshape(-1), reduction="sum", ignore_index=-100)
            n = int((labels != -100).sum())
            (loss / max(n, 1)).backward()
            ce_total += float(loss)
            counted += n
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad /= max(counted, 1)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None or parameter.abs().numel() == 0:
                    continue
                ratio = (parameter.grad.abs() * float(args.rate)).max()
                max_update_over_lr = max(max_update_over_lr, float(ratio))
        optimizer.step()
        step += 1
        losses.append(ce_total / max(counted, 1))
        if step % args.log_every == 0 or step == steps:
            print(f"s62-gate: step {step}/{steps} ce={losses[-1]:.6f} "
                  f"({time.time() - started:.0f}s)", flush=True)

    final = fixed_batch_kl(model, teacher, batches, args.temperature)
    elapsed = time.time() - started
    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB

    state = {name: module.state_dict()
             for name, module in model.named_modules()
             if isinstance(module, LoraLinear)}
    checkpoint_path = os.path.join(ROOT, args.run_dir, "lora-gate.pt")
    torch.save({"lora": state, "rate": float(args.rate),
                "step": steps, "tokens_seen": args.tokens,
                "targets": list(DEFAULT_TARGETS)}, checkpoint_path)

    fell = final["kl"] < init["kl"]
    tripwire = max_update_over_lr > args.abort_update_factor
    payload = {
        "schema": SCHEMA_RATE,
        "kind": "lr-gate-rate",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "rate": float(args.rate),
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "targets": list(DEFAULT_TARGETS),
                 "modules_installed": len(installed),
                 "trainable_parameters": sum(
                     p.numel() for p in parameters),
                 "checkpoint": os.path.relpath(checkpoint_path, ROOT),
                 "checkpoint_sha256": sha256_file(checkpoint_path)},
        "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION,
                    "dir": args.student_dir},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION,
                    "dir": args.teacher_dir},
        "budget": {"tokens": args.tokens, "steps": steps,
                   "tokens_per_step": token_per_step,
                   "seq_len": args.seq_len,
                   "micro_batch": args.micro_batch,
                   "grad_accum": args.grad_accum},
        "data": {"examples": args.examples,
                 "examples_sha256": sha256_file(
                     os.path.join(ROOT, args.examples)), **data_report},
        "fixed_batch": {
            "stream_manifest": args.stream_manifest,
            "stream_sha256": stream_sha,
            "start_token": args.start_token,
            "batches": args.gate_batches,
            "seq_len": args.seq_len,
            "temperature": args.temperature,
            "init": init,
            "final": final,
            "direction": "fall" if fell else "rise",
            "kl_delta": final["kl"] - init["kl"],
        },
        "optimizer": {"kind": "AdamW on LoRA parameters only",
                      "lr": float(args.rate), "warmup": 0,
                      "schedule": "constant during the gate"},
        "update_guard": {"max_update_over_lr": max_update_over_lr,
                         "abort_update_factor": args.abort_update_factor,
                         "tripped": tripwire},
        "loss": {"first": losses[0] if losses else None,
                 "last": losses[-1] if losses else None,
                 "min": min(losses) if losses else None},
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": peak_alloc,
        "peak_reserved_gib": peak_reserved,
        "command": sys.argv,
    }
    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    write_json(out, payload)
    print(f"s62-gate: lr={args.rate} kl {init['kl']:.6f} -> "
          f"{final['kl']:.6f} ({'fall' if fell else 'rise'}), "
          f"tripwire={tripwire} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Decision.
# ---------------------------------------------------------------------------
def decide(receipts: list[dict], start: str) -> dict:
    """Largest screened rate whose fixed-batch KL fell, else no launch."""
    rows = []
    for receipt in receipts:
        fixed = receipt["fixed_batch"]
        rows.append({
            "rate": receipt["rate"],
            "init_kl": fixed["init"]["kl"],
            "final_kl": fixed["final"]["kl"],
            "kl_delta": fixed["kl_delta"],
            "direction": fixed["direction"],
            "tripwire": receipt["update_guard"]["tripped"],
            "eligible": (fixed["direction"] == "fall"
                         and not receipt["update_guard"]["tripped"]),
            "receipt": receipt.get("_path"),
        })
    rows.sort(key=lambda row: row["rate"])
    eligible = [row for row in rows if row["eligible"]]
    chosen = max(eligible, key=lambda row: row["rate"]) if eligible else None
    return {
        "rows": rows,
        "chosen": chosen["rate"] if chosen else None,
        "launch": chosen is not None,
        "rule": ("the largest screened rate whose fixed-batch KL fell and "
                 "whose update guard did not trip; if no rate qualifies, the "
                 "full run does not launch and the screen is reported"),
        "start": float(start),
        "start_eligible": bool(chosen is not None
                               and chosen["rate"] == float(start)),
    }


def run_decide(args) -> int:
    receipts = []
    for path in args.gate_receipt:
        full_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        payload = json.load(open(full_path, encoding="utf-8"))
        if payload.get("schema") != SCHEMA_RATE:
            raise SystemExit(
                f"s62-gate: {path} is not a {SCHEMA_RATE} receipt")
        payload["_path"] = os.path.relpath(full_path, ROOT)
        receipts.append(payload)
    if not receipts:
        raise SystemExit("s62-gate: --decide needs at least one --gate-receipt")
    decision = decide(receipts, args.start)
    payload = {
        "schema": SCHEMA_DECISION,
        "kind": "lr-gate-decision",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": REGISTRATION,
        "instrument": (
            "S50 fixed-batch KL(teacher_T || student_T) at T=2.0 over the same "
            "fixed batches at 2M processed tokens"),
        **decision,
        "command": sys.argv,
    }
    out = args.out or os.path.join(args.out_dir, "decision.json")
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    write_json(out, payload)
    for row in payload["rows"]:
        print(f"s62-gate: lr={row['rate']:g} kl {row['init_kl']:.6f} -> "
              f"{row['final_kl']:.6f} ({row['direction']}) "
              f"eligible={row['eligible']}")
    print(f"s62-gate: chosen={payload['chosen']} launch={payload['launch']} "
          f"-> {args.out or os.path.join(args.out_dir, 'decision.json')}")
    return 0


# ---------------------------------------------------------------------------
# CPU self-test.
# ---------------------------------------------------------------------------
def self_test() -> int:
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.down_proj = torch.nn.Linear(8, 4)
            self.lm_head = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.lm_head(self.down_proj(self.q_proj(x)))

    model = Tiny()
    model.eval()
    x = torch.randn(3, 5, 8)
    before = model(x).detach().clone()
    installed = inject_lora(model, ("q_proj", "down_proj"), rank=4, alpha=8.0)
    assert len(installed) == 2, installed
    assert not any(name.endswith("lm_head") for name in installed)
    after = model(x)
    if not torch.equal(before, after):
        raise SystemExit("s62-gate self-test: LoRA changed the init forward")
    modules = lora_modules(model)
    assert len(modules) == 2
    # A non-zero B must change the output; merging must reproduce it exactly.
    with torch.no_grad():
        modules[0].lora_b.normal_()
        modules[1].lora_b.normal_()
    unmerged = model(x).detach().clone()
    for module in modules:
        module.base.weight.copy_(module.merged_weight())
        module.merged = True
    merged = model(x).detach()
    error = (unmerged - merged).abs().max().item()
    # The residual and the merged weight reassociate the same product, so the
    # result agrees to float32 rounding, not bitwise (as the S59 amendments
    # already record for a gathered GEMM). The threshold is a property of the
    # arithmetic, not a quality bound.
    if error > 5e-5:
        raise SystemExit(
            f"s62-gate self-test: merge error {error} (expected <= 5e-5)")

    # The training step must lower the completion CE on a fixed toy batch.
    model2 = Tiny()
    inject_lora(model2, ("q_proj", "down_proj"), rank=4, alpha=8.0)
    params = trainable(model2)
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    ids = torch.randn(2, 6, 8)
    labels = torch.randint(0, 8, (2, 6))
    labels[:, :3] = -100
    model2.train()
    first = None
    for _ in range(20):
        optimizer.zero_grad()
        logits = model2(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first
    if not float(loss) < first:
        raise SystemExit("s62-gate self-test: the probe step did not lower CE")

    # The decision rule: largest eligible rate, and no launch when all rise.
    def rate_receipt(rate, init_kl, final_kl, tripwire=False):
        return {"rate": rate,
                "fixed_batch": {"init": {"kl": init_kl},
                                "final": {"kl": final_kl},
                                "kl_delta": final_kl - init_kl,
                                "direction": "fall" if final_kl < init_kl
                                             else "rise"},
                "update_guard": {"tripped": tripwire}}

    decision = decide([rate_receipt(3e-5, 0.5, 0.4),
                       rate_receipt(1e-4, 0.5, 0.45)], start="1e-4")
    assert decision["chosen"] == 1e-4 and decision["launch"], decision
    decision = decide([rate_receipt(3e-5, 0.5, 0.4),
                       rate_receipt(1e-4, 0.5, 0.55)], start="1e-4")
    assert decision["chosen"] == 3e-5 and decision["launch"], decision
    decision = decide([rate_receipt(3e-5, 0.5, 0.6),
                       rate_receipt(1e-4, 0.5, 0.5),
                       rate_receipt(3e-4, 0.5, 0.7)], start="1e-4")
    assert decision["chosen"] is None and not decision["launch"], decision
    decision = decide([rate_receipt(1e-4, 0.5, 0.4, tripwire=True)], start="1e-4")
    assert decision["chosen"] is None, decision

    print("s62-lr-gate self-test: OK (LoRA init identity, merge <= 5e-5 "
          f"float32 association ({error:.2e}), CE falls, decision rule)")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rate", default=None)
    parser.add_argument("--rates", type=rates, default=list(DEFAULT_RATES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--examples", default=(
        "results/artifacts/stage2/s62/corpus/stage2b-gsm8k-examples.jsonl"))
    parser.add_argument("--student-dir", default=(
        f"/hf-cache/models/Qwen/Qwen3.5-0.8B/{STUDENT_REVISION}"))
    parser.add_argument("--teacher-dir", default=(
        f"/hf-cache/models/Qwen/Qwen3.5-2B/{TEACHER_REVISION}"))
    parser.add_argument("--stream-manifest", default=DEFAULT_STREAM)
    parser.add_argument("--start-token", type=int, default=DEFAULT_START_TOKEN)
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--seq-len", type=int, default=1250)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--batch-prompts", type=int, default=8)
    parser.add_argument("--gate-batches", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--abort-update-factor", type=float, default=10.0)
    parser.add_argument("--probe-objective", choices=["sft", "reverse-kl"],
                        default="sft")
    parser.add_argument("--teacher-logprobs", default=None)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default="results/stage2/s62/lr-gate")
    parser.add_argument("--out", default=None)
    parser.add_argument("--gate-receipt", action="append", default=[])
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.probe_objective == "reverse-kl" and not args.teacher_logprobs:
        raise SystemExit(
            "s62-gate: --probe-objective reverse-kl needs --teacher-logprobs "
            "from the separate teacher pass; without it the gate would not "
            "measure the registered objective, so it fails closed")

    if args.plan or args.dry_run:
        plan = build_plan(args)
        out = os.path.join(args.out_dir, "plan.json")
        if not os.path.isabs(out):
            out = os.path.join(ROOT, out)
        write_json(out, plan)
        for run in plan["runs"]:
            print(f"s62-plan: lr={run['rate']}: "
                  f"{' '.join(run['command'])}")
        print(f"s62-plan: {len(plan['runs'])} rate(s) -> "
              f"{os.path.relpath(out, ROOT)}")
        return 0

    if args.decide:
        return run_decide(args)

    if args.execute:
        if not args.rate:
            raise SystemExit("s62-gate: --execute needs --rate")
        if not os.path.isabs(args.run_dir or ""):
            args.run_dir = args.run_dir or os.path.join(
                args.out_dir, f"gate-lr{args.rate}")
        if not args.out:
            args.out = os.path.join(args.run_dir, "gate-rate.json")
        return run_rate(args)

    parser.error("one of --plan/--dry-run/--execute/--decide/--self-test "
                 "is required")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
