#!/usr/bin/env python3
"""Stage-2 training loop: frozen anchor, trainable memory, online teacher KL.

The core is frozen. Only the arm module (the memory or the parameter-matched
dense adapter) receives gradients. The objective is

    L = CE(student, next token) + lambda * T^2 * KL(teacher_T || student_T)

where the teacher distribution is computed **online** on the same batch — no
logit cache is built or read. `T` and `lambda` are command-line values written
into the run manifest before the first step; they are not tuned here.

Weight-update numerics are recorded too (S42). A bf16 weight update below half
an ulp rounds to zero under round-to-nearest; `optim8bit` applies stochastic
rounding by default and can be asked for nearest or an error-compensation buffer.
On measured steps the optimiser reports the fraction of parameter elements left
bitwise unchanged, beside the round-to-nearest fraction and the sub-half-ulp
fraction, and the run receipt carries the peak-LR and minimum-LR records.

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
import glob
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, corrective, full, matched_controls, memory, optim8bit  # noqa: E402
from frontier import identity  # noqa: E402
from frontier import sft_data  # noqa: E402
from frontier import same_prefix  # noqa: E402
from frontier.stream import load_stream, sha256_file  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-2B-Base",
                           "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
GIB = float(1 << 30)
#: The unit of learning. `example`: each microbatch is averaged, then the
#: gradient is divided by grad_accum, so every example carries equal weight
#: regardless of length. `token`: the step's CE is summed over supervised
#: positions and divided by the step's total supervised positions, so every
#: supervised token carries equal weight (S60 fix for the stage-2a confound).
#: `difficulty` (p(1-p)-weighted step) belongs to the on-policy trainer and is
#: registered in `docs/ws19-post-trained-route-stage2-revised-s60-v1.json`; it
#: is not implemented by this SFT loop.
LOSS_UNITS = ("auto", "example", "token")


def resolve_loss_unit(requested: str, objective: str) -> str:
    """Resolve `auto` without changing a historical arm's recipe.

    The completion-masked SFT objectives were the ones whose token accounting
    did not match their gradient, so they default to `token`. The earlier
    `ce_kl` and `correction` arms remain `example`, exactly as they ran, so
    their config hashes and resumes do not move.
    """
    if requested != "auto":
        return requested
    return "token" if objective in ("sft", "sft-kl") else "example"


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
               vocab_chunk: int = 32,
               normaliser: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CE plus full-vocabulary online KL, chunked over positions for memory.

    `normaliser` is the supervised-position count of the whole optimiser step.
    When given, every returned term is divided by it, so summing the
    microbatches of a step yields exactly `sum CE / step supervised positions`.
    When omitted, each microbatch is normalised by its own count (the historical
    example-weighted behaviour).
    """
    flat_student = student_logits.reshape(-1, student_logits.shape[-1])
    flat_targets = targets.reshape(-1)
    ce_total = torch.zeros((), device=student_logits.device)
    positions = 0
    for start in range(0, flat_student.shape[0], vocab_chunk):
        s = flat_student[start:start + vocab_chunk].float()
        t = flat_targets[start:start + vocab_chunk]
        ce_total = ce_total + F.cross_entropy(s, t, reduction="sum")
        # Completion-only SFT: prompt and padding positions are -100 and are
        # neither a loss nor part of the normalising count.
        positions += int((t != -100).sum())
    ce = ce_total / max(1.0, float(positions if normaliser is None else normaliser))
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
    kl = total / max(1.0, float(count if normaliser is None else normaliser))
    return ce + kl_weight * (temperature ** 2) * kl, ce.detach(), kl.detach()


# ---------------------------------------------------------------------------
# arm construction


def build_arm(kind: str, model, args, device: str):
    if kind == "full":
        arm = full.FullBackboneArm(model, freeze_embedding=args.freeze_embedding)
        if args.gradient_checkpointing:
            # transformers 5.x renamed the switch; the old name is deprecated and
            # is a no-op there, which made this flag silently do nothing (S59
            # smoke: the peak was byte-identical with and without it).
            if hasattr(model, "activation_checkpointing_enable"):
                model.activation_checkpointing_enable()
            else:
                model.gradient_checkpointing_enable()
        report = arm.active_parameter_report()
        report["gradient_checkpointing"] = bool(args.gradient_checkpointing)
        report["optimizer"] = args.optimizer
        return arm, [], report
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
    if kind == "corrective":
        checkpoint = torch.load(args.memory_init, map_location="cpu",
                                weights_only=False)
        config = checkpoint["config"]
        if config["sites"] != [args.site]:
            raise SystemExit(f"memory-init site {config['sites']} != --site {args.site}")
        module = corrective.CorrectiveMemory(
            hidden=config["hidden"], value_dim=config["value_dim"],
            rows=config["rows"], sites=tuple(config["sites"]),
            conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
            seed=config["seed"])
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        module = module.to(device)
        handles = arms.install_memory(model, module)
        report = module.active_parameter_report()
        report["graft"] = checkpoint["graft"]
        report["init_checkpoint"] = args.memory_init
        report["init_checkpoint_sha256"] = sha256_file(args.memory_init)
        report["identity_at_init"] = module.identity_at_init()
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
    if kind in matched_controls.CONTROL_KINDS:
        # S47 matched controls: the same data and budget without the external
        # memory (`adapter-matched`) and the same gate over a dense residual
        # (`gated-dense`). Both are installed through the adapter hook the
        # scoring path already uses, so a trained control and a scored control
        # are the same module. Fail closed if the accounting is not the
        # registered union: the whole point of the control is the equal count.
        module, handles, report = matched_controls.build_control(
            kind, SimpleNamespace(model=model, hidden=args.hidden,
                                  site=args.site), device)
        if report["active_per_token"] != matched_controls.CORRECTIVE_ACTIVE:
            raise SystemExit(
                f"{kind}: active union {report['active_per_token']} != registered "
                f"{matched_controls.CORRECTIVE_ACTIVE}")
        return module, handles, report
    raise SystemExit(f"unknown trainable arm {kind!r}")


# ---------------------------------------------------------------------------
# batch and loop


def learning_rate(step: int, args, steps: int, warmup: int) -> float:
    """Linear warmup, then the registered schedule."""
    if step < warmup:
        return args.lr * (min(1.0, (step + 1) / warmup))
    if args.schedule == "constant":
        return args.lr
    if args.schedule == "cosine":
        span = max(1, steps - warmup)
        progress = min(1.0, (step - warmup) / span)
        floor = args.min_lr_ratio
        return args.lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise SystemExit(f"unknown schedule {args.schedule!r}")


def tripwire_decision(ce_history: list[float], factor: float,
                      baseline_steps: int, window: int) -> dict | None:
    """Evaluate the registered early-loss tripwire from the CE history.

    The baseline is the median CE over the first `baseline_steps` optimiser
    steps; the monitored statistic is the median over the trailing `window`
    steps. The tripwire fires when the trailing median exceeds `factor` times
    the baseline. Nothing is checked until both a full baseline and a full
    trailing window exist, so a short run cannot trip it by construction.
    """
    if factor <= 0.0 or baseline_steps <= 0 or window <= 0:
        return None
    if len(ce_history) < baseline_steps + window:
        return None
    baseline = statistics.median(ce_history[:baseline_steps])
    trailing = statistics.median(ce_history[-window:])
    if baseline <= 0.0:
        return None
    ratio = trailing / baseline
    return {
        "baseline_median_ce": baseline,
        "trailing_median_ce": trailing,
        "ratio": ratio,
        "fired": ratio > factor,
    }


def update_tripwire_decision(max_update_over_lr: float | None,
                             factor: float) -> dict | None:
    """Per-step tripwire on the largest Adam update direction.

    A corrected Adam step has an update of order one per element: the first
    moment is a normalised moving average of the gradient and the denominator is
    its root-mean-square, so `|update| / lr` is the step the parameter takes in
    units of the learning rate. With beta1 = 0.9 the theoretical bound is
    `1 / (1 - beta1) = 10`; in practice it stays near one. A value above the
    registered factor can only come from a collapsing denominator (a zeroed
    second moment), so it aborts the run instead of being logged and ignored.

    A disabled factor (<= 0) or a step that produced no update is silent.
    """
    if factor <= 0.0 or max_update_over_lr is None:
        return None
    return {
        "max_update_over_lr": float(max_update_over_lr),
        "factor": float(factor),
        "fired": float(max_update_over_lr) > float(factor),
    }


def prune_checkpoints(run_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    paths = sorted(glob.glob(os.path.join(run_dir, "ckpt-*.pt")))
    for old in paths[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


def next_batch(stream: torch.Tensor, offset: int, seq_len: int) -> tuple[torch.Tensor, int]:
    end = offset + seq_len + 1
    if end > stream.numel():
        raise SystemExit("stream exhausted")
    chunk = stream[offset:end].to(torch.int64)
    return chunk[None, :], offset + seq_len


def length_aware_plan(packed, rows_per_step: int, seed: int) -> dict:
    """Group complete SFT rows into length-homogeneous optimiser steps.

    Rows are sorted by their true token length and cut into contiguous groups of
    `rows_per_step`. Each group is one optimiser step, so its microbatches share
    a similar width and are trimmed to that width before the model call. The
    group order is a seeded permutation: batches stay homogeneous, while the
    step sequence still mixes short and long examples instead of teaching a
    length curriculum by accident. The plan is a pure function of the packed
    rows and the seed, so a resumed run rebuilds the same order.

    A trailing partial group is not used (the launcher requires the row count
    to be a multiple of rows/step); it is reported so it cannot disappear.
    """
    if rows_per_step <= 0:
        raise ValueError("rows_per_step must be positive")
    lengths = sft_data.row_lengths(packed)
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    usable = (len(order) // rows_per_step) * rows_per_step
    groups = [order[start:start + rows_per_step]
              for start in range(0, usable, rows_per_step)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    leftover = order[usable:]
    return {
        "groups": groups,
        "lengths": lengths,
        "rows_per_step": int(rows_per_step),
        "steps": len(groups),
        "leftover_rows": len(leftover),
        "min_group_width": min((max(lengths[index] for index in group)
                                for group in groups), default=0),
        "max_group_width": max((max(lengths[index] for index in group)
                                for group in groups), default=0),
        "note": "rows sorted by true length, grouped, then the groups are "
                "seeded-shuffled; each microbatch is trimmed to its own width",
    }


def materialise_step(packed, stream, *, step: int, row_cursor: int, offset: int,
                     plan: dict | None, args) -> tuple[list, int, int]:
    """Build every microbatch of one optimiser step up front.

    Building the step before the forward passes is what makes the global
    token normaliser exact: the supervised-position count of the whole step is
    known from the labels alone, with no model call and no extra memory beyond
    the batch tensors that were about to be built anyway.
    """
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    if packed is None:
        for micro in range(args.grad_accum):
            batch, _ = next_batch(stream, offset + micro * args.seq_len,
                                  args.seq_len)
            batches.append((batch[:, :-1], batch[:, 1:]))
        return batches, row_cursor, offset + args.micro_batch * args.seq_len * args.grad_accum
    rows = packed.input_ids.shape[0]
    if plan is None:
        for _ in range(args.grad_accum):
            indices = torch.arange(row_cursor, row_cursor + args.micro_batch) % rows
            row_cursor += args.micro_batch
            batches.append((packed.input_ids[indices, :-1],
                            packed.labels[indices, 1:]))
        return batches, row_cursor, offset
    group = plan["groups"][step % plan["steps"]]
    lengths = plan["lengths"]
    for micro in range(args.grad_accum):
        chunk = group[micro * args.micro_batch:(micro + 1) * args.micro_batch]
        width = max(lengths[index] for index in chunk)
        indices = torch.tensor(chunk, dtype=torch.long)
        batches.append((packed.input_ids[indices, :width - 1],
                        packed.labels[indices, 1:width]))
    return batches, row_cursor, offset


def step_supervised_positions(batches) -> int:
    """Supervised positions in the step, from labels only."""
    return sum(int((targets != sft_data.IGNORE_INDEX).sum())
               for _, targets in batches)


def summarise_numerics(records: list[dict]) -> dict | None:
    """Peak-LR and minimum-LR unchanged fractions from measured steps."""
    if not records:
        return None
    peak = max(records, key=lambda record: record["lr"])
    low = min(records, key=lambda record: record["lr"])

    def pick(record: dict) -> dict:
        return {
            "step": record["step"], "lr": record["lr"],
            "unchanged_fraction": record["unchanged_fraction"],
            "unchanged_fraction_round_to_nearest": record[
                "unchanged_fraction_round_to_nearest"],
            "sub_half_ulp_fraction": record["sub_half_ulp_fraction"],
        }

    return {
        "records": records,
        "peak_lr": pick(peak),
        "min_lr": pick(low),
        "reading": (
            "unchanged_fraction is measured bitwise, element by element, after "
            "the applied update; the round-to-nearest column is the outcome the "
            "same exact fp32 result would have had under nearest-even. A large "
            "measured fraction beside a large sub-half-ulp fraction means the "
            "update is below the grid, not that the weight did not move."),
    }


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

    manifest = None
    stream = None
    packed = None
    example_source = None
    data_report = None
    train_offset = 0
    if args.sft_examples:
        if args.objective != "sft":
            raise SystemExit(
                "train: --sft-examples is the completion-masked SFT path; it "
                "requires --objective sft")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.student_dir, local_files_only=True, trust_remote_code=False)
        eos_id = int(tokenizer.eos_token_id)
        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                     else eos_id)
        examples = sft_data.load_examples(args.sft_examples,
                                         vocab_size=len(tokenizer))
        packed = sft_data.pack_examples(
            examples, args.seq_len, pad_id, eos_id,
            examples_per_row=args.sft_examples_per_row,
            overflow=args.sft_overflow)
        example_source = sft_data.describe_source(args.sft_examples)
        data_report = {"kind": "sft_examples", **example_source,
                       **packed.report}
        print(f"train: packed {packed.report['packed_examples']} examples into "
              f"{packed.report['rows']} rows of {args.seq_len}; "
              f"supervised_tokens={packed.report['supervised_tokens']}",
              flush=True)
    else:
        if not args.stream_manifest:
            raise SystemExit("train: --stream-manifest is required unless "
                             "--sft-examples is given")
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
        data_report = {"kind": "token_stream",
                       "manifest": args.stream_manifest,
                       "stream_sha256": manifest["sha256"],
                       "tokens": int(stream.numel())}

    student_identity = None
    if args.student_lock:
        if not args.student_id or not args.student_revision:
            raise SystemExit(
                "train: --student-lock requires --student-id and "
                "--student-revision")
        try:
            student_identity = identity.verify_student(
                args.student_dir, args.student_lock, args.student_id,
                args.student_revision)
        except identity.IdentityMismatch as exc:
            raise SystemExit(f"train: student identity check failed: {exc}")
        print(f"train: student identity verified {args.student_id}@"
              f"{args.student_revision[:12]} weight {student_identity['weight_sha256'][:12]}",
              flush=True)
    print(f"train: loading student {args.student_dir}", flush=True)
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    if args.arm == "full":
        # The full arm decides its own frozen/trainable partition (tied
        # embedding frozen, every other text-tower identity trained).
        frozen = 0
        frozen_semantics = "set by the full arm: tied embedding only"
    else:
        frozen = arms.freeze_backbone(model)
        frozen_semantics = "whole core"
    model.eval()
    if args.arm != "full":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    arm, handles, arm_report = build_arm(args.arm, model, args, device)
    if args.arm in ("memory", "corrective"):
        for parameter in arm.parameters():
            parameter.requires_grad_(True)
    corrective_module = arm if args.arm == "corrective" else None
    objective = args.objective or (
        "correction" if args.arm == "corrective" else "ce_kl")
    if objective == "correction" and corrective_module is None:
        raise SystemExit("train: the correction objective requires --arm corrective")
    loss_unit = resolve_loss_unit(args.loss_unit, objective)
    if loss_unit == "token" and objective == "correction":
        raise SystemExit(
            "train: --loss-unit token is not defined for the correction "
            "objective; the registered corrective arms are example-averaged")
    plan = None
    if args.length_aware_batching:
        if packed is None:
            raise SystemExit(
                "train: --length-aware-batching requires --sft-examples; the "
                "token stream has fixed-length chunks by construction")
        plan = length_aware_plan(packed, args.grad_accum * args.micro_batch,
                                 args.seed)
        print(f"train: length-aware plan {plan['steps']} steps of "
              f"{plan['rows_per_step']} rows, widths "
              f"{plan['min_group_width']}..{plan['max_group_width']}, "
              f"leftover_rows={plan['leftover_rows']}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    teacher = None
    teacher_params = 0
    if args.teacher_dir and args.kl_weight > 0.0 and objective != "sft":
        print(f"train: loading teacher {args.teacher_dir}", flush=True)
        teacher = arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)
        teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    if args.arm == "full" and objective != "sft" and (
            teacher is None or args.kl_weight <= 0.0):
        raise SystemExit("train: the full-backbone control requires the online "
                         "teacher and kl_weight > 0; use --objective sft for "
                         "verified-solution cross-entropy")

    trainable = [parameter for parameter in arm.parameters() if parameter.requires_grad]
    if args.optimizer == "adamw":
        if args.arm == "full" and not args.allow_dense_optimizer:
            raise SystemExit(
                "train: the full arm does not fit with fp32 AdamW moments; use "
                "--optimizer adam8bit (or pass --allow-dense-optimizer deliberately)")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                      weight_decay=args.weight_decay)
        optimizer_report = {"kind": "stage2_adamw", "name": "AdamW",
                            "betas": [0.9, 0.95]}
    elif args.optimizer == "adam8bit":
        rounding_generator = torch.Generator(device=device)
        rounding_generator.manual_seed(args.seed + 1)
        optimizer = optim8bit.Adam8bit(
            trainable, lr=args.lr, betas=(0.9, 0.95),
            weight_decay=args.weight_decay, block=args.optim_block,
            weight_rounding=args.optim_numerics,
            generator=rounding_generator)
        optimizer_report = optimizer.receipt()
    else:
        raise SystemExit(f"train: unknown optimizer {args.optimizer!r}")
    arm_report["optimizer"] = optimizer_report
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
        "stream_sha256": (manifest["sha256"] if manifest is not None else None),
        "data": data_report,
        "student_dir": args.student_dir,
        "teacher_dir": args.teacher_dir if teacher is not None else None,
        "teacher_parameters": teacher_params,
        "deterministic_algorithms": True,
        "data_order": ("packed examples, deterministic row order, cycled; no "
                       "shuffling" if packed is not None else
                       "pinned stream order; no shuffling"),
        "completion_only_loss": bool(packed is not None),
        "memory_table_in_active_union": False,
    }
    if loss_unit != "example" or args.loss_unit != "auto":
        # Added only when the resolved unit is not the historical example
        # average (or when the caller stated it explicitly), so the earlier
        # arms' config hashes and resumes are unchanged.
        config.update({
            "loss_unit": loss_unit,
            "loss_unit_requested": args.loss_unit,
            "loss_normalisation": (
                "sum CE over the step's supervised positions / the step's "
                "supervised positions" if loss_unit == "token" else
                "microbatch mean / grad_accum: each example has equal weight"),
        })
    if student_identity is not None:
        config["student_identity"] = student_identity
    if plan is not None:
        config.update({
            "length_aware_batching": True,
            "length_aware_plan": {
                "steps": plan["steps"],
                "rows_per_step": plan["rows_per_step"],
                "leftover_rows": plan["leftover_rows"],
                "min_group_width": plan["min_group_width"],
                "max_group_width": plan["max_group_width"],
                "note": plan["note"],
            },
        })
    if args.arm == "full":
        # Added only for the full arm so existing adapter/memory config hashes
        # (and therefore their resumes) are unchanged by this arm's arrival.
        config.update({
            "optimizer": args.optimizer,
            "optim_block_elements": args.optim_block,
            "schedule": args.schedule,
            "min_lr_ratio": args.min_lr_ratio,
            "freeze_embedding": bool(args.freeze_embedding),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "sft_logits_gather": bool(args.sft_logits_gather),
            "optimizer_numerics": args.optim_numerics,
            "optimizer_measure_every": int(args.optim_measure_every),
            "max_update_over_lr_factor": float(args.abort_update_factor),
        })
    if args.arm == "corrective":
        config.update({
            "objective": objective,
            "objective_preserve_weight": args.preserve_weight,
            "objective_interference_weight": args.interference_weight,
            "objective_advantage": "KL(teacher_T || base_T) per position, detached",
            "objective_advantage_weighting": args.advantage_weighting,
            "objective_signed_advantage": (
                "log p_T(y_t) - log p_B(y_t) on the supervised target at T=1.0"),
        })
    if args.abort_ce_factor > 0.0:
        # Added only when the registered tripwire is active, so no existing
        # arm's config hash (and therefore no resume) changes.
        config.update({
            "abort_ce_factor": args.abort_ce_factor,
            "abort_ce_baseline_steps": args.abort_ce_baseline_steps,
            "abort_ce_window": args.abort_ce_window,
        })
    if args.arm in matched_controls.CONTROL_KINDS:
        # Added only for the S47 controls, so no existing arm's config hash (and
        # therefore no resume) changes. `control_role` is the attribution the
        # registration reads: the data, or the gate.
        config.update({
            "control": args.arm,
            "control_role": ("matched-data: no external memory, no gate"
                             if args.arm == "adapter-matched" else
                             "gate-only: the corrective gate over a dense residual"),
            "control_active_union": int(arm_report["active_per_token"]),
            "control_identity_at_init": bool(arm_report.get("identity_at_init")),
        })
    config["config_sha256"] = config_hash(config)
    os.makedirs(args.run_dir, exist_ok=True)
    write_json(os.path.join(args.run_dir, "manifest.json"), {
        **config, "created_utc": utcnow(), "revision": revision(),
        "frozen_backbone_elements": int(
            arm_report.get("frozen_embedding_elements", frozen)
            if args.arm == "full" else frozen),
        "frozen_backbone_elements_semantics": frozen_semantics,
        "trainable_arm_elements": int(sum(p.numel() for p in arm.parameters())),
        "arm_accounting": arm_report,
        "gpu_free_gib": free_bytes / GIB,
        "gpu_total_gib": total_bytes / GIB,
        "vram_limit_gib": args.vram_limit_gib,
        "command": sys.argv,
    })

    start_step = 0
    offset = train_offset
    row_cursor = 0
    if args.resume:
        path = args.resume
        if path == "latest":
            path = os.path.join(args.run_dir, "latest.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config_sha256"] != config["config_sha256"]:
            raise SystemExit("train: resume config hash mismatch")
        recorded_stream = payload.get("stream_sha256")
        if recorded_stream != (manifest["sha256"] if manifest is not None else None):
            raise SystemExit("train: resume stream hash mismatch")
        arm.load_state_dict(payload["arm_state"])
        arm.to(device)
        optimizer.load_state_dict(payload["optimizer"])
        if args.optimizer == "adam8bit":
            optimizer.set_rng_state(payload.get("rounding_rng"))
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        start_step = int(payload["step"])
        offset = int(payload["offset"])
        if packed is not None:
            row_cursor = (start_step * args.grad_accum * args.micro_batch
                          % packed.input_ids.shape[0])
        print(f"train: resumed step={start_step} offset={offset}", flush=True)

    log_path = os.path.join(args.run_dir, "log.jsonl")
    log = open(log_path, "a" if start_step else "w", encoding="utf-8")
    # Measure the bitwise outcome at the start, at the first full-LR step and at
    # the final (minimum-LR) step; `--optim-measure-every` adds periodic points.
    measure_at = {0, min(max(warmup - 1, 0), steps - 1), steps - 1}
    if args.optim_measure_every > 0:
        measure_at.update(range(0, steps, args.optim_measure_every))
    numerics_log: list[dict] = []
    step_time = 0.0
    run_tokens = start_step * tokens_per_step
    actual_tokens = 0
    ce_history: list[float] = []
    update_max_ever = 0.0
    if start_step and args.abort_ce_factor > 0.0:
        # A resumed run must keep the tripwire's baseline and trajectory.
        with open(log_path, "r", encoding="utf-8") as prior:
            for line in prior:
                try:
                    ce_history.append(float(json.loads(line)["ce"]))
                except (ValueError, KeyError):
                    continue
    global_started = time.time()
    for step in range(start_step, steps):
        lr = learning_rate(step, args, steps, warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        kl_sum = 0.0
        loss_sum = 0.0
        objective_sum: dict[str, float] = {}
        step_started = time.time()
        batches, row_cursor, offset = materialise_step(
            packed, stream, step=step, row_cursor=row_cursor, offset=offset,
            plan=plan, args=args)
        supervised_positions = step_supervised_positions(batches)
        if loss_unit == "token":
            if supervised_positions == 0:
                raise SystemExit(
                    "train: an optimiser step has no supervised positions; a "
                    "globally normalised loss has no denominator")
            normaliser: float | None = float(supervised_positions)
            backward_scale = 1.0
            log_scale = 1.0
        else:
            normaliser = None
            backward_scale = 1.0 / args.grad_accum
            log_scale = 1.0 / args.grad_accum
        for micro, (inputs, targets) in enumerate(batches):
            inputs = inputs.to(device)
            targets = targets.to(device)
            actual_tokens += int(inputs.numel())
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=inputs, use_cache=False).logits
            gather = objective == "sft" and args.sft_logits_gather
            if gather:
                # Completion-only CE only needs the logits at supervised
                # positions. Masked positions contribute zero loss and zero
                # gradient, so restricting the LM head to them is exact; it is
                # what keeps a 1,249-position row inside the device when the
                # desktop already holds part of the card (S59 smoke).
                if args.micro_batch != 1:
                    raise SystemExit(
                        "train: --sft-logits-gather requires --micro-batch 1")
                keep = (targets[0] != sft_data.IGNORE_INDEX).nonzero(
                    as_tuple=True)[0]
                if keep.numel() == 0:
                    student = model(input_ids=inputs, use_cache=False).logits
                    loss, ce, kl = token_loss(student, None, targets,
                                              args.temperature, 0.0,
                                              normaliser=normaliser)
                else:
                    student = model(input_ids=inputs, use_cache=False,
                                    logits_to_keep=keep).logits
                    loss, ce, kl = token_loss(student, None, targets[:, keep],
                                              args.temperature, 0.0,
                                              normaliser=normaliser)
            elif objective == "sft":
                # Verified-solution cross-entropy: no teacher, no KL.
                student = model(input_ids=inputs, use_cache=False).logits
                loss, ce, kl = token_loss(student, None, targets,
                                          args.temperature, 0.0,
                                          normaliser=normaliser)
            elif objective == "sft-kl":
                # The registered stage-2a objective: completion-only CE plus
                # teacher KL at the SAME causal prefix. `inputs` is the
                # student's own prefix, and the teacher consumed exactly those
                # ids above; the guard fails closed if that ever stops being
                # true.
                student = model(input_ids=inputs, use_cache=False).logits
                if teacher_logits is None:
                    raise SystemExit(
                        "train: sft-kl requires the online teacher")
                if tuple(teacher_logits.shape[:2]) != tuple(inputs.shape):
                    raise SystemExit(
                        "train: teacher logits do not describe the student "
                        f"prefix: {tuple(teacher_logits.shape[:2])} vs "
                        f"{tuple(inputs.shape)}")
                loss, ce, kl = same_prefix.sft_with_same_prefix_kl(
                    student, teacher_logits, targets, args.temperature,
                    args.kl_weight, normaliser=normaliser)
            elif (corrective_module is not None and objective == "correction"):
                if teacher_logits is None:
                    raise SystemExit(
                        "train: the correction objective requires the online teacher")
                student = model(input_ids=inputs, use_cache=False).logits
                # The frozen base logits are needed for the advantage and the
                # preservation term. `bypass` returns exactly zero and does not
                # advance the convolution state, so this is a clean base pass.
                with torch.no_grad():
                    corrective_module.bypass = True
                    base_logits = model(input_ids=inputs, use_cache=False).logits
                    corrective_module.bypass = False
                gate = corrective_module.last_gate
                if gate is None:
                    raise SystemExit("train: the corrective gate was not recorded "
                                     "on the student forward")
                loss, ce, kl, diagnostics = corrective.corrective_objective(
                    student, base_logits, teacher_logits, targets,
                    args.temperature, args.kl_weight, objective=objective,
                    weighting=args.advantage_weighting,
                    gate=gate,
                    preserve_weight=args.preserve_weight,
                    interference_weight=args.interference_weight)
                for key, value in diagnostics.items():
                    if isinstance(value, (int, float)):
                        objective_sum[key] = objective_sum.get(key, 0.0) + float(value)
            else:
                student = model(input_ids=inputs, use_cache=False).logits
                loss, ce, kl = token_loss(student, teacher_logits, targets,
                                          args.temperature, args.kl_weight,
                                          normaliser=normaliser)
            (loss * backward_scale).backward()
            loss_sum += float(loss.detach())
            ce_sum += float(ce)
            kl_sum += float(kl)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        measure = args.optimizer == "adam8bit" and step in measure_at
        if measure:
            optimizer.step(measure=True)
            numerics_log.append({"step": step + 1, **optimizer.last_numerics})
        else:
            optimizer.step()
        step_max_update = None
        if args.optimizer == "adam8bit":
            step_max_update = optimizer.last_max_update_over_lr
            if step_max_update is not None:
                update_max_ever = max(update_max_ever, step_max_update)
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
            "loss": loss_sum * log_scale,
            "ce": ce_sum * log_scale,
            "kl": kl_sum * log_scale,
            "loss_unit": loss_unit,
            "supervised_positions_in_step": supervised_positions,
            "actual_input_tokens": actual_tokens,
            "lr": lr,
            "step_seconds": round(step_time, 4),
            "tokens_per_second": round(tokens_per_step / step_time, 1),
            "peak_allocated_gib": round(peak_alloc, 3),
            "peak_reserved_gib": round(peak_reserved, 3),
            "active_per_token": arm_report.get("active_per_token"),
        }
        if measure:
            entry["numerics"] = optimizer.last_numerics
        if step_max_update is not None:
            entry["max_update_over_lr"] = step_max_update
        if objective_sum and corrective_module is not None:
            entry["objective"] = {
                key: value / args.grad_accum
                for key, value in objective_sum.items()
                if key != "objective"}
            entry["gate"] = corrective_module.gate_summary()
        if "gate" not in entry and getattr(arm, "last_gate", None) is not None:
            # The matched gated-dense control records the same gate tensor; a
            # module with no gate (the dense adapter) records nothing.
            gate_tensor = arm.last_gate.detach().float()
            entry["gate"] = {
                "tokens": int(gate_tensor.numel()),
                "mean": float(gate_tensor.mean()),
                "mean_abs": float(gate_tensor.abs().mean()),
                "min": float(gate_tensor.min()),
                "max": float(gate_tensor.max()),
            }
        if packed is None and step % args.collision_every == 0 and args.arm == "memory":
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
        if args.optimizer == "adam8bit":
            decision = update_tripwire_decision(step_max_update,
                                                args.abort_update_factor)
            if decision is not None and decision["fired"]:
                # The update tripwire is a registered stop: a collapsing Adam
                # denominator is a storage defect, not a training regime.
                abort = {
                    "schema": "selvedge/stage2-train-abort/v1",
                    "kind": "train-abort",
                    "created_utc": utcnow(),
                    "stamp": stamp(),
                    "revision": revision(),
                    "arm": args.arm,
                    "reason": ("update tripwire: max|update|/lr exceeded the "
                               "registered factor; the Adam denominator is "
                               "collapsing (zeroed second moment)"),
                    "step": step + 1,
                    "tokens_seen": run_tokens,
                    **decision,
                    "optimizer": optimizer.receipt(),
                    "log": log_path,
                    "config": config,
                    "command": sys.argv,
                }
                write_json(os.path.join(args.run_dir, "abort.json"), abort)
                log.close()
                print(f"train: UPDATE TRIPWIRE step {step + 1} "
                      f"max|update|/lr={decision['max_update_over_lr']:.3f} > "
                      f"{args.abort_update_factor:.3f}; aborted", flush=True)
                return 86
        if args.abort_ce_factor > 0.0:
            ce_history.append(entry["ce"])
            decision = tripwire_decision(
                ce_history, args.abort_ce_factor,
                args.abort_ce_baseline_steps, args.abort_ce_window)
            if decision is not None and decision["fired"]:
                # The tripwire is a registered stop, not an exception: the
                # abort record is the evidence, the log is left closed and no
                # checkpoint is written from this path.
                abort = {
                    "schema": "selvedge/stage2-train-abort/v1",
                    "kind": "train-abort",
                    "created_utc": utcnow(),
                    "stamp": stamp(),
                    "revision": revision(),
                    "arm": args.arm,
                    "reason": ("early-loss tripwire: the trailing-window "
                               "median CE exceeded the registered factor of "
                               "the baseline median CE"),
                    "step": step + 1,
                    "tokens_seen": run_tokens,
                    "baseline_steps": args.abort_ce_baseline_steps,
                    "window_steps": args.abort_ce_window,
                    "factor": args.abort_ce_factor,
                    **decision,
                    "log": log_path,
                    "config": config,
                    "command": sys.argv,
                }
                write_json(os.path.join(args.run_dir, "abort.json"), abort)
                log.close()
                print(f"train: TRIPWIRE step {step + 1} ratio="
                      f"{decision['ratio']:.3f} > {args.abort_ce_factor:.3f}; "
                      f"aborted", flush=True)
                return 86
        if (step + 1) % args.ckpt_every == 0 or step + 1 == steps:
            payload = {
                "schema": "selvedge/stage2-ckpt/v1",
                "step": step + 1,
                "offset": offset,
                "tokens_seen": run_tokens,
                "config_sha256": config["config_sha256"],
                "stream_sha256": (manifest["sha256"]
                                   if manifest is not None else None),
                "arm": args.arm,
                "arm_state": arm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rounding_rng": (optimizer.rng_state()
                                 if args.optimizer == "adam8bit" else None),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }
            ckpt = os.path.join(args.run_dir, f"ckpt-{step + 1:06d}.pt")
            torch.save(payload, ckpt)
            torch.save(payload, os.path.join(args.run_dir, "latest.pt"))
            prune_checkpoints(args.run_dir, args.keep_checkpoints)
            print(f"train: checkpoint {ckpt}", flush=True)

    log.close()
    if args.optimizer == "adam8bit":
        # The construction receipt is stale by now; report allocated state and
        # the final measured numerics as saved.
        optimizer_report = optimizer.receipt()
    numerics_summary = summarise_numerics(numerics_log)
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
        "frozen_backbone_elements": int(
            arm_report.get("frozen_embedding_elements", frozen)
            if args.arm == "full" else frozen),
        "frozen_backbone_elements_semantics": frozen_semantics,
        "trainable_arm_elements": int(sum(p.numel() for p in arm.parameters())),
        "arm_accounting": arm_report,
        "optimizer_state": optimizer_report,
        "optimizer_numerics": numerics_summary,
        "tripwire": ({
            "factor": args.abort_ce_factor,
            "baseline_steps": args.abort_ce_baseline_steps,
            "window_steps": args.abort_ce_window,
            "tripped": False,
        } if args.abort_ce_factor > 0.0 else None),
        "update_tripwire": ({
            "factor": args.abort_update_factor,
            "max_update_over_lr_ever": update_max_ever,
            "tripped": False,
        } if args.optimizer == "adam8bit" else None),
        "objective": objective,
        "loss_unit": loss_unit,
        "actual_input_tokens": actual_tokens,
        "length_aware_batching": bool(plan is not None),
        "student_identity": student_identity,
        "correction_gate_final": (
            corrective_module.gate_summary() if corrective_module is not None
            else None),
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


def validate_objective_args(args) -> None:
    """Fail-closed wiring shared by main and the CPU tests."""
    if args.objective == "correction" and args.arm != "corrective":
        raise SystemExit("--objective correction requires --arm corrective")
    if args.objective == "sft" and args.kl_weight != 0.0:
        raise SystemExit("--objective sft requires --kl-weight 0")
    if args.objective == "sft-kl" and args.kl_weight <= 0.0:
        raise SystemExit("--objective sft-kl requires --kl-weight > 0")
    if args.objective == "sft-kl" and not args.teacher_dir:
        raise SystemExit("--objective sft-kl requires --teacher-dir")
    if args.arm == "full" and args.objective != "sft" and args.kl_weight <= 0.0:
        raise SystemExit("--arm full needs kl_weight > 0 unless --objective sft")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",
                        choices=["memory", "corrective", "adapter", "full",
                                 *matched_controls.CONTROL_KINDS],
                        required=True)
    parser.add_argument("--stream-manifest", default=None)
    parser.add_argument("--sft-examples", default=None,
                        help="JSONL of prompt_ids/solution_ids; enables the "
                             "completion-masked SFT path (no stream chunking)")
    parser.add_argument("--sft-examples-per-row", type=int, default=1,
                        help="complete examples packed per row; >1 is not "
                             "boundary-safe without a per-example attention reset")
    parser.add_argument("--sft-overflow", choices=["error", "drop"],
                        default="error",
                        help="what to do when an example exceeds seq_len; "
                             "error never splits a solution")
    parser.add_argument("--loss-unit", choices=list(LOSS_UNITS), default="auto",
                        help="unit of learning: 'token' sums CE over the step's "
                             "supervised positions and divides by that step total; "
                             "'example' averages each microbatch and divides by "
                             "grad_accum; 'auto' selects token for sft/sft-kl and "
                             "example for the earlier ce_kl/correction arms")
    parser.add_argument("--length-aware-batching", action="store_true",
                        help="group complete SFT rows into length-homogeneous "
                             "optimiser steps and trim each microbatch to its own "
                             "width, so long examples are preserved without "
                             "paying full-length activations on short batches")
    parser.add_argument("--student-lock", default=None,
                        help="identity lock to verify the student against before "
                             "loading it (requires --student-id and "
                             "--student-revision)")
    parser.add_argument("--student-id", default=None)
    parser.add_argument("--student-revision", default=None)
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
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--optim-block", type=int, default=256)
    parser.add_argument("--optim-numerics",
                        choices=list(optim8bit.NUMERICS),
                        default=optim8bit.DEFAULT_NUMERICS,
                        help="weight-update rounding for the 8-bit optimizer")
    parser.add_argument("--optim-measure-every", type=int, default=0,
                        help="also measure unchanged-element fractions every N steps")
    parser.add_argument("--objective", choices=list(corrective.OBJECTIVES) + ["sft", "sft-kl"],
                        default=None,
                        help="correction (default for --arm corrective), ce_kl, or "
                             "sft (verified-solution cross-entropy, kl_weight 0)")
    parser.add_argument("--advantage-weighting",
                        choices=list(corrective.WEIGHTINGS), default="kl",
                        help="the correction objective's position weight: the "
                             "registered KL-disagreement weight, the signed "
                             "teacher advantage on the target, or both")
    parser.add_argument("--preserve-weight", type=float, default=0.5)
    parser.add_argument("--interference-weight", type=float, default=0.05)
    parser.add_argument("--allow-dense-optimizer", action="store_true")
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--abort-ce-factor", type=float, default=0.0,
                        help="early-loss tripwire: abort when the trailing-window "
                             "median CE exceeds this factor of the baseline median")
    parser.add_argument("--abort-ce-baseline-steps", type=int, default=100,
                        help="optimiser steps whose median CE is the tripwire baseline")
    parser.add_argument("--abort-ce-window", type=int, default=50,
                        help="trailing optimiser steps whose median CE is monitored")
    parser.add_argument("--abort-update-factor", type=float, default=10.0,
                        help="update tripwire: abort when max|update|/lr over a "
                             "step exceeds this factor (0 disables); the Adam "
                             "bound with beta1=0.9 is 1/(1-beta1)=10")
    parser.add_argument("--freeze-embedding", dest="freeze_embedding",
                        action="store_true", default=True)
    parser.add_argument("--no-freeze-embedding", dest="freeze_embedding",
                        action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--sft-logits-gather", action="store_true",
                        help="completion-only SFT: compute the LM head only at "
                             "supervised positions (exact; masked positions "
                             "contribute no loss or gradient)")
    parser.add_argument("--keep-checkpoints", type=int, default=0,
                        help="keep only the newest N ckpt-*.pt files (0 keeps all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--collision-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vram-limit-gib", type=float, default=12.0)
    args = parser.parse_args(argv)
    if args.arm in ("memory", "corrective") and not args.memory_init:
        raise SystemExit(f"--memory-init is required for --arm {args.arm}")
    validate_objective_args(args)
    if args.arm != "full" and (args.optimizer != "adamw" or args.schedule != "constant"):
        raise SystemExit("the registered adapter/memory/corrective arms are AdamW + "
                         "constant LR; do not change them retroactively")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
