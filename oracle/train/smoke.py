#!/usr/bin/env python3
"""T1 smoke tests: every architecture trains, checkpoints, resumes and evaluates.

Per arm (A, B, C):

  1. verify the Rust admission record for the arm config;
  2. verify step-0 identity against the frozen base (K12);
  3. train 20 registered-shape steps (65,536 tokens per step);
  4. train 10 steps, resume from the step-10 checkpoint, train to 20, and
     compare the continuation loss with the uninterrupted run (K11 resume
     evidence, reported as a numerical delta — GPU atomics in grouped routing
     are not bitwise reproducible);
  5. evaluate on 5 HumanEval problems, 2 held-out windows per domain and a
     fixed determinism probe (the same 5 prompts generated twice);
  6. record throughput, VRAM peak and steps/s for the budget table.

Writes results/train/smoke-<stamp>.json. No full run is started here.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

import harness  # noqa: E402
import loop  # noqa: E402
import tasks  # noqa: E402
from modeling import build_arm, load_admission, resource_report  # noqa: E402


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_losses(path: str) -> dict[int, float]:
    losses = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            losses[int(record["step"])] = float(record["loss"])
    return losses


def run_loop(arm: str, config: str, admission: str, out: str, steps: int,
             resume: str | None, zero_point: bool, micro_batch: int,
             accum: int, profile: str = "t1", warmup_steps: int = 0,
             calibration_windows: int = 32, router_steps: int = 200,
             labels_out: str | None = None, load_balance: bool = False,
             grad_checkpoint: bool = False, deterministic: bool = False,
             micro_positions: int = 1024) -> None:
    argv = ["--arm", arm, "--config", config, "--admission", admission,
            "--out", out, "--steps", str(steps), "--checkpoint-every", "10",
            "--micro-batch", str(micro_batch), "--accum", str(accum),
            "--micro-positions", str(micro_positions),
            "--profile", profile]
    if resume:
        argv += ["--resume", resume]
    if zero_point:
        argv += ["--zero-point"]
    if profile == "t2":
        argv += ["--warmup-steps", str(warmup_steps),
                 "--calibration-windows", str(calibration_windows),
                 "--router-steps", str(router_steps)]
        if labels_out:
            argv += ["--labels-out", labels_out]
    if load_balance:
        argv += ["--load-balance"]
    if grad_checkpoint:
        argv += ["--grad-checkpoint"]
    if deterministic:
        argv += ["--deterministic"]
    loop.main(argv)


def checkpoint_digest(path: str) -> str:
    return loop.sha256_file(path)


def evaluate_checkpoint(arm: str, config: str, admission: str, checkpoint: str,
                        hf_root: str, dataset_root: str, split_dir: str,
                        split_receipts: str, device, profile: str = "t1") -> dict:
    from transformers import AutoModelForCausalLM

    config_payload, admission_payload = load_admission(config, admission)
    size = "1.7b" if arm in ("A", "A0") else "0.6b"
    base = AutoModelForCausalLM.from_pretrained(
        harness.model_path(hf_root, size), dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, attn_implementation="sdpa")
    base.to(device)
    base.eval()
    arm_model = build_arm(base, arm, config_payload, admission_payload, 1001,
                          profile=profile)
    arm_model.model.to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    arm_model.model.load_state_dict(state["model"], strict=False)
    arm_model.model.eval()
    tokenizer = harness.load_tokenizer(hf_root)
    split, split_sha = tasks.load_split(split_dir, split_receipts)
    windows = tasks.heldout_windows(split)
    humaneval = tasks.load_humaneval(dataset_root)
    gsm8k = tasks.gsm8k_subset(dataset_root, 5)
    truthfulqa = tasks.load_truthfulqa(dataset_root)[:5]
    teacher = harness.load_model(harness.model_path(hf_root, "1.7b"), device)
    result = harness.smoke(arm_model.model, teacher, tokenizer,
                           {"humaneval": humaneval, "gsm8k": gsm8k,
                            "truthfulqa": truthfulqa}, windows, device, problems=5)
    prompts = [tasks.humaneval_prompt(item) for item in humaneval[:5]]
    first = harness.generate_texts(arm_model.model, tokenizer, prompts, 32,
                                   harness.GEN_BATCH, device)
    second = harness.generate_texts(arm_model.model, tokenizer, prompts, 32,
                                    harness.GEN_BATCH, device)
    result["determinism"] = {"problems": len(prompts), "identical": first == second}
    result["resources"] = resource_report(arm_model)
    result["checkpoint_step"] = int(state["step"])
    result["split_sha256"] = split_sha
    result["vram_peak_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    del teacher, base, arm_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["t1", "t2"], default="t1")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--determinism-steps", type=int, default=10)
    parser.add_argument("--resume-split", type=int, default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--admission-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--receipt", default="results/train")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--calibration-windows", type=int, default=32)
    parser.add_argument("--router-steps", type=int, default=200)
    parser.add_argument("--load-balance", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-checkpoint-arms", default="H,H*",
                        help="T2 arms that need activation checkpointing to fit 16 GiB")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--micro-positions", type=int, default=None)
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=64)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("smoke requires the GPU court (--gpu)")
    device = torch.device("cuda")
    t2 = args.profile == "t2"
    defaults = {
        "arms": "D,M,X,H,H*" if t2 else "A,B,C",
        "steps": 50 if t2 else 20,
        "resume-split": 25 if t2 else 10,
        "config-dir": "results/train/admission-configs/t2" if t2
        else "results/train/admission-configs",
        "admission-dir": "results/train/admission/t2" if t2
        else "results/train/admission",
        "out-dir": "results/artifacts/train/smoke-t2" if t2
        else "results/artifacts/train/smoke",
    }
    for name, value in defaults.items():
        if getattr(args, name.replace("-", "_")) is None:
            setattr(args, name.replace("-", "_"), value)
    arms = [item for item in args.arms.split(",") if item]
    reports = {}
    for arm in arms:
        config = os.path.join(args.config_dir, f"{arm}.json")
        admission = os.path.join(args.admission_dir, f"{arm}.json")
        if not os.path.exists(admission):
            raise SystemExit(f"missing admission record {admission}; "
                             "run scripts/train-admission.sh")
        out = os.path.join(args.out_dir, arm)
        resume_out = os.path.join(args.out_dir, f"{arm}-resume")
        determinism_out = os.path.join(args.out_dir, f"{arm}-determinism")
        import shutil

        for name in (out, resume_out, determinism_out):
            shutil.rmtree(name, ignore_errors=True)
        common = dict(profile=args.profile, warmup_steps=args.warmup_steps,
                      calibration_windows=args.calibration_windows,
                      router_steps=args.router_steps,
                      load_balance=args.load_balance and arm != "H*",
                      grad_checkpoint=(args.grad_checkpoint or
                                       (t2 and arm in args.grad_checkpoint_arms.split(","))),
                      deterministic=args.deterministic,
                      micro_positions=args.micro_positions or (512 if t2 else 1024))
        print(f"smoke: arm {arm} training {args.steps} steps "
              f"(resident {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GiB)",
              flush=True)
        torch.cuda.reset_peak_memory_stats()
        run_loop(arm, config, admission, out, args.steps, None, True,
                 args.micro_batch, args.accum, **common)
        full_losses = read_losses(os.path.join(out, "train.jsonl"))

        # Determinism: the same seed twice, compared on the loss trajectory and
        # the step-N checkpoint hash (T2 execution is deterministic, T2.4).
        torch.cuda.reset_peak_memory_stats()
        run_loop(arm, config, admission, determinism_out, args.determinism_steps, None,
                 False, args.micro_batch, args.accum, **common)
        first_losses = read_losses(os.path.join(determinism_out, "train.jsonl"))
        first_hash = checkpoint_digest(
            os.path.join(determinism_out, f"step-{args.determinism_steps:06d}.pt"))
        shutil.rmtree(determinism_out)
        run_loop(arm, config, admission, determinism_out, args.determinism_steps, None,
                 False, args.micro_batch, args.accum, **common)
        second_losses = read_losses(os.path.join(determinism_out, "train.jsonl"))
        second_hash = checkpoint_digest(
            os.path.join(determinism_out, f"step-{args.determinism_steps:06d}.pt"))
        determinism = {
            "steps": args.determinism_steps,
            "loss_trajectories_identical": first_losses == second_losses,
            "checkpoint_sha256_first": first_hash,
            "checkpoint_sha256_second": second_hash,
            "checkpoint_hash_identical": first_hash == second_hash,
            "max_abs_loss_delta": max(
                (abs(first_losses[step] - second_losses[step]) for step in first_losses),
                default=None),
            "note": "T2 uses a deterministic slot-wise combine; a hash mismatch is a "
                    "kernel defect, not hardware noise (T2.4)",
        }

        run_loop(arm, config, admission, resume_out, args.resume_split, None, False,
                 args.micro_batch, args.accum, **common)
        checkpoint = os.path.join(resume_out, f"step-{args.resume_split:06d}.pt")
        run_loop(arm, config, admission, resume_out, args.steps, checkpoint, False,
                 args.micro_batch, args.accum, **common)
        resumed_losses = read_losses(os.path.join(resume_out, "train.jsonl"))
        compared = sorted(set(full_losses) & set(resumed_losses))
        compared = [step for step in compared if step > args.resume_split]
        deltas = [abs(full_losses[step] - resumed_losses[step]) for step in compared]
        resume = {
            "steps_compared": len(compared),
            "max_abs_loss_delta": float(max(deltas)) if deltas else None,
            "mean_abs_loss_delta": float(np.mean(deltas)) if deltas else None,
            "full_losses": {str(step): full_losses[step] for step in compared},
            "resumed_losses": {str(step): resumed_losses[step] for step in compared},
            "note": "T2 continuation must be bitwise identical after a resume; any "
                    "delta is reported verbatim",
        }
        summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
        zero_point = json.load(open(os.path.join(out, "zero-point.json"),
                                    encoding="utf-8"))
        eval_result = None
        if not args.skip_eval:
            final = os.path.join(out, f"step-{args.steps:06d}.pt")
            eval_result = evaluate_checkpoint(arm, config, admission, final,
                                              args.hf_root, args.dataset_root,
                                              args.split_dir, args.split_receipts,
                                              device, profile=args.profile)
            torch.cuda.empty_cache()
        reports[arm] = {
            "train": summary,
            "resume": resume,
            "determinism": determinism,
            "zero_point": zero_point,
            "eval": eval_result,
        }
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        deltas_msg = (f"{resume['max_abs_loss_delta']:.3e}"
                      if resume["max_abs_loss_delta"] is not None else "n/a")
        print(f"smoke: arm {arm} tok/s={summary['tokens_per_second']:.0f} "
              f"vram={summary['vram_peak_gib']:.2f}GiB "
              f"zero_point={zero_point['ok']} resume_delta={deltas_msg} "
              f"deterministic={determinism['checkpoint_hash_identical']}", flush=True)

    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/train-smoke/v1",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "finished_utc": utc_now(),
        "profile": args.profile,
        "steps": args.steps,
        "determinism_steps": args.determinism_steps,
        "tokens_per_step": 64 * 1024,
        "warmup_steps": args.warmup_steps,
        "calibration_windows": args.calibration_windows,
        "grad_checkpoint": args.grad_checkpoint,
        "arms": reports,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "command": " ".join(sys.argv),
    }
    path = os.path.join(args.receipt, f"smoke-{stamp}.json")
    write_json_atomic(path, receipt)
    print(f"smoke: receipt={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
