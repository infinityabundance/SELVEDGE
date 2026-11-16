#!/usr/bin/env python3
"""Evaluate one T1 arm on the full registered suite and write a receipt.

  * D: unmodified dense Qwen3-0.6B (compliant baseline)
  * E: unmodified dense Qwen3-1.7B (non-compliant reference; never in compliant
    results)
  * A, A0, B, C, C2: an arm checkpoint produced by oracle/train/loop.py, built
    with the same wrappers and admitted config the run used

Every receipt carries the model/checkpoint/config hashes, the pinned data
revisions, the harness prompt definitions, per-item and per-window raw values,
the paired-bootstrap CIs, timings and the physical-compliance label.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "train"))

import harness
import tasks

HARNESS_VERSION = "selvedge/eval-harness/v1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--profile", choices=["t1", "t2", "t3"], default="t1")
    parser.add_argument("--latent-spec", default=None,
                        help="T3: the latent geometry spec JSON; builds the arm with "
                             "oracle/train/latent_geometry.py instead of build_arm")
    parser.add_argument("--model-size", default="0.6b", choices=["0.6b", "1.7b"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--admission", default=None)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--split-dir", default="results/artifacts")
    parser.add_argument("--split-receipts", default="results/splits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--humaneval-limit", type=int, default=164)
    parser.add_argument("--gsm8k-limit", type=int, default=300)
    parser.add_argument("--truthfulqa-limit", type=int, default=817)
    parser.add_argument("--skip-truthfulqa", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--mask-order", choices=["legacy", "pair"], default="legacy",
                        help="overflow validity-mask placement (P12.1); `legacy` is "
                             "what every registered arm was trained and evaluated "
                             "with, `pair` is the corrected placement")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("evaluation requires the GPU court (--gpu)")
    device = torch.device("cuda")
    started = time.time()

    tokenizer = harness.load_tokenizer(args.hf_root)
    split, split_sha = tasks.load_split(args.split_dir, args.split_receipts)
    windows = tasks.heldout_windows(split)
    humaneval = tasks.load_humaneval(args.dataset_root)
    gsm8k = tasks.gsm8k_subset(args.dataset_root, args.gsm8k_limit)
    truthfulqa = tasks.load_truthfulqa(args.dataset_root)

    teacher = harness.load_model(
        harness.model_path(args.hf_root, "1.7b"), device)
    built = None
    latent_spec_obj = None
    if args.arm == "E":
        student = teacher
    else:
        base = harness.load_model(harness.model_path(args.hf_root, args.model_size), device)
        if args.checkpoint and args.config and args.admission:
            from modeling import load_admission

            config, admission = load_admission(args.config, args.admission)
            if args.latent_spec:
                import latent_geometry

                spec = latent_geometry.LatentSpec.load(args.latent_spec)
                latent_spec_obj = spec
                arm_model = latent_geometry.build_latent_arm(base, spec, args.seed,
                                                             admission)
                args.arm = arm_model.arm
            else:
                from modeling import build_arm

                arm_model = build_arm(base, args.arm, config, admission, args.seed,
                                      profile=args.profile,
                                      mask_order=args.mask_order)
            arm_model.model.to(device)
            state = torch.load(args.checkpoint, map_location=device, weights_only=False)
            _missing, unexpected = arm_model.model.load_state_dict(state["model"],
                                                                   strict=False)
            if unexpected:
                raise SystemExit(f"checkpoint has unexpected keys: {unexpected[:5]}")
            loaded = len(state["model"])
            trainable = [name for name, parameter in arm_model.model.named_parameters()
                         if parameter.requires_grad]
            unloaded = [name for name in trainable if name not in state["model"]]
            if unloaded:
                raise SystemExit(f"checkpoint misses trainable tensors: {unloaded[:5]}")
            print(f"eval: loaded {loaded} tensors; {len(trainable)} trainable")
            arm_model.model.eval()
            student = arm_model.model
            built = arm_model
        elif args.checkpoint:
            raise SystemExit("--checkpoint requires --config and --admission")
        else:
            student = base

    compliance = {
        "D": "compliant_dense_baseline",
        "E": "research_noncompliant_teacher",
        "A": "compliant_routed_upcycle",
        "A0": "compliant_routed_untrained",
        "B": "compliant_routed_specialists",
        "C": "compliant_dense_control",
        "C2": "compliant_dense_ceiling_control",
    }.get(args.arm, "unlabelled")
    if args.profile == "t2":
        compliance = {
            "D": "compliant_dense_control",
            "M": "compliant_hashed_memory",
            "X": "compliant_routed_specialists",
            "H": "compliant_naive_hybrid",
            "H*": "compliant_error_directed_hybrid",
        }.get(args.arm, "unlabelled")
    if args.profile == "t3":
        compliance = "compliant_latent_specialists"

    results: dict = {}

    determinism = None
    if args.determinism_check and args.arm != "E":
        probe = humaneval[:5]
        first = harness.generate_texts(student, tokenizer,
                                       [tasks.humaneval_prompt(item) for item in probe],
                                       64, harness.GEN_BATCH, device)
        second = harness.generate_texts(student, tokenizer,
                                        [tasks.humaneval_prompt(item) for item in probe],
                                        64, harness.GEN_BATCH, device)
        determinism = {"problems": len(probe), "identical": first == second}

    if args.smoke:
        results["distributional"] = harness.distributional(
            student, teacher,
            {domain: entries[:2] for domain, entries in windows.items()}, device)
        results["humaneval"] = harness.run_humaneval(
            student, tokenizer, humaneval[:args.humaneval_limit], device)
        results["gsm8k"] = harness.run_gsm8k(
            student, tokenizer, gsm8k[:args.gsm8k_limit], device)
        if not args.skip_truthfulqa:
            results["truthfulqa"] = harness.run_truthfulqa(
                student, tokenizer, truthfulqa[:args.truthfulqa_limit], device)
    else:
        results["distributional"] = harness.distributional(student, teacher, windows,
                                                           device)
        results["humaneval"] = harness.run_humaneval(
            student, tokenizer, humaneval[:args.humaneval_limit], device)
        results["gsm8k"] = harness.run_gsm8k(
            student, tokenizer, gsm8k, device)
        if not args.skip_truthfulqa:
            results["truthfulqa"] = harness.run_truthfulqa(
                student, tokenizer, truthfulqa[:args.truthfulqa_limit], device)

    vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    resources = None
    if built is not None and args.profile == "t3":
        resources = {
            "arm": built.arm, "profile": "t3",
            "active_unique_params": admission["active_unique_params"],
            "total_accessible": config["total_accessible"],
            "trainable_elements": built.trainable_elements(),
            "spec": latent_spec_obj.report(),
            "basis": "admitted config (the Rust ledger is the source of truth)",
        }
    elif built is not None:
        from modeling import resource_report

        resources = resource_report(built)
    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/eval-receipt/v1",
        "harness": HARNESS_VERSION,
        "arm": args.arm,
        "compliance_label": compliance,
        "compliant_results": args.arm != "E",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "started_utc": utc_now(),
        "seconds": round(time.time() - started, 3),
        "vram_peak_gib": round(vram, 3),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "command": " ".join(sys.argv),
        "model": {
            "size": args.model_size,
            "path": harness.model_path(args.hf_root, args.model_size),
            "revision": harness.MODEL_REVISIONS[args.model_size],
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
            "config": args.config,
            "config_sha256": sha256_file(args.config) if args.config else None,
            "admission": args.admission,
        },
        "data": {
            "split_sha256": split_sha,
            "split_seed": split["seed"],
            "datasets": {
                "humaneval_items": len(humaneval),
                "gsm8k_items": len(gsm8k),
                "truthfulqa_items": len(truthfulqa),
                "distributional_windows": {domain: len(entries)
                                           for domain, entries in windows.items()},
            },
            "licences": {"humaneval": "MIT", "gsm8k": "MIT",
                         "truthfulqa": "Apache-2.0",
                         "wikitext": "CC-BY-SA-3.0 AND GFDL"},
        },
        "instrument": {
            "decoding": "greedy (do_sample=false, num_beams=1)",
            "humaneval_max_new_tokens": harness.HUMANEVAL_MAX_NEW,
            "humaneval_stop_strings": tasks.HUMANEVAL_STOP,
            "humaneval_timeout_seconds": harness.HUMANEVAL_TIMEOUT,
            "humaneval_address_space_bytes": harness.HUMANEVAL_ADDRESS_SPACE,
            "gsm8k_max_new_tokens": harness.GSM8K_MAX_NEW,
            "gsm8k_extractor": "last number after '####' else last number",
            "truthfulqa_scoring": "length-normalised log-likelihood, argmax",
            "kl": "KL(teacher || student) in nats/token, f32 logits",
            "precision": "bf16 forward, f32 logits for metrics",
            "seed": 20261115,
            "dispatch_mask_order": (
                built.mask_order() if built is not None else args.mask_order),
        },
        "determinism": determinism,
        "dispatch": built.dispatch_exposure() if built is not None else None,
        "resources": resources,
        "results": results,
    }
    write_json_atomic(args.out, receipt)
    print(json.dumps({
        "arm": args.arm,
        "pass_at_1": results["humaneval"]["pass_at_1"] if "humaneval" in results else None,
        "gsm8k": results["gsm8k"]["exact_match"] if "gsm8k" in results else None,
        "mc1": results["truthfulqa"]["mc1"] if "truthfulqa" in results else None,
        "kl_mixed": results["distributional"]["mixed"]["kl"]["mean"],
        "seconds": receipt["seconds"],
        "vram_peak_gib": receipt["vram_peak_gib"],
        "out": args.out,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
