#!/usr/bin/env python3
"""Exact bytes for the full-backbone distillation control (S40).

No number here is typed. The parameter counts come from the pinned checkpoint's
safetensors header, the teacher count comes from the registered run manifest,
the measured envelope comes from the running reference arm's step log, and the
optimizer bytes are summed per tensor from its own block arithmetic.

Projection, stated as a projection. The reference arm (`control-20m`) measures a
*fully frozen* core. Its peak above weights and teacher is the forward/logit/KL/
workspace envelope. The full arm adds three things and keeps everything else:

* bf16 gradients for the 498,113,344 trained elements;
* block-wise 8-bit Adam moments and scales;
* retained activations, because the autograd graph now runs through all 24
  layers instead of stopping at the injected site.

The retained-activation term is modelled, not measured: the saved input of every
2-D linear weight is `seq_len * in_features` elements, summed over the text
tower, with a lower bound of exactly those saved inputs and an upper bound of
twice that for elementwise intermediates. The first launch of this arm must
therefore be a short smoke whose measured peak replaces this projection before
the 20M-token run. The registered `--vram-limit-gib` is the upper projection plus
1 GiB of headroom, and the run fails closed if it is exceeded.

Usage:
    SELVEDGE_MEM=2g scripts/court python -- python3 \
        scripts/s40_full_arm_arithmetic.py \
        --out results/stage2/s40-full-arithmetic-<stamp>.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GIB = float(1 << 30)
CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
REFERENCE_RUN = "results/stage2/runs/control-20m"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def numel(entry: dict) -> int:
    total = 1
    for size in entry["shape"]:
        total *= int(size)
    return total


def padded(value: int, block: int) -> int:
    return ((value + block - 1) // block) * block


def optimizer_bytes(shapes: list[tuple[int, ...]], block: int) -> dict:
    moment_bytes = 0
    scale_bytes = 0
    padded_elements = 0
    parameters = 0
    for shape in shapes:
        count = 1
        for size in shape:
            count *= int(size)
        parameters += count
        block_count = padded(count, block) // block
        padded_elements += block_count * block
        moment_bytes += 2 * block_count * block          # two int8 moments
        scale_bytes += 2 * block_count * 4               # two float32 absmax
    return {
        "block_elements": block,
        "parameters": parameters,
        "padded_elements": padded_elements,
        "moment_bytes": moment_bytes,
        "scale_bytes": scale_bytes,
        "bytes": moment_bytes + scale_bytes,
        "bytes_per_parameter": (moment_bytes + scale_bytes) / max(1, parameters),
        "basis": "two int8 moments plus two float32 scales per 256-element block",
    }


def last_log_entry(path: str) -> dict:
    entry = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
    if entry is None:
        raise SystemExit(f"no log entries in {path}")
    return entry


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--reference-run", default=REFERENCE_RUN)
    parser.add_argument("--teacher-parameters", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("results", "stage2",
                                f"s40-full-arithmetic-{stamp()}.json")

    files = sorted(glob.glob(os.path.join(args.student_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors in {args.student_dir}")
    if len(files) != 1:
        raise SystemExit("this script expects a single-file text checkpoint")
    header = safetensors_header(files[0])

    text = {name: entry for name, entry in header.items()
            if name.startswith("model.language_model.")}
    if not text:
        raise SystemExit("no model.language_model tensors in the header")
    text_elements = sum(numel(entry) for entry in text.values())
    embedding_name = "model.language_model.embed_tokens.weight"
    if embedding_name not in text:
        raise SystemExit("the tied embedding tensor is absent")
    embedding_elements = numel(text[embedding_name])
    trainable = {name: entry for name, entry in text.items()
                 if name != embedding_name}
    trainable_elements = sum(numel(entry) for entry in trainable.values())

    shapes = [tuple(entry["shape"]) for entry in trainable.values()]
    optimizer = optimizer_bytes(shapes, args.block)
    in_features = sum(int(entry["shape"][1]) for entry in trainable.values()
                      if len(entry["shape"]) == 2)

    reference_manifest = json.load(
        open(os.path.join(REPO, args.reference_run, "manifest.json")))
    reference_log = os.path.join(REPO, args.reference_run, "log.jsonl")
    reference = last_log_entry(reference_log)
    teacher_parameters = args.teacher_parameters or int(
        reference_manifest["teacher_parameters"])
    teacher_bytes = teacher_parameters * 2

    student_bytes = text_elements * 2
    measured_peak_gib = float(reference["peak_allocated_gib"])
    measured_reserved_gib = float(reference.get("peak_reserved_gib", measured_peak_gib))
    envelope_bytes = int(round(measured_peak_gib * GIB)) - student_bytes - teacher_bytes
    if envelope_bytes < 0:
        raise SystemExit("the reference peak is below its resident weights")

    saved_inputs = 2 * args.seq_len * in_features
    retention_lower = saved_inputs
    retention_upper = 2 * saved_inputs

    gradients = trainable_elements * 2
    projected_low = (reference["peak_allocated_gib"] * GIB + gradients
                     + optimizer["bytes"] + retention_lower) / GIB
    projected_high = (reference["peak_allocated_gib"] * GIB + gradients
                      + optimizer["bytes"] + retention_upper) / GIB
    limit = math.ceil((projected_high + 1.0) * 2.0) / 2.0

    payload = {
        "schema": "selvedge/stage2-full-arm-arithmetic/v1",
        "kind": "full-backbone-distillation-arithmetic",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "student": {
            "dir": args.student_dir,
            "checkpoint": files[0],
            "checkpoint_sha256": sha256_file(files[0]),
            "text_tower_elements": text_elements,
            "frozen_embedding_elements": embedding_elements,
            "trainable_elements": trainable_elements,
            "deployed_active_union_elements": text_elements,
            "deployed_active_union_basis": "finetuning changes values, not identities",
            "frozen_identities": [embedding_name,
                                  "the tied output projection (no lm_head.weight exists)"],
            "text_tensors": len(text),
            "trainable_tensors": len(trainable),
        },
        "teacher": {
            "parameters": teacher_parameters,
            "source": f"{args.reference_run}/manifest.json:teacher_parameters",
            "bf16_bytes": teacher_bytes,
        },
        "byte_arithmetic": {
            "dtype_bytes": 2,
            "student_text_bytes": student_bytes,
            "frozen_embedding_bytes": embedding_elements * 2,
            "trainable_weights_bytes": trainable_elements * 2,
            "gradients_bytes": gradients,
            "optimizer": optimizer,
            "total_resident_and_training_bytes": (
                student_bytes + teacher_bytes + gradients + optimizer["bytes"]),
        },
        "reference_measurement": {
            "run": args.reference_run,
            "arm": reference_manifest.get("arm"),
            "log": os.path.relpath(reference_log, REPO),
            "step": reference.get("step"),
            "tokens_seen": reference.get("tokens_seen"),
            "measured_peak_allocated_gib": measured_peak_gib,
            "measured_peak_reserved_gib": measured_reserved_gib,
            "reference_student_bytes": student_bytes,
            "reference_teacher_bytes": teacher_bytes,
            "derived_non_weight_envelope_bytes": envelope_bytes,
            "derived_non_weight_envelope_gib": envelope_bytes / GIB,
            "reading": (
                "the reference arm freezes the core, so this envelope is forward "
                "logits, the teacher forward, the KL/CE buffers and workspace; it "
                "does not include gradient or retained-activation memory"
            ),
        },
        "activation_retention_model": {
            "seq_len": args.seq_len,
            "linear_in_features_per_token": in_features,
            "saved_linear_inputs_bytes_lower": retention_lower,
            "saved_plus_intermediates_bytes_upper": retention_upper,
            "basis": (
                "every 2-D linear weight saves its input [T, in_features] at bf16; "
                "the upper bound doubles that for elementwise intermediates"
            ),
            "status": "MODELLED — replace with the smoke measurement before the run",
        },
        "projection": {
            "projected_peak_gib_low": projected_low,
            "projected_peak_gib_high": projected_high,
            "vram_limit_gib": limit,
            "headroom_gib": limit - projected_high,
            "device_total_gib": reference_manifest.get("gpu_total_gib"),
            "free_at_reference_start_gib": reference_manifest.get("gpu_free_gib"),
            "status": "PROJECTED from the measured reference envelope + exact weights",
            "smoke_requirement": (
                "launch 20 optimizer steps under the limit, record the measured "
                "peak, and re-derive the limit from it before the 20M-token run"
            ),
        },
        "command": sys.argv,
    }
    write_json(args.out, payload)
    print(f"full-arm: text={text_elements} frozen_embedding={embedding_elements} "
          f"trainable={trainable_elements}")
    print(f"full-arm: optimizer bytes={optimizer['bytes']} "
          f"({optimizer['bytes_per_parameter']:.4f} B/param)")
    print(f"full-arm: reference peak={measured_peak_gib:.3f} GiB "
          f"envelope={envelope_bytes / GIB:.3f} GiB")
    print(f"full-arm: projected peak={projected_low:.2f}..{projected_high:.2f} GiB "
          f"limit={limit:.1f} GiB")
    print(f"full-arm: receipt {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
