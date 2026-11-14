#!/usr/bin/env python3
"""Capture real FFN input/output activations from a pinned teacher checkpoint.

This is instrumentation, not a model implementation. It runs the Hugging Face
teacher on a pinned, original prompt set and records, for every transformer
block, the exact input to the MLP module and the exact module output, in two
dtype paths:

* `bf16` — the checkpoint's native dtype, which is what the model actually
  executes in normal inference;
* `f32` — the same weights upcast exactly to float32, which is the reference the
  Rust group arithmetic is compared against.

Only the MLP boundary is instrumented. No gradient, no cache, no generation and
no `trust_remote_code` are used. The activation file is a plain safetensors
file of f32 tensors so the Rust verifier can read it with the S2 reader.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SCHEMA = "selvedge/activation-capture/v1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture_pass(model, tokenizer, prompts, device, max_tokens, label: str) -> dict:
    layers = len(model.model.layers)
    buffers: dict[int, dict[str, list[torch.Tensor]]] = {
        index: {"input": [], "output": []} for index in range(layers)
    }
    handles = []

    def make_hook(index: int):
        def hook(_module, inputs, output):
            if not inputs:
                raise RuntimeError(f"layer {index}: mlp hook received no input")
            buffers[index]["input"].append(
                inputs[0].detach().to(torch.float32).cpu().reshape(-1, inputs[0].shape[-1]).contiguous()
            )
            if isinstance(output, tuple):
                output = output[0]
            buffers[index]["output"].append(
                output.detach().to(torch.float32).cpu().reshape(-1, output.shape[-1]).contiguous()
            )

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(make_hook(index)))
    per_prompt = []
    try:
        with torch.no_grad():
            for prompt in prompts["prompts"]:
                encoded = tokenizer(
                    prompt["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"].to(device)
                model(input_ids=input_ids, use_cache=False)
                per_prompt.append(
                    {
                        "id": prompt["id"],
                        "prompt_sha256": sha256_text(prompt["text"]),
                        "tokens": int(input_ids.shape[1]),
                        "token_ids_sha256": sha256_text(
                            ",".join(str(value) for value in input_ids[0].tolist())
                        ),
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    tensors = {}
    rows = 0
    output_abs_max = 0.0
    output_abs_sum = 0.0
    output_sq_sum = 0.0
    output_count = 0
    input_abs_max = 0.0
    for index in range(layers):
        input_rows = buffers[index]["input"]
        output_rows = buffers[index]["output"]
        if not input_rows or len(input_rows) != len(output_rows):
            raise RuntimeError(f"layer {index}: incomplete capture in the {label} pass")
        inputs = torch.cat(input_rows, dim=0)
        outputs = torch.cat(output_rows, dim=0)
        tensors[f"layer{index}.input_{label}"] = inputs
        tensors[f"layer{index}.output_{label}"] = outputs
        rows = int(inputs.shape[0])
        abs_outputs = outputs.abs()
        output_abs_max = max(output_abs_max, float(abs_outputs.max()))
        output_abs_sum += float(abs_outputs.sum())
        output_sq_sum += float((outputs * outputs).sum())
        output_count += outputs.numel()
        input_abs_max = max(input_abs_max, float(inputs.abs().max()))
    stats = {
        "output_abs_max": output_abs_max,
        "output_abs_mean": output_abs_sum / output_count if output_count else 0.0,
        "output_rms": (output_sq_sum / output_count) ** 0.5 if output_count else 0.0,
        "input_abs_max": input_abs_max,
    }
    return {
        "tensors": tensors,
        "rows": rows,
        "per_prompt": per_prompt,
        "layers": layers,
        "stats": stats,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--prompts", default="fixtures/prompts/qwen3-teacher-prompts.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    with open(args.prompts, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if prompts.get("schema") != "selvedge/prompt-set/v1":
        raise SystemExit(f"{args.prompts}: unexpected schema")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make the arithmetic mode explicit before any model runs. The f32 pass is
    # only a reference if TF32 is off; the bf16 pass is only reproducible if
    # reduced-precision reduction is off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    arithmetic_mode = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=False, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    model.to(device)

    bf16 = capture_pass(model, tokenizer, prompts, device, args.max_tokens, "bf16")

    # Upcast in place on the target device. The checkpoint is bf16, so this is
    # an exact widening; no new weights are invented.
    model.float()
    f32 = capture_pass(model, tokenizer, prompts, device, args.max_tokens, "f32")

    tensors = {}
    tensors.update(bf16["tensors"])
    tensors.update(f32["tensors"])
    for name, tensor in tensors.items():
        tensors[name] = tensor.to(torch.float32).contiguous()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    metadata = {
        "schema": SCHEMA,
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt_set": prompts["id"],
        "rows": str(bf16["rows"]),
    }
    save_file(tensors, args.out, metadata=metadata)

    summary = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "timestamp_utc": utc_now(),
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "directory": args.model_dir,
        },
        "prompt_set": {
            "id": prompts["id"],
            "path": args.prompts,
            "max_tokens": args.max_tokens,
            "per_prompt": bf16["per_prompt"],
        },
        "dtype_paths": ["bf16", "f32"],
        "bf16_output_stats": bf16["stats"],
        "f32_output_stats": f32["stats"],
        "arithmetic_mode": arithmetic_mode,
        "aligned_rows": bf16["rows"],
        "f32_rows": f32["rows"],
        "layers": bf16["layers"],
        "hidden": int(model.config.hidden_size),
        "intermediate": int(model.config.intermediate_size),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        ),
        "versions": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
        "activations": {
            "path": args.out,
            "bytes": os.path.getsize(args.out),
            "sha256": sha256_file(args.out),
            "tensor_count": len(tensors),
        },
        "notes": [
            "the bf16 path is the checkpoint's native dtype and is the teacher output",
            "the f32 path is the same weights widened exactly, used as the arithmetic reference",
            "only the MLP module boundary is instrumented",
            "trust_remote_code=False and local_files_only=True",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"capture: {args.model_id}@{args.revision} device={device} "
            f"layers={summary['layers']} rows={summary['aligned_rows']} "
            f"bytes={summary['activations']['bytes']}"
        )
        print(f"capture: activations {args.out}")
        print(f"capture: summary {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
