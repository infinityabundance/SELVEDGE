#!/usr/bin/env python3
"""Parity driver for the Qwen3.5 text-path reference.

Produces one receipt under `results/reference/` containing:

* the pinned `transformers` version and the reference file's hash;
* per-prompt logits comparison between `transformers` (vendor reference) and
  `oracle/reference/qwen35.py` (independent derivation);
* greedy continuations from both, decoded, so a broken reference is visible as
  gibberish rather than as a tolerance that was quietly widened;
* a bitwise determinism check (the same input twice in the same process);
* a dtype-path report (bf16 vs fp32 on the same weights);
* one artifact per prompt: the reference's last-position logits as `.npy`, with
  its SHA-256, plus the full-logits hash and the top-8 tokens.

Run inside the oracle court:

    scripts/docker-court.sh oracle --gpu -- python3 scripts/s33_reference_parity.py \
        --out results/reference/<stamp>-qwen35-reference.json \
        --stamp <stamp> --revision <rev>

Standard library + the court's pinned wheels. Reads the model cache, writes only
under `results/` and `docs/`-adjacent paths given on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from oracle.reference import qwen35  # noqa: E402

PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "2 + 2 =",
    "The chemical symbol for water is",
    "Once upon a time there was a",
)

SCHEMA = "selvedge/qwen35-reference-parity/v1"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    fa = a.detach().float()
    fb = b.detach().float()
    diff = (fa - fb).abs()
    scale = fa.abs().clamp_min(1e-9)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / scale).max().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(fa.flatten(), fb.flatten(), dim=0).item()
        ),
        "argmax_equal": bool(
            torch.equal(fa.argmax(dim=-1), fb.argmax(dim=-1))
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=qwen35.DEFAULT_CHECKPOINT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--reference-steps", type=int, default=8)
    args = parser.parse_args(argv)

    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifact_dir = args.artifact_dir or os.path.join(
        REPO, "results", "reference", f"{stamp}-qwen35"
    )
    os.makedirs(artifact_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("the parity driver requires the GPU court entry point")

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reference_path = os.path.join(REPO, "oracle", "reference", "qwen35.py")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    vendor = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to(device)
        .eval()
    )
    if hasattr(vendor, "language_model"):
        text_model = vendor.language_model
    else:
        text_model = vendor.model

    reference, config = qwen35.load_reference(
        args.checkpoint, torch.bfloat16, device
    )

    prompts: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        ids = input_ids[0]
        with torch.no_grad():
            vendor_logits = vendor(input_ids=input_ids, use_cache=False).logits[0]
            first = reference(ids)
            second = reference(ids)
        determinism = sha256_tensor(first) == sha256_tensor(second)
        parity = compare(vendor_logits, first)

        with torch.no_grad():
            vendor_ids = vendor.generate(
                input_ids.clone(),
                max_new_tokens=args.steps,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
            reference_new = reference.greedy(ids, args.reference_steps)

        vendor_new = vendor_ids[ids.shape[0] :].tolist()
        shared = min(len(vendor_new), len(reference_new))
        agreement = sum(
            1 for a, b in zip(vendor_new[:shared], reference_new[:shared]) if a == b
        )

        artifact_path = os.path.join(artifact_dir, f"prompt-{index}.npy")
        last = first[-1].float().cpu().numpy()
        np.save(artifact_path, last)
        top = torch.topk(first[-1].float(), 8)
        prompts.append(
            {
                "index": index,
                "prompt": prompt,
                "tokens": int(ids.shape[0]),
                "logits_shape": list(first.shape),
                "parity_vs_transformers": parity,
                "reference_is_deterministic": determinism,
                "vendor_continuation": tokenizer.decode(vendor_new),
                "vendor_continuation_ids": vendor_new,
                "reference_continuation": tokenizer.decode(reference_new),
                "reference_continuation_ids": reference_new,
                "continuation_token_agreement": {
                    "compared": shared,
                    "equal": agreement,
                },
                "artifact": {
                    "path": os.path.relpath(artifact_path, REPO),
                    "kind": "last-position logits, float32 npy",
                    "sha256": sha256_file(artifact_path),
                    "full_logits_sha256": sha256_tensor(first),
                    "top8_ids": [int(i) for i in top.indices.tolist()],
                    "top8_logits": [float(v) for v in top.values.tolist()],
                },
            }
        )
        print(
            f"[{index}] {prompt!r} parity_max_abs={parity['max_abs']:.3e} "
            f"argmax_equal={parity['argmax_equal']} agree={agreement}/{shared}",
            flush=True,
        )

    # dtype path: cast the reference to fp32 and re-run the first prompt.
    fp32_reference = qwen35.Qwen35TextReference(config)
    fp32_reference.load_state(
        {
            qwen35.TEXT_PREFIX + name: tensor
            for name, tensor in reference.named_weights().items()
        }
    )
    fp32_reference = fp32_reference.to(device=device, dtype=torch.float32).eval()
    ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"][0].to(device)
    with torch.no_grad():
        bf16_logits = reference(ids)
        fp32_logits = fp32_reference(ids)
    dtype_report = {
        "bf16_vs_fp32": compare(bf16_logits, fp32_logits),
        "bf16_last_position_argmax": int(bf16_logits[-1].float().argmax().item()),
        "fp32_last_position_argmax": int(fp32_logits[-1].float().argmax().item()),
        "bf16_token": tokenizer.decode(
            [int(bf16_logits[-1].float().argmax().item())]
        ),
        "fp32_token": tokenizer.decode(
            [int(fp32_logits[-1].float().argmax().item())]
        ),
    }

    # linear-attention state: carried across a chunk boundary, and reset.
    state_test = qwen35.linear_attention_state_test(reference, layer=0)
    print(
        f"state test: carry_max_abs={state_test['carry_max_abs']:.3e} "
        f"reset_max_abs={state_test['reset_max_abs']:.3e} "
        f"carry_matches={state_test['carry_matches']} "
        f"reset_differs={state_test['reset_differs']}",
        flush=True,
    )

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": args.revision,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "checkpoint_config_sha256": sha256_file(
            os.path.join(args.checkpoint, "config.json")
        ),
        "reference_implementation": {
            "path": os.path.relpath(reference_path, REPO),
            "sha256": sha256_file(reference_path),
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "dtype_path": {
            "vendor_reference": "bfloat16 weights, fp32 norms/softmax/delta-rule",
            "independent_reference": "bfloat16 weights, fp32 norms/softmax/delta-rule "
            "(the same convention, implemented separately)",
            "dtype_comparison": dtype_report,
        },
        "config_echo": {
            "layers": config.layers,
            "hidden": config.hidden,
            "rotary_dim": config.rotary_dim,
            "linear_layers": sum(
                1 for kind in config.layer_types if kind == "linear_attention"
            ),
            "attention_layers": sum(
                1 for kind in config.layer_types if kind == "full_attention"
            ),
            "conv_kernel": config.conv_kernel,
            "tie_word_embeddings": config.tie_word_embeddings,
        },
        "prompts": prompts,
        "linear_attention_state_test": state_test,
        "artifact_dir": os.path.relpath(artifact_dir, REPO),
        "notes": [
            "The vendor reference is transformers==5.23.0, which ships "
            "Qwen3_5ForConditionalGeneration; the checkpoint's MTP head is ignored by "
            "that loader (_keys_to_ignore_on_load_unexpected includes ^mtp.*), matching "
            "the text-only declared path.",
            "The independent reference loads model.language_model.* directly from the "
            "safetensors files and re-derives every equation in its module docstring.",
            "Continuation agreement is compared token by token on the shorter of the "
            "two lengths; a disagreement in the first few tokens is a real defect, not "
            "a sampling artifact, because both runs are greedy.",
            "max_rel is dominated by logits of magnitude near zero and is reported for "
            "completeness only; max_abs, the cosine similarity and argmax equality are "
            "the metrics this receipt is judged on.",
            "The state test runs one real DeltaNet layer (layer 0) three ways: whole "
            "sequence, two chunks with state carried, and the second chunk from a fresh "
            "state. Carry must reproduce the whole-sequence output; reset must not.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"receipt: {args.out}")
    print(f"artifacts: {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
