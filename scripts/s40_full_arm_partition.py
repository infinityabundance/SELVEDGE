#!/usr/bin/env python3
"""Verify the full arm's trainable partition on the real transformer model.

`s40_full_arm_arithmetic.py` derives the counts from the safetensors header. This
script loads the pinned checkpoint through `AutoModelForCausalLM` on CPU and
checks that the object the trainer will actually see has the same partition:

* the whole text tower is `752,393,024` elements;
* the tied embedding is exactly `254,279,680`, and the output projection is the
  same `Parameter` object (so freezing it freezes the head);
* every other text-tower identity is trainable, `498,113,344` elements;
* the exported trainable state does not contain the embedding.

It emits a small receipt so the claim is evidence, not a comment.

Usage:
    SELVEDGE_MEM=6g scripts/court oracle -- python3 \
        scripts/s40_full_arm_partition.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

from frontier import arms, full  # noqa: E402

CACHE = os.environ.get("HF_HOME", "/hf-cache")
STUDENT_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                           "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


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


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", default=STUDENT_DIR)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("results", "stage2",
                                f"s40-full-partition-{stamp()}.json")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.student_dir, dtype=torch.bfloat16, attn_implementation="eager")
    arm = full.FullBackboneArm(model)
    report = arm.active_parameter_report()

    embedding = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    tied = head is not None and head.weight.data_ptr() == embedding.weight.data_ptr()
    state_names = list(arm.state_dict().keys())
    carries_embedding = any(name == "embed_tokens.weight" for name in state_names)
    trainable_counts = {name: parameter.numel()
                        for name, parameter in arm.named_trainable()}
    missing = sorted(set(trainable_counts) - set(state_names))
    extra = sorted(set(state_names) - set(trainable_counts))

    payload = {
        "schema": "selvedge/stage2-full-partition/v1",
        "kind": "full-arm-runtime-partition",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "student_dir": args.student_dir,
        "model_class": type(model).__name__,
        "text_model_class": type(arm.text_model).__name__,
        "report": report,
        "checks": {
            "text_tower_elements": report["text_tower_elements"],
            "frozen_embedding_elements": report["frozen_embedding_elements"],
            "trainable_elements": report["trainable_elements"],
            "sum_is_text_tower": (report["frozen_embedding_elements"]
                                  + report["trainable_elements"]
                                  == report["text_tower_elements"]),
            "frozen_is_exactly_the_tied_parameter": tied,
            "tied_output_projection_has_no_separate_weight": tied,
            "trainable_state_carries_embedding": carries_embedding,
            "trainable_state_missing_keys": missing,
            "trainable_state_extra_keys": extra,
            "trainable_tensor_count": len(trainable_counts),
            "param_dtype": str(arm.parameters()[0].dtype),
        },
        "expected": {
            "text_tower_elements": 752393024,
            "frozen_embedding_elements": 254279680,
            "trainable_elements": 498113344,
            "deployed_active_union_elements": 752393024,
        },
        "command": sys.argv,
    }
    write_json(args.out, payload)
    print(f"partition: text={report['text_tower_elements']} "
          f"frozen={report['frozen_embedding_elements']} "
          f"trainable={report['trainable_elements']}")
    print(f"partition: tied={tied} embedding_in_state={carries_embedding} "
          f"missing={len(missing)} extra={len(extra)}")
    print(f"partition: receipt {args.out}", flush=True)
    ok = (payload["checks"]["sum_is_text_tower"] and tied
          and not carries_embedding and not missing and not extra
          and report["text_tower_elements"] == 752393024
          and report["frozen_embedding_elements"] == 254279680
          and report["trainable_elements"] == 498113344)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
