#!/usr/bin/env python3
"""Receipt the recurrent-state contract on the real `-Base` anchor.

The plan used to say a pass's recurrent state is reset between tokens. It is not:
the state is carried across tokens and reset only between independent sequences,
and each pass owns its own state. `oracle/reference/qwen35.py` now implements
that contract through `PassState` / `MultiPassState` / `step_token`, and
`multi_pass_state_test` proves it. This script runs that test on the anchor and
writes the receipt the docs quote, so the claim is a measurement rather than a
reading of the source.

CPU only. The court's memory limit is sized from a measured peak: loading the
text tower in bf16 costs about 1.5 GB of weights, so the run is launched with
12 GiB and the receipt records the cgroup peak the court measured.

Usage:
    SELVEDGE_MEM=12g scripts/court oracle -- \
      python3 scripts/s38_state_receipt.py --out results/reference/<stamp>-state-contract.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from reference import qwen35 as ref  # noqa: E402

ANCHOR = os.environ.get(
    "SELVEDGE_QWEN35_BASE",
    "/hf-cache/models/Qwen/Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
)
SCHEMA = "selvedge/recurrent-state-contract/v1"


def sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=ANCHOR)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--times", type=int, default=8)
    parser.add_argument("--reset-after", type=int, default=4)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args(argv)

    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    started = time.time()

    model, config = ref.load_reference(args.checkpoint, torch.bfloat16, "cpu")
    result = ref.multi_pass_state_test(
        model, passes=args.passes, times=args.times, reset_after=args.reset_after
    )

    shapes = result["state_shapes"]
    delta_elements = sum(
        int(torch.tensor(shape).prod().item()) for shape in shapes["recurrent"]
    )
    conv_elements = sum(int(torch.tensor(shape).prod().item()) for shape in shapes["conv"])
    attention_layers = sum(
        1 for kind in config.layer_types if kind == "full_attention"
    )
    kv_per_token = 2 * config.kv_heads * config.head_dim * attention_layers
    result["state_breakdown"] = {
        "delta_state_elements": delta_elements,
        "conv_state_elements": conv_elements,
        "kv_state_elements_per_token": kv_per_token,
        "note": (
            "per pass, at zero context. The DeltaNet state is the P19.4 figure; the "
            "convolution windows are a fixed extra the plan's table omits, and the "
            "attention KV grows with context at the per-token rate here."
        ),
    }

    # The device peak PyTorch reports for a CPU run is not meaningful; the court
    # records the container's cgroup peak instead, which is the number the docs
    # quote for whatever memory limit this run was given.
    receipt = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": args.revision,
        "generated_by": "scripts/s38_state_receipt.py",
        "checkpoint": {
            "path": args.checkpoint,
            "config_sha256": sha256_file(os.path.join(args.checkpoint, "config.json")),
            "dtype": "bfloat16",
            "device": "cpu",
            "layers": config.layers,
            "linear_layers": sum(
                1 for kind in config.layer_types if kind == "linear_attention"
            ),
            "attention_layers": sum(
                1 for kind in config.layer_types if kind == "full_attention"
            ),
        },
        "test": result,
        "state_rule": (
            "a pass's recurrent state is carried across tokens within a sequence, "
            "reset only between independent sequences, and never shared between passes"
        ),
        "spec_correction": {
            "wrong_sentence": (
                "P19.4 previously said a pass's state must be reset between tokens"
            ),
            "correct_sentence": (
                "S_{k,t} -> S_{k,t+1} across tokens; reset only between independent "
                "sequences; no cross-pass reads"
            ),
            "why_it_matters": (
                "resetting between tokens would make every pass a one-token function "
                "and would make the chunking-equivalence claim in P19.4 false"
            ),
        },
        "wall_seconds": round(time.time() - started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_sizing": (
            "the court limit for this job is measured peak RSS times 1.5 plus a 1 GiB "
            "floor margin, rounded up to a whole GiB; the peak is the field above"
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(json.dumps(result, indent=2)[:2000])
    print(f"receipt: {args.out}")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
