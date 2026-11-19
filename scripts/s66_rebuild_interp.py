#!/usr/bin/env python3
"""S66: rebuild the selected S65 interpolation checkpoint bit for bit.

Inputs are exactly the two endpoints named by the S65 identity receipt:

* the post-trained Qwen3.5-0.8B snapshot at revision
  ``2fc06364715b967f1860aea9cf38778875588b17`` (t = 0), and
* ``results/stage2/runs/s58-stage2a/ckpt-004000.pt`` (t = 1), and
* the scalar coefficient t = 0.375.

The arithmetic is the S60 routine, imported rather than copied:
``interpolate_state_dict`` mixes in float32 and rounds back to each base
tensor's dtype. The checkpoint payload is written with the same schema and key
order as the S65 artifact. The script then hashes the result and refuses a
mismatch against ``--expect-sha256``.

Usage (CPU, inside the Python court):
    SELVEDGE_MEM=10g scripts/court python -- \
      python3 scripts/s66_rebuild_interp.py \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --checkpoint results/stage2/runs/s58-stage2a/ckpt-004000.pt \
        --coefficient 0.375 \
        --out results/stage2/s65-interp-selected/rebuilt-t0.375-full.pt \
        --expect-sha256 10ba2357beef052a434735c682009726b380866e450661de719eb437a5558e0a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from s60_interpolate import (  # noqa: E402
    _base_arm_state, interpolate_state_dict, sha256_file, torch_load, torch_save,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-sha256", default=None)
    args = parser.parse_args(argv)

    checkpoint = os.path.join(ROOT, args.checkpoint)
    payload = torch_load(checkpoint)
    if payload.get("arm") != "full":
        raise SystemExit("s66-rebuild: the candidate is not a full-arm checkpoint")

    import torch
    from transformers import AutoModelForCausalLM

    # The endpoint is read at the snapshot's native bf16 dtype. No kernel is
    # needed: this is a parameter-level mix.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False)
    base_state = _base_arm_state(model, payload["arm_state"])
    mixed = interpolate_state_dict(base_state, payload["arm_state"],
                                   args.coefficient)
    out_payload = {
        "schema": "selvedge/stage2-ckpt/v1",
        "interpolation": {
            "coefficient": args.coefficient,
            "base": "post-trained 0.8B (untrained endpoint)",
            "candidate": args.checkpoint,
            "candidate_sha256": sha256_file(checkpoint),
        },
        "step": payload.get("step"),
        "config_sha256": payload.get("config_sha256"),
        "arm": "full",
        "arm_state": mixed,
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch_save(out_payload, out)
    actual = sha256_file(out)
    size = os.path.getsize(out)
    result = {
        "schema": "selvedge/s66-interpolation-rebuild/v1",
        "out": args.out,
        "bytes": size,
        "sha256": actual,
        "coefficient": args.coefficient,
        "candidate": args.checkpoint,
        "candidate_sha256": sha256_file(checkpoint),
        "model_dir": args.model_dir,
        "expected_sha256": args.expect_sha256,
        "match": (actual == args.expect_sha256
                  if args.expect_sha256 else None),
        "state_elements": sum(
            int(value.numel()) for value in mixed.values()
            if hasattr(value, "numel")),
    }
    receipt = out + ".json"
    with open(receipt, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s66-rebuild: {args.out} bytes={size} sha256={actual}")
    if args.expect_sha256 and actual != args.expect_sha256:
        raise SystemExit(
            "s66-rebuild: hash mismatch; the rebuild is not the released "
            f"artifact (expected {args.expect_sha256})")
    print(f"s66-rebuild: match={result['match']} receipt={receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
