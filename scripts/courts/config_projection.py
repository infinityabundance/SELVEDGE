#!/usr/bin/env python3
"""Architecture-level parameter projection from a Qwen3 config alone.

This is not a substitute for `selvedge inspect-model`: it reads no weights, so
it cannot verify shapes or the embedding/output tie. It exists to answer one
question before a large download: would this architecture's always-on backbone
fit under the active-parameter ceiling at all? The report says so explicitly
and carries `kind: "config_projection"`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

CEILING = 1_000_000_000


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--revision-git", required=True)
    args = parser.parse_args(argv)

    raw = open(args.config, "rb").read()
    config = json.loads(raw)
    if config.get("model_type") != "qwen3":
        raise SystemExit("config_projection: only model_type qwen3 is supported")

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", hidden // heads))
    vocab = int(config["vocab_size"])
    tied = bool(config.get("tie_word_embeddings", False))

    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    embed = vocab * hidden
    attention_per_layer = q_width * hidden + 2 * (kv_width * hidden) + hidden * q_width + 2 * head_dim
    norm_per_layer = 2 * hidden
    ffn_per_layer = 3 * intermediate * hidden
    backbone = embed + layers * (attention_per_layer + norm_per_layer) + hidden
    ffn = layers * ffn_per_layer
    unique = backbone + (0 if tied else embed) + ffn
    checkpoint_elements = backbone + (0 if tied else embed) + ffn + (embed if tied else 0)
    max_ffn = max(0, CEILING - backbone)
    if unique <= CEILING:
        verdict = "not_binding"
        binding = False
        dense_compliant = True
        backbone_fits = True
    elif backbone <= CEILING:
        verdict = "binding_backbone_fits"
        binding = True
        dense_compliant = False
        backbone_fits = True
    else:
        verdict = "impossible_backbone_over_ceiling"
        binding = True
        dense_compliant = False
        backbone_fits = False

    report = {
        "schema": "selvedge/config-projection/v1",
        "kind": "config_projection",
        "ok": True,
        "stamp": args.stamp,
        "revision": args.revision_git,
        "generated_by": "scripts/courts/config_projection.py",
        "notes": [
            "reads config.json only; no weights are downloaded or measured",
            "cannot verify tensor shapes or whether the embedding/output tie exists in the bytes",
            "for an exact count on real weights use `selvedge inspect-model`",
        ],
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "hidden": hidden,
            "intermediate": intermediate,
            "layers": layers,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "vocab": vocab,
            "tie_word_embeddings": tied,
        },
        "totals": {
            "embedding": embed,
            "attention": layers * attention_per_layer,
            "norm": layers * norm_per_layer + hidden,
            "ffn": ffn,
            "lm_head": embed,
        },
        "budget": {
            "ceiling": CEILING,
            "checkpoint_tensor_elements": checkpoint_elements,
            "unique_learned_elements": unique,
            "always_active_backbone_elements": backbone,
            "removable_ffn_elements": ffn,
            "headroom_elements": CEILING - unique,
            "max_compliant_ffn_elements": max_ffn,
            "max_compliant_ffn_fraction": (min(1.0, max_ffn / ffn) if ffn else 0.0),
            "ceiling_binding": binding,
            "dense_execution_compliant": dense_compliant,
            "backbone_fits": backbone_fits,
            "verdict": verdict,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"config-projection: {args.model_id}@{args.revision} unique={unique} "
        f"backbone={backbone} ffn={ffn} verdict={verdict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
