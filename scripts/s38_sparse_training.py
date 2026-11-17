#!/usr/bin/env python3
"""The arithmetic of training the 8M-row memory table sparsely.

The 8,388,608-row table cannot be trained with a dense optimiser: AdamW over the
whole table needs a float32 master weight and two float32 moments, which is three
times the table in float32 — 96 GiB for the table that is already 16 GiB in bf16,
i.e. 112 GiB of accelerator memory before a single activation. This script
computes that number, the sparse design's per-step working set, the host and NVMe
tiers it needs, and the staleness window, from declared design constants and from
receipted model facts (hidden width, layer counts, per-pass state). It writes a
receipt; the docs quote the receipt between generated sentinels.

Everything here is arithmetic over declared constants and two tracked receipts.
No number is a measurement of a run that has not happened, and none is typed into
a document by hand.

Usage:
    scripts/court python -- python3 scripts/s38_sparse_training.py \
        --out results/frontier-models/<stamp>-sparse-memory-training.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = "selvedge/sparse-memory-training/v1"

ACCOUNTING_REL = "results/accounting/20261117T001252Z-qwen35-08b-base-text-only.json"
STATE_REL = "results/reference/20261117T003300Z-qwen35-base-state-contract.json"

#: Declared design constants. Each is registered with the sentence it supports in
#: `docs/WS19-EXECUTION-REGISTRATION.md` §5; changing one changes a registered
#: quantity and must be a dated amendment, not an edit.
DESIGN = {
    "rows_registered": [65536, 1048576, 8388608],
    "rows_primary": 8388608,
    # the stage-2 module reads 8 rows per token at one site
    "reads_per_token": 8,
    # P19.4 registers K = 4 passes; training the table is single-pass until P19.4
    "passes_for_training": 1,
    # sequence 512 x micro-batch 1 x gradient accumulation 4
    "sequence_length": 512,
    "grad_accumulation": 4,
    # bytes per element
    "bf16_bytes": 2,
    "fp32_bytes": 4,
    # AdamW: master weight + first moment + second moment, all fp32
    "optimizer_fp32_vectors": 3,
    # rows whose optimizer state may stay resident on the host before LRU eviction
    "host_optimizer_rows": 1048576,
    # an update bucket is flushed every K optimizer steps; a row's applied update
    # therefore uses gradients at most K steps old
    "update_bucket_steps": 16,
    # the LR of a row touched once every J steps is corrected by its own step count
    "lr_correction": "per-row step count, not the global step",
}


def gib(value: int | float) -> float:
    return value / (1024**3)


def mib(value: int | float) -> float:
    return value / (1024**2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(os.path.join(REPO, ACCOUNTING_REL), encoding="utf-8") as handle:
        accounting = json.load(handle)
    with open(os.path.join(REPO, STATE_REL), encoding="utf-8") as handle:
        state_receipt = json.load(handle)

    hidden = int(accounting["model"]["text"]["hidden"])
    layers = int(accounting["model"]["text"]["layers"])
    linear_layers = int(accounting["model"]["text"]["linear_layers"])
    attention_layers = int(accounting["model"]["text"]["attention_layers"])
    state_breakdown = state_receipt["test"]["state_breakdown"]

    tokens_per_step = (
        DESIGN["sequence_length"] * DESIGN["grad_accumulation"]
    )
    reads_per_step = tokens_per_step * DESIGN["reads_per_token"] * DESIGN["passes_for_training"]
    row_bytes_bf16 = hidden * DESIGN["bf16_bytes"]
    row_bytes_opt_fp32 = hidden * DESIGN["fp32_bytes"] * DESIGN["optimizer_fp32_vectors"]

    scales = []
    for rows in DESIGN["rows_registered"]:
        elements = rows * hidden
        table_bf16 = elements * DESIGN["bf16_bytes"]
        dense_opt = elements * DESIGN["fp32_bytes"] * DESIGN["optimizer_fp32_vectors"]
        scales.append(
            {
                "rows": rows,
                "elements": elements,
                "table_bf16_gib": gib(table_bf16),
                "dense_optimizer_state_gib": gib(dense_opt),
                "dense_total_gib": gib(table_bf16 + dense_opt),
                "table_fits_device": bool(gib(table_bf16) < 16.0),
                "dense_optimizer_fits_device": bool(gib(dense_opt) < 16.0),
            }
        )

    primary = next(item for item in scales if item["rows"] == DESIGN["rows_primary"])

    sparse = {
        "tokens_per_optimizer_step": tokens_per_step,
        "row_reads_per_step": reads_per_step,
        "distinct_rows_per_step_upper_bound": min(
            reads_per_step, DESIGN["rows_primary"]
        ),
        "device_row_bytes_bf16_value_and_grad": row_bytes_bf16 * 2,
        "device_working_set_mib": mib(
            min(reads_per_step, DESIGN["rows_primary"]) * row_bytes_bf16 * 2
        ),
        "host_optimizer_row_bytes_fp32": row_bytes_opt_fp32,
        "host_optimizer_cap_rows": DESIGN["host_optimizer_rows"],
        "host_optimizer_cap_gib": gib(
            DESIGN["host_optimizer_rows"] * row_bytes_opt_fp32
        ),
        "optimizer_steps_to_cover_table_once": (
            DESIGN["rows_primary"] // min(reads_per_step, DESIGN["rows_primary"])
        )
        + (1 if DESIGN["rows_primary"] % min(reads_per_step, DESIGN["rows_primary"]) else 0),
        "staleness_bound_steps": DESIGN["update_bucket_steps"],
        "staleness_bound_tokens": DESIGN["update_bucket_steps"] * tokens_per_step,
        "nvme_write_bytes_per_optimizer_step": min(
            reads_per_step, DESIGN["rows_primary"]
        )
        * row_bytes_opt_fp32,
        "nvme_write_mib_per_optimizer_step": mib(
            min(reads_per_step, DESIGN["rows_primary"]) * row_bytes_opt_fp32
        ),
        "nvme_write_amortised_mib_per_step": mib(
            min(reads_per_step, DESIGN["rows_primary"])
            * row_bytes_opt_fp32
            / DESIGN["update_bucket_steps"]
        ),
    }

    # Peak accelerator memory is an inequality, because the activation term is a
    # measurement this design does not have yet: the throughput job measures it
    # (`docs/WS19-EXECUTION-REGISTRATION.md` §4) and the arithmetic then closes.
    peak = {
        "backbone_bf16_gib": gib(
            int(accounting["towers"]["text"]["bytes"])
        ),
        "table_on_device_gib": 0.0,
        "optimizer_state_on_device_gib": 0.0,
        "sparse_working_set_mib": sparse["device_working_set_mib"],
        "per_pass_state_mib": mib(
            (state_breakdown["delta_state_elements"] * 4)
            + (state_breakdown["conv_state_elements"] * DESIGN["bf16_bytes"])
        ),
        "activation_term": "measured by the throughput job, not assumed here",
        "rule": (
            "peak VRAM = backbone + measured activation peak + sparse working set "
            "+ per-pass state; the table and its optimiser state contribute zero "
            "accelerator bytes by construction"
        ),
        "dense_alternative_shortfall_gib": (
            primary["dense_total_gib"] - 16.0
        ),
    }

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": args.revision,
        "generated_by": "scripts/s38_sparse_training.py",
        "inputs": {
            "accounting": {"path": ACCOUNTING_REL, "stamp": accounting["stamp"]},
            "state_contract": {"path": STATE_REL, "stamp": state_receipt["stamp"]},
            "hidden": hidden,
            "layers": layers,
            "linear_layers": linear_layers,
            "attention_layers": attention_layers,
        },
        "design": DESIGN,
        "scales": scales,
        "primary": primary,
        "sparse": sparse,
        "peak": peak,
        "statement": (
            "the 8,388,608-row table is untrainable with a dense optimizer on a "
            "16 GiB device; the sparse-row, offloaded-optimizer design is the "
            "registered route, and its acceptance evidence is a measured peak and a "
            "sparse-equals-dense equivalence at 65,536 rows"
        ),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(json.dumps(receipt["scales"], indent=1))
    print(json.dumps(sparse, indent=1))
    print(json.dumps(peak, indent=1))
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
