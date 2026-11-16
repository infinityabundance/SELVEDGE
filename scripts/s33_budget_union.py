#!/usr/bin/env python3
"""Per-token active-union arithmetic for the P19 planned execution path.

The mandate's rule is that a token's audit shows fewer than 1e9 **unique**
active learned elements. This script computes that union for the declared
design point, from the component counts the auditor already produced, so no
number is typed by hand.

What is counted, and why each term is what it is
-----------------------------------------------
A *union* counts distinct elements, not executions. That distinction decides two
of the terms, so both readings are computed and reported:

1. `text_backbone` — the text tower minus its FFN (`always_active_elements` in
   the accounting receipt). Every token reads it.
2. `text_ffn` — the FFN of all 24 layers, declared **dense-active**. It is
   removable in a sharded design; this budget does not assume any sharding, so
   the number below is an upper bound and the headroom is a floor.
3. `depth_lora` — rank-16 adapters on the FFN triple (gate, up, down) of 8
   layers. Per layer: `3 * r * (H + I)` elements, because
   `[r,H]+[I,r]` twice plus `[r,I]+[H,r]`. Reported twice:
   * **one adapter set reused across the 4 passes** (union counts it once), and
   * **a distinct adapter set per pass** (union counts it four times).
   Reuse changes the union not at all and the per-token *work* by the pass count;
   conflating the two is how a budget lies.
4. `memory_projections_gates` — two projections and two gates per memory, each
   `H x H`, i.e. `4 * H^2` per memory. Reported with the same reuse distinction
   as the LoRA term.
5. `memory_rows_read` — `rows_per_token * passes * H` distinct rows touched. The
   memory *bank* is not counted: only the rows a token actually reads. A design
   that reads a different row set per pass is counted as such; the union of row
   *indices* is assumed not to overlap across passes, which is the pessimistic
   reading.

The design constants are declared, not measured. They are recorded verbatim in
the receipt so a later revision that changes them produces a different receipt
rather than a silently different claim.

Usage
-----
    python3 scripts/s33_budget_union.py \
        --accounting results/accounting/<stamp>-qwen35-08b-text-only.json \
        --out results/accounting/<stamp>-qwen35-budget-union.json \
        --stamp <stamp> --revision <rev>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCHEMA = "selvedge/qwen35-budget-union/v1"

# Declared design point. Changing any of these changes the receipt.
DESIGN = {
    "ceiling": 1_000_000_000,
    "lora_rank": 16,
    "lora_layers": 8,
    "passes": 4,
    "memory_projections_and_gates_per_memory": 4,
    "memory_rows_per_token": 256,
    "memory_bank_rows": 65_536,
    "ffn_dense_active": True,
}


def terms(accounting: dict, design: dict) -> dict[str, object]:
    towers = accounting["towers"]
    text = towers["text"]
    vision = towers.get("vision")
    mtp = towers.get("mtp")
    hidden = accounting["model"]["text"]["hidden"]
    intermediate = accounting["model"]["text"]["intermediate"]

    backbone = text["always_active_elements"]
    ffn = text["ffn_elements"]

    rank = design["lora_rank"]
    layers = design["lora_layers"]
    passes = design["passes"]

    lora_one = 3 * rank * (hidden + intermediate) * layers
    lora_per_pass = lora_one * passes

    memory_once = design["memory_projections_and_gates_per_memory"] * hidden * hidden
    memory_per_pass = memory_once * passes

    # Rows read: distinct rows per pass, assumed non-overlapping across passes.
    rows_read = design["memory_rows_per_token"] * hidden * passes
    bank = design["memory_bank_rows"] * hidden

    shared = backbone + ffn + lora_one + memory_once + rows_read
    distinct = backbone + ffn + lora_per_pass + memory_per_pass + rows_read

    vision_elements = vision["checkpoint_elements"] if vision else 0
    mtp_elements = mtp["checkpoint_elements"] if mtp else 0

    def verdict(total: int) -> dict[str, object]:
        return {
            "active_union_elements": total,
            "headroom_elements": design["ceiling"] - total,
            "within_ceiling": total <= design["ceiling"],
        }

    return {
        "terms": {
            "text_backbone_elements": backbone,
            "text_ffn_elements": ffn,
            "depth_lora_one_adapter_set_elements": lora_one,
            "depth_lora_distinct_per_pass_elements": lora_per_pass,
            "memory_projections_gates_one_set_elements": memory_once,
            "memory_projections_gates_distinct_per_pass_elements": memory_per_pass,
            "memory_rows_read_elements": rows_read,
            "memory_bank_total_elements_not_counted": bank,
            "vision_tower_elements_if_active": vision_elements,
            "mtp_head_elements_if_active": mtp_elements,
        },
        "union": {
            "text_plus_plan_shared_adapters": verdict(shared),
            "text_plus_plan_distinct_per_pass": verdict(distinct),
            "text_plus_plan_shared_adapters_plus_vision": verdict(shared + vision_elements),
            "text_plus_plan_shared_adapters_plus_mtp": verdict(shared + mtp_elements),
            "text_plus_plan_shared_adapters_plus_vision_plus_mtp": verdict(
                shared + vision_elements + mtp_elements
            ),
            "text_plus_plan_distinct_per_pass_plus_vision_plus_mtp": verdict(
                distinct + vision_elements + mtp_elements
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(args.accounting, encoding="utf-8") as handle:
        accounting = json.load(handle)
    if accounting.get("schema") != "selvedge/model-accounting/v2":
        raise SystemExit(
            f"{args.accounting} is not a v2 accounting receipt; the budget reads "
            "component counts only from that schema"
        )
    if accounting["active_path"]["declared"] != "text-only":
        raise SystemExit("the budget is defined against the text-only accounting receipt")

    computed = terms(accounting, DESIGN)
    receipt = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": args.revision,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_receipt": {
            "path": args.accounting,
            "stamp": accounting.get("stamp"),
            "revision": accounting.get("revision"),
            "checkpoint_digest": accounting["model"].get("checkpoint_digest"),
        },
        "design": DESIGN,
        **computed,
        "notes": [
            "Every term is a distinct-element count, not an execution count.",
            "The FFN is counted dense-active; a sharded execution can only lower "
            "the union, so the headroom printed here is a floor.",
            "The memory bank is not counted; only the rows a token reads are.",
            "The vision and MTP variants answer 'what if the declared path changed', "
            "not 'what does this path execute'.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"terms": computed["terms"], "union": computed["union"]}, indent=2))
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
