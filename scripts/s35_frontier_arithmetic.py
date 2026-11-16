#!/usr/bin/env python3
"""The frontier program's arithmetic: teacher route, memory scale, per-pass state.

Three registered design questions were written in prose and priced nowhere. This
script prices them from receipts, so the plan can quote numbers instead of
adjectives:

1. **Teacher route.** The proposal budgeted a top-k teacher-logit cache on disk.
   An online teacher removes that cache entirely. What remains is a *throughput*
   requirement, and it is computed here from the measured student token rate and
   the receipted parameter counts — labelled as an estimate, because no teacher
   forward has been timed on this device.
2. **Memory scale.** Rows are the architecture question; the parameter count of a
   table follows from rows times the anchor's hidden width, and so do the bytes
   per token and the resident footprint. The 65K-row table is one configuration,
   not the architecture.
3. **Per-pass state.** If every recurrent pass carries its own temporal state —
   DeltaNet state and attention KV — that state is part of the budget. It is
   computed here per pass and for the registered pass count.

Every input is a tracked receipt; every output is a computed field. `--check`
re-runs the arithmetic and compares it to an existing receipt, so a plan that
quotes a stale number fails.

Usage:
    scripts/court python -- python3 scripts/s35_frontier_arithmetic.py \
        --out results/frontier-models/20261116T193500Z-frontier-arithmetic.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

SCHEMA = "selvedge/frontier-arithmetic/v1"

ACCOUNTING_REL = "results/accounting/20261116T183609Z-qwen35-08b-text-only.json"
CEILING_REL = "results/frontier-models/20261116-ceiling.json"
VERDICT_REL = "results/t2-verdict.json"
DISK_REL = "results/frontier-models/20261116-disk.json"
HARDWARE_REL = "results/hardware/probe-20261114T184849Z.json"

#: Declared design constants. These are choices, and each is registered in
#: `docs/PROGRAM-PLAN.md` §24 next to the number it produces.
DESIGN = {
    # teacher route
    "teacher_corpus_tokens_k32": 250_000_000,
    "teacher_corpus_tokens_k64": 100_000_000,
    "teacher_k_primary": 32,
    "teacher_k_fidelity": 64,
    "cache_bytes_per_entry_u32_id_plus_bf16_logit": 6,
    "cache_bytes_per_token_header": 4,
    "optional_eval_cache_tokens": 5_000_000,
    # the forward pass costs about a third of a fwd+bwd step; declared, not measured
    "teacher_forward_fraction_of_training_step": 1.0 / 3.0,
    # memory scale
    "memory_rows": (65_536, 1_048_576, 8_388_608),
    "memory_rows_read_per_token_per_pass": 8,
    "memory_passes": 4,
    # per-pass temporal state
    "passes": 4,
    "state_contexts": (8_192, 32_768),
    "state_bytes_per_element": 2,  # bf16
}


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def revision_of(root: str) -> str:
    result = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def read_json(root: str, rel: str) -> dict:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        raise SystemExit(f"s35_frontier_arithmetic: {rel} is missing")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def gib(value: int | float) -> float:
    return value / float(1 << 30)


def compute(root: str) -> dict:
    accounting = read_json(root, ACCOUNTING_REL)
    ceiling = read_json(root, CEILING_REL)
    verdict = read_json(root, VERDICT_REL)
    disk = read_json(root, DISK_REL)
    hardware = read_json(root, HARDWARE_REL)

    text = accounting["model"]["text"]
    hidden = int(text["hidden"])
    student_elements = int(accounting["towers"]["text"]["unique_elements"])
    models = {entry["id"]: entry for entry in ceiling["models"] if entry.get("ok")}
    teacher_2b = int(models["Qwen/Qwen3.5-2B-Base"]["text_elements"])
    teacher_4b = int(models["Qwen/Qwen3.5-4B-Base"]["text_elements"])
    vram_raw = hardware["gpu"]["devices"][0]["vram_total"]
    if isinstance(vram_raw, str) and vram_raw.strip().endswith("MiB"):
        vram_bytes = int(vram_raw.split()[0]) * (1 << 20)
    elif isinstance(vram_raw, int):
        vram_bytes = vram_raw
    else:
        raise SystemExit(
            f"s35_frontier_arithmetic: cannot read the device memory from "
            f"{HARDWARE_REL}: {vram_raw!r}")
    cache_free = int(disk["filesystems"]["cache_free_bytes"])

    # ---- 1. teacher route -------------------------------------------------
    def cache_bytes(tokens: int, k: int) -> int:
        per_token = (k * DESIGN["cache_bytes_per_entry_u32_id_plus_bf16_logit"]
                     + DESIGN["cache_bytes_per_token_header"])
        return tokens * per_token

    cache_k32 = cache_bytes(DESIGN["teacher_corpus_tokens_k32"],
                            DESIGN["teacher_k_primary"])
    cache_k64 = cache_bytes(DESIGN["teacher_corpus_tokens_k64"],
                            DESIGN["teacher_k_fidelity"])
    eval_cache = cache_bytes(DESIGN["optional_eval_cache_tokens"],
                             DESIGN["teacher_k_primary"])
    teacher_vram_2b = teacher_2b * 2  # bf16
    teacher_vram_4b_8bit = teacher_4b * 1  # 8-bit weights
    student_rate = float(verdict["training"]["D"]["tokens_per_second"])
    size_ratio = teacher_2b / student_elements
    forward_fraction = DESIGN["teacher_forward_fraction_of_training_step"]
    required_rate_2b = student_rate * size_ratio * forward_fraction
    required_rate_4b = student_rate * (teacher_4b / student_elements) * forward_fraction

    teacher_route = {
        "cache_removed_bytes_k32": cache_k32,
        "cache_removed_bytes_k64": cache_k64,
        "cache_removed_gib_k32": gib(cache_k32),
        "cache_removed_gib_k64": gib(cache_k64),
        "optional_eval_cache_bytes": eval_cache,
        "optional_eval_cache_gib": gib(eval_cache),
        "disk_saving_fraction_of_k32_cache": 1.0 - eval_cache / cache_k32,
        "cache_free_bytes_at_registration": cache_free,
        "k32_cache_fits_without_online_teacher": cache_k32 <= cache_free,
        "teacher_vram_bytes_2b_bf16": teacher_vram_2b,
        "teacher_vram_gib_2b_bf16": gib(teacher_vram_2b),
        "teacher_vram_bytes_4b_8bit": teacher_vram_4b_8bit,
        "teacher_vram_gib_4b_8bit": gib(teacher_vram_4b_8bit),
        "device_vram_gib": gib(vram_bytes),
        "teacher_2b_fits_device": teacher_vram_2b < vram_bytes,
        "teacher_4b_8bit_fits_device": teacher_vram_4b_8bit < vram_bytes,
        "student_text_elements": student_elements,
        "teacher_2b_text_elements": teacher_2b,
        "teacher_4b_text_elements": teacher_4b,
        "student_tokens_per_second_measured": student_rate,
        "required_teacher_tokens_per_second_2b": required_rate_2b,
        "required_teacher_tokens_per_second_4b_8bit": required_rate_4b,
        "estimate_note": "arithmetic over a measured student rate and declared "
                         "parameter ratios; no teacher forward has been timed on "
                         "this device, so the requirement is an estimate and the "
                         "first stage must measure the teacher's actual rate",
    }

    # ---- 2. memory scale --------------------------------------------------
    rows_read = min(DESIGN["memory_rows_read_per_token_per_pass"],
                    min(DESIGN["memory_rows"]))
    pass_reads = rows_read * DESIGN["memory_passes"]
    memory_scale = []
    for rows in DESIGN["memory_rows"]:
        params = rows * hidden
        bytes_bf16 = params * 2
        distinct_rows = min(pass_reads, rows)
        active_elements = distinct_rows * hidden
        memory_scale.append({
            "rows": rows,
            "width": hidden,
            "parameters": params,
            "resident_bytes_bf16": bytes_bf16,
            "resident_gib_bf16": gib(bytes_bf16),
            "distinct_rows_read_per_token": distinct_rows,
            "active_elements_per_token": active_elements,
            "active_read_bytes_per_token_bf16": active_elements * 2,
            "active_read_bytes_per_token_fp32": active_elements * 4,
            "table_over_checkpoint": bytes_bf16 / int(ceiling["models"][0]["total_elements"] * 2),
        })
    dev = memory_scale[0]
    memory = {
        "scales": memory_scale,
        "development_configuration": {
            "rows": dev["rows"],
            "parameters": dev["parameters"],
            "statement": "one configuration, not the architecture: the architecture "
                         "question is how quality and memory traffic move with rows, "
                         "measured at every registered scale",
        },
        "passes": DESIGN["memory_passes"],
        "rows_read_per_token_per_pass": DESIGN["memory_rows_read_per_token_per_pass"],
        "read_bytes_are": "modelled distinct-identity reads at the declared dtype, "
                          "not measured DRAM or PCIe traffic",
    }

    # ---- 3. per-pass temporal state --------------------------------------
    delta_per_layer = (int(text["linear_value_heads"]) * int(text["linear_value_head_dim"])
                       * int(text["linear_key_head_dim"]))
    delta_total = delta_per_layer * int(text["linear_layers"])
    kv_per_token = (int(text["attention_layers"]) * int(text["attention_kv_heads"])
                    * 2 * int(text["attention_head_dim"]))
    byte_per = DESIGN["state_bytes_per_element"]
    contexts = []
    for context in DESIGN["state_contexts"]:
        kv_elements = kv_per_token * context
        single_pass = (delta_total + kv_elements) * byte_per
        all_passes = single_pass * DESIGN["passes"]
        contexts.append({
            "context_tokens": context,
            "delta_state_elements_per_pass": delta_total,
            "kv_elements_per_pass": kv_elements,
            "single_pass_state_bytes_bf16": single_pass,
            "single_pass_state_gib": gib(single_pass),
            "passes": DESIGN["passes"],
            "all_pass_state_bytes_bf16": all_passes,
            "all_pass_state_gib": gib(all_passes),
            "extra_bytes_vs_shared_state": all_passes - single_pass,
            "extra_gib_vs_shared_state": gib(all_passes - single_pass),
        })
    state = {
        "delta_state_elements_per_layer": delta_per_layer,
        "delta_state_elements_all_linear_layers": delta_total,
        "kv_elements_per_token": kv_per_token,
        "contexts": contexts,
        "requirement": "each pass carries its own recurrent state; no pass may read "
                       "another pass's DeltaNet state or KV, and a chunked forward "
                       "must reproduce the token-by-token state exactly",
        "note": "measured device memory must include this; it is not part of the "
                "per-token unique-active weight union, which counts weights",
    }

    return {
        "schema": SCHEMA,
        "stamp": None,  # filled by main
        "revision": None,
        "created_utc": None,
        "inputs": {
            "accounting": ACCOUNTING_REL,
            "ceiling": CEILING_REL,
            "verdict": VERDICT_REL,
            "disk": DISK_REL,
            "hardware": HARDWARE_REL,
        },
        "design_constants": DESIGN,
        "teacher_route": teacher_route,
        "memory_scale": memory,
        "pass_state": state,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=repo_root())
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--check", default=None,
                        help="recompute and compare against an existing receipt")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.repo)
    payload = compute(root)

    if args.check:
        with open(os.path.join(root, args.check), "r", encoding="utf-8") as handle:
            recorded = json.load(handle)
        problems = []
        for section in ("teacher_route", "memory_scale", "pass_state"):
            if recorded.get(section) != payload[section]:
                problems.append(f"{section} differs from the recomputed arithmetic")
        if problems:
            for problem in problems:
                print(f"s35_frontier_arithmetic: {problem}", file=sys.stderr)
            return 1
        print("s35_frontier_arithmetic: OK — the receipt matches the recomputation")
        return 0

    stamp = args.stamp or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    payload["stamp"] = stamp
    payload["revision"] = revision_of(root)
    payload["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    t = payload["teacher_route"]
    print(
        "s35_frontier_arithmetic: cache removed {:.1f} GiB (k=32) / {:.1f} GiB "
        "(k=64); required teacher rate {:.0f} tok/s (2B bf16) / {:.0f} tok/s "
        "(4B 8-bit); 2B fits {:.1f} GiB of {:.1f} GiB".format(
            t["cache_removed_gib_k32"], t["cache_removed_gib_k64"],
            t["required_teacher_tokens_per_second_2b"],
            t["required_teacher_tokens_per_second_4b_8bit"],
            t["teacher_vram_gib_2b_bf16"], t["device_vram_gib"])
    )
    for row in payload["memory_scale"]["scales"]:
        print("  rows {:>9,}: {:>12,} params, {:>8.2f} GiB bf16, {:>7} active "
              "elements/token".format(row["rows"], row["parameters"],
                                     row["resident_gib_bf16"],
                                     row["active_elements_per_token"]))
    for row in payload["pass_state"]["contexts"]:
        print("  ctx {:>6,}: per-pass state {:.3f} GiB, all {} passes {:.3f} GiB"
              .format(row["context_tokens"], row["single_pass_state_gib"],
                      row["passes"], row["all_pass_state_gib"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
