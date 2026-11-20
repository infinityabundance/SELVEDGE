#!/usr/bin/env python3
"""S67: select the soup coefficient on the enlarged train-split calibration.

Scores every soup checkpoint on the registered 1,000-item GSM8K-cal and
MATH-cal splits (`docs/s66-calibration-splits-v1.json`) under the S54
instrument, attaches the measured 2,000-item calibration MMLU as the retention
filter, and applies the registered selection rule: mean gain over the 0.0
endpoint on the two calibration tasks, MMLU drop above 1.0 point ineligible,
ties to the smaller coefficient. No test split is read.

A completed row is written to ``<out>.partial`` after every coefficient, so a
long GPU run is resumable without re-scoring. MMLU receipts are read from
``--mmlu-dir`` (files ``mmlu-<alpha>.json`` written by
``scripts/s43_score_total_cap.py`` with ``--total-cap-map mmlu=2000``).

Usage (GPU):
    SELVEDGE_MEM=12g scripts/court --gpu eval -- \
      python3 scripts/s67_soup_calibration.py \
        --soup-dir results/stage2/s67-soup \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc064... \
        --grid 0,0.375,0.5,0.625,0.75 \
        --out results/stage2/s67-soup/calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from s60_interpolate import (  # noqa: E402
    _load_into, _tokenizer, choose_coefficient, sha256_file, torch_load,
)

SCHEMA = "selvedge/s67-soup-calibration/v1"
REGISTRATION = "docs/s66-calibration-splits-v1.json"
PREREG = "docs/s67-resolution-instrument-preregistration-v1.json"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_mmlu(directory: str) -> dict[float, dict]:
    values: dict[float, dict] = {}
    if not os.path.isdir(directory):
        return values
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("mmlu-") and name.endswith(".json")):
            continue
        path = os.path.join(directory, name)
        payload = json.load(open(path, encoding="utf-8"))
        coefficient = payload.get("interpolation_coefficient")
        if coefficient is None:
            raise SystemExit(f"s67-cal: {name} has no coefficient")
        entries = [entry for entry in payload.get("tasks") or []
                   if entry.get("task") == "mmlu"]
        if not entries:
            raise SystemExit(f"s67-cal: {name} has no mmlu task entry")
        entry = entries[0]
        metrics = entry.get("all_metrics") or {}
        value = metrics.get(entry.get("primary_metric") or "acc,none")
        if not isinstance(value, (int, float)):
            raise SystemExit(f"s67-cal: {name} has no numeric mmlu accuracy")
        values[float(coefficient)] = {
            "accuracy": float(value),
            "receipt": os.path.relpath(path, ROOT),
            "receipt_sha256": sha256_file(path),
        }
    return values


def evaluate(model, tokenizer, items, task, max_new_tokens, batch_size,
             device) -> dict:
    import s54_post_trained_eval as harness

    return harness.evaluate_task(
        model, tokenizer, items, task, "non-thinking", "greedy",
        max_new_tokens, device, None, batch_size)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soup-dir", default="results/stage2/s67-soup")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--registration", default=REGISTRATION)
    parser.add_argument("--grid", default="0,0.375,0.5,0.625")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-row", default=None,
                        help="receipt containing an already-scored alpha=0 row "
                             "on the same 1,000-item splits")
    parser.add_argument("--verify-base-equivalence", default=None,
                        help="checkpoint the base row was scored from; it must "
                             "be tensor-for-tensor equal to the soup alpha=0 "
                             "checkpoint before the row is reused")
    parser.add_argument("--out", default="results/stage2/s67-soup/calibration.json")
    args = parser.parse_args(argv)

    registration = json.load(open(os.path.join(ROOT, args.registration),
                                  encoding="utf-8"))
    gsm = load_jsonl(os.path.join(ROOT,
                                  registration["splits"]["gsm8k_cal"]["path"]))
    math = load_jsonl(os.path.join(ROOT,
                                   registration["splits"]["math_cal"]["path"]))
    for row in gsm:
        row["task"] = "gsm8k"
    for row in math:
        row["task"] = "math500"
    if len(gsm) != 1000 or len(math) != 1000:
        raise SystemExit(
            f"s67-cal: expected 1,000 items per task, got {len(gsm)}/{len(math)}")

    soup_dir = args.soup_dir if os.path.isabs(args.soup_dir) else os.path.join(
        ROOT, args.soup_dir)
    grid = sorted({float(part) for part in args.grid.split(",") if part.strip()})
    out_path = os.path.join(ROOT, args.out)
    partial_path = out_path + ".partial"
    rows: list[dict] = []
    if args.resume and os.path.exists(partial_path):
        partial = json.load(open(partial_path, encoding="utf-8"))
        rows = list(partial.get("rows") or [])
        done = {float(row["coefficient"]) for row in rows}
        grid = [value for value in grid if value not in done]
        print(f"s67-cal: resumed {len(done)} coefficient(s); remaining {grid}",
              flush=True)

    if args.base_row and not any(float(row["coefficient"]) == 0.0
                                 for row in rows):
        external = json.load(open(os.path.join(ROOT, args.base_row),
                                  encoding="utf-8"))
        base_rows = [row for row in external.get("rows") or []
                     if float(row["coefficient"]) == 0.0]
        if not base_rows:
            raise SystemExit(f"s67-cal: {args.base_row} has no alpha=0 row")
        external_base = dict(base_rows[0])
        external_items = (external.get("registration") or {}).get(
            "items_per_task")
        row_items = external_base.get("gsm8k_items")
        if external_items != 1000 or (row_items is not None and row_items != 1000):
            raise SystemExit(
                "s67-cal: the external alpha=0 row was not scored on the "
                f"registered 1,000-item splits (items_per_task={external_items}, "
                f"row items={row_items})")
        external_sha = (external.get("registration") or {}).get("sha256")
        if external_sha != sha256_file(os.path.join(ROOT, args.registration)):
            raise SystemExit(
                "s67-cal: the external alpha=0 row used a different "
                "calibration registration; refusing to mix rows")
        if not args.verify_base_equivalence:
            raise SystemExit(
                "s67-cal: reusing an external alpha=0 row requires "
                "--verify-base-equivalence")
        import torch

        ours_path = os.path.join(soup_dir, "interp-0.000-arm.pt")
        ours = torch_load(ours_path)["arm_state"]
        theirs = torch_load(os.path.join(ROOT, args.verify_base_equivalence))[
            "arm_state"]
        if sorted(ours) != sorted(theirs):
            raise SystemExit("s67-cal: base checkpoints have different keys")
        for key in ours:
            if not torch.equal(ours[key], theirs[key]):
                raise SystemExit(
                    f"s67-cal: {key} differs between the reused base "
                    "checkpoint and the soup alpha=0 checkpoint; refusing "
                    "to mix rows")
        external_base["source"] = args.base_row
        external_base["verified_tensor_equal_to"] = args.verify_base_equivalence
        rows.append(external_base)
        print(f"s67-cal: reused verified alpha=0 row from {args.base_row}",
              flush=True)

    if grid:
        import torch
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False).to(device)
        model.eval()
        tokenizer = _tokenizer(args.model_dir)
        for alpha in grid:
            checkpoint = os.path.join(soup_dir, f"interp-{alpha:.3f}-arm.pt")
            if not os.path.exists(checkpoint):
                raise SystemExit(f"s67-cal: missing checkpoint {checkpoint}")
            payload = torch_load(checkpoint)
            if payload.get("arm") != "full":
                raise SystemExit(f"s67-cal: {checkpoint} is not a full arm")
            _load_into(model, payload["arm_state"])
            gsm_result = evaluate(model, tokenizer, gsm, "gsm8k",
                                  args.max_new_tokens, args.batch_size, device)
            math_result = evaluate(model, tokenizer, math, "math500",
                                   args.max_new_tokens, args.batch_size, device)
            row = {
                "coefficient": alpha,
                "checkpoint": os.path.relpath(checkpoint, ROOT),
                "checkpoint_sha256": sha256_file(checkpoint),
                "gsm8k_cal": gsm_result["flexible"]["accuracy"],
                "math_cal": math_result["flexible"]["accuracy"],
                "gsm8k_items": gsm_result["items"],
                "math_items": math_result["items"],
                "gsm8k_correct": gsm_result["flexible"]["correct"],
                "math_correct": math_result["flexible"]["correct"],
                "wall_seconds": (gsm_result.get("wall_seconds", 0)
                                 + math_result.get("wall_seconds", 0)),
                "generated_tokens": (gsm_result.get("generated_tokens", 0)
                                     + math_result.get("generated_tokens", 0)),
            }
            rows.append(row)
            write_json(partial_path, {
                "schema": SCHEMA,
                "kind": "soup-calibration-partial",
                "created_utc": utcnow(),
                "registration_sha256": sha256_file(
                    os.path.join(ROOT, args.registration)),
                "rows": rows,
            })
            print(f"s67-cal: alpha={alpha:.3f} "
                  f"gsm8k-cal={row['gsm8k_cal']:.4f} "
                  f"math-cal={row['math_cal']:.4f}", flush=True)

    mmlu = load_mmlu(soup_dir)
    for row in rows:
        entry = mmlu.get(float(row["coefficient"]))
        if entry is not None:
            row["mmlu_cal"] = entry["accuracy"]
            row["mmlu_cal_receipt"] = entry["receipt"]
            row["mmlu_cal_receipt_sha256"] = entry["receipt_sha256"]
    missing_mmlu = [row["coefficient"] for row in rows
                    if not isinstance(row.get("mmlu_cal"), (int, float))]
    if missing_mmlu:
        raise SystemExit(
            "s67-cal: the retention filter is required by the registration; "
            f"MMLU-cal is missing for {missing_mmlu}")

    rows.sort(key=lambda row: row["coefficient"])
    selection = choose_coefficient(rows, retention_limit=0.01)
    selection["retention_required"] = True
    selection["objective"] = ("mean gain over alpha=0 on GSM8K-cal and "
                              "MATH-cal; MMLU-cal drop > 1.0 point ineligible")
    payload = {
        "schema": SCHEMA,
        "kind": "soup-calibration",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": PREREG,
            "sha256": sha256_file(os.path.join(ROOT, PREREG)),
        },
        "calibration_registration": {
            "path": args.registration,
            "sha256": sha256_file(os.path.join(ROOT, args.registration)),
            "items_per_task": 1000,
            "source": "train split only; no test split read",
        },
        "instrument": {
            "tasks": "GSM8K-cal and MATH-cal (1,000 each)",
            "decoding": "greedy non-thinking",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "test_split_read": False,
        },
        "rows": rows,
        "selection": selection,
        "claims": {
            "test_split_used_for_selection": False,
            "mmlu_is_a_retention_filter": True,
            "selection_rule_is_registered": True,
        },
    }
    write_json(out_path, payload)
    print(f"s67-cal: selected alpha={selection.get('coefficient')} "
          f"gain={selection.get('calibration_gain')} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
