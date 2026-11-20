#!/usr/bin/env python3
"""S66: re-run the interpolation alpha grid on the enlarged calibration split.

This reads the registered 1,000-item train-split splits in
`docs/s66-calibration-splits-v1.json`, loads the already-written interpolation
checkpoints, and scores GSM8K-cal and MATH-cal under the S54 instrument. It
writes no checkpoint and reads no test split.

The selection rule is the registered one: mean gain over t=0 on the two
calibration tasks, with the calibration MMLU retention values from
`results/stage2/s60-interpolation/selected-receipt.json` as the retention input.
The old 300-item subset is scored from the same per-problem rows so the
enlargement's effect on selection is visible directly.

Usage (GPU):
    SELVEDGE_MEM=12g scripts/court --gpu eval -- \
      python3 scripts/s66_large_calibration.py \
        --interp-dir results/stage2/s60-interpolation \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --registration docs/s66-calibration-splits-v1.json \
        --out results/stage2/s66-calibration/large-grid.json
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

from s60_interpolate import _load_into, _tokenizer, choose_coefficient  # noqa: E402

SCHEMA = "selvedge/s66-large-calibration/v1"


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(rows: list[dict], keep: set[str]) -> dict:
    chosen = [row for row in rows if row["problem_id"] in keep]
    correct = sum(bool(row["flexible_correct"]) for row in chosen)
    return {"items": len(chosen), "correct": correct,
            "accuracy": (correct / len(chosen) if chosen else None)}


def evaluate(model, tokenizer, items: list[dict], task: str,
             max_new_tokens: int, batch_size: int, device: str) -> dict:
    import s54_post_trained_eval as harness

    return harness.evaluate_task(
        model, tokenizer, items, task, "non-thinking", "greedy",
        max_new_tokens, device, None, batch_size)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interp-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--grid", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke-only item limit per task; omit for the full split")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    registration = json.load(open(os.path.join(ROOT, args.registration),
                                  encoding="utf-8"))
    gsm = load_jsonl(os.path.join(ROOT, registration["splits"]["gsm8k_cal"]["path"]))
    math = load_jsonl(os.path.join(ROOT, registration["splits"]["math_cal"]["path"]))
    old_gsm = set(registration["splits"]["gsm8k_cal"]["s60_300_ids"])
    old_math = set(registration["splits"]["math_cal"]["s60_300_ids"])
    for row in gsm:
        row["task"] = "gsm8k"
    for row in math:
        row["task"] = "math500"
    if args.limit is not None:
        gsm = gsm[:args.limit]
        math = math[:args.limit]

    selection_receipt = json.load(open(os.path.join(
        ROOT, "results/stage2/s60-interpolation/selected-receipt.json"),
        encoding="utf-8"))
    mmlu = {float(key): float(value) for key, value in
            (selection_receipt.get("mmlu") or {}).items()}

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False).to(device)
    model.eval()
    tokenizer = _tokenizer(args.model_dir)

    grid = sorted({float(part) for part in args.grid.split(",") if part.strip()})
    interp_dir = os.path.join(ROOT, args.interp_dir)
    rows = []
    for coefficient in grid:
        checkpoint = os.path.join(interp_dir, f"interp-{coefficient:.3f}-arm.pt")
        if not os.path.exists(checkpoint):
            raise SystemExit(f"s66-large-cal: missing checkpoint {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("arm") != "full":
            raise SystemExit(f"s66-large-cal: {checkpoint} is not a full arm")
        _load_into(model, payload["arm_state"])
        gsm_result = evaluate(model, tokenizer, gsm, "gsm8k",
                              args.max_new_tokens, args.batch_size, device)
        math_result = evaluate(model, tokenizer, math, "math500",
                               args.max_new_tokens, args.batch_size, device)
        row = {
            "coefficient": coefficient,
            "checkpoint": os.path.relpath(checkpoint, ROOT),
            "checkpoint_sha256": sha256_file(checkpoint),
            "gsm8k_cal": gsm_result["flexible"]["accuracy"],
            "math_cal": math_result["flexible"]["accuracy"],
            "gsm8k_cal_300": accuracy(gsm_result["per_problem"], old_gsm),
            "math_cal_300": accuracy(math_result["per_problem"], old_math),
            "gsm8k_raw": gsm_result.get("summary"),
            "math_raw": math_result.get("summary"),
        }
        if coefficient in mmlu:
            row["mmlu_cal"] = mmlu[coefficient]
        rows.append(row)
        old_gsm_value = row['gsm8k_cal_300']['accuracy']
        old_math_value = row['math_cal_300']['accuracy']
        old_text = ("n/a" if old_gsm_value is None or old_math_value is None else
                    f"({old_gsm_value:.4f},{old_math_value:.4f})")
        print(f"s66-large-cal: t={coefficient:.3f} "
              f"gsm8k-cal={row['gsm8k_cal']:.4f} "
              f"math-cal={row['math_cal']:.4f} old300={old_text}", flush=True)

    selection = (choose_coefficient(rows)
                 if any(row["coefficient"] == 0.0 for row in rows)
                 else None)
    old_selection = None
    old_rows = [
        {"coefficient": row["coefficient"],
         "gsm8k_cal": row["gsm8k_cal_300"]["accuracy"],
         "math_cal": row["math_cal_300"]["accuracy"],
         **({"mmlu_cal": row["mmlu_cal"]} if "mmlu_cal" in row else {})}
        for row in rows]
    if (len(rows) > 1 and all(row["gsm8k_cal"] is not None
                              and row["math_cal"] is not None
                              for row in old_rows)
            and any(row["coefficient"] == 0.0 for row in old_rows)):
        old_selection = choose_coefficient(old_rows)
    payload = {
        "schema": SCHEMA,
        "kind": "large-calibration-grid",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": {
            "path": args.registration,
            "sha256": sha256_file(os.path.join(ROOT, args.registration)),
            "gsm8k_sha256": registration["splits"]["gsm8k_cal"]["sha256"],
            "math_sha256": registration["splits"]["math_cal"]["sha256"],
            "items_per_task": len(gsm),
            "limit": args.limit,
        },
        "instrument": {
            "tasks": "GSM8K-cal and MATH-cal",
            "decoding": "greedy non-thinking",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "test_split_read": False,
        },
        "grid": grid,
        "rows": rows,
        "selection": selection,
        "selection_300": old_selection,
        "stable": (selection["coefficient"] == old_selection["coefficient"]
                   if selection is not None and old_selection is not None else None),
        "claims": {
            "test_split_used_for_selection": False,
            "calibration_is_train_split_only": True,
            "enlarged_split_is_superset": True,
        },
    }
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    print(f"s66-large-cal: selection 1000="
          f"{selection['coefficient'] if selection else None} "
          f"300={old_selection['coefficient'] if old_selection else None} "
          f"stable={payload['stable']}")
    print(f"s66-large-cal: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
