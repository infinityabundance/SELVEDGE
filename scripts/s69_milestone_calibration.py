#!/usr/bin/env python3
"""S69: score preference milestones on the registered calibration splits.

Two receipted subcommands:

``score``
    Load each merged full-arm checkpoint over the post-trained model directory
    and score the registered 1,000-item GSM8K-cal and MATH-cal splits with the
    S54 instrument (greedy, non-thinking, cap 2048). A completed row is written
    to ``<out>.partial`` after every checkpoint, so a long GPU run resumes
    without re-scoring. Per-problem rows are kept in the payload so a later
    paired comparison does not need to re-generate.

``select``
    Apply the registered selection rule from the score receipt and the
    MMLU-cal receipts: highest MATH-cal among checkpoints whose MMLU-cal is
    within 1.0 point of the soup's; adoption requires MATH-cal strictly above
    the soup's and GSM8K-cal within 1.0 point. No test split is read.

The soup's own calibration row is read from the S67 calibration receipt and is
never regenerated; the same splits, protocol and extraction are used.

Usage
-----
    SELVEDGE_MEM=14g scripts/court --gpu eval -- python3 \
      scripts/s69_milestone_calibration.py score \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --checkpoints 500000=results/.../ckpt-000123.pt,1000000=... \
        --out results/stage2/s69-calibration/preference.json

    SELVEDGE_MEM=4g scripts/court python -- python3 \
      scripts/s69_milestone_calibration.py select \
        --calibration results/stage2/s69-calibration/preference.json \
        --mmlu 500000=results/.../mmlu-500000.json,... \
        --out results/stage2/s69-calibration/selection.json
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
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

SCHEMA_SCORE = "selvedge/s69-milestone-calibration/v1"
SCHEMA_SELECT = "selvedge/s69-preference-selection/v1"
REGISTRATION = "docs/s69-preference-from-champion-preregistration-v1.json"
CALIBRATION_REGISTRATION = "docs/s66-calibration-splits-v1.json"
SOUP_CALIBRATION = "results/stage2/s67-soup/calibration.json"
SOUP_COEFFICIENT = 0.375
RETENTION_BOUND = 0.01


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


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_pairs(text: str) -> list[tuple[int, str]]:
    """``cadence=path`` comma list -> sorted [(cadence, path)]."""
    values: list[tuple[int, str]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"s69-cal: checkpoint entry {part!r} is not "
                             "cadence=path")
        cadence_text, path = part.split("=", 1)
        try:
            cadence = int(cadence_text)
        except ValueError:
            raise SystemExit(f"s69-cal: cadence {cadence_text!r} is not an int")
        values.append((cadence, path.strip()))
    if not values:
        raise SystemExit("s69-cal: no checkpoints given")
    return sorted(values)


def soup_row(path: str = SOUP_CALIBRATION) -> dict:
    payload = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    rows = [row for row in payload.get("rows") or []
            if float(row.get("coefficient")) == SOUP_COEFFICIENT]
    if not rows:
        raise SystemExit(f"s69-cal: {path} has no alpha={SOUP_COEFFICIENT} row")
    row = dict(rows[0])
    row["receipt"] = path
    row["receipt_sha256"] = sha256_file(os.path.join(ROOT, path))
    return row


def evaluate(model, tokenizer, items, task, max_new_tokens, batch_size,
             device) -> dict:
    import s54_post_trained_eval as harness

    return harness.evaluate_task(
        model, tokenizer, items, task, "non-thinking", "greedy",
        max_new_tokens, device, None, batch_size)


def mmlu_value(path: str) -> dict:
    payload = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    entries = [entry for entry in payload.get("tasks") or []
               if entry.get("task") == "mmlu"]
    if not entries:
        raise SystemExit(f"s69-cal: {path} has no mmlu task entry")
    entry = entries[0]
    metrics = entry.get("all_metrics") or {}
    value = metrics.get(entry.get("primary_metric") or "acc,none")
    if not isinstance(value, (int, float)):
        raise SystemExit(f"s69-cal: {path} has no numeric mmlu accuracy")
    return {
        "accuracy": float(value),
        "receipt": path,
        "receipt_sha256": sha256_file(os.path.join(ROOT, path)),
        "items": (entry.get("sample_count") or {}).get(
            entry.get("primary_metric") or "acc,none"),
    }


# ---------------------------------------------------------------------------
# score


def run_score(args) -> int:
    registration = json.load(
        open(os.path.join(ROOT, args.calibration_registration), encoding="utf-8"))
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
            f"s69-cal: expected 1,000 items per task, got {len(gsm)}/{len(math)}")

    checkpoints = parse_pairs(args.checkpoints)
    out_path = os.path.join(ROOT, args.out)
    partial_path = out_path + ".partial"
    rows: list[dict] = []
    if args.resume and os.path.exists(partial_path):
        partial = json.load(open(partial_path, encoding="utf-8"))
        rows = list(partial.get("rows") or [])
        done = {int(row["cadence"]) for row in rows}
        checkpoints = [(cadence, path) for cadence, path in checkpoints
                       if cadence not in done]
        print(f"s69-cal: resumed {len(done)} checkpoint(s); remaining "
              f"{[cadence for cadence, _ in checkpoints]}", flush=True)

    if checkpoints:
        import torch
        from transformers import AutoModelForCausalLM
        from s60_interpolate import _load_into, _tokenizer, torch_load

        if not torch.cuda.is_available():
            raise SystemExit("s69-cal: CUDA required; run through "
                             "scripts/court --gpu eval")
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa").to(device)
        model.eval()
        tokenizer = _tokenizer(args.model_dir)
        for cadence, checkpoint in checkpoints:
            path = os.path.join(ROOT, checkpoint)
            payload = torch_load(path)
            if payload.get("arm") != "full":
                raise SystemExit(f"s69-cal: {checkpoint} is not a full arm")
            _load_into(model, payload["arm_state"])
            model.eval()
            gsm_result = evaluate(model, tokenizer, gsm, "gsm8k",
                                  args.max_new_tokens, args.batch_size, device)
            math_result = evaluate(model, tokenizer, math, "math500",
                                   args.max_new_tokens, args.batch_size, device)
            row = {
                "cadence": cadence,
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_step": payload.get("step"),
                "checkpoint_processed_tokens": payload.get("processed_tokens"),
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
                "gsm8k_per_problem": gsm_result["per_problem"],
                "math_per_problem": math_result["per_problem"],
            }
            rows.append(row)
            write_json(partial_path, {
                "schema": SCHEMA_SCORE,
                "kind": "s69-milestone-calibration-partial",
                "created_utc": utcnow(),
                "rows": rows,
            })
            print(f"s69-cal: cadence={cadence} "
                  f"gsm8k-cal={row['gsm8k_cal']:.4f} "
                  f"math-cal={row['math_cal']:.4f}", flush=True)

    rows.sort(key=lambda row: row["cadence"])
    payload = {
        "schema": SCHEMA_SCORE,
        "kind": "s69-milestone-calibration",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": args.registration,
            "sha256": sha256_file(os.path.join(ROOT, args.registration)),
        },
        "calibration_registration": {
            "path": args.calibration_registration,
            "sha256": sha256_file(os.path.join(
                ROOT, args.calibration_registration)),
            "items_per_task": 1000,
            "source": "train split only; no test split read",
        },
        "instrument": {
            "tasks": "GSM8K-cal and MATH-cal (1,000 each)",
            "decoding": "greedy non-thinking",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "attention": "sdpa",
            "test_split_read": False,
        },
        "soup_reference": soup_row(args.soup_calibration),
        "rows": rows,
        "claims": {
            "test_split_used_for_selection": False,
            "soup_row_not_regenerated": True,
        },
    }
    write_json(out_path, payload)
    print(f"s69-cal: wrote {len(rows)} row(s) -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# select


def run_select(args) -> int:
    calibration = json.load(open(os.path.join(ROOT, args.calibration),
                                 encoding="utf-8"))
    rows = {int(row["cadence"]): row for row in calibration.get("rows") or []}
    if not rows:
        raise SystemExit("s69-cal: the calibration receipt has no rows")
    mmlu = {}
    for part in args.mmlu:
        if "=" not in part:
            raise SystemExit(f"s69-cal: --mmlu entry {part!r} is not cadence=path")
        cadence_text, path = part.split("=", 1)
        mmlu[int(cadence_text)] = mmlu_value(path)
    missing = sorted(set(rows) - set(mmlu))
    if missing:
        raise SystemExit(
            "s69-cal: the retention filter is required by the registration; "
            f"MMLU-cal is missing for cadence(s) {missing}")
    soup = calibration.get("soup_reference") or soup_row()
    soup_mmlu = float(soup.get("mmlu_cal"))

    for cadence, row in rows.items():
        entry = mmlu[cadence]
        row["mmlu_cal"] = entry["accuracy"]
        row["mmlu_cal_receipt"] = entry["receipt"]
        row["mmlu_cal_receipt_sha256"] = entry["receipt_sha256"]
        row["eligible"] = entry["accuracy"] >= soup_mmlu - RETENTION_BOUND

    eligible = [row for row in rows.values() if row["eligible"]]
    selected = None
    if eligible:
        selected = max(eligible, key=lambda row: (
            row["math_cal"], row["gsm8k_cal"], -int(row["cadence"])))
    adoption = bool(
        selected is not None
        and selected["math_cal"] > float(soup["math_cal"])
        and selected["gsm8k_cal"] >= float(soup["gsm8k_cal"]) - RETENTION_BOUND)
    payload = {
        "schema": SCHEMA_SELECT,
        "kind": "s69-preference-selection",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": calibration.get("preregistration"),
        "calibration": {
            "path": args.calibration,
            "sha256": sha256_file(os.path.join(ROOT, args.calibration)),
        },
        "soup_reference": soup,
        "rows": [rows[cadence] for cadence in sorted(rows)],
        "selection": {
            "rule": ("highest MATH-cal among checkpoints whose MMLU-cal is "
                     "within 1.0 point of the soup's; tie-break higher "
                     "GSM8K-cal, then the earlier cadence"),
            "cadence": (selected or {}).get("cadence"),
            "checkpoint": (selected or {}).get("checkpoint"),
            "checkpoint_sha256": (selected or {}).get("checkpoint_sha256"),
            "math_cal": (selected or {}).get("math_cal"),
            "gsm8k_cal": (selected or {}).get("gsm8k_cal"),
            "mmlu_cal": (selected or {}).get("mmlu_cal"),
            "soup_math_cal": soup.get("math_cal"),
            "soup_gsm8k_cal": soup.get("gsm8k_cal"),
            "soup_mmlu_cal": soup.get("mmlu_cal"),
            "adoption_gate": ("MATH-cal strictly above the soup's and "
                              "GSM8K-cal within 1.0 point of the soup's"),
            "adopted": adoption,
            "test_split_read_for_selection": False,
        },
        "claims": {
            "selection_is_calibration_only": True,
            "mmlu_is_a_retention_filter": True,
            "no_test_split_read": True,
        },
    }
    out_path = os.path.join(ROOT, args.out)
    write_json(out_path, payload)
    print(f"s69-cal: selected cadence={(selected or {}).get('cadence')} "
          f"adopted={adoption} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score milestones on calibration")
    score.add_argument("--model-dir", required=True)
    score.add_argument("--checkpoints", required=True,
                       help="comma-separated cadence=path entries")
    score.add_argument("--registration", default=REGISTRATION)
    score.add_argument("--calibration-registration",
                       default=CALIBRATION_REGISTRATION)
    score.add_argument("--soup-calibration", default=SOUP_CALIBRATION)
    score.add_argument("--max-new-tokens", type=int, default=2048)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--resume", action="store_true")
    score.add_argument("--out", required=True)
    score.set_defaults(func=run_score)

    select = sub.add_parser("select", help="apply the selection rule")
    select.add_argument("--calibration", required=True)
    select.add_argument("--mmlu", action="append", default=[],
                        help="cadence=path to an MMLU total-cap receipt; "
                             "repeatable")
    select.add_argument("--out", required=True)
    select.set_defaults(func=run_select)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
