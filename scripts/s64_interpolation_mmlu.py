#!/usr/bin/env python3
"""S64: MMLU retention passes for the interpolation grid, then the selection.

The interpolation diagnostic (`scripts/s60_interpolate.py`) scores GSM8K-cal and
MATH-cal drawn from the train split and writes one checkpoint per coefficient.
MMLU is the **retention input**, not an accuracy objective: a coefficient whose
calibration MMLU drop exceeds the registered 1.0-point bound is ineligible.

This wrapper does two things:

``run`` (GPU)
    invokes the total-cap harness once per coefficient with
    ``--arm interpolated`` (the exact full-arm loader, labelled honestly) and
    ``--tasks mmlu --total-cap-map mmlu=2000``, serially, one model at a time.
    Then it applies the retention bound to the evaluation already in the main
    receipt and writes a selected receipt. It never re-runs the interpolation
    evaluation and never scores a test split.

``select`` (CPU)
    applies the same selection to receipts that already exist, without a GPU.

The S60 queue's earlier ``--arm interp-<t>`` spelling is not accepted by the
scorer's arm choices; the ``interpolated`` arm loads the checkpoint through the
exact ``full-distill`` contract and writes ``interpolation_coefficient``.

Usage
-----
    SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 \
      scripts/s64_interpolation_mmlu.py run \
        --interp-dir results/stage2/s60-interpolation \
        --grid 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
        --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --model-id Qwen/Qwen3.5-0.8B \
        --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
        --receipt results/stage2/s60-interpolation/receipt.json \
        --out results/stage2/s60-interpolation/selected-receipt.json
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

from s60_interpolate import choose_coefficient  # noqa: E402

SCHEMA = "selvedge/s64-interpolation-selection/v1"
REGISTRATION = "docs/ws19-anchored-subset-amendment-preregistration-v2.json"
BOUND = 0.01


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
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


def parse_grid(text: str) -> list[float]:
    return sorted({float(part) for part in text.split(",") if part.strip()})


def load_mmlu(paths: list[str]) -> dict[float, float]:
    """coefficient -> MMLU accuracy, from `s43_score_total_cap.py` receipts."""
    values: dict[float, float] = {}
    for path in paths:
        payload = json.load(open(path, encoding="utf-8"))
        coefficient = payload.get("interpolation_coefficient")
        if coefficient is None:
            arm_report = payload.get("arm_report") or {}
            coefficient = (arm_report.get("interpolation") or {}).get(
                "coefficient")
        for entry in payload.get("tasks") or []:
            if entry.get("task") != "mmlu":
                continue
            metrics = entry.get("all_metrics") or {}
            primary = entry.get("primary_metric") or "acc,none"
            value = metrics.get(primary)
            if coefficient is None or not isinstance(value, (int, float)):
                raise SystemExit(
                    f"s64-mmlu: {path}: cannot tie an MMLU accuracy to a "
                    "coefficient")
            values[float(coefficient)] = float(value)
    return values


def apply_selection(receipt: dict, mmlu: dict[float, float],
                    retention_limit: float = BOUND) -> dict:
    """Attach MMLU to the calibration rows and apply the registered rule.

    The trade block is arithmetic on the calibration rows: it states whether a
    coefficient keeps at least the S58 endpoint's GSM8K-cal accuracy while
    sitting above the S58 endpoint's MATH-cal accuracy. It is a description of
    the segment, not a new selection bar.
    """
    evaluated = [dict(row) for row in receipt["evaluation"]]
    for row in evaluated:
        coefficient = float(row["coefficient"])
        if coefficient in mmlu:
            row["mmlu_cal"] = mmlu[coefficient]
    selection = choose_coefficient(evaluated, retention_limit)
    selection["provisional"] = not bool(mmlu)
    selection["mmlu_present"] = bool(mmlu)
    if not mmlu:
        selection["why_provisional"] = (
            "no MMLU calibration receipts were supplied; the retention bound "
            "was not applied")

    base = next(row for row in evaluated if float(row["coefficient"]) == 0.0)
    s58 = next(row for row in evaluated if float(row["coefficient"]) == 1.0)
    s58_gain = s58["gsm8k_cal"] - base["gsm8k_cal"]
    rows = []
    for row in evaluated:
        coefficient = float(row["coefficient"])
        gsm_delta = row["gsm8k_cal"] - base["gsm8k_cal"]
        rows.append({
            "coefficient": coefficient,
            "gsm8k_cal": row["gsm8k_cal"],
            "math_cal": row["math_cal"],
            "mmlu_cal": row.get("mmlu_cal"),
            "gsm8k_delta_vs_post": gsm_delta,
            "math_delta_vs_post": row["math_cal"] - base["math_cal"],
            "math_delta_vs_s58": row["math_cal"] - s58["math_cal"],
            "s58_gain_fraction": (gsm_delta / s58_gain if s58_gain else None),
            "retains_full_s58_gain": (
                row["gsm8k_cal"] >= s58["gsm8k_cal"]),
            "less_math_damage_than_s58": (
                row["math_cal"] > s58["math_cal"]),
        })
    for row in rows:
        row["retains_gain_with_less_damage"] = (
            row["retains_full_s58_gain"]
            and row["less_math_damage_than_s58"])
    return {
        "evaluated": evaluated,
        "selection": selection,
        "trade": {
            "post_coefficient": 0.0,
            "s58_coefficient": 1.0,
            "s58_gsm8k_gain_calibration": s58_gain,
            "any_coefficient_retains_full_gain_with_less_math_damage": any(
                row["retains_gain_with_less_damage"] for row in rows),
            "rows": rows,
            "basis": (
                "GSM8K-cal and MATH-cal are train-split calibration; the "
                "condition is gsm8k_cal(t) >= gsm8k_cal(1.0) and "
                "math_cal(t) > math_cal(1.0). No test split is read."),
        },
    }


def run_passes(args) -> int:
    grid = parse_grid(args.grid)
    interp_dir = (args.interp_dir if os.path.isabs(args.interp_dir)
                  else os.path.join(ROOT, args.interp_dir))
    os.makedirs(interp_dir, exist_ok=True)
    receipts = []
    for coefficient in grid:
        tag = f"{coefficient:.3f}"
        checkpoint = os.path.join(interp_dir, f"interp-{tag}-arm.pt")
        out = os.path.join(interp_dir, f"mmlu-{tag}.json")
        if not os.path.exists(checkpoint):
            raise SystemExit(
                f"s64-mmlu: {checkpoint} is missing; run the interpolation "
                "checkpoint pass first")
        if os.path.exists(out) and not args.overwrite:
            payload = json.load(open(out, encoding="utf-8"))
            if payload.get("complete") and payload.get(
                    "interpolation_coefficient") is not None:
                print(f"s64-mmlu: {tag} already receipted; skipping", flush=True)
                receipts.append(out)
                continue
        command = [
            sys.executable, os.path.join(HERE, "s43_score_total_cap.py"),
            "--arm", "interpolated",
            "--full-checkpoint", checkpoint,
            "--student-dir", args.student_dir,
            "--model-id", args.model_id,
            "--model-revision", args.model_revision,
            "--tasks", "mmlu",
            "--total-cap-map", f"mmlu={args.mmlu_cap}",
            "--registration", args.registration,
            "--vram-limit-gib", str(args.vram_limit_gib),
            "--out", out,
        ]
        print(f"s64-mmlu: t={tag} start", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(
                f"s64-mmlu: t={tag} failed with exit {completed.returncode}")
        receipts.append(out)
    return select(args, receipts)


def select(args, receipts: list[str] | None = None) -> int:
    receipt_path = (args.receipt if os.path.isabs(args.receipt)
                    else os.path.join(ROOT, args.receipt))
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    if "evaluation" not in receipt:
        raise SystemExit(
            "s64-mmlu: the interpolation receipt has no evaluation rows; run "
            "the interpolation evaluation first")
    paths = receipts or args.mmlu_receipt
    if not paths and args.interp_dir:
        interp_dir = (args.interp_dir if os.path.isabs(args.interp_dir)
                      else os.path.join(ROOT, args.interp_dir))
        paths = sorted(os.path.join(interp_dir, name)
                       for name in os.listdir(interp_dir)
                       if name.startswith("mmlu-") and name.endswith(".json"))
    mmlu = load_mmlu(paths)
    result = apply_selection(receipt, mmlu)
    payload = {
        "schema": SCHEMA,
        "kind": "interpolation-selection-with-mmlu-retention",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "registration": (
            "docs/ws19-stage2-correction-s63-preregistration-v1.json"),
        "source_receipt": {
            "path": os.path.relpath(receipt_path, ROOT),
            "sha256": sha256_file(receipt_path),
            "stamp": receipt.get("stamp"),
            "revision": receipt.get("revision"),
        },
        "mmlu_receipts": [
            {"path": os.path.relpath(path, ROOT),
             "sha256": sha256_file(path)} for path in paths],
        "mmlu": {f"{key:.3f}": value for key, value in sorted(mmlu.items())},
        "selection": result["selection"],
        "trade": result["trade"],
        "evaluation": result["evaluated"],
        "selection_rule": (
            "mean gain over the 0.0 endpoint on GSM8K-cal and MATH-cal; a "
            "coefficient whose calibration MMLU drop exceeds 1.0 point is "
            "ineligible; ties choose the smaller coefficient"),
        "claims": {
            "test_split_used_for_selection": False,
            "mmlu_is_a_retention_input": True,
            "provisional": result["selection"]["provisional"],
        },
    }
    out = (args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out))
    write_json(out, payload)
    selection = result["selection"]
    print(f"s64-mmlu: selected t={selection['coefficient']} "
          f"provisional={selection['provisional']}")
    for row in result["trade"]["rows"]:
        print(f"s64-mmlu: t={row['coefficient']:.3f} "
              f"gsm8k-cal={row['gsm8k_cal']:.4f} "
              f"math-cal={row['math_cal']:.4f} "
              f"mmlu={row['mmlu_cal']} "
              f"retains_full_gain={row['retains_full_s58_gain']} "
              f"less_math_damage={row['less_math_damage_than_s58']}")
    print(f"s64-mmlu: selection -> {args.out}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the MMLU passes, then select")
    run.add_argument("--interp-dir", required=True)
    run.add_argument("--grid", required=True)
    run.add_argument("--student-dir", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--model-revision", required=True)
    run.add_argument("--registration", default=REGISTRATION)
    run.add_argument("--mmlu-cap", type=int, default=2000)
    run.add_argument("--vram-limit-gib", type=float, default=13.0)
    run.add_argument("--receipt", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=run_passes)

    sel = sub.add_parser("select", help="apply the selection to existing receipts")
    sel.add_argument("--receipt", required=True)
    sel.add_argument("--interp-dir", default=None)
    sel.add_argument("--mmlu-receipt", action="append", default=[])
    sel.add_argument("--out", required=True)
    sel.set_defaults(func=select)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
