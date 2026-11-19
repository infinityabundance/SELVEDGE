#!/usr/bin/env python3
"""S60: weight-interpolation diagnostic between the post-trained model and S58.

S58's stage-2a arm improved GSM8K (+2.88 points, paired interval barely above
zero) and lost MATH-500 (11.0 points, interval entirely below zero). A plain
weight interpolation asks a different question from either training or data
selection: is there a point on the segment

    theta(t) = (1 - t) * theta_post_trained + t * theta_s58_best

that keeps the GSM8K gain without the MATH loss? If yes, the S58 update moved
the model along a harmful direction at full strength, and the diagnostic says
so without another training run. If no point on the segment improves both, the
segment itself is the evidence and the next route is not interpolation.

Discipline
----------
* The grid is fixed before the run (`--grid`), and every coefficient is written
  as a checkpoint in the same schema `train.py` wrote, so the S54 eval driver
  reads it without a special path.
* Coefficients are chosen on a **calibration split drawn from the train split**
  (GSM8K-cal and MATH-cal), never on MATH-500 test or GSM8K test. The test
  instruments are only used after selection, once, if at all.
* MMLU is a retention input, not an accuracy objective. The selection rule
  rejects a coefficient whose calibration MMLU drop exceeds the registered
  1.0-point bound; if MMLU receipts are absent the selection is labelled
  provisional and is not a score.
* The checkpoint's `arm_state` holds the trainable text-tower parameters (the
  tied embedding is frozen in both endpoints and is identical), so the
  interpolation is exact for the deployed active union.

CPU-testable interpolation math is in `interpolate_state_dict`; the model and
eval imports happen only inside `evaluate`.

Usage (queued, GPU):
    SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 scripts/s60_interpolate.py \
        --checkpoint results/stage2/runs/s58-stage2a/ckpt-004000.pt \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --grid 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
        --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined.jsonl \
        --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
        --calibration-limit 300 --evaluate \
        --out-dir results/stage2/s60-interpolation \
        --out results/stage2/s60-interpolation/receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import rft  # noqa: E402

SCHEMA = "selvedge/s60-interpolation/v1"


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


def parse_grid(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise SystemExit("s60-interpolate: empty grid")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise SystemExit("s60-interpolate: grid coefficients must be in [0, 1]")
    return sorted(set(values))


def interpolate_state_dict(base: dict, candidate: dict,
                           coefficient: float) -> dict:
    """Linear interpolation of two state dicts over the same keys.

    `coefficient` 0.0 returns the base values and 1.0 the candidate values
    exactly. A key present in only one dict is a hard error: a partial
    interpolation would silently mix parameter identities.
    """
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("coefficient must be in [0, 1]")
    if set(base) != set(candidate):
        missing = sorted(set(base) - set(candidate))[:5]
        extra = sorted(set(candidate) - set(base))[:5]
        raise ValueError(f"state dict keys differ: missing {missing}, extra {extra}")
    mixed = {}
    for key, base_value in base.items():
        candidate_value = candidate[key]
        if tuple(base_value.shape) != tuple(candidate_value.shape):
            raise ValueError(
                f"{key}: shape {tuple(base_value.shape)} != "
                f"{tuple(candidate_value.shape)}")
        if not hasattr(base_value, "dtype"):
            mixed[key] = (base_value if coefficient < 0.5 else candidate_value)
            continue
        if coefficient == 0.0:
            mixed[key] = base_value.clone()
        elif coefficient == 1.0:
            mixed[key] = candidate_value.clone()
        else:
            mixed[key] = (base_value.float() * (1.0 - coefficient)
                          + candidate_value.float() * coefficient).to(
                              base_value.dtype)
    return mixed


def _base_arm_state(model, checkpoint_state: dict) -> dict:
    """The checkpoint's keys read from the untrained post-trained model.

    The full arm exports only the trainable parameters, keyed relative to the
    text model. Reading the same keys from the freshly loaded model gives the
    untrained endpoint on exactly the same parameter set.
    """
    parameters = dict(model.named_parameters())
    state = {}
    missing = []
    for name in checkpoint_state:
        if name not in parameters:
            missing.append(name)
            continue
        state[name] = parameters[name].detach().to("cpu")
    if missing:
        raise SystemExit(
            f"s60-interpolate: checkpoint keys absent from the model: {missing[:5]}")
    return state


def build_calibration(dataset: str, exclude_ids: set[str], limit: int,
                      seed: int) -> list[dict]:
    """A labelled calibration split drawn from the train split.

    The question is the train item's own prompt text (the item stores it with
    its `Question:`/`Problem:` framing), and the gold is the train label. No
    test item is read here.
    """
    source = "gsm8k-train" if dataset == "gsm8k" else "math-train"
    items = rft.load_items(source)
    items = [item for item in items if item["id"] not in exclude_ids]
    rng = random.Random(seed)
    rng.shuffle(items)
    items = sorted(items[:limit], key=lambda item: item["id"])
    calibration = []
    for item in items:
        prompt = item["prompt"]
        if dataset == "gsm8k":
            question = prompt[len("Question: "):-len("\nAnswer:")]
            task = "gsm8k"
        else:
            question = prompt[len("Problem:\n"):-len("\n\nSolution:")]
            task = "math500"
        calibration.append({
            "problem_id": item["id"],
            "task": task,
            "question": question,
            "gold": item["target"],
            "subject": item.get("subject"),
            "level": item.get("difficulty"),
        })
    return calibration


def _load_exclusions(paths: list[str]) -> set[str]:
    excluded = set()
    for path in paths:
        with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("id"):
                    excluded.add(row["id"])
    return excluded


def choose_coefficient(rows: list[dict], retention_limit: float = 0.01) -> dict:
    """Pick a coefficient on calibration only.

    `rows` entries carry `coefficient`, `gsm8k_cal`, `math_cal` and optionally
    `mmlu_cal`. The score is the mean gain over the 0.0 endpoint on the two
    calibration tasks. A row whose calibration MMLU drop exceeds
    `retention_limit` is ineligible when MMLU is present. Ties go to the smaller
    coefficient, i.e. the smaller departure from the post-trained model.

    No test-set number is an input to this function.
    """
    base = next((row for row in rows if row["coefficient"] == 0.0), None)
    if base is None:
        raise ValueError("the grid must contain coefficient 0.0")
    has_mmlu = all(isinstance(row.get("mmlu_cal"), (int, float)) for row in rows)
    scored = []
    for row in rows:
        if row["coefficient"] == 0.0:
            continue
        gain = 0.5 * ((row["gsm8k_cal"] - base["gsm8k_cal"])
                      + (row["math_cal"] - base["math_cal"]))
        eligible = True
        if has_mmlu and row["mmlu_cal"] < base["mmlu_cal"] - retention_limit:
            eligible = False
        scored.append((eligible, gain, row["coefficient"]))
    eligible_rows = [item for item in scored if item[0]]
    if not eligible_rows:
        return {"coefficient": 0.0, "reason": "no coefficient met the retention bound",
                "eligible": False, "basis": "calibration only",
                "mmlu_present": has_mmlu}
    best = max(eligible_rows, key=lambda item: (item[1], -item[2]))
    return {
        "coefficient": best[2],
        "calibration_gain": best[1],
        "eligible": True,
        "basis": "calibration only; ties choose the smaller coefficient",
        "mmlu_present": has_mmlu,
        "retention_limit": retention_limit if has_mmlu else None,
    }


def evaluate(model, tokenizer, items: list[dict], max_new_tokens: int,
             batch_size: int, device: str) -> dict:
    """Greedy accuracy on one calibration split, via the S54 instrument."""
    import s54_post_trained_eval as harness

    result = harness.evaluate_task(
        model, tokenizer, items, items[0]["task"], "non-thinking", "greedy",
        max_new_tokens, device, None, batch_size)
    return {"accuracy": result.get("flexible", {}).get("accuracy"),
            "accuracy_primary": result.get("accuracy", {}).get("accuracy"),
            "raw": {key: value for key, value in result.items()
                    if key != "per_problem"}}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="the S58 best checkpoint (full-arm state)")
    parser.add_argument("--model-dir", required=True,
                        help="the post-trained 0.8B snapshot (the 0.0 endpoint)")
    parser.add_argument("--grid", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0")
    parser.add_argument("--exclude-ids", action="append", default=[],
                        help="JSONL whose `id` fields are removed from calibration")
    parser.add_argument("--calibration-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate calibration accuracies; without it only "
                             "the interpolated checkpoints are written")
    parser.add_argument("--mmlu-receipts", action="append", default=[],
                        help="optional MMLU receipts per coefficient; the "
                             "selection is provisional without them")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(ROOT, args.checkpoint)
    payload = torch_load(checkpoint_path)
    if payload.get("arm") != "full":
        raise SystemExit(
            f"s60-interpolate: {args.checkpoint} is not a full-arm checkpoint")
    candidate_state = payload["arm_state"]
    grid = parse_grid(args.grid)

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False).to(device)
    model.eval()
    base_state = _base_arm_state(model, candidate_state)

    rows = []
    for coefficient in grid:
        mixed = interpolate_state_dict(base_state, candidate_state, coefficient)
        path = os.path.join(out_dir, f"interp-{coefficient:.3f}-arm.pt")
        torch_save({
            "schema": "selvedge/stage2-ckpt/v1",
            "interpolation": {
                "coefficient": coefficient,
                "base": "post-trained 0.8B (untrained endpoint)",
                "candidate": args.checkpoint,
                "candidate_sha256": sha256_file(checkpoint_path),
            },
            "step": payload.get("step"),
            "config_sha256": payload.get("config_sha256"),
            "arm": "full",
            "arm_state": mixed,
        }, path)
        rows.append({"coefficient": coefficient,
                     "checkpoint": os.path.relpath(path, ROOT),
                     "sha256": sha256_file(path)})

    receipt = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "grid": grid,
        "checkpoints": rows,
        "source_checkpoint": {"path": args.checkpoint,
                              "sha256": sha256_file(checkpoint_path)},
        "model_dir": args.model_dir,
        "calibration": {
            "source": "train split only; the test splits are never read",
            "gsm8k_items": args.calibration_limit,
            "math_items": args.calibration_limit,
            "math500_test_used": False,
            "gsm8k_test_used": False,
            "excluded_files": args.exclude_ids,
        },
        "selection": None,
    }
    if not args.evaluate:
        write_json(os.path.join(ROOT, args.out), receipt)
        print(f"s60-interpolate: wrote {len(rows)} interpolated checkpoints -> "
              f"{args.out}")
        return 0

    tokenizer = _tokenizer(args.model_dir)
    excluded = _load_exclusions(args.exclude_ids)
    gsm_cal = build_calibration("gsm8k", excluded, args.calibration_limit,
                                args.seed)
    math_cal = build_calibration("math", excluded, args.calibration_limit,
                                 args.seed + 1)

    evaluated = []
    for coefficient in grid:
        mixed = interpolate_state_dict(base_state, candidate_state, coefficient)
        _load_into(model, mixed)
        gsm = evaluate(model, tokenizer, gsm_cal, args.max_new_tokens,
                       args.batch_size, device)
        math_result = evaluate(model, tokenizer, math_cal, args.max_new_tokens,
                               args.batch_size, device)
        evaluated.append({
            "coefficient": coefficient,
            "gsm8k_cal": gsm["accuracy"],
            "math_cal": math_result["accuracy"],
            "gsm8k_raw": gsm["raw"].get("summary"),
            "math_raw": math_result["raw"].get("summary"),
        })
        print(f"s60-interpolate: t={coefficient:.3f} gsm8k-cal="
              f"{gsm['accuracy']} math-cal={math_result['accuracy']}", flush=True)

    mmlu = _load_mmlu(args.mmlu_receipts)
    for row in evaluated:
        if row["coefficient"] in mmlu:
            row["mmlu_cal"] = mmlu[row["coefficient"]]
    selection = choose_coefficient(evaluated)
    if not mmlu:
        selection["provisional"] = True
        selection["why_provisional"] = (
            "no MMLU calibration receipts were supplied; the retention bound "
            "was not applied, so this is not a final selection")
    receipt["evaluation"] = evaluated
    receipt["selection"] = selection
    receipt["command"] = sys.argv
    write_json(os.path.join(ROOT, args.out), receipt)
    print(f"s60-interpolate: selected t={selection['coefficient']} "
          f"(provisional={selection.get('provisional', False)}) -> {args.out}")
    return 0


# -- lazy torch helpers (the CPU test imports the math without torch weights) --


def torch_load(path: str):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def torch_save(payload, path: str) -> None:
    import torch
    torch.save(payload, path)


def _tokenizer(model_dir: str):
    import s54_post_trained_eval as harness
    return harness.load_tokenizer(model_dir)


def _load_into(model, state: dict) -> None:
    import torch
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device,
                                             dtype=parameters[name].dtype))


def _load_mmlu(paths: list[str]) -> dict:
    """Map coefficient -> MMLU primary accuracy from harness receipts.

    The receipt may declare `interpolation_coefficient` directly, or name the
    interpolated checkpoint (`interp-<t>-arm.pt`) in its checkpoint fields.
    """
    import re

    values = {}
    for path in paths:
        payload = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
        coefficient = payload.get("interpolation_coefficient")
        if coefficient is None:
            text = json.dumps(payload)
            match = re.search(r"interp-([0-9.]+)-arm", text)
            if match:
                coefficient = float(match.group(1).rstrip("."))
        for entry in payload.get("tasks") or []:
            if entry.get("task") != "mmlu":
                continue
            metrics = entry.get("all_metrics") or {}
            primary = entry.get("primary_metric") or "acc,none"
            if coefficient is None:
                raise SystemExit(
                    f"s60-interpolate: {path} cannot be tied to a coefficient")
            if isinstance(metrics.get(primary), (int, float)):
                values[float(coefficient)] = float(metrics[primary])
    return values


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
