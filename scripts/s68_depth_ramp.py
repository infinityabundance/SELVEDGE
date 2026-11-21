#!/usr/bin/env python3
"""S68: depth-ramped weight interpolation on the seed-delta soup.

The registered experiment (`docs/s68-depth-ramp-preregistration-v1.json`) asks
whether letting the interpolation coefficient vary with depth beats the
constant coefficient that S67 selected. The ramp is

    alpha(l) = alpha_shallow + (alpha_deep - alpha_shallow) * l / (L - 1)

over the arm's own `layers.<l>.` keys, with `L = 24` read from the model
config. The parameter update is the S67 soup delta:

    theta_ramp(key) = theta_0 + alpha(l) * delta_bar
    delta_bar       = mean_s (theta_s - theta_0)

`norm.weight` has no layer index and is mixed at `alpha_deep`; the receipt
lists it. The tied embedding/output projection is frozen and inherited.

Subcommands
-----------
``build``      (CPU) write the nine ramp checkpoints and their receipt.
``calibrate``  (GPU) score GSM8K-cal and MATH-cal for every ramp arm.
``mmlu``       (GPU) score the 2,000-item calibration MMLU per arm.
``select``     (CPU) apply the registered rule and the adoption test.

No subcommand reads a test split. Test scoring happens only after ``select``
reports `adopted: true`, and is driven by the existing S54/S43 instruments.

Usage (CPU, oracle court):
    SELVEDGE_MEM=24g scripts/court oracle -- \
      python3 scripts/s68_depth_ramp.py build \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --seed results/stage2/runs/s58-stage2a/ckpt-004000.pt \
        --seed results/stage2/runs/s66-stage2a-seed1-guard20/ckpt-004000.pt \
        --seed results/stage2/runs/s66-stage2a-seed2-guard20/ckpt-004000.pt \
        --out-dir results/stage2/s68-ramp \
        --out results/stage2/s68-ramp/build.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from s60_interpolate import (  # noqa: E402
    _base_arm_state, _load_into, _tokenizer, sha256_file, torch_load,
    torch_save,
)

SCHEMA = "selvedge/s68-depth-ramp/v1"
PREREG = "docs/s68-depth-ramp-preregistration-v1.json"
REGISTRATION = "docs/s66-calibration-splits-v1.json"
BASE_CALIBRATION = "results/stage2/s67-soup/calibration.json"
BASE_CHECKPOINT = "results/stage2/s67-soup/interp-0.000-arm.pt"
CONTROL_CHECKPOINT = "results/stage2/s67-soup/interp-0.375-arm.pt"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
RETENTION_LIMIT = 0.01


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
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> dict:
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def load_jsonl(path: str) -> list[dict]:
    with open(os.path.join(ROOT, path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_tag(shallow: float, deep: float) -> str:
    return f"ramp-s{shallow:.3f}-d{deep:.3f}"


def alpha_at(layer: int, shallow: float, deep: float, layers: int) -> float:
    if layers < 2:
        raise ValueError("the ramp needs at least two layers")
    return shallow + (deep - shallow) * (layer / (layers - 1))


# ---------------------------------------------------------------------------
# build


def build(args) -> int:
    prereg = read_json(args.prereg)
    grid = prereg["grid"]
    shallow_values = sorted({float(value) for value in grid["shallow"]})
    deep_values = sorted({float(value) for value in grid["deep"]})
    if not shallow_values or not deep_values:
        raise SystemExit("s68-ramp: the registered grid is empty")
    if args.diagnostics:
        diagnostics = read_json(args.diagnostics)
        if not (diagnostics.get("decision") or {}).get(
                "directions_compatible"):
            raise SystemExit(
                "s68-ramp: the registered seed-delta diagnostics are not "
                "green; the registered fallback is the seed-0 checkpoint")

    for path, expected in (
            (BASE_CHECKPOINT, prereg["grid"]["base"]["sha256"]),
            (CONTROL_CHECKPOINT, prereg["grid"]["control"]["sha256"])):
        actual = sha256_file(os.path.join(ROOT, path))
        if actual != expected:
            raise SystemExit(
                f"s68-ramp: {path} sha256 {actual} != registered {expected}")

    import torch
    from transformers import AutoModelForCausalLM

    checkpoints = [torch_load(os.path.join(ROOT, path)) for path in args.seed]
    for path, payload in zip(args.seed, checkpoints):
        if payload.get("arm") != "full":
            raise SystemExit(f"s68-ramp: {path} is not a full arm")
    keys = sorted(checkpoints[0]["arm_state"])
    for path, payload in zip(args.seed[1:], checkpoints[1:]):
        if sorted(payload["arm_state"]) != keys:
            raise SystemExit(f"s68-ramp: {path} key set differs")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False)
    text_config = getattr(model.config, "text_config", model.config)
    layers = int(getattr(text_config, "num_hidden_layers"))
    if layers != int(prereg["model"]["layers"]):
        raise SystemExit(
            f"s68-ramp: model has {layers} layers, the registration says "
            f"{prereg['model']['layers']}")
    base_state = _base_arm_state(model, checkpoints[0]["arm_state"])

    layer_of: dict[str, int | None] = {}
    for key in keys:
        match = LAYER_RE.match(key)
        layer_of[key] = int(match.group(1)) if match else None
    non_layer = sorted(key for key, value in layer_of.items() if value is None)

    delta_bar = {}
    for key in keys:
        base = base_state[key]
        if not hasattr(base, "dtype") or not torch.is_floating_point(base):
            delta_bar[key] = None
            continue
        total = torch.zeros_like(base, dtype=torch.float32)
        for payload in checkpoints:
            total += payload["arm_state"][key].float() - base.float()
        delta_bar[key] = total / len(checkpoints)
        del total

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [{"checkpoint": path,
              "checkpoint_sha256": sha256_file(os.path.join(ROOT, path))}
             for path in args.seed]
    arms = []
    for shallow in shallow_values:
        for deep in deep_values:
            tag = arm_tag(shallow, deep)
            per_layer = [alpha_at(layer, shallow, deep, layers)
                         for layer in range(layers)]
            mixed = {}
            for key in keys:
                base = base_state[key]
                delta = delta_bar[key]
                layer = layer_of[key]
                alpha = deep if layer is None else per_layer[layer]
                if delta is None or alpha == 0.0:
                    mixed[key] = base.clone()
                else:
                    mixed[key] = (base.float() + alpha * delta).to(base.dtype)
            payload = {
                "schema": "selvedge/stage2-ckpt/v1",
                "interpolation": {
                    "kind": "depth-ramp",
                    "coefficient": round(0.5 * (shallow + deep), 6),
                    "alpha_shallow": shallow,
                    "alpha_deep": deep,
                    "layers": layers,
                    "formula": prereg["formula"]["definition"],
                    "non_layer_alpha": "alpha_deep",
                    "non_layer_keys": non_layer,
                    "base": "post-trained 0.8B (untrained endpoint)",
                    "candidate": "mean of the three S66 seed deltas",
                    "seeds": seeds,
                },
                "step": checkpoints[0].get("step"),
                "config_sha256": checkpoints[0].get("config_sha256"),
                "arm": "full",
                "arm_state": mixed,
            }
            path = os.path.join(out_dir, f"{tag}-arm.pt")
            torch_save(payload, path)
            record = {
                "tag": tag,
                "alpha_shallow": shallow,
                "alpha_deep": deep,
                "mean_coefficient": round(0.5 * (shallow + deep), 6),
                "alpha_by_layer": per_layer,
                "path": os.path.relpath(path, ROOT),
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
            arms.append(record)
            print(f"s68-ramp: {tag} -> {record['path']} "
                  f"sha256={record['sha256']}", flush=True)
            del payload, mixed

    payload = {
        "schema": SCHEMA,
        "kind": "depth-ramp-build",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": args.prereg,
            "sha256": sha256_file(os.path.join(ROOT, args.prereg)),
        },
        "formula": prereg["formula"],
        "layers": layers,
        "non_layer_keys": non_layer,
        "model_dir": args.model_dir,
        "base": {"checkpoint": BASE_CHECKPOINT,
                 "sha256": prereg["grid"]["base"]["sha256"]},
        "control": {"checkpoint": CONTROL_CHECKPOINT,
                    "sha256": prereg["grid"]["control"]["sha256"]},
        "seeds": seeds,
        "diagnostics": args.diagnostics,
        "grid": {"shallow": shallow_values, "deep": deep_values},
        "arms": arms,
        "state_elements": sum(
            int(value.numel()) for value in delta_bar.values()
            if value is not None),
        "frozen_elements_inherited": (
            "the tied embedding/output projection is not mixed; it is "
            "inherited from the post-trained snapshot"),
        "claims": {"reads_no_test_split": True, "selection_performed": False},
    }
    write_json(args.out, payload)
    print(f"s68-ramp: wrote {len(arms)} ramp arms -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# calibration


def evaluate(model, tokenizer, items, task, max_new_tokens, batch_size,
             device) -> dict:
    import s54_post_trained_eval as harness

    return harness.evaluate_task(
        model, tokenizer, items, task, "non-thinking", "greedy",
        max_new_tokens, device, None, batch_size)


def base_rows() -> tuple[dict, dict]:
    calibration = read_json(BASE_CALIBRATION)
    rows = calibration.get("rows") or []
    base = next((row for row in rows if float(row.get("coefficient", -1)) == 0.0),
                None)
    control = next((row for row in rows
                    if float(row.get("coefficient", -1)) == 0.375), None)
    if base is None or control is None:
        raise SystemExit("s68-ramp: the S67 calibration has no 0.0/0.375 row")
    return base, control


def calibrate(args) -> int:
    build_receipt = read_json(args.build)
    registration = read_json(args.registration)
    gsm = load_jsonl(registration["splits"]["gsm8k_cal"]["path"])
    math = load_jsonl(registration["splits"]["math_cal"]["path"])
    for row in gsm:
        row["task"] = "gsm8k"
    for row in math:
        row["task"] = "math500"
    if len(gsm) != 1000 or len(math) != 1000:
        raise SystemExit(
            f"s68-ramp: expected 1,000 calibration items, got "
            f"{len(gsm)}/{len(math)}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(
        ROOT, args.out)
    partial_path = out_path + ".partial"
    rows: list[dict] = []
    done: set[str] = set()
    if args.resume and os.path.exists(partial_path):
        partial = json.load(open(partial_path, encoding="utf-8"))
        rows = list(partial.get("rows") or [])
        done = {row["tag"] for row in rows}
        print(f"s68-ramp: resumed {len(done)} arm(s)", flush=True)

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=False).to(device)
    model.eval()
    tokenizer = _tokenizer(args.model_dir)

    for arm in build_receipt["arms"]:
        if arm["tag"] in done:
            continue
        checkpoint = os.path.join(ROOT, arm["path"])
        payload = torch_load(checkpoint)
        if payload.get("arm") != "full":
            raise SystemExit(f"s68-ramp: {arm['path']} is not a full arm")
        _load_into(model, payload["arm_state"])
        gsm_result = evaluate(model, tokenizer, gsm, "gsm8k",
                              args.max_new_tokens, args.batch_size, device)
        math_result = evaluate(model, tokenizer, math, "math500",
                               args.max_new_tokens, args.batch_size, device)
        row = {
            "tag": arm["tag"],
            "alpha_shallow": arm["alpha_shallow"],
            "alpha_deep": arm["alpha_deep"],
            "mean_coefficient": arm["mean_coefficient"],
            "checkpoint": arm["path"],
            "checkpoint_sha256": arm["sha256"],
            "gsm8k_cal": gsm_result["flexible"]["accuracy"],
            "math_cal": math_result["flexible"]["accuracy"],
            "gsm8k_items": len(gsm),
            "math_items": len(math),
            "gsm8k_raw": gsm_result.get("summary"),
            "math_raw": math_result.get("summary"),
        }
        rows.append(row)
        print(f"s68-ramp: {row['tag']} gsm8k-cal={row['gsm8k_cal']:.4f} "
              f"math-cal={row['math_cal']:.4f}", flush=True)
        write_json(partial_path, {
            "schema": SCHEMA,
            "kind": "depth-ramp-calibration-partial",
            "created_utc": utcnow(),
            "build_receipt": {"path": args.build,
                              "sha256": sha256_file(
                                  os.path.join(ROOT, args.build))},
            "rows": rows,
        })
        del payload

    base, control = base_rows()
    write_json(out_path, {
        "schema": SCHEMA,
        "kind": "depth-ramp-calibration",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "build_receipt": {
            "path": args.build,
            "sha256": sha256_file(os.path.join(ROOT, args.build)),
            "stamp": build_receipt.get("stamp"),
        },
        "instrument": {
            "tasks": "GSM8K-cal and MATH-cal (1,000 each)",
            "decoding": "greedy non-thinking",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "test_split_read": False,
            "registration": args.registration,
        },
        "base_row": {key: base.get(key) for key in (
            "coefficient", "gsm8k_cal", "math_cal", "mmlu_cal",
            "checkpoint", "checkpoint_sha256")},
        "control_row": {key: control.get(key) for key in (
            "coefficient", "gsm8k_cal", "math_cal", "mmlu_cal",
            "checkpoint", "checkpoint_sha256")},
        "rows": rows,
    })
    print(f"s68-ramp: calibration -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# calibration MMLU


def mmlu(args) -> int:
    build_receipt = read_json(args.build)
    out_dir = args.mmlu_dir if os.path.isabs(args.mmlu_dir) else os.path.join(
        ROOT, args.mmlu_dir)
    os.makedirs(out_dir, exist_ok=True)
    values = {}
    for arm in build_receipt["arms"]:
        tag = arm["tag"]
        out = os.path.join(out_dir, f"mmlu-{tag}.json")
        if os.path.exists(out):
            payload = json.load(open(out, encoding="utf-8"))
            if payload.get("complete") and payload.get("tasks"):
                values[tag] = accuracy_of(payload)
                print(f"s68-ramp: {tag} already receipted", flush=True)
                continue
        command = [
            sys.executable, os.path.join(ROOT, "scripts/s43_score_total_cap.py"),
            "--arm", "interpolated",
            "--full-checkpoint", os.path.join(ROOT, arm["path"]),
            "--student-dir", args.model_dir,
            "--model-id", MODEL_ID,
            "--model-revision", MODEL_REVISION,
            "--tasks", "mmlu",
            "--total-cap-map", "mmlu=2000",
            "--registration", args.prereg,
            "--vram-limit-gib", str(args.vram_limit_gib),
            "--out", out,
        ]
        print(f"s68-ramp: {tag} MMLU start", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(
                f"s68-ramp: {tag} MMLU failed with exit "
                f"{completed.returncode}")
        payload = json.load(open(out, encoding="utf-8"))
        values[tag] = accuracy_of(payload)
        print(f"s68-ramp: {tag} mmlu-cal={values[tag]:.4f}", flush=True)
    write_json(args.out, {
        "schema": SCHEMA,
        "kind": "depth-ramp-calibration-mmlu",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "items": 2000,
        "build_receipt": {"path": args.build,
                          "sha256": sha256_file(
                              os.path.join(ROOT, args.build))},
        "values": values,
        "receipts": {
            tag: os.path.relpath(os.path.join(out_dir, f"mmlu-{tag}.json"), ROOT)
            for tag in values},
    })
    print(f"s68-ramp: MMLU values -> {args.out}")
    return 0


def accuracy_of(payload: dict) -> float:
    for entry in payload.get("tasks") or []:
        if entry.get("task") != "mmlu":
            continue
        metrics = entry.get("all_metrics") or {}
        primary = entry.get("primary_metric") or "acc,none"
        value = metrics.get(primary)
        if isinstance(value, (int, float)):
            return float(value)
    raise SystemExit("s68-ramp: an MMLU receipt has no numeric accuracy")


# ---------------------------------------------------------------------------
# selection


def select(args) -> int:
    prereg = read_json(PREREG)
    calibration = read_json(args.calibration)
    mmlu_values = {}
    if args.mmlu:
        mmlu_values = (read_json(args.mmlu).get("values") or {})
    base = calibration["base_row"]
    control = calibration["control_row"]
    base_gsm = float(base["gsm8k_cal"])
    base_math = float(base["math_cal"])
    base_mmlu = float(base["mmlu_cal"])
    base_mean = 0.5 * (base_gsm + base_math)
    control_mean = 0.5 * (float(control["gsm8k_cal"])
                          + float(control["math_cal"]))

    scored = []
    for row in calibration["rows"]:
        tag = row["tag"]
        mmlu_cal = mmlu_values.get(tag)
        gain = 0.5 * ((float(row["gsm8k_cal"]) - base_gsm)
                      + (float(row["math_cal"]) - base_math))
        eligible = (mmlu_cal is not None
                    and float(mmlu_cal) >= base_mmlu - RETENTION_LIMIT)
        scored.append({
            **row,
            "mmlu_cal": mmlu_cal,
            "calibration_mean": 0.5 * (float(row["gsm8k_cal"])
                                       + float(row["math_cal"])),
            "gain_vs_alpha0": gain,
            "eligible": eligible,
        })
    eligible = [row for row in scored if row["eligible"]]
    if not eligible:
        raise SystemExit("s68-ramp: no ramp arm passed the MMLU filter")
    eligible.sort(key=lambda row: (-row["gain_vs_alpha0"],
                                   row["mean_coefficient"],
                                   row["alpha_deep"], row["tag"]))
    best = eligible[0]
    adopted = best["calibration_mean"] > control_mean
    payload = {
        "schema": SCHEMA,
        "kind": "depth-ramp-selection",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "preregistration": {
            "path": PREREG,
            "sha256": sha256_file(os.path.join(ROOT, PREREG)),
        },
        "calibration": {
            "path": args.calibration,
            "sha256": sha256_file(os.path.join(ROOT, args.calibration)),
            "stamp": calibration.get("stamp"),
        },
        "base": {
            "coefficient": base.get("coefficient"),
            "checkpoint": base.get("checkpoint"),
            "checkpoint_sha256": base.get("checkpoint_sha256"),
            "gsm8k_cal": base_gsm,
            "math_cal": base_math,
            "mmlu_cal": base_mmlu,
            "calibration_mean": base_mean,
        },
        "control": {
            "coefficient": control.get("coefficient"),
            "checkpoint": control.get("checkpoint"),
            "checkpoint_sha256": control.get("checkpoint_sha256"),
            "gsm8k_cal": control["gsm8k_cal"],
            "math_cal": control["math_cal"],
            "mmlu_cal": control.get("mmlu_cal"),
            "calibration_mean": control_mean,
        },
        "rows": scored,
        "selection": {
            "rule": prereg["selection_rule"]["objective"],
            "retention_limit": RETENTION_LIMIT,
            "tie_break": prereg["selection_rule"]["tie_break"],
            "tag": best["tag"],
            "alpha_shallow": best["alpha_shallow"],
            "alpha_deep": best["alpha_deep"],
            "calibration_mean": best["calibration_mean"],
            "gain_vs_alpha0": best["gain_vs_alpha0"],
            "gain_over_control": best["calibration_mean"] - control_mean,
            "adopted": adopted,
            "adoption_rule": prereg["selection_rule"]["adoption_test"],
            "test_scoring_authorised": bool(adopted),
        },
        "claims": {
            "test_split_read_for_selection": False,
            "selection_is_calibration_only": True,
            "constant_control_is_not_a_candidate": True,
        },
    }
    write_json(args.out, payload)
    print(f"s68-ramp: selected {best['tag']} "
          f"mean={best['calibration_mean']:.4f} "
          f"control={control_mean:.4f} adopted={adopted} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--seed", action="append", required=True)
    p.add_argument("--out-dir", default="results/stage2/s68-ramp")
    p.add_argument("--out", default="results/stage2/s68-ramp/build.json")
    p.add_argument("--prereg", default=PREREG)
    p.add_argument("--diagnostics",
                   default="results/stage2/s67-soup/delta-diagnostics.json")
    p.set_defaults(func=build)

    p = sub.add_parser("calibrate")
    p.add_argument("--build", default="results/stage2/s68-ramp/build.json")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--registration", default=REGISTRATION)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out",
                   default="results/stage2/s68-ramp/calibration.json")
    p.set_defaults(func=calibrate)

    p = sub.add_parser("mmlu")
    p.add_argument("--build", default="results/stage2/s68-ramp/build.json")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prereg", default=PREREG)
    p.add_argument("--mmlu-dir", default="results/stage2/s68-ramp/mmlu")
    p.add_argument("--vram-limit-gib", type=float, default=13.0)
    p.add_argument("--out",
                   default="results/stage2/s68-ramp/calibration-mmlu.json")
    p.set_defaults(func=mmlu)

    p = sub.add_parser("select")
    p.add_argument("--calibration",
                   default="results/stage2/s68-ramp/calibration.json")
    p.add_argument("--mmlu",
                   default="results/stage2/s68-ramp/calibration-mmlu.json")
    p.add_argument("--out",
                   default="results/stage2/s68-ramp/selection.json")
    p.set_defaults(func=select)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
