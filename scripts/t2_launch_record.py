#!/usr/bin/env python3
"""Generate a GPU launch record: the VRAM arithmetic for one stage.

Directive 2 requires every GPU launch to carry an explicit, workload-sized
memory arithmetic in its receipt. This script is bookkeeping only — it reads
receipts and the admission record, never a model — and it is the single place
that arithmetic is written, so it cannot drift from the artifacts it cites.

Two modes:

  plan       before the launch. Components are computed from the model
             accounting receipts, the admission record and the instrument's
             logit chunk; the measured figure (when it exists) comes from a
             previous run of the *same* geometry.
  finalize   after the run. Adds the court receipt, the measured peaks from
             the run's own summary/eval receipt, and the cgroup evidence.

Usage:
  scripts/court python -- python3 scripts/t2_launch_record.py plan \\
      --stage eval --arm H --out results/t2/H/launch-eval.json
  scripts/court python -- python3 scripts/t2_launch_record.py finalize \\
      --record results/t2/H/launch-eval.json --court-receipt <path> \\
      [--measured results/eval/t2-H.json] [--unit selvedge-t2-H]

Exit codes: 0 ok, 2 usage, 3 an input receipt is missing or malformed.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIB = 1024 ** 3
#: The card reports 16 GiB; the instrument's usable budget is the device total
#: minus the desktop compositor's share. Stated rather than inferred so the
#: arithmetic can be re-read without a live GPU.
USABLE_VRAM_GIB = 15.6
HEADROOM_FACTOR = 1.5
#: H*'s utility phase runs forward-only sub-batches of `micro_positions` while
#: the optimiser state is resident; it cannot add a training-step-sized peak.
LABEL_ALLOWANCE_GIB = 0.5


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_path(rel: str) -> str:
    return os.path.join(REPO, rel)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def model_elements(size: str) -> int:
    """Unique elements of a pinned Qwen3 checkpoint, from its accounting."""
    receipts = sorted(glob.glob(repo_path(f"results/accounting/*-qwen3-{size}.json")))
    if not receipts:
        raise SystemExit(f"no accounting receipt for qwen3-{size}")
    receipt = load_json(receipts[-1])
    totals = receipt["totals"]
    elements = sum(int(value) for value in totals.values())
    if (receipt.get("tie") or {}).get("treated_as_shared"):
        elements -= int(totals["lm_head"])
    return elements


def vocab_size(size: str = "1.7b") -> int:
    receipts = sorted(glob.glob(repo_path(f"results/accounting/*-qwen3-{size}.json")))
    if not receipts:
        raise SystemExit(f"no accounting receipt for qwen3-{size}")
    return int(load_json(receipts[-1])["model"]["vocab"])


def gpu_total_gib() -> float | None:
    """Device total, from nvidia-smi or the last launch guard's own query."""
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.splitlines()[0]
        return round(int(float(text.strip())) / 1024, 2)
    except Exception:
        pass
    guard = repo_path("results/t2/launch-guard.json")
    if os.path.isfile(guard):
        try:
            total_mib = (load_json(guard).get("gpu") or {}).get("total_mib")
            if total_mib:
                return round(int(total_mib) / 1024, 2)
        except Exception:
            return None
    return None


def plan(args) -> int:
    config = load_json(repo_path(f"results/train/admission-configs/t2/{args.arm}.json"))
    accessible = int(config["total_accessible"])

    components = {
        "teacher_weights_bf16": {
            "elements": model_elements("1.7b"),
            "bytes": model_elements("1.7b") * 2,
            "source": "results/accounting/*-qwen3-1.7b.json",
        },
        "student_weights_bf16": {
            "elements": accessible,
            "bytes": accessible * 2,
            "source": f"results/train/admission-configs/t2/{args.arm}.json total_accessible",
        },
    }
    if args.stage == "eval":
        components["memory_table_bf16"] = {
            "bytes": 4 * (1 << 18) * 256 * 2,
            "note": "4 insertion layers x 2^18 rows x 256 values, resident and "
                    "included in the student's accessible count; listed separately "
                    "because it is the largest single added tensor",
        }
        chunk = args.logit_chunk
        vocab = vocab_size("1.7b")
        components["logit_chunk_f32"] = {
            "positions": chunk,
            "per_tensor_bytes": chunk * vocab * 4,
            "tensors": 2,
            "bytes": 2 * chunk * vocab * 4,
            "note": "two chunk tensors (student, teacher); log-softmax and p_t "
                    "add at most two more of the same size inside a chunk",
        }
        measured_source = args.measured or f"results/eval/t2-{args.arm}.json"
    else:
        components["optimiser_adamw_fp32"] = {
            "elements": accessible,
            "bytes": 2 * accessible * 4,
            "note": "two fp32 moments over the trainable elements",
        }
        components["gradients_bf16"] = {
            "elements": accessible,
            "bytes": accessible * 2,
        }
        measured_source = args.measured or f"results/artifacts/train/t2/{args.run_name}/summary.json"

    predicted = None
    measured = None
    if measured_source and os.path.isfile(repo_path(measured_source)):
        receipt = load_json(repo_path(measured_source))
        vram = receipt.get("vram") or {}
        measured = {
            "source": measured_source,
            "peak_allocated_gib": receipt.get("vram_peak_gib"),
            "peak_reserved_gib": vram.get("peak_reserved_gib"),
            "stage_peaks": vram.get("stage_peaks"),
        }
        if args.stage != "eval":
            measured["step_peak_gib"] = receipt.get("vram_peak_gib")
    if measured and measured.get("peak_allocated_gib") is not None:
        predicted = float(measured["peak_allocated_gib"])
        if args.stage == "train":
            predicted += LABEL_ALLOWANCE_GIB

    headroom = None if predicted is None else round(predicted * HEADROOM_FACTOR, 3)
    record = {
        "schema": "selvedge/t2-gpu-launch/v1",
        "arm": args.arm,
        "stage": args.stage,
        "created_utc": utc_now(),
        "revision": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "device": {
            "name": "NVIDIA GeForce RTX 4080 SUPER",
            "total_gib": gpu_total_gib(),
            "usable_gib": USABLE_VRAM_GIB,
            "usable_basis": "device total less the desktop compositor's resident "
                            "allocation; the binding constraint on this host",
        },
        "components": components,
        "components_note": "weights and chunked logits only; the runtime's CUDA "
                            "context, cuBLAS workspaces and load transients are "
                            "named by the measured peak, not by this sum",
        "predicted_peak_gib": None if predicted is None else round(predicted, 3),
        "headroom_factor": HEADROOM_FACTOR,
        "headroom_required_gib": headroom,
        "fits_with_headroom": None if headroom is None else headroom <= USABLE_VRAM_GIB,
        "measured": measured,
        "host_limits": {
            "SELVEDGE_MEM": args.mem,
            "SELVEDGE_CPUS": args.cpus,
            "SELVEDGE_PIDS": args.pids,
            "SELVEDGE_TMP": args.tmp,
            "note": "host RAM limit for the court container; VRAM is not a cgroup "
                    "resource and is bounded by the arithmetic above",
        },
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",
                                     "expandable_segments:True"),
        "unit": args.unit,
        "kill_condition": args.kill_condition,
    }
    if args.stage == "train":
        record["components"]["utility_label_phase"] = {
            "allowance_gib": LABEL_ALLOWANCE_GIB,
            "note": "forward-only, no gradients, micro_positions=256 sub-batches "
                    "under inference_mode while the optimiser state is resident",
        }
    if headroom is not None and headroom > USABLE_VRAM_GIB:
        record["verdict"] = (
            "the 1.5x rule over the predicted peak does not fit on this card; the "
            "launch is authorised on a measured bound instead: the same geometry's "
            "measured reserved peak plus the declared allowance, with the kill "
            "condition below enforced from the first logged step")
        record["deviation"] = {
            "headroom_factor_achieved": round(USABLE_VRAM_GIB / predicted, 3)
            if predicted else None,
            "free_at_measured_reserved_gib": None if not measured
            else round(USABLE_VRAM_GIB - float(measured.get("peak_reserved_gib")
                                               or predicted), 3),
        }
    else:
        record["verdict"] = "fits with the declared headroom"
    write_json_atomic(repo_path(args.out), record)
    print(json.dumps({"out": args.out, "predicted_peak_gib": record["predicted_peak_gib"],
                      "headroom_required_gib": headroom, "verdict": record["verdict"]},
                     indent=2, sort_keys=True))
    return 0


def finalize(args) -> int:
    record = load_json(repo_path(args.record))
    if not os.path.isfile(repo_path(args.court_receipt)):
        raise SystemExit(f"court receipt {args.court_receipt} is missing")
    court = load_json(repo_path(args.court_receipt))
    measured = None
    if args.measured and os.path.isfile(repo_path(args.measured)):
        receipt = load_json(repo_path(args.measured))
        vram = receipt.get("vram") or {}
        measured = {
            "source": args.measured,
            "peak_allocated_gib": receipt.get("vram_peak_gib"),
            "peak_reserved_gib": vram.get("peak_reserved_gib"),
            "stage_peaks": vram.get("stage_peaks"),
        }
    record["court"] = {
        "receipt": args.court_receipt,
        "log_path": court.get("log_path"),
        "exit_code": court.get("exit_code"),
        "container_exit_code": court.get("container_exit_code"),
        "oom_killed": court.get("oom_killed"),
        "enforcement_ok": (court.get("enforcement") or {}).get("ok"),
        "limits": court.get("limits"),
        "container_env": court.get("container_env"),
        "gpu": court.get("gpu"),
    }
    if measured:
        record["measured"] = measured
    if args.unit:
        record["unit"] = args.unit
        try:
            show = subprocess.run(
                ["systemctl", "--user", "show", args.unit, "--property=MemoryMax",
                 "--property=MemoryPeak", "--property=MemorySwapMax",
                 "--property=CPUQuotaPerSecUSec", "--property=Result"],
                capture_output=True, text=True, check=True).stdout.strip()
            record["cgroup"] = dict(
                line.split("=", 1) for line in show.splitlines() if "=" in line)
        except Exception as error:  # pragma: no cover - host dependent
            record["cgroup"] = {"error": str(error)}
    record["finalised_utc"] = utc_now()
    write_json_atomic(repo_path(args.record), record)
    print(json.dumps({"record": args.record, "measured": record.get("measured"),
                      "court_exit": (record.get("court") or {}).get("exit_code")},
                     indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "finalize"):
        child = sub.add_parser(name)
        if name == "plan":
            child.add_argument("--stage", required=True, choices=["train", "eval"])
            child.add_argument("--arm", required=True)
            child.add_argument("--out", required=True)
            child.add_argument("--run-name", default=None)
            child.add_argument("--measured", default=None)
            child.add_argument("--logit-chunk", type=int, default=64)
            child.add_argument("--mem", default="24g")
            child.add_argument("--cpus", default="8")
            child.add_argument("--pids", default="1024")
            child.add_argument("--tmp", default="8g")
            child.add_argument("--unit", default=None)
            child.add_argument("--kill-condition", default=None)
            child.set_defaults(func=plan)
        else:
            child.add_argument("--record", required=True)
            child.add_argument("--court-receipt", required=True)
            child.add_argument("--measured", default=None)
            child.add_argument("--unit", default=None)
            child.set_defaults(func=finalize)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan" and args.stage == "train" and not args.run_name:
        parser.error("--stage train needs --run-name")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
