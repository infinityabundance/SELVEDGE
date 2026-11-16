#!/usr/bin/env python3
"""T2 pilot bookkeeping: launch guards, per-arm receipts, progress and ETA.

This module is orchestration only. It reads artifacts, queries `df` and
`nvidia-smi`, and writes JSON state under `results/t2/`. It never imports torch,
never loads a model and never runs a training or evaluation step; every one of
those runs through `scripts/court` (directive 1).

Subcommands:

  guard         free-disk and GPU-idle arithmetic before the pilot starts
  targets       is the shared teacher-target cache complete for the budget?
  state-init    create results/t2/pilot-state.json (idempotent)
  state-set     set one dotted field, atomically
  state-get     print one dotted field
  state-archive-failure  move `failure` into the `failures` history on resume
  arm-receipt   compose results/t2/<arm>/run.json from the run artifacts
  eval-receipt  compose results/t2/<arm>/eval.json from the eval receipt
  progress      human-readable progress, remaining work and resume command
  estimate      remaining wall-clock, as JSON

Exit codes: 0 ok, 1 not satisfied, 2 usage, 3 blocked by a guard.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_REL = "results/t2/pilot-state.json"
ARTIFACT_REL = "results/artifacts/train"
ARMS = ("D", "M", "X", "H", "H*")

# Registered geometry (docs/EXPERIMENT-CONTRACT.md T2.1/T2.2/T2.3). These are
# the constants the configs and the admission records are built from; they are
# duplicated here only to size the checkpoint arithmetic before anything runs,
# and every one of them is cross-checked against the measured smoke checkpoints
# in `guard`.
LAYERS = 28
HIDDEN = 1024
DENSE_WIDTH = 2048
SPECIALISTS = 8
SPEC_WIDTH = 512
MEM_INSERTS = 4
MEM_ROWS = 1 << 18
MEM_VALUE_DIM = 256
MEM_READS = 2

TRAINABLE = {
    "D": LAYERS * 3 * HIDDEN * DENSE_WIDTH,
    "M": MEM_INSERTS * MEM_ROWS * MEM_VALUE_DIM
         + MEM_INSERTS * (MEM_READS * MEM_VALUE_DIM * HIDDEN + HIDDEN),
    "X": LAYERS * (SPECIALISTS * 3 * SPEC_WIDTH * HIDDEN + HIDDEN * SPECIALISTS),
}
TRAINABLE["H"] = TRAINABLE["X"] + TRAINABLE["M"]
TRAINABLE["H*"] = TRAINABLE["H"]

#: bf16 memory table, fp32 everything else; AdamW keeps two fp32 moments.
MEMORY_TABLE_ELEMENTS = MEM_INSERTS * MEM_ROWS * MEM_VALUE_DIM
FP32_BYTES_PER_ELEMENT = 12
BF16_BYTES_PER_ELEMENT = 6

#: Measured smoke checkpoints (results/artifacts/train/smoke-t2/<arm>/), used
#: only to prove the analytic size is not optimistic.
SMOKE_CHECKPOINT_BYTES = {
    "D": 2_114_029_497,
    "M": 1_635_842_957,
    "X": 4_230_748_073,
    "H": 5_866_590_113,
    "H*": 5_866_590_113,
}

RECORD_BYTES = 100
#: H*'s utility checkpoint holds the per-position utilities and the cached
#: pre-expert features (28 layers x 32,768 positions x 1,024 floats).
LABEL_BYTES = (LAYERS * 32_768 * SPECIALISTS * 4
               + LAYERS * 32_768 * HIDDEN * 4)
RESERVE_BYTES = 10 * 1024 ** 3


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# checkpoint arithmetic


def arm_checkpoint_bytes(arm: str) -> int:
    """Analytic checkpoint size: parameters plus AdamW's two moments."""
    trainable = TRAINABLE[arm]
    table = MEMORY_TABLE_ELEMENTS if arm in ("M", "H", "H*") else 0
    other = trainable - table
    return FP32_BYTES_PER_ELEMENT * other + BF16_BYTES_PER_ELEMENT * table


def disk_arithmetic(budget_tokens: int) -> dict:
    targets_bytes = budget_tokens * RECORD_BYTES
    per_arm = {arm: arm_checkpoint_bytes(arm) for arm in ARMS}
    retained = 2 * sum(per_arm.values())      # the loop keeps the latest two
    in_flight = max(per_arm.values())         # a third copy exists while writing
    labels = LABEL_BYTES if "H*" in ARMS else 0
    cache_path = repo_path(os.path.join(ARTIFACT_REL, "teacher-targets.bin"))
    existing_cache = os.path.getsize(cache_path) if os.path.exists(cache_path) else 0
    # The build writes .tmp then renames, so the old cache coexists with the new
    # one; that phase runs before any checkpoint exists.
    build_peak = targets_bytes + existing_cache + RESERVE_BYTES
    training_peak = targets_bytes + retained + in_flight + labels + RESERVE_BYTES
    cross_check = {
        arm: {
            "analytic_bytes": per_arm[arm],
            "measured_smoke_bytes": SMOKE_CHECKPOINT_BYTES[arm],
            "analytic_le_measured": per_arm[arm] <= SMOKE_CHECKPOINT_BYTES[arm],
            "ratio": round(per_arm[arm] / SMOKE_CHECKPOINT_BYTES[arm], 6),
        }
        for arm in ARMS
    }
    return {
        "schema": "selvedge/t2-disk-arithmetic/v1",
        "budget_tokens": budget_tokens,
        "targets_bytes": targets_bytes,
        "existing_cache_bytes": existing_cache,
        "checkpoint_bytes": per_arm,
        "retained_two_per_arm_bytes": retained,
        "in_flight_checkpoint_bytes": in_flight,
        "utility_label_bytes": labels,
        "reserve_bytes": RESERVE_BYTES,
        "build_peak_bytes": build_peak,
        "training_peak_bytes": training_peak,
        "required_bytes": max(build_peak, training_peak),
        "checkpoint_size_cross_check": cross_check,
        "notes": [
            "the target build writes teacher-targets.bin.tmp before the atomic "
            "rename, so the old and the new cache coexist during the build",
            "the loop retains the latest two checkpoints per arm and deletes older "
            "ones, and a third copy exists while a new one is being written",
            "H*'s utility-label checkpoint holds the cached 1024-wide features for "
            "28 layers x 32,768 calibration positions as well as the labels",
        ],
    }


def free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def gpu_state() -> dict:
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as error:  # pragma: no cover - host without nvidia-smi
        return {"available": False, "error": str(error)}
    used_mib = total_mib = None
    first = used.splitlines()[0] if used else ""
    if "," in first:
        left, right = first.split(",", 1)
        used_mib, total_mib = int(left.strip()), int(right.strip())
    apps = []
    for line in processes.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "name": parts[1], "used_mib": int(parts[2])})
    compute_mib = sum(app["used_mib"] for app in apps)
    return {
        "available": True,
        "used_mib": used_mib,
        "total_mib": total_mib,
        "compute_apps": apps,
        "compute_mib": compute_mib,
    }


def cmd_guard(args) -> int:
    os.makedirs(repo_path("results/t2"), exist_ok=True)
    arithmetic = disk_arithmetic(args.budget_tokens)
    available = free_bytes(REPO)
    arithmetic["filesystem"] = REPO
    arithmetic["available_bytes"] = available
    arithmetic["fits"] = available >= arithmetic["required_bytes"]
    arithmetic["headroom_bytes"] = available - arithmetic["required_bytes"]

    gpu = gpu_state()
    allow_busy = os.environ.get("SELVEDGE_T2_ALLOW_GPU_BUSY") == "1"
    busy_threshold_mib = 3000
    gpu_ok = bool(gpu.get("available"))
    gpu_reason = ""
    if not gpu_ok:
        gpu_reason = f"nvidia-smi unavailable: {gpu.get('error', 'unknown')}"
    elif (gpu.get("used_mib") or 0) > busy_threshold_mib and not allow_busy:
        gpu_reason = (f"GPU already holds {gpu['used_mib']} MiB "
                      f"(threshold {busy_threshold_mib} MiB)")
        gpu_ok = False
    if gpu_ok and not allow_busy:
        heavy = [app for app in gpu.get("compute_apps", []) if app["used_mib"] >= 512]
        if heavy:
            gpu_ok = False
            gpu_reason = ("heavy compute processes on the GPU: "
                          + ", ".join(f"{a['name']}({a['used_mib']}MiB)" for a in heavy))

    try:
        containers = subprocess.run(
            ["docker", "ps", "--filter", "name=selvedge-", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        containers = []

    payload = {
        "schema": "selvedge/t2-launch-guard/v1",
        "stamp": stamp_now(),
        "utc": utc_now(),
        "revision": git_revision(),
        "disk": arithmetic,
        "gpu": gpu,
        "gpu_idle": gpu_ok,
        "gpu_reason": gpu_reason,
        "selvedge_containers": containers,
        "ok": bool(arithmetic["fits"] and gpu_ok and not containers),
    }
    if containers and not allow_busy:
        payload["ok"] = False
        payload["gpu_reason"] = (payload["gpu_reason"] or "") + \
            f" selvedge containers already running: {containers}"
    write_json_atomic(repo_path("results/t2/launch-guard.json"), payload)
    print(json.dumps({
        "ok": payload["ok"],
        "disk_required_bytes": arithmetic["required_bytes"],
        "disk_available_bytes": available,
        "disk_headroom_bytes": arithmetic["headroom_bytes"],
        "gpu_idle": payload["gpu_idle"],
        "gpu_reason": payload["gpu_reason"],
        "containers": containers,
    }, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 3


def cmd_targets(args) -> int:
    manifest_path = repo_path(os.path.join(ARTIFACT_REL, "teacher-targets.json"))
    index_path = repo_path(os.path.join(ARTIFACT_REL, "train.index.json"))
    result = {"required_tokens": args.budget_tokens, "manifest": None, "ok": False}
    if not os.path.exists(manifest_path) or not os.path.exists(index_path):
        result["reason"] = "no target manifest or no stream index"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    manifest = load_json(manifest_path)
    index = load_json(index_path)
    path = repo_path(os.path.join(ARTIFACT_REL, "teacher-targets.bin"))
    result["manifest"] = {
        "tokens": manifest.get("tokens"),
        "bytes": manifest.get("bytes"),
        "teacher_revision": manifest.get("teacher_revision"),
        "built_utc": manifest.get("built_utc"),
        "chunks": len(manifest.get("chunks") or []),
    }
    result["stream_sha256"] = index.get("stream_sha256")
    result["manifest_stream_sha256"] = manifest.get("stream_sha256")
    problems = []
    if (manifest.get("tokens") or 0) < args.budget_tokens:
        problems.append(f"cache has {manifest.get('tokens')} tokens < {args.budget_tokens}")
    if not os.path.exists(path):
        problems.append("cache file is missing")
    elif os.path.getsize(path) != manifest.get("bytes"):
        problems.append("cache size does not match the manifest")
    if manifest.get("stream_sha256") != index.get("stream_sha256"):
        problems.append("cache stream sha256 does not match the stream index")
    if len(manifest.get("chunks") or []) != manifest.get("tokens", 0) // 65536:
        problems.append("chunk count does not cover the token count")
    result["problems"] = problems
    result["ok"] = not problems
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# state


def state_path(override: str | None) -> str:
    return os.path.abspath(override) if override else repo_path(STATE_REL)


def split_get(payload, dotted: str):
    node = payload
    for part in dotted.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def split_set(payload, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = payload
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def cmd_state_init(args) -> int:
    path = state_path(args.state)
    previous = load_json(path) if os.path.exists(path) else None
    index = load_json(repo_path(os.path.join(ARTIFACT_REL, "train.index.json")))
    budget_manifest = load_json(repo_path(
        f"results/train/budget-{args.budget_stamp}.json")) if args.budget_stamp else None
    configs = {}
    admissions = {}
    for arm in ARMS:
        config_rel = f"results/train/admission-configs/t2/{arm}.json"
        admission_rel = f"results/train/admission/t2/{arm}.json"
        configs[arm] = {
            "path": config_rel,
            "sha256": sha256_file(repo_path(config_rel)) if os.path.exists(repo_path(config_rel)) else None,
        }
        if os.path.exists(repo_path(admission_rel)):
            record = load_json(repo_path(admission_rel))
            admissions[arm] = {
                "path": admission_rel,
                "sha256": sha256_file(repo_path(admission_rel)),
                "admitted": record.get("admitted"),
                "active_unique_params": record.get("active_unique_params"),
                "config_sha256": record.get("config_sha256"),
            }
    state = previous or {
        "schema": "selvedge/t2-pilot-state/v1",
        "pilot": "t2",
        "created_utc": utc_now(),
        "arms": {arm: {"status": "pending"} for arm in ARMS},
        "phases": {},
        "status": "initialised",
        "failure": None,
    }
    state.update({
        "updated_utc": utc_now(),
        "revision": git_revision(),
        "branch": subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "budget_tokens": args.budget_tokens,
        "steps": args.steps,
        "seed": args.seed,
        "tokens_per_step": 65_536,
        "data_order": {
            "stream_sha256": index.get("stream_sha256"),
            "tokens": index.get("tokens"),
            "spec_sha256": index.get("spec_sha256"),
            "contamination": index.get("contamination"),
        },
        "configs": configs,
        "admissions": admissions,
        "budget_receipt": None if budget_manifest is None else {
            "path": f"results/train/budget-{args.budget_stamp}.json",
            "stamp": budget_manifest.get("stamp"),
            "revision": budget_manifest.get("revision"),
        },
    })
    if not state["arms"]:
        state["arms"] = {arm: {"status": "pending"} for arm in ARMS}
    write_json_atomic(path, state)
    print(json.dumps({"state": path, "revision": state["revision"],
                      "budget_tokens": state["budget_tokens"]}, indent=2))
    return 0


def cmd_state_set(args) -> int:
    path = state_path(args.state)
    state = load_json(path)
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    split_set(state, args.field, value)
    state["updated_utc"] = utc_now()
    write_json_atomic(path, state)
    return 0


def cmd_state_archive_failure(args) -> int:
    """Move `failure` into the `failures` history; nothing is deleted.

    A resume supersedes the recorded failure without erasing it: the live state
    must describe the run in progress, and the operator must still be able to
    read what stopped the previous one.
    """
    path = state_path(args.state)
    state = load_json(path)
    failure = state.get("failure")
    if not failure:
        print(json.dumps({"archived": None}, indent=2, sort_keys=True))
        return 0
    state.setdefault("failures", []).append(failure)
    state["failure"] = None
    state["resumed_utc"] = utc_now()
    write_json_atomic(path, state)
    print(json.dumps({"archived": failure}, indent=2, sort_keys=True))
    return 0


def cmd_state_get(args) -> int:
    value = split_get(load_json(state_path(args.state)), args.field)
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, sort_keys=True))
    elif value is None:
        return 1
    else:
        print(value)
    return 0


# ---------------------------------------------------------------------------
# receipts


def read_jsonl_tail(path: str, limit: int = 1) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) > limit * 4:
                rows = rows[-limit * 4:]
    return rows[-limit:]


def count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def copy_evidence(arm: str, out_dir: str, run_dir: str) -> dict:
    copied = {}
    for name in ("train.jsonl", "summary.json", "run-meta.json", "zero-point.json",
                 "router-phase.json", "latest.json"):
        source = os.path.join(run_dir, name)
        if os.path.exists(source):
            target = os.path.join(out_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(source, target)
            copied[name] = {"path": os.path.relpath(target, REPO),
                            "sha256": sha256_file(target),
                            "bytes": os.path.getsize(target)}
    return copied


def cmd_arm_receipt(args) -> int:
    arm = args.arm
    run_dir = repo_path(os.path.join(ARTIFACT_REL, "t2", args.run_name or arm))
    out_dir = repo_path(os.path.join("results/t2", arm))
    os.makedirs(out_dir, exist_ok=True)
    meta = load_json(os.path.join(run_dir, "run-meta.json"))
    summary = load_json(os.path.join(run_dir, "summary.json")) \
        if os.path.exists(os.path.join(run_dir, "summary.json")) else None
    rows = read_jsonl_tail(os.path.join(run_dir, "train.jsonl"), 1)
    last = rows[-1] if rows else {}
    steps_logged = count_jsonl(os.path.join(run_dir, "train.jsonl"))

    checkpoint = None
    candidates = sorted(name for name in os.listdir(run_dir)
                        if re.fullmatch(r"step-\d+\.pt", name)) if os.path.isdir(run_dir) else []
    if candidates:
        name = candidates[-1]
        path = os.path.join(run_dir, name)
        sidecar = path + ".sha256"
        digest = None
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
            if raw.startswith("{"):
                # loop.save_checkpoint writes a JSON sidecar with sha256 and bytes
                digest = (json.loads(raw) or {}).get("sha256")
            elif raw:
                digest = raw.split()[0]
        checkpoint = {
            "path": os.path.relpath(path, REPO),
            "name": name,
            "bytes": os.path.getsize(path),
            "sha256": digest,
            "sha256_source": "sidecar" if digest else "absent",
        }

    argv = None
    if args.training_state and os.path.exists(args.training_state):
        argv = load_json(args.training_state).get("argv")
        if args.consume:
            os.unlink(args.training_state)

    receipt = {
        "schema": "selvedge/t2-arm-receipt/v1",
        "arm": arm,
        "stamp": stamp_now(),
        "utc": utc_now(),
        "revision": meta.get("revision"),
        "git_sha_at_launch": args.launch_revision,
        "seed": meta.get("seed"),
        "config": {"path": meta.get("config"), "sha256": meta.get("config_sha256")},
        "admission": {
            "path": meta.get("admission"),
            "active_unique_params": meta.get("admission_active_unique"),
            "selector_unique_params": meta.get("admitted_selector_unique"),
            "execution_unique_params": meta.get("admitted_execution_unique"),
            "declared_active": meta.get("declared_active"),
        },
        "total_accessible": meta.get("total_accessible"),
        "trainable_elements": meta.get("trainable_elements"),
        "data_order": {
            "stream_sha256": meta.get("stream_sha256"),
            "targets_tokens": meta.get("targets_tokens"),
            "teacher_manifest_revision": meta.get("teacher_manifest"),
        },
        "budget": {
            "steps": meta.get("steps"),
            "tokens_per_step": meta.get("tokens_per_step"),
            "registered_budget_tokens": meta.get("registered_budget_tokens"),
            "micro_batch": meta.get("micro_batch"),
            "accum": meta.get("accum"),
            "window": meta.get("window"),
        },
        "phases": {
            "warmup_steps": meta.get("warmup_steps"),
            "load_balance": meta.get("load_balance"),
            "router_phase": (summary or {}).get("router_phase"),
        },
        "resource": {
            "gpu": meta.get("gpu"),
            "torch": meta.get("torch"),
            "grad_checkpoint": meta.get("grad_checkpoint"),
            "memory": meta.get("memory"),
            "extra_ledger": meta.get("extra_ledger"),
            "active_extra_elements": meta.get("active_extra_elements"),
        },
        "run": {
            "directory": os.path.relpath(run_dir, REPO),
            "argv": argv,
            "container": args.container,
            "court_receipt": args.court_receipt,
            "court_log": args.court_log,
            "exit_code": args.exit_code,
            "exit_code_source": ("container (docker wait / inspect)"
                                 if args.exit_code is not None
                                 else "unobserved: no court attribution"),
            "started_utc": args.started_utc,
            "finished_utc": args.finished_utc,
            "steps_logged": steps_logged,
            "final_step": last.get("step"),
            "summary": summary,
        },
        "measured": {
            "tokens": (summary or {}).get("tokens", last.get("tokens")),
            "train_seconds": (summary or {}).get("train_seconds"),
            "wall_seconds": (summary or {}).get("wall_seconds"),
            "tokens_per_second": (summary or {}).get("tokens_per_second"),
            "vram_peak_gib": (summary or {}).get("vram_peak_gib",
                                                 last.get("vram_peak_gib")),
            "vram_peak_source": "cuda.max_memory_allocated over the whole run",
            "last_loss": last.get("loss"),
            "last_grad_norm": last.get("grad_norm"),
            "gpu_hours": None if not (summary or {}).get("wall_seconds")
            else round(summary["wall_seconds"] / 3600.0, 4),
        },
        "checkpoint": checkpoint,
        "evidence": copy_evidence(arm, out_dir, run_dir),
    }
    measured = receipt["measured"]
    if measured.get("tokens") and measured.get("train_seconds"):
        receipt["compute"] = {
            "training_tokens": measured["tokens"],
            "tokens_per_second": measured["tokens_per_second"],
            "training_flops": None,
            "note": "FLOPs are the admission record's declared per-token count; the "
                    "receipt stores the declared value rather than a counter",
        }
        if meta.get("admission"):
            admission = load_json(repo_path(meta["admission"]))
            declared = admission.get("total_flops")
            if isinstance(declared, dict):
                per_token = sum(int(value) for value in declared.values()
                                if isinstance(value, (int, float)))
                break_down = {key: int(value) for key, value in declared.items()
                              if isinstance(value, (int, float))}
            else:
                per_token = int(declared) if declared else 0
                break_down = None
            receipt["compute"]["declared_total_flops_per_token"] = per_token
            receipt["compute"]["declared_flops_breakdown"] = break_down
            receipt["compute"]["declared_flops_trained"] = per_token * int(measured["tokens"])
    write_json_atomic(os.path.join(out_dir, "run.json"), receipt)
    print(json.dumps({
        "arm": arm,
        "out": os.path.relpath(os.path.join(out_dir, "run.json"), REPO),
        "final_step": receipt["run"]["final_step"],
        "tokens": receipt["measured"]["tokens"],
        "tokens_per_second": receipt["measured"]["tokens_per_second"],
        "vram_peak_gib": receipt["measured"]["vram_peak_gib"],
        "checkpoint": checkpoint,
    }, indent=2, sort_keys=True))
    return 0


def cmd_eval_receipt(args) -> int:
    receipt = load_json(repo_path(args.eval))
    results = receipt.get("results") or {}
    task = {
        "humaneval_pass_at_1": (results.get("humaneval") or {}).get("pass_at_1"),
        "gsm8k_exact_match": (results.get("gsm8k") or {}).get("exact_match"),
        "truthfulqa_mc1": (results.get("truthfulqa") or {}).get("mc1"),
    }
    scores = [value for value in task.values() if value is not None]
    primary = sum(scores) / len(scores) if scores else None
    distributional = results.get("distributional") or {}
    payload = {
        "schema": "selvedge/t2-eval-summary/v1",
        "arm": args.arm,
        "stamp": stamp_now(),
        "utc": utc_now(),
        "eval_receipt": args.eval,
        "revision": receipt.get("revision"),
        "eval_receipt_stamp": receipt.get("stamp"),
        "eval_revision": receipt.get("revision"),
        "checkpoint_sha256": (receipt.get("model") or {}).get("checkpoint_sha256"),
        "config_sha256": (receipt.get("model") or {}).get("config_sha256"),
        "compliance_label": receipt.get("compliance_label"),
        "primary_composite_pc": primary,
        "primary_kl_mixed_p1": (distributional.get("mixed") or {}).get("kl"),
        "per_task": task,
        "per_task_ci": {
            "humaneval": (results.get("humaneval") or {}).get("pass_at_1_ci"),
            "gsm8k": (results.get("gsm8k") or {}).get("exact_match_ci"),
            "truthfulqa": (results.get("truthfulqa") or {}).get("mc1_ci"),
        },
        "per_domain": {
            domain: {
                "kl": values.get("kl"),
                "ppl": values.get("ppl"),
                "top1": values.get("top1"),
                "window_count": values.get("window_count"),
                "documents": len(values.get("documents") or []),
            }
            for domain, values in (distributional.get("domains") or {}).items()
            if isinstance(values, dict)
        },
        "pooled": {
            "kl": distributional.get("kl"),
            "ppl": distributional.get("ppl"),
            "top1": distributional.get("top1"),
        },
        "mixed": distributional.get("mixed"),
        "resources": receipt.get("resources"),
        "instrument": receipt.get("instrument"),
        "seconds": receipt.get("seconds"),
        "vram_peak_gib": receipt.get("vram_peak_gib"),
        "command": receipt.get("command"),
        "determinism": receipt.get("determinism"),
    }
    os.makedirs(repo_path(os.path.join("results/t2", args.arm)), exist_ok=True)
    out = repo_path(os.path.join("results/t2", args.arm, "eval.json"))
    write_json_atomic(out, payload)
    print(json.dumps({
        "arm": args.arm,
        "out": os.path.relpath(out, REPO),
        "pc": primary,
        "kl_mixed": payload["primary_kl_mixed_p1"],
        "per_task": task,
    }, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# progress


def arm_status(arm: str, state: dict) -> dict:
    entry = dict((state.get("arms") or {}).get(arm) or {})
    run_dir = repo_path(os.path.join(ARTIFACT_REL, "t2", entry.get("run_name") or arm))
    rows = read_jsonl_tail(os.path.join(run_dir, "train.jsonl"), 1)
    target = int(state.get("steps") or 1024)
    if rows:
        last = rows[-1]
        entry["last_step"] = last.get("step")
        entry["tokens"] = last.get("tokens")
        entry["tokens_per_second"] = last.get("tokens_per_second")
        entry["vram_peak_gib"] = last.get("vram_peak_gib")
        entry["loss"] = last.get("loss")
        if entry.get("last_step"):
            entry["fraction"] = round(min(1.0, entry["last_step"] / target), 4)
            if entry.get("tokens_per_second"):
                remaining_tokens = (target - entry["last_step"]) * 65_536
                entry["eta_seconds"] = int(remaining_tokens / entry["tokens_per_second"])
    entry["run_directory"] = os.path.relpath(run_dir, REPO)
    entry["has_summary"] = os.path.exists(os.path.join(run_dir, "summary.json"))
    entry["has_eval"] = os.path.exists(repo_path(os.path.join("results/t2", arm, "eval.json")))
    return entry


def remaining_estimates(state: dict) -> dict:
    per_arm_hours = {}
    fallback = {"D": 2.35, "M": 2.00, "X": 3.70, "H": 3.91, "H*": 3.72}
    for arm in ARMS:
        entry = arm_status(arm, state)
        if entry.get("status") in ("evaluated", "skipped"):
            per_arm_hours[arm] = 0.0
            continue
        if entry.get("eta_seconds") is not None:
            per_arm_hours[arm] = round(entry["eta_seconds"] / 3600.0, 2)
        elif entry.get("summary_seconds"):
            per_arm_hours[arm] = round(entry["summary_seconds"] / 3600.0, 2)
        else:
            per_arm_hours[arm] = fallback.get(arm, 4.0)
    evaluation_hours = sum(
        0.08 for arm in ARMS
        if not arm_status(arm, state).get("has_eval")
        and arm_status(arm, state).get("status") not in ("pending",))
    targets = (state.get("phases") or {}).get("targets") or {}
    targets_hours = 0.0 if targets.get("verified") else 1.2
    return {
        "per_arm_training_hours": per_arm_hours,
        "training_hours": round(sum(per_arm_hours.values()), 2),
        "evaluation_hours": round(evaluation_hours, 2),
        "targets_hours": targets_hours,
        "total_hours": round(sum(per_arm_hours.values()) + evaluation_hours + targets_hours, 2),
    }


def cmd_progress(args) -> int:
    path = state_path(args.state)
    if not os.path.exists(path):
        print("t2 pilot: no state file at " + path)
        return 1
    state = load_json(path)
    rows = []
    for arm in ARMS:
        rows.append({"arm": arm, **arm_status(arm, state)})
    estimates = remaining_estimates(state)
    payload = {
        "schema": "selvedge/t2-progress/v1",
        "utc": utc_now(),
        "status": state.get("status"),
        "updated_utc": state.get("updated_utc"),
        "revision": state.get("revision"),
        "budget_tokens": state.get("budget_tokens"),
        "steps": state.get("steps"),
        "phases": state.get("phases"),
        "failure": state.get("failure"),
        "arms": rows,
        "remaining": estimates,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"T2 pilot — status {state.get('status')} (updated {state.get('updated_utc')})")
    print(f"revision {state.get('revision')}  budget {state.get('budget_tokens')} tokens"
          f"  steps {state.get('steps')}  seed {state.get('seed')}")
    targets = (state.get("phases") or {}).get("targets") or {}
    print(f"targets: {targets.get('status', 'unknown')} "
          f"tokens={targets.get('tokens')} verified={targets.get('verified')}")
    print()
    header = f"{'arm':4} {'status':10} {'step':>6} {'tok/s':>8} {'vram':>6} {'loss':>8} {'eta':>7} {'eval':>5}"
    print(header)
    for row in rows:
        eta = row.get("eta_seconds")
        eta_text = "-" if eta is None else f"{eta / 60:.0f}m"
        print(f"{row['arm']:4} {str(row.get('status')):10} "
              f"{str(row.get('last_step', '-')):>6} "
              f"{('%.0f' % row['tokens_per_second']) if row.get('tokens_per_second') else '-':>8} "
              f"{('%.2f' % row['vram_peak_gib']) if row.get('vram_peak_gib') else '-':>6} "
              f"{('%.4f' % row['loss']) if row.get('loss') is not None else '-':>8} "
              f"{eta_text:>7} {str(row.get('has_eval')):>5}")
    print()
    print("remaining: training {training_hours} h, evaluation {evaluation_hours} h, "
          "targets {targets_hours} h -> {total_hours} h".format(**estimates))
    if state.get("failure"):
        print(f"FAILURE: {json.dumps(state['failure'], sort_keys=True)}")
    print()
    print("resume:  scripts/run-t2-pilot.sh --resume")
    print("monitor: scripts/poll-t2-pilot.sh --follow")
    if state.get("status") == "done":
        print("publish: scripts/finish-t2-pilot.sh   "
              "# renders results/T2-PILOT.md and runs the courts")
    elif state.get("status") == "partial":
        print("some arms are missing: rerun with scripts/run-t2-pilot.sh --resume")
    return 0


def cmd_estimate(args) -> int:
    state = load_json(state_path(args.state))
    print(json.dumps(remaining_estimates(state), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------


def _exit_code(value: str):
    """An exit code, or None when the run has no court attribution (a
    verified-checkpoint harvest after the orchestrator died)."""
    if value is None or str(value).strip().lower() in ("", "none", "unobserved"):
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    guard = sub.add_parser("guard")
    guard.add_argument("--budget-tokens", type=int, default=67_108_864)
    guard.set_defaults(func=cmd_guard)

    targets = sub.add_parser("targets")
    targets.add_argument("--budget-tokens", type=int, default=67_108_864)
    targets.set_defaults(func=cmd_targets)

    for name in ("state-init", "state-set", "state-get", "state-archive-failure",
                 "progress", "estimate"):
        child = sub.add_parser(name)
        child.add_argument("--state", default=None)
        if name == "state-init":
            child.add_argument("--budget-tokens", type=int, default=67_108_864)
            child.add_argument("--steps", type=int, default=1024)
            child.add_argument("--seed", type=int, default=1001)
            child.add_argument("--budget-stamp", default=None)
            child.set_defaults(func=cmd_state_init)
        elif name == "state-set":
            child.add_argument("--field", required=True)
            child.add_argument("--value", required=True)
            child.set_defaults(func=cmd_state_set)
        elif name == "state-get":
            child.add_argument("--field", required=True)
            child.set_defaults(func=cmd_state_get)
        elif name == "state-archive-failure":
            child.set_defaults(func=cmd_state_archive_failure)
        elif name == "progress":
            child.add_argument("--json", action="store_true")
            child.set_defaults(func=cmd_progress)
        else:
            child.set_defaults(func=cmd_estimate)

    arm = sub.add_parser("arm-receipt")
    arm.add_argument("--arm", required=True, choices=list(ARMS))
    arm.add_argument("--run-name", default=None)
    arm.add_argument("--training-state", default=None)
    arm.add_argument("--consume", action="store_true")
    arm.add_argument("--launch-revision", default=None)
    arm.add_argument("--container", default=None)
    arm.add_argument("--court-receipt", default=None)
    arm.add_argument("--court-log", default=None)
    arm.add_argument("--exit-code", type=_exit_code, default=None,
                     help="container exit code, or 'unobserved' when no court "
                          "attribution exists (a verified-checkpoint harvest)")
    arm.add_argument("--started-utc", default=None)
    arm.add_argument("--finished-utc", default=None)
    arm.set_defaults(func=cmd_arm_receipt)

    evaluation = sub.add_parser("eval-receipt")
    evaluation.add_argument("--arm", required=True, choices=list(ARMS) + ["D0"])
    evaluation.add_argument("--eval", required=True,
                            help="path to the oracle/eval/baseline.py receipt")
    evaluation.set_defaults(func=cmd_eval_receipt)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
