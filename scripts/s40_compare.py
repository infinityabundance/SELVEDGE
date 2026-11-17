#!/usr/bin/env python3
"""Score every stage-2 arm with one instrument and emit one table (S40).

Arms, built from their registered artifacts and scored identically:

    base               frozen Qwen3.5-0.8B-Base, no hook
    adapter-2m         frozen core + parameter-matched dense adapter
    memory-2m          trained grafted table, evaluated after the S40 fix
    memory-ungrafted   the module at its deterministic init (zero value projection)
    memory-grafted     the step-0 teacher graft, no training
    full-distill       full-backbone distillation, tied embedding frozen

Two measurements:

1. **Held-out CE/KL** on a stream disjoint from training and grafting. The
   instrument runs every arm over the same non-overlapping token chunks and
   reports mean CE, mean KL at the registered temperature, their standard errors
   across chunks, and paired per-chunk deltas against `full-distill`.
2. **Anchored harness**, if per-arm receipts exist under `--harness-dir`, using
   the same task/few-shot/batch/config registration as the S36 anchors. Missing
   receipts are recorded as `not_scored`, never as zero.

The `--self-test` path builds tiny fake checkpoints for all six arms and runs the
same code on CPU; it is the test that the instrument works, not a result.

Usage (real):
    SELVEDGE_MEM=12g scripts/court --gpu oracle -- python3 scripts/s40_compare.py \
        --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B-Base/dc7cdfe2... \
        --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B-Base/b1485b2f... \
        --heldout-manifest results/stage2/s40-heldout-<stamp>.json \
        --memory-init results/stage2/graft/20261116T231451Z/memory-init.pt \
        --memory-checkpoint results/stage2/runs/memory-2m/ckpt-000976.pt \
        --adapter-checkpoint results/stage2/runs/adapter-2m/ckpt-000976.pt \
        --full-checkpoint results/stage2/runs/full-20m/ckpt-009765.pt \
        --harness-dir results/stage2/eval \
        --out results/stage2/compare-<stamp>.json

Usage (CPU test):
    scripts/court python -- python3 scripts/s40_compare.py --self-test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

from frontier import arms, compare, full, memory  # noqa: E402

ARMS = ("base", "adapter-2m", "memory-2m", "memory-ungrafted",
        "memory-grafted", "full-distill")
REFERENCE_ARM = "full-distill"
FAMILIES = {"knowledge": ["mmlu_pro", "mmlu"],
            "math": ["gsm8k", "minerva_math500"]}
DEFAULT_STUDENT = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
DEFAULT_TEACHER = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen",
    "Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
GIB = float(1 << 30)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_manifest(path: str) -> tuple[dict, np.ndarray]:
    manifest = json.load(open(path, encoding="utf-8"))
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit(f"{path}: stream sha {got} != {manifest['sha256']}")
    tokens = np.fromfile(manifest["stream_path"], dtype=np.uint32)
    if tokens.size != int(manifest["tokens"]):
        raise SystemExit(f"{path}: {tokens.size} tokens, manifest says {manifest['tokens']}")
    return manifest, tokens


def build_memory(init_path: str, checkpoint: str | None, device: str,
                 ungrafted: bool):
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    context = memory.TokenContext()
    module = memory.ConditionalMemory(
        hidden=config["hidden"], value_dim=config["value_dim"],
        rows=config["rows"], sites=tuple(config["sites"]),
        conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
        seed=config["seed"], context=context)
    if not ungrafted:
        state = payload["state_dict"]
        if checkpoint:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu",
                                            weights_only=False)
            if checkpoint_payload.get("arm") not in (None, "memory"):
                raise SystemExit(f"{checkpoint}: not a memory checkpoint")
            state = checkpoint_payload["arm_state"]
        module.load_state_dict(state, strict=True)
    return module.to(device), {
        "memory_init": init_path,
        "memory_init_sha256": sha256_file(init_path),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "grafted": not ungrafted,
    }


def build_named_arm(name: str, model, args, device: str):
    """Install one arm on a fresh backbone. Returns (model, handles, report)."""
    if name == "base":
        return model, [], {"kind": "stage2_base", "added_elements": 0}
    if name == "adapter-2m":
        if not args.adapter_checkpoint:
            raise SystemExit("--adapter-checkpoint is required for adapter-2m")
        payload = torch.load(args.adapter_checkpoint, map_location="cpu",
                             weights_only=False)
        module = arms.DenseAdapter(hidden=args.hidden).to(device)
        module.load_state_dict(payload["arm_state"], strict=True)
        handles = arms.install_adapter(model, module, (args.site,))
        report = module.active_parameter_report()
        report.update({"checkpoint": args.adapter_checkpoint,
                       "checkpoint_sha256": sha256_file(args.adapter_checkpoint),
                       "checkpoint_step": payload.get("step")})
        return model, handles, report
    if name in ("memory-2m", "memory-ungrafted", "memory-grafted"):
        if not args.memory_init:
            raise SystemExit("--memory-init is required for the memory arms")
        if name == "memory-2m" and not args.memory_checkpoint:
            raise SystemExit("--memory-checkpoint is required for memory-2m")
        if name == "memory-ungrafted" and args.memory_checkpoint:
            checkpoint = None
        elif name == "memory-grafted":
            checkpoint = None
        else:
            checkpoint = args.memory_checkpoint
        module, report = build_memory(
            args.memory_init, checkpoint, device, ungrafted=(name == "memory-ungrafted"))
        handles = arms.install_memory(model, module)
        report.update(module.active_parameter_report())
        report["extension"] = report.pop("kind", "stage2_memory")
        report["kind"] = "stage2_memory_arm"
        return model, handles, report
    if name == "full-distill":
        if not args.full_checkpoint:
            raise SystemExit("--full-checkpoint is required for full-distill")
        payload = torch.load(args.full_checkpoint, map_location="cpu",
                             weights_only=False)
        if payload.get("arm") not in (None, "full"):
            raise SystemExit(f"{args.full_checkpoint}: not a full-arm checkpoint")
        arm = full.load_full_state_into(model, payload["arm_state"])
        report = arm.active_parameter_report()
        report.update({"checkpoint": args.full_checkpoint,
                       "checkpoint_sha256": sha256_file(args.full_checkpoint),
                       "checkpoint_step": payload.get("step")})
        return model, [], report
    raise SystemExit(f"unknown arm {name!r}")


def load_backbone(args, device: str):
    if args.fake:
        from frontier import fake
        torch.manual_seed(0)
        return fake.TinyLM(args.vocab, args.hidden, args.layers).to(device)
    return arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)


def load_teacher(args, device: str):
    if args.fake:
        from frontier import fake
        torch.manual_seed(1)
        return fake.TinyLM(args.vocab, args.hidden, args.layers).to(device)
    return arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)


def score_all(args, device: str):
    manifest, tokens = read_manifest(args.heldout_manifest)
    tokens = torch.from_numpy(tokens.astype(np.int64))
    teacher = load_teacher(args, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    scored = {}
    reports = {}
    paths = {}
    for name in ARMS:
        print(f"compare: {name}", flush=True)
        model = load_backbone(args, device)
        model.eval()
        model, handles, report = build_named_arm(name, model, args, device)
        model.eval()
        result = compare.score_stream(model, teacher, tokens, args.seq_len, device,
                                      args.temperature, args.vocab_chunk)
        scored[name] = result
        reports[name] = report
        paths[name] = checkpoint_path(name, args)
        arms.uninstall(handles)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    del teacher
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return manifest, scored, reports, paths


def checkpoint_path(name: str, args) -> dict:
    mapping = {"adapter-2m": args.adapter_checkpoint,
               "memory-2m": args.memory_checkpoint,
               "memory-ungrafted": None,
               "memory-grafted": args.memory_init,
               "full-distill": args.full_checkpoint,
               "base": None}
    path = mapping.get(name)
    return {"path": path, "sha256": sha256_file(path) if path else None}


def harness_section(directory: str | None) -> dict:
    if not directory:
        return {"status": "not_scored",
                "reason": "no --harness-dir was given; no arm has an anchored receipt"}
    per_arm: dict[str, dict[str, dict]] = {}
    missing = []
    for name in ARMS:
        path = os.path.join(directory, name, "eval.json")
        if not os.path.exists(path):
            missing.append(name)
            continue
        payload = json.load(open(path, encoding="utf-8"))
        tasks = {row["task"]: {"score": row["score"], "stderr": row.get("stderr")}
                 for row in payload.get("tasks", [])}
        per_arm[name] = tasks
    families = compare.family_means(per_arm, FAMILIES)
    reference = families.get(REFERENCE_ARM, {})
    deltas = {}
    for name, rows in families.items():
        deltas[name] = {}
        for family, row in rows.items():
            base = reference.get(family)
            if base is None:
                continue
            combined = (None if row["stderr"] is None or base["stderr"] is None
                        else (row["stderr"] ** 2 + base["stderr"] ** 2) ** 0.5)
            deltas[name][family] = {
                "mean": row["mean"], "stderr": row["stderr"],
                "delta_vs_reference": row["mean"] - base["mean"],
                "combined_stderr": combined,
            }
    return {"status": "scored" if per_arm else "not_scored",
            "directory": directory, "families": families,
            "deltas_vs_reference": deltas, "missing_arms": missing,
            "reference_arm": REFERENCE_ARM}


def table_rows(scored: dict) -> list[dict]:
    reference = scored[REFERENCE_ARM]
    rows = []
    for name in ARMS:
        summary = compare.arm_summary(scored[name], reference)
        summary["arm"] = name
        rows.append(summary)
    return rows


def run(args) -> int:
    device = "cpu" if args.fake else "cuda"
    if not args.fake:
        if not torch.cuda.is_available():
            raise SystemExit("compare: CUDA required; run through scripts/court --gpu oracle")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if args.vram_limit_gib * GIB > free_bytes:
            raise SystemExit(f"compare: limit {args.vram_limit_gib} GiB exceeds free "
                             f"{free_bytes / GIB:.2f} GiB")
        print(f"compare: device free {free_bytes / GIB:.2f} GiB", flush=True)
    manifest, scored, reports, paths = score_all(args, device)
    rows = table_rows(scored)
    text = compare.format_table(rows, REFERENCE_ARM)
    print(text, flush=True)

    payload = {
        "schema": "selvedge/stage2-arm-comparison/v1",
        "kind": "arm-comparison",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "heldout": {
            "manifest": args.heldout_manifest,
            "manifest_sha256": sha256_file(args.heldout_manifest),
            "stream_sha256": manifest.get("sha256"),
            "tokens": int(manifest.get("tokens", 0)),
            "tokens_scored": int(scored[REFERENCE_ARM]["tokens_scored"]),
            "seq_len": args.seq_len,
            "chunks": scored[REFERENCE_ARM]["chunks"],
            "chunking": "non-overlapping next-token chunks, partial tail dropped",
        },
        "temperature": args.temperature,
        "reference_arm": REFERENCE_ARM,
        "arms": {
            name: {
                **compare.arm_summary(scored[name], scored[REFERENCE_ARM]),
                "report": reports[name],
                "checkpoint": paths[name],
            }
            for name in ARMS
        },
        "table": text,
        "harness": harness_section(args.harness_dir),
        "fake": bool(args.fake),
        "command": sys.argv,
    }
    if args.out:
        write_json(args.out, payload)
        print(f"compare: receipt {args.out}", flush=True)
    return 0


def self_test() -> int:
    from frontier import fake
    root = tempfile.mkdtemp(prefix="s40-selftest-")
    paths = fake.write_fake_checkpoints(root)
    args = SimpleNamespace(
        fake=True, vocab=paths["vocab"], hidden=paths["hidden"],
        layers=paths["layers"], student_dir=None, teacher_dir=None,
        adapter_checkpoint=paths["adapter"],
        memory_init=paths["memory_init"],
        memory_checkpoint=paths["memory_trained"],
        full_checkpoint=paths["full"],
        heldout_manifest=paths["stream_manifest"],
        seq_len=paths["seq_len"], temperature=2.0, vocab_chunk=4,
        site=2, harness_dir=None,
        out=os.path.join(root, "compare.json"),
        vram_limit_gib=12.0)
    code = run(args)
    receipt = json.load(open(args.out, encoding="utf-8"))
    assert set(receipt["arms"]) == set(ARMS), receipt["arms"].keys()
    for name in ARMS:
        assert receipt["arms"][name]["ce"]["mean"] is not None
        assert receipt["arms"][name]["kl"]["mean"] is not None
    # the untrained ungrafted module has a zero value projection: it must equal
    # the frozen base to float tolerance
    base_ce = receipt["arms"]["base"]["ce"]["mean"]
    ungrafted_ce = receipt["arms"]["memory-ungrafted"]["ce"]["mean"]
    assert abs(base_ce - ungrafted_ce) < 1e-9, (base_ce, ungrafted_ce)
    # the grafted step-0 arm is a real function change
    assert receipt["arms"]["memory-grafted"]["ce"]["mean"] != base_ce
    print(f"selftest: six arms scored, ungrafted == base, receipt {args.out}", flush=True)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--student-dir", default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-dir", default=DEFAULT_TEACHER)
    parser.add_argument("--heldout-manifest", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--memory-checkpoint", default=None)
    parser.add_argument("--full-checkpoint", default=None)
    parser.add_argument("--harness-dir", default=None)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--site", type=int, default=memory.DEFAULT_SITE)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-chunk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--vram-limit-gib", type=float, default=13.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.heldout_manifest:
        raise SystemExit("--heldout-manifest is required (or use --self-test)")
    if args.out is None:
        args.out = os.path.join("results", "stage2", f"compare-{stamp()}.json")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
