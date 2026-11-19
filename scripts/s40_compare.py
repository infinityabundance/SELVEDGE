#!/usr/bin/env python3
"""Score every stage-2 arm with one instrument and emit one table (S40).

Arms, built from their registered artifacts and scored identically:

    base               frozen Qwen3.5-0.8B-Base, no hook
    adapter-2m         frozen core + parameter-matched dense adapter
    memory-2m          trained grafted table, evaluated after the S40 fix
    memory-ungrafted   the module at its deterministic init (zero value projection)
    memory-grafted     the step-0 teacher graft, no training
    full-distill       full-backbone distillation, tied embedding frozen

A seventh arm, `adapter-20m`, is the S39 20M-token adapter control. It is not
part of the registered six-arm comparison (whose default list is unchanged);
it exists so the 10x-token question can be scored by this instrument with an
honest label instead of reusing the `adapter-2m` slot. Select it with `--arms`.
The reference arm is `full-distill` unless `--reference-arm` names a selected
arm, so an intermediate table can be produced before the full arm exists.

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

from frontier import arms, compare, corrective, full, matched_controls, memory  # noqa: E402

ARMS = ("base", "adapter-2m", "memory-2m", "memory-ungrafted",
        "memory-grafted", "full-distill")
CONTROL_ARM = "adapter-20m"
#: Stage-2b corrective arms. They are scored through the corrective module with
#: the trained gate, on the same anchored instrument as the other arms.
CORRECTIVE_ARMS = ("corrective-kl-20m-math", "corrective-signed-20m-math",
                   "corrective-signed-true-20m-math",
                   "corrective-both-20m-math")
#: S47 mandatory matched controls: the same data without the external memory,
#: and the same gate over a dense residual instead of the memory.
MATCHED_CONTROLS = ("adapter-20m-math-matched", "gated-dense-20m-math")
#: S48 screen variants of the same arms at the 2M budget. The names differ so a
#: receipt can never be mistaken for a 20M arm's receipt; the construction,
#: accounting and installed module are identical.
SCREEN_CORRECTIVE_ARMS = ("corrective-kl-2m-math",
                          "corrective-signed-exp-2m-math",
                          "corrective-signed-true-2m-math")
SCREEN_MATCHED_CONTROLS = ("adapter-2m-math-matched", "gated-dense-2m-math")
#: The weight-interpolation diagnostic loads one interpolated full-arm
#: checkpoint through the exact `full-distill` contract, but is labelled by its
#: own name so a receipt can never be mistaken for a trained full arm. The
#: checkpoint carries the coefficient in its `interpolation` block.
INTERPOLATED_ARM = "interpolated"
SCORED_ARMS = (ARMS + (CONTROL_ARM,) + CORRECTIVE_ARMS + MATCHED_CONTROLS
               + SCREEN_CORRECTIVE_ARMS + SCREEN_MATCHED_CONTROLS
               + (INTERPOLATED_ARM,))
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
    model, handles, report = _build_named_arm(name, model, args, device)
    report.setdefault(
        "eval_attention_implementation",
        getattr(getattr(model, "config", None), "_attn_implementation", None))
    report.setdefault(
        "eval_attention_source",
        "transformers default, to match the anchor instrument")
    return model, handles, report


def _build_named_arm(name: str, model, args, device: str):
    """Install one arm on a fresh backbone. Returns (model, handles, report)."""
    if name == "base":
        return model, [], {"kind": "stage2_base", "added_elements": 0}
    if name in ("adapter-2m", CONTROL_ARM):
        checkpoint = (args.control_checkpoint if name == CONTROL_ARM
                      else args.adapter_checkpoint)
        if not checkpoint:
            flag = "--control-checkpoint" if name == CONTROL_ARM else "--adapter-checkpoint"
            raise SystemExit(f"{flag} is required for {name}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        module = arms.DenseAdapter(hidden=args.hidden).to(device)
        module.load_state_dict(payload["arm_state"], strict=True)
        handles = arms.install_adapter(model, module, (args.site,))
        report = module.active_parameter_report()
        report.update({"checkpoint": checkpoint,
                       "checkpoint_sha256": sha256_file(checkpoint),
                       "checkpoint_step": payload.get("step"),
                       "checkpoint_tokens_seen": payload.get("tokens_seen")})
        if name == CONTROL_ARM:
            report["control"] = "S39 20M-token frozen-backbone adapter"
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
    if name in MATCHED_CONTROLS + SCREEN_MATCHED_CONTROLS:
        if name in ("adapter-20m-math-matched", "adapter-2m-math-matched"):
            checkpoint = getattr(args, "matched_adapter_checkpoint", None)
            if not checkpoint:
                raise SystemExit(
                    f"--matched-adapter-checkpoint is required for {name}")
            module = matched_controls.MatchedDenseAdapter(
                hidden=args.hidden).to(device)
        else:
            checkpoint = getattr(args, "gated_dense_checkpoint", None)
            if not checkpoint:
                raise SystemExit(
                    f"--gated-dense-checkpoint is required for {name}")
            module = matched_controls.IdentityGatedDenseResidual(
                hidden=args.hidden).to(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        module.load_state_dict(payload["arm_state"], strict=True)
        handles = arms.install_adapter(model, module, (args.site,))
        report = module.active_parameter_report()
        report.update({
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_step": payload.get("step"),
            "checkpoint_tokens_seen": payload.get("tokens_seen"),
            "arm_family": name,
        })
        return model, handles, report
    if name in CORRECTIVE_ARMS + SCREEN_CORRECTIVE_ARMS:
        if not args.memory_init:
            raise SystemExit("--memory-init is required for a corrective arm")
        init_payload = torch.load(args.memory_init, map_location="cpu",
                                  weights_only=False)
        config = init_payload["config"]
        module = corrective.CorrectiveMemory(
            hidden=config["hidden"], value_dim=config["value_dim"],
            rows=config["rows"], sites=tuple(config["sites"]),
            conv_kernel=config["conv_kernel"], table_dtype=torch.float32,
            seed=config["seed"])
        module.load_state_dict(init_payload["state_dict"], strict=True)
        report = {
            "memory_init": args.memory_init,
            "memory_init_sha256": sha256_file(args.memory_init),
            "init_variant": config.get("variant"),
            "checkpoint": None,
            "checkpoint_sha256": None,
            "grafted": True,
            "gate_initialisation": "zero (the untrained corrective arm is the base)",
        }
        if args.memory_checkpoint:
            payload = torch.load(args.memory_checkpoint, map_location="cpu",
                                 weights_only=False)
            if payload.get("arm") not in (None, "corrective"):
                raise SystemExit(f"{args.memory_checkpoint}: not a corrective checkpoint")
            module.load_state_dict(payload["arm_state"], strict=True)
            report.update({
                "checkpoint": args.memory_checkpoint,
                "checkpoint_sha256": sha256_file(args.memory_checkpoint),
                "checkpoint_step": payload.get("step"),
                "checkpoint_tokens_seen": payload.get("tokens_seen"),
            })
        module = module.to(device)
        module.eval()
        handles = arms.install_memory(model, module)
        report.update(module.active_parameter_report())
        report["extension"] = report.pop("kind", "stage2_corrective")
        report["kind"] = "stage2_corrective_arm"
        report["arm_family"] = name
        return model, handles, report
    if name in ("full-distill", INTERPOLATED_ARM):
        if not args.full_checkpoint:
            raise SystemExit("--full-checkpoint is required for full-distill")
        payload = torch.load(args.full_checkpoint, map_location="cpu",
                             weights_only=False)
        if payload.get("arm") not in (None, "full"):
            raise SystemExit(f"{args.full_checkpoint}: not a full-arm checkpoint")
        interpolation = None
        if name == INTERPOLATED_ARM:
            interpolation = payload.get("interpolation") or {}
            if "coefficient" not in interpolation:
                raise SystemExit(
                    f"{args.full_checkpoint}: the interpolated arm needs an "
                    "interpolation block with a coefficient")
        arm = full.load_full_state_into(model, payload["arm_state"])
        report = arm.active_parameter_report()
        report.update({"checkpoint": args.full_checkpoint,
                       "checkpoint_sha256": sha256_file(args.full_checkpoint),
                       "checkpoint_step": payload.get("step")})
        if interpolation is not None:
            report["interpolation"] = interpolation
        return model, [], report
    raise SystemExit(f"unknown arm {name!r}")


def load_backbone(args, device: str, attention: str | None = None):
    if args.fake:
        from frontier import fake
        torch.manual_seed(0)
        return fake.TinyLM(args.vocab, args.hidden, args.layers).to(device)
    return arms.load_model(args.student_dir, dtype=torch.bfloat16,
                           device=device, attn_implementation=attention)


def load_teacher(args, device: str):
    if args.fake:
        from frontier import fake
        torch.manual_seed(1)
        return fake.TinyLM(args.vocab, args.hidden, args.layers).to(device)
    return arms.load_model(args.teacher_dir, dtype=torch.bfloat16, device=device)


def registration_block(args) -> dict | None:
    """Name the registration this comparison acts under, if one was given.

    `prereg_provenance` reads this key to order a registration against its first
    score; the anchor instrument stays under `harness`.
    """
    path = getattr(args, "registration", None)
    if not path:
        return None
    return {"path": path, "sha256": sha256_file(path)}


def selected_arms(args) -> list[str]:
    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    if not names:
        raise SystemExit("--arms selected no arm")
    unknown = sorted(set(names) - set(SCORED_ARMS))
    if unknown:
        raise SystemExit(f"--arms has unknown arm(s) {unknown}; known {list(SCORED_ARMS)}")
    if len(set(names)) != len(names):
        raise SystemExit(f"--arms repeats an arm: {names}")
    if args.reference_arm not in names:
        raise SystemExit(f"--reference-arm {args.reference_arm!r} is not in --arms")
    return names


def score_all(args, device: str, names: list[str]):
    manifest, tokens = read_manifest(args.heldout_manifest)
    tokens = torch.from_numpy(tokens.astype(np.int64))
    teacher = load_teacher(args, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    scored = {}
    reports = {}
    paths = {}
    for name in names:
        print(f"compare: {name}", flush=True)
        model = load_backbone(args, device, attention="eager")
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
               CONTROL_ARM: args.control_checkpoint,
               "memory-2m": args.memory_checkpoint,
               "memory-ungrafted": None,
               "memory-grafted": args.memory_init,
               "full-distill": args.full_checkpoint,
               "base": None}
    for name in CORRECTIVE_ARMS + SCREEN_CORRECTIVE_ARMS:
        mapping[name] = args.memory_checkpoint
    for name in SCREEN_MATCHED_CONTROLS:
        mapping[name] = (args.matched_adapter_checkpoint
                         if name == "adapter-2m-math-matched"
                         else args.gated_dense_checkpoint)
    path = mapping.get(name)
    return {"path": path, "sha256": sha256_file(path) if path else None}


def harness_section(directory: str | None, names: list[str],
                    reference_arm: str) -> dict:
    if not directory:
        return {"status": "not_scored",
                "reason": "no --harness-dir was given; no arm has an anchored receipt"}
    per_arm: dict[str, dict[str, dict]] = {}
    missing = []
    for name in names:
        path = os.path.join(directory, name, "eval.json")
        if not os.path.exists(path):
            missing.append(name)
            continue
        payload = json.load(open(path, encoding="utf-8"))
        tasks = {row["task"]: {"score": row["score"], "stderr": row.get("stderr")}
                 for row in payload.get("tasks", [])}
        per_arm[name] = tasks
    families = compare.family_means(per_arm, FAMILIES)
    reference = families.get(reference_arm, {})
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
            "reference_arm": reference_arm}


def table_rows(scored: dict, names: list[str], reference_arm: str) -> list[dict]:
    reference = scored[reference_arm]
    rows = []
    for name in names:
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
    names = selected_arms(args)
    manifest, scored, reports, paths = score_all(args, device, names)
    rows = table_rows(scored, names, args.reference_arm)
    text = compare.format_table(rows, args.reference_arm)
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
            "tokens_scored": int(scored[args.reference_arm]["tokens_scored"]),
            "seq_len": args.seq_len,
            "chunks": scored[args.reference_arm]["chunks"],
            "chunking": "non-overlapping next-token chunks, partial tail dropped",
        },
        "temperature": args.temperature,
        "reference_arm": args.reference_arm,
        "selected_arms": names,
        "preregistration": registration_block(args),
        "arms": {
            name: {
                **compare.arm_summary(scored[name], scored[args.reference_arm]),
                "report": reports[name],
                "checkpoint": paths[name],
            }
            for name in names
        },
        "table": text,
        "harness": harness_section(args.harness_dir, names, args.reference_arm),
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
        control_checkpoint=None,
        memory_init=paths["memory_init"],
        memory_checkpoint=paths["memory_trained"],
        full_checkpoint=paths["full"],
        heldout_manifest=paths["stream_manifest"],
        seq_len=paths["seq_len"], temperature=2.0, vocab_chunk=4,
        site=2, harness_dir=None, arms=",".join(ARMS),
        reference_arm=REFERENCE_ARM, registration=None,
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
    # a selected subset with a non-default reference must score only what it
    # names and pair every delta against that reference
    subset = SimpleNamespace(**vars(args))
    subset.arms = "base,adapter-2m"
    subset.reference_arm = "base"
    subset.out = os.path.join(root, "compare-subset.json")
    run(subset)
    sub = json.load(open(subset.out, encoding="utf-8"))
    assert sub["selected_arms"] == ["base", "adapter-2m"], sub["selected_arms"]
    assert set(sub["arms"]) == {"base", "adapter-2m"}, sub["arms"].keys()
    assert sub["reference_arm"] == "base"
    assert sub["arms"]["base"]["delta_ce_vs_reference"]["mean"] == 0.0
    try:
        bad = SimpleNamespace(**vars(args))
        bad.arms = "base,memory-2m"
        bad.reference_arm = "adapter-2m"
        selected_arms(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("selected_arms accepted a reference outside --arms")
    print(f"selftest: six arms scored, ungrafted == base, subset + reference "
          f"selection pass, receipts {root}", flush=True)
    return code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma-separated arms to score; default is the "
                             "registered six-arm comparison")
    parser.add_argument("--reference-arm", default=REFERENCE_ARM,
                        help="arm the paired deltas are against; must be selected")
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--student-dir", default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-dir", default=DEFAULT_TEACHER)
    parser.add_argument("--heldout-manifest", default=None)
    parser.add_argument("--adapter-checkpoint", default=None)
    parser.add_argument("--control-checkpoint", default=None,
                        help="S39 20M-token adapter checkpoint for adapter-20m")
    parser.add_argument("--matched-adapter-checkpoint", default=None,
                        help="S47 matched dense adapter on the 20M math stream")
    parser.add_argument("--gated-dense-checkpoint", default=None,
                        help="S47 identity-gated dense residual on the 20M math stream")
    parser.add_argument("--memory-init", default=None)
    parser.add_argument("--memory-checkpoint", default=None)
    parser.add_argument("--full-checkpoint", default=None)
    parser.add_argument("--harness-dir", default=None)
    parser.add_argument("--registration", default=None,
                        help="registration this comparison acts under; "
                             "recorded as `preregistration` for provenance")
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
