#!/usr/bin/env python3
"""S55 stage 2a: held-out solution cross-entropy for every saved checkpoint.

The stage-2a registration stops at the held-out-solution-CE minimum. This
script measures that curve: for each `ckpt-*.pt` of a run it restores the arm
state on top of the pinned post-trained student and computes the mean
cross-entropy over **supervised** positions only (prompt and padding are
`-100`), per dataset, on the held-out slice written by
`scripts/s55_stage2a_examples.py`.

It also reports the same CE for the base (step-0) student, so the delta is a
paired number rather than an absolute one.

Usage
-----
    SELVEDGE_MEM=6g scripts/court --gpu oracle -- python3 \
        scripts/s55_heldout_ce.py \
        --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc0636... \
        --holdout results/artifacts/stage2/s55-stage2a/stage2a-holdout.jsonl \
        --run-dir results/stage2/runs/s55-stage2a \
        --seq-len 768 --out results/stage2/s55-stage2a-heldout-ce.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import arms, full, sft_data  # noqa: E402

SCHEMA = "selvedge/s55-stage2a-heldout-ce/v1"
GIB = float(1 << 30)


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
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def evaluate(model, inputs, targets, groups, chunk: int = 32) -> dict:
    """Sum CE over supervised positions, split by dataset group."""
    import torch
    import torch.nn.functional as F

    totals: dict[str, dict] = {}
    with torch.no_grad():
        logits = model(input_ids=inputs, use_cache=False).logits
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_groups = []
        for index, group in enumerate(groups):
            flat_groups.extend([group] * targets.shape[1])
        for start in range(0, flat_logits.shape[0], chunk):
            s = flat_logits[start:start + chunk].float()
            t = flat_targets[start:start + chunk]
            supervised = (t != -100)
            if not bool(supervised.any()):
                continue
            losses = F.cross_entropy(s, t, reduction="none", ignore_index=-100)
            for offset in range(s.shape[0]):
                if not bool(supervised[offset]):
                    continue
                group = flat_groups[start + offset]
                entry = totals.setdefault(group, {"sum": 0.0, "positions": 0})
                entry["sum"] += float(losses[offset])
                entry["positions"] += 1
    return totals


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="explicit checkpoint paths; default every ckpt-*.pt")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--vram-limit-gib", type=float, default=12.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("s55-heldout: CUDA required; run through scripts/court --gpu")
    free_bytes, _total = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free_bytes:
        raise SystemExit(f"s55-heldout: vram limit {args.vram_limit_gib} GiB "
                         f"exceeds free {free_bytes / GIB:.2f} GiB")
    device = "cuda"
    torch.use_deterministic_algorithms(True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student_dir,
                                              local_files_only=True,
                                              trust_remote_code=False)
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                 else eos_id)
    examples = sft_data.load_examples(os.path.join(ROOT, args.holdout),
                                      vocab_size=len(tokenizer))
    packed = sft_data.pack_examples(examples, args.seq_len, pad_id, eos_id,
                                    examples_per_row=1, overflow="drop")
    groups = [example.get("dataset") or "unknown" for example in examples]

    model = arms.load_model(args.student_dir, dtype=torch.bfloat16, device=device)
    arm = full.FullBackboneArm(model, freeze_embedding=True)
    arm.eval()
    torch.cuda.reset_peak_memory_stats()

    checkpoints = args.checkpoints
    if not checkpoints:
        checkpoints = sorted(glob.glob(os.path.join(ROOT, args.run_dir, "ckpt-*.pt")))
        if not checkpoints:
            raise SystemExit("s55-heldout: no checkpoints in the run directory")

    results = []

    def measure(label: str, payload_state: dict | None) -> dict:
        if payload_state is None:
            # The base arm is the post-trained checkpoint as loaded.
            pass
        else:
            arm.load_state_dict(payload_state)
        total_sum = 0.0
        total_positions = 0
        per_group: dict[str, dict] = {}
        for start in range(0, packed.input_ids.shape[0], args.micro_batch):
            stop = start + args.micro_batch
            inputs = packed.input_ids[start:stop, :-1].to(device)
            targets = packed.labels[start:stop, 1:].to(device)
            batch_groups = groups[start:stop]
            totals = evaluate(model, inputs, targets, batch_groups)
            for group, entry in totals.items():
                target = per_group.setdefault(group, {"sum": 0.0, "positions": 0})
                target["sum"] += entry["sum"]
                target["positions"] += entry["positions"]
                total_sum += entry["sum"]
                total_positions += entry["positions"]
        return {
            "label": label,
            "ce": (total_sum / total_positions) if total_positions else None,
            "positions": total_positions,
            "per_dataset": {group: {
                "ce": entry["sum"] / entry["positions"],
                "positions": entry["positions"]}
                for group, entry in sorted(per_group.items())},
        }

    started = time.time()
    results.append(measure("base", None))
    print(f"s55-heldout: base ce={results[-1]['ce']:.5f}", flush=True)
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["arm_state"]
        # Restore the base first, then the checkpoint, so a checkpoint can never
        # inherit values from the previous arm state.
        base_model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                                     device=device)
        arm = full.FullBackboneArm(base_model, freeze_embedding=True)
        arm.eval()
        model = base_model
        result = measure(os.path.basename(path), state)
        result["step"] = int(payload.get("step") or 0)
        result["tokens_seen"] = int(payload.get("tokens_seen") or 0)
        result["sha256_path"] = os.path.relpath(path, ROOT)
        results.append(result)
        print(f"s55-heldout: {result['label']} step={result['step']} "
              f"ce={result['ce']:.5f}", flush=True)
        del base_model, arm
        torch.cuda.empty_cache()

    chosen = min((entry for entry in results if entry["label"] != "base"),
                 key=lambda entry: entry["ce"])
    payload = {
        "schema": SCHEMA,
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "student_dir": args.student_dir,
        "holdout": os.path.relpath(os.path.join(ROOT, args.holdout), ROOT),
        "seq_len": args.seq_len,
        "packing": packed.report,
        "results": results,
        "selected": {
            "checkpoint": chosen["sha256_path"],
            "step": chosen["step"],
            "ce": chosen["ce"],
            "base_ce": results[0]["ce"],
            "delta": chosen["ce"] - results[0]["ce"],
            "rule": "minimum held-out supervised-position CE across saved "
                    "checkpoints",
        },
        "elapsed_seconds": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "command": sys.argv,
    }
    write_json(os.path.join(ROOT, args.out), payload)
    print(f"s55-heldout: selected {payload['selected']['checkpoint']} "
          f"ce={payload['selected']['ce']:.5f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
