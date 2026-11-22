#!/usr/bin/env python3
"""S71: generation diagnostics for already-scored calibration checkpoints.

The S69 calibration scorer keeps accuracy rows but not token ids, so it cannot
say whether a sequence ended with EOS, was cut by the registered loop rule, or
ran into the 2048-token cap. This probe regenerates a prefix of each calibration
split with the identical greedy protocol and batch composition, records the
generated ids, and reports:

* generated tokens per task and per checkpoint,
* the fraction of sequences that hit the 2048-token cap,
* the fraction that emitted EOS,
* the fraction the registered loop rule matches, and where it cuts, using
  ``s59_student_rollouts.loop_cut_index`` (imported, never reimplemented).

The probe is diagnostic. It does not change a score, select anything, or read a
test split. It fails closed unless every sampled problem reproduces the stored
``flexible_correct`` value from the official receipt: the ids are regenerated,
so the accuracy they imply must equal the accuracy that was stored.

Usage
-----
    SELVEDGE_MEM=6g scripts/court --gpu eval -- python3 \
      scripts/s71_generation_diagnostics.py \
        --arm matched-sft \
        --checkpoints 500000=results/.../ckpt-000742.pt,1000000=... \
        --official results/stage2/s69-calibration/matched-sft.json \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
        --out results/stage2/s71-overshoot/generation-diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, HERE)

from frontier import rft  # noqa: E402
from s54_post_trained_eval import (  # noqa: E402
    _decode,
    cgroup_memory,
    extract_flexible,
    load_tokenizer,
    render_prompt,
    revision,
    sha256_file,
    stamp,
    utcnow,
    write_json,
)
from s59_student_rollouts import loop_cut_index  # noqa: E402

SCHEMA = "selvedge/s71-generation-diagnostics/v1"
CALIBRATION_REGISTRATION = "docs/s66-calibration-splits-v1.json"
BOOTSTRAP_NOTE = (
    "loop status is read post hoc from the generated ids with the registered "
    "helper; this probe is not the training-time sampler and cuts nothing")


def load_split(task: str) -> list[dict]:
    registration = json.load(open(
        os.path.join(ROOT, CALIBRATION_REGISTRATION), encoding="utf-8"))
    key = "gsm8k_cal" if task == "gsm8k" else "math_cal"
    path = os.path.join(ROOT, registration["splits"][key]["path"])
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_pairs(text: str) -> list[tuple[int, str]]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"s71-diag: {part!r} is not cadence=path")
        cadence_text, path = part.split("=", 1)
        values.append((int(cadence_text), path.strip()))
    if not values:
        raise SystemExit("s71-diag: no checkpoints given")
    return sorted(values)


def generate_greedy(model, tokenizer, prompts: list[str], *,
                    max_new_tokens: int, device: str) -> dict:
    """Greedy generation with the S54 batch kwargs, returning raw ids.

    The kwargs are the ones ``s54_post_trained_eval.generate_batch`` uses on its
    greedy path; only the return value differs (ids instead of decoded text).
    """
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = int(input_ids.shape[1])
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    wall = time.time() - started
    generated = output[:, prompt_length:]
    pad = tokenizer.pad_token_id
    ids_rows = []
    for row in generated:
        ids = row.tolist()
        while ids and ids[-1] == pad:
            ids.pop()
        ids_rows.append(ids)
    return {
        "ids": ids_rows,
        "statistics": {
            "prompt_tokens": int(attention_mask.sum().item()),
            "wall_seconds": wall,
            "batch_size": len(prompts),
        },
        "decoded": [_decode(tokenizer, row) for row in generated],
    }


def summarise(rows: list[dict], max_new_tokens: int) -> dict:
    n = len(rows)
    if not n:
        return {"sequences": 0}
    lengths = [row["new_tokens"] for row in rows]
    cuts = [row["loop_cut_index"] for row in rows
            if row["loop_cut_index"] is not None]
    return {
        "sequences": n,
        "generated_tokens": int(sum(lengths)),
        "mean_new_tokens": sum(lengths) / n,
        "min_new_tokens": min(lengths),
        "max_new_tokens": max(lengths),
        "hit_cap": sum(1 for row in rows if row["hit_cap"]),
        "hit_cap_fraction": sum(1 for row in rows if row["hit_cap"]) / n,
        "emitted_eos": sum(1 for row in rows if row["emitted_eos"]),
        "emitted_eos_fraction": sum(1 for row in rows if row["emitted_eos"]) / n,
        "loop_cut": len(cuts),
        "loop_cut_fraction": len(cuts) / n,
        "loop_cut_index_min": min(cuts) if cuts else None,
        "loop_cut_index_max": max(cuts) if cuts else None,
        "cap_tokens": max_new_tokens,
        "basis": BOOTSTRAP_NOTE,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--official", required=True,
                        help="the s69 calibration receipt for this arm")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--items-per-task", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    checkpoints = parse_pairs(args.checkpoints)
    official = json.load(open(os.path.join(ROOT, args.official),
                              encoding="utf-8"))
    official_rows = {int(row["cadence"]): row for row in official["rows"]}
    for cadence, _ in checkpoints:
        if cadence not in official_rows:
            raise SystemExit(f"s71-diag: {args.official} has no cadence "
                             f"{cadence} row to check parity against")

    out_path = os.path.join(ROOT, args.out)
    payload = {
        "schema": SCHEMA,
        "kind": "s71-generation-diagnostics",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "label": "diagnostic",
        "test_read": False,
        "preregistration": {
            "path": "docs/s71-overshoot-probe-preregistration-v1.json",
            "sha256": sha256_file(os.path.join(
                ROOT, "docs/s71-overshoot-probe-preregistration-v1.json")),
        },
        "instrument": {
            "model_dir": args.model_dir,
            "official_receipt": args.official,
            "official_receipt_sha256": sha256_file(
                os.path.join(ROOT, args.official)),
            "items_per_task": args.items_per_task,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy non-thinking",
            "attention": "sdpa",
            "parity_requirement": ("every sampled problem's flexible_correct "
                                   "must equal the stored row's value"),
            "loop_helper": ("scripts/s59_student_rollouts.py::loop_cut_index, "
                            "imported"),
        },
        "runs": [],
        "claims": {
            "scores_not_changed": True,
            "subsample_not_a_score": True,
            "no_test_split_read": True,
        },
    }
    if args.resume and os.path.exists(out_path):
        previous = json.load(open(out_path, encoding="utf-8"))
        payload["runs"] = list(previous.get("runs") or [])
        payload["created_utc"] = previous.get("created_utc", payload["created_utc"])
    done = {(run["arm"], int(run["cadence"])) for run in payload["runs"]}
    pending = [(cadence, path) for cadence, path in checkpoints
               if (args.arm, cadence) not in done]
    if pending:
        import torch
        from transformers import AutoModelForCausalLM
        from s60_interpolate import _load_into, torch_load

        if not torch.cuda.is_available():
            raise SystemExit("s71-diag: CUDA required; run through "
                             "scripts/court --gpu eval")
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa").to(device)
        model.eval()
        tokenizer = load_tokenizer(args.model_dir)
        splits = {
            "gsm8k": load_split("gsm8k")[:args.items_per_task],
            "math500": load_split("math500")[:args.items_per_task],
        }
        for cadence, checkpoint in pending:
            path = os.path.join(ROOT, checkpoint)
            stored_payload = torch_load(path)
            if stored_payload.get("arm") != "full":
                raise SystemExit(f"s71-diag: {checkpoint} is not a full arm")
            _load_into(model, stored_payload["arm_state"])
            model.eval()
            official_row = official_rows[cadence]
            tasks = []
            for task, items in splits.items():
                by_id = {row["problem_id"]: row
                         for row in official_row[f"{task if task == 'gsm8k' else 'math'}_per_problem"]}
                sequences = []
                parity = []
                for index in range(0, len(items), args.batch_size):
                    chunk = items[index:index + args.batch_size]
                    prompts = [render_prompt(tokenizer, item["question"], False)
                               for item in chunk]
                    generated = generate_greedy(
                        model, tokenizer, prompts,
                        max_new_tokens=args.max_new_tokens, device=device)
                    for offset, item in enumerate(chunk):
                        ids = generated["ids"][offset]
                        text = generated["decoded"][offset]
                        extracted = extract_flexible(text, task)
                        correct = bool(
                            extracted["value"] is not None
                            and rft.answers_equal(extracted["value"],
                                                  item["gold"]))
                        stored = by_id.get(item["problem_id"])
                        parity.append({
                            "problem_id": item["problem_id"],
                            "flexible_correct": correct,
                            "stored_flexible_correct": (
                                None if stored is None
                                else bool(stored["flexible_correct"])),
                            "match": (stored is not None
                                      and bool(stored["flexible_correct"]) == correct),
                        })
                        eos = int(tokenizer.eos_token_id)
                        sequences.append({
                            "problem_id": item["problem_id"],
                            "new_tokens": len(ids),
                            "hit_cap": (len(ids) >= args.max_new_tokens
                                        and (not ids or ids[-1] != eos)),
                            "emitted_eos": bool(ids) and ids[-1] == eos,
                            "loop_cut_index": loop_cut_index(ids),
                        })
                mismatches = [row for row in parity if not row["match"]]
                if mismatches:
                    raise SystemExit(
                        "s71-diag: parity failed for "
                        f"{len(mismatches)}/{len(parity)} sampled problems at "
                        f"cadence {cadence} task {task}; refusing to report "
                        f"diagnostics. First: {mismatches[0]}")
                tasks.append({
                    "task": task,
                    "items": len(sequences),
                    "summary": summarise(sequences, args.max_new_tokens),
                    "parity": {
                        "compared": len(parity),
                        "matched": len(parity) - len(mismatches),
                        "all_match": True,
                    },
                })
                print(f"s71-diag: cadence={cadence} task={task} "
                      f"loop={tasks[-1]['summary']['loop_cut_fraction']:.3f} "
                      f"cap={tasks[-1]['summary']['hit_cap_fraction']:.3f} "
                      f"eos={tasks[-1]['summary']['emitted_eos_fraction']:.3f}",
                      flush=True)
            payload["runs"].append({
                "arm": args.arm,
                "cadence": cadence,
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_step": stored_payload.get("step"),
                "checkpoint_processed_tokens": stored_payload.get(
                    "processed_tokens"),
                "tasks": tasks,
            })
            payload["cgroup"] = cgroup_memory()
            write_json(out_path, payload)
    write_json(out_path, payload)
    print(f"s71-diag: wrote {len(payload['runs'])} run(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
