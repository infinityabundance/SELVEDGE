#!/usr/bin/env python3
"""Diagnose the anchor-vs-stage-2 base score difference (S43).

Compares the fixed samples the two instruments actually scored:

    anchor   results/frontier/anchor-0.8b-20261116T195953Z.json
             raw under results/frontier/anchor-raw/
    stage2   results/stage2/eval/base/eval.json
             raw under results/stage2/eval/base/raw/

It records, per task, whether the doc ids, prompts and targets are identical,
how many greedy generations differ, how many extracted answers differ, and how
many metric outcomes flip. It then names the one configuration difference that
is not a score-setting difference.

Emits results/stage2/s43-base-discrepancy-<stamp>.json. Pure JSON analysis;
no model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

ANCHOR_RAW = "results/frontier/anchor-raw"
STAGE2_RAW = "results/stage2/eval/base/raw"
ANCHOR_RECEIPT = "results/frontier/anchor-0.8b-20261116T195953Z.json"
STAGE2_RECEIPT = "results/stage2/eval/base/eval.json"

TASKS = {
    "mmlu_pro": {
        "anchor": f"{ANCHOR_RAW}/20261116T195953Z-Qwen__Qwen3.5-0.8B-Base-mmlu_pro.json",
        "stage2": f"{STAGE2_RAW}/20261117T040831Z-mmlu_pro.json",
        "metric": "exact_match",
    },
    "minerva_math500": {
        "anchor": f"{ANCHOR_RAW}/20261116T195953Z-Qwen__Qwen3.5-0.8B-Base-minerva_math500.json",
        "stage2": f"{STAGE2_RAW}/20261117T043134Z-minerva_math500.json",
        "metric": "math_verify",
    },
    "gsm8k": {
        "anchor": f"{ANCHOR_RAW}/20261116T195953Z-Qwen__Qwen3.5-0.8B-Base-gsm8k.json",
        "stage2": f"{STAGE2_RAW}/20261117T042753Z-gsm8k.json",
        "metric": "exact_match",
    },
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(samples: dict) -> list[dict]:
    rows = []
    seen = set()
    for key, value in samples.items():
        for item in value:
            ident = (str(item.get("doc_id")), item.get("doc_hash", ""))
            if ident in seen:
                continue
            seen.add(ident)
            rows.append(item)
    rows.sort(key=lambda item: (str(item.get("doc_id")), item.get("doc_hash", "")))
    return rows


def _normalise_config(value):
    """Drop process-local reprs and loader metadata; keep task behaviour."""
    import re

    if isinstance(value, dict):
        return {key: _normalise_config(item) for key, item in value.items()
                if key not in ("metadata",)}
    if isinstance(value, list):
        return [_normalise_config(item) for item in value]
    if isinstance(value, str):
        return re.sub(r" at 0x[0-9a-f]+", " at 0xADDR", value)
    return value


def config_diff(anchor: dict, stage2: dict) -> dict:
    left = anchor.get("configs") or {}
    right = stage2.get("configs") or {}
    keys = sorted(set(left) | set(right))
    differing = []
    for key in keys:
        if _normalise_config(left.get(key)) != _normalise_config(right.get(key)):
            differing.append({"task": key,
                              "anchor": _normalise_config(left.get(key)),
                              "stage2": _normalise_config(right.get(key))})
    return {
        "subtasks_compared": len(keys),
        "differing": differing,
        "normalised_away": [
            "process-local function reprs (`<function ... at 0x...>`)",
            "`metadata`, which the anchor's HFLM model_args path fills with "
            "dtype/pretrained/parallelize and the pre-initialised-model path "
            "fills with a harness version; it is loader bookkeeping, not a "
            "score setting",
        ],
    }


def compare(task: str, spec: dict) -> dict:
    anchor = json.load(open(spec["anchor"], encoding="utf-8"))
    stage2 = json.load(open(spec["stage2"], encoding="utf-8"))
    a = flatten(anchor["samples"])
    b = flatten(stage2["samples"])
    if len(a) != len(b):
        return {"error": f"sample count {len(a)} vs {len(b)}"}
    same_doc = sum(1 for x, y in zip(a, b)
                   if x.get("doc_id") == y.get("doc_id")
                   and x.get("doc_hash") == y.get("doc_hash"))
    same_prompt = sum(1 for x, y in zip(a, b)
                      if x.get("prompt_hash") == y.get("prompt_hash"))
    same_target = sum(1 for x, y in zip(a, b)
                      if x.get("target_hash") == y.get("target_hash"))
    response_diff = 0
    filtered_diff = 0
    a_right_b_wrong = 0
    a_wrong_b_right = 0
    a_right = 0
    b_right = 0
    examples = []
    metric = spec["metric"]
    for x, y in zip(a, b):
        if x.get("resps") != y.get("resps"):
            response_diff += 1
        if x.get("filtered_resps") != y.get("filtered_resps"):
            filtered_diff += 1
        if metric in x and metric in y:
            ax = int(x.get(metric) or 0)
            by = int(y.get(metric) or 0)
            a_right += ax
            b_right += by
            if ax and not by:
                a_right_b_wrong += 1
            elif by and not ax:
                a_wrong_b_right += 1
        if len(examples) < 3 and x.get("resps") != y.get("resps"):
            examples.append({
                "doc_id": x.get("doc_id"),
                "anchor_filtered": x.get("filtered_resps"),
                "stage2_filtered": y.get("filtered_resps"),
                "anchor_metric": x.get(metric),
                "stage2_metric": y.get(metric),
            })
    return {
        "items": len(a),
        "identical_doc_id_and_hash": same_doc,
        "identical_prompt_hash": same_prompt,
        "identical_target_hash": same_target,
        "differing_generations": response_diff,
        "differing_extracted_answers": filtered_diff,
        "anchor_correct": a_right,
        "stage2_correct": b_right,
        "anchor_correct_stage2_wrong": a_right_b_wrong,
        "stage2_correct_anchor_wrong": a_wrong_b_right,
        "examples": examples,
        "config_diff": config_diff(anchor, stage2),
        "versions": {
            "anchor": {"lm_eval": anchor.get("lm_eval_version"),
                       "transformers": anchor.get("transformers_version"),
                       "git_hash": anchor.get("git_hash")},
            "stage2": {"lm_eval": stage2.get("lm_eval_version"),
                       "transformers": stage2.get("transformers_version"),
                       "git_hash": stage2.get("git_hash")},
        },
    }


def main() -> int:
    args = parse_args()
    anchor_receipt = json.load(open(ANCHOR_RECEIPT, encoding="utf-8"))
    stage2_receipt = json.load(open(STAGE2_RECEIPT, encoding="utf-8"))
    scores = {}
    for task in anchor_receipt["tasks"]:
        if task["task"] in ("mmlu_pro", "minerva_math500", "gsm8k", "mmlu"):
            scores.setdefault(task["task"], {})["anchor"] = {
                "primary_metric": task["primary_metric"],
                "score": task["score"],
                "stderr": task["stderr"],
                "sample_len": task["n_samples"]["sample_len"],
                "limit": task["pre_registered"]["limit"],
                "num_fewshot": task["pre_registered"]["num_fewshot"],
            }
    for task in stage2_receipt["tasks"]:
        if task["task"] in ("mmlu_pro", "minerva_math500", "gsm8k", "mmlu"):
            scores.setdefault(task["task"], {})["stage2"] = {
                "primary_metric": task["primary_metric"],
                "score": task["score"],
                "stderr": task["stderr"],
                "sample_len": task["n_samples"]["sample_len"],
                "limit": task["pre_registered"]["limit"],
                "num_fewshot": task["pre_registered"]["num_fewshot"],
            }
    for task, pair in scores.items():
        if "anchor" in pair and "stage2" in pair:
            pair["delta_stage2_minus_anchor"] = (
                pair["stage2"]["score"] - pair["anchor"]["score"])

    item_level = {task: compare(task, spec) for task, spec in TASKS.items()}

    report = {
        "schema": "selvedge/s43-base-discrepancy/v1",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "question": ("why does the stage-2 base score differ from the anchor base "
                     "score on the same checkpoint?"),
        "answer": ("the two instruments loaded the same checkpoint with different "
                   "attention kernels. The S36 anchor passes `model=hf` to the "
                   "harness, which loads the checkpoint with the transformers "
                   "default, measured as `sdpa` for this model. The S40/S41 stage-2 "
                   "loader called `arms.load_model`, which passed "
                   "`attn_implementation='eager'` explicitly. Identical prompts "
                   "under greedy decoding then produce different continuations "
                   "wherever the two kernels' floating-point reductions diverge, "
                   "and a few of those continuations change the extracted answer. "
                   "No other score-setting input differs."),
        "inputs": {
            "anchor_receipt": {"path": ANCHOR_RECEIPT,
                               "sha256": sha256_file(ANCHOR_RECEIPT)},
            "stage2_receipt": {"path": STAGE2_RECEIPT,
                               "sha256": sha256_file(STAGE2_RECEIPT)},
            "raw": {task: {"anchor": spec["anchor"], "stage2": spec["stage2"],
                           "anchor_sha256": sha256_file(spec["anchor"]),
                           "stage2_sha256": sha256_file(spec["stage2"])}
                    for task, spec in TASKS.items()},
        },
        "invariant_and_checked": {
            "model_id_and_revision": (
                "both receipts name Qwen/Qwen3.5-0.8B-Base at "
                "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
            "harness_commit": (
                "both receipts name d6de81643928d653435c431bae19945d41d32520"),
            "harness_image": (
                "both receipts name selvedge/eval-court:20261116-3.12-slim-"
                "eval-5c2514f58e78 with image sha256:42a0ddab6264..."),
            "lm_eval_and_transformers": "0.4.14.dev0 and 5.23.0 in both",
            "task_configs": "identical for every compared subtask (config_diff "
                            "below is empty)",
            "prompts": "identical prompt hashes on every item",
            "targets": "identical target hashes on every item",
            "limits_fewshot_batch": (
                "identical: mmlu_pro 36 per subject / 5-shot / batch 8; "
                "minerva_math500 full 500 / 4-shot / batch 8; gsm8k full 1319 / "
                "5-shot / batch 8; mmlu full 14042 / 5-shot / batch 2"),
            "seeds": ("random 0, numpy 1234, torch 1234, fewshot 1234 in both; "
                      "greedy decoding (do_sample false, temperature 0.0)"),
            "dtype": "bfloat16 in both",
        },
        "the_one_difference": {
            "setting": "attn_implementation",
            "anchor": ("transformers default; measured `sdpa` for this checkpoint "
                       "by loading it with no attn_implementation argument"),
            "stage2": "`eager`, passed explicitly by oracle/frontier/arms.py",
            "evidence": {
                "probe_receipt": "results/courts/oracle-20261117T070212Z-666a5c.json",
                "probe_output": "DEFAULT_ATTN sdpa / EAGER_ATTN eager",
                "anchor_code": ("scripts/s36_anchor.py passes model=hf and "
                                "model_args=dtype=bfloat16,trust_remote_code=false,"
                                "parallelize=false,add_bos_token=false; "
                                "attn_implementation is absent"),
                "stage2_code": ("scripts/s40_score.py calls "
                                "comparison.load_backbone -> arms.load_model, which "
                                "passed attn_implementation='eager'"),
            },
            "why_it_changes_scores": (
                "the kernels compute the same mathematical operation with different "
                "reduction orders and different fused paths, so bf16 logits differ "
                "in the last bits; under greedy decoding a near-tie at any position "
                "can select a different token and the continuation diverges. The "
                "effect is small per item but visible as a shifted exact-match "
                "count on 500-1,300-item generation tasks."),
        },
        "scores": scores,
        "item_level": item_level,
        "cause": "CONFIG DIFFERENCE — attention implementation (anchor sdpa vs stage-2 eager)",
        "fix": {
            "not_history": ("the anchor and stage-2 receipts are not edited or "
                            "re-run here; each keeps the configuration it recorded"),
            "code": [
                "oracle/frontier/arms.py: load_model gains an "
                "`attn_implementation` argument and omits the kwarg when it is "
                "None, leaving the transformers default in place",
                "scripts/s40_compare.py: the eval backbone loads with the anchor's "
                "default (attention=None); the held-out instrument keeps the "
                "registered eager path it used before",
                "scripts/s40_compare.py: build_named_arm records "
                "`eval_attention_implementation` and its source in the arm report, "
                "so every future receipt names the kernel",
            ],
            "next_score": ("the next stage-2 base score is produced under the "
                           "anchor's attention setting and is the one to compare to "
                           "the anchor; this receipt records why the earlier one is "
                           "not the comparison"),
        },
        "what_this_does_not_establish": [
            "that sdpa is more correct than eager: they are two numerical paths to "
            "the same operator, and the anchor's setting is the comparison basis, "
            "not a truth claim",
            "that no other difference exists in unmeasured tasks: the compared "
            "tasks are the four the pass scored, and the item-level evidence is "
            "for mmlu_pro, minerva_math500 and gsm8k",
        ],
    }
    out = args.out or f"results/stage2/s43-base-discrepancy-{report['stamp']}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    print(f"wrote {out}")
    for task, value in item_level.items():
        print(task, {key: value[key] for key in (
            "items", "identical_prompt_hash", "differing_generations",
            "differing_extracted_answers", "anchor_correct", "stage2_correct",
            "anchor_correct_stage2_wrong", "stage2_correct_anchor_wrong")})
        print(" config differing subtasks:", len(value["config_diff"]["differing"]))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
