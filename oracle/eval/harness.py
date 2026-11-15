#!/usr/bin/env python3
"""T1 evaluation harness: teacher KL/PPL, HumanEval pass@1, GSM8K EM and
TruthfulQA MC1, all pinned and deterministic.

Every reported number is produced here, from a pinned checkpoint and pinned
items. Greedy decoding for pass@1 and exact match; MC1 by length-normalised
log-likelihood; no sampling in the primary metrics. Per-window and per-item
values are written to the receipt so the paired bootstrap can be recomputed.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

import tasks
from metrics import bootstrap_mean, paired_bootstrap, stratified_bootstrap

MODEL_REVISIONS = {
    "0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
    "1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
KL_BATCH = 4
GEN_BATCH = 8
HUMANEVAL_MAX_NEW = 256
GSM8K_MAX_NEW = 512
HUMANEVAL_TIMEOUT = 3.0
HUMANEVAL_ADDRESS_SPACE = 1_500_000_000


def model_path(hf_root: str, size: str) -> str:
    directory = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B"}[size]
    path = os.path.join(hf_root, "models", "Qwen", directory,
                        MODEL_REVISIONS[size])
    if not os.path.isdir(path):
        raise SystemExit(f"model path {path} is absent")
    return path


def load_model(path: str, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return model


def load_tokenizer(hf_root: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path(hf_root, "0.6b"),
                                         local_files_only=True,
                                         trust_remote_code=False)


# ---------------------------------------------------------------------------
# distributional metrics


@torch.inference_mode()
def window_stats(student, teacher, tokens: list[int], device) -> dict:
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
        student_logits = student(input_ids=ids, use_cache=False).logits.float()
        teacher_logits = teacher(input_ids=ids, use_cache=False).logits.float()
    # Predict tokens 1..n-1.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    target = ids[:, 1:]
    logp_s = F.log_softmax(s, dim=-1)
    logp_t = F.log_softmax(t, dim=-1)
    p_t = logp_t.exp()
    kl = (p_t * (logp_t - logp_s)).sum(dim=-1)
    nll = -logp_s.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    top1 = (s.argmax(dim=-1) == target).float()
    return {
        "kl": kl.mean().item(),
        "ppl": float(torch.exp(nll.mean()).item()),
        "nll": nll.mean().item(),
        "top1": top1.mean().item(),
        "positions": int(target.numel()),
    }


def distributional(student, teacher, windows: dict[str, list[dict]], device) -> dict:
    result = {"domains": {}, "per_window": {}}
    for domain, items in windows.items():
        rows = []
        for item in items:
            stats = window_stats(student, teacher, item["tokens"], device)
            stats["doc_id"] = item["doc_id"]
            stats["window_index"] = item["window_index"]
            rows.append(stats)
        result["per_window"][domain] = rows
        # Cluster by document: average windows within a document first.
        by_doc: dict[str, list[dict]] = {}
        for row in rows:
            by_doc.setdefault(row["doc_id"], []).append(row)
        documents = []
        for doc_id in sorted(by_doc):
            group = by_doc[doc_id]
            documents.append({
                "doc_id": doc_id,
                "kl": float(np.mean([row["kl"] for row in group])),
                "ppl": float(np.mean([row["ppl"] for row in group])),
                "top1": float(np.mean([row["top1"] for row in group])),
                "windows": len(group),
            })
        result["domains"][domain] = {
            "documents": documents,
            "kl": bootstrap_mean(np.array([doc["kl"] for doc in documents])),
            "ppl": float(np.mean([doc["ppl"] for doc in documents])),
            "top1": float(np.mean([doc["top1"] for doc in documents])),
            "window_count": len(rows),
        }
    result["mixed"] = {
        "kl": stratified_bootstrap({domain: np.array(
            [doc["kl"] for doc in result["domains"][domain]["documents"]])
            for domain in result["domains"]}),
        "top1": stratified_bootstrap({domain: np.array(
            [doc["top1"] for doc in result["domains"][domain]["documents"]])
            for domain in result["domains"]}),
    }
    return result


# ---------------------------------------------------------------------------
# generation


def to_prompts(tokenizer, prompts: list[str]):
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048)
    return encoded


@torch.inference_mode()
def generate_texts(model, tokenizer, prompts: list[str], max_new_tokens: int,
                   batch: int, device, seed: int = 20261115) -> list[str]:
    torch.manual_seed(seed)
    outputs: list[str] = []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        encoded = to_prompts(tokenizer, chunk)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        if start % (batch * 10) == 0:
            print(f"eval: generated {min(start + batch, len(prompts))}/{len(prompts)}",
                  flush=True)
    return outputs


# ---------------------------------------------------------------------------
# HumanEval


def _limit_address_space() -> None:
    resource.setrlimit(resource.RLIMIT_AS,
                       (HUMANEVAL_ADDRESS_SPACE, HUMANEVAL_ADDRESS_SPACE))


def run_humaneval_program(program: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="selvedge-he-") as directory:
        path = os.path.join(directory, "candidate.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)
        environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0",
                       "HOME": directory, "TMPDIR": directory}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", path],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=HUMANEVAL_TIMEOUT, preexec_fn=_limit_address_space,
            )
            return {"ok": completed.returncode == 0, "returncode": completed.returncode,
                    "stderr": completed.stderr[-800:], "timeout": False}
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": None, "stderr": "timeout", "timeout": True}


def run_humaneval(model, tokenizer, items: list[dict], device) -> dict:
    prompts = [tasks.humaneval_prompt(item) for item in items]
    completions = generate_texts(model, tokenizer, prompts, HUMANEVAL_MAX_NEW, GEN_BATCH,
                                 device)
    rows = []
    for item, completion in zip(items, completions):
        body = tasks.extract_code(completion)
        program = (item["prompt"] + body + "\n" + item["test"] +
                   f"\ncheck({item['entry_point']})\n")
        outcome = run_humaneval_program(program)
        rows.append({"id": item["id"], "passed": bool(outcome["ok"]),
                     "timeout": outcome["timeout"], "returncode": outcome["returncode"],
                     "completion": body, "stderr": outcome["stderr"]})
    values = np.array([1.0 if row["passed"] else 0.0 for row in rows])
    return {
        "items": rows,
        "pass_at_1": float(values.mean()) if values.size else 0.0,
        "pass_at_1_ci": bootstrap_mean(values) if values.size else None,
        "timeouts": int(sum(1 for row in rows if row["timeout"])),
    }


# ---------------------------------------------------------------------------
# GSM8K


def normalise_number(text: str) -> str:
    text = text.strip().replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def run_gsm8k(model, tokenizer, items: list[dict], device) -> dict:
    prompts = [tasks.gsm8k_prompt(item) for item in items]
    completions = generate_texts(model, tokenizer, prompts, GSM8K_MAX_NEW, GEN_BATCH,
                                 device)
    rows = []
    for item, completion in zip(items, completions):
        predicted = tasks.extract_gsm8k(completion)
        correct = predicted is not None and normalise_number(predicted) == normalise_number(
            item["gold"])
        rows.append({"id": item["id"], "predicted": predicted, "gold": item["gold"],
                     "correct": bool(correct), "completion": completion})
    values = np.array([1.0 if row["correct"] else 0.0 for row in rows])
    return {
        "items": rows,
        "exact_match": float(values.mean()) if values.size else 0.0,
        "exact_match_ci": bootstrap_mean(values) if values.size else None,
    }


# ---------------------------------------------------------------------------
# TruthfulQA MC1


@torch.inference_mode()
def answer_loglikelihood(model, tokenizer, prompt: str, choice: str, device) -> float:
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    choice_ids = tokenizer(" " + choice, return_tensors="pt", add_special_tokens=False).input_ids
    if choice_ids.numel() == 0:
        return float("-inf")
    ids = torch.cat([prefix, choice_ids], dim=1).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
        logits = model(input_ids=ids, use_cache=False).logits.float()
    start = prefix.shape[1]
    log_probs = F.log_softmax(logits[:, start - 1:-1, :], dim=-1)
    gathered = log_probs.gather(-1, choice_ids.to(device).unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item())


def run_truthfulqa(model, tokenizer, items: list[dict], device) -> dict:
    rows = []
    for index, item in enumerate(items):
        scores = [answer_loglikelihood(model, tokenizer,
                                       tasks.truthfulqa_prompt(item, ""),
                                       choice, device)
                  for choice in item["choices"]]
        best = int(np.argmax(scores))
        correct = item["labels"][best] == 1
        rows.append({"id": item["id"], "predicted_index": best,
                     "correct": bool(correct), "scores": scores})
        if index % 100 == 0:
            print(f"eval: truthfulqa {index}/{len(items)}", flush=True)
    values = np.array([1.0 if row["correct"] else 0.0 for row in rows])
    return {
        "items": rows,
        "mc1": float(values.mean()) if values.size else 0.0,
        "mc1_ci": bootstrap_mean(values) if values.size else None,
        "scoring": "length-normalised log-likelihood of the choice given 'Q: ...\\nA:'",
    }


# ---------------------------------------------------------------------------
# cheap smoke eval


def smoke(student, teacher, tokenizer, items, windows, device, problems: int = 5) -> dict:
    subset = {}
    for domain, entries in windows.items():
        subset[domain] = entries[:2]
    small = items["humaneval"][:problems]
    return {
        "distributional": distributional(student, teacher, subset, device),
        "humaneval": run_humaneval(student, tokenizer, small, device),
        "probes": {
            "gsm8k": tasks.gsm8k_prompt(items["gsm8k"][0]),
            "truthfulqa": tasks.truthfulqa_prompt(items["truthfulqa"][0], ""),
        },
    }
