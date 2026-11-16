#!/usr/bin/env python3
"""T1 evaluation harness: teacher KL/PPL, HumanEval pass@1, GSM8K EM and
TruthfulQA MC1, all pinned and deterministic.

Every reported number is produced here, from a pinned checkpoint and pinned
items. Greedy decoding for pass@1 and exact match; MC1 by length-normalised
log-likelihood; no sampling in the primary metrics. Per-window and per-item
values are written to the receipt so the paired bootstrap can be recomputed.

The distributional metric keeps the teacher and the student resident at the
same time, so it must not also hold two full `[positions, vocab]` logit
tensors: on this vocabulary (151,936) and a 512-position window that is 311 MiB
per f32 tensor before softmax temporaries, which is what OOMed the H
evaluation. Instead, each model's final hidden states are computed once and
the vocabulary head is applied `LOGIT_CHUNK` positions at a time, reduced in
f32 and freed. Nothing about the registered metric changes: every position is
still scored against the full vocabulary, with the same KL(teacher||student),
NLL, PPL and top-1 definitions.
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

#: Positions per logits chunk on the distributional path. Two f32 chunk tensors
#: (student and teacher) are the peak: 64 positions x 151,936 vocabulary is
#: 78 MiB each, before the log-softmax temporaries. The peak is therefore
#: bounded by `2 * LOGIT_CHUNK * vocab * 4` bytes regardless of window length.
LOGIT_CHUNK = 64


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


def output_head(model):
    """The causal LM's vocabulary head: the only module that makes full logits."""
    head = getattr(model, "lm_head", None)
    if head is None:
        getter = getattr(model, "get_output_embeddings", None)
        head = getter() if getter is not None else None
    if not isinstance(head, torch.nn.Module):
        raise RuntimeError(
            "the streaming metric path needs a causal LM with an output head; "
            f"{type(model).__name__} has none")
    return head


def final_hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """Run the model once and return its final hidden states, with no logits.

    The vocabulary head is replaced for the duration of the call, so the whole
    forward pass runs exactly as it does in training — including every hook the
    arm installed (the memory arms set their token context from a forward
    pre-hook on the model) — while the `[positions, vocab]` logits tensor is
    never materialised. Callers stream the head over position chunks with
    `logits_chunk` afterwards.
    """
    head = output_head(model)
    captured: dict[str, torch.Tensor] = {}

    def capture(hidden_states, *args, **kwargs):
        captured["hidden"] = hidden_states
        return hidden_states.new_empty((0,))

    had_own_forward = "forward" in head.__dict__
    original = head.forward
    head.forward = capture
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        if had_own_forward:
            head.forward = original
        else:
            del head.forward
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("the model's output head was never reached")
    return hidden


def logits_chunk(model, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """f32 logits for positions `[start, end)`, as the full forward would give."""
    return output_head(model)(hidden[:, start:end, :]).float()


@torch.inference_mode()
def window_stats(student, teacher, tokens: list[int], device,
                 chunk: int = LOGIT_CHUNK) -> dict:
    """Registered per-window statistics, streamed over the vocabulary head.

    KL(teacher||student) in nats/token, the student's NLL/PPL and top-1, all
    against the *full* vocabulary at every position. Only the storage changes:
    one hidden-state pass per model, then `chunk` positions of logits per head
    application, reduced in f32 and dropped before the next chunk.
    """
    if chunk < 1:
        raise ValueError("chunk must be at least one position")
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    if ids.shape[1] < 2:
        raise ValueError("a window needs at least two tokens to score a position")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
        hidden_s = final_hidden_states(student, ids)
        hidden_t = final_hidden_states(teacher, ids)
    # Predict tokens 1..n-1.
    target = ids[0, 1:]
    positions = int(target.numel())
    kl_parts: list[torch.Tensor] = []
    nll_parts: list[torch.Tensor] = []
    top1_parts: list[torch.Tensor] = []
    for start in range(0, positions, chunk):
        end = min(start + chunk, positions)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            s = logits_chunk(student, hidden_s, start, end)
            t = logits_chunk(teacher, hidden_t, start, end)
        logp_s = F.log_softmax(s, dim=-1)
        logp_t = F.log_softmax(t, dim=-1)
        p_t = logp_t.exp()
        kl_parts.append((p_t * (logp_t - logp_s)).sum(dim=-1).reshape(-1))
        nll_parts.append(
            -logp_s.gather(-1, target[start:end].view(1, -1, 1)).squeeze(-1).reshape(-1))
        top1_parts.append(
            (s.argmax(dim=-1) == target[start:end]).float().reshape(-1))
        del s, t, logp_s, logp_t, p_t
    kl = torch.cat(kl_parts)
    nll = torch.cat(nll_parts)
    top1 = torch.cat(top1_parts)
    del hidden_s, hidden_t
    return {
        "kl": kl.mean().item(),
        "ppl": float(torch.exp(nll.mean()).item()),
        "nll": nll.mean().item(),
        "top1": top1.mean().item(),
        "positions": positions,
    }


def distributional(student, teacher, windows: dict[str, list[dict]], device,
                   chunk: int = LOGIT_CHUNK) -> dict:
    result = {"domains": {}, "per_window": {}}
    for domain, items in windows.items():
        rows = []
        for item in items:
            stats = window_stats(student, teacher, item["tokens"], device, chunk)
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
def answer_loglikelihood(model, tokenizer, prompt: str, choice: str, device,
                         chunk: int = LOGIT_CHUNK) -> float:
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    choice_ids = tokenizer(" " + choice, return_tensors="pt", add_special_tokens=False).input_ids
    if choice_ids.numel() == 0:
        return float("-inf")
    ids = torch.cat([prefix, choice_ids], dim=1).to(device)
    start = prefix.shape[1]
    # The first choice token sits at position `start`; logits at `start - 1`
    # predict it. Only the choice's positions are ever scored, so only they are
    # streamed off the head.
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
        hidden = final_hidden_states(model, ids)
    gathered = []
    for offset in range(0, choice_ids.shape[1], chunk):
        stop = min(offset + chunk, choice_ids.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            logits = logits_chunk(model, hidden, start - 1 + offset, start - 1 + stop)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered.append(log_probs.gather(
            -1, choice_ids[:, offset:stop].to(device).unsqueeze(-1)).squeeze(-1))
    return float(torch.cat(gathered, dim=-1).mean().item())


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
            "gsm8k": tasks.gsm8k_prompt(items["gsm8k"][0]) if items.get("gsm8k") else None,
            "truthfulqa": tasks.truthfulqa_prompt(items["truthfulqa"][0], "")
            if items.get("truthfulqa") else None,
        },
    }
