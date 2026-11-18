#!/usr/bin/env python3
"""S54 stage 1: score the post-trained frontier with its own chat protocol.

This is the executable half of the stage-1 measurement registered in
`docs/ws19-post-trained-route-registration-v1.json` and amended in
`docs/ws19-post-trained-route-stage2-amendment-s54-v1.json`:

* GSM8K test (1,319 items) and MATH-500 (500 items);
* zero-shot chain-of-thought, final answer in ``\\boxed{}``;
* each checkpoint's own tokenizer and ``chat_template.jinja``;
* non-thinking (``enable_thinking=False``) and thinking (``enable_thinking=True``);
* greedy (temperature 0, top_p 1) and majority-of-16 (temperature 0.8, top_p 0.95,
  seed ``5300 + item_index``);
* ``max_new_tokens = 4096`` and EOS-only stopping.

Engine
------
vLLM 0.34.0 is the registered engine for a *future* pinned image, but it is not
installed in the offline eval image used here. The eval image's pinned
transformers stack (5.23.0 / torch 2.15.0) loads these checkpoints directly, so
every number below is a transformers number. The receipt records
``engine = transformers`` and ``vllm_available = false``; no vLLM number is
claimed.

Extraction is reported on three bases side by side:

``strict``
    the last balanced ``\\boxed{...}`` only; two distinct boxed answers or a
    boxed ``or`` alternative are ambiguous;
``flexible``
    the registered fallbacks after a missing box: for GSM8K the last ``####``
    value then the last number; for MATH-500 the last "final answer is ..."
    phrase;
``exact``
    the same flexible extraction compared with ``math_verify`` (0.9.0) symbolic
    equivalence, labelled accordingly. If ``math_verify`` is unavailable the
    receipt says ``numeric-canonical`` instead of silently changing the basis.

No number is typed by hand: every accuracy, standard error, throughput, VRAM
peak and gap is computed from the generations in the raw JSONL named by the
receipt.

Usage
-----
    SELVEDGE_MEM=.. scripts/court --gpu eval -- python3 \
        scripts/s54_post_trained_eval.py \
        --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc063... \
        --model-id Qwen/Qwen3.5-0.8B --revision 2fc063... \
        --out results/stage1/s54-post-0.8b-greedy-nonthinking.json

The container limit is not chosen here. The launcher sizes it from the smoke
run's own cgroup peak; this script only refuses a CUDA free-memory check and
caps the torch allocator at ``--vram-limit-gib``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import arms, rft  # noqa: E402

GIB = float(1 << 30)
CACHE = os.environ.get("HF_HOME", "/hf-cache")
GSM8K_TEST = os.path.join(
    CACHE, "datasets", "openai___gsm8k", "main", "0.0.0",
    "740312add88f781978c0658806c59bc2815b9866", "gsm8k-test.arrow")
MATH500_TEST = os.path.join(
    CACHE, "datasets", "HuggingFaceH4___math-500", "default", "0.0.0",
    "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "math-500-test.arrow")

SYSTEM_PROMPT = None
USER_TEMPLATE = "{question}\n\nSolve the problem step by step. Put the final answer in \\boxed{{}}."
MODES = ("non-thinking", "thinking")
DECODINGS = ("greedy", "maj16")
TASKS = ("gsm8k", "math500")
MAX_NEW_TOKENS = 4096
MAJ_K = 16
MAJ_TEMPERATURE = 0.8
MAJ_TOP_P = 0.95
MAJ_SEED_BASE = 5300
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 53


# ---------------------------------------------------------------------------
# receipts and environment


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def cgroup_memory() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    events = {}
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                events[key] = int(value)
    except (OSError, ValueError):
        events = {}
    report["memory_events"] = events
    return report


def bundle_hashes(model_dir: str) -> dict:
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "model.safetensors.index.json",
                 "LICENSE"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            files[name] = {"bytes": os.path.getsize(path),
                           "sha256": sha256_file(path)}
    return files


def weight_sha256(model_dir: str) -> str | None:
    """Hash the single-file checkpoint, or the first shard of a sharded one."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        payload = json.load(open(index, encoding="utf-8"))
        shards = sorted(set(payload.get("weight_map", {}).values()))
        if shards:
            return sha256_file(os.path.join(model_dir, shards[0]))
        return None
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return sha256_file(single)
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            return sha256_file(os.path.join(model_dir, name))
    return None


# ---------------------------------------------------------------------------
# data


def _ipc_rows(path: str) -> list[dict]:
    import pyarrow.ipc as ipc

    if not os.path.isfile(path):
        raise SystemExit(f"s54-stage1: missing dataset {path}")
    return ipc.open_stream(path).read_all().to_pylist()


def load_task(task: str, limit: int | None) -> list[dict]:
    items = []
    if task == "gsm8k":
        for index, row in enumerate(_ipc_rows(GSM8K_TEST)):
            target = rft.extract(row["answer"], "gsm8k")
            if not target.ok:
                continue
            items.append({
                "problem_id": f"gsm8k-test:{index}",
                "task": "gsm8k",
                "question": row["question"],
                "gold": target.value,
                "gold_source": "#### value in the GSM8K reference answer",
            })
    elif task == "math500":
        for index, row in enumerate(_ipc_rows(MATH500_TEST)):
            gold = row.get("answer")
            if gold is None:
                solution = rft.extract(row.get("solution", ""), "math")
                gold = solution.value
            if gold is None:
                continue
            items.append({
                "problem_id": f"math500:{row.get('unique_id') or index}",
                "task": "math500",
                "question": row["problem"],
                "gold": str(gold),
                "gold_source": "MATH-500 answer field",
                "subject": row.get("subject"),
                "level": row.get("level"),
            })
    else:
        raise SystemExit(f"s54-stage1: unknown task {task!r}")
    if limit is not None:
        items = items[:limit]
    return items


# ---------------------------------------------------------------------------
# extraction


def _distinct(values: list[str]) -> list[str]:
    distinct: list[str] = []
    for candidate in values:
        if not any(rft.answers_equal(candidate, other) for other in distinct):
            distinct.append(candidate)
    return distinct


def extract_strict(text: str) -> dict:
    """The last balanced boxed answer, or an explicit refusal."""
    answers = rft._boxed_answers(text)
    if not answers:
        return {"value": None, "method": "none", "status": "unparseable",
                "reason": "no \\boxed{} answer"}
    distinct = _distinct(answers)
    if len(distinct) > 1:
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous",
                "reason": f"{len(distinct)} distinct boxed answers"}
    if len(answers) == 1 and rft._OR.search(answers[0]):
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous",
                "reason": "boxed alternative ('or')"}
    return {"value": answers[-1], "method": "last \\boxed{}", "status": "ok",
            "reason": "single boxed answer"}


def extract_flexible(text: str, task: str) -> dict:
    """The registered fallback chain after the box, shared with rft.extract."""
    boxed = extract_strict(text)
    if boxed["value"] is not None:
        return boxed
    if boxed["status"] == "ambiguous":
        # two distinct boxed answers are ambiguous, not "use the last one"
        return boxed
    dataset = "gsm8k" if task == "gsm8k" else "math"
    extraction = rft.extract(text, dataset)
    if extraction.ambiguous:
        status = "ambiguous"
    elif extraction.value is None:
        status = "unparseable"
    else:
        status = "ok"
    return {"value": extraction.value, "method": extraction.method,
            "status": status, "reason": extraction.reason}


_MATH_VERIFY = None
_MATH_VERIFY_BASIS = "unavailable"


def _load_math_verify():
    global _MATH_VERIFY, _MATH_VERIFY_BASIS
    if _MATH_VERIFY is not None:
        return _MATH_VERIFY
    try:
        from math_verify import parse, verify  # type: ignore
        _MATH_VERIFY = (parse, verify)
        try:
            import importlib.metadata as metadata
            _MATH_VERIFY_BASIS = "math_verify " + metadata.version("math-verify")
        except Exception:  # noqa: BLE001 - label only, never a score basis
            _MATH_VERIFY_BASIS = "math_verify"
    except Exception:  # noqa: BLE001 - the receipt records the fallback
        _MATH_VERIFY = False
        _MATH_VERIFY_BASIS = "numeric-canonical (math_verify unavailable)"
    return _MATH_VERIFY


def compare_exact(extracted: str, gold: str) -> bool:
    loaded = _load_math_verify()
    if loaded:
        parse, verify = loaded
        try:
            wrapped_pred = extracted if "$" in extracted else f"${extracted}$"
            wrapped_gold = gold if "$" in gold else f"${gold}$"
            return bool(verify(parse(wrapped_gold), parse(wrapped_pred)))
        except Exception:  # noqa: BLE001 - a parse failure is a wrong answer
            return False
    return rft.answers_equal(extracted, gold)


# ---------------------------------------------------------------------------
# statistics


def _accuracy(flags: list[bool]) -> tuple[float, float]:
    n = len(flags)
    if n == 0:
        return 0.0, 0.0
    p = sum(1 for flag in flags if flag) / n
    return p, (p * (1.0 - p) / n) ** 0.5


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_prompt_bootstrap(rows: list[dict], arm: str, reference: str,
                            resamples: int = BOOTSTRAP_RESAMPLES,
                            seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap over problems, not continuations.

    Each `rows` entry is one problem with `correct[arm]` and
    `correct[reference]` booleans. The resampling unit is the problem, so a
    prompt with 16 continuations contributes one paired observation per draw;
    resampling continuations would understate the interval.
    """
    deltas = [float(row["correct"][arm]) - float(row["correct"][reference])
              for row in rows]
    if not deltas:
        return {"problems": 0, "delta": None, "ci95": None,
                "basis": "prompt-level paired bootstrap"}
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    n = len(deltas)
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        draws.append(total / n)
    return {
        "problems": n,
        "delta": observed,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "resamples": resamples,
        "seed": seed,
        "basis": ("prompt-level paired bootstrap; the resampling unit is the "
                  "distinct problem, never the continuation"),
    }


# ---------------------------------------------------------------------------
# generation


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_prompt(tokenizer, question: str, thinking: bool) -> str:
    content = USER_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(thinking))


def _decode(tokenizer, row) -> str:
    return tokenizer.decode(row, skip_special_tokens=True)


def generate_batch(model, tokenizer, prompts: list[str], *, greedy: bool,
                   max_new_tokens: int, device: str,
                   num_return_sequences: int, seed: int | None = None) -> tuple[list[str], dict]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = int(input_ids.shape[1])
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences,
    )
    if greedy:
        kwargs.update(do_sample=False)
    else:
        kwargs.update(do_sample=True, temperature=MAJ_TEMPERATURE,
                      top_p=MAJ_TOP_P)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    wall = time.time() - started
    generated = output[:, prompt_length:]
    completions = [_decode(tokenizer, row) for row in generated]
    eos = tokenizer.eos_token_id
    truncated = []
    new_tokens = []
    for row in generated:
        ids = row.tolist()
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
        new_tokens.append(len(ids))
        truncated.append(len(ids) >= max_new_tokens and
                         (not ids or ids[-1] != eos))
    stats = {
        "wall_seconds": wall,
        "generated_tokens": int(sum(new_tokens)),
        "new_tokens_per_sequence": new_tokens,
        "truncated": truncated,
        "tokens_per_second": (sum(new_tokens) / wall) if wall > 0 else None,
        "prompt_tokens": int(attention_mask.sum().item()),
    }
    return completions, stats


def evaluate_task(model, tokenizer, items: list[dict], task: str, mode: str,
                  decoding: str, max_new_tokens: int, device: str,
                  raw_handle, batch_size: int) -> dict:
    import torch

    greedy = decoding == "greedy"
    thinking = mode == "thinking"
    per_problem = []
    started = time.time()
    generated_tokens = 0
    decode_seconds = 0.0
    truncated_count = 0

    index = 0
    while index < len(items):
        stop = index + 1 if not greedy else min(index + batch_size, len(items))
        chunk = items[index:stop]
        prompts = [render_prompt(tokenizer, item["question"], thinking)
                   for item in chunk]
        completions, stats = generate_batch(
            model, tokenizer, prompts, greedy=greedy,
            max_new_tokens=max_new_tokens, device=device,
            num_return_sequences=1 if greedy else MAJ_K,
            seed=(MAJ_SEED_BASE + index) if not greedy else None)
        generated_tokens += stats["generated_tokens"]
        decode_seconds += stats["wall_seconds"]
        truncated_count += sum(1 for value in stats["truncated"] if value)
        for offset, item in enumerate(chunk):
            if greedy:
                samples = [completions[offset]]
            else:
                samples = completions[offset * MAJ_K:(offset + 1) * MAJ_K]
            sample_rows = []
            for sample_index, completion in enumerate(samples):
                strict = extract_strict(completion)
                flexible = extract_flexible(completion, task)
                strict_correct = (strict["value"] is not None
                                  and rft.answers_equal(strict["value"], item["gold"]))
                flexible_correct = (flexible["value"] is not None
                                    and rft.answers_equal(flexible["value"], item["gold"]))
                exact_correct = (flexible["value"] is not None
                                 and compare_exact(flexible["value"], item["gold"]))
                if not greedy:
                    truncated = bool(stats["truncated"][offset * MAJ_K + sample_index])
                    new_tokens = int(stats["new_tokens_per_sequence"][
                        offset * MAJ_K + sample_index])
                else:
                    truncated = bool(stats["truncated"][offset])
                    new_tokens = int(stats["new_tokens_per_sequence"][offset])
                sample_rows.append({
                    "problem_id": item["problem_id"],
                    "sample_index": sample_index,
                    "completion": completion,
                    "new_tokens": new_tokens,
                    "truncated": truncated,
                    "strict": strict,
                    "flexible": flexible,
                    "strict_correct": bool(strict_correct),
                    "flexible_correct": bool(flexible_correct),
                    "exact_correct": bool(exact_correct),
                })
                if raw_handle is not None:
                    raw_handle.write(json.dumps({
                        "task": task, "mode": mode, "decoding": decoding,
                        "problem_id": item["problem_id"],
                        "sample_index": sample_index,
                        "gold": item["gold"],
                        "completion": completion,
                        "new_tokens": new_tokens,
                        "truncated": truncated,
                        "strict": strict["value"],
                        "strict_status": strict["status"],
                        "flexible": flexible["value"],
                        "flexible_status": flexible["status"],
                        "strict_correct": bool(strict_correct),
                        "flexible_correct": bool(flexible_correct),
                        "exact_correct": bool(exact_correct),
                    }, sort_keys=True) + "\n")
                    raw_handle.flush()

            # majority-of-16: the canonicalised extracted answer; ties go to the
            # earliest sample and a sample with no extractable answer does not
            # vote. Each extraction tier gets its own vote so strict, flexible
            # and exact are not conflated.
            def vote(value_of) -> tuple[str | None, str]:
                counts: dict[str, int] = {}
                first_seen: dict[str, int] = {}
                canonical_to_value: dict[str, str] = {}
                for sample_index, row in enumerate(sample_rows):
                    value = value_of(row)
                    if value is None:
                        continue
                    key = rft._canonical_math(value)
                    counts[key] = counts.get(key, 0) + 1
                    first_seen.setdefault(key, sample_index)
                    canonical_to_value.setdefault(key, value)
                if not counts:
                    return None, "unparseable"
                best = max(counts, key=lambda key: (counts[key],
                                                    -first_seen[key]))
                return canonical_to_value[best], "ok"

            strict_vote, strict_status = (None, None) if greedy else (
                vote(lambda row: row["strict"]["value"]))
            flexible_vote, flexible_status = (None, None) if greedy else (
                vote(lambda row: row["flexible"]["value"]))
            exact_vote, exact_status = (None, None) if greedy else (
                vote(lambda row: (row["flexible"]["value"]
                                  if row["flexible"]["value"] is not None
                                  else None)))
            per_problem.append({
                "problem_id": item["problem_id"],
                "gold": item["gold"],
                "strict_correct": sample_rows[0]["strict_correct"] if greedy else None,
                "flexible_correct": sample_rows[0]["flexible_correct"] if greedy else None,
                "exact_correct": sample_rows[0]["exact_correct"] if greedy else None,
                "strict_vote_correct": (None if greedy else (
                    strict_vote is not None
                    and rft.answers_equal(strict_vote, item["gold"]))),
                "flexible_vote_correct": (None if greedy else (
                    flexible_vote is not None
                    and rft.answers_equal(flexible_vote, item["gold"]))),
                "exact_vote_correct": (None if greedy else (
                    exact_vote is not None
                    and compare_exact(exact_vote, item["gold"]))),
                "vote_value": flexible_vote,
                "vote_status": flexible_status,
                "vote_correct": (None if greedy else (
                    flexible_vote is not None
                    and rft.answers_equal(flexible_vote, item["gold"]))),
                "strict_vote_status": strict_status,
                "exact_vote_status": exact_status,
                "samples": len(sample_rows),
                "answered_samples": sum(
                    1 for row in sample_rows
                    if row["flexible"]["status"] == "ok"),
                "truncated_samples": sum(
                    1 for row in sample_rows if row["truncated"]),
                "boxed_samples": sum(
                    1 for row in sample_rows
                    if row["strict"]["status"] == "ok"),
            })
        index = stop

    if greedy:
        strict_flags = [bool(row["strict_correct"]) for row in per_problem]
        flexible_flags = [bool(row["flexible_correct"]) for row in per_problem]
        exact_flags = [bool(row["exact_correct"]) for row in per_problem]
        primary_flags = flexible_flags
    else:
        strict_flags = [bool(row["strict_vote_correct"]) for row in per_problem]
        flexible_flags = [bool(row["flexible_vote_correct"]) for row in per_problem]
        exact_flags = [bool(row["exact_vote_correct"]) for row in per_problem]
        primary_flags = [bool(row["vote_correct"]) for row in per_problem]

    def scores(flags):
        accuracy, stderr = _accuracy(flags)
        return {"correct": sum(1 for flag in flags if flag),
                "items": len(flags), "accuracy": accuracy, "stderr": stderr}

    answered = sum(row["answered_samples"] for row in per_problem)
    total_samples = sum(row["samples"] for row in per_problem)
    boxed = sum(row["boxed_samples"] for row in per_problem)
    return {
        "task": task,
        "mode": mode,
        "decoding": decoding,
        "items": len(per_problem),
        "distinct_problems": len({row["problem_id"] for row in per_problem}),
        "samples_per_problem": 1 if greedy else MAJ_K,
        "accuracy": scores(primary_flags),
        "strict": scores(strict_flags),
        "flexible": scores(flexible_flags),
        "exact": scores(exact_flags),
        "exact_basis": _MATH_VERIFY_BASIS,
        "answered_samples": answered,
        "total_samples": total_samples,
        "boxed_samples": boxed,
        "format_correct_fraction": boxed / total_samples if total_samples else None,
        "truncated_samples": truncated_count,
        "truncated_fraction": truncated_count / total_samples if total_samples else None,
        "mean_new_tokens": (generated_tokens / total_samples
                            if total_samples else None),
        "generated_tokens": generated_tokens,
        "decode_wall_seconds": decode_seconds,
        "tokens_per_second": (generated_tokens / decode_seconds
                              if decode_seconds > 0 else None),
        "peak_vram_gib": (torch.cuda.max_memory_allocated() / GIB
                          if device == "cuda" else None),
        "peak_vram_reserved_gib": (torch.cuda.max_memory_reserved() / GIB
                                   if device == "cuda" else None),
        "wall_seconds": round(time.time() - started, 1),
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# main


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", default="gsm8k,math500")
    parser.add_argument("--modes", default="non-thinking")
    parser.add_argument("--decodings", default="greedy")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N items per task (smoke)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vram-limit-gib", type=float, default=14.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args(argv)

    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    decodings = [value.strip() for value in args.decodings.split(",")
                 if value.strip()]
    for name in tasks:
        if name not in TASKS:
            raise SystemExit(f"s54-stage1: unknown task {name!r}")
    for name in modes:
        if name not in MODES:
            raise SystemExit(f"s54-stage1: unknown mode {name!r}")
    for name in decodings:
        if name not in DECODINGS:
            raise SystemExit(f"s54-stage1: unknown decoding {name!r}")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("s54-stage1: CUDA required; run through "
                         "scripts/court --gpu eval")
    free, total = torch.cuda.mem_get_info()
    if args.vram_limit_gib * GIB > free:
        raise SystemExit(
            f"s54-stage1: vram limit {args.vram_limit_gib:.1f} GiB exceeds free "
            f"{free / GIB:.2f} GiB")
    torch.cuda.set_per_process_memory_fraction(
        min(0.99, args.vram_limit_gib * GIB / total))
    torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = load_tokenizer(args.model_dir)
    model = arms.load_model(args.model_dir, dtype=torch.bfloat16,
                            device="cuda", attn_implementation="sdpa")
    model.eval()

    raw_dir = args.raw_dir or (os.path.splitext(args.out)[0] + "-raw")
    os.makedirs(raw_dir, exist_ok=True)
    task_results = []
    raw_paths = {}
    started = time.time()
    for task in tasks:
        items = load_task(task, args.limit)
        for mode in modes:
            for decoding in decodings:
                label = f"{task}-{mode}-{decoding}"
                raw_path = os.path.join(raw_dir, label + ".jsonl")
                raw_paths[label] = raw_path
                with open(raw_path, "w", encoding="utf-8") as raw_handle:
                    result = evaluate_task(
                        model, tokenizer, items, task, mode, decoding,
                        args.max_new_tokens, "cuda", raw_handle,
                        args.batch_size)
                task_results.append(result)
                print(f"s54-stage1: {label} acc={result['accuracy']['accuracy']:.4f} "
                      f"strict={result['strict']['accuracy']:.4f} "
                      f"flexible={result['flexible']['accuracy']:.4f} "
                      f"exact={result['exact']['accuracy']:.4f} "
                      f"tokens/s={result['tokens_per_second']:.1f}", flush=True)
                write_json(args.out, {
                    "schema": "selvedge/s54-stage1-eval/v1",
                    "created_utc": utcnow(), "stamp": stamp(),
                    "revision": revision(),
                    "complete": False,
                    "model": {
                        "id": args.model_id, "revision": args.revision,
                        "dir": args.model_dir,
                        "text_tower_elements": sum(
                            parameter.numel() for parameter in model.parameters()),
                        "bundle": bundle_hashes(args.model_dir),
                        "weight_sha256_first_shard": weight_sha256(args.model_dir),
                    },
                    "engine": {
                        "name": "transformers",
                        "version": __import__("transformers").__version__,
                        "torch_version": torch.__version__,
                        "attention": "sdpa",
                        "dtype": "bfloat16",
                        "vllm_available": False,
                        "vllm_note": ("vLLM 0.34.0 is not installed in the "
                                      "offline eval image; the pinned "
                                      "transformers stack loads these "
                                      "checkpoints and is the engine for every "
                                      "number here"),
                    },
                    "protocol": {
                        "system": SYSTEM_PROMPT,
                        "user_template": USER_TEMPLATE,
                        "few_shot": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "stop": "EOS only",
                        "greedy": {"temperature": 0.0, "top_p": 1.0},
                        "maj16": {"k": MAJ_K, "temperature": MAJ_TEMPERATURE,
                                  "top_p": MAJ_TOP_P,
                                  "seed_rule": "torch.manual_seed(5300 + item_index) per item",
                                  "vote_unit": "canonicalised extracted answer",
                                  "tie": "earliest sample",
                                  "no_answer": "does not vote"},
                        "extraction": {
                            "strict": "last balanced \\boxed{} only",
                            "flexible": ("boxed, then ####/last number for GSM8K "
                                         "and 'final answer is' for MATH-500"),
                            "exact": _MATH_VERIFY_BASIS,
                        },
                    },
                    "tasks": task_results,
                    "raw_paths": raw_paths,
                    "cgroup_memory": cgroup_memory(),
                    "wall_seconds": round(time.time() - started, 1),
                    "command": sys.argv,
                })
                gc.collect()
                torch.cuda.empty_cache()

    # gaps are computed from the same problem ids, paired at the problem level
    gaps = {}
    for task in tasks:
        for mode in modes:
            greedy = next((row for row in task_results
                           if row["task"] == task and row["mode"] == mode
                           and row["decoding"] == "greedy"), None)
            maj = next((row for row in task_results
                        if row["task"] == task and row["mode"] == mode
                        and row["decoding"] == "maj16"), None)
            if greedy is None or maj is None:
                continue
            maj_by_id = {row["problem_id"]: row for row in maj["per_problem"]}
            paired = []
            for row in greedy["per_problem"]:
                other = maj_by_id.get(row["problem_id"])
                if other is None:
                    continue
                paired.append({"correct": {
                    "maj16": bool(other["vote_correct"]),
                    "greedy": bool(row["flexible_correct"]),
                }})
            gaps[f"{task}-{mode}"] = paired_prompt_bootstrap(paired, "maj16",
                                                              "greedy")

    payload = json.load(open(args.out, encoding="utf-8"))
    payload["complete"] = True
    payload["created_utc"] = utcnow()
    payload["stamp"] = stamp()
    payload["greedy_vs_maj16"] = gaps
    payload["wall_seconds"] = round(time.time() - started, 1)
    payload["raw_sha256"] = {label: sha256_file(path)
                             for label, path in raw_paths.items()}
    write_json(args.out, payload)
    print(f"s54-stage1: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
