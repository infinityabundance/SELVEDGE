#!/usr/bin/env python3
"""S45 next route: rejection-sampling fine-tuning and on-policy distillation.

The unit that can be tested on CPU is the **verifier**.  A solution is accepted
only when one answer can be extracted unambiguously and it equals the reference
answer exactly, under a documented extractor:

* numerical values are compared as exact numbers (integers, decimals, simple
  fractions, and braced/unbraced ``\\frac``) with a 1e-9 relative tolerance;
* everything else is compared as a canonical string after a fixed, listed set of
  LaTeX rewrites (``\\dfrac``/``\\tfrac`` -> ``\\frac``, ``\\left``/``\\right``
  removed, ``\\cdot``/``\\times`` -> ``*``, ``\\div`` -> ``/``, spacing removed);
* it is **not** a symbolic verifier.  Two algebraically equal expressions that
  do not canonicalise to the same string are rejected.  That is intentional:
  a false accept is worse than a false reject for this pipeline.

Ambiguity is rejected rather than resolved: two distinct ``####`` answers in a
GSM8K solution, two distinct ``\\boxed{}`` answers in a MATH solution, or a
boxed alternative ("... or ...") is ``ambiguous`` and is not kept.

Subcommands
-----------
``sample``
    Generate ``--samples`` solutions per training prompt from the teacher
    (``--policy teacher``) or the student (``--policy student``), verify, keep
    only verified-correct solutions, deduplicate within a prompt, and write
    ``accepted.jsonl`` plus a receipt with acceptance rates.  This is the GPU
    job; it is CPU-runnable but slow.  ``--sample-source FILE`` reads
    pre-generated samples instead and is the CPU test path.

``verify``
    Verify one hand-written solution (used by tests and by an operator).

``contamination``
    Re-run the n-gram contamination check on an accepted file.

Usage
-----
    python3 oracle/frontier/rft.py sample --dataset gsm8k-train --limit 100 \\
        --samples 8 --policy teacher --temperature 0.8 --top-p 0.95 \\
        --max-new-tokens 512 --seed 451 --out-dir results/stage2/rft/gsm8k
    python3 oracle/frontier/rft.py sample --sample-source fixtures/s45/samples.jsonl \\
        --out-dir /tmp/s45-rft
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("HF_HOME", "/hf-cache")
TEACHER_DIR = os.path.join(
    CACHE, "models", "Qwen", "Qwen3.5-2B-Base",
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
STUDENT_DIR = os.path.join(
    CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
    "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
GSM8K_TRAIN = os.path.join(
    CACHE, "datasets", "openai___gsm8k", "main", "0.0.0",
    "740312add88f781978c0658806c59bc2815b9866", "gsm8k-train.arrow")
GSM8K_TEST = os.path.join(
    CACHE, "datasets", "openai___gsm8k", "main", "0.0.0",
    "740312add88f781978c0658806c59bc2815b9866", "gsm8k-test.arrow")
MATH_TRAIN = "results/artifacts/train-corpus/hendrycks-math-train"
MATH500_RAW = "results/stage2/eval/base/raw/20261117T043134Z-minerva_math500.json"

SHINGLE_WIDTH = 5
DISCARD_JACCARD = 0.2
FAIL_JACCARD = 0.5
NUMERIC_TOLERANCE = 1e-9

_GSM_MARK = re.compile(r"####\s*([^\n]+?)(?=\s*####|\n|$)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FRAC_BRACED = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_FRAC_BARE = re.compile(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])")
_OR = re.compile(r"(?:\\text\{\s*or\s*\}|(?:\A|\s)or(?:\s|\Z))", re.IGNORECASE)


# ---------------------------------------------------------------------------
# extraction


def _balanced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index
    return None


def _boxed_answers(text: str) -> list[str]:
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        brace = text.index("{", match.start())
        parsed = _balanced(text, brace)
        if parsed is not None:
            answers.append(parsed[0])
    return answers


def _final_answer_phrase(text: str) -> str | None:
    markers = list(re.finditer(r"final answer is\s+(.+)", text, re.IGNORECASE))
    if not markers:
        return None
    value = markers[-1].group(1).strip()
    return value.rstrip(".").strip()


def _gsm_answers(text: str) -> list[str]:
    return [match.strip() for match in _GSM_MARK.findall(text)]


def _canonical_math(value: str) -> str:
    text = value.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("\\:", "").replace("\\ ", " ")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\div", "/")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    # Remove braces around a single token: \sqrt{2} and \sqrt2 are the same
    # written expression for this extractor; this is a rewrite, not algebra.
    text = re.sub(r"\{([A-Za-z0-9])\}", r"\1", text)
    while text.startswith("{") and text.endswith("}"):
        parsed = _balanced(text, 0)
        if parsed is None or parsed[1] != len(text) - 1:
            break
        text = parsed[0]
    return text


def _numeric(value: str) -> float | None:
    text = _canonical_math(value)
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    text = text.replace(",", "")
    match = _FRAC_BRACED.fullmatch(text) or _FRAC_BARE.fullmatch(text)
    if match:
        try:
            number = float(match.group(1)) / float(match.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    else:
        try:
            number = float(text)
        except ValueError:
            return None
    return number / 100.0 if percent else number


def answers_equal(left: str, right: str) -> bool:
    left_number = _numeric(left)
    right_number = _numeric(right)
    if left_number is not None and right_number is not None:
        scale = max(1.0, abs(left_number), abs(right_number))
        return abs(left_number - right_number) <= NUMERIC_TOLERANCE * scale
    return _canonical_math(left) == _canonical_math(right)


@dataclass(frozen=True)
class Extraction:
    value: str | None
    candidates: tuple[str, ...]
    method: str
    ambiguous: bool
    reason: str

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.ambiguous


def extract(text: str, dataset: str) -> Extraction:
    """Extract exactly one answer, or say why not."""
    if dataset == "gsm8k":
        candidates = _gsm_answers(text)
        method = "last #### value"
        if not candidates:
            numbers = _NUMBER.findall(text)
            if not numbers:
                return Extraction(None, (), "none", False, "no #### and no number")
            candidates = [numbers[-1]]
            method = "last number fallback"
    elif dataset == "math":
        candidates = _boxed_answers(text)
        method = "last \\boxed{}"
        if not candidates:
            phrase = _final_answer_phrase(text)
            if phrase is None:
                return Extraction(None, (), "none", False, "no boxed answer and no 'final answer is'")
            candidates = [phrase]
            method = "final answer phrase fallback"
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")

    distinct: list[str] = []
    for candidate in candidates:
        canonical = _canonical_math(candidate)
        if not any(answers_equal(canonical, other) for other in distinct):
            distinct.append(candidate)
    if len(distinct) > 1:
        return Extraction(None, tuple(candidates), method, True,
                          f"{len(distinct)} distinct answers")
    if len(candidates) == 1 and _OR.search(candidates[0]):
        return Extraction(None, tuple(candidates), method, True,
                          "boxed alternative ('or')")
    if not candidates:
        return Extraction(None, (), method, False, "empty candidate")
    return Extraction(candidates[-1], tuple(candidates), method, False, "single answer")


def verify(solution: str, target: str, dataset: str) -> dict:
    """Return a verdict in {correct, wrong, ambiguous, unparseable}."""
    extraction = extract(solution, dataset)
    if extraction.ambiguous:
        verdict = "ambiguous"
    elif extraction.value is None:
        verdict = "unparseable"
    elif answers_equal(extraction.value, target):
        verdict = "correct"
    else:
        verdict = "wrong"
    return {
        "verdict": verdict,
        "extracted": extraction.value,
        "candidates": list(extraction.candidates),
        "method": extraction.method,
        "reason": extraction.reason,
        "target": target,
    }


# ---------------------------------------------------------------------------
# training data


def load_items(dataset: str, limit: int | None = None,
               start: int = 0) -> list[dict]:
    """Deterministic training prompts with an extractable reference answer."""
    items: list[dict] = []
    skipped = 0
    if dataset == "gsm8k-train":
        import pyarrow.ipc as ipc

        table = ipc.open_stream(GSM8K_TRAIN).read_all()
        rows = table.to_pylist()
        for index, row in enumerate(rows):
            target = extract(row["answer"], "gsm8k")
            if not target.ok:
                skipped += 1
                continue
            items.append({
                "id": f"gsm8k-train:{index}",
                "dataset": "gsm8k",
                "split": "train",
                "subject": None,
                "prompt": f"Question: {row['question']}\nAnswer:",
                "target": target.value,
                "reference_solution": row["answer"],
                # GSM8K has no level field; the calculator annotations `<<...>>`
                # are the reference solution's own operation steps and are the
                # registered difficulty proxy.
                "difficulty": row["answer"].count("<<"),
                "difficulty_source": "calculator annotations in the reference solution",
                "reasoning_length": len(_words(row["answer"])),
            })
    elif dataset == "math-train":
        import pyarrow.parquet as pq

        paths = sorted(glob.glob(os.path.join(REPO, MATH_TRAIN, "*", "*.parquet")))
        if not paths:
            raise SystemExit(f"no MATH-train parquet under {MATH_TRAIN}")
        for path in paths:
            subject = os.path.basename(os.path.dirname(path))
            rows = pq.read_table(path).to_pylist()
            for index, row in enumerate(rows):
                target = extract(row["solution"], "math")
                if not target.ok:
                    skipped += 1
                    continue
                items.append({
                    "id": f"math-train:{subject}:{index}",
                    "dataset": "math",
                    "split": "train",
                    "subject": subject,
                    "prompt": f"Problem:\n{row['problem']}\n\nSolution:",
                    "target": target.value,
                    "reference_solution": row["solution"],
                    # Hendrycks MATH carries an explicit level 1-5.
                    "difficulty": row.get("level"),
                    "difficulty_source": "Hendrycks MATH level field",
                    "reasoning_length": len(_words(row["solution"])),
                })
    else:
        raise SystemExit(f"unknown dataset {dataset!r}")

    items = items[start:]
    if limit is not None:
        items = items[:limit]
    for item in items:
        item["skipped_reference_items"] = skipped
    return items


def load_pool(datasets: list[str]) -> list[dict]:
    """Load every requested train split, keeping each item's own source."""
    pool: list[dict] = []
    for dataset in datasets:
        pool.extend(load_items(dataset))
    return pool


def _difficulty_bucket(item: dict) -> str:
    value = item.get("difficulty")
    if item.get("dataset") == "math" and value is not None:
        return f"level{int(value)}"
    steps = int(value or 0)
    return f"steps{min(steps, 3)}"


def _length_bucket(length: int, boundaries: list[int]) -> int:
    index = 0
    for boundary in boundaries:
        if length >= boundary:
            index += 1
    return min(index, len(boundaries))


def stratified_select(items: list[dict], count: int | None, seed: int
                      ) -> tuple[list[dict], dict]:
    """Seeded stratified selection instead of a prefix truncation.

    Strata are `(source, difficulty bucket, reasoning-length quartile)`. The
    pool is shuffled inside each stratum with a seeded RNG, then strata are
    consumed round-robin so every represented stratum appears before any one
    stratum contributes twice. This is what replaces `items[:limit]`, which
    would take only the first GSM8K questions (all easy) when both sources are
    pooled.
    """
    import random

    if not items:
        raise SystemExit("stratified_select received no items")
    lengths = sorted(int(item.get("reasoning_length") or 0) for item in items)
    if len(lengths) >= 4:
        boundaries = [lengths[len(lengths) * numerator // 4]
                      for numerator in (1, 2, 3)]
    else:
        boundaries = []
    strata: dict[str, list[dict]] = {}
    for item in items:
        key = "|".join((str(item.get("dataset") or "unknown"),
                        _difficulty_bucket(item),
                        str(_length_bucket(int(item.get("reasoning_length") or 0),
                                           boundaries))))
        strata.setdefault(key, []).append(item)
    rng = random.Random(seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    keys = sorted(strata)
    target = len(items) if count is None else min(int(count), len(items))
    selected: list[dict] = []
    cursor = {key: 0 for key in keys}
    while len(selected) < target:
        progressed = False
        for key in keys:
            if cursor[key] >= len(strata[key]):
                continue
            selected.append(strata[key][cursor[key]])
            cursor[key] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    report = {
        "kind": "stratified_selection",
        "seed": int(seed),
        "requested": None if count is None else int(count),
        "pool": len(items),
        "selected": len(selected),
        "length_boundaries_words": boundaries,
        "strata": [
            {"key": key, "pool": len(strata[key]), "selected": cursor[key]}
            for key in keys
        ],
        "replaces": "prefix truncation (items[start:start+limit])",
    }
    return selected, report


def select_items(datasets: list[str], limit: int | None, start: int,
                 sampling: str, seed: int) -> tuple[list[dict], dict]:
    """The registered selection modes.

    `prefix` is the old behaviour and is retained only for reproducing earlier
    runs. `stratified` is the corrected mode; it refuses `--start` because a
    start offset is meaningless once the pool is shuffled by stratum.
    """
    if sampling == "prefix":
        if len(datasets) != 1:
            raise SystemExit("prefix selection is single-source; use --sampling "
                             "stratified for a pooled run")
        items = load_items(datasets[0], limit, start)
        return items, {"kind": "prefix", "requested": limit, "start": start,
                       "selected": len(items),
                       "replaces": "nothing; this is the old mode"}
    if sampling != "stratified":
        raise SystemExit(f"unknown sampling mode {sampling!r}")
    if start:
        raise SystemExit("--start is prefix-only; stratified selection is seeded")
    pool = load_pool(datasets)
    selected, report = stratified_select(pool, limit, seed)
    for item in selected:
        item["skipped_reference_items"] = 0
    report["sources"] = datasets
    return selected, report


def format_curriculum(solution: str, mode: str) -> str:
    """Target-side formatting only; the eval prompt is never changed."""
    if mode == "none":
        return solution
    text = solution.strip()
    if mode in ("prefix", "prefix+steps"):
        text = "Solution:\n" + text
    if mode in ("steps", "prefix+steps"):
        lines = []
        step = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if re.match(r"^(?:Step\s+\d+|Solution:)", stripped, re.IGNORECASE):
                lines.append(line)
                continue
            step += 1
            lines.append(f"Step {step}: {line}")
        text = "\n".join(lines)
    return text


def dedupe_key(solution: str) -> str:
    return re.sub(r"\s+", " ", solution).strip()


# ---------------------------------------------------------------------------
# contamination


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    if len(words) < SHINGLE_WIDTH:
        return {tuple(words)} if words else set()
    return {tuple(words[index:index + SHINGLE_WIDTH])
            for index in range(len(words) - SHINGLE_WIDTH + 1)}


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_items() -> list[dict]:
    """The evaluation test splits the accepted data must not overlap."""
    import pyarrow.ipc as ipc

    items = []
    table = ipc.open_stream(GSM8K_TEST).read_all()
    for row in table.to_pylist():
        items.append({"task": "gsm8k-test",
                      "text": f"{row['question']}\n{row['answer']}"})
    path = os.path.join(REPO, MATH500_RAW)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        samples = payload["samples"]
        key = next(iter(samples))
        seen = set()
        for sample in samples[key]:
            if sample.get("filter") != "none" or sample["doc_id"] in seen:
                continue
            seen.add(sample["doc_id"])
            doc = sample.get("doc", {})
            items.append({
                "task": "math500-test",
                "text": f"{doc.get('problem', '')}\n{doc.get('solution', '')}",
            })
    return items


def _shingle_bag(words: list[str]) -> set[tuple[str, ...]]:
    if len(words) < SHINGLE_WIDTH:
        return {tuple(words)} if words else set()
    return {tuple(words[index:index + SHINGLE_WIDTH])
            for index in range(len(words) - SHINGLE_WIDTH + 1)}


def contamination_check(items: list[dict], tests: list[dict] | None = None) -> dict:
    tests = tests if tests is not None else test_items()
    test_bags = [_shingle_bag(_words(entry["text"])) for entry in tests]
    inverted: dict[tuple[str, ...], list[int]] = {}
    for index, bag in enumerate(test_bags):
        for shingle in bag:
            inverted.setdefault(shingle, []).append(index)
    worst = 0.0
    above_discard = 0
    above_fail = 0
    examples = []
    for item in items:
        bag = _shingle_bag(_words(item["prompt"] + "\n" + item["solution"]))
        shared: dict[int, int] = {}
        for shingle in bag:
            for index in inverted.get(shingle, ()):
                shared[index] = shared.get(index, 0) + 1
        best = 0.0
        best_task = None
        for index, count in shared.items():
            union = len(bag) + len(test_bags[index]) - count
            score = count / union if union else 0.0
            if score > best:
                best = score
                best_task = tests[index]["task"]
        if best > worst:
            worst = best
        if best >= DISCARD_JACCARD:
            above_discard += 1
        if best >= FAIL_JACCARD:
            above_fail += 1
            if len(examples) < 5:
                examples.append({"id": item["id"], "task": best_task,
                                 "jaccard": best})
    return {
        "items": len(items),
        "test_items": len(tests),
        "shingle_width": SHINGLE_WIDTH,
        "max_jaccard": worst,
        "at_or_above_discard": above_discard,
        "at_or_above_fail": above_fail,
        "discard_jaccard": DISCARD_JACCARD,
        "fail_jaccard": FAIL_JACCARD,
        "fail_closed": above_fail > 0,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# sampling and rejection


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _sample_source(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_generator(model_dir: str, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=False)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _generate(torch, tokenizer, model, prompt: str, samples: int,
              temperature: float, top_p: float, max_new_tokens: int,
              seed: int, device: str) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=samples,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
            for row in output]


def _rejection(rows: list[dict], dataset: str, curriculum: str) -> tuple[list[dict], dict]:
    accepted = []
    per_prompt = []
    total_sampled = 0
    total_verified = 0
    for row in rows:
        row_dataset = row.get("dataset", dataset)
        samples = row["samples"]
        total_sampled += len(samples)
        per_prompt_entry = {"id": row["id"], "sampled": len(samples),
                            "verdicts": {"correct": 0, "wrong": 0,
                                         "ambiguous": 0, "unparseable": 0},
                            "verified": 0, "kept": 0}
        seen: dict[str, int] = {}
        for index, solution in enumerate(samples):
            result = verify(solution, row["target"], row_dataset)
            per_prompt_entry["verdicts"][result["verdict"]] += 1
            if result["verdict"] != "correct":
                continue
            per_prompt_entry["verified"] += 1
            total_verified += 1
            key = dedupe_key(solution)
            if key in seen:
                continue
            seen[key] = index
            accepted.append({
                "id": row["id"],
                "dataset": row_dataset,
                "prompt": row["prompt"],
                "target": row["target"],
                "solution": format_curriculum(solution, curriculum),
                "raw_solution": solution,
                "extracted": result["extracted"],
                "verification_method": result["method"],
                "sample_index": index,
                "curriculum": curriculum,
            })
            per_prompt_entry["kept"] += 1
        per_prompt.append(per_prompt_entry)
    accepted.sort(key=lambda entry: (entry["id"], entry["sample_index"]))
    summary = {
        "prompts": len(rows),
        "sampled_solutions": total_sampled,
        "verified_solutions": total_verified,
        "kept_solutions": len(accepted),
        "verified_rate": total_verified / total_sampled if total_sampled else 0.0,
        "dedup_retention": len(accepted) / total_verified if total_verified else 0.0,
        "prompts_with_a_kept_solution": sum(
            1 for entry in per_prompt if entry["kept"] > 0),
        "per_prompt": per_prompt,
    }
    return accepted, summary


def sample(args) -> int:
    if args.sample_source:
        rows = _sample_source(args.sample_source)
        if args.limit is not None:
            rows = rows[:args.limit]
        source = {
            "kind": "file",
            "path": args.sample_source,
            "sha256": sha256_file(args.sample_source),
        }
        tokenizer_sha = None
        model_dir = None
        device = "cpu"
        elapsed = 0.0
    else:
        started = time.time()
        torch, tokenizer, model, device = _load_generator(args.model_dir, args.device)
        datasets = (["gsm8k-train", "math-train"]
                    if args.dataset == "all-train" else [args.dataset])
        items, selection = select_items(
            datasets, args.limit, args.start, args.sampling, args.select_seed)
        rows = []
        for index, item in enumerate(items):
            samples = _generate(
                torch, tokenizer, model, item["prompt"], args.samples,
                args.temperature, args.top_p, args.max_new_tokens,
                args.seed + index, device)
            rows.append({**item, "samples": samples})
            if args.progress and (index + 1) % args.progress == 0:
                print(f"sampled {index + 1}/{len(items)}", flush=True)
        elapsed = time.time() - started
        source = {
            "kind": "generation",
            "policy": args.policy,
            "model_dir": args.model_dir,
            "device": device,
            "samples": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "tokens_per_second": (
                sum(len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                    + sum(len(tokenizer.encode(text, add_special_tokens=False))
                          for text in row["samples"])
                    for row in rows) / elapsed if elapsed > 0 else None),
        }
        tokenizer_sha = hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
        model_dir = args.model_dir

    dataset = args.dataset
    if args.sample_source:
        datasets = sorted({row.get("dataset", dataset) for row in rows})
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = "mixed:" + ",".join(datasets)
        selection = {"kind": "file", "requested": args.limit,
                     "selected": len(rows)}

    accepted, summary = _rejection(rows, dataset, args.curriculum)
    contamination = contamination_check(accepted)
    receipt = {
        "schema": "selvedge/s45-rft-sampling/v1",
        "kind": "rejection_sampling",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "dataset": dataset,
        "objective": args.objective,
        "policy": args.policy,
        "curriculum": args.curriculum,
        "verifier": {
            "definition": "one unambiguous extracted answer, compared exactly "
                          "(numeric 1e-9 relative, else canonical string)",
            "dataset": dataset,
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
        "source": source,
        "selection": selection,
        "tokenizer_vocab_sha256": tokenizer_sha,
        "model_dir": model_dir,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "contamination": contamination,
    }
    out_dir = args.out_dir
    os.makedirs(os.path.join(REPO, out_dir), exist_ok=True)
    accepted_path = os.path.join(REPO, out_dir, "accepted.jsonl")
    with open(accepted_path, "w", encoding="utf-8") as handle:
        for entry in accepted:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    receipt["accepted_path"] = os.path.join(out_dir, "accepted.jsonl")
    receipt["accepted_sha256"] = sha256_file(accepted_path)
    receipt_path = os.path.join(REPO, out_dir, "receipt.json")
    write_json(receipt_path, receipt)
    print(f"prompts={summary['prompts']} sampled={summary['sampled_solutions']} "
          f"verified={summary['verified_solutions']} kept={summary['kept_solutions']} "
          f"verified_rate={summary['verified_rate']:.4f} "
          f"dedup_retention={summary['dedup_retention']:.4f} "
          f"contamination_max={contamination['max_jaccard']:.4f}")
    print(f"receipt: {os.path.join(out_dir, 'receipt.json')}")
    if contamination["fail_closed"]:
        raise SystemExit("contamination check failed closed")
    return 0


# ---------------------------------------------------------------------------


def _self_test() -> int:
    cases = [
        ("#### 12", "12", "gsm8k", "correct"),
        ("#### 13", "12", "gsm8k", "wrong"),
        ("#### 1,000", "1000", "gsm8k", "correct"),
        ("#### 12 and #### 13", "12", "gsm8k", "ambiguous"),
        ("I do not know", "12", "gsm8k", "unparseable"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", "math", "correct"),
        ("\\boxed{\\dfrac{3}{4}}", "\\frac{3}{4}", "math", "correct"),
        ("\\boxed{0.5}", "\\frac{1}{2}", "math", "correct"),
        ("\\boxed{1}\\text{ or }\\boxed{2}", "1", "math", "ambiguous"),
        ("\\boxed{\\sqrt{2}}", "\\sqrt2", "math", "correct"),
    ]
    failures = 0
    for solution, target, dataset, expected in cases:
        got = verify(solution, target, dataset)["verdict"]
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"{status} {dataset} expected={expected} got={got} "
              f"solution={solution!r} target={target!r}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--dataset",
                               choices=["gsm8k-train", "math-train", "all-train"],
                               default="gsm8k-train")
    sample_parser.add_argument("--limit", type=int, default=None)
    sample_parser.add_argument("--start", type=int, default=0)
    sample_parser.add_argument("--sampling", choices=["prefix", "stratified"],
                               default="stratified",
                               help="corrected default: seeded stratified selection "
                                    "over source, difficulty and reasoning-length "
                                    "quartile; prefix is the old behaviour")
    sample_parser.add_argument("--select-seed", type=int, default=451)
    sample_parser.add_argument("--samples", type=int, default=8)
    sample_parser.add_argument("--policy", choices=["teacher", "student"],
                               default="teacher")
    sample_parser.add_argument("--objective", choices=["sft", "kd"], default="sft")
    sample_parser.add_argument("--model-dir", default=None)
    sample_parser.add_argument("--device", default=None)
    sample_parser.add_argument("--temperature", type=float, default=0.8)
    sample_parser.add_argument("--top-p", type=float, default=0.95)
    sample_parser.add_argument("--max-new-tokens", type=int, default=512)
    sample_parser.add_argument("--seed", type=int, default=451)
    sample_parser.add_argument("--curriculum", choices=["none", "prefix", "steps",
                                                       "prefix+steps"],
                               default="none")
    sample_parser.add_argument("--sample-source", default=None)
    sample_parser.add_argument("--out-dir", required=True)
    sample_parser.add_argument("--progress", type=int, default=100)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--solution", required=True)
    verify_parser.add_argument("--target", required=True)
    verify_parser.add_argument("--dataset", choices=["gsm8k", "math"],
                               required=True)

    contamination_parser = sub.add_parser("contamination")
    contamination_parser.add_argument("--accepted", required=True)

    sub.add_parser("self-test")
    args = parser.parse_args(argv)

    if args.command == "self-test":
        return _self_test()

    if args.command == "sample":
        if args.model_dir is None:
            args.model_dir = TEACHER_DIR if args.policy == "teacher" else STUDENT_DIR
        return sample(args)
    if args.command == "verify":
        print(json.dumps(verify(args.solution, args.target, args.dataset),
                         indent=2, sort_keys=True))
        return 0
    if args.command == "contamination":
        rows = _sample_source(os.path.join(REPO, args.accepted))
        report = contamination_check(rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["fail_closed"] else 0
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
