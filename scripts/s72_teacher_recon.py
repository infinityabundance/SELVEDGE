#!/usr/bin/env python3
"""S72 stage 0 prerequisite: receipt recon of teacher quality.

Reads only files already in the repository. It never loads a model, never
touches CUDA and never generates a token. It answers two questions:

1. which instruments have we *already* measured, in our S54 protocol, on the
   distillation teacher (``Qwen/Qwen3.5-2B``) and on every other locally cached
   candidate; and
2. how was ``Qwen/Qwen3-0.6B``'s MATH-500 = 48.80 produced: which extraction
   basis, which cap, how many of its answers came from the non-boxed fallback,
   how many completions were truncated, and what strict / flexible look like on
   the same 500 problems as our soup.

Unmeasured cells are reported as ``null`` (printed ``unknown``); nothing is
inferred. The extraction ``method`` is recomputed from the stored completions
with the same :mod:`frontier.rft` code the scorer used: the receipt's
per-problem rows do not keep the method string, so the method census is a
deterministic replay of ``extract_strict``/``extract_flexible``
(``scripts/s54_post_trained_eval.py`` lines 276-310), not a new generation.

Usage (CPU only, inside the python court):

    SELVEDGE_MEM=4g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
      scripts/court python scripts/s72_teacher_recon.py \
      --out results/stage2/s72-recon/teacher-recon.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s72_rft", os.path.join(ROOT, "oracle", "frontier", "rft.py"))
rft = importlib.util.module_from_spec(_spec)
sys.modules["s72_rft"] = rft
_spec.loader.exec_module(rft)

TEACHERS = {
    "Qwen/Qwen3.5-2B": "distillation teacher (stage-1 control)",
    "Qwen/Qwen3.5-2B-Base": "teacher base (not instruction-tuned)",
    "Qwen/Qwen3.5-0.8B": "post-trained student / soup base",
    "Qwen/Qwen3.5-4B": "candidate: config+tokenizer only on disk",
    "Qwen/Qwen3.5-9B": "candidate: config+tokenizer only on disk",
    "Qwen/Qwen3-4B": "literature candidate: not downloaded",
    "Qwen/Qwen3-1.7B": "cached, never scored in our protocol",
    "Qwen/Qwen3-0.6B": "external position check",
    "Qwen/Qwen2.5-Math-7B-Instruct": "literature candidate: not present",
    "Qwen/Qwen2.5-Math-1.5B-Instruct": "literature candidate: not present",
    "amd/ReasonLite-0.6B": "external position check (lower bound)",
}

CANDIDATE_PREFIXES = ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
                      "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B",
                      "Qwen/Qwen2.5-Math", "amd/ReasonLite")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _accuracy(block) -> float | None:
    if isinstance(block, dict):
        return block.get("accuracy")
    if isinstance(block, (int, float)):
        return float(block)
    return None


def scan_receipts() -> list[dict]:
    """Every receipt in the tree that carries a per-task accuracy block."""
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.json"),
                                 recursive=True)):
        try:
            if os.path.getsize(path) > 20_000_000:
                continue
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:  # noqa: BLE001 - not every JSON in the tree is a receipt
            continue
        if not isinstance(payload, dict):
            continue
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            continue
        if not all(isinstance(task, dict) and "task" in task for task in tasks):
            continue
        model = payload.get("model")
        model_id = model.get("id") if isinstance(model, dict) else model
        protocol = payload.get("protocol") or {}
        for task in tasks:
            flexible = _accuracy(task.get("flexible"))
            strict = _accuracy(task.get("strict"))
            exact = _accuracy(task.get("exact"))
            if flexible is None and strict is None and exact is None:
                continue
            rows.append({
                "receipt": os.path.relpath(path, ROOT),
                "receipt_sha256": sha256_file(path),
                "stamp": payload.get("stamp") or payload.get("created_utc"),
                "revision": payload.get("revision"),
                "model_id": model_id,
                "task": task.get("task"),
                "items": task.get("items") or task.get("distinct_problems"),
                "flexible": flexible,
                "strict": strict,
                "exact": exact,
                "mode": task.get("mode"),
                "max_new_tokens": protocol.get("max_new_tokens"),
                "truncated_samples": task.get("truncated_samples"),
                "truncated_fraction": task.get("truncated_fraction"),
            })
    return rows


# ---------------------------------------------------------------------------
# 0.6B extraction audit


def _distinct(values: list[str]) -> list[str]:
    distinct: list[str] = []
    for candidate in values:
        if not any(rft.answers_equal(candidate, other) for other in distinct):
            distinct.append(candidate)
    return distinct


def extract_strict(text: str) -> dict:
    """Identical to scripts/s54_post_trained_eval.py::extract_strict."""
    answers = rft._boxed_answers(text)
    if not answers:
        return {"value": None, "method": "none", "status": "unparseable"}
    distinct = _distinct(answers)
    if len(distinct) > 1:
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous"}
    if len(answers) == 1 and rft._OR.search(answers[0]):
        return {"value": None, "method": "last \\boxed{}", "status": "ambiguous"}
    return {"value": answers[-1], "method": "last \\boxed{}", "status": "ok"}


def extract_flexible(text: str, task: str) -> dict:
    """Identical to scripts/s54_post_trained_eval.py::extract_flexible."""
    boxed = extract_strict(text)
    if boxed["value"] is not None:
        return boxed
    if boxed["status"] == "ambiguous":
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
            "status": status}


def audit_raw(path: str, task: str) -> dict:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    methods = collections.Counter()
    strict_methods = collections.Counter()
    fallback_used = 0
    fallback_correct = 0
    strict_correct = 0
    flexible_correct = 0
    truncated = 0
    new_tokens = []
    for row in rows:
        completion = row.get("completion") or ""
        strict = extract_strict(completion)
        flexible = extract_flexible(completion, task)
        strict_methods[strict["method"]] += 1
        methods[flexible["method"]] += 1
        if strict["value"] is None and flexible["value"] is not None:
            fallback_used += 1
            if row.get("flexible_correct"):
                fallback_correct += 1
        strict_correct += bool(row.get("strict_correct"))
        flexible_correct += bool(row.get("flexible_correct"))
        truncated += bool(row.get("truncated"))
        new_tokens.append(int(row.get("new_tokens") or 0))
    new_tokens.sort()
    count = len(rows)

    def pct(index: float) -> int:
        return new_tokens[min(count - 1, int(index * count))]

    return {
        "path": os.path.relpath(path, ROOT),
        "sha256": sha256_file(path),
        "task": task,
        "rows": count,
        "strict_correct": strict_correct,
        "flexible_correct": flexible_correct,
        "strict_accuracy": strict_correct / count,
        "flexible_accuracy": flexible_correct / count,
        "strict_methods": dict(strict_methods),
        "flexible_methods": dict(methods),
        "fallback_used_but_strict_empty": fallback_used,
        "fallback_rescued_correct": fallback_correct,
        "truncated_samples": truncated,
        "truncated_fraction": truncated / count,
        "new_tokens_p50": pct(0.50),
        "new_tokens_p90": pct(0.90),
        "new_tokens_p99": pct(0.99),
        "new_tokens_max": new_tokens[-1],
        "new_tokens_at_or_above_cap_4096": sum(1 for value in new_tokens
                                               if value >= 4096),
    }


def paired_compare(left_path: str, right_path: str,
                   left_label: str, right_label: str) -> dict:
    left = {}
    with open(left_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                left[row["problem_id"]] = row
    right = {}
    with open(right_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                right[row["problem_id"]] = row
    shared = sorted(set(left) & set(right))
    cells = collections.Counter()
    strict_cells = collections.Counter()
    for key in shared:
        cells[(bool(left[key]["flexible_correct"]),
               bool(right[key]["flexible_correct"]))] += 1
        strict_cells[(bool(left[key]["strict_correct"]),
                      bool(right[key]["strict_correct"]))] += 1
    n = len(shared)
    return {
        "left": left_label,
        "right": right_label,
        "shared_problems": n,
        "flexible": {
            "left_correct": sum(1 for key in shared
                                if left[key]["flexible_correct"]),
            "right_correct": sum(1 for key in shared
                                 if right[key]["flexible_correct"]),
            "both_correct": cells[(True, True)],
            "left_only": cells[(True, False)],
            "right_only": cells[(False, True)],
            "neither": cells[(False, False)],
        },
        "strict": {
            "left_correct": sum(1 for key in shared
                                if left[key]["strict_correct"]),
            "right_correct": sum(1 for key in shared
                                 if right[key]["strict_correct"]),
            "both_correct": strict_cells[(True, True)],
            "left_only": strict_cells[(True, False)],
            "right_only": strict_cells[(False, True)],
            "neither": strict_cells[(False, False)],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/stage2/s72-recon/teacher-recon.json")
    parser.add_argument("--frontier-dir", default="results/stage2/s68-frontier")
    parser.add_argument("--soup-dir", default="results/stage2/s67-final")
    args = parser.parse_args()

    receipts = scan_receipts()
    candidates = [row for row in receipts
                  if row["model_id"] and row["model_id"].startswith(CANDIDATE_PREFIXES)]

    qwen06 = audit_raw(os.path.join(ROOT, args.frontier_dir,
                                    "qwen3-0.6b-raw",
                                    "math500-non-thinking-greedy.jsonl"), "math500")
    qwen06_gsm = audit_raw(os.path.join(ROOT, args.frontier_dir,
                                        "qwen3-0.6b-raw",
                                        "gsm8k-non-thinking-greedy.jsonl"), "gsm8k")
    soup_raw = os.path.join(ROOT, args.soup_dir, "soup-raw",
                            "math500-non-thinking-greedy.jsonl")
    paired = paired_compare(os.path.join(ROOT, args.frontier_dir,
                                         "qwen3-0.6b-raw",
                                         "math500-non-thinking-greedy.jsonl"),
                            soup_raw, "Qwen/Qwen3-0.6B", "S67 soup")
    soup_audit = audit_raw(soup_raw, "math500")

    disk = {}
    for name in ("Qwen3.5-2B", "Qwen3.5-2B-Base", "Qwen3.5-0.8B",
                 "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3-0.6B", "Qwen3-1.7B"):
        root = os.path.join(os.environ.get("HF_HOME", "/hf-cache"),
                            "models", "Qwen", name)
        weights = []
        for path in glob.glob(os.path.join(root, "*", "*.safetensors")):
            weights.append(os.path.basename(path))
        disk[name] = {"root": root, "weight_files": sorted(weights),
                      "weights_present": bool(weights)}

    receipt = {
        "schema": "selvedge/s72-teacher-recon/v1",
        "stamp": now_stamp(),
        "created_utc": now_iso(),
        "kind": "receipt-recon",
        "instrument": "S54 stage-1 (greedy, non-thinking, zero-shot CoT, "
                      "\\boxed{}, extraction basis reported per row)",
        "model_roles": TEACHERS,
        "candidate_receipts": candidates,
        "qwen3_0.6b_math500_audit": qwen06,
        "qwen3_0.6b_gsm8k_audit": qwen06_gsm,
        "soup_math500_audit": soup_audit,
        "qwen3_0.6b_vs_soup_math500_paired": paired,
        "local_weights": disk,
    }
    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    print(f"{len(candidates)} candidate task rows from existing receipts")
    exact_rows = [row["exact"] for row in candidates
                  if row["model_id"] == "Qwen/Qwen3-0.6B"
                  and row["task"] == "math500"]
    print("Qwen3-0.6B MATH-500: strict %.3f flexible %.3f exact %s"
          % (qwen06["strict_accuracy"], qwen06["flexible_accuracy"],
             exact_rows[0] if exact_rows else "unknown"))
    print("  fallback_used=%d fallback_rescued_correct=%d truncated=%d/%d"
          % (qwen06["fallback_used_but_strict_empty"],
             qwen06["fallback_rescued_correct"], qwen06["truncated_samples"],
             qwen06["rows"]))
    print("  flexible methods: %s" % qwen06["flexible_methods"])
    print("Qwen3-0.6B vs soup paired (math500): flexible %s" % paired["flexible"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
