#!/usr/bin/env python3
"""Pinned evaluation items and the registered prompt/extractor definitions.

All items come from the pinned dataset cache; no item text is committed. The
held-out KL windows are the exact token ids the M1/M2 split produced, read from
`results/artifacts/split/m1-split.json` (gitignored derived file whose SHA-256
is in the tracked split receipt).

The prompt templates and extractors here are part of the instrument. They are
fixed before measurement and recorded in every eval receipt.
"""

from __future__ import annotations

import json
import os
import re

HUMANEVAL_STOP = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


def load_split(artifact_dir: str, receipt_dir: str) -> tuple[dict, str]:
    path = os.path.join(artifact_dir, "split", "m1-split.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing derived split {path}; run oracle/dataset_prep.py once (S5 recipe)")
    import hashlib

    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    receipts = sorted(os.listdir(receipt_dir)) if os.path.isdir(receipt_dir) else []
    if not receipts:
        raise SystemExit(f"no split receipt in {receipt_dir}")
    return json.load(open(path, encoding="utf-8")), digest


def heldout_windows(split: dict) -> dict[str, list[dict]]:
    """Domain -> [{doc_id, window_index, tokens}] in a fixed order."""
    windows = split["windows"]
    result = {}
    for domain, key in (("general", "heldout_general"),
                        ("code", "heldout_code"),
                        ("math", "heldout_math")):
        items = []
        for document in windows[key]:
            for index, tokens in enumerate(document["windows"]):
                items.append({
                    "doc_id": document["doc_id"],
                    "window_index": index,
                    "tokens": [int(token) for token in tokens],
                })
        result[domain] = items
    return result


def load_humaneval(dataset_root: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "humaneval-test", "test-00000-of-00001.parquet")
    table = pq.read_table(path, columns=["task_id", "prompt", "canonical_solution",
                                        "test", "entry_point"]).to_pylist()
    items = []
    for row in sorted(table, key=lambda item: item["task_id"]):
        items.append({
            "id": row["task_id"],
            "prompt": row["prompt"],
            # The canonical solution is kept only so the execution harness can
            # be smoke-tested; it is never fed to a model.
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return items


def gsm8k_subset(dataset_root: str, count: int) -> list[dict]:
    import hashlib

    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "gsm8k-test", "test-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    keyed = []
    for row in rows:
        digest = hashlib.sha256(re.sub(r"\s+", " ", row["question"]).strip().encode()).hexdigest()
        keyed.append((digest, row))
    keyed.sort(key=lambda item: item[0])
    selected = []
    for digest, row in keyed[:count]:
        answer = row["answer"]
        if "####" not in answer:
            continue
        gold = answer.split("####")[-1].strip().replace(",", "")
        selected.append({"id": digest[:16], "question": row["question"], "gold": gold})
    return selected


def load_truthfulqa(dataset_root: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = os.path.join(dataset_root, "truthfulqa-mc-validation",
                        "validation-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["question", "mc1_targets"]).to_pylist()
    items = []
    for index, row in enumerate(rows):
        targets = row["mc1_targets"]
        items.append({
            "id": f"truthfulqa-{index:04d}",
            "question": row["question"],
            "choices": list(targets["choices"]),
            "labels": [int(value) for value in targets["labels"]],
        })
    return items


# ---------------------------------------------------------------------------
# prompt templates and extractors


def humaneval_prompt(item: dict) -> str:
    return item["prompt"]


def gsm8k_prompt(item: dict) -> str:
    return f"Question: {item['question']}\nAnswer:"


def truthfulqa_prompt(item: dict, choice: str) -> str:
    return f"Q: {item['question']}\nA: {choice}"


NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_gsm8k(text: str) -> str | None:
    """The registered GSM8K extractor (fixed before measurement).

    1. after the last `####`, the first number;
    2. else after the last case-insensitive `answer is` / `answer:` / `final
       answer` marker, the first number;
    3. else the last number in the completion.
    """
    if "####" in text:
        tail = text.split("####")[-1]
        match = NUMBER.search(tail)
        if match:
            return match.group(0).replace(",", "")
    lowered = text.lower()
    markers = ["final answer", "the answer is", "answer is", "answer:"]
    best = -1
    for marker in markers:
        best = max(best, lowered.rfind(marker))
    if best != -1:
        match = NUMBER.search(text[best:])
        if match:
            return match.group(0).replace(",", "")
    numbers = NUMBER.findall(text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def extract_code(completion: str) -> str:
    for stop in HUMANEVAL_STOP:
        position = completion.find(stop)
        if position != -1:
            completion = completion[:position]
    return completion
