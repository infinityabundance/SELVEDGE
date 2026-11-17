#!/usr/bin/env python3
"""S45: diagnose the generative-math collapse of the stage-2 arms.

Two modes, both bounded and read-only with respect to earlier receipts:

``--artifacts``
    Reads the stored lm-evaluation-harness sample files for ``base``,
    ``memory-2m``, ``adapter-2m`` and ``adapter-20m`` on GSM8K and MATH-500,
    pairs every item by ``doc_id`` (and checks that ``prompt_hash`` is equal),
    and reports:

    * accuracy and the transitions ``base correct -> arm wrong`` /
      ``base wrong -> arm correct``;
    * generated-response length buckets and the accuracy delta in each;
    * the first character at which an arm's response diverges from the base's,
      its class (digit, operator, LaTeX, letter, space), its relative position,
      and whether it falls in the reasoning or in the final-answer region;
    * whether the extracted final answer changed, and whether the change is a
      digit change;
    * the subset whose responses are byte-identical, which is a control that
      must show a zero delta.

    This mode needs no torch and runs at a small memory limit.

``--forward``
    Teacher-forces a small, selected set of GSM8K prompts plus one arm's stored
    generation through base, ``memory-2m``, ``adapter-2m`` and the 2B teacher on
    CPU, and records per completion position: CE of each arm, KL(teacher||arm)
    at T=1, the memory arm's per-read gate magnitude and residual norm, the
    adapter residual norm, and the first position at which CE divergence
    crosses a threshold.  Prompt and completion windows are capped; the cap is
    written into the receipt and is identical for every arm.

Every number written by this script comes from these files; nothing is typed by
hand.  The JSON receipt names the raw files and their SHA-256 hashes, the git
revision, and the cgroup peak when the forward pass runs inside the court.

Usage
-----
    scripts/court python -- python3 scripts/s45_collapse_diagnosis.py --artifacts
    SELVEDGE_MEM=... scripts/court oracle -- python3 scripts/s45_collapse_diagnosis.py \
        --forward --items 3 --prompt-tokens 256 --completion-tokens 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = "results/stage2/eval"
OUT_DIR = "results/stage2"

RAW = {
    "base": {
        "gsm8k": "results/stage2/eval/base/raw/20261117T042753Z-gsm8k.json",
        "minerva_math500": "results/stage2/eval/base/raw/20261117T043134Z-minerva_math500.json",
    },
    "memory-2m": {
        "gsm8k": "results/stage2/eval/memory-2m/raw/20261117T055151Z-gsm8k.json",
        "minerva_math500": "results/stage2/eval/memory-2m/raw/20261117T055544Z-minerva_math500.json",
    },
    "adapter-2m": {
        "gsm8k": "results/stage2/eval/adapter-2m/raw/20261117T050638Z-gsm8k.json",
        "minerva_math500": "results/stage2/eval/adapter-2m/raw/20261117T051019Z-minerva_math500.json",
    },
    "adapter-20m": {
        "gsm8k": "results/stage2/eval/adapter-20m/raw/20261117T075433Z-gsm8k.json",
        "minerva_math500": "results/stage2/eval/adapter-20m/raw/20261117T075807Z-minerva_math500.json",
    },
}

FILTER = {"gsm8k": "strict-match", "minerva_math500": "none"}
SCHEMA = "selvedge/s45-collapse-diagnosis/v1"

#: forward-pass checkpoints, fixed by the stage-2 receipts.
MEMORY_CKPT = "results/stage2/runs/memory-2m/ckpt-000976.pt"
ADAPTER_CKPT = "results/stage2/runs/adapter-2m/ckpt-000976.pt"
STUDENT_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen", "Qwen3.5-0.8B-Base",
    "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
TEACHER_DIR = os.path.join(
    os.environ.get("HF_HOME", "/hf-cache"), "models", "Qwen", "Qwen3.5-2B-Base",
    "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")

GIB = float(1 << 30)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# artifacts


def load_items(path: str, filter_name: str, metric: str = "exact_match") -> dict[int, dict]:
    """One sample per doc_id, keeping only the registered filter and metric."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload["samples"]
    key = next(iter(samples))
    items: dict[int, dict] = {}
    for sample in samples[key]:
        if sample.get("filter") != filter_name:
            continue
        doc_id = int(sample["doc_id"])
        response = "".join(sample.get("resps", [[]])[0])
        value = sample.get(metric)
        if value is None:
            value = sample.get("exact_match", 0.0)
        items[doc_id] = {
            "doc_id": doc_id,
            "prompt": "".join(sample["arguments"][0][0]),
            "prompt_hash": sample.get("prompt_hash"),
            "target": sample.get("target"),
            "response": response,
            "exact_match": float(value),
        }
    return items


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def first_difference(left: str, right: str) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    if len(left) == len(right):
        return None
    return limit


def char_class(char: str | None) -> str:
    if char is None:
        return "end"
    if char.isdigit():
        return "digit"
    if char in "+-*/=<>^%":
        return "operator"
    if char == "\\":
        return "latex"
    if char.isspace():
        return "space"
    if char.isalpha():
        return "letter"
    return "other"


_GSM_ANSWER = re.compile(r"####\s*([^\n]+)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "").replace("$", "").replace(" ", "")
    value = value.rstrip(".")
    match = _NUMBER.search(value)
    if not match:
        return value
    text = match.group(0)
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return repr(number)


def extract_gsm_answer(text: str) -> str | None:
    matches = _GSM_ANSWER.findall(text)
    if matches:
        return normalize_number(matches[-1])
    numbers = _NUMBER.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def _balanced(text: str, start: int) -> str | None:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return None


def extract_math_answer(text: str) -> str | None:
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(positions):
        brace = text.index("{", start)
        answer = _balanced(text, brace)
        if answer is not None:
            return normalize_text(answer)
    markers = list(re.finditer(r"final answer is\s+(.+)", text, re.IGNORECASE))
    if markers:
        return normalize_text(markers[-1].group(1)).rstrip(".")
    return None


def answer_marker(text: str, task: str) -> int | None:
    if task == "gsm8k":
        positions = [match.start() for match in _GSM_ANSWER.finditer(text)]
        return positions[-1] if positions else None
    positions = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if positions:
        return positions[-1]
    positions = [match.start() for match in
                 re.finditer(r"final answer is", text, re.IGNORECASE)]
    return positions[-1] if positions else None


def extract_answer(text: str, task: str) -> str | None:
    return extract_gsm_answer(text) if task == "gsm8k" else extract_math_answer(text)


def digits_of(text: str | None) -> str | None:
    if text is None:
        return None
    return "".join(char for char in text if char.isdigit())


def analyse_task(task: str, arms: list[str]) -> dict:
    metric = "exact_match" if task == "gsm8k" else "math_verify"
    reference = load_items(os.path.join(REPO, RAW["base"][task]), FILTER[task], metric)
    loaded = {}
    for arm in arms:
        path = os.path.join(REPO, RAW[arm][task])
        loaded[arm] = load_items(path, FILTER[task], metric)
    doc_ids = sorted(reference)
    for arm in arms:
        if sorted(loaded[arm]) != doc_ids:
            raise SystemExit(f"{task}: {arm} doc ids differ from base")
        for doc_id in doc_ids:
            if loaded[arm][doc_id]["prompt_hash"] != reference[doc_id]["prompt_hash"]:
                raise SystemExit(f"{task}: {arm} prompt hash differs at doc {doc_id}")

    outputs = {}
    for arm in arms:
        losses = []
        gains = []
        identical = []
        by_length_bucket: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
        by_position_bucket: dict[str, list[dict]] = {
            "0-10%": [], "10-25%": [], "25-50%": [], "50-100%": []}
        divergence_classes: dict[str, int] = {}
        loss_divergence_classes: dict[str, int] = {}
        loss_reasoning_first = 0
        loss_answer_first = 0
        answer_changed = 0
        answer_digit_changed = 0
        reasoning_first = 0
        answer_first = 0
        records = []
        base_lengths = sorted(len(normalize_text(reference[d]["response"]))
                              for d in doc_ids)
        cuts = [base_lengths[int(0.25 * len(base_lengths))],
                base_lengths[int(0.50 * len(base_lengths))],
                base_lengths[int(0.75 * len(base_lengths))]]

        for doc_id in doc_ids:
            base = reference[doc_id]
            sample = loaded[arm][doc_id]
            base_ok = base["exact_match"] >= 0.5
            arm_ok = sample["exact_match"] >= 0.5
            base_text = normalize_text(base["response"])
            arm_text = normalize_text(sample["response"])
            same = base_text == arm_text
            diff = first_difference(base_text, arm_text)
            base_answer = extract_answer(base["response"], task)
            arm_answer = extract_answer(sample["response"], task)
            marker_base = answer_marker(base["response"], task)
            marker_arm = answer_marker(sample["response"], task)
            length = len(base_text)
            bucket = 0 if length <= cuts[0] else 1 if length <= cuts[1] \
                else 2 if length <= cuts[2] else 3

            entry = {
                "doc_id": doc_id,
                "base_ok": base_ok,
                "arm_ok": arm_ok,
                "identical": same,
                "base_length": length,
                "arm_length": len(arm_text),
                "first_difference": diff,
                "first_difference_class": char_class(
                    arm_text[diff] if diff is not None and diff < len(arm_text)
                    else None),
                "base_answer": base_answer,
                "arm_answer": arm_answer,
                "answer_changed": base_answer != arm_answer,
                "answer_digit_changed": digits_of(base_answer) != digits_of(arm_answer),
            }
            records.append(entry)
            by_length_bucket[bucket].append(entry)

            if base_ok and not arm_ok:
                losses.append(entry)
            if arm_ok and not base_ok:
                gains.append(entry)
            if same:
                identical.append(entry)
            if not same and diff is not None:
                divergence_classes[entry["first_difference_class"]] = \
                    divergence_classes.get(entry["first_difference_class"], 0) + 1
                relative = diff / max(1, min(len(base_text), len(arm_text)))
                key = "0-10%" if relative < 0.10 else "10-25%" if relative < 0.25 \
                    else "25-50%" if relative < 0.50 else "50-100%"
                by_position_bucket[key].append(entry)
                in_answer = (marker_arm is not None and diff >= marker_arm) or \
                            (marker_base is not None and diff >= marker_base)
                if in_answer:
                    answer_first += 1
                else:
                    reasoning_first += 1
                if base_ok and not arm_ok:
                    loss_divergence_classes[entry["first_difference_class"]] = \
                        loss_divergence_classes.get(entry["first_difference_class"], 0) + 1
                    if in_answer:
                        loss_answer_first += 1
                    else:
                        loss_reasoning_first += 1
            if base_ok and not arm_ok and not same:
                if base_answer != arm_answer:
                    answer_changed += 1
                if digits_of(base_answer) != digits_of(arm_answer):
                    answer_digit_changed += 1

        def bucket_summary(entries: list[dict]) -> dict:
            return {
                "n": len(entries),
                "base_accuracy": sum(1 for e in entries if e["base_ok"]) / max(1, len(entries)),
                "arm_accuracy": sum(1 for e in entries if e["arm_ok"]) / max(1, len(entries)),
                "delta": sum(1 for e in entries if e["arm_ok"])
                         / max(1, len(entries))
                         - sum(1 for e in entries if e["base_ok"]) / max(1, len(entries)),
                "answer_changed_fraction": sum(1 for e in entries if e["answer_changed"])
                                           / max(1, len(entries)),
            }

        losses_diverged = [e for e in losses if not e["identical"]]
        outputs[arm] = {
            "n": len(records),
            "accuracy": sum(1 for e in records if e["arm_ok"]) / len(records),
            "base_accuracy": sum(1 for e in records if e["base_ok"]) / len(records),
            "transitions": {
                "base_correct_arm_wrong": len(losses),
                "base_wrong_arm_correct": len(gains),
                "net": len(gains) - len(losses),
            },
            "identical_responses": {
                **bucket_summary(identical),
                "base_correct_arm_wrong": sum(
                    1 for e in identical if e["base_ok"] and not e["arm_ok"]),
            },
            "losses": {
                "n": len(losses),
                "diverged": len(losses_diverged),
                "identical": len(losses) - len(losses_diverged),
                "answer_changed": answer_changed,
                "answer_digit_changed": answer_digit_changed,
                "first_divergence_in_reasoning": loss_reasoning_first,
                "first_divergence_in_answer": loss_answer_first,
                "first_divergence_classes": loss_divergence_classes,
                "all_diverged_first_divergence_in_reasoning": reasoning_first,
                "all_diverged_first_divergence_in_answer": answer_first,
                "all_diverged_first_divergence_classes": divergence_classes,
                "first_divergence_examples": [
                    {
                        "doc_id": e["doc_id"],
                        "class": e["first_difference_class"],
                        "relative": round(e["first_difference"]
                                          / max(1, min(e["base_length"], e["arm_length"])), 4),
                        "snippet": normalize_text(
                            loaded[arm][e["doc_id"]]["response"])[
                            max(0, e["first_difference"] - 30):e["first_difference"] + 30],
                    }
                    for e in losses_diverged[:10]
                ],
            },
            "by_base_length_quartile": {
                f"q{index + 1}": bucket_summary(entries)
                for index, entries in by_length_bucket.items()
            },
            "by_first_divergence_position": {
                key: bucket_summary(entries)
                for key, entries in by_position_bucket.items()
            },
            "items": records,
        }
    return {
        "task": task,
        "filter": FILTER[task],
        "metric": metric,
        "n": len(doc_ids),
        "answer_extractor": "last `#### value` (GSM8K) / last `\\boxed{}` else "
                            "'final answer is' (MATH-500); numbers normalised",
        "arms": outputs,
    }


def run_artifacts(args) -> int:
    arms = ["memory-2m", "adapter-2m", "adapter-20m"]
    result = {
        "schema": SCHEMA,
        "kind": "artifact_generative_collapse",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "raw": {
            arm: {task: {"path": RAW[arm][task],
                         "sha256": sha256_file(os.path.join(REPO, RAW[arm][task]))}
                  for task in RAW[arm]}
            for arm in ["base"] + arms
        },
        "tasks": {},
    }
    for task in ("gsm8k", "minerva_math500"):
        result["tasks"][task] = analyse_task(task, arms)
    path = args.out or os.path.join(
        OUT_DIR, f"s45-collapse-artifacts-{result['stamp']}.json")
    write_json(os.path.join(REPO, path), result)
    print_summary(result)
    print(f"receipt: {path}")
    return 0


def print_summary(result: dict) -> None:
    for task, payload in result["tasks"].items():
        print(f"== {task} n={payload['n']} filter={payload['filter']}")
        for arm, data in payload["arms"].items():
            losses = data["losses"]
            print(f"  {arm}: acc {data['accuracy']:.4f} vs base {data['base_accuracy']:.4f} "
                  f"| -{data['transitions']['base_correct_arm_wrong']} "
                  f"+{data['transitions']['base_wrong_arm_correct']} "
                  f"| identical {data['identical_responses']['n']} "
                  f"(delta {data['identical_responses']['delta']:+.4f})")
            print(f"    losses: {losses['n']} (identical {losses['identical']}, "
                  f"answer changed {losses['answer_changed']}, digit changed "
                  f"{losses['answer_digit_changed']}) first diff "
                  f"reasoning {losses['first_divergence_in_reasoning']} / "
                  f"answer {losses['first_divergence_in_answer']} | "
                  f"classes {losses['first_divergence_classes']}")
            print("    by length: " + ", ".join(
                f"{k} {v['delta']:+.4f} (n={v['n']})"
                for k, v in data["by_base_length_quartile"].items()))
            print("    by first-diff position: " + ", ".join(
                f"{k} {v['delta']:+.4f} (n={v['n']})"
                for k, v in data["by_first_divergence_position"].items()))


# ---------------------------------------------------------------------------
# post-analysis of a stored forward receipt (no torch)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _class_summary(values: list[float], classes: list[str]) -> dict:
    grouped: dict[str, list[float]] = {}
    for value, klass in zip(values, classes):
        grouped.setdefault(klass, []).append(value)
    return {klass: {"n": len(items), "mean": sum(items) / len(items)}
            for klass, items in sorted(grouped.items())}


def analyse_forward_payload(payload: dict, thresholds: list[float]) -> dict:
    rows = []
    for item in payload["items"]:
        positions = item["positions"]
        tokens = positions["token"]
        classes = positions["class"]
        delta = positions["ce_delta_memory_minus_base"]
        kl_delta = [m - b for m, b in zip(positions["kl_memory_teacher"],
                                          positions["kl_base_teacher"])]
        gate = positions.get("memory_gate_mean_abs") or []
        norm = positions.get("memory_residual_norm") or []
        cumulative = []
        running = 0.0
        for value in delta:
            running += value
            cumulative.append(running)
        marker = None
        for index, token in enumerate(tokens):
            text = token.strip()
            if text == "####" or "boxed" in text:
                marker = index
                break
        before = delta[:marker] if marker is not None else delta
        after = delta[marker:] if marker is not None else []
        row = {
            "doc_id": item["doc_id"],
            "source": item["source"],
            "base_ok": item["base_ok"],
            "memory_ok": item["memory_ok"],
            "completion_tokens": len(delta),
            "ce_delta_sum": sum(delta),
            "ce_delta_mean": sum(delta) / len(delta),
            "ce_delta_positive_positions": sum(1 for value in delta if value > 0),
            "ce_delta_negative_positions": sum(1 for value in delta if value < 0),
            "ce_delta_max": max(delta),
            "ce_delta_min": min(delta),
            "ce_delta_by_class": _class_summary(delta, classes),
            "kl_delta_sum": sum(kl_delta),
            "kl_delta_mean": sum(kl_delta) / len(kl_delta),
            "kl_delta_by_class": _class_summary(kl_delta, classes),
            "ce_delta_first_10": sum(delta[:10]),
            "ce_delta_first_20": sum(delta[:20]),
            "ce_delta_after_first_10": sum(delta[10:]),
            "ce_delta_before_answer_marker": (sum(before) / len(before)) if before else None,
            "ce_delta_from_answer_marker": (sum(after) / len(after)) if after else None,
            "answer_marker_index": marker,
            "pearson_position_vs_ce_delta": _pearson(
                list(range(len(delta))), delta),
            "pearson_gate_vs_abs_ce_delta": _pearson(
                gate, [abs(value) for value in delta]) if gate else None,
            "memory_gate_mean_abs": (sum(gate) / len(gate)) if gate else None,
            "memory_gate_min_abs": min(gate) if gate else None,
            "memory_gate_max_abs": max(gate) if gate else None,
            "memory_residual_norm_mean": (sum(norm) / len(norm)) if norm else None,
            "cumulative_ce_delta": cumulative,
            "top_positive_positions": [
                {"position": index, "token": tokens[index],
                 "class": classes[index], "delta": delta[index]}
                for index in sorted(range(len(delta)),
                                    key=lambda i: delta[i], reverse=True)[:5]
            ],
        }
        for threshold in thresholds:
            row[f"fraction_abs_delta_ge_{threshold}"] = sum(
                1 for value in delta if abs(value) >= threshold) / len(delta)
            row[f"first_abs_delta_ge_{threshold}"] = next(
                (index for index, value in enumerate(delta)
                 if abs(value) >= threshold), None)
        rows.append(row)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["source"], []).append(row)
    aggregate = {}
    for source, entries in groups.items():
        aggregate[source] = {
            "items": len(entries),
            "mean_ce_delta_mean": sum(r["ce_delta_mean"] for r in entries) / len(entries),
            "total_ce_delta_sum": sum(r["ce_delta_sum"] for r in entries),
            "mean_fraction_positive": sum(
                r["ce_delta_positive_positions"] / r["completion_tokens"]
                for r in entries) / len(entries),
            "mean_ce_delta_first_10": sum(r["ce_delta_first_10"] for r in entries)
            / len(entries),
            "mean_ce_delta_after_first_10": sum(r["ce_delta_after_first_10"]
                                                for r in entries) / len(entries),
            "mean_gate_abs": sum(r["memory_gate_mean_abs"] for r in entries
                                 if r["memory_gate_mean_abs"] is not None)
            / max(1, sum(1 for r in entries if r["memory_gate_mean_abs"] is not None)),
            "mean_residual_norm": sum(r["memory_residual_norm_mean"] for r in entries
                                      if r["memory_residual_norm_mean"] is not None)
            / max(1, sum(1 for r in entries
                         if r["memory_residual_norm_mean"] is not None)),
        }
    return {"aggregate": aggregate, "items": rows}


def run_analyse_forward(args) -> int:
    with open(os.path.join(REPO, args.analyse_forward), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    thresholds = [float(value) for value in args.analyse_thresholds.split(",")]
    analysis = analyse_forward_payload(payload, thresholds)
    result = {
        "schema": SCHEMA,
        "kind": "forward_positional_analysis",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "source_receipt": args.analyse_forward,
        "source_receipt_sha256": sha256_file(
            os.path.join(REPO, args.analyse_forward)),
        "thresholds": thresholds,
        **analysis,
    }
    path = args.out or os.path.join(
        OUT_DIR, f"s45-forward-analysis-{result['stamp']}.json")
    write_json(os.path.join(REPO, path), result)
    for source, summary in analysis["aggregate"].items():
        print(f"== source={source} items={summary['items']} "
              f"mean CE delta {summary['mean_ce_delta_mean']:+.4f} nats/token, "
              f"total {summary['total_ce_delta_sum']:+.3f} nats, "
              f"positive positions {summary['mean_fraction_positive']:.1%}, "
              f"first-10 {summary['mean_ce_delta_first_10']:+.3f}, "
              f"after-10 {summary['mean_ce_delta_after_first_10']:+.3f}, "
              f"gate |a| {summary['mean_gate_abs']:.3f}, "
              f"residual norm {summary['mean_residual_norm']:.3f}")
    for row in analysis["items"]:
        print(f"  doc {row['doc_id']} {row['source']}: mean {row['ce_delta_mean']:+.4f} "
              f"sum {row['ce_delta_sum']:+.3f} pos {row['ce_delta_positive_positions']}/"
              f"{row['completion_tokens']} pearson(pos) {row['pearson_position_vs_ce_delta']} "
              f"classes " + ", ".join(
                  f"{k} {v['mean']:+.3f}" for k, v in row["ce_delta_by_class"].items()))
    print(f"receipt: {path}")
    return 0


# ---------------------------------------------------------------------------
# forward pass (CPU, bounded)


def cgroup_memory() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    return report


def rss_peak_bytes() -> int | None:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def _select_items(tok, args) -> list[dict]:
    """Shortest GSM8K prompts where base is correct and memory-2m is wrong."""
    base = load_items(os.path.join(REPO, RAW["base"]["gsm8k"]), "strict-match")
    memory = load_items(os.path.join(REPO, RAW["memory-2m"]["gsm8k"]), "strict-match")
    candidates = []
    controls = []
    for doc_id, item in base.items():
        other = memory[doc_id]
        base_ok = item["exact_match"] >= 0.5
        memory_ok = other["exact_match"] >= 0.5
        prompt_ids = tok(item["prompt"], add_special_tokens=False).input_ids
        if len(prompt_ids) < 64:
            continue
        record = {
            "doc_id": doc_id,
            "prompt": item["prompt"],
            "base_response": item["response"],
            "memory_response": other["response"],
            "base_ok": base_ok,
            "memory_ok": memory_ok,
            "prompt_tokens": len(prompt_ids),
        }
        if base_ok and not memory_ok:
            candidates.append(record)
        elif base_ok and memory_ok:
            controls.append(record)
    candidates.sort(key=lambda r: r["prompt_tokens"])
    controls.sort(key=lambda r: r["prompt_tokens"])
    selected = candidates[:args.items]
    if args.controls and controls:
        selected = selected + controls[:args.controls]
    return selected


def _build_sequence(tok, item: dict, prompt_tokens: int, completion_tokens: int,
                    source: str):
    prompt = item["prompt"]
    completion = item[f"{source}_response"]
    full_ids = tok(prompt + completion, add_special_tokens=False).input_ids
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    split = 0
    limit = min(len(prompt_ids), len(full_ids))
    while split < limit and prompt_ids[split] == full_ids[split]:
        split += 1
    if split < max(1, int(0.8 * len(prompt_ids))):
        raise SystemExit(f"doc {item['doc_id']}: prompt tokenisation is not a prefix")
    completion_ids = full_ids[split:]
    if not completion_ids:
        return None
    prompt_window = full_ids[max(0, split - prompt_tokens):split]
    completion_ids = completion_ids[:completion_tokens]
    ids = prompt_window + completion_ids
    return {
        "input_ids": ids,
        "prompt_len": len(prompt_window),
        "completion_len": len(completion_ids),
        "completion_ids": completion_ids,
    }


def _load_models():
    import torch
    sys.path.insert(0, os.path.join(REPO, "oracle"))
    from frontier import arms, memory

    student = arms.load_model(STUDENT_DIR, dtype=torch.bfloat16, device="cpu",
                              attn_implementation="eager")
    teacher = arms.load_model(TEACHER_DIR, dtype=torch.bfloat16, device="cpu",
                              attn_implementation="eager")
    mem_payload = torch.load(os.path.join(REPO, MEMORY_CKPT), map_location="cpu",
                             weights_only=False)
    mem_module = memory.ConditionalMemory(hidden=1024, value_dim=256, rows=65536,
                                          sites=(2,), conv_kernel=4,
                                          table_dtype=torch.float32, seed=0)
    mem_module.load_state_dict(mem_payload["arm_state"], strict=True)
    mem_module = mem_module.to("cpu")
    adapter_payload = torch.load(os.path.join(REPO, ADAPTER_CKPT),
                                 map_location="cpu", weights_only=False)
    adapter = arms.DenseAdapter(hidden=1024)
    adapter.load_state_dict(adapter_payload["arm_state"], strict=True)
    adapter = adapter.to("cpu")
    return torch, arms, memory, student, teacher, mem_module, adapter


def _logits(torch, model, ids):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long))
    return out.logits[0].float()


def _ce(position_logits, targets):
    import torch.nn.functional as F
    return F.cross_entropy(position_logits, targets, reduction="none")


def _kl(teacher_logits, student_logits):
    import torch.nn.functional as F
    log_t = F.log_softmax(teacher_logits.float(), dim=-1)
    log_s = F.log_softmax(student_logits.float(), dim=-1)
    return (log_t.exp() * (log_t - log_s)).sum(-1)


def run_forward(args) -> int:
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(STUDENT_DIR, trust_remote_code=False)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_DIR, trust_remote_code=False)
    if tok.get_vocab() != teacher_tok.get_vocab():
        raise SystemExit("student and teacher tokenizers differ")

    items = _select_items(tok, args)
    sequences = []
    for item in items:
        for source in args.source:
            built = _build_sequence(tok, item, args.prompt_tokens,
                                    args.completion_tokens, source)
            if built is not None:
                sequences.append((item, source, built))

    torch, arms, memory, student, teacher, mem_module, adapter = _load_models()
    load_rss_peak = rss_peak_bytes()
    base_model = student
    results = {
        "schema": SCHEMA,
        "kind": "forward_positional_diagnosis",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": revision(),
        "device": "cpu",
        "dtype": "bfloat16",
        "attention": "eager",
        "prompt_tokens_cap": args.prompt_tokens,
        "completion_tokens_cap": args.completion_tokens,
        "threshold_nats": args.threshold,
        "checkpoints": {
            "memory": {"path": MEMORY_CKPT,
                        "sha256": sha256_file(os.path.join(REPO, MEMORY_CKPT))},
            "adapter": {"path": ADAPTER_CKPT,
                         "sha256": sha256_file(os.path.join(REPO, ADAPTER_CKPT))},
        },
        "student": STUDENT_DIR,
        "teacher": TEACHER_DIR,
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest(),
        "cgroup": cgroup_memory(),
        "rss_peak_after_load_bytes": load_rss_peak,
        "items": [],
    }

    for item, source, built in sequences:
        ids = built["input_ids"]
        prompt_len = built["prompt_len"]
        completion_len = built["completion_len"]
        if completion_len < 2:
            continue
        input_tensor = torch.tensor([ids], dtype=torch.long)
        targets = torch.tensor(ids[prompt_len:], dtype=torch.long)
        positions = list(range(prompt_len - 1, prompt_len - 1 + completion_len))

        base_logits = _logits(torch, base_model, ids)
        teacher_logits = _logits(torch, teacher, ids)

        memory_logits = None
        gate_mean = gate_max = mem_norm = None
        handles = arms.install_memory(base_model, mem_module)
        try:
            stats = {}

            def hook(module, inputs, output):
                stats["norm"] = output.detach().float().norm(dim=-1)[0]
                if module.last_gate is not None:
                    gate = module.last_gate.float().abs()
                    stats["gate_mean"] = gate.mean(dim=-1)[0]
                    stats["gate_max"] = gate.amax(dim=-1)[0]

            handle = mem_module.register_forward_hook(hook)
            mem_module.reset_state()
            memory_logits = _logits(torch, base_model, ids)
            handle.remove()
            if "norm" in stats:
                mem_norm = stats["norm"][positions].tolist()
                gate_mean = stats["gate_mean"][positions].tolist()
                gate_max = stats["gate_max"][positions].tolist()
        finally:
            arms.uninstall(handles)
            mem_module.reset_state()

        adapter_logits = None
        adapter_norm = None
        handles = arms.install_adapter(base_model, adapter, (2,))
        try:
            stats = {}

            def adapter_hook(module, inputs, output):
                stats["norm"] = output.detach().float().norm(dim=-1)[0]

            handle = adapter.register_forward_hook(adapter_hook)
            adapter_logits = _logits(torch, base_model, ids)
            handle.remove()
            if "norm" in stats:
                adapter_norm = stats["norm"][positions].tolist()
        finally:
            arms.uninstall(handles)

        def slice_logits(logits):
            return logits[positions]

        b = slice_logits(base_logits)
        m = slice_logits(memory_logits)
        a = slice_logits(adapter_logits)
        t = slice_logits(teacher_logits)
        ce_base = _ce(b, targets).tolist()
        ce_mem = _ce(m, targets).tolist()
        ce_adapter = _ce(a, targets).tolist()
        ce_teacher = _ce(t, targets).tolist()
        kl_base = _kl(t, b).tolist()
        kl_mem = _kl(t, m).tolist()
        kl_adapter = _kl(t, a).tolist()

        delta = [mem - base for mem, base in zip(ce_mem, ce_base)]
        first_ce_divergence = next(
            (index for index, value in enumerate(delta) if abs(value) >= args.threshold),
            None)
        token_text = [tok.decode([token_id]) for token_id in built["completion_ids"]]
        token_class = [char_class(text[0] if text else None) for text in token_text]

        def by_position(values: list[float]) -> dict:
            buckets: dict[str, list[float]] = {
                "0-4": [], "5-9": [], "10-19": [], "20+": []}
            for index, value in enumerate(values):
                key = "0-4" if index < 5 else "5-9" if index < 10 \
                    else "10-19" if index < 20 else "20+"
                buckets[key].append(value)
            return {key: {"n": len(vals),
                          "mean": sum(vals) / len(vals) if vals else None}
                    for key, vals in buckets.items()}

        results["items"].append({
            "doc_id": item["doc_id"],
            "source": source,
            "base_ok": item["base_ok"],
            "memory_ok": item["memory_ok"],
            "prompt_tokens_used": prompt_len,
            "completion_tokens_used": completion_len,
            "first_ce_divergence": first_ce_divergence,
            "first_ce_divergence_token": None if first_ce_divergence is None
            else token_text[first_ce_divergence],
            "first_ce_divergence_class": None if first_ce_divergence is None
            else token_class[first_ce_divergence],
            "positions": {
                "token": token_text,
                "class": token_class,
                "ce_base": ce_base,
                "ce_memory": ce_mem,
                "ce_adapter": ce_adapter,
                "ce_teacher": ce_teacher,
                "kl_base_teacher": kl_base,
                "kl_memory_teacher": kl_mem,
                "kl_adapter_teacher": kl_adapter,
                "ce_delta_memory_minus_base": delta,
                "memory_gate_mean_abs": gate_mean,
                "memory_gate_max_abs": gate_max,
                "memory_residual_norm": mem_norm,
                "adapter_residual_norm": adapter_norm,
            },
            "summary": {
                "ce_base_mean": sum(ce_base) / len(ce_base),
                "ce_memory_mean": sum(ce_mem) / len(ce_mem),
                "ce_adapter_mean": sum(ce_adapter) / len(ce_adapter),
                "ce_teacher_mean": sum(ce_teacher) / len(ce_teacher),
                "ce_delta_memory_minus_base": by_position(delta),
                "kl_base_teacher_mean": sum(kl_base) / len(kl_base),
                "kl_memory_teacher_mean": sum(kl_mem) / len(kl_mem),
                "kl_adapter_teacher_mean": sum(kl_adapter) / len(kl_adapter),
                "kl_delta_memory_minus_base": by_position(
                    [x - y for x, y in zip(kl_mem, kl_base)]),
                "memory_gate_mean_abs": None if gate_mean is None
                else sum(gate_mean) / len(gate_mean),
                "memory_residual_norm_mean": None if mem_norm is None
                else sum(mem_norm) / len(mem_norm),
                "adapter_residual_norm_mean": None if adapter_norm is None
                else sum(adapter_norm) / len(adapter_norm),
            },
        })
        print(f"doc {item['doc_id']} source={source} "
              f"P={prompt_len} C={completion_len} "
              f"CE base/mem/adapt/teacher="
              f"{results['items'][-1]['summary']['ce_base_mean']:.3f}/"
              f"{results['items'][-1]['summary']['ce_memory_mean']:.3f}/"
              f"{results['items'][-1]['summary']['ce_adapter_mean']:.3f}/"
              f"{results['items'][-1]['summary']['ce_teacher_mean']:.3f} "
              f"first CE divergence={first_ce_divergence}")

    results["cgroup"] = cgroup_memory()
    results["rss_peak_bytes"] = rss_peak_bytes()
    path = args.out or os.path.join(
        OUT_DIR, f"s45-forward-positional-{results['stamp']}.json")
    write_json(os.path.join(REPO, path), results)
    print(f"receipt: {path}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--analyse-forward", default=None,
                        help="post-analyse a stored --forward receipt")
    parser.add_argument("--analyse-thresholds", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--out", default=None)
    parser.add_argument("--items", type=int, default=3)
    parser.add_argument("--controls", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--completion-tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--source", action="append", default=None,
                        help="response source(s): base and/or memory (default memory)")
    args = parser.parse_args(argv)
    if args.source is None:
        args.source = ["memory"]
    if args.artifacts and args.forward:
        raise SystemExit("choose one mode")
    if args.artifacts:
        return run_artifacts(args)
    if args.analyse_forward:
        return run_analyse_forward(args)
    if args.forward:
        return run_forward(args)
    print(__doc__)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
