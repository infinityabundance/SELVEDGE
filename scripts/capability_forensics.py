#!/usr/bin/env python3
"""Prompt forensics and response-entropy screen for the capability court (S30).

Run in the oracle court (it loads a tokenizer, no GPU and no model weights):

    SELVEDGE_MEM=4g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
    scripts/court oracle -- python3 scripts/capability_forensics.py \
      --suite fixtures/capability-calibrated \
      --generations D=results/capability/calibration/D/generations.json \
      --out results/capability/forensics-<stamp>.json

It writes one receipt: the verbatim rendered prompt for a few selection items,
the prompt audit (candidates present, labels unique, reference position, example
letter in the instruction), the tokenizer audit against the adapter's 4096-token
truncation, the measured answer distribution of each supplied arm and format, and
the per-format entropy screen. Nothing here scores an item.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import forensics, formats, suite as suite_module  # noqa: E402

SCHEMA = "selvedge/capability-forensics/v1"


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pairs(values: list, flag: str) -> dict:
    named = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        named[name] = path
    return named


def revision() -> str:
    """The commit this receipt was produced at (perf-number blocks cite it)."""
    import subprocess

    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tokenizer_helpers(hf_root: str):
    """(tokenize, decode, provenance) using the pinned Qwen3 tokenizer.

    The adapter tokenises with `AutoTokenizer(...)(prompt)` defaults, i.e. no chat
    template and the tokenizer's own truncation side. This mirrors exactly that,
    so a token count here is the count the adapter would use.
    """
    from transformers import AutoTokenizer

    path = os.path.join(hf_root, "models", "Qwen", "Qwen3-0.6B",
                        "c1899de289a04d12100db370d81485cdf75e47ca")
    if not os.path.isdir(path):
        raise SystemExit(f"tokenizer path {path} is absent")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                              trust_remote_code=False)

    def tokenize(text: str) -> list:
        return tokenizer(text, add_special_tokens=True)["input_ids"]

    def decode(ids: list) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    config = load(os.path.join(path, "tokenizer_config.json"))
    provenance = {
        "path": path,
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "add_bos_token": config.get("add_bos_token"),
        "bos_token": config.get("bos_token"),
        "eos_token": config.get("eos_token"),
        "pad_token": config.get("pad_token"),
        "has_chat_template": bool(config.get("chat_template")),
        "chat_template_used_at_generation": False,
        "model_max_length": config.get("model_max_length"),
    }
    if config.get("chat_template"):
        message = [{"role": "user", "content": "Reply with the single letter A."}]
        provenance["chat_template_would_render_example"] = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
    return tokenize, decode, provenance


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=os.path.join(REPO, "fixtures",
                                                        "capability-calibrated"))
    parser.add_argument("--generations", action="append", default=[],
                        metavar="ARM=PATH")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--patch-prompts", type=int, default=3)
    parser.add_argument("--diagnostic-prompts", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    collections = suite_module.load_suite(args.suite)
    tasks = collections["suite"]
    by_format: dict[str, list] = {}
    for task in tasks:
        by_format.setdefault(formats.task_format(task), []).append(task)
    for values in by_format.values():
        values.sort(key=lambda task: task.id)

    wanted = []
    wanted.extend(by_format.get(formats.FORMAT_PATCH, [])[:args.patch_prompts])
    wanted.extend(by_format.get(formats.FORMAT_DIAGNOSTIC, [])[:args.diagnostic_prompts])

    tokenize, decode, tokenizer_provenance = tokenizer_helpers(args.hf_root)

    prompts = []
    for task in wanted:
        prompt = suite_module.render_prompt(task)
        audit = forensics.prompt_audit(task, prompt)
        audit["tokenizer"] = forensics.tokenizer_audit(
            task, prompt, tokenize, forensics.ADAPTER_MAX_PROMPT_TOKENS, decode=decode)
        audit["prompt"] = prompt
        prompts.append(audit)

    # Token shapes for the whole calibration split, not only the dumped prompts:
    # the truncation question is about every item the adapter ran.
    shapes = []
    for task in tasks:
        prompt = suite_module.render_prompt(task)
        ids = list(tokenize(prompt))
        shapes.append({
            "id": task.id,
            "format": formats.task_format(task),
            "prompt_tokens": len(ids),
            "truncated_by_adapter": len(ids) > forensics.ADAPTER_MAX_PROMPT_TOKENS,
        })
    by_shape: dict = {}
    for row in shapes:
        entry = by_shape.setdefault(row["format"], {"items": 0, "max_tokens": 0,
                                                    "truncated": 0, "over_2048": 0})
        entry["items"] += 1
        entry["max_tokens"] = max(entry["max_tokens"], row["prompt_tokens"])
        entry["truncated"] += 1 if row["truncated_by_adapter"] else 0
        entry["over_2048"] += 1 if row["prompt_tokens"] > 2048 else 0

    generation_paths = pairs(args.generations, "--generations")
    arms = {}
    for arm, path in sorted(generation_paths.items()):
        receipt = load(path)
        outputs = {item["id"]: (item.get("output") or {}).get("text")
                   for item in receipt.get("items", [])}
        formats_seen = sorted({formats.task_format(task) for task in tasks})
        screens = {fmt: forensics.format_answer_screen(tasks, outputs, fmt)
                   for fmt in formats_seen}
        collapse = {
            "patch": forensics.answer_collapse(tasks, outputs, formats.FORMAT_PATCH),
            "diagnostic": forensics.answer_collapse(
                tasks, outputs, formats.FORMAT_DIAGNOSTIC),
        }
        arms[arm] = {
            "path": path,
            "sha256": sha256_file(path),
            "stamp": receipt.get("stamp"),
            "revision": receipt.get("revision"),
            "decoding": receipt.get("decoding"),
            "finish_reasons": sorted({
                (item.get("output") or {}).get("finish_reason")
                for item in receipt.get("items", [])}),
            "screens": screens,
            "collapse": collapse,
            "diagnosis": forensics.collapse_diagnosis(collapse["patch"],
                                                      collapse["diagnostic"]),
        }

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "created_utc": utc_now(),
        "revision": revision(),
        "command": " ".join(sys.argv),
        "suite": {
            "root": os.path.abspath(args.suite),
            "suite_sha256": suite_module.build_manifest(args.suite)["suite_sha256"],
        },
        "tokenizer": tokenizer_provenance,
        "adapter_truncation": {
            "max_prompt_tokens": forensics.ADAPTER_MAX_PROMPT_TOKENS,
            "truncation_side": "right (tokenizer default)",
            "in_scope": ("the tail of the prompt is the candidate list and the "
                         "answer-format instruction, so truncation removes the "
                         "question"),
            "by_format": by_shape,
            "max_prompt_tokens": max((row["prompt_tokens"] for row in shapes),
                                     default=0),
            "truncated_items": sum(1 for row in shapes if row["truncated_by_adapter"]),
        },
        "prompts": prompts,
        "arms": arms,
        "screen": {
            "min_answer_entropy_bits": forensics.MIN_ANSWER_ENTROPY_BITS,
            "max_modal_answer_share": forensics.MAX_MODAL_ANSWER_SHARE,
            "rule": ("a format whose answers are near-constant across items is "
                     "excluded for that arm, not scored"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    truncation = payload["adapter_truncation"]
    print(f"capability_forensics: wrote {args.out}")
    print(f"  prompt tokens: max {truncation['max_prompt_tokens']} of "
          f"{forensics.ADAPTER_MAX_PROMPT_TOKENS} limit; "
          f"{truncation['truncated_items']} item(s) truncated")
    for audit in prompts:
        print(f"  {audit['id']}: ok={audit['ok']} example={audit['example_letter']} "
              f"reference={audit['reference']} "
              f"tokens={audit['tokenizer']['prompt_tokens']}")
    for arm, entry in arms.items():
        print(f"  arm {arm}: patch {entry['collapse']['patch']['by_letter']} "
              f"diag {entry['collapse']['diagnostic']['by_letter']} "
              f"-> {entry['diagnosis']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
