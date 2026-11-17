#!/usr/bin/env python3
"""S45: pack accepted RFT/on-policy solutions into a training stream.

The existing stage-2 trainer consumes a pinned `uint32` token stream and its
manifest. This builder produces one from one or more `accepted.jsonl` files:

* documents are ordered by `(input index, id, sample_index)`, so the stream is a
  deterministic function of the accepted files;
* each document is `prompt + "\\n" + solution`, tokenised with the anchor's own
  Qwen3.5 tokenizer, followed by EOS; no special tokens are added;
* the accepted corpus is repeated in whole cycles until the requested token
  count is filled, and the number of passes is recorded (`epochs`). The stream
  is truncated at exactly the requested token count;
* every input's `receipt.json` is re-read and its `accepted_sha256` is compared
  with the file's SHA-256; a mismatch fails closed. The sampling receipt's
  contamination report is carried into this manifest;
* the manifest carries `graft_tokens: 0`, so the whole stream is training data
  for `oracle/frontier/train.py`.

Usage
-----
    python3 scripts/s45_rft_stream.py --input results/stage2/rft/gsm8k \
        --input results/stage2/rft/math --tokens 20000000 --label rft-both
    python3 scripts/s45_rft_stream.py --input results/stage2/s45-rft-fixture \
        --tokens 50000 --label fixture --out-dir /tmp/s45 --receipt-dir /tmp/s45
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("HF_HOME", "/hf-cache")
MODEL_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                         "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str) -> str:
    import hashlib

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


def load_input(input_dir: str) -> tuple[list[dict], dict]:
    accepted_path = os.path.join(REPO, input_dir, "accepted.jsonl")
    receipt_path = os.path.join(REPO, input_dir, "receipt.json")
    if not os.path.exists(accepted_path):
        raise SystemExit(f"missing {accepted_path}")
    accepted_sha = sha256_file(accepted_path)
    receipt = None
    if os.path.exists(receipt_path):
        with open(receipt_path, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
        declared = receipt.get("accepted_sha256")
        if declared != accepted_sha:
            raise SystemExit(
                f"{accepted_path}: sha256 {accepted_sha} != receipt {declared}")
    rows = []
    with open(accepted_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: (row["id"], int(row.get("sample_index", 0))))
    return rows, {
        "accepted_path": accepted_path,
        "accepted_sha256": accepted_sha,
        "receipt_path": receipt_path if receipt is not None else None,
        "receipt": receipt,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="directory containing accepted.jsonl (+ receipt.json)")
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--label", default="rft")
    parser.add_argument("--out-dir", default="results/artifacts/s45-rft")
    parser.add_argument("--receipt-dir", default="results/stage2")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                              trust_remote_code=False)
    eos = int(tokenizer.eos_token_id)
    vocab = int(getattr(tokenizer, "vocab_size", 0)) or 248_320
    tokenizer_sha = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()

    inputs = []
    documents = []
    for input_index, input_dir in enumerate(args.input):
        rows, receipt = load_input(input_dir)
        inputs.append({"index": input_index, "dir": input_dir, **receipt})
        for row in rows:
            text = row["prompt"] + "\n" + row["solution"]
            documents.append((input_index, row["id"], int(row.get("sample_index", 0)),
                              row.get("dataset"), row["prompt"], row["solution"],
                              text))
    if not documents:
        raise SystemExit("no accepted documents")

    cycle_arrays = []
    example_rows = []
    corpus_tokens = 0
    prompt_tokens_total = 0
    solution_tokens_total = 0
    for (input_index, doc_id, sample_index, dataset, prompt, solution,
         text) in documents:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        array = np.asarray(ids + [eos], dtype=np.uint32)
        if int(array.max()) >= max(vocab, 248_320):
            raise SystemExit("token id outside the anchor vocabulary")
        cycle_arrays.append(array)
        corpus_tokens += int(array.size)
        # The SFT path supervises the solution and the EOS only. Prompt and
        # solution are tokenised separately (the separator is part of the
        # solution) so `train.py --sft-examples` can set the prompt labels to
        # -100 without guessing a boundary.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        solution_ids = tokenizer("\n" + solution, add_special_tokens=False)["input_ids"]
        if not solution_ids:
            raise SystemExit(f"{doc_id}: solution tokenises to zero tokens")
        if max(prompt_ids + solution_ids) >= max(vocab, 248_320):
            raise SystemExit("token id outside the anchor vocabulary")
        prompt_tokens_total += len(prompt_ids)
        solution_tokens_total += len(solution_ids)
        example_rows.append({
            "id": doc_id,
            "dataset": dataset,
            "source_index": input_index,
            "sample_index": sample_index,
            "prompt_ids": prompt_ids,
            "solution_ids": solution_ids,
        })
    if corpus_tokens == 0:
        raise SystemExit("accepted corpus tokenises to zero tokens")
    if corpus_tokens > args.tokens:
        # One pass is already larger than the budget: truncate the first cycle.
        stream = np.concatenate(cycle_arrays)[:args.tokens]
        epochs = 1
        passes = args.tokens / corpus_tokens
    else:
        repeats = -(-args.tokens // corpus_tokens)
        stream = np.concatenate(cycle_arrays * repeats)[:args.tokens]
        epochs = repeats
        passes = repeats
    if int(stream.size) != args.tokens:
        raise SystemExit(f"built {stream.size} tokens, wanted {args.tokens}")

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_stamp = stamp()
    stream_path = os.path.join(out_dir, f"{args.label}-stream-{out_stamp}.u32")
    stream.tofile(stream_path)
    examples_path = os.path.join(out_dir, f"{args.label}-examples-{out_stamp}.jsonl")
    with open(examples_path, "w", encoding="utf-8") as handle:
        for entry in example_rows:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    manifest = {
        "schema": "selvedge/stage2-stream/v1",
        "kind": "s45-rft-build",
        "created_utc": utcnow(),
        "stamp": out_stamp,
        "revision": revision(),
        "label": args.label,
        "tokens": int(stream.size),
        "bytes": int(stream.size * 4),
        "dtype": "uint32-le",
        "sha256": sha256_file(stream_path),
        "stream_path": os.path.relpath(stream_path, REPO),
        "graft_tokens": 0,
        "train_tokens": int(stream.size),
        "split": "all tokens are training data for the RFT/KD arm",
        "tokenizer": {
            "path": MODEL_DIR,
            "tokenizer_sha256": tokenizer_sha,
            "note": "the anchor's own Qwen3.5 tokenizer",
        },
        "sources": {
            "inputs": [
                {"index": entry["index"], "dir": entry["dir"],
                 "accepted_sha256": entry["accepted_sha256"],
                 "receipt_path": entry["receipt_path"]}
                for entry in inputs
            ],
            "documents": len(cycle_arrays),
            "corpus_tokens_per_pass": corpus_tokens,
            "epochs": epochs,
            "passes": passes,
            "document_order": "(input index, id, sample_index), ascending",
            "separator": "eos_token",
            "eos_token_id": eos,
        },
        "examples": {
            "path": os.path.relpath(examples_path, REPO),
            "sha256": sha256_file(examples_path),
            "rows": len(example_rows),
            "prompt_tokens": prompt_tokens_total,
            "solution_tokens": solution_tokens_total,
            "note": ("prompt_ids and solution_ids are separate so "
                     "train.py --sft-examples can mask the prompt with -100; "
                     "the stream above remains an untyped legacy input"),
        },
        "contamination": [
            {"input": entry["dir"],
             "sampling_receipt": (entry["receipt"] or {}).get("contamination")}
            for entry in inputs
        ],
        "command": sys.argv,
    }
    manifest_path = os.path.join(REPO, args.receipt_dir,
                                 f"s45-rft-stream-{out_stamp}.json")
    write_json(manifest_path, manifest)
    print(f"stream: {stream.size} tokens sha256={manifest['sha256']} "
          f"corpus={corpus_tokens} epochs={epochs} -> {stream_path}")
    print(f"examples: {len(example_rows)} rows "
          f"prompt={prompt_tokens_total} solution={solution_tokens_total} -> "
          f"{examples_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
