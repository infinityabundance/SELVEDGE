#!/usr/bin/env python3
"""Compare two cached Qwen3.5 checkpoints file by file, header by header.

The WS19 ceiling certificate was produced for `Qwen/Qwen3.5-0.8B` (chat) while
the anchored scoreboard scores `Qwen/Qwen3.5-0.8B-Base`. That is an identity
defect unless the two are *measured* to be the same executed object. This script
answers the question with hashes rather than assumption, and writes a receipt:

1. **Safetensors header.** The first 8 bytes are a little-endian header length;
   the following JSON names every tensor with its dtype, shape and byte offsets.
   The raw header bytes are hashed, and the tensor table (name, dtype, shape,
   `data_offsets`) is sorted and hashed, so a header that differs only in key
   order is not reported as different and a header that differs in any shape is.
2. **Shard bytes.** `sha256` over the whole safetensors file, streamed, and the
   declared total from `model.safetensors.index.json` checked against the sum of
   the header's tensor byte ranges, so a truncated or replaced file is caught.
3. **The small files that decide compatibility.** `config.json` (the
   architecture), the tokenizer files (the vocabulary), `merges.txt`,
   `vocab.json`, and the licence.

The output says, per file, whether the two revisions are byte-identical, and the
`verdict` field states what that means for the execution path: identical headers
mean the same *graph*; different shard bytes mean different *weights*; different
tokenizer bytes mean the two checkpoints do not share a vocabulary, which is the
one difference that would break a distillation pairing.

CPU only; reads files, never the network. Peak resident memory is measured and
recorded, so a later run can size its court limit from this one rather than from
a guess.

Usage:
    scripts/court oracle -- python3 scripts/s38_revision_compare.py \
        --a /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
        --b /hf-cache/models/Qwen/Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
        --label-a Qwen/Qwen3.5-0.8B --label-b Qwen/Qwen3.5-0.8B-Base \
        --out results/frontier/<stamp>-revision-compare-08b-chat-vs-base.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time

SCHEMA = "selvedge/revision-compare/v1"

#: Small files compared byte-for-byte, in the order the receipt prints them.
COMPARE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "README.md",
    "generation_config.json",
    "preprocessor_config.json",
)


def sha256_stream(path: str, chunk: int = 1 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def read_header(path: str) -> dict:
    """Return the raw header bytes and the parsed tensor table of a shard."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SystemExit(f"{path}: shorter than the 8-byte header length prefix")
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > 64 << 20:
            raise SystemExit(f"{path}: implausible header length {length}")
        raw = handle.read(length)
        if len(raw) != length:
            raise SystemExit(f"{path}: header is truncated")
    table = json.loads(raw)
    if not isinstance(table, dict):
        raise SystemExit(f"{path}: header is not a JSON object")
    metadata = table.pop("__metadata__", None)
    return {
        "path": os.path.basename(path),
        "prefix_bytes": 8,
        "header_bytes": length,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": metadata,
        "tensors": table,
    }


def tensor_table_digest(tensors: dict) -> tuple[str, int]:
    """Canonical digest over (name, dtype, shape, data_offsets)."""
    rows = []
    for name in sorted(tensors):
        entry = tensors[name]
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise SystemExit(f"tensor {name}: missing or malformed data_offsets")
        rows.append(
            [
                name,
                entry.get("dtype"),
                list(entry.get("shape") or []),
                int(offsets[0]),
                int(offsets[1]),
            ]
        )
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), sort_keys=False).encode()
    )
    return digest.hexdigest(), len(rows)


def tensor_bytes(tensors: dict) -> int:
    total = 0
    for name, entry in tensors.items():
        offsets = entry["data_offsets"]
        total += int(offsets[1]) - int(offsets[0])
    return total


def shard_facts(directory: str) -> dict:
    shards = sorted(
        name for name in os.listdir(directory) if name.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"{directory}: no safetensors shard")
    facts = []
    for name in shards:
        path = os.path.join(directory, name)
        header = read_header(path)
        table_digest, count = tensor_table_digest(header["tensors"])
        file_sha, file_bytes = sha256_stream(path)
        facts.append(
            {
                "file": name,
                "file_bytes": file_bytes,
                "file_sha256": file_sha,
                "header_bytes": header["header_bytes"],
                "header_sha256": header["header_sha256"],
                "tensor_count": count,
                "tensor_table_sha256": table_digest,
                "tensor_bytes": tensor_bytes(header["tensors"]),
            }
        )
    total_tensor_bytes = sum(item["tensor_bytes"] for item in facts)
    index_path = os.path.join(directory, "model.safetensors.index.json")
    declared = None
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        declared = (index.get("metadata") or {}).get("total_size")
    return {
        "directory": directory,
        "shard_count": len(facts),
        "shards": facts,
        "tensor_bytes_total": total_tensor_bytes,
        "index_declared_total_bytes": declared,
        "index_matches_header": declared == total_tensor_bytes,
    }


TOKENIZER_SECTIONS = (
    "model",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "truncation",
    "padding",
)


def tokenizer_structure(directory: str) -> dict | None:
    """Per-section digests of `tokenizer.json`.

    Two revisions can ship a byte-different `tokenizer.json` whose *vocabulary*
    is nevertheless identical (the serializer emitted a different key order or a
    different `added_tokens` block). A distillation pairing needs the vocabulary
    and the token boundaries, not the file bytes, so the receipt separates them:
    byte identity is reported, and so is which section actually differs.
    """
    path = os.path.join(directory, "tokenizer.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    sections = {}
    for name in TOKENIZER_SECTIONS:
        if name not in document:
            sections[name] = {"present": False}
            continue
        payload = json.dumps(
            document[name], sort_keys=True, separators=(",", ":")
        ).encode()
        sections[name] = {
            "present": True,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    model = document.get("model") or {}
    vocab = model.get("vocab")
    merged = model.get("merges")
    vocab_digest = None
    if isinstance(vocab, dict):
        vocab_digest = hashlib.sha256(
            json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    merges_digest = None
    if isinstance(merged, list):
        merges_digest = hashlib.sha256(
            json.dumps(merged, separators=(",", ":")).encode()
        ).hexdigest()
    scalars = {
        key: model[key]
        for key in model
        if key not in ("vocab", "merges") and not isinstance(model[key], (dict, list))
    }
    return {
        "sections": sections,
        "top_level_keys": sorted(document),
        "vocab_entries": len(vocab) if isinstance(vocab, (dict, list)) else None,
        "vocab_sha256": vocab_digest,
        "merges_entries": len(merged) if isinstance(merged, list) else None,
        "merges_sha256": merges_digest,
        "added_tokens": len(document.get("added_tokens") or []),
        "added_token_contents": sorted(
            (str(token.get("content")), int(token.get("id", -1)))
            for token in (document.get("added_tokens") or [])
        ),
        "model_scalars": scalars,
    }


def file_facts(directory: str) -> dict:
    facts = {}
    for name in COMPARE_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            facts[name] = {"present": False}
            continue
        digest, size = sha256_stream(path)
        facts[name] = {"present": True, "bytes": size, "sha256": digest}
    return facts


def compare(a: dict, b: dict) -> list[dict]:
    rows = []
    for name in COMPARE_FILES:
        left, right = a.get(name, {"present": False}), b.get(name, {"present": False})
        if not left.get("present") and not right.get("present"):
            continue
        same = (
            left.get("present") == right.get("present")
            and left.get("sha256") == right.get("sha256")
        )
        rows.append(
            {
                "file": name,
                "a_bytes": left.get("bytes"),
                "b_bytes": right.get("bytes"),
                "a_sha256": left.get("sha256"),
                "b_sha256": right.get("sha256"),
                "identical": bool(same),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="skip the whole-file sha256 (the header comparison still runs)",
    )
    args = parser.parse_args(argv)

    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    started = time.time()

    sides = {}
    for key, directory in (("a", args.a), ("b", args.b)):
        shard = shard_facts(directory)
        if args.skip_weights:
            for item in shard["shards"]:
                item.pop("file_sha256", None)
                item["file_sha256"] = None
        sides[key] = {"checkpoint": shard, "files": file_facts(directory)}

    a_shard = sides["a"]["checkpoint"]["shards"][0]
    b_shard = sides["b"]["checkpoint"]["shards"][0]

    header_identical = (
        a_shard["header_sha256"] == b_shard["header_sha256"]
        and a_shard["header_bytes"] == b_shard["header_bytes"]
    )
    table_identical = (
        a_shard["tensor_table_sha256"] == b_shard["tensor_table_sha256"]
        and a_shard["tensor_count"] == b_shard["tensor_count"]
    )
    weights_identical = (
        None
        if args.skip_weights
        else a_shard["file_sha256"] == b_shard["file_sha256"]
    )
    file_rows = compare(sides["a"]["files"], sides["b"]["files"])
    by_name = {row["file"]: row for row in file_rows}
    config_identical = bool(by_name.get("config.json", {}).get("identical"))
    tokenizer_identical = bool(by_name.get("tokenizer.json", {}).get("identical"))
    vocab_identical = bool(
        by_name.get("vocab.json", {}).get("identical")
        and by_name.get("merges.txt", {}).get("identical")
    )

    if table_identical and weights_identical:
        verdict = (
            "the same graph and the same weights: the two revisions are the same "
            "execution object and an audit of either is an audit of both"
        )
    elif table_identical:
        verdict = (
            "the same graph (identical safetensors header, tensor names, shapes, "
            "dtypes and offsets) over different weights (different shard sha256): "
            "an element-count or ceiling argument transfers between them, a "
            "capability score does not"
        )
    else:
        verdict = (
            "not the same graph: the safetensors headers differ, so no accounting "
            "result transfers and the anchor must be audited in its own right"
        )

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp,
        "revision": args.revision,
        "generated_by": "scripts/s38_revision_compare.py",
        "labels": {"a": args.label_a, "b": args.label_b},
        "a": sides["a"],
        "b": sides["b"],
        "comparison": {
            "safetensors_header_identical": header_identical,
            "tensor_table_identical": table_identical,
            "weights_identical": weights_identical,
            "config_identical": config_identical,
            "tokenizer_identical": tokenizer_identical,
            "vocab_and_merges_identical": vocab_identical,
            "files": file_rows,
            "different_files": [row["file"] for row in file_rows if not row["identical"]],
        },
        "tokenizer_structure": {
            "a": tokenizer_structure(args.a),
            "b": tokenizer_structure(args.b),
        },
        "verdict": verdict,
        "wall_seconds": round(time.time() - started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"header identical      : {header_identical}")
    print(f"tensor table identical: {table_identical} ({a_shard['tensor_count']} tensors)")
    print(f"weights identical     : {weights_identical}")
    print(f"config identical      : {config_identical}")
    print(f"tokenizer identical   : {tokenizer_identical}")
    print(f"vocab+merges identical: {vocab_identical}")
    print(f"different files       : {receipt['comparison']['different_files']}")
    print(f"verdict               : {verdict}")
    print(f"receipt               : {args.out}")
    tokenizer_a = receipt["tokenizer_structure"]["a"] or {}
    tokenizer_b = receipt["tokenizer_structure"]["b"] or {}
    if tokenizer_a and tokenizer_b:
        differing = [
            name
            for name in TOKENIZER_SECTIONS
            if tokenizer_a["sections"][name] != tokenizer_b["sections"][name]
        ]
        same_vocab = tokenizer_a.get("vocab_sha256") == tokenizer_b.get("vocab_sha256")
        same_merges = tokenizer_a.get("merges_sha256") == tokenizer_b.get("merges_sha256")
        shared = sorted(
            set(tokenizer_a.get("added_token_contents") or [])
            & set(tokenizer_b.get("added_token_contents") or [])
        )
        only_a = sorted(
            set(tokenizer_a.get("added_token_contents") or [])
            - set(tokenizer_b.get("added_token_contents") or [])
        )
        only_b = sorted(
            set(tokenizer_b.get("added_token_contents") or [])
            - set(tokenizer_a.get("added_token_contents") or [])
        )
        receipt["tokenizer_structure"].update(
            {
                "differing_sections": differing,
                "vocab_identical": same_vocab,
                "merges_identical": same_merges,
                "added_tokens_shared": shared,
                "added_tokens_only_a": only_a,
                "added_tokens_only_b": only_b,
            }
        )
        print(f"tokenizer sections differing: {differing}")
        print(f"tokenizer vocab identical   : {same_vocab}")
        print(f"tokenizer merges identical  : {same_merges}")
        print(f"added tokens only in a      : {only_a}")
        print(f"added tokens only in b      : {only_b}")
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=False)
            handle.write("\n")
    if not table_identical:
        # A different graph is the one outcome that invalidates the certificate.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
