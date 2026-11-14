#!/usr/bin/env python3
"""Record the tokenizer identity used by the teacher oracle.

The tokenizer is part of the experimental identity: two runs that tokenise
differently are not comparable. This probe loads the cached tokenizer offline,
records its class, size, special-token ids, and the SHA-256 of every tokenizer
file, and never touches the network.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import tokenizers
from transformers import AutoTokenizer


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--revision-git", required=True)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=False, local_files_only=True
    )
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = os.path.join(args.model_dir, name)
        if os.path.exists(path):
            files[name] = {
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
    report = {
        "schema": "selvedge/tokenizer-identity/v1",
        "stamp": args.stamp,
        "revision": args.revision_git,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "model": {"id": args.model_id, "revision": args.revision},
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "is_fast": bool(tokenizer.is_fast),
            "length": len(tokenizer),
            "vocab_size": int(tokenizer.vocab_size),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "tokenizers_library_version": tokenizers.__version__,
            "files": files,
        },
        "notes": [
            "loaded with trust_remote_code=False and local_files_only=True",
            "the tokenizer length can be smaller than the model vocabulary; the difference is unused embedding rows, not extra tokens",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"tokenizer: {args.model_id} class={report['tokenizer']['class']} "
        f"len={report['tokenizer']['length']} vocab={report['tokenizer']['vocab_size']} "
        f"eos={report['tokenizer']['eos_token_id']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
