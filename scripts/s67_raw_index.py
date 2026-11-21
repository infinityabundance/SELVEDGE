#!/usr/bin/env python3
"""S67: gzip raw evaluator dumps and index both digests.

The raw lm-evaluation-harness dumps are hundreds of megabytes and are ignored
by git; the tracked form is the lossless ``.json.gz`` beside it, with an
``INDEX.json`` that pins the byte size and SHA-256 of the raw and compressed
files. This mirrors the S65 raw-dump convention.

Usage (CPU):
    SELVEDGE_MEM=2g scripts/court python -- \
      python3 scripts/s67_raw_index.py results/stage2/s67-final
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args(argv)

    index = {}
    for root in args.roots:
        raw_dirs = [os.path.join(root, "raw")]
        if os.path.isdir(root):
            raw_dirs.extend(sorted(
                os.path.join(root, name) for name in os.listdir(root)
                if name.endswith("-raw") and os.path.isdir(os.path.join(root, name))))
        for raw_dir in raw_dirs:
            if not os.path.isdir(raw_dir):
                continue
            for name in sorted(os.listdir(raw_dir)):
                if not (name.endswith(".json") or name.endswith(".jsonl")):
                    continue
                path = os.path.join(raw_dir, name)
                gz_path = path + ".gz"
                if not os.path.exists(gz_path):
                    with open(path, "rb") as source, \
                            gzip.open(gz_path, "wb", compresslevel=9) as target:
                        for block in iter(lambda: source.read(1 << 22), b""):
                            target.write(block)
                for candidate in (path, gz_path):
                    index[os.path.relpath(candidate, ".")] = {
                        "bytes": os.path.getsize(candidate),
                        "sha256": sha256_file(candidate),
                    }
        # The per-task receipts beside raw/ are tracked too.
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name.endswith(".json") and os.path.isfile(path):
                index[os.path.relpath(path, ".")] = {
                    "bytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                }
    out = os.path.join(args.roots[0], "INDEX.json")
    with open(out + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(out + ".tmp", out)
    print(f"s67-raw-index: {len(index)} entries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
