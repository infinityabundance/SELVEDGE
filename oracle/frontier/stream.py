#!/usr/bin/env python3
"""Stage-2 pinned token stream and its leakage guard (WS19 P19.2 / S37).

The stream is built from the already-fetched, permissively licensed T1 corpus
cache (`results/artifacts/train-corpus`), re-tokenised with the **anchor's own**
Qwen3.5 tokenizer (the T1 stream was tokenised with Qwen3, whose vocabulary is
different, so it cannot be reused byte-for-byte).

Discipline:

* documents are ordered by the SHA-256 of their whitespace-normalised text, so
  the stream is a deterministic function of the cached bytes;
* every document is checked against the cached evaluation sets with the same
  5-word-shingle Jaccard rule as `oracle/train/corpus.py`: >= 0.2 is discarded,
  >= 0.5 fails the build closed;
* the output is little-endian uint32 plus a manifest carrying the token count,
  the stream SHA-256, the tokenizer hash, the source revisions and the
  contamination statistics. No corpus text is committed.

The smoke uses the first `--graft-tokens` tokens for teacher grafting and the
remaining `--train-tokens` for training; the split is recorded, not implicit.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

CACHE = os.environ.get("HF_HOME", "/hf-cache")
MODEL_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                         "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
CORPUS_ROOT = "results/artifacts/train-corpus"
DATASET_ROOT = os.path.join(CACHE, "datasets")
DOMAIN = "SELVEDGE stage-2 stream"
SHINGLE_WIDTH = 5
DISCARD = 0.2
FAIL_CLOSED = 0.5


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"\S+", normalize(text).lower())


def shingle_hashes(tokens: list[str], width: int = SHINGLE_WIDTH) -> set[int]:
    """Stable 64-bit hashes of the word shingles (process-independent)."""
    if len(tokens) < width:
        return set()
    out = set()
    for index in range(len(tokens) - width + 1):
        piece = " ".join(tokens[index:index + width]).encode("utf-8")
        out.add(int.from_bytes(hashlib.blake2b(piece, digest_size=8).digest(), "little"))
    return out


# ---------------------------------------------------------------------------
# evaluation sets and the leakage index


def _row_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_row_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_row_text(item) for item in value.values())
    return ""


def evaluation_items(dataset_root: str = DATASET_ROOT) -> list[tuple[str, str]]:
    """Every cached evaluation row that is part of a test split."""
    import pyarrow.ipc as ipc

    items: list[tuple[str, str]] = []
    pattern = os.path.join(dataset_root, "**", "*.arrow")
    for path in sorted(glob.glob(pattern, recursive=True)):
        base = os.path.basename(path)
        if base.startswith("cache-") or "test" not in base:
            continue
        try:
            table = ipc.open_stream(path).read_all()
        except Exception:
            continue
        name = path.replace(dataset_root + os.sep, "")
        for row in table.to_pylist():
            text = _row_text(row)
            if len(text) >= 20:
                items.append((name, text))
    return items


class LeakageIndex:
    """Inverted 5-word-shingle index over the evaluation items."""

    def __init__(self, items: list[tuple[str, str]]):
        self.items = items
        self.bags: list[set[int]] = []
        self.inverted: dict[int, list[int]] = {}
        for index, (_name, text) in enumerate(items):
            bag = shingle_hashes(words(text))
            self.bags.append(bag)
            for shingle in bag:
                self.inverted.setdefault(shingle, []).append(index)

    def worst(self, text: str) -> tuple[float, str | None]:
        bag = shingle_hashes(words(text))
        if not bag:
            return 0.0, None
        shared: dict[int, int] = {}
        for shingle in bag:
            for index in self.inverted.get(shingle, ()):
                shared[index] = shared.get(index, 0) + 1
        worst = 0.0
        worst_name = None
        for index, count in shared.items():
            other = self.bags[index]
            union = len(bag) + len(other) - count
            score = count / union if union else 0.0
            if score > worst:
                worst, worst_name = score, self.items[index][0]
        return worst, worst_name

    def report(self) -> dict:
        return {"items": len(self.items),
                "distinct_shingles": len(self.inverted),
                "width_words": SHINGLE_WIDTH}


# ---------------------------------------------------------------------------
# corpus documents


def _fineweb(root: str):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(os.path.join(root, "fineweb-edu-sample-10bt",
                                              "**", "*.parquet"), recursive=True)):
        table = pq.read_table(path, columns=["text"])
        for value in table["text"].to_pylist():
            yield os.path.relpath(path, root), value


def _hendrycks(root: str):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(os.path.join(root, "hendrycks-math-train",
                                              "**", "*.parquet"), recursive=True)):
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield os.path.relpath(path, root), (f"Problem: {row['problem']}\n"
                                                f"Solution: {row['solution']}")


def candidate_documents(root: str, limit: int = 6000):
    """Deterministic candidates: fineweb first, then MIT-licensed math."""
    seen: set[str] = set()
    produced = 0
    for source in (_fineweb, _hendrycks):
        for path, text in source(root):
            if not text:
                continue
            normalized = normalize(text)
            if len(normalized) < 200:
                continue
            digest = sha256_text(normalized)
            if digest in seen:
                continue
            seen.add(digest)
            yield digest, path, text
            produced += 1
            if produced >= limit:
                return


def build_stream(tokens_wanted: int, graft_tokens: int, out_dir: str,
                 receipt_dir: str, corpus_root: str = CORPUS_ROOT,
                 dataset_root: str = DATASET_ROOT, vocabulary_check: bool = True,
                 candidates_limit: int = 6000):
    from transformers import AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                        trust_remote_code=False)
    tokenizer_sha = sha256_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    items = evaluation_items(dataset_root)
    index = LeakageIndex(items)
    print(f"stream: eval items={len(items)} shingles={len(index.inverted)}", flush=True)

    # 2000 tokenizer-token carry margin: subword counts are not known up front.
    # The candidate cap is a length parameter, not a change of order or guard:
    # the 6000-document default yields ~6M tokens, so a 20M-token stream needs a
    # larger cap. It is recorded in the manifest.
    candidates = list(candidate_documents(corpus_root, limit=candidates_limit))
    candidates.sort(key=lambda item: item[0])
    print(f"stream: {len(candidates)} candidate documents", flush=True)

    eos = tok.eos_token_id
    pieces: list[np.ndarray] = []
    total = 0
    accepted = 0
    discarded = 0
    hard: list[str] = []
    worst_seen = 0.0
    worst_source = None
    per_source: dict[str, int] = {}
    for digest, path, text in candidates:
        score, worst_name = index.worst(text)
        if score > worst_seen:
            worst_seen, worst_source = score, worst_name
        if score >= DISCARD:
            discarded += 1
            if score >= FAIL_CLOSED:
                hard.append(f"{digest[:16]}:{score:.3f}:{worst_name}")
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        array = np.asarray(ids + [eos], dtype=np.uint32)
        pieces.append(array)
        total += int(array.size)
        accepted += 1
        per_source[path] = per_source.get(path, 0) + int(array.size)
        if total >= tokens_wanted + 2000:
            break
    if hard:
        raise SystemExit(f"stream: contamination >= {FAIL_CLOSED}: {hard[:5]}")

    stream = np.concatenate(pieces)
    if total < tokens_wanted:
        raise SystemExit(f"stream: only {total} tokens available, wanted {tokens_wanted}")
    stream = stream[:tokens_wanted]
    if vocabulary_check:
        limit = int(getattr(tok, "vocab_size", 0)) or 248_320
        if int(stream.max()) >= max(limit, 248_320):
            raise SystemExit("stream: token id outside the anchor vocabulary")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(receipt_dir, exist_ok=True)
    start_stamp = stamp()
    stream_path = os.path.join(out_dir, f"stage2-stream-{start_stamp}.u32")
    stream.tofile(stream_path)
    manifest = {
        "schema": "selvedge/stage2-stream/v1",
        "kind": "build",
        "created_utc": utcnow(),
        "stamp": start_stamp,
        "revision": _revision(),
        "tokens": int(stream.size),
        "bytes": int(stream.size * 4),
        "dtype": "uint32-le",
        "sha256": sha256_file(stream_path),
        "stream_path": stream_path,
        "graft_tokens": int(graft_tokens),
        "train_tokens": int(stream.size - graft_tokens),
        "split": "first graft_tokens for teacher grafting, remainder for training",
        "tokenizer": {
            "path": MODEL_DIR,
            "tokenizer_sha256": tokenizer_sha,
            "note": "the anchor's own Qwen3.5 tokenizer (same file in 0.8B-Base "
                    "and 2B-Base; hashes verified equal)",
        },
        "sources": {
            "root": corpus_root,
            "documents_accepted": accepted,
            "documents_discarded_contamination": discarded,
            "candidate_limit": int(candidates_limit),
            "document_order": "sha256 of normalised text, ascending",
            "separator": "eos_token",
            "per_path_tokens": per_source,
        },
        "contamination": {
            "evaluation_items": len(items),
            "eval_split_filter": "cached arrow files whose basename contains 'test'",
            "shingle_width_words": SHINGLE_WIDTH,
            "threshold_discard": DISCARD,
            "threshold_fail_closed": FAIL_CLOSED,
            "max_shingle_jaccard_seen": worst_seen,
            "max_shingle_jaccard_eval_set": worst_source,
            "index": index.report(),
        },
        "wall_seconds": round(time.time() - started, 3),
        "container_memory": _cgroup_memory(),
    }
    manifest_path = os.path.join(receipt_dir, f"stage2-stream-{start_stamp}.json")
    _write_json(manifest_path, manifest)
    print(f"stream: {stream.size} tokens sha256={manifest['sha256']} "
          f"-> {stream_path} ({manifest['wall_seconds']} s)", flush=True)
    print(f"stream: receipt {manifest_path}", flush=True)
    return stream_path, manifest_path


def _revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(HERE)), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _cgroup_memory() -> dict:
    """cgroup v2 peak/limit inside the court, so the builder's limit is measured."""
    report = {}
    for name in ("memory.peak", "memory.max"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    return report


def load_stream(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--tokens", type=int, default=2_250_000)
    build.add_argument("--graft-tokens", type=int, default=250_000)
    build.add_argument("--candidates", type=int, default=6000)
    build.add_argument("--out-dir", default="results/artifacts/stage2")
    build.add_argument("--receipt-dir", default="results/stage2")
    build.add_argument("--corpus-root", default=CORPUS_ROOT)
    build.add_argument("--dataset-root", default=DATASET_ROOT)
    check = sub.add_parser("verify")
    check.add_argument("manifest")
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.graft_tokens >= args.tokens:
            raise SystemExit("--graft-tokens must be smaller than --tokens")
        build_stream(args.tokens, args.graft_tokens, args.out_dir,
                     args.receipt_dir, args.corpus_root, args.dataset_root,
                     candidates_limit=args.candidates)
        return 0
    manifest = json.load(open(args.manifest))
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit(f"stream: sha256 mismatch {got} != {manifest['sha256']}")
    print(f"stream: verified {manifest['tokens']} tokens sha256={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
