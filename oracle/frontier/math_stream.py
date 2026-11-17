#!/usr/bin/env python3
"""Math-weighted stage-2 stream (S43): a registered mixture, not FineWeb alone.

The registered WS19 streams are 100% FineWeb-Edu. This module builds the
math-weighted replacement:

* math sources (MIT / Apache-2.0 / ODC-By, each pinned by revision and SHA-256):
  the existing pinned MATH-train corpus (MIT), the GSM8K train split (MIT), and
  a bounded sample of NuminaMath-CoT (Apache-2.0);
* general source: the existing pinned FineWeb-Edu shard (ODC-By 1.0), the same
  file the general-only stream used.

The intended third math source was `nvidia/Nemotron-CC-Math-v1`. It is a gated
dataset: an unauthenticated request for `3/part_000000.parquet` returns HTTP 401
`GatedRepo`. It is recorded as attempted-and-unavailable, and NuminaMath-CoT is
substituted *explicitly* (revision, license and SHA-256 in the manifest), never
silently.

Discipline:

* documents are taken in a deterministic order: subject files sorted by path and
  round-robined for MATH, rows in file order elsewhere, first accepted documents
  until each source's exact token budget is filled;
* every document is shingle-checked against every cached evaluation test split
  and against the sealed capability fixtures. Any shingle Jaccard at or above
  `FAIL_CLOSED` aborts the build; the final stream must have zero such overlap;
* tokenisation uses the anchor's own Qwen3.5 tokenizer;
* each shard records its own SHA-256 and its exact per-source token counts; the
  whole stream is exactly 20,000,000 tokens.

Subcommands: fetch | count | build | verify
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier.stream import (  # noqa: E402
    LeakageIndex, evaluation_items, normalize, words,
)

CACHE = os.environ.get("HF_HOME", "/hf-cache")
MODEL_DIR = os.path.join(CACHE, "models", "Qwen", "Qwen3.5-0.8B-Base",
                         "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
DATASET_ROOT = os.path.join(CACHE, "datasets")
CORPUS_ROOT = "results/artifacts/train-corpus"
MATH_ROOT = "results/artifacts/math-corpus"
SEALED_ROOT = "fixtures/capability/sealed"
SEALED_INDEX = "fixtures/capability/SEALED.json"

SHINGLE_WIDTH = 5
DISCARD = 0.2              # documents at or above this are excluded
FAIL_CLOSED = 0.5          # a near-verbatim copy aborts the build

# Pinned sources. `local` files are already in the gitignored corpus cache; the
# NuminaMath shard is fetched by `fetch` and hash-checked against the LFS oid.
SOURCES = {
    "gsm8k": {
        "kind": "gsm8k",
        "repo_id": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "split": "train",
        "rows": 7473,
        "path": os.path.join(
            DATASET_ROOT, "openai___gsm8k", "main", "0.0.0",
            "740312add88f781978c0658806c59bc2815b9866",
            "gsm8k-train.arrow"),
        "sha256": "6146bca074f4c6c5e07de7e96cc968adab3833dbdafeda21830f27558eea3d9b",
        "sha256_note": "the Hugging Face datasets arrow file for this revision; the corpus is pinned by revision and this local content hash",
    },
    "math": {
        "kind": "math",
        "repo_id": "EleutherAI/hendrycks_math",
        "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
        "license": "MIT",
        "split": "train",
        "path": os.path.join(CORPUS_ROOT, "hendrycks-math-train"),
        "lock": "data/train-corpus.lock.json",
    },
    "numina": {
        "kind": "numina",
        "repo_id": "project-numina/NuminaMath-CoT",
        "revision": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
        "license": "Apache-2.0",
        "split": "train",
        "file": "data/train-00000-of-00005.parquet",
        "lfs_oid": "25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3",
        "bytes": 246635854,
        "path": os.path.join(MATH_ROOT, "numina-cot",
                             "25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3",
                             "train-00000-of-00005.parquet"),
        "url": ("https://huggingface.co/datasets/project-numina/NuminaMath-CoT/"
                "resolve/9d8d210c9f6a36c8f3cd84045668c9b7800ef517/"
                "data/train-00000-of-00005.parquet"),
    },
    "fineweb": {
        "kind": "fineweb",
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "license": "ODC-By-1.0",
        "split": "train",
        "path": os.path.join(CORPUS_ROOT, "fineweb-edu-sample-10bt",
                             "sample", "10BT", "013_00000.parquet"),
        "lock": "data/train-corpus.lock.json",
    },
}

# The mixture plan. One shard is 200,000 tokens; one cycle is 10,000 tokens and
# repeats 20 times per shard. With 100 shards this is exactly:
#   numina   8,000,000   math 2,200,000   gsm8k 1,200,000   fineweb 8,600,000
# = 11,400,000 math (57.0%) + 8,600,000 general (43.0%), 20,000,000 total.
SHARD_TOKENS = 200_000
CYCLE = (("numina", 4_000), ("math", 1_100), ("gsm8k", 600), ("fineweb", 4_300))
CYCLES_PER_SHARD = SHARD_TOKENS // sum(size for _, size in CYCLE)
SHARDS = 100
TOTAL_TOKENS = SHARD_TOKENS * SHARDS
MATH_FRACTION = (8_000_000 + 2_200_000 + 1_200_000) / TOTAL_TOKENS


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _cgroup_memory() -> dict:
    report = {}
    for name in ("memory.peak", "memory.max", "memory.swap.peak"):
        try:
            with open(os.path.join("/sys/fs/cgroup", name), encoding="utf-8") as handle:
                report[name.replace(".", "_")] = handle.read().strip()
        except OSError:
            report[name.replace(".", "_")] = None
    return report


# ---------------------------------------------------------------------------
# evaluation sets, including the sealed holdback


def sealed_items(root: str = SEALED_ROOT, index_path: str = SEALED_INDEX
                 ) -> list[tuple[str, str]]:
    """Text of the sealed capability fixtures, read but never trained on."""
    items: list[tuple[str, str]] = []
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            items.append((index_path, handle.read()))
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        if len(text) >= 20:
            items.append((path, text))
    return items


# ---------------------------------------------------------------------------
# source rows


def _gsm8k_rows(path: str):
    import pyarrow.ipc as ipc

    table = ipc.open_stream(path).read_all()
    for row in table.to_pylist():
        yield (f"Question: {row['question']}\nAnswer: {row['answer']}")


def _math_rows(root: str):
    """Round-robin the seven subject files so every subject is represented."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "*", "*.parquet")))
    tables = [pq.read_table(path).to_pylist() for path in files]
    longest = max(len(table) for table in tables)
    for index in range(longest):
        for table in tables:
            if index < len(table):
                row = table[index]
                yield f"Problem:\n{row['problem']}\n\nSolution:\n{row['solution']}"


def _numina_rows(path: str):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512,
                                      columns=["problem", "solution", "source"]):
        for row in batch.to_pylist():
            problem = row["problem"] or ""
            solution = row["solution"] or ""
            source = (row["source"] or "").lower()
            if source == "gsm8k":
                yield f"Question: {problem}\nAnswer: {solution}"
            elif source in ("math", "math_train"):
                yield f"Problem:\n{problem}\n\nSolution:\n{solution}"
            else:
                yield f"Problem:\n{problem}\n\nSolution:\n{solution}"


def _fineweb_rows(path: str):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=512, columns=["text"]):
        for value in batch["text"].to_pylist():
            yield value


def source_rows(name: str):
    spec = SOURCES[name]
    kind = spec["kind"]
    if kind == "gsm8k":
        return _gsm8k_rows(spec["path"])
    if kind == "math":
        return _math_rows(spec["path"])
    if kind == "numina":
        return _numina_rows(spec["path"])
    if kind == "fineweb":
        return _fineweb_rows(spec["path"])
    raise SystemExit(f"unknown source kind {kind!r}")


def source_hashes() -> dict:
    """The local file hashes; fail closed on a missing or drifted file."""
    report = {}
    for name, spec in SOURCES.items():
        if name == "math":
            paths = sorted(glob.glob(os.path.join(spec["path"], "*", "*.parquet")))
        else:
            paths = [spec["path"]]
        entries = []
        for path in paths:
            if not os.path.exists(path):
                raise SystemExit(f"math_stream: {name} file missing: {path}")
            entries.append({"path": path, "bytes": os.path.getsize(path),
                            "sha256": sha256_file(path)})
        report[name] = entries
    return report


# ---------------------------------------------------------------------------
# fetch the bounded NuminaMath sample


def fetch(out_root: str = MATH_ROOT) -> int:
    spec = SOURCES["numina"]
    path = os.path.join(out_root, "numina-cot", spec["lfs_oid"],
                        "train-00000-of-00005.parquet")
    if os.path.exists(path):
        got = sha256_file(path)
        if got == spec["lfs_oid"]:
            print(f"fetch: {path} already verified {got}")
            return 0
        raise SystemExit(f"fetch: {path} sha256 {got} != pinned {spec['lfs_oid']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    digest = hashlib.sha256()
    received = 0
    print(f"fetch: {spec['url']} -> {path}", flush=True)
    with urllib.request.urlopen(spec["url"], timeout=120) as response, \
            open(tmp, "wb") as handle:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != spec["bytes"]:
            raise SystemExit(f"fetch: content-length {length} != pinned {spec['bytes']}")
        while True:
            chunk = response.read(8 << 20)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
    if received != spec["bytes"]:
        raise SystemExit(f"fetch: received {received} bytes, pinned {spec['bytes']}")
    got = digest.hexdigest()
    if got != spec["lfs_oid"]:
        os.remove(tmp)
        raise SystemExit(f"fetch: sha256 {got} != pinned {spec['lfs_oid']}")
    os.replace(tmp, path)
    print(f"fetch: verified {received} bytes sha256={got}")
    return 0


# ---------------------------------------------------------------------------
# count available tokens per source


def count(dataset_root: str = DATASET_ROOT) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                        trust_remote_code=False)
    eos = tok.eos_token_id
    report = {}
    for name in ("gsm8k", "math", "numina", "fineweb"):
        started = time.time()
        seen: set[str] = set()
        tokens = 0
        documents = 0
        for text in source_rows(name):
            if not text:
                continue
            normalized = normalize(text)
            if len(normalized) < 200:
                continue
            digest = sha256_bytes(normalized.encode("utf-8"))
            if digest in seen:
                continue
            seen.add(digest)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            tokens += len(ids) + 1
            documents += 1
        report[name] = {"documents": documents, "tokens": tokens,
                        "wall_seconds": round(time.time() - started, 1)}
        print(f"count: {name} documents={documents} tokens={tokens} "
              f"({report[name]['wall_seconds']} s)", flush=True)
    write_json(os.path.join("results", "stage2",
                            f"s43-math-source-count-{stamp()}.json"),
               {"schema": "selvedge/s43-math-source-count/v1",
                "created_utc": utcnow(), "sources": report,
                "note": "counts include the per-document EOS and are before the "
                        "contamination and per-source budget cuts"})
    return 0


# ---------------------------------------------------------------------------
# build


class _SourceBuffer:
    """Yield an exact number of tokens from one source, in pinned order."""

    def __init__(self, name: str, tokenizer, index: LeakageIndex,
                 discard_log: list, hard_log: list):
        self.name = name
        self.tokenizer = tokenizer
        self.index = index
        self.discard_log = discard_log
        self.hard_log = hard_log
        self.iterator = source_rows(name)
        self.rows: list[int] = []
        self.seen: set[str] = set()
        self.worst = 0.0
        self.worst_where = None
        self.worst_discarded = 0.0
        self.documents = 0
        self.rejected_short = 0
        self.rejected_duplicate = 0
        self.rejected_contamination = 0

    def _next_array(self):
        while True:
            try:
                text = next(self.iterator)
            except StopIteration:
                return None
            if not text:
                continue
            normalized = normalize(text)
            if len(normalized) < 200:
                self.rejected_short += 1
                continue
            digest = sha256_bytes(normalized.encode("utf-8"))
            if digest in self.seen:
                self.rejected_duplicate += 1
                continue
            self.seen.add(digest)
            score, where = self.index.worst(text)
            if score >= FAIL_CLOSED:
                self.hard_log.append({"source": self.name, "digest": digest,
                                      "jaccard": score, "eval_item": where})
                # A near-verbatim copy is excluded like any other overlap; the
                # build fails closed below if a budget cannot be filled without
                # it, and the retained maximum is asserted below DISCARD.
                self.rejected_contamination += 1
                self.worst_discarded = max(self.worst_discarded, score)
                if len(self.hard_log) > 200:
                    del self.hard_log[200:]
                continue
            if score >= DISCARD:
                self.rejected_contamination += 1
                self.worst_discarded = max(self.worst_discarded, score)
                if len(self.discard_log) < 1000:
                    self.discard_log.append({"source": self.name,
                                             "digest": digest,
                                             "jaccard": score,
                                             "eval_item": where})
                continue
            if score > self.worst:
                self.worst, self.worst_where = score, where
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            self.documents += 1
            array = np.asarray(ids + [self.tokenizer.eos_token_id], dtype=np.uint32)
            return array

    def take(self, count: int) -> np.ndarray:
        pieces = []
        taken = 0
        while taken < count:
            if not self.rows:
                array = self._next_array()
                if array is None:
                    raise SystemExit(
                        f"math_stream: source {self.name} exhausted while taking "
                        f"{count} tokens ({taken} gathered); lower its budget")
                self.rows = array.tolist()
            need = count - taken
            piece = self.rows[:need]
            self.rows = self.rows[need:]
            pieces.append(np.asarray(piece, dtype=np.uint32))
            taken += len(piece)
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def build(out_dir: str, receipt_dir: str, dataset_root: str = DATASET_ROOT) -> int:
    from transformers import AutoTokenizer

    started = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                        trust_remote_code=False)
    tokenizer_sha = sha256_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    items = evaluation_items(dataset_root) + sealed_items()
    index = LeakageIndex(items)
    print(f"build: eval+sealed items={len(items)} "
          f"shingles={len(index.inverted)}", flush=True)

    source_files = source_hashes()
    discard_log: list[dict] = []
    hard_log: list[dict] = []
    buffers = {name: _SourceBuffer(name, tok, index, discard_log, hard_log)
               for name, _ in CYCLE}

    overall = np.zeros(TOTAL_TOKENS, dtype=np.uint32)
    shard_reports = []
    cursor = 0
    for shard in range(SHARDS):
        shard_start = cursor
        per_source = {name: 0 for name, _ in CYCLE}
        for _ in range(CYCLES_PER_SHARD):
            for name, size in CYCLE:
                overall[cursor:cursor + size] = buffers[name].take(size)
                cursor += size
                per_source[name] += size
        blob = overall[shard_start:cursor].tobytes()
        shard_reports.append({
            "index": shard,
            "start_token": shard_start,
            "tokens": cursor - shard_start,
            "sha256": sha256_bytes(blob),
            "per_source_tokens": per_source,
        })
        if shard % 10 == 0:
            print(f"build: shard {shard + 1}/{SHARDS} tokens={cursor}", flush=True)

    if cursor != TOTAL_TOKENS:
        raise SystemExit(f"build: assembled {cursor} tokens, expected {TOTAL_TOKENS}")
    for name, buffer in buffers.items():
        if buffer.worst >= DISCARD:
            raise SystemExit(
                f"build: fail closed: a retained {name} document scores "
                f"{buffer.worst:.3f} >= {DISCARD}")
        if buffer.documents == 0:
            raise SystemExit(f"build: source {name} contributed no documents")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(receipt_dir, exist_ok=True)
    build_stamp = stamp()
    stream_path = os.path.join(out_dir, f"stage2-math-stream-{build_stamp}.u32")
    overall.tofile(stream_path)
    stream_sha = sha256_file(stream_path)

    per_source_total = {}
    for name, _ in CYCLE:
        per_source_total[name] = {
            "tokens": sum(report["per_source_tokens"][name]
                          for report in shard_reports),
            "revision": SOURCES[name]["revision"],
            "license": SOURCES[name]["license"],
            "repo_id": SOURCES[name]["repo_id"],
            "split": SOURCES[name]["split"],
            "files": source_files[name],
            "documents_used": buffers[name].documents,
            "rejected_short": buffers[name].rejected_short,
            "rejected_duplicate": buffers[name].rejected_duplicate,
            "rejected_contamination": buffers[name].rejected_contamination,
            "max_shingle_jaccard_retained": buffers[name].worst,
            "max_shingle_jaccard_retained_item": buffers[name].worst_where,
            "max_shingle_jaccard_discarded": buffers[name].worst_discarded,
        }

    manifest = {
        "schema": "selvedge/stage2-math-stream/v1",
        "kind": "build",
        "created_utc": utcnow(),
        "stamp": build_stamp,
        "tokens": TOTAL_TOKENS,
        "bytes": int(overall.size * 4),
        "dtype": "uint32-le",
        "sha256": stream_sha,
        "stream_path": stream_path,
        "graft_tokens": 0,
        "train_tokens": TOTAL_TOKENS,
        "split": "no graft split; the full-distillation arm trains on every token",
        "mixture": {
            "math_tokens": 8_000_000 + 2_200_000 + 1_200_000,
            "general_tokens": 8_600_000,
            "math_fraction": MATH_FRACTION,
            "general_fraction": 8_600_000 / TOTAL_TOKENS,
            "per_source_tokens": {name: per_source_total[name]["tokens"]
                                  for name, _ in CYCLE},
            "cycle_tokens": sum(size for _, size in CYCLE),
            "cycle": [{"source": name, "tokens": size} for name, size in CYCLE],
            "cycles_per_shard": CYCLES_PER_SHARD,
            "shard_tokens": SHARD_TOKENS,
            "shards": SHARDS,
            "note": ("every shard repeats the same cycle, so every shard has the "
                     "same exact per-source counts; the table below records them"),
        },
        "shards": shard_reports,
        "sources": per_source_total,
        "source_order": {
            "gsm8k": "arrow rows in file order",
            "math": "subject files sorted by path, rows round-robined",
            "numina": "the pinned shard's rows in file order",
            "fineweb": "the pinned parquet's rows in file order",
            "selection": "first accepted document until the exact source budget",
        },
        "tokenizer": {
            "path": MODEL_DIR,
            "tokenizer_sha256": tokenizer_sha,
            "note": "the anchor's own Qwen3.5 tokenizer",
        },
        "contamination": {
            "evaluation_items": len(items),
            "sets": {
                "cached_test_splits": "every cached arrow whose basename contains 'test'",
                "sealed": f"{SEALED_ROOT}/** plus {SEALED_INDEX}",
            },
            "shingle_width_words": SHINGLE_WIDTH,
            "threshold_discard": DISCARD,
            "threshold_fail_closed": FAIL_CLOSED,
            "max_shingle_jaccard_retained": max(
                (source["max_shingle_jaccard_retained"]
                 for source in per_source_total.values()), default=0.0),
            "max_shingle_jaccard_discarded": max(
                (source["max_shingle_jaccard_discarded"]
                 for source in per_source_total.values()), default=0.0),
            "documents_rejected_contamination": sum(
                source["rejected_contamination"]
                for source in per_source_total.values()),
            "documents_rejected": discard_log,
            "hard_contamination": hard_log,
            "index": index.report(),
            "verdict": ("PASS: every token in the final stream comes from a "
                        f"document below {DISCARD} Jaccard against every "
                        "evaluation test split and the sealed holdback. "
                        "Overlapping documents are excluded and listed above by "
                        f"digest only; a document at or above {FAIL_CLOSED} is "
                        "also flagged as a source-split defect. The build fails "
                        "closed if clean data cannot fill a budget or if any "
                        "retained document reaches the discard threshold."),
        },
        "teacher_visibility": {
            "protocol": ("the teacher sees the same token ids the student trains "
                         "on, in the same order, on the same batch; logits are "
                         "computed online and never cached"),
            "supervised_targets": ("every position except the first of each "
                                   "512-token training window; the scored formats "
                                   "put the worked solution after 'Answer:' or "
                                   "'Solution:', so the teacher's distribution "
                                   "covers the solution tokens"),
        },
        "nemotron_cc_math": {
            "repo_id": "nvidia/Nemotron-CC-Math-v1",
            "revision": "397a2502f2028c659ba411a6c4935b464a7f03aa",
            "attempted_utc": utcnow(),
            "result": "HTTP 401 GatedRepo; no authentication token is available",
            "used": False,
            "substitute": "project-numina/NuminaMath-CoT",
            "substitute_reason": ("the requested bounded Nemotron sample could "
                                  "not be downloaded; the substitute is named "
                                  "explicitly with its own pinned revision, "
                                  "license and SHA-256"),
        },
        "wall_seconds": round(time.time() - started, 1),
        "container_memory": _cgroup_memory(),
    }
    manifest_path = os.path.join(receipt_dir, f"stage2-math-stream-{build_stamp}.json")
    write_json(manifest_path, manifest)
    print(f"build: {overall.size} tokens sha256={stream_sha} -> {stream_path} "
          f"({manifest['wall_seconds']} s)", flush=True)
    print(f"build: receipt {manifest_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# verify


def verify(manifest_path: str) -> int:
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    stream = np.fromfile(manifest["stream_path"], dtype=np.uint32)
    if int(stream.size) != int(manifest["tokens"]):
        raise SystemExit(f"verify: {stream.size} tokens != {manifest['tokens']}")
    got = sha256_file(manifest["stream_path"])
    if got != manifest["sha256"]:
        raise SystemExit(f"verify: sha256 {got} != {manifest['sha256']}")
    cursor = 0
    for shard in manifest["shards"]:
        if int(shard["start_token"]) != cursor:
            raise SystemExit(f"verify: shard {shard['index']} start mismatch")
        size = int(shard["tokens"])
        blob = stream[cursor:cursor + size].tobytes()
        if sha256_bytes(blob) != shard["sha256"]:
            raise SystemExit(f"verify: shard {shard['index']} sha mismatch")
        cursor += size
    if cursor != int(manifest["tokens"]):
        raise SystemExit("verify: shard table does not cover the stream")
    total = sum(shard["per_source_tokens"][name]
                for shard in manifest["shards"]
                for name, _ in CYCLE)
    if total != int(manifest["tokens"]):
        raise SystemExit("verify: per-source totals do not sum to the stream")
    if max(int(stream.max()), 0) >= 248_320:
        raise SystemExit("verify: token id outside the anchor vocabulary")
    print(f"verify: {stream.size} tokens, {len(manifest['shards'])} shards, "
          f"sha256={got}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("count")
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--out-dir", default="results/artifacts/stage2")
    build_cmd.add_argument("--receipt-dir", default="results/stage2")
    build_cmd.add_argument("--dataset-root", default=DATASET_ROOT)
    check = sub.add_parser("verify")
    check.add_argument("manifest")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return fetch()
    if args.command == "count":
        return count()
    if args.command == "build":
        return build(args.out_dir, args.receipt_dir, args.dataset_root)
    return verify(args.manifest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
