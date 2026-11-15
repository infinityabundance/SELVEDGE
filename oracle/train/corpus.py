#!/usr/bin/env python3
"""Pinned T1 training corpus: fetch, build the deterministic stream, verify.

Lives inside the oracle court. Network is only used by `fetch`, which is run
through `scripts/fetch-train-corpus.sh` with `--net`. `build` and `verify` are
offline and read only the local cache plus the pinned tokenizer.

Discipline (docs/EXPERIMENT-CONTRACT.md T1.3, T1.5):

  * only the files named in `data/train-corpus-spec.json` are downloaded;
  * every download is hashed against the Hugging Face LFS object id and the
    resolved tree size; a mismatch fails closed and never replaces a file;
  * documents are deduplicated by the SHA-256 of the normalised text, ordered by
    that hash, and filtered for per-row licence (code source only) and for
    contamination against every evaluation item;
  * the tokenised stream is interleaved 70:20:10 at 1024-token blocks and
    written as little-endian uint32 with a sidecar index carrying hashes, source
    spans and repeat factors.

Subcommands: fetch | build | verify
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request

SCHEMA_SPEC = "selvedge/train-corpus-spec/v1"
SCHEMA_LOCK = "selvedge/train-corpus-lock/v1"
SCHEMA_RECEIPT = "selvedge/train-corpus/v1"
USER_AGENT = "selvedge-research/0.1 (offline research program)"
CHUNK = 4 * 1024 * 1024


class CorpusError(Exception):
    pass


# ---------------------------------------------------------------------------
# small helpers


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".corpus-", delete=False)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\S+", normalize(text).lower()))


def shingles(tokens: tuple[str, ...], width: int = 5) -> set[str]:
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


# ---------------------------------------------------------------------------
# fetch


def http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def file_meta(repo_id: str, repo_type: str, revision: str, path: str) -> dict | None:
    """Metadata for one file, from the parent-directory listing.

    The recursive tree listing of a multi-terabyte dataset is paginated and does
    not reliably include every shard, so each file is looked up in its own
    directory instead.
    """
    kind = "datasets" if repo_type == "dataset" else "models"
    parent = os.path.dirname(path)
    url = f"https://huggingface.co/api/{kind}/{repo_id}/tree/{revision}"
    if parent:
        url = f"{url}/{parent}"
    data = http_json(url)
    name = os.path.basename(path)
    for entry in data:
        if entry.get("type") != "file":
            continue
        entry_path = entry.get("path", "")
        if entry_path != path and os.path.basename(entry_path) != name:
            continue
        lfs = entry.get("lfs") or {}
        return {"size": entry.get("size"), "lfs_oid": lfs.get("oid")}
    return None


def download(url: str, target: str) -> tuple[int, str, str]:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    if os.path.exists(partial):
        os.unlink(partial)
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response, open(partial, "wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            digest.update(block)
            handle.write(block)
            size += len(block)
        handle.flush()
        os.fsync(handle.fileno())
    return size, digest.hexdigest(), partial


def fetch(spec_path: str, lock_path: str, cache_root: str, receipt_dir: str,
          network: bool) -> int:
    spec = load_json(spec_path)
    if spec.get("schema") != SCHEMA_SPEC:
        raise CorpusError(f"{spec_path}: unexpected schema {spec.get('schema')!r}")
    previous = {}
    if os.path.exists(lock_path):
        lock = load_json(lock_path)
        if lock.get("schema") != SCHEMA_LOCK:
            raise CorpusError(f"{lock_path}: unexpected schema")
        previous = {(item["source"], item["file"]): item for item in lock.get("files", [])}

    entries = []
    for source in spec["sources"]:
        for name in source["files"]:
            target = os.path.join(cache_root, source["id"], name)
            key = (source["id"], name)
            prior = previous.get(key)
            if os.path.exists(target):
                if prior is None:
                    raise CorpusError(
                        f"{target} exists but is not in the lock; remove it deliberately")
                size = os.path.getsize(target)
                if size != prior["bytes"]:
                    raise CorpusError(f"{target}: size {size} != lock {prior['bytes']}")
                got = sha256_file(target)
                if got != prior["sha256"]:
                    raise CorpusError(f"{target}: sha256 {got} != lock {prior['sha256']}")
                entries.append(prior)
                continue
            if not network:
                raise CorpusError(f"{target} is absent; run the fetch with --net once")
            meta = file_meta(source["repo_id"], source.get("repo_type", "dataset"),
                             source["revision"], name)
            if meta is None:
                raise CorpusError(f"{source['repo_id']}@{source['revision']}: {name} not found")
            url = (f"https://huggingface.co/{source.get('repo_type', 'dataset')}s/"
                   f"{source['repo_id']}/resolve/{source['revision']}/{name}")
            size, sha, partial = download(url, target)
            lfs_oid = meta.get("lfs_oid")
            if lfs_oid and sha != lfs_oid:
                os.unlink(partial)
                raise CorpusError(f"{source['repo_id']}/{name}: sha256 {sha} != LFS {lfs_oid}")
            if meta.get("size") is not None and size != meta["size"]:
                os.unlink(partial)
                raise CorpusError(
                    f"{source['repo_id']}/{name}: {size} bytes, tree reports {meta['size']}")
            os.replace(partial, target)
            entries.append({
                "source": source["id"],
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "file": name,
                "bytes": size,
                "sha256": sha,
                "verified_source": "hf_lfs_oid" if lfs_oid else "computed_sha256",
                "license": source["license"],
                "license_tag": source.get("license_tag"),
            })
            print(f"corpus: fetched {source['id']}/{name} {size} bytes sha256={sha}")

    write_json_atomic(lock_path, {
        "schema": SCHEMA_LOCK,
        "updated_utc": utc_now(),
        "cache_relative_root": cache_root,
        "spec": spec_path,
        "spec_sha256": sha256_file(spec_path),
        "files": entries,
    })
    stamp = stamp_now()
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "kind": "fetch",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "network": network,
        "spec": spec_path,
        "spec_sha256": sha256_file(spec_path),
        "lock": lock_path,
        "lock_sha256": sha256_file(lock_path),
        "files": entries,
    }
    out = os.path.join(receipt_dir, f"fetch-{stamp}.json")
    write_json_atomic(out, receipt)
    print(f"corpus: lock={lock_path} receipt={out}")
    return 0


# ---------------------------------------------------------------------------
# document readers


def iter_documents(source: dict, cache_root: str):
    """Yield dictionaries of the source's text fields for each row."""
    for name in source["files"]:
        path = os.path.join(cache_root, source["id"], name)
        if source["format"] == "parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            columns = source["text_fields"]
            for batch in parquet.iter_batches(batch_size=2048, columns=columns):
                data = batch.to_pydict()
                for index in range(len(data[columns[0]])):
                    row = {column: data[column][index] for column in columns}
                    yield row
        elif source["format"] == "jsonl-gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        else:
            raise CorpusError(f"{source['id']}: unsupported format {source['format']}")


def document_text(source: dict, row: dict) -> str | None:
    try:
        parts = [row[key] for key in source["text_fields"]]
    except (KeyError, TypeError):
        return None
    if any(part is None for part in parts):
        return None
    template = source.get("text_template")
    if template:
        return template.format(**dict(zip(source["text_fields"], parts)))
    return "\n".join(str(part) for part in parts)


def row_license_allowed(source: dict, row: dict) -> bool:
    field = source.get("row_license_field")
    if not field:
        return True
    allow = {value.lower() for value in source["permissive_allowlist"]}
    value = row.get(field)
    return isinstance(value, str) and value.strip().lower() in allow


# ---------------------------------------------------------------------------
# evaluation items for contamination


def evaluation_items(dataset_root: str) -> list[tuple[str, str]]:
    """(kind, text) pairs for every evaluation item the corpus must not contain."""
    import pyarrow.parquet as pq

    items: list[tuple[str, str]] = []
    humaneval = os.path.join(dataset_root, "humaneval-test",
                             "test-00000-of-00001.parquet")
    gsm8k = os.path.join(dataset_root, "gsm8k-test", "test-00000-of-00001.parquet")
    truthfulqa = os.path.join(dataset_root, "truthfulqa-mc-validation",
                              "validation-00000-of-00001.parquet")
    wikitext = os.path.join(dataset_root, "wikitext-2-raw-v1-validation",
                            "validation-00000-of-00001.parquet")
    for path, column, kind in (
        (humaneval, "prompt", "humaneval"),
        (gsm8k, "question", "gsm8k"),
        (truthfulqa, "question", "truthfulqa"),
        (wikitext, "text", "wikitext"),
    ):
        if not os.path.exists(path):
            raise CorpusError(f"missing pinned evaluation file {path}; fetch datasets first")
        table = pq.read_table(path, columns=[column])
        for value in table[column].to_pylist():
            if value:
                items.append((kind, value))
    return items


def tokenizer_path(hf_root: str) -> str:
    base = os.path.join(hf_root, "models", "Qwen", "Qwen3-0.6B",
                        "c1899de289a04d12100db370d81485cdf75e47ca")
    if not os.path.isdir(base):
        raise CorpusError(f"tokenizer path {base} is absent; fetch models first")
    return base


def load_tokenizer(hf_root: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path(hf_root), local_files_only=True,
                                         trust_remote_code=False)


# ---------------------------------------------------------------------------
# build


def build(spec_path: str, lock_path: str, cache_root: str, dataset_root: str,
          hf_root: str, artifact_dir: str, receipt_dir: str) -> int:
    import numpy as np

    spec = load_json(spec_path)
    if spec.get("schema") != SCHEMA_SPEC:
        raise CorpusError(f"{spec_path}: unexpected schema")
    lock = load_json(lock_path)
    locked = {(item["source"], item["file"]): item for item in lock["files"]}
    for source in spec["sources"]:
        for name in source["files"]:
            entry = locked.get((source["id"], name))
            path = os.path.join(cache_root, source["id"], name)
            if entry is None or not os.path.exists(path):
                raise CorpusError(f"{path}: not in the lock or absent; run fetch")
            if sha256_file(path) != entry["sha256"]:
                raise CorpusError(f"{path}: sha256 mismatch against the lock")

    tokenizer = load_tokenizer(hf_root)
    eos = tokenizer.eos_token_id
    stream_spec = spec["stream"]
    block_tokens = int(stream_spec["block_tokens"])
    budget = int(stream_spec["token_budget"])
    headroom = float(stream_spec["headroom_fraction"])
    target_total = int(round(budget * (1.0 + headroom)))

    eval_pairs = evaluation_items(dataset_root)
    eval_bags: list[set[str]] = []
    inverted: dict[str, list[int]] = {}
    for index, (_kind, text) in enumerate(eval_pairs):
        bag = shingles(words(text))
        eval_bags.append(bag)
        for shingle in bag:
            inverted.setdefault(shingle, []).append(index)

    mixes = {entry["source"]: entry["fraction"] for entry in stream_spec["mix"]}
    if abs(sum(mixes.values()) - 1.0) > 1e-9:
        raise CorpusError("stream mix does not sum to 1")
    mix_sources = [entry["source"] for entry in stream_spec["mix"]]
    required = {name: int(round(target_total * mixes[name])) for name in mix_sources}

    tokenized: dict[str, list[tuple[str, "np.ndarray"]]] = {name: [] for name in mix_sources}
    stats = {name: {"rows": 0, "licence_discarded": 0, "duplicate_discarded": 0,
                    "contamination_discarded": 0, "tokens": 0} for name in mix_sources}
    seen_hashes: set[str] = set()
    max_contamination = 0.0
    hard_contamination: list[str] = []

    for source in spec["sources"]:
        name = source["id"]
        if name not in required:
            continue
        need = required[name]
        for row in iter_documents(source, cache_root):
            stats[name]["rows"] += 1
            if not row_license_allowed(source, row):
                stats[name]["licence_discarded"] += 1
                continue
            text = document_text(source, row)
            if not text or len(normalize(text)) < 200:
                continue
            normalized = normalize(text)
            doc_hash = sha256_bytes(normalized.encode("utf-8"))
            if doc_hash in seen_hashes:
                stats[name]["duplicate_discarded"] += 1
                continue
            bag = shingles(words(text))
            intersection: dict[int, int] = {}
            for shingle in bag:
                for key in inverted.get(shingle, ()):
                    intersection[key] = intersection.get(key, 0) + 1
            worst = 0.0
            for key, shared in intersection.items():
                other = eval_bags[key]
                union = len(bag) + len(other) - shared
                score = shared / union if union else 0.0
                if score > worst:
                    worst = score
            if worst > max_contamination:
                max_contamination = worst
            if worst >= 0.2:
                stats[name]["contamination_discarded"] += 1
                if worst >= 0.5:
                    hard_contamination.append(f"{name}:{doc_hash[:16]}:{worst:.3f}")
                continue
            seen_hashes.add(doc_hash)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            tokens = np.asarray(ids, dtype=np.uint32)
            tokenized[name].append((doc_hash, tokens))
            stats[name]["tokens"] += int(tokens.size)
            if stats[name]["tokens"] >= need:
                break
        if stats[name]["tokens"] == 0:
            raise CorpusError(f"{name}: no usable documents after filtering; corpus build stops")

    if hard_contamination:
        raise CorpusError(
            f"contamination >= 0.5 for {len(hard_contamination)} document(s): "
            f"{hard_contamination[:5]}; corpus build fails closed (K10)")

    # Deterministic order: by document hash, then interleave in 1024-token blocks.
    per_source_blocks: dict[str, list["np.ndarray"]] = {}
    repeat_factor: dict[str, float] = {}
    for name in mix_sources:
        docs = sorted(tokenized[name], key=lambda item: item[0])
        flat = np.concatenate([tokens for _, tokens in docs]) if docs else np.zeros(0, np.uint32)
        available = int(flat.size)
        need = required[name]
        repeats = 0
        parts = [flat]
        total = available
        while total < need + block_tokens:
            parts.append(flat)
            total += available
            repeats += 1
        joined = np.concatenate(parts)
        repeat_factor[name] = 1 + repeats
        blocks = [joined[start:start + block_tokens]
                  for start in range(0, joined.size - block_tokens + 1, block_tokens)]
        per_source_blocks[name] = blocks

    # Weighted round-robin at block granularity: 7 text, 2 code, 1 math blocks.
    schedule: list[str] = []
    counts = {name: 0 for name in mix_sources}
    total_blocks = (target_total + block_tokens - 1) // block_tokens
    weights = {name: int(round(mixes[name] * 10)) for name in mix_sources}
    while len(schedule) < total_blocks:
        for name in mix_sources:
            for _ in range(weights[name]):
                if len(schedule) >= total_blocks:
                    break
                schedule.append(name)
    cursor = {name: 0 for name in mix_sources}
    pieces = []
    source_of_block = []
    for name in schedule:
        index = cursor[name] % len(per_source_blocks[name])
        cursor[name] += 1
        pieces.append(per_source_blocks[name][index])
        source_of_block.append(name)
    out_stream = np.empty(len(pieces) * (block_tokens + 1), dtype="<u4")
    for index, piece in enumerate(pieces):
        offset = index * (block_tokens + 1)
        out_stream[offset:offset + block_tokens] = piece
        out_stream[offset + block_tokens] = eos

    os.makedirs(artifact_dir, exist_ok=True)
    stream_path = os.path.join(artifact_dir, "train.bin")
    tmp = stream_path + ".tmp"
    out_stream.tofile(tmp)
    os.replace(tmp, stream_path)
    stream_sha = sha256_file(stream_path)

    index_payload = {
        "schema": "selvedge/train-stream/v1",
        "spec": spec_path,
        "spec_sha256": sha256_file(spec_path),
        "lock_sha256": sha256_file(lock_path),
        "stream_file": os.path.relpath(stream_path, os.path.dirname(artifact_dir) or "."),
        "stream_sha256": stream_sha,
        "dtype": "uint32_le",
        "block_tokens": block_tokens,
        "separator_token": eos,
        "blocks": len(pieces),
        "tokens": int(out_stream.size),
        "registered_budget_tokens": budget,
        "schedule": {
            "pattern": mix_sources,
            "weights_per_10": weights,
            "blocks_per_source": {name: source_of_block.count(name) for name in mix_sources},
        },
        "sources": {
            name: {
                "stats": stats[name],
                "repeat_factor": repeat_factor[name],
                "blocks_available": len(per_source_blocks[name]),
                "blocks_used": schedule.count(name),
            }
            for name in mix_sources
        },
        "contamination": {
            "eval_items": len(eval_pairs),
            "max_shingle_jaccard": max_contamination,
            "threshold_discard": 0.2,
            "threshold_fail_closed": 0.5,
        },
    }
    index_path = os.path.join(artifact_dir, "train.index.json")
    write_json_atomic(index_path, index_payload)

    stamp = stamp_now()
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "kind": "build",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "spec_sha256": index_payload["spec_sha256"],
        "stream_sha256": stream_sha,
        "stream_bytes": os.path.getsize(stream_path),
        "tokens": int(out_stream.size),
        "registered_budget_tokens": budget,
        "index": index_path,
        "index_sha256": sha256_file(index_path),
        "sources": index_payload["sources"],
        "contamination": index_payload["contamination"],
        "tokenizer": tokenizer_path(os.environ.get("HF_HOME", "/hf-cache")),
    }
    out = os.path.join(receipt_dir, f"corpus-{stamp}.json")
    write_json_atomic(out, receipt)
    print(json.dumps({key: receipt[key] for key in
                      ("stamp", "tokens", "stream_sha256", "stream_bytes", "contamination")},
                     indent=2, sort_keys=True))
    print(f"corpus: stream={stream_path} index={index_path} receipt={out}")
    return 0


# ---------------------------------------------------------------------------
# verify


def verify(spec_path: str, lock_path: str, cache_root: str, artifact_dir: str,
           receipt_dir: str) -> int:
    spec = load_json(spec_path)
    lock = load_json(lock_path)
    for item in lock["files"]:
        path = os.path.join(cache_root, item["source"], item["file"])
        got = sha256_file(path)
        if got != item["sha256"] or os.path.getsize(path) != item["bytes"]:
            raise CorpusError(f"{path}: lock mismatch")
    index_path = os.path.join(artifact_dir, "train.index.json")
    index = load_json(index_path)
    stream_path = os.path.join(artifact_dir, "train.bin")
    got = sha256_file(stream_path)
    if got != index["stream_sha256"]:
        raise CorpusError(f"{stream_path}: sha256 {got} != index {index['stream_sha256']}")
    if os.path.getsize(stream_path) != 4 * index["tokens"]:
        raise CorpusError(f"{stream_path}: size is not 4 bytes x {index['tokens']}")
    receipts = sorted(os.listdir(receipt_dir)) if os.path.isdir(receipt_dir) else []
    print(f"corpus: verified {len(lock['files'])} files, {index['tokens']} tokens, "
          f"sha256={got}")
    print(f"corpus: {len(receipts)} receipts in {receipt_dir}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "build", "verify"])
    parser.add_argument("--spec", default="data/train-corpus-spec.json")
    parser.add_argument("--lock", default="data/train-corpus.lock.json")
    parser.add_argument("--cache-root", default="results/artifacts/train-corpus")
    parser.add_argument("--dataset-root", default="results/artifacts/datasets")
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--receipt-dir", default="results/train")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return fetch(args.spec, args.lock, args.cache_root, args.receipt_dir, network=True)
    if args.command == "build":
        return build(args.spec, args.lock, args.cache_root, args.dataset_root, args.hf_root,
                     args.artifact_dir, args.receipt_dir)
    return verify(args.spec, args.lock, args.cache_root, args.artifact_dir, args.receipt_dir)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CorpusError as error:
        print(f"corpus: {error}", file=sys.stderr)
        sys.exit(1)
