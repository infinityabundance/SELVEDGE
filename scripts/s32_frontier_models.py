#!/usr/bin/env python3
"""Verify the proposed frontier models against the Hugging Face Hub, exactly.

This script exists because `docs/PROGRAM-PLAN.md` WS19 proposes a target model
family that SELVEDGE has never fetched. Nothing about those models may be
asserted from memory: every fact in the receipt is either a byte the Hub
returned, a hash computed here, or arithmetic over a safetensors header read
with an HTTP range request (so accounting costs no weight download).

Subcommands:

  probe  -- for every candidate id: resolve the revision, list the tree, read
             the small files (config/tokenizer/generation/LICENSE/README), and
             account every tensor from the safetensors header by range read.
             Also records the Hub search that was used when an id is missing.
  fetch  -- download one model's weight files into the persistent cache,
             verify each LFS SHA-256 against the Hub, and record the result.
  disk   -- arithmetic over the probe receipt: exact checkpoint bytes and the
             teacher top-k logit-cache sizes at the requested token budgets.

Limits: the process is memory-bounded (it never holds a response larger than
--max-response-bytes and hashes weights in a fixed-size chunk). It is run inside
the Docker court with an explicit SELVEDGE_MEM; it sets no limits itself.

Every run writes one JSON receipt with `stamp` and `revision` top-level fields so
`scripts/perf_numbers.py` can source published numbers from it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA_PROBE = "selvedge/frontier-model-probe/v1"
SCHEMA_FETCH = "selvedge/frontier-model-fetch/v1"
SCHEMA_DISK = "selvedge/frontier-disk-arithmetic/v1"
USER_AGENT = "SELVEDGE-frontier-verification/1 (+model identity verification)"

CANDIDATES = [
    ("Qwen/Qwen3.5-0.8B", "chat/instruct; proposed anchor"),
    ("Qwen/Qwen3.5-0.8B-Base", "base checkpoint; proposed anchor"),
    ("Qwen/Qwen3.5-2B", "chat/instruct; proposed gap upper end"),
    ("Qwen/Qwen3.5-2B-Base", "base checkpoint; proposed gap upper end"),
    ("Qwen/Qwen3.5-4B", "chat/instruct; size reference"),
    ("Qwen/Qwen3.5-4B-Base", "base checkpoint; size reference"),
    ("Qwen/Qwen3.5-9B", "chat/instruct; size reference"),
    ("Qwen/Qwen3.5-9B-Base", "base checkpoint; size reference"),
    ("LiquidAI/LFM2.5-1.2B", "proposed stretch model"),
    ("LiquidAI/LFM2.5-1.2B-Instruct", "proposed stretch model (instruct)"),
    ("LiquidAI/LFM2.5-1.2B-Base", "proposed stretch model (base)"),
    ("google/gemma-3-1b-it", "proposed 1B-class reference"),
]

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

# Small files only: never a weight file here. Weights are `fetch`'s job.
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "LICENSE",
    "LICENSE.txt",
    "README.md",
]


class ProbeError(Exception):
    pass


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# HTTP


def open_url(url: str, extra_headers: dict[str, str] | None = None):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)


def http_json(url: str, max_bytes: int) -> tuple[int, object | None, str]:
    """GET a JSON document; returns (status, parsed-or-None, detail)."""
    try:
        with open_url(url) as response:
            status = response.status
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", "replace")
        return exc.code, None, detail
    except Exception as exc:  # network failure is a finding, not a crash
        return 0, None, f"{type(exc).__name__}: {exc}"
    if len(body) > max_bytes:
        return status, None, f"response exceeded {max_bytes} bytes; refused to buffer"
    try:
        return status, json.loads(body), ""
    except ValueError as exc:
        return status, None, f"not JSON: {exc}"


def resolve(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def range_bytes(url: str, start: int, length: int, limit: int) -> tuple[bytes, str]:
    """Read exactly `length` bytes at `start`; refuses to over-read."""
    if length <= 0 or length > limit:
        raise ProbeError(f"refusing range read of {length} bytes (limit {limit})")
    end = start + length - 1
    try:
        with open_url(url, {"Range": f"bytes={start}-{end}"}) as response:
            if response.status == 206:
                body = response.read(length)
            else:
                # Server ignored Range: take only what we need and hang up.
                body = response.read(length)
                return body, f"server returned {response.status}; read first {length} bytes"
    except urllib.error.HTTPError as exc:
        return b"", f"HTTP {exc.code}"
    except Exception as exc:
        return b"", f"{type(exc).__name__}: {exc}"
    return body, ""


# ---------------------------------------------------------------------------
# safetensors header accounting


def header_accounting(url: str, limit: int) -> dict:
    """Read a safetensors header by range request and account every tensor."""
    prefix, detail = range_bytes(url, 0, 8, limit)
    if len(prefix) != 8:
        return {"ok": False, "error": f"cannot read length prefix: {detail or 'short read'}"}
    header_len = int.from_bytes(prefix, "little")
    if header_len <= 0 or header_len > limit:
        return {"ok": False, "error": f"header length {header_len} outside (0, {limit}]"}
    header, detail = range_bytes(url, 8, header_len, limit)
    if len(header) != header_len:
        return {"ok": False, "error": f"short header read: {detail or len(header)}"}
    try:
        document = json.loads(header)
    except ValueError as exc:
        return {"ok": False, "error": f"header is not JSON: {exc}"}

    tensors = []
    total_bytes = 0
    for name, meta in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            return {"ok": False, "error": f"tensor {name!r} metadata is not an object"}
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            return {"ok": False, "error": f"tensor {name!r} has unknown dtype {dtype!r}"}
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            return {"ok": False, "error": f"tensor {name!r} has malformed shape/offsets"}
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                return {"ok": False, "error": f"tensor {name!r} has bad dimension {dimension!r}"}
            elements *= dimension
        if offsets[1] - offsets[0] != elements * DTYPE_BYTES[dtype]:
            return {
                "ok": False,
                "error": f"tensor {name!r} byte span disagrees with shape and dtype",
            }
        total_bytes += elements * DTYPE_BYTES[dtype]
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "elements": elements,
                "bytes": elements * DTYPE_BYTES[dtype],
            }
        )
    tensors.sort(key=lambda item: item["name"])
    return {
        "ok": True,
        "header_bytes": header_len,
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "metadata": document.get("__metadata__"),
        "tensors": tensors,
    }


def classify(name: str) -> str:
    """Bucket a tensor name for the text/vision/MTP split.

    Returns "unknown" rather than guessing; the probe records the count so an
    unclassified tensor is a visible finding, never a silent omission.
    """
    lowered = name.lower()
    if lowered.startswith("mtp") or ".mtp." in lowered:
        return "mtp"
    if "visual" in lowered or "vision" in lowered:
        return "vision"
    if (
        "language_model" in lowered
        or lowered.startswith("model.layers")
        or lowered.startswith("model.embed_tokens")
        or lowered.startswith("model.embedding_norm")
        or lowered.startswith("model.norm")
        or lowered.startswith("lm_head")
    ):
        return "text"
    return "unknown"


# ---------------------------------------------------------------------------
# probe


def probe_model(repo: str, role: str, args) -> dict:
    record = {
        "id": repo,
        "role": role,
        "urls": [],
        "exists": False,
        "accessed_utc": utcnow(),
    }
    api_url = f"https://huggingface.co/api/models/{repo}"
    record["urls"].append({"method": "hf_api_model", "url": api_url})
    status, meta, detail = http_json(api_url, args.max_response_bytes)
    record["api_status"] = status
    if status != 200 or not isinstance(meta, dict):
        record["missing"] = {
            "status": status,
            "detail": detail[:400],
            "search_url": f"https://huggingface.co/api/models?search={repo.split('/')[-1]}",
        }
        search_status, search, search_detail = http_json(
            record["missing"]["search_url"], args.max_response_bytes
        )
        record["missing"]["search_status"] = search_status
        if isinstance(search, list):
            record["missing"]["search_candidates"] = [item.get("id") for item in search[:20]]
        else:
            record["missing"]["search_detail"] = search_detail[:400]
        return record

    record["exists"] = True
    record["revision"] = meta.get("sha")
    record["last_modified"] = meta.get("lastModified")
    record["created_at"] = meta.get("createdAt")
    record["gated"] = meta.get("gated")
    record["downloads"] = meta.get("downloads")
    record["used_storage_bytes"] = meta.get("usedStorage")
    record["card_license"] = (meta.get("cardData") or {}).get("license")
    record["tags"] = sorted(meta.get("tags") or [])
    record["architectures"] = ((meta.get("config") or {}).get("architectures") or [])
    record["hub_safetensors"] = meta.get("safetensors")

    revision = record["revision"]
    if not isinstance(revision, str) or len(revision) < 7:
        record["error"] = "hub returned no commit sha; cannot pin"
        return record

    tree_url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    record["urls"].append({"method": "hf_tree", "url": tree_url})
    tree_status, tree, tree_detail = http_json(tree_url, args.max_response_bytes)
    record["tree_status"] = tree_status
    if tree_status != 200 or not isinstance(tree, list):
        record["error"] = f"tree listing failed: {tree_status} {tree_detail[:200]}"
        return record
    files = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        files.append(
            {
                "path": entry.get("path"),
                "bytes": entry.get("size") if lfs.get("size") is None else lfs.get("size"),
                "lfs_oid": lfs.get("oid"),
            }
        )
    files.sort(key=lambda item: item["path"] or "")
    record["files"] = files
    record["file_count"] = len(files)

    weight_files = [item for item in files if item["path"].endswith(".safetensors")]
    record["weight_files"] = weight_files
    record["weight_bytes"] = sum(int(item["bytes"] or 0) for item in weight_files)
    record["total_bytes"] = sum(int(item["bytes"] or 0) for item in files)

    # Small files, hashed and cached. Weights are never downloaded here.
    cache_root = args.cache_dir
    by_path = {item["path"]: item for item in files}
    small = []
    for path in SMALL_FILES:
        if path not in by_path or args.small_files == "none":
            continue
        target_url = resolve(repo, revision, path)
        expected = int(by_path[path]["bytes"] or 0)
        if expected > args.max_small_file_bytes:
            small.append({"path": path, "bytes": expected, "skipped": "too large to hash"})
            continue
        entry = {"path": path, "bytes": expected, "url": target_url}
        try:
            with open_url(target_url) as response:
                body = response.read(args.max_small_file_bytes + 1)
        except urllib.error.HTTPError as exc:
            entry["error"] = f"HTTP {exc.code}"
            small.append(entry)
            continue
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            small.append(entry)
            continue
        entry["fetched_bytes"] = len(body)
        entry["sha256"] = hashlib.sha256(body).hexdigest()
        entry["lfs_oid"] = by_path[path].get("lfs_oid")
        if by_path[path].get("lfs_oid") and by_path[path]["lfs_oid"] != entry["sha256"]:
            entry["error"] = "computed sha256 does not match the Hub LFS oid"
        if len(body) != expected:
            entry["error"] = f"byte count {len(body)} does not match the tree listing {expected}"
        if cache_root and "error" not in entry:
            directory = os.path.join(cache_root, "models", repo, revision)
            os.makedirs(directory, exist_ok=True)
            destination = os.path.join(directory, os.path.basename(path))
            with open(destination, "wb") as handle:
                handle.write(body)
            entry["cached_at"] = destination
        if path == "config.json":
            try:
                record["config"] = json.loads(body)
            except ValueError as exc:
                record["config_error"] = str(exc)
        if path in ("LICENSE", "LICENSE.txt"):
            text = body.decode("utf-8", "replace")
            record["license_file"] = {
                "path": path,
                "sha256": entry["sha256"],
                "first_lines": text.strip().splitlines()[:6],
                "bytes": len(body),
            }
        small.append(entry)
    record["small_files"] = small

    # Exact per-tensor accounting from the header, one range read per shard.
    shards = []
    for item in weight_files[: args.max_shards]:
        target_url = resolve(repo, revision, item["path"])
        accounting = header_accounting(target_url, args.max_header_bytes)
        accounting["path"] = item["path"]
        accounting["url"] = target_url
        accounting["declared_bytes"] = item["bytes"]
        if accounting.get("ok"):
            accounting["bytes_match_declared"] = (
                int(accounting["tensor_bytes"]) + accounting["header_bytes"] + 8
            ) == int(item["bytes"] or 0)
        shards.append(accounting)
    record["shards"] = shards

    totals = {"text": 0, "vision": 0, "mtp": 0, "unknown": 0}
    by_dtype: dict[str, int] = {}
    unknown_names: list[str] = []
    dtypes_by_group: dict[str, dict[str, int]] = {}
    num_shards = 0
    for shard in shards:
        if not shard.get("ok"):
            continue
        num_shards += 1
        for tensor in shard["tensors"]:
            group = classify(tensor["name"])
            totals[group] += tensor["elements"]
            by_dtype[tensor["dtype"]] = by_dtype.get(tensor["dtype"], 0) + tensor["elements"]
            dtypes_by_group.setdefault(group, {})
            dtypes_by_group[group][tensor["dtype"]] = (
                dtypes_by_group[group].get(tensor["dtype"], 0) + tensor["elements"]
            )
            if group == "unknown" and len(unknown_names) < 64:
                unknown_names.append(tensor["name"])
    record["accounting"] = {
        "shards_read": num_shards,
        "shards_declared": len(weight_files),
        "complete": num_shards == len(weight_files) and 0 <= len(weight_files) <= args.max_shards,
        "elements_by_group": totals,
        "elements_by_dtype": by_dtype,
        "elements_by_group_and_dtype": dtypes_by_group,
        "total_elements": sum(totals.values()),
        "unknown_tensor_names": unknown_names,
        "classification_complete": totals["unknown"] == 0,
    }
    hub_total = (meta.get("safetensors") or {}).get("total")
    record["accounting"]["hub_reported_total"] = hub_total
    if record["accounting"]["complete"] and isinstance(hub_total, int):
        record["accounting"]["matches_hub_total"] = hub_total == record["accounting"]["total_elements"]
    return record


def cmd_probe(args) -> int:
    receipt = {
        "schema": SCHEMA_PROBE,
        "stamp": stamp(),
        "revision": git_revision(),
        "accessed_utc": utcnow(),
        "method": (
            "Hugging Face Hub HTTP API: /api/models/<id> resolves the commit sha; "
            "/api/models/<id>/tree/<sha>?recursive=true lists every file with its "
            "size and LFS oid; small files are fetched and SHA-256 hashed; every "
            "safetensors tensor is enumerated from the file header (8-byte length "
            "prefix + JSON) obtained with an HTTP range request, so no weight is "
            "downloaded to count it. Parameter counts are element counts; tie "
            "handling is read from the config and from whether lm_head.weight is "
            "present in the header."
        ),
        "identities": [{"id": repo, "role": role} for repo, role in CANDIDATES],
        "python": sys.version.split()[0],
        "models": [],
    }
    failures = 0
    for repo, role in CANDIDATES:
        started = time.time()
        try:
            record = probe_model(repo, role, args)
        except Exception as exc:  # never lose the whole receipt to one model
            record = {"id": repo, "role": role, "exists": False, "error": f"{type(exc).__name__}: {exc}"}
        record["elapsed_s"] = round(time.time() - started, 2)
        receipt["models"].append(record)
        state = "EXISTS" if record.get("exists") else "MISSING"
        total = (record.get("accounting") or {}).get("total_elements")
        print(f"{repo}: {state} revision={record.get('revision')} params={total}", flush=True)
        if not record.get("exists"):
            failures += 1
        time.sleep(args.sleep)

    receipt["exist_count"] = sum(1 for model in receipt["models"] if model.get("exists"))
    receipt["missing_count"] = len(receipt["models"]) - receipt["exist_count"]
    write_receipt(args.out, receipt)
    print(f"probe: {receipt['exist_count']} exist, {receipt['missing_count']} missing -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# fetch


def cmd_fetch(args) -> int:
    """Download weight files for one pinned revision and verify every hash."""
    api_url = f"https://huggingface.co/api/models/{args.id}"
    status, meta, detail = http_json(api_url, 8 * 1024 * 1024)
    if status != 200 or not isinstance(meta, dict):
        raise ProbeError(f"cannot resolve {args.id}: {status} {detail[:200]}")
    revision = meta.get("sha")
    tree_url = f"https://huggingface.co/api/models/{args.id}/tree/{revision}?recursive=true"
    tree_status, tree, tree_detail = http_json(tree_url, 64 * 1024 * 1024)
    if tree_status != 200 or not isinstance(tree, list):
        raise ProbeError(f"cannot list {args.id}: {tree_status} {tree_detail[:200]}")
    files = [
        {
            "path": entry["path"],
            "bytes": (entry.get("lfs") or {}).get("size", entry.get("size")),
            "lfs_oid": (entry.get("lfs") or {}).get("oid"),
        }
        for entry in tree
        if entry.get("type") == "file"
    ]
    wanted = [item for item in files if item["path"].endswith(".safetensors") or item["path"].endswith(".json")]
    wanted = [item for item in wanted if item["path"] != "chat_template.jinja"]
    directory = os.path.join(args.cache_dir, "models", args.id, revision)
    os.makedirs(directory, exist_ok=True)

    free_before = shutil_free(args.cache_dir)
    needed = sum(int(item["bytes"] or 0) for item in wanted)
    if needed + args.headroom_bytes > free_before:
        raise ProbeError(
            f"refusing to fetch: {needed} bytes needed + {args.headroom_bytes} headroom "
            f"exceeds {free_before} bytes free on {args.cache_dir}"
        )

    receipt = {
        "schema": SCHEMA_FETCH,
        "stamp": stamp(),
        "revision": git_revision(),
        "accessed_utc": utcnow(),
        "id": args.id,
        "model_revision": revision,
        "cache_dir": args.cache_dir,
        "cache_free_bytes_before": free_before,
        "cache_free_bytes_after": None,
        "requested_bytes": needed,
        "files": [],
    }
    for item in wanted:
        target = os.path.join(directory, item["path"])
        url = resolve(args.id, revision, item["path"])
        digest = hashlib.sha256()
        written = 0
        started = time.time()
        try:
            with open_url(url) as response, open(target, "wb") as handle:
                while True:
                    chunk = response.read(args.chunk_bytes)
                    if not chunk:
                        break
                    digest.update(chunk)
                    handle.write(chunk)
                    written += len(chunk)
                    if written > int(item["bytes"] or 0) + 1:
                        raise ProbeError(f"{item['path']} exceeded its declared size")
        except Exception as exc:
            receipt["files"].append({"path": item["path"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        sha = digest.hexdigest()
        entry = {
            "path": item["path"],
            "url": url,
            "bytes": written,
            "declared_bytes": item["bytes"],
            "sha256": sha,
            "lfs_oid": item["lfs_oid"],
            "sha256_matches_lfs_oid": (item["lfs_oid"] == sha) if item["lfs_oid"] else None,
            "bytes_match_declared": written == int(item["bytes"] or 0),
            "elapsed_s": round(time.time() - started, 2),
            "cached_at": target,
        }
        receipt["files"].append(entry)
        print(f"  {item['path']}: {written} bytes sha={sha[:16]} ok={entry['sha256_matches_lfs_oid']}", flush=True)
    receipt["cache_free_bytes_after"] = shutil_free(args.cache_dir)
    receipt["failures"] = [entry["path"] for entry in receipt["files"] if "error" in entry]
    write_receipt(args.out, receipt)
    return 0 if not receipt["failures"] else 1


def shutil_free(path: str) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


# ---------------------------------------------------------------------------
# disk arithmetic

CACHE_BUDGETS = [
    (500_000_000, 64),
    (500_000_000, 32),
    (250_000_000, 64),
    (250_000_000, 32),
    (100_000_000, 64),
    (100_000_000, 32),
]


def logit_cache_bytes(tokens: int, k: int) -> int:
    """n_tokens x (k x (4 + 2) + 4) bytes: per entry a u32 token id and a bf16
    logit, plus a 4-byte per-token header."""
    return tokens * (k * (4 + 2) + 4)


def cmd_disk(args) -> int:
    with open(args.probe, "r", encoding="utf-8") as handle:
        probe = json.load(handle)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_stat = os.statvfs(repo_root)
    cache_stat = os.statvfs(args.cache_dir)
    repo_free = repo_stat.f_bavail * repo_stat.f_frsize
    cache_free = cache_stat.f_bavail * cache_stat.f_frsize

    models = []
    for record in probe["models"]:
        if not record.get("exists"):
            continue
        accounting = record.get("accounting") or {}
        models.append(
            {
                "id": record["id"],
                "revision": record.get("revision"),
                "weight_bytes": record.get("weight_bytes"),
                "total_repo_bytes": record.get("total_bytes"),
                "total_elements": accounting.get("total_elements"),
                "elements_by_group": accounting.get("elements_by_group"),
                "matches_hub_total": accounting.get("matches_hub_total"),
            }
        )

    caches = []
    for tokens, k in CACHE_BUDGETS:
        caches.append(
            {
                "tokens": tokens,
                "k": k,
                "bytes": logit_cache_bytes(tokens, k),
                "fits_repo_with_headroom": logit_cache_bytes(tokens, k)
                <= repo_free - args.repo_headroom_bytes,
            }
        )
    inside = [item for item in caches if item["fits_repo_with_headroom"]]
    largest = max(inside, key=lambda item: item["bytes"]) if inside else None
    largest_k64 = None
    inside64 = [item for item in inside if item["k"] == 64]
    if inside64:
        largest_k64 = max(inside64, key=lambda item: item["bytes"])

    receipt = {
        "schema": SCHEMA_DISK,
        "stamp": stamp(),
        "revision": git_revision(),
        "accessed_utc": utcnow(),
        "probe_receipt": args.probe,
        "probe_stamp": probe.get("stamp"),
        "probe_revision": probe.get("revision"),
        "filesystems": {
            "repo_root": repo_root,
            "repo_free_bytes": repo_free,
            "cache_dir": args.cache_dir,
            "cache_free_bytes": cache_free,
        },
        "headroom": {
            "repo_headroom_bytes": args.repo_headroom_bytes,
            "cache_headroom_bytes": args.cache_headroom_bytes,
        },
        "models": models,
        "checkpoint_bytes": {item["id"]: item["weight_bytes"] for item in models},
        "cache_formula": "n_tokens * (k * (4 + 2) + 4)",
        "teacher_caches": caches,
        "largest_fitting_cache": largest,
        "largest_fitting_k64_cache": largest_k64,
        "plan_190gb_fits": 190_000_000_000 <= repo_free - args.repo_headroom_bytes,
        "cache_download_budget_bytes": cache_free - args.cache_headroom_bytes,
    }
    derived = []
    for item in models:
        elements = item.get("total_elements")
        if not elements:
            continue
        derived.append(
            {
                "id": item["id"],
                "bf16_bytes": elements * 2,
                "f32_bytes": elements * 4,
                "note": "an uncompressed SELVEDGE bundle at the stated dtype; the "
                "storage format is defined in docs/STORAGE.md, not assumed here",
            }
        )
    receipt["derived_bundles"] = derived
    write_receipt(args.out, receipt)
    print(json.dumps({"models": len(models), "caches": len(caches)}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# ceiling arithmetic

FFN_MARKERS = (".mlp.", ".feed_forward.")
ATTENTION_MARKERS = (".self_attn.", ".linear_attn.")
EMBEDDING_MARKERS = ("embed_tokens", "embedding_norm", "lm_head")


def bucket_tensor(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in FFN_MARKERS):
        return "ffn"
    if any(marker in lowered for marker in ATTENTION_MARKERS):
        return "attention"
    if any(marker in lowered for marker in EMBEDDING_MARKERS):
        return "embedding"
    return "norm"


def cmd_ceiling(args) -> int:
    """Exact always-active vs removable split, for the 1e9 ceiling arithmetic."""
    with open(args.probe, "r", encoding="utf-8") as handle:
        probe = json.load(handle)
    models = []
    for record in probe["models"]:
        accounting = record.get("accounting") or {}
        if not accounting.get("complete"):
            models.append(
                {
                    "id": record["id"],
                    "ok": False,
                    "reason": "accounting incomplete (weights not readable without access)",
                }
            )
            continue
        components = {
            "text": {"ffn": 0, "attention": 0, "embedding": 0, "norm": 0},
            "mtp": {"ffn": 0, "attention": 0, "embedding": 0, "norm": 0},
            "vision": {"ffn": 0, "attention": 0, "embedding": 0, "norm": 0},
        }
        lm_head_present = False
        for shard in record["shards"]:
            if not shard.get("ok"):
                continue
            for tensor in shard["tensors"]:
                group = classify(tensor["name"])
                if group == "unknown":
                    raise ProbeError(f"unclassified tensor {tensor['name']!r}")
                if tensor["name"] == "lm_head.weight":
                    lm_head_present = True
                components[group][bucket_tensor(tensor["name"])] += tensor["elements"]
        text = sum(components["text"].values())
        config = record.get("config") or {}
        text_config = config.get("text_config") or config
        tie_declared = bool(
            text_config.get("tie_word_embeddings", config.get("tie_word_embeddings"))
        )
        total = sum(sum(bucket.values()) for bucket in components.values())
        mtp = sum(components["mtp"].values())
        layer_types = text_config.get("layer_types") or []
        attention_layers = [
            index for index, kind in enumerate(layer_types) if kind == "full_attention"
        ]
        linear_layers = [
            index for index, kind in enumerate(layer_types) if kind == "linear_attention"
        ]
        models.append(
            {
                "id": record["id"],
                "ok": True,
                "revision": record.get("revision"),
                "attention_layer_count": len(attention_layers),
                "linear_attention_layer_count": len(linear_layers),
                "attention_layer_indices": attention_layers,
                "other_layer_types": sorted(
                    {kind for kind in layer_types if kind not in ("full_attention", "linear_attention")}
                ),
                "layer_type_total": len(layer_types),
                "components": components,
                "text_elements": text,
                "vision_elements": sum(components["vision"].values()),
                "mtp_elements": mtp,
                "total_elements": total,
                "text_always_active_elements": text - components["text"]["ffn"],
                "text_ffn_elements": components["text"]["ffn"],
                "text_executed_per_token_elements": text,
                "text_plus_mtp_per_token_elements": text + mtp,
                "tie_word_embeddings_declared": tie_declared,
                "lm_head_present": lm_head_present,
                "tie_verified_at_checkpoint": tie_declared and not lm_head_present,
                "under_ceiling_full_checkpoint": total < args.ceiling,
                "under_ceiling_text_path": text < args.ceiling,
                "headroom_to_ceiling_elements": args.ceiling - text,
            }
        )
    receipt = {
        "schema": "selvedge/frontier-ceiling/v1",
        "stamp": stamp(),
        "revision": git_revision(),
        "accessed_utc": utcnow(),
        "probe_receipt": args.probe,
        "probe_stamp": probe.get("stamp"),
        "probe_revision": probe.get("revision"),
        "ceiling_elements": args.ceiling,
        "definition": (
            "A dense token's unique-active elements are the elements its execution "
            "touches, each counted once (ties once). The 'text path' excludes the "
            "vision tower because a text-only runtime never executes it; it is "
            "reported with and without the MTP head because whether MTP executes per "
            "token depends on the decoding mode."
        ),
        "models": models,
    }
    write_receipt(args.out, receipt)
    for model in models:
        if model["ok"]:
            print(
                f"{model['id']}: text={model['text_elements']} "
                f"always_active={model['text_always_active_elements']} "
                f"ffn={model['text_ffn_elements']} mtp={model['mtp_elements']}"
            )
    return 0


# ---------------------------------------------------------------------------


def write_receipt(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe")
    probe.add_argument("--out", required=True)
    probe.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/hf-cache"))
    probe.add_argument("--sleep", type=float, default=0.5)
    probe.add_argument("--small-files", choices=["hash", "none"], default="hash")
    probe.add_argument("--max-small-file-bytes", type=int, default=32 * 1024 * 1024)
    probe.add_argument("--max-response-bytes", type=int, default=64 * 1024 * 1024)
    probe.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    probe.add_argument("--max-shards", type=int, default=32)
    probe.set_defaults(func=cmd_probe)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--id", required=True)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/hf-cache"))
    fetch.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    fetch.add_argument("--headroom-bytes", type=int, default=2 * 1024**3)
    fetch.set_defaults(func=cmd_fetch)

    disk = sub.add_parser("disk")
    disk.add_argument("--probe", required=True)
    disk.add_argument("--out", required=True)
    disk.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/hf-cache"))
    disk.add_argument("--repo-headroom-bytes", type=int, default=20 * 1024**3)
    disk.add_argument("--cache-headroom-bytes", type=int, default=2 * 1024**3)
    disk.set_defaults(func=cmd_disk)

    ceiling = sub.add_parser("ceiling")
    ceiling.add_argument("--probe", required=True)
    ceiling.add_argument("--out", required=True)
    ceiling.add_argument("--ceiling", type=int, default=1_000_000_000)
    ceiling.set_defaults(func=cmd_ceiling)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProbeError as exc:
        print(f"s32-frontier-models: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
