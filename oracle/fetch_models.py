#!/usr/bin/env python3
"""Fetch the pinned Qwen3 checkpoints into the persistent cache, or verify them.

The fetch step is the only part of the real-model path that uses the network.
It is explicit (`scripts/fetch-models.sh` passes `--net` to the oracle court),
idempotent, and fails closed:

  * every file is downloaded to `<target>.partial`, hashed while streaming, and
    only then renamed into place;
  * when the Hugging Face API reports an LFS object id, it is treated as the
    expected SHA-256 and a mismatch is fatal;
  * an already-present file whose hash matches is never re-downloaded and never
    overwritten;
  * an already-present file whose hash does not match is fatal and is not
    replaced — the operator must remove it deliberately;
  * the resolved commit revision is compared with the pinned revision in
    `models/model-spec.json`.

`--verify-only` performs no network access and checks the cache against
`models/MODELS.lock.json`. That is how offline courts prove they are reading a
verified checkpoint.

The lock file records, for every downloaded file, its repository-relative path,
byte size and SHA-256, plus the license file hash and the tokenizer files. No
weights are written into the repository. The receipt under `results/models/`
carries the court stamp and git revision and is what `docs/MODELS.md` numbers
are generated from.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

SCHEMA_LOCK = "selvedge/models-lock/v1"
SCHEMA_RECEIPT = "selvedge/model-fetch/v1"
USER_AGENT = "selvedge-research/0.1 (offline research program)"
CHUNK = 4 * 1024 * 1024


class FetchError(Exception):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".models-", delete=False
    )
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


def http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def tree_index(model_id: str, revision: str) -> dict[str, dict]:
    """Return {path: {size, lfs_oid}} for a pinned revision."""
    data = http_json(
        f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
    )
    index: dict[str, dict] = {}
    for entry in data:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        index[entry["path"]] = {
            "size": entry.get("size"),
            "lfs_oid": lfs.get("oid"),
        }
    return index


def download(url: str, target: str) -> tuple[int, str, str]:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    if os.path.exists(partial):
        os.unlink(partial)
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    return total, digest.hexdigest(), partial


def free_bytes(path: str) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def verify_file(path: str, expected_sha: str, expected_size: int | None) -> None:
    if not os.path.exists(path):
        raise FetchError(f"missing verified cache file {path}")
    size = os.path.getsize(path)
    if expected_size is not None and size != expected_size:
        raise FetchError(
            f"{path} has {size} bytes but the lock records {expected_size}"
        )
    actual = sha256_file(path)
    if actual != expected_sha:
        raise FetchError(
            f"{path} sha256 {actual} does not match the locked {expected_sha}"
        )


def fetch_model(
    model: dict,
    cache_root: str,
    previous: dict | None,
    network: bool,
) -> dict:
    model_id = model["id"]
    revision = model["revision"]
    slug = model_id.replace("/", "__")
    target_dir = os.path.join(cache_root, "models", model_id, revision)
    previous_files = {}
    if previous:
        if previous.get("revision") != revision:
            raise FetchError(
                f"{model_id}: lock revision {previous.get('revision')} != pinned {revision}; "
                "refusing to mix revisions"
            )
        previous_files = {item["path"]: item for item in previous.get("files", [])}

    index: dict[str, dict] = {}
    if network:
        index = tree_index(model_id, revision)
        for name in model["files"]:
            if name not in index:
                raise FetchError(
                    f"{model_id}@{revision}: spec file {name!r} is not in the resolved tree"
                )

    files = []
    for name in model["files"]:
        target = os.path.join(target_dir, name)
        expected = previous_files.get(name)
        if os.path.exists(target):
            if expected is None:
                raise FetchError(
                    f"{target} exists but is not in the lock; refusing to overwrite an "
                    "unverified file (remove it deliberately to re-fetch)"
                )
            verify_file(target, expected["sha256"], expected.get("bytes"))
            files.append(expected)
            continue
        if not network:
            raise FetchError(
                f"{target} is absent and --verify-only was requested; run "
                "scripts/fetch-models.sh once with network access"
            )
        meta = index[name]
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/{name}"
        size, sha, partial = download(url, target)
        lfs_oid = meta.get("lfs_oid")
        if lfs_oid and sha != lfs_oid:
            os.unlink(partial)
            raise FetchError(
                f"{model_id}/{name}: downloaded sha256 {sha} != HF LFS oid {lfs_oid}"
            )
        if meta.get("size") is not None and size != meta["size"]:
            os.unlink(partial)
            raise FetchError(
                f"{model_id}/{name}: downloaded {size} bytes, tree reports {meta['size']}"
            )
        os.replace(partial, target)
        files.append(
            {
                "path": name,
                "bytes": size,
                "sha256": sha,
                "verified_source": "hf_lfs_oid" if lfs_oid else "computed_sha256",
                "hf_lfs_oid": lfs_oid,
            }
        )

    license_entry = next((item for item in files if item["path"] == "LICENSE"), None)
    if license_entry is None:
        raise FetchError(f"{model_id}: the spec must include the LICENSE file")

    tokenizer_files = [
        item
        for item in files
        if item["path"]
        in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    ]
    weight_files = [
        item for item in files if item["path"].endswith(".safetensors")
    ]
    return {
        "id": model_id,
        "revision": revision,
        "license": model["license"],
        "role": model.get("role"),
        "cache_relative_dir": os.path.relpath(target_dir, cache_root),
        "license_file": "LICENSE",
        "license_sha256": license_entry["sha256"],
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(item["bytes"] for item in weight_files),
        "files": files,
        "tokenizer": {
            "files": [
                {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
                for item in tokenizer_files
            ]
        },
        "slug": slug,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="models/model-spec.json")
    parser.add_argument("--lock", default="models/MODELS.lock.json")
    parser.add_argument("--cache-root", default=os.environ.get("SELVEDGE_MODEL_CACHE", "/hf-cache"))
    parser.add_argument("--receipt-dir", default="results/models")
    parser.add_argument("--verify-only", action="store_true", help="no network; verify the cache")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    network = not args.verify_only
    spec = load_json(args.spec)
    if spec.get("schema") != "selvedge/model-spec/v1":
        raise FetchError(f"{args.spec}: unexpected schema {spec.get('schema')!r}")

    previous_lock = None
    if os.path.exists(args.lock):
        previous_lock = load_json(args.lock)
        if previous_lock.get("schema") != SCHEMA_LOCK:
            raise FetchError(f"{args.lock}: unexpected schema")
    elif args.verify_only:
        raise FetchError(f"{args.lock} is missing; a verify-only run needs the lock")

    previous_by_id = {}
    if previous_lock:
        previous_by_id = {item["id"]: item for item in previous_lock.get("models", [])}

    os.makedirs(os.path.join(args.cache_root, "models"), exist_ok=True)
    before_free = free_bytes(args.cache_root)

    models = []
    for model in spec["models"]:
        models.append(
            fetch_model(
                model,
                args.cache_root,
                previous_by_id.get(model["id"]),
                network,
            )
        )

    lock = {
        "schema": SCHEMA_LOCK,
        "updated_utc": utc_now(),
        "cache_relative_root": "models",
        "models": models,
        "reference_models_not_downloaded": spec.get(
            "reference_models_not_downloaded", []
        ),
    }
    if previous_lock and previous_lock.get("models") == models:
        lock["updated_utc"] = previous_lock.get("updated_utc", lock["updated_utc"])
    write_json_atomic(args.lock, lock)

    after_free = free_bytes(args.cache_root)
    stamp = stamp_now()
    revision = os.environ.get("SELVEDGE_REVISION", "unknown")
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "stamp": stamp,
        "revision": revision,
        "started_utc": utc_now(),
        "network": network,
        "cache_root_note": "host ~/.cache/selvedge/hf mounted at /hf-cache",
        "cache_filesystem": {
            "free_bytes_before": before_free,
            "free_bytes_after": after_free,
        },
        "models": models,
        "reference_models_not_downloaded": spec.get(
            "reference_models_not_downloaded", []
        ),
        "lock": args.lock,
        "lock_sha256": sha256_file(args.lock),
    }
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, f"fetch-{stamp}.json")
    write_json_atomic(receipt_path, receipt)

    summary = {
        "ok": True,
        "network": network,
        "receipt": receipt_path,
        "lock": args.lock,
        "models": [
            {
                "id": item["id"],
                "revision": item["revision"],
                "license": item["license"],
                "license_sha256": item["license_sha256"],
                "weight_bytes": item["weight_bytes"],
                "files": len(item["files"]),
                "cache_relative_dir": item["cache_relative_dir"],
            }
            for item in models
        ],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"fetch-models: network={network} lock={args.lock}")
        for item in models:
            print(
                f"  {item['id']}@{item['revision']} license={item['license']} "
                f"license_sha256={item['license_sha256']} files={len(item['files'])} "
                f"weight_bytes={item['weight_bytes']} dir={item['cache_relative_dir']}"
            )
        print(f"fetch-models: receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except FetchError as exc:
        print(f"fetch-models: FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"fetch-models: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
