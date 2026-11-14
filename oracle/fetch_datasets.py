#!/usr/bin/env python3
"""Fetch the pinned evaluation datasets, or verify the local cache.

This is the only part of the experiment that downloads data. It runs inside the
oracle court with `--net` through `scripts/fetch-datasets.sh`, is idempotent, and
fails closed:

  * only the files named in `data/dataset-spec.json` are downloaded;
  * the URL uses the full pinned commit revision, and the resolved tree is
    checked against that revision;
  * every file is streamed to `<target>.partial`, hashed while streaming, and
    renamed into place only after its size (and LFS object id, when the API
    reports one) matches;
  * an already-present file whose hash matches is never re-downloaded;
  * an already-present file whose hash does not match the lock is fatal and is
    not replaced;
  * `--verify-only` performs no network access and checks the cache against
    `data/datasets.lock.json`.

The cache is `results/artifacts/datasets/` (gitignored). No corpus text is
committed. The lock records each file's bytes and SHA-256; the receipt under
`results/datasets/` carries the court stamp and git revision.
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
import urllib.request

SCHEMA_SPEC = "selvedge/dataset-spec/v1"
SCHEMA_LOCK = "selvedge/datasets-lock/v1"
SCHEMA_RECEIPT = "selvedge/dataset-fetch/v1"
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
        "w", encoding="utf-8", dir=directory, prefix=".datasets-", delete=False
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


def tree_index(repo_id: str, revision: str) -> dict[str, dict]:
    data = http_json(
        f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=true"
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


def verify_file(path: str, expected_sha: str, expected_size: int) -> None:
    if not os.path.exists(path):
        raise FetchError(f"missing verified cache file {path}")
    size = os.path.getsize(path)
    if size != expected_size:
        raise FetchError(f"{path} has {size} bytes but the lock records {expected_size}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise FetchError(f"{path} sha256 {actual} does not match the locked {expected_sha}")


def fetch_dataset(dataset: dict, cache_root: str, previous: dict | None, network: bool) -> dict:
    dataset_id = dataset["id"]
    repo_id = dataset["repo_id"]
    revision = dataset["revision"]
    name = dataset["file"]
    target = os.path.join(cache_root, dataset_id, os.path.basename(name))

    if previous is not None and previous.get("id") != dataset_id:
        raise FetchError(f"{dataset_id}: lock id mismatch")

    if os.path.exists(target):
        if previous is None or "sha256" not in previous:
            raise FetchError(
                f"{target} exists but is not in the lock; refusing to overwrite an "
                "unverified file (remove it deliberately to re-fetch)"
            )
        verify_file(target, previous["sha256"], previous["bytes"])
        entry = dict(previous)
        entry["cache_relative_path"] = os.path.relpath(target, cache_root)
        return entry

    if not network:
        raise FetchError(
            f"{target} is absent and --verify-only was requested; run "
            "scripts/fetch-datasets.sh once with network access"
        )

    index = tree_index(repo_id, revision)
    if name not in index:
        raise FetchError(f"{repo_id}@{revision}: {name!r} is not in the resolved tree")
    meta = index[name]
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{name}"
    size, sha, partial = download(url, target)
    lfs_oid = meta.get("lfs_oid")
    if lfs_oid and sha != lfs_oid:
        os.unlink(partial)
        raise FetchError(f"{repo_id}/{name}: downloaded sha256 {sha} != HF LFS oid {lfs_oid}")
    if meta.get("size") is not None and size != meta["size"]:
        os.unlink(partial)
        raise FetchError(
            f"{repo_id}/{name}: downloaded {size} bytes, tree reports {meta['size']}"
        )
    os.replace(partial, target)
    return {
        "id": dataset_id,
        "repo_id": repo_id,
        "revision": revision,
        "file": name,
        "bytes": size,
        "sha256": sha,
        "verified_source": "hf_lfs_oid" if lfs_oid else "computed_sha256",
        "hf_lfs_oid": lfs_oid,
        "license": dataset["license"],
        "license_url": dataset["license_url"],
        "split": dataset["split"],
        "text_field": dataset["text_field"],
        "role": dataset["role"],
        "cache_relative_path": os.path.relpath(target, cache_root),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="data/dataset-spec.json")
    parser.add_argument("--lock", default="data/datasets.lock.json")
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("SELVEDGE_DATASET_CACHE", "results/artifacts/datasets"),
    )
    parser.add_argument("--receipt-dir", default="results/datasets")
    parser.add_argument("--verify-only", action="store_true", help="no network; verify the cache")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    network = not args.verify_only
    spec = load_json(args.spec)
    if spec.get("schema") != SCHEMA_SPEC:
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
        previous_by_id = {item["id"]: item for item in previous_lock.get("datasets", [])}

    os.makedirs(args.cache_root, exist_ok=True)
    before_free = shutil.disk_usage(args.cache_root).free

    datasets = []
    for dataset in spec["datasets"]:
        datasets.append(
            fetch_dataset(
                dataset,
                args.cache_root,
                previous_by_id.get(dataset["id"]),
                network,
            )
        )

    lock = {
        "schema": SCHEMA_LOCK,
        "updated_utc": utc_now(),
        "cache_relative_root": args.cache_root,
        "datasets": datasets,
    }
    if previous_lock and previous_lock.get("datasets") == datasets:
        lock["updated_utc"] = previous_lock.get("updated_utc", lock["updated_utc"])
    write_json_atomic(args.lock, lock)

    after_free = shutil.disk_usage(args.cache_root).free
    stamp = stamp_now()
    revision = os.environ.get("SELVEDGE_REVISION", "unknown")
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "stamp": stamp,
        "revision": revision,
        "started_utc": utc_now(),
        "network": network,
        "spec": args.spec,
        "spec_sha256": sha256_file(args.spec),
        "lock": args.lock,
        "lock_sha256": sha256_file(args.lock),
        "cache_relative_root": args.cache_root,
        "disk_free_bytes_before": before_free,
        "disk_free_bytes_after": after_free,
        "datasets": datasets,
        "notes": [
            "only the files named in data/dataset-spec.json are downloaded",
            "every URL uses the full pinned commit revision",
            "verify-only performs no network access",
        ],
    }
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, f"fetch-{stamp}.json")
    write_json_atomic(receipt_path, receipt)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        for item in datasets:
            print(
                f"datasets: {item['id']} {item['repo_id']}@{item['revision']} "
                f"{item['bytes']} bytes sha256={item['sha256']}"
            )
        print(f"datasets: lock {args.lock}")
        print(f"datasets: receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except FetchError as exc:
        print(f"fetch-datasets: {exc}", file=sys.stderr)
        sys.exit(1)
