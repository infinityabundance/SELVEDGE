#!/usr/bin/env python3
"""S68: fetch pinned external frontier checkpoints with hashes and licences.

This is the network step of the S68 position check. It is explicit, idempotent
and fails closed:

  * the pinned revision is the SHA named on the command line; if the Hub's
    current `main` differs, the receipt records both and still fetches the pin;
  * a gated repository is recorded as not fetched, together with the HTTP status
    its file endpoint returns (401 for a manual-gate repository) — no attempt is
    made to bypass the gate;
  * every downloaded file is streamed to `<path>.partial`, SHA-256 hashed while
    streaming, and compared with the Hub's LFS object id before the rename;
  * an already-present file whose hash matches is not re-downloaded.

The layout matches `oracle/fetch_models.py`:
`<cache>/models/<org>/<name>/<revision>/<file>`.

Usage (network, oracle court):
    SELVEDGE_MEM=4g scripts/court --net oracle -- \
      python3 scripts/s68_fetch_frontier.py \
        --model amd/ReasonLite-0.6B=3aad2248... \
        --model facebook/MobileLLM-R1.5-950M=d4af70ac... \
        --out results/models/s68-frontier-fetch.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

SCHEMA = "selvedge/s68-frontier-fetch/v1"
USER_AGENT = "selvedge-research/0.1 (offline research program)"
CHUNK = 4 * 1024 * 1024


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def revision() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def open_url(url: str, headers: dict | None = None):
    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=merged),
                                  timeout=120)


def http_json(url: str) -> tuple[int, object | None, str]:
    try:
        with open_url(url) as response:
            return response.status, json.loads(response.read()), ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, exc.read(2048).decode("utf-8", "replace")
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def parse_model(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise SystemExit(f"s68-fetch: expected repo=revision, got {text!r}")
    repo, pin = text.rsplit("=", 1)
    if not repo or not pin:
        raise SystemExit(f"s68-fetch: empty repo or revision in {text!r}")
    return repo, pin


def fetch_one(repo: str, pin: str, cache_dir: str) -> dict:
    status, meta, detail = http_json(
        f"https://huggingface.co/api/models/{repo}")
    if status != 200 or not isinstance(meta, dict):
        raise SystemExit(
            f"s68-fetch: cannot resolve {repo}: HTTP {status} {detail[:200]}")
    hub_main = meta.get("sha")
    record = {
        "id": repo,
        "pinned_revision": pin,
        "hub_main_sha": hub_main,
        "revision_is_hub_main": hub_main == pin,
        "gated": meta.get("gated", False),
        "private": meta.get("private", False),
        "license": ((meta.get("cardData") or {}).get("license")
                    or next((tag.split(":", 1)[1] for tag in
                             meta.get("tags") or []
                             if tag.startswith("license:")), None)),
        "license_name": (meta.get("cardData") or {}).get("license_name"),
        "used_storage_bytes": meta.get("usedStorage"),
        "fetched": False,
    }
    file_url = (f"https://huggingface.co/{repo}/resolve/{pin}/config.json")
    if record["gated"]:
        try:
            with open_url(file_url) as response:
                record["gate_probe"] = {"url": file_url,
                                        "status": response.status}
        except urllib.error.HTTPError as exc:
            record["gate_probe"] = {"url": file_url, "status": exc.code,
                                    "detail": exc.read(512).decode(
                                        "utf-8", "replace")}
        except Exception as exc:
            record["gate_probe"] = {"url": file_url, "status": 0,
                                    "detail": f"{type(exc).__name__}: {exc}"}
        record["not_fetched_reason"] = (
            "the repository is gated; the file endpoint refuses anonymous "
            "access and the program does not hold acceptance credentials")
        return record

    tree_status, tree, tree_detail = http_json(
        f"https://huggingface.co/api/models/{repo}/tree/{pin}?recursive=true")
    if tree_status != 200 or not isinstance(tree, list):
        raise SystemExit(
            f"s68-fetch: cannot list {repo}@{pin}: HTTP {tree_status} "
            f"{tree_detail[:200]}")
    files = [entry for entry in tree if entry.get("type") == "file"]
    wanted = [entry for entry in files
              if (entry["path"].endswith(".safetensors")
                  or entry["path"].endswith(".json")
                  or entry["path"].startswith("LICENSE")
                  or entry["path"] == "README.md")]
    directory = os.path.join(cache_dir, "models", repo, pin)
    os.makedirs(directory, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    needed = sum(int((entry.get("lfs") or {}).get("size", entry.get("size"))
                     or 0) for entry in wanted)
    if needed + (2 << 30) > free:
        raise SystemExit(
            f"s68-fetch: refusing to fetch {needed} bytes plus 2 GiB headroom "
            f"with {free} bytes free on {cache_dir}")
    record["cache_dir"] = directory
    record["cache_free_before"] = free
    record["files"] = []
    for entry in wanted:
        path = entry["path"]
        target = os.path.join(directory, path)
        declared = int((entry.get("lfs") or {}).get(
            "size", entry.get("size")) or 0)
        oid = (entry.get("lfs") or {}).get("oid")
        if os.path.exists(target):
            digest = sha256_file(target)
            if digest == oid or (oid is None and
                                 os.path.getsize(target) == declared):
                record["files"].append({
                    "path": path, "bytes": declared, "sha256": digest,
                    "lfs_oid": oid, "cached": True,
                    "sha256_matches_lfs_oid": (oid == digest) if oid else None,
                })
                continue
        url = f"https://huggingface.co/{repo}/resolve/{pin}/{path}"
        partial = target + ".partial"
        digest = hashlib.sha256()
        written = 0
        with open_url(url) as response, open(partial, "wb") as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                digest.update(block)
                handle.write(block)
                written += len(block)
                if written > declared + 1:
                    raise SystemExit(
                        f"s68-fetch: {path} exceeded its declared size")
        value = digest.hexdigest()
        if oid is not None and value != oid:
            raise SystemExit(
                f"s68-fetch: {path} sha256 {value} != hub lfs oid {oid}")
        if written != declared:
            raise SystemExit(
                f"s68-fetch: {path} wrote {written} bytes, declared {declared}")
        os.replace(partial, target)
        record["files"].append({
            "path": path, "bytes": written, "sha256": value, "lfs_oid": oid,
            "cached": False,
            "sha256_matches_lfs_oid": (oid == value) if oid else None,
        })
        print(f"s68-fetch: {repo}:{path} {written} bytes sha={value[:16]}",
              flush=True)
    record["fetched"] = True
    record["cache_free_after"] = shutil.disk_usage(cache_dir).free
    license_files = [entry for entry in record["files"]
                     if entry["path"].startswith("LICENSE")]
    record["license_files"] = license_files
    weights = [entry for entry in record["files"]
               if entry["path"].endswith(".safetensors")]
    record["weight_bytes"] = sum(entry["bytes"] for entry in weights)
    record["weight_files"] = weights
    return record


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="repo=revision")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME",
                                                              "/hf-cache"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    models = []
    for text in args.model:
        repo, pin = parse_model(text)
        print(f"s68-fetch: {repo}@{pin}", flush=True)
        models.append(fetch_one(repo, pin, args.cache_dir))
    write_json(args.out, {
        "schema": SCHEMA,
        "kind": "frontier-fetch",
        "created_utc": utcnow(),
        "stamp": stamp(),
        "revision": revision(),
        "network": True,
        "cache_dir": args.cache_dir,
        "models": models,
    })
    print(f"s68-fetch: -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
