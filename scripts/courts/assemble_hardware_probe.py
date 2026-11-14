#!/usr/bin/env python3
"""Assemble the hardware probe receipt from the court runs that produced it.

Every field carries the container run it came from; nothing is copied from a
previous report and nothing is hand-edited. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional(path: str) -> dict:
    if not path or path == "none" or not os.path.exists(path):
        return {}
    try:
        return load_json(path)
    except ValueError:
        return {}


def toolchain_of(receipt: dict) -> dict:
    return receipt.get("toolchain") or {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dirty", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu", default="none")
    parser.add_argument("--gpu-status", default="not attempted")
    parser.add_argument("--gpu-court-receipt", default="")
    parser.add_argument("--host-court-receipt", required=True)
    parser.add_argument("--rust-receipt", required=True)
    parser.add_argument("--python-receipt", required=True)
    parser.add_argument("--docker-client", required=True)
    parser.add_argument("--docker-server", required=True)
    parser.add_argument("--cgroup-version", required=True)
    parser.add_argument("--host-uname", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args(argv)

    host = load_json(args.host)
    gpu = load_optional(args.gpu)
    gpu_receipt = load_optional(args.gpu_court_receipt)
    rust_receipt = load_json(args.rust_receipt)
    python_receipt = load_json(args.python_receipt)

    rust_tc = toolchain_of(rust_receipt)
    python_tc = toolchain_of(python_receipt)

    def rel(path: str) -> str | None:
        if not path or path == "none":
            return None
        return os.path.relpath(path, args.repo)

    payload = {
        "schema": "selvedge/hardware-probe/v1",
        "stamp": args.stamp,
        "revision": args.revision,
        "worktree_dirty": args.dirty == "true",
        "host_uname": args.host_uname,
        "kernel": host.get("kernel", {}),
        "cpu": host.get("cpu", {}),
        "memory": host.get("memory", {}),
        "storage": host.get("storage", {}),
        "container_limits_observed": host.get("container_limits", {}),
        "docker": {
            "client_version": args.docker_client,
            "server_version": args.docker_server,
            "cgroup_version": args.cgroup_version,
            "source": "docker CLI/daemon queried by scripts/probe-hardware.sh on the host",
        },
        "toolchains": {
            "rustc": rust_tc.get("rustc"),
            "cargo": rust_tc.get("cargo"),
            "python3": python_tc.get("python3"),
            "cc": rust_tc.get("cc"),
            "git": rust_tc.get("git"),
            "rust_court_image": rust_receipt.get("image_ref"),
            "python_court_image": python_receipt.get("image_ref"),
            "source": "scripts/courts/probe-toolchain.sh inside the courts named above",
        },
        "gpu": {
            "court_status": args.gpu_status,
            "container_access": bool(gpu.get("available")),
            "access_mode": (gpu_receipt.get("gpu") or {}).get("mode"),
            "gpus_all_error": (gpu_receipt.get("gpu") or {}).get("detail"),
            "devices": gpu.get("devices", []),
            "cuda_version": gpu.get("cuda_version"),
            "result": gpu or None,
            "source": "scripts/courts/gpu_probe.py inside a --gpu court container",
        },
        "evidence": {
            "host_probe_court_receipt": rel(args.host_court_receipt),
            "gpu_probe_court_receipt": rel(args.gpu_court_receipt),
            "rust_toolchain_court_receipt": rel(args.rust_receipt),
            "python_toolchain_court_receipt": rel(args.python_receipt),
        },
    }

    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.out)
    print(f"probe-hardware: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
