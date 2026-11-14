#!/usr/bin/env python3
"""Write a SELVEDGE court receipt and verify the container limits were enforced.

This helper is deliberately the only place that serializes receipt JSON: the
court script passes plain scalar arguments, so no shell quoting can corrupt a
receipt and no log content is ever interpolated into JSON by hand.

Exit codes:
  0  receipt written, every limit verified as applied
  2  receipt written, but the runtime did not apply the requested limits
  3  receipt could not be written or the inspect data was unusable
Standard library only. Reads/writes files, never the network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

SCHEMA = "selvedge/court-receipt/v1"
SIZE_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


class ReceiptError(Exception):
    pass


def parse_size_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9]+)([kmg])", text.strip().lower())
    if not match:
        raise ReceiptError(f"cannot parse size {text!r}")
    return int(match.group(1)) * SIZE_UNITS[match.group(2)]


def parse_tmpfs_size(options: str) -> int | None:
    for part in options.split(","):
        part = part.strip()
        if part.startswith("size="):
            raw = part[len("size=") :].strip().lower()
            match = re.fullmatch(r"([0-9]+)([kmg]?)", raw)
            if not match:
                return None
            return int(match.group(1)) * SIZE_UNITS[match.group(2)]
    return None


def read_toolchain(path: str) -> tuple[dict[str, str], str]:
    if not path or path == "-" or not os.path.exists(path):
        return {}, "probe-missing"
    values: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                key, sep, value = line.partition("\t")
                if not sep:
                    key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip()
                if key:
                    values[key] = value.strip()
    except OSError as exc:
        return {}, f"probe-unreadable: {exc}"
    return values, "probe-ok" if values else "probe-empty"


def load_inspect(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ReceiptError(f"cannot read docker inspect data {path!r}: {exc}")
    if isinstance(data, list):
        if not data:
            raise ReceiptError("docker inspect returned no container")
        data = data[0]
    if not isinstance(data, dict):
        raise ReceiptError("docker inspect returned unexpected shape")
    return data


def build_receipt(args: argparse.Namespace) -> dict:
    inspect = load_inspect(args.inspect)
    host_config = inspect.get("HostConfig") or {}
    state = inspect.get("State") or {}
    toolchain, toolchain_source = read_toolchain(args.toolchain)

    mem_bytes = parse_size_bytes(args.memory)
    tmp_bytes = parse_size_bytes(args.tmpfs_size)
    nano_cpus = int(round(float(args.cpus) * 1_000_000_000))

    observed: dict[str, object] = {}
    mismatches: list[str] = []

    def compare(name: str, actual: object, expected: object) -> None:
        observed[name] = {"expected": expected, "actual": actual}
        if actual != expected:
            mismatches.append(f"{name}: expected {expected!r}, runtime reports {actual!r}")

    compare("memory_bytes", host_config.get("Memory"), mem_bytes)
    compare("memory_swap_bytes", host_config.get("MemorySwap"), mem_bytes)
    compare("pids_limit", host_config.get("PidsLimit"), args.pids)
    compare("nano_cpus", host_config.get("NanoCpus"), nano_cpus)
    compare("network_mode", host_config.get("NetworkMode"), args.network)
    compare("init", bool(host_config.get("Init")), True)

    security_opt = host_config.get("SecurityOpt") or []
    has_no_new_privs = any(
        str(entry).split(":")[0] == "no-new-privileges" for entry in security_opt
    )
    compare("no_new_privileges", has_no_new_privs, True)

    tmpfs = host_config.get("Tmpfs") or {}
    tmpfs_options = tmpfs.get("/tmp", "")
    compare("tmpfs_bytes", parse_tmpfs_size(str(tmpfs_options)), tmp_bytes)

    ulimits = host_config.get("Ulimits") or []
    nofile = None
    for entry in ulimits:
        if isinstance(entry, dict) and str(entry.get("Name")) == "nofile":
            nofile = {"soft": entry.get("Soft"), "hard": entry.get("Hard")}
            break
    compare("nofile", nofile, {"soft": args.nofile, "hard": args.nofile})

    if args.user:
        compare("user", (inspect.get("Config") or {}).get("User"), args.user)

    read_only = bool(host_config.get("ReadonlyRootfs"))
    observed["readonly_rootfs"] = read_only

    exit_code = int(args.exit_code)
    duration_ms = int(args.duration_ms)
    oom_killed = bool(state.get("OOMKilled"))
    container_exit = state.get("ExitCode")

    receipt = {
        "schema": SCHEMA,
        "court": args.court,
        "stamp": args.stamp,
        "started_utc": args.started_utc,
        "finished_utc": args.finished_utc,
        "revision": args.revision,
        "worktree_dirty": args.worktree_dirty == "true",
        "argv": list(args.argv),
        "container_name": args.container_name,
        "container_id": inspect.get("Id"),
        "image_ref": args.image_ref,
        "image_id": args.image_id,
        "image_digest": args.image_digest,
        "image_digest_source": args.image_digest_source,
        "limits": {
            "memory": args.memory,
            "memory_bytes": mem_bytes,
            "memory_swap": args.memory,
            "memory_swap_bytes": mem_bytes,
            "cpus": args.cpus,
            "nano_cpus": nano_cpus,
            "pids": args.pids,
            "tmpfs": "/tmp",
            "tmpfs_size": args.tmpfs_size,
            "tmpfs_bytes": tmp_bytes,
            "nofile": args.nofile,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "init": True,
        },
        "network_mode": args.network,
        "network_requested": args.network != "none",
        "gpu": {"mode": args.gpu_mode, "detail": args.gpu_detail or None},
        "user": args.user or None,
        "mounts": [m for m in args.mount if m],
        "exit_code": exit_code,
        "container_exit_code": container_exit,
        "oom_killed": oom_killed,
        "duration_ms": duration_ms,
        "host": {
            "uname": args.host_uname,
            "docker_client": args.docker_client,
            "docker_server": args.docker_server,
            "cgroup_version": args.cgroup_version,
        },
        "toolchain": toolchain,
        "toolchain_probe": toolchain_source,
        "enforcement": {
            "ok": not mismatches,
            "observed": observed,
            "mismatches": mismatches,
        },
        "log_path": args.log,
    }
    return receipt


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".receipt-", delete=False
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--court", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--started-utc", required=True)
    parser.add_argument("--finished-utc", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--worktree-dirty", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--inspect", required=True)
    parser.add_argument("--toolchain", default="")
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--image-digest-source", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--pids", type=int, required=True)
    parser.add_argument("--tmpfs-size", required=True)
    parser.add_argument("--nofile", type=int, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--gpu-mode", default="none")
    parser.add_argument("--gpu-detail", default="")
    parser.add_argument("--user", default="")
    parser.add_argument("--mount", action="append", default=[])
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--duration-ms", type=int, required=True)
    parser.add_argument("--host-uname", required=True)
    parser.add_argument("--docker-client", required=True)
    parser.add_argument("--docker-server", required=True)
    parser.add_argument("--cgroup-version", default="unknown")
    parser.add_argument("--log", required=True)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    argv_tail = list(args.argv)
    if argv_tail and argv_tail[0] == "--":
        argv_tail = argv_tail[1:]
    args.argv = argv_tail

    try:
        receipt = build_receipt(args)
        write_json_atomic(args.receipt_out, receipt)
    except ReceiptError as exc:
        print(f"write-receipt: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"write-receipt: cannot write receipt: {exc}", file=sys.stderr)
        return 3

    if receipt["enforcement"]["mismatches"]:
        for mismatch in receipt["enforcement"]["mismatches"]:
            print(f"write-receipt: LIMIT NOT ENFORCED — {mismatch}", file=sys.stderr)
        print(
            f"write-receipt: receipt {args.receipt_out} records the failure; "
            "refusing to treat this run as valid",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
