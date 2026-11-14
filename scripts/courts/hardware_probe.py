#!/usr/bin/env python3
"""Collect host-visible hardware facts from inside a court container.

Invoked by scripts/probe-hardware.sh as:

    scripts/court python --host-root-ro -- python3 scripts/courts/hardware_probe.py

The container has the host root mounted read-only at /hostroot, so /proc, /sys
and the real mount table are the host's, not a container-local imitation. The
container's own cgroup limits are recorded too, which is independent evidence
that the court's mem/cpu/pids limits were applied.

Prints one JSON object to stdout. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys

HOSTROOT = os.environ.get("SELVEDGE_HOSTROOT", "/hostroot")


def host_path(path: str) -> str:
    return os.path.join(HOSTROOT, path.lstrip("/"))


def read_text(path: str) -> str:
    try:
        with open(host_path(path), "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def parse_key_values(text: str, separator: str = ":") -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(separator)
        if sep:
            values[key.strip()] = value.strip()
    return values


def unescape_mount_field(field: str) -> str:
    return (
        field.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def probe_cpu() -> dict:
    text = read_text("/proc/cpuinfo")
    processors = [block for block in text.split("\n\n") if block.strip()]
    first = parse_key_values(processors[0]) if processors else {}
    atom_pairs = set()
    physical_ids = set()
    for block in processors:
        fields = parse_key_values(block)
        physical_ids.add(fields.get("physical id", "0"))
        if "core id" in fields:
            atom_pairs.add((fields.get("physical id", "0"), fields["core id"]))
    flags = first.get("flags", "").split()
    return {
        "model": first.get("model name", "unknown").strip(),
        "vendor": first.get("vendor_id", "unknown").strip(),
        "logical_cores": len(processors),
        "physical_cores": len(atom_pairs) if atom_pairs else len(processors),
        "sockets": len(physical_ids) if physical_ids else 1,
        "flags": sorted(flags),
        "flag_count": len(flags),
        "microcode": first.get("microcode", "unknown"),
        "cache_kb": {
            "l2": first.get("cache size", "unknown"),
        },
        "source": "container read of /hostroot/proc/cpuinfo",
    }


def probe_memory() -> dict:
    meminfo = parse_key_values(read_text("/proc/meminfo"))
    swaps: list[dict] = []
    for line in read_text("/proc/swaps").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            swaps.append(
                {
                    "filename": parts[0],
                    "type": parts[1],
                    "size_kib": int(parts[2]) if parts[2].isdigit() else None,
                    "used_kib": int(parts[3]) if parts[3].isdigit() else None,
                }
            )

    def kib(name: str) -> int | None:
        raw = meminfo.get(name, "")
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None

    total = kib("MemTotal")
    available = kib("MemAvailable")
    return {
        "total_kib": total,
        "total_bytes": total * 1024 if total is not None else None,
        "total_gib": round(total / 1024 / 1024, 2) if total is not None else None,
        "available_kib": available,
        "available_gib": round(available / 1024 / 1024, 2) if available is not None else None,
        "free_kib": kib("MemFree"),
        "swap_total_kib": kib("SwapTotal"),
        "swap_total_gib": round(kib("SwapTotal") / 1024 / 1024, 2) if kib("SwapTotal") else 0.0,
        "swap_free_kib": kib("SwapFree"),
        "swap_devices": swaps,
        "source": "container read of /hostroot/proc/meminfo and /proc/swaps",
    }


def probe_storage() -> dict:
    devices: list[dict] = []
    block_dir = host_path("/sys/block")
    try:
        names = sorted(os.listdir(block_dir))
    except OSError:
        names = []
    for name in names:
        if name.startswith(("loop", "ram", "zram", "dm-", "md")):
            continue
        base = os.path.join(block_dir, name)
        try:
            with open(os.path.join(base, "size"), "r", encoding="utf-8") as handle:
                sectors = int(handle.read().strip())
        except (OSError, ValueError):
            continue
        def read_attr(relative: str) -> str:
            try:
                with open(os.path.join(base, relative), "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read().strip()
            except OSError:
                return ""

        rotational = read_attr("queue/rotational")
        devices.append(
            {
                "name": name,
                "size_bytes": sectors * 512,
                "size_gib": round(sectors * 512 / 1024**3, 2),
                "model": read_attr("device/model"),
                "vendor": read_attr("device/vendor"),
                "rotational": rotational == "1",
                "scheduler": read_attr("queue/scheduler"),
            }
        )

    mounts: list[dict] = []
    seen = set()
    for line in read_text("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        source = unescape_mount_field(parts[0])
        target = unescape_mount_field(parts[1])
        fstype = parts[2]
        if not source.startswith("/dev/"):
            continue
        # Inside the court the host's mount table appears under /hostroot. The
        # container's own bind mounts appear unprefixed (e.g. /work, /cargo) and
        # the host's docker overlay paths are court plumbing, not storage facts.
        if not target.startswith(HOSTROOT + "/"):
            continue
        host_target = target[len(HOSTROOT) :]
        if "/docker/overlay2/" in host_target:
            continue
        if host_target in seen:
            continue
        seen.add(host_target)
        entry = {"source": source, "target": host_target, "fstype": fstype}
        try:
            stats = os.statvfs(host_path(host_target))
            total = stats.f_blocks * stats.f_frsize
            free = stats.f_bavail * stats.f_frsize
            entry["total_bytes"] = total
            entry["free_bytes"] = free
            entry["total_gib"] = round(total / 1024**3, 2)
            entry["free_gib"] = round(free / 1024**3, 2)
        except OSError:
            pass
        mounts.append(entry)
    mounts.sort(key=lambda item: item["target"])

    return {
        "devices": devices,
        "mounts": mounts,
        "source": "container read of /hostroot/sys/block, /proc/mounts and statvfs",
    }


def probe_kernel() -> dict:
    osrelease = read_text("/proc/sys/kernel/osrelease").strip()
    version = read_text("/proc/version").strip()
    return {
        "release": osrelease,
        "version": version,
        "arch": os.uname().machine,
        "source": "container read of /proc/sys/kernel/osrelease",
    }


def probe_container_limits() -> dict:
    def read_cgroup(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    return {
        "memory_max": read_cgroup("/sys/fs/cgroup/memory.max"),
        "memory_swap_max": read_cgroup("/sys/fs/cgroup/memory.swap.max"),
        "cpu_max": read_cgroup("/sys/fs/cgroup/cpu.max"),
        "pids_max": read_cgroup("/sys/fs/cgroup/pids.max"),
        "source": "container read of its own cgroup v2 control files",
    }


def main() -> int:
    payload = {
        "cpu": probe_cpu(),
        "memory": probe_memory(),
        "storage": probe_storage(),
        "kernel": probe_kernel(),
        "container_limits": probe_container_limits(),
        "container": {
            "image": os.environ.get("SELVEDGE_COURT_IMAGE", "unknown"),
            "memory_limit_requested": os.environ.get("SELVEDGE_COURT_MEM", "unknown"),
            "cpus_requested": os.environ.get("SELVEDGE_COURT_CPUS", "unknown"),
            "pids_requested": os.environ.get("SELVEDGE_COURT_PIDS", "unknown"),
        },
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
