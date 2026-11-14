#!/usr/bin/env python3
"""Probe the NVIDIA GPU from inside a GPU court container.

Invoked by scripts/probe-hardware.sh through scripts/docker-court.sh --gpu, so
the answer is "can a court container actually use the GPU", not "does the host
have a driver". Prints one JSON object. Standard library only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

QUERY_FIELDS = "name,memory.total,driver_version,compute_cap,uuid,pci.bus_id"


def run(argv: list[str]) -> dict:
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "ok": False, "error": str(exc)}
    return {
        "argv": argv,
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip()[:2000],
    }


def main() -> int:
    smi = run(["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader"])
    devices: list[dict] = []
    if smi.get("ok"):
        for line in smi["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 6:
                devices.append(
                    {
                        "name": parts[0],
                        "vram_total": parts[1],
                        "driver_version": parts[2],
                        "compute_capability": parts[3],
                        "uuid": parts[4],
                        "pci_bus_id": parts[5],
                    }
                )

    cuda_driver = run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
    payload = {
        "available": bool(devices),
        "devices": devices,
        "query": smi,
        "cuda_version": cuda_driver.get("stdout") if cuda_driver.get("ok") else None,
        "device_nodes": sorted(
            name for name in os.listdir("/dev") if name.startswith("nvidia")
        ),
        "libs": sorted(
            name
            for name in os.listdir("/usr/lib/x86_64-linux-gnu")
            if name.startswith(("libcuda.so", "libnvidia-"))
        )
        if os.path.isdir("/usr/lib/x86_64-linux-gnu")
        else [],
        "source": "nvidia-smi executed inside the court container",
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
