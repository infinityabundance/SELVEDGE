#!/usr/bin/env python3
"""Allocate memory until the cgroup OOM killer stops us.

Used by the court self-verification to prove that SELVEDGE_MEM is a hard bound
rather than a suggestion. Each step allocates and *touches* one chunk, so the
pages are resident and counted by the cgroup memory controller.

Exit behaviour:
  * killed by the cgroup OOM killer (expected under a working cap): the process
    dies, docker reports exit 137 and State.OOMKilled=true;
  * reaching the target allocation: exit 0 and print the peak — which means the
    limit was NOT enforced and the court run must be treated as a failure.

Standard library only. Never writes files.
"""

from __future__ import annotations

import os
import sys
import time

CHUNK = 32 * 1024 * 1024  # 32 MiB
DEFAULT_TARGET = 4 * 1024 * 1024 * 1024  # 4 GiB


def main() -> int:
    target = DEFAULT_TARGET
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    limit = int(os.environ.get("SELVEDGE_COURT_MEM_BYTES", "0") or 0)
    chunks: list[bytearray] = []
    allocated = 0
    print(
        f"oom_probe: target={target} bytes, cgroup limit={limit} bytes, pid={os.getpid()}",
        flush=True,
    )
    while allocated < target:
        try:
            block = bytearray(CHUNK)
            for offset in range(0, CHUNK, 4096):
                block[offset] = 1
        except MemoryError:
            print(f"oom_probe: MemoryError at {allocated} bytes", flush=True)
            return 0
        chunks.append(block)
        allocated += CHUNK
        print(f"oom_probe: allocated {allocated} bytes", flush=True)
        time.sleep(0.01)
    print(f"oom_probe: SURVIVED to {allocated} bytes", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
