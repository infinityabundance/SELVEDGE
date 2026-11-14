#!/usr/bin/env python3
"""Record free disk space for the paths a real-model phase writes to.

Runs inside a court and emits a receipt-shaped JSON with the git revision from
the court environment. The numbers are measured with `shutil.disk_usage`, not
copied from a hardware probe of an earlier session.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/disk")
    args = parser.parse_args(argv)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    for path in args.paths:
        usage = shutil.disk_usage(path)
        entries.append(
            {
                "path": path,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    report = {
        "schema": "selvedge/disk-free/v1",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "timestamp_utc": now,
        "paths": entries,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    target = os.path.join(args.out_dir, f"disk-{stamp}.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"disk-free: wrote {target}")
    for entry in entries:
        print(
            f"  {entry['path']} total={entry['total_bytes']} free={entry['free_bytes']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
