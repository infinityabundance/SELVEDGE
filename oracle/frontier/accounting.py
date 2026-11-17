#!/usr/bin/env python3
"""Exact stage-2 accounting receipt (CPU, no model, no GPU).

Every parameter count the stage-2 report quotes is printed here from the same
`MemoryGeometry` the module executes. The receipt covers the development table
and the two registered scale questions, the four-site variant that the existing
PROGRAM-PLAN number implicitly assumed, the parameter-matched dense adapter's
exact match, and the parameter-free NGM control's read set.

Nothing here is a measurement of throughput or quality; it is the declared
arithmetic. A measured claim cites the smoke/graft receipts instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, ORACLE)

from frontier import arms, memory  # noqa: E402


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(HERE)), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/stage2")
    args = parser.parse_args(argv)
    stamp_now = stamp()
    device = memory.MemoryGeometry()
    one_site = {str(rows): memory.scale_table(rows, sites=1)
                for rows in memory.REGISTERED_ROW_SCALES}
    four_site = {str(rows): memory.scale_table(rows, sites=4)
                 for rows in memory.REGISTERED_ROW_SCALES}
    adapter = arms.DenseAdapter(hidden=device.hidden)
    receipt = {
        "schema": "selvedge/stage2-accounting/v1",
        "kind": "accounting",
        "created_utc": utcnow(),
        "stamp": stamp_now,
        "revision": revision(),
        "default_geometry": device.report(),
        "one_site": one_site,
        "four_site": four_site,
        "dense_adapter": {
            **adapter.active_parameter_report(),
            "memory_active_per_token_per_site": device.active_per_token_per_site,
            "exact_match": adapter.active_elements == device.active_per_token_per_site,
            "delta": adapter.active_elements - device.active_per_token_per_site,
            "rank_choice": "round((active - hidden) / (2 * hidden))",
        },
        "ngm": arms.NGMInjector(hidden=device.hidden).active_parameter_report(),
        "notes": {
            "active": "per token, one site: 8 rows x 256 values + key projection "
                      "+ value projection + output bias + depthwise conv",
            "accessible": "table + the same trainable parameter sets",
            "table_charged_to_active": False,
            "four_site_note": "the PROGRAM-PLAN P19.2 table-size generated "
                              "blocks count rows x 1024 in one number; that is "
                              "four 256-value sites, which is why the four_site "
                              "table here reproduces those totals while the "
                              "default one-site module does not",
            "read_bytes": "modelled distinct-identity lower bound, not measured traffic",
        },
    }
    path = os.path.join(args.out_dir, f"stage2-accounting-{stamp_now}.json")
    write_json(path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("default_geometry", "dense_adapter", "ngm")}, indent=2))
    print(f"accounting: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
