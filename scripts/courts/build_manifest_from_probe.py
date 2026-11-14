#!/usr/bin/env python3
"""Rebuild docs/perf-numbers.json from a hardware probe receipt.

The probe names its own evidence (the court receipts for host facts, GPU and
toolchains), so the manifest can be regenerated after a re-probe without
hand-editing a receipt path, stamp or revision. The block table below is the
only hand-maintained input, and every entry names a JSON pointer that
`perf_numbers.py --write` will resolve and render.

Usage (inside the Python court, or on the host for authoring):

    python3 scripts/courts/build_manifest_from_probe.py \
        --probe results/hardware/probe-<stamp>.json

Then:

    scripts/court perf_numbers --write
    scripts/court perf_numbers
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DOC = "docs/PERF-NUMBERS.md"

# (id, source key, JSON pointer, format)
#   source "probe"  -> the hardware probe receipt
#   source "rust"   -> the Rust toolchain court receipt the probe names
#   source "python" -> the Python toolchain court receipt the probe names
BLOCKS = [
    ("cpu-logical-cores", "probe", "/cpu/logical_cores", "{value}"),
    ("cpu-physical-cores", "probe", "/cpu/physical_cores", "{value}"),
    ("ram-total-gib", "probe", "/memory/total_gib", "{value:.2f} GiB"),
    ("swap-total-gib", "probe", "/memory/swap_total_gib", "{value:.2f} GiB"),
    ("repo-fs-total-gib", "probe", "/storage/mounts/0/total_gib", "{value:.2f} GiB"),
    ("kernel-release", "probe", "/kernel/release", "{value}"),
    ("rustc-version", "rust", "/toolchain/rustc", "{value}"),
    ("python-version", "python", "/toolchain/python3", "{value}"),
    ("gpu-name", "probe", "/gpu/devices/0/name", "{value}"),
    ("gpu-vram", "probe", "/gpu/devices/0/vram_total", "{value}"),
]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", default="docs/perf-numbers.json")
    parser.add_argument("--doc", default=DOC)
    args = parser.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    probe_path = args.probe if os.path.isabs(args.probe) else os.path.join(repo, args.probe)
    with open(probe_path, "r", encoding="utf-8") as handle:
        probe = json.load(handle)
    if probe.get("schema") != "selvedge/hardware-probe/v1":
        raise SystemExit(f"not a hardware probe receipt: {args.probe}")

    evidence = probe.get("evidence") or {}
    sources = {
        "probe": (os.path.relpath(probe_path, repo), probe),
        "rust": (evidence["rust_toolchain_court_receipt"], None),
        "python": (evidence["python_toolchain_court_receipt"], None),
    }
    for key in ("rust", "python"):
        rel_path, _ = sources[key]
        with open(os.path.join(repo, rel_path), "r", encoding="utf-8") as handle:
            sources[key] = (rel_path, json.load(handle))

    blocks = []
    for ident, source_key, pointer, fmt in BLOCKS:
        receipt_rel, receipt = sources[source_key]
        blocks.append(
            {
                "id": ident,
                "doc": args.doc,
                "sentinel": ident,
                "source": {"receipt": receipt_rel, "pointer": pointer},
                "format": fmt,
                "declared_receipt_stamp": receipt["stamp"],
                "declared_revision": receipt["revision"],
            }
        )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(repo, args.out)
    temporary = out_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump({"schema": "selvedge/perf-numbers/v1", "blocks": blocks}, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, out_path)
    print(f"build-manifest: wrote {os.path.relpath(out_path, repo)} from {os.path.relpath(probe_path, repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
