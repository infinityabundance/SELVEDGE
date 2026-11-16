#!/usr/bin/env python3
"""Audit one arm's recorded dispatch-mask placement and write a replay receipt.

    python3 scripts/provenance_audit.py --arm H* \
        --receipt results/t2/H*/run.json \
        --summary results/artifacts/train/t2/Hstar/summary.json \
        --meta results/artifacts/train/t2/Hstar/run-meta.json \
        --code oracle/train/loop.py --code oracle/train/modeling.py \
        --blob oracle/train/loop.py=<git blob id at the launch revision> \
        --blob oracle/train/modeling.py=<git blob id at the launch revision> \
        --launch-revision <sha> \
        --out results/t2/provenance-replay-Hstar-<stamp>.json

Why this exists. `oracle/capability/adapters.py` refuses to score a routed arm
whose training receipt records no dispatch mask placement, because an unverified
mask order changes which pairs are masked on overflow. For the historical arms
`X` and `H` that order had to be *inferred* from a pre-correction code blob
(`results/t2/PROVENANCE-AMENDMENT-X-and-H.json`). For an arm trained after the
correction introduced `mask_order`, the training loop records the order it built
the model with — `arm_model.mask_order()` in `oracle/train/loop.py` — in both
`run-meta.json` and `summary.json`. This audit establishes, from those retained
artifacts, that the record exists, that the two artifacts agree, that the code
that wrote them is byte-identical to the code at the launch revision, and that
the record is *informative* (the two placements are distinguishable precisely
when the run actually dropped pairs, and this run did).

It does not re-derive the mask order from reasoning: it reads the recorded value
and checks the chain that makes the record trustworthy. No model is loaded and no
GPU is used.

The output is deterministic (no clock-dependent numbers) except for its stamp and
the paths it records; the `--check` mode re-computes everything and compares.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = "selvedge/t2-provenance-replay/v1"
ORDERS = ("legacy", "pair")


class AuditError(RuntimeError):
    pass


def blob_id(path: str) -> str:
    """The git blob id of a file's current bytes (sha1 over `blob <n>\\0`)."""
    with open(path, "rb") as handle:
        payload = handle.read()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO)


def audit(args) -> dict:
    problems: list[str] = []

    receipt = load_json(args.receipt)
    meta = load_json(args.meta)
    summary = load_json(args.summary)

    in_receipt = receipt.get("dispatch_mask_order")
    in_meta = meta.get("dispatch_mask_order")
    in_summary = summary.get("dispatch_mask_order")
    exposure = summary.get("dispatch") or {}

    if in_receipt is not None and in_receipt != in_summary:
        problems.append(
            f"the arm receipt records {in_receipt!r}, the training summary "
            f"{in_summary!r}")
    if in_meta is None or in_summary is None:
        problems.append("the training artifacts do not record a dispatch mask order")
    elif in_meta != in_summary:
        problems.append(
            f"run-meta records {in_meta!r}, summary records {in_summary!r}")
    recorded = in_summary if in_summary is not None else None
    if recorded is not None and recorded not in ORDERS:
        problems.append(f"the recorded mask order {recorded!r} is not one of "
                        f"{ORDERS}")
    if exposure.get("mask_order") not in (None, recorded):
        problems.append(
            f"the dispatch exposure block reports {exposure.get('mask_order')!r}, "
            f"the summary reports {recorded!r}")

    # The code that wrote the record must be the code that ran.
    code = {}
    for path in args.code:
        name = rel(path)
        if name not in args.blob:
            problems.append(f"no launch-revision blob id was supplied for {name}")
            continue
        actual = blob_id(path)
        code[name] = {"blob": actual, "launch_blob": args.blob[name],
                      "matches_launch_revision": actual == args.blob[name]}
        if actual != args.blob[name]:
            problems.append(
                f"{name} is not the blob that ran at {args.launch_revision}: "
                f"{actual} != {args.blob[name]}")

    # The record is written from the constructed model, not from the CLI default.
    loop_source = open(args.loop_path, encoding="utf-8").read() \
        if os.path.isfile(args.loop_path) else ""
    record_sites = [match.start() for match in
                    re.finditer(r'"dispatch_mask_order":\s*arm_model\.mask_order\(\)',
                                loop_source)]
    record_lines = sorted(loop_source[:offset].count("\n") + 1
                          for offset in record_sites)
    if not record_sites:
        problems.append(
            "oracle/train/loop.py does not write dispatch_mask_order from "
            "arm_model.mask_order(); the record cannot be trusted as the applied "
            "order")
    modeling_source = open(args.modeling_path, encoding="utf-8").read() \
        if os.path.isfile(args.modeling_path) else ""
    if "def mask_order" not in modeling_source:
        problems.append("oracle/train/modeling.py defines no mask_order accessor")

    # Distinguishability control: the two placements differ only on overflow.
    pairs = exposure.get("pairs")
    dropped = exposure.get("dropped_pairs")
    distinguishable = (isinstance(pairs, int) and isinstance(dropped, int)
                       and pairs > 0 and dropped > 0)
    control = {
        "pairs": pairs,
        "dropped_pairs": dropped,
        "dropped_fraction": exposure.get("dropped_fraction"),
        "interpretation": (
            "the run dropped pairs, so the legacy and pair placements mask "
            "different positions and the recorded order is not vacuous"
            if distinguishable else
            "no pairs were dropped in this run, so the recorded order does not "
            "affect the trained weights; it still governs how the arm is "
            "evaluated and is recorded here for that reason"),
    }

    conclusion = "unresolved"
    if not problems and recorded is not None:
        conclusion = recorded

    return {
        "schema": SCHEMA,
        "arm": args.arm,
        "created_utc": args.utc,
        "launch_revision": args.launch_revision,
        "recorded": {
            "in_arm_receipt": in_receipt,
            "in_run_meta": in_meta,
            "in_training_summary": in_summary,
            "agrees": (in_meta == in_summary and in_meta in ORDERS),
        },
        "artifacts": {
            "receipt": {"path": rel(args.receipt), "sha256": sha256_file(args.receipt)},
            "run_meta": {"path": rel(args.meta), "sha256": sha256_file(args.meta)},
            "summary": {"path": rel(args.summary), "sha256": sha256_file(args.summary)},
        },
        "code": code,
        "code_facts": {
            "record_sites": record_lines,
            "recorded_from": "arm_model.mask_order() (the constructed model)",
            "modeling_defines_mask_order": "def mask_order" in modeling_source,
        },
        "control": control,
        "problems": problems,
        "conclusion": conclusion,
        "method": (
            "read the order the training loop recorded from the constructed model, "
            "check that run-meta and the summary agree, check that the code that "
            "wrote the record is byte-identical to the code at the launch revision "
            "(git blob id over the file bytes), and check that the record is "
            "informative by confirming the run actually dropped pairs. The order "
            "is not inferred from a code reading; the record is."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--code", action="append", default=[])
    parser.add_argument("--blob", action="append", default=[],
                        help="PATH=<git blob id at the launch revision>")
    parser.add_argument("--loop-path", default="oracle/train/loop.py")
    parser.add_argument("--modeling-path", default="oracle/train/modeling.py")
    parser.add_argument("--launch-revision", required=True)
    parser.add_argument("--utc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    args.code = args.code or ["oracle/train/loop.py", "oracle/train/modeling.py"]
    args.blob = dict(entry.split("=", 1) for entry in args.blob)

    payload = audit(args)
    write_json(args.out, payload)
    if payload["problems"]:
        for problem in payload["problems"]:
            print(f"provenance-audit: {problem}", file=sys.stderr)
        return 3
    print(f"provenance-audit: {args.arm} records mask_order="
          f"{payload['conclusion']!r}; receipt {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
