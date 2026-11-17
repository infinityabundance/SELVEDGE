#!/usr/bin/env python3
"""S50: invalidate the S49 stability sweep and its bar by the optimiser defect.

Additive only. The amendment, the run receipts, the checkpoints, the evaluation
receipts and the selection markers stay exactly as they are; this script writes

* ``INVALIDATED.json`` inside every affected run/eval directory, and
* one master record under ``results/stage2/`` naming every affected artefact and
  its SHA-256,

so a reader of any directory sees that the numbers beside it may not be used as
a bar, baseline or selection input, and no receipt has been deleted or edited.

It does not edit the pre-registration or its amendments: those are immutable
evidence of what was registered, not of what the corrected optimiser produced.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNS = [
    ("results/stage2/runs/full-20m-math", "the original 20M bar at lr 1e-4"),
    ("results/stage2/runs/full-20m-math-s49",
     "the S49 20M bar at the swept lr 3e-5"),
    ("results/stage2/runs/stability-lr3e-5-2m", "stability sweep, lr 3e-5, 2M"),
    ("results/stage2/runs/stability-lr3e-5-2m-smoke", "stability smoke, lr 3e-5"),
    ("results/stage2/runs/stability-lr1e-5-2m", "stability sweep, lr 1e-5, 2M"),
    ("results/stage2/runs/stability-lr1e-5-2m-smoke", "stability smoke, lr 1e-5"),
    ("results/stage2/runs/stability-lr5e-6-2m", "stability sweep, lr 5e-6, 2M"),
    ("results/stage2/runs/stability-lr5e-6-2m-smoke", "stability smoke, lr 5e-6"),
]

EVALS = [
    ("results/stage2/eval/full-20m-math", "anchored score of the original 20M bar"),
    ("results/stage2/eval/full-20m-math-s49",
     "anchored score of the S49 20M bar (the m5 pass was stopped incomplete)"),
]

SELECTIONS = [
    ("results/stage2/s49-stability-decision-20261117T172316Z.json",
     "S49 stability decision"),
    ("results/stage2/s49-stability-decision-20261117T172328Z.json",
     "S49 stability decision"),
    ("results/stage2/runs/full-20m-math-s49-selected.json",
     "S49 bar selection (lr 3e-5)"),
    ("results/stage2/s49-heldout-stability-lr3e-5-2m.json",
     "S49 held-out comparison, lr 3e-5"),
    ("results/stage2/s49-heldout-stability-lr1e-5-2m.json",
     "S49 held-out comparison, lr 1e-5"),
    ("results/stage2/s49-heldout-stability-lr5e-6-2m.json",
     "S49 held-out comparison, lr 5e-6"),
]

AMENDMENTS = [
    ("docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json",
     "the stability-sweep amendment (registration retained verbatim)"),
    ("docs/ws19-full-distillation-amendment-stability-s49b-preregistration-v1.json",
     "the bar-selection amendment (registration retained verbatim)"),
]

EVIDENCE = [
    ("results/stage2/s50-optim-defect-sim-20261117T203840Z.json",
     "reproduction of the linear-absmax second-moment defect"),
    ("results/stage2/s50-parity-components-20261117T203840Z.json",
     "attribution of the remaining parity error to the int8 first moment"),
]

SUPERSEDED_BY = {
    "run": "results/stage2/runs/s50-bar-20m-lr1e-5",
    "lr": 1e-5,
    "revalidation": "results/stage2/runs/s50-revalidate-200-lr1e-5",
    "fixed_batch_kl_training_1e-5": (
        "results/stage2/s50-fixed-batch-kl-math-lr1e-5-20261117T205600Z.json"),
    "fixed_batch_kl_training_1e-4": (
        "results/stage2/s50-fixed-batch-kl-math-20261117T205500Z.json"),
}

REASON = (
    "Adam8bit stored the second moment as a per-256-element linear absmax int8 "
    "code. Any squared gradient below about absmax/254 rounded to zero, and Adam "
    "then divided by eps whenever the current gradient was also small. The "
    "reproduction measures 57.7% of elements with a zeroed stored second moment "
    "and a nonzero first moment, a maximum update 3.2e4 times fp32 AdamW's, and "
    "a 313% parameter error after 50 steps. Every number produced by these runs "
    "is therefore a property of the defective optimiser; the learning-rate sweep "
    "reduced the damage but did not remove it, and the selected 3e-5 rate is not "
    "used. Receipts and checkpoints are retained verbatim and may not be used as "
    "a bar, baseline or selection input.")


def sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    master_rel = f"results/stage2/s50-invalidation-{stamp}.json"
    master_path = os.path.join(ROOT, master_rel)
    markers = []
    affected = {"runs": [], "evaluations": [], "selections": [],
                "amendments": [], "evidence": []}

    for path, label in RUNS:
        directory = os.path.join(ROOT, path)
        if not os.path.isdir(directory):
            continue
        receipt = os.path.join(directory, "receipt.json")
        affected["runs"].append({
            "path": path, "label": label, "invalidated": True,
            "receipt_sha256": sha256(receipt),
            "kept": sorted(os.path.basename(item)
                           for item in os.listdir(directory)),
        })
        markers.append((directory, path))

    for path, label in EVALS:
        directory = os.path.join(ROOT, path)
        if not os.path.isdir(directory):
            continue
        affected["evaluations"].append({
            "path": path, "label": label, "invalidated": True, "kept": True})
        markers.append((directory, path))

    for path, label in SELECTIONS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            continue
        affected["selections"].append({
            "path": path, "label": label, "invalidated": True,
            "sha256": sha256(full)})

    for path, label in AMENDMENTS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            continue
        affected["amendments"].append({
            "path": path, "label": label, "invalidated_as_evidence": True,
            "edited": False, "sha256": sha256(full)})

    for path, label in EVIDENCE:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            continue
        affected["evidence"].append({
            "path": path, "label": label, "sha256": sha256(full)})

    record = {
        "schema": "selvedge/s50-invalidation/v1",
        "kind": "s50-invalidation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": stamp,
        "revision": revision(),
        "reason": REASON,
        "defect": {
            "module": "oracle/frontier/optim8bit.py",
            "defective_revision": "e57df2f",
            "second_moment_pre_fix": "linear absmax int8 per 256-element block",
            "second_moment_post_fix": "bfloat16, unquantised",
            "first_moment_post_fix": "linear absmax int8 per 256-element block",
        },
        "affected": affected,
        "disposition": (
            "retained verbatim; no number from these runs may be used as a bar, "
            "baseline, selection input or comparison; they are not scored as the "
            "bar in any S50 or later report"),
        "superseded_by": SUPERSEDED_BY,
        "markers_written": [path for _, path in markers],
        "master_record": master_rel,
    }

    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

    marker = {
        "schema": "selvedge/s50-invalidation-marker/v1",
        "kind": "s50-invalidation-marker",
        "created_utc": record["created_utc"],
        "revision": record["revision"],
        "invalidated": True,
        "reason": REASON,
        "master_record": master_rel,
        "superseded_by": SUPERSEDED_BY,
    }
    for directory, path in markers:
        payload = dict(marker, path=path)
        with open(os.path.join(directory, "INVALIDATED.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(f"s50-invalidate: master {master_rel}")
    print(f"s50-invalidate: markers {len(markers)} "
          f"runs={len(affected['runs'])} evals={len(affected['evaluations'])} "
          f"selections={len(affected['selections'])} "
          f"amendments={len(affected['amendments'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
