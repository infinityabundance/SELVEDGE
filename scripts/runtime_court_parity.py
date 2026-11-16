#!/usr/bin/env python3
"""Court E (real model, parity oracle): gate the runtime against NumPy.

Reads a ``selvedge run --json`` receipt produced with ``--compare-logits`` and
fails closed unless the comparison passed element by element *and* the argmax
agrees at every prefill position. The receipt is the evidence; this script only
refuses to let a failing comparison look like a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt")
    args = parser.parse_args()

    document = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    comparison = document.get("comparison") or {}
    if not comparison:
        print(f"FAIL {args.receipt} carries no comparison")
        return 1
    print(json.dumps(comparison, indent=2))
    failures = []
    if not comparison.get("passed"):
        failures.append("the elementwise comparison did not pass")
    if comparison.get("top1_matches") != comparison.get("positions"):
        failures.append("the argmax disagrees with the reference at some position")
    if document.get("exactness", {}).get("status") != "proven":
        failures.append("the run was not proven exact")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
