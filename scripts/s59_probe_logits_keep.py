#!/usr/bin/env python3
"""S59 probe: can the LM head emit logits for a position subset, exactly?

The stage-2b smoke OOMs by ~50-300 MiB because the completion-only SFT row
computes logits for all 1,249 positions although only the supervised positions
carry a loss. If `logits_to_keep` accepts an index tensor, the head can emit
*only* the supervised positions. That is arithmetically exact: cross-entropy on
masked (`-100`) positions contributes neither loss nor gradient, so the kept
positions produce the same total and the same parameter gradient.

This probe loads the student on CPU, runs one short prefix, and compares

    full.logits[:, keep]  against  subset.logits

bitwise for the kept positions. Any difference means the path is not exact and
must not be used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-dir", required=True)
    args = parser.parse_args(argv)

    import torch
    from frontier import arms

    torch.manual_seed(0)
    model = arms.load_model(args.student_dir, dtype=torch.bfloat16,
                           device="cpu")
    model.eval()
    report: dict = {"class": type(model).__name__}
    ids = torch.tensor([[248045, 846, 198, 623, 264, 6673, 1228, 11, 4246,
                         16060, 10598, 430, 1691, 430, 279, 3140, 5284, 13,
                         1368, 279, 8166, 5284, 369, 220, 24, 23, 321, 279,
                         2020, 314, 279, 11857, 369, 220, 22, 20, 11, 1092,
                         557, 4246, 579, 5284, 30, 271, 50, 3815, 279, 3377]])
    keep = torch.tensor([5, 9, 17, 30, 44])
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        try:
            subset = model(input_ids=ids, use_cache=False,
                           logits_to_keep=keep).logits
            report["supported"] = True
        except Exception as error:  # noqa: BLE001 - the failure is the answer
            report["supported"] = False
            report["error"] = f"{type(error).__name__}: {error}"
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
    report["full_shape"] = list(full.shape)
    report["subset_shape"] = list(subset.shape)
    report["bitwise_equal"] = bool(torch.equal(full[:, keep, :], subset))
    report["max_abs_difference"] = float(
        (full[:, keep, :].float() - subset.float()).abs().max())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
