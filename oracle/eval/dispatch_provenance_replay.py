#!/usr/bin/env python3
"""Replay the routed-dispatch mask order of a historical checkpoint (P12.1).

Purpose. The T2 arms `X` and `H` were trained *before* commit `f13581f` added the
`mask_order` parameter to `oracle/train/modeling.py`. Their receipts therefore
carry no `dispatch_mask_order`, and the capability adapter refuses a routed
checkpoint whose mask order is unverified. This script does not score anything
and does not retrain: it replays the combine semantics of the revision in force
on a **fixed, deterministic batch** and records which mask order that revision
actually applied.

It compares four things on the same dispatch plan:

  * the historical combine (restore -> weight -> mask), re-implemented here
    exactly as `git show <rev>:oracle/train/modeling.py` has it for the
    pre-`f13581f` revisions;
  * the current `_RoutedDispatch` in `mask_order="legacy"` (same semantics);
  * the current `_RoutedDispatch` in `mask_order="pair"` (the correction);
  * the independent reference (`oracle/train/reference_dispatch.py`) in
    `"sorted"` (legacy) and `"pair"` modes.

If the historical combine is bitwise equal to legacy and differs from pair on
this batch, the historical mask order is `legacy`; that is the amendment the
adapter may consume. If they were equal (no overflow), the batch is not
discriminating and the script fails closed rather than certifying.

Run inside the oracle court (CPU is sufficient), e.g.:

  SELVEDGE_MEM=8g scripts/court oracle -- python3 oracle/eval/dispatch_provenance_replay.py \
      --out results/t2/provenance-replay-<stamp>.json

No network, no checkpoints, no writes outside `--out`. Deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import modeling  # noqa: E402
import reference_dispatch  # noqa: E402

HIDDEN = 4
TOKENS = 8
K = 2
EXPERTS = 4
FACTOR = 1.25


def _fixed_batch():
    """A deterministic batch whose routing overflows exactly one expert.

    token-major pair order is `pair = token * k + slot`. Expert 0 receives six
    pairs, experts 1/2 receive one each, expert 3 receives none. Capacity for
    this geometry is `max(1, ceil(8*2/4*1.25)) = 5`, so at least one expert-0
    pair is dropped and the sort permutation is not the identity: the batch is
    discriminating.
    """
    top_index = torch.tensor(
        [[0, 1], [0, 2], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]],
        dtype=torch.long,
    )
    flat = torch.arange(TOKENS * HIDDEN, dtype=torch.float32).reshape(TOKENS, HIDDEN)
    flat = flat / 10.0
    pair_weight = (torch.arange(TOKENS * K, dtype=torch.float32) + 1.0) / 10.0
    return top_index, flat, pair_weight


def historical_combine(output, row, order, valid, pair_weight, tokens, hidden):
    """The pre-`f13581f` combine, transcribed from the git blob.

    That revision had no `mask_order`: it restored to token-major order first,
    multiplied by the pair weight, and *then* applied `valid`, which is indexed
    by sorted position. This is the definition of the legacy placement.
    """
    pairs = tokens * K
    flat_out = output.reshape(-1, hidden).index_select(0, row)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(pairs, dtype=order.dtype))
    flat_out = flat_out.index_select(0, inverse)
    flat_out = flat_out * pair_weight.unsqueeze(-1).to(flat_out.dtype)
    flat_out = flat_out * valid.unsqueeze(-1).to(flat_out.dtype)
    return flat_out.view(tokens, K, hidden).sum(dim=1)


def _current(mask_order, top_index, flat, pair_weight, tokens):
    dispatch = modeling._RoutedDispatch(HIDDEN, EXPERTS, K, mask_order=mask_order)
    buffer, row, order, valid = dispatch.dispatch(flat, top_index)
    out = dispatch.combine(buffer, row, order, valid, pair_weight, tokens)
    return out, dispatch


def _reference(mask_mode, top_index, flat, pair_weight):
    plan = reference_dispatch.plan(top_index, EXPERTS, FACTOR)
    buffer = reference_dispatch.build_buffer(flat, plan)
    return reference_dispatch.assemble(buffer, plan, pair_weight, mask_mode=mask_mode)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--historical-revision", default="")
    ap.add_argument("--historical-blob", default="")
    args = ap.parse_args(argv)

    torch.manual_seed(20261116)
    top_index, flat, pair_weight = _fixed_batch()

    legacy, legacy_mod = _current("legacy", top_index, flat, pair_weight, TOKENS)
    corrected, corrected_mod = _current("pair", top_index, flat, pair_weight, TOKENS)
    h_buffer, h_row, h_order, h_valid = legacy_mod.dispatch(flat, top_index)
    historical = historical_combine(
        h_buffer, h_row, h_order, h_valid, pair_weight, TOKENS, HIDDEN,
    )
    ref_sorted = _reference("sorted", top_index, flat, pair_weight)
    ref_pair = _reference("pair", top_index, flat, pair_weight)

    # A second, overflow-free batch is the control: both placements must agree
    # there, which proves the divergence below is caused by overflow, not by an
    # unrelated arithmetic difference between the two code paths.
    top_ok = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.long)
    flat_ok = torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN) / 10.0
    pw_ok = (torch.arange(4 * K, dtype=torch.float32) + 1.0) / 10.0
    ok_legacy, _ = _current("legacy", top_ok, flat_ok, pw_ok, 4)
    ok_pair, _ = _current("pair", top_ok, flat_ok, pw_ok, 4)

    pairs = TOKENS * K
    dropped = int((~legacy_mod.dispatch(flat, top_index)[3]).sum().item())

    # The mask-order question is *which token-major pair is zeroed*, independent
    # of which buffer row a mislabelled pair reads. `valid` is sorted-position
    # indexed. Legacy gates token-major pair p by valid[p]; the correction gates
    # it by valid[inverse[p]]. Count the pairs where those disagree.
    _buf, _row, order, valid = legacy_mod.dispatch(flat, top_index)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(pairs, dtype=order.dtype))
    legacy_gate = valid
    pair_gate = valid.index_select(0, inverse)
    misassigned = int((legacy_gate != pair_gate).sum().item())

    evidence = {
        "schema": "selvedge/p12.1-provenance-replay/v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch": {
            "hidden": HIDDEN, "tokens": TOKENS, "k": K, "experts": EXPERTS,
            "capacity_factor": FACTOR, "pairs": pairs, "dropped_pairs": dropped,
            "top_index": top_index.tolist(),
            "seed": 20261116,
            "discriminating": dropped > 0,
        },
        "historical_revision": args.historical_revision,
        "historical_modeling_blob": args.historical_blob,
        "comparisons": {
            "historical_vs_current_legacy_max_abs": _max_abs(historical, legacy),
            "historical_vs_current_pair_max_abs": _max_abs(historical, corrected),
            "current_legacy_vs_reference_sorted_max_abs": _max_abs(legacy, ref_sorted),
            "current_pair_vs_reference_pair_max_abs": _max_abs(corrected, ref_pair),
            "legacy_vs_pair_max_abs": _max_abs(legacy, corrected),
            "control_no_overflow_legacy_vs_pair_max_abs": _max_abs(ok_legacy, ok_pair),
        },
        "masks": {
            "legacy_gate_token_major": [bool(x) for x in legacy_gate.tolist()],
            "pair_gate_token_major": [bool(x) for x in pair_gate.tolist()],
            "pairs_gated_differently": misassigned,
            "note": "valid is sorted-position indexed; legacy applies it to the "
                    "restored token-major tensor, the correction applies it before "
                    "the restore via the inverse permutation",
        },
        "agreements": {
            "historical_equals_current_legacy": bool(torch.equal(historical, legacy)),
            "historical_equals_current_pair": bool(torch.equal(historical, corrected)),
            "current_legacy_equals_reference_sorted": bool(torch.equal(legacy, ref_sorted)),
            "current_pair_equals_reference_pair": bool(torch.equal(corrected, ref_pair)),
            "placements_agree_without_overflow": bool(torch.equal(ok_legacy, ok_pair)),
        },
        "outputs": {
            "historical": historical.tolist(),
            "current_legacy": legacy.tolist(),
            "current_pair": corrected.tolist(),
        },
    }

    a = evidence["agreements"]
    if not a["historical_equals_current_legacy"]:
        evidence["conclusion"] = "REFUSED: historical combine does not match legacy"
        code = 2
    elif not a["historical_equals_current_pair"]:
        evidence["conclusion"] = "legacy"
        code = 0
    elif a["placements_agree_without_overflow"]:
        evidence["conclusion"] = "REFUSED: batch did not discriminate the two placements"
        code = 3
    else:
        evidence["conclusion"] = "REFUSED: historical matches both placements"
        code = 4
    if not a["current_pair_equals_reference_pair"]:
        evidence["conclusion"] += " (reference pair disagreement)"
        code = max(code, 5)
    if code == 0 and misassigned == 0:
        evidence["conclusion"] = "REFUSED: masks do not differ on this batch"
        code = 6
    # The reference models a dropped pair as reading the last buffer row, while the
    # shipped code reads `expert*capacity + slot` (which can land in the next
    # expert's block). On a batch with cross-over misassignment those row choices
    # differ even though the *mask* is identical, so a numeric difference here is
    # documented rather than treated as a mask-order failure.
    if not a["current_legacy_equals_reference_sorted"]:
        evidence["reference_sorted_numeric_note"] = (
            "legacy and reference-sorted select the same gate (valid[p]) but model "
            "the buffer row of a mislabelled pair differently; the mask comparison "
            "below is the decisive record for mask order"
        )

    evidence["evidence_sha256"] = _sha(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(evidence["conclusion"]))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
