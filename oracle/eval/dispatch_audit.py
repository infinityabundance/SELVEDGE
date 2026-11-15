#!/usr/bin/env python3
"""P12.1 — routed-overflow validity-mask audit, with a receipt.

Three questions, one receipt:

1. **Is the shipped mask applied to the pairs it describes?** The toy cases below
   compare `modeling._RoutedDispatch` against `reference_dispatch`, an
   independent transcription of the documented semantics, in both mask
   placements. The `legacy` placement (the default, what every registered T2 arm
   ran) must reproduce the *sorted-order* mask exactly; the `pair` placement must
   reproduce the documented semantics exactly.
2. **Did the registered arms actually overflow?** Every T2 training log records
   the per-layer expert counts and the dropped-pair count every 20th step.
3. **If they did, how many pairs were mis-assigned by the legacy mask?** The
   recorded counts determine *which* sorted positions are valid but not which
   pair sits at each position (that needs the hidden states). The audit therefore
   reports (a) the exact dropped count, and (b) the two mis-assignment counters
   under a uniform-random pairing of pairs to sorted positions, with the analytic
   expectation and the interval the counters can occupy. (b) is a model of the
   permutation, labelled as such; it is not a measurement of the permutation.

Nothing here needs a GPU, changes an arm, or touches the pilot.

Usage:

    scripts/court oracle -- python3 oracle/eval/dispatch_audit.py \
        --out results/t2/dispatch-audit-$(date -u +%Y%m%dT%H%M%SZ).json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ORACLE, "train"))
sys.path.insert(0, os.path.join(ORACLE, "eval"))

import modeling  # noqa: E402
import reference_dispatch as reference  # noqa: E402

REPO = os.path.dirname(ORACLE)
SCHEMA = "selvedge/p12.1-dispatch-audit/v1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 1. the toy cases


class _ForcedBank(modeling.SpecialistBank):
    """A bank whose routing decision is supplied instead of computed."""

    def __init__(self, hidden, width, experts, k, top_index, pair_weight):
        super().__init__(hidden, width, experts, k,
                         modeling._router(hidden, experts, seed=5))
        with torch.no_grad():
            self.down.normal_(generator=torch.Generator().manual_seed(3))
        self._top_index = top_index
        self._pair_weight = pair_weight

    def routing(self, flat, router_weight):
        probs = torch.full((flat.shape[0], self.experts), 1.0 / self.experts)
        return probs, torch.zeros_like(probs), self._top_index, self._pair_weight


def _expert_fn(bank, flat):
    import torch.nn.functional as F

    def call(expert, tokens):
        rows = flat.index_select(0, tokens)
        gate = F.linear(rows, bank.gate[expert])
        up = F.linear(rows, bank.up[expert])
        return F.linear(F.silu(gate) * up, bank.down[expert])
    return call


CASES = (
    {"name": "no-overflow", "top": [[0, 1], [2, 3], [1, 0], [3, 2], [0, 3]],
     "experts": 4, "k": 2, "factor": 1.25},
    {"name": "overflow-one-expert", "top": [[0, 0], [0, 1], [0, 2], [0, 3]],
     "experts": 4, "k": 2, "factor": 0.5},
    {"name": "overflow-last-expert-clamps", "top": [[3, 0], [3, 1], [3, 2]],
     "experts": 4, "k": 2, "factor": 0.5},
    {"name": "ties", "top": [[0, 1], [0, 1], [0, 1], [0, 1]],
     "experts": 4, "k": 2, "factor": 0.5},
    {"name": "duplicate-ids", "top": [[2, 2, 2], [0, 2, 1], [2, 0, 0]],
     "experts": 4, "k": 3, "factor": 0.75},
    {"name": "k-one", "top": [[1], [1], [1], [2]], "experts": 3, "k": 1,
     "factor": 0.5},
    {"name": "k-equals-experts", "top": [[0, 1, 2], [2, 1, 0]], "experts": 3,
     "k": 3, "factor": 0.5},
    {"name": "single-token", "top": [[1, 2]], "experts": 4, "k": 2, "factor": 0.25},
    {"name": "empty-tokens", "top": [], "experts": 4, "k": 2, "factor": 1.25},
    {"name": "k-zero", "top": [[], [], []], "experts": 4, "k": 0, "factor": 1.25},
)


def toy_case(case: dict) -> dict:
    tokens, hidden, width = len(case["top"]), 6, 3
    top_index = torch.tensor(case["top"], dtype=torch.long).reshape(tokens, case["k"])
    weight = torch.arange(1, tokens * case["k"] + 1, dtype=torch.float32)
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(19))
    bank = _ForcedBank(hidden, width, case["experts"], case["k"], top_index, weight)
    bank.capacity_factor = case["factor"]
    fn = _expert_fn(bank, flat)
    specification = reference.plan(top_index, experts=case["experts"],
                                   factor=case["factor"])
    buffer, _row, _order, _valid = bank.dispatch(flat, top_index)
    buffer_equal = bool(torch.equal(buffer, reference.build_buffer(flat, specification)))

    results = {}
    for placement, mode in (("pair", modeling.MASK_ORDER_PAIR),
                            ("sorted", modeling.MASK_ORDER_LEGACY)):
        expected = reference.contribution(flat, top_index, weight, fn,
                                          experts=case["experts"],
                                          factor=case["factor"], mask_mode=placement)
        bank.mask_order = mode
        got = bank(flat.reshape(1, tokens, hidden)).reshape(tokens, hidden)
        results[placement] = {
            "agrees_with_reference": bool(torch.allclose(got.double(),
                                                         expected.double(), atol=1e-6)),
            "max_abs_difference": float((got.detach().double()
                                          - expected.detach().double()).abs().max())
            if tokens else 0.0,
        }

    pair_mode = reference.contribution(flat, top_index, weight, fn,
                                       experts=case["experts"],
                                       factor=case["factor"], mask_mode="pair")
    sorted_mode = reference.contribution(flat, top_index, weight, fn,
                                         experts=case["experts"],
                                         factor=case["factor"], mask_mode="sorted")
    pair_counts = {
        "capacity": specification.capacity,
        "pairs": specification.pairs,
        "dropped": len(specification.dropped()),
        "both_placements_agree": bool(torch.allclose(pair_mode, sorted_mode)),
    }
    return {"case": case["name"], "tokens": tokens, "experts": case["experts"],
            "k": case["k"], "capacity_factor": case["factor"],
            "dispatch_halves_bitwise_equal": buffer_equal,
            "placements": results, **pair_counts}


def headline_case() -> dict:
    """A kept overflow pair that reads another token's activation."""
    hidden, width, experts, k, tokens, factor = 4, 2, 4, 2, 3, 0.5
    top_index = torch.tensor([[3, 0], [3, 1], [3, 2]])
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    flat = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(11))
    bank = _ForcedBank(hidden, width, experts, k, top_index, weight)
    bank.capacity_factor = factor
    fn = _expert_fn(bank, flat)
    specification = reference.plan(top_index, experts=experts, factor=factor)
    pair_mode = reference.contribution(flat, top_index, weight, fn, experts=experts,
                                       factor=factor, mask_mode="pair")
    sorted_mode = reference.contribution(flat, top_index, weight, fn, experts=experts,
                                         factor=factor, mask_mode="sorted")
    wrong = weight[2] * fn(3, torch.tensor([0]))[0]
    return {
        "tokens": tokens, "experts": experts, "k": k, "capacity_factor": factor,
        "capacity": specification.capacity,
        "overflow_pairs": len(specification.dropped()),
        "overflow_pair_receives_another_tokens_activation": bool(
            torch.allclose(sorted_mode[1].double(), (pair_mode[1] + wrong).double(),
                           atol=1e-6)),
        "valid_pair_dropped_by_the_legacy_mask": bool(
            torch.allclose(sorted_mode[2].double(),
                           torch.zeros(hidden, dtype=torch.float64), atol=1e-9)
            and float(pair_mode[2].detach().abs().sum()) > 1e-6),
        "max_abs_difference": float((sorted_mode.detach() - pair_mode.detach()).abs().max()),
        "dispatch_halves_bitwise_equal": bool(
            torch.equal(bank.dispatch(flat, top_index)[0],
                        reference.build_buffer(flat, specification))),
    }


# ---------------------------------------------------------------------------
# 2 and 3. the pilot's recorded overflow, and the mis-assignment model


def misassignment(counts: list[int], capacity: int, draws: int,
                  seed: int) -> dict:
    """Mis-assignment counters under a uniform-random pair->position pairing.

    `counts[e]` is the number of pairs routed to expert `e`. In sorted order the
    expert blocks are contiguous, so `valid` (a property of the sorted position)
    is fully determined by `counts`. What is *not* determined is which
    token-major pair sits at which sorted position.

    A pair `p` is gated (in the shipped placement) by `valid[p]`, while its own
    validity is `valid[inverse[p]]`. The two counters are:

    * **retained overflow** — `valid[p]` and not `valid[inverse[p]]`: the pair
      overflowed, was not dropped, and reads a clamped buffer row (possibly
      another token's activation);
    * **silently dropped** — not `valid[p]` and `valid[inverse[p]]`: the pair
      was executed in the buffer and its contribution is discarded.

    Both are reported per draw; the mean, min and max over draws are returned.
    """
    pairs = sum(counts)
    if not pairs:
        return {"draws": 0}
    valid = []
    for expert, count in enumerate(counts):
        valid.extend([slot < capacity for slot in range(count)])
    assert len(valid) == pairs
    retained, dropped = [], []
    generator = random.Random(seed)
    for _draw in range(draws):
        permutation = list(range(pairs))
        generator.shuffle(permutation)
        # permutation[position] = pair sitting at that sorted position
        inverse = [0] * pairs
        for position, pair in enumerate(permutation):
            inverse[pair] = position
        keep_retained = 0
        keep_dropped = 0
        for pair in range(pairs):
            gated = valid[pair]
            own = valid[inverse[pair]]
            if gated and not own:
                keep_retained += 1
            elif not gated and own:
                keep_dropped += 1
        retained.append(keep_retained)
        dropped.append(keep_dropped)
    p = sum(1 for item in valid if not item) / pairs
    return {
        "draws": draws,
        "dropped_pairs": pairs - sum(1 for item in valid if item),
        "dropped_fraction": p,
        "retained_overflow_mean": sum(retained) / len(retained),
        "retained_overflow_min": min(retained),
        "retained_overflow_max": max(retained),
        "silently_dropped_mean": sum(dropped) / len(dropped),
        "misassigned_fraction_mean": (sum(retained) + sum(dropped)) / (pairs * len(retained)),
        "misassigned_fraction_analytic": 2.0 * p * (1.0 - p),
        "misassigned_pairs_bound": [0, 2 * (pairs - round(p * pairs))],
    }


def pilot_arm(log_path: str, k: int, draws: int, seed: int) -> dict:
    samples = 0
    layers = 0
    entries = 0
    entries_with_overflow = 0
    pairs_total = 0
    dropped_total = 0
    worst_fraction = 0.0
    worst_step = None
    per_sample = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            routing = record.get("routing") or {}
            if not routing:
                continue
            samples += 1
            layers = max(layers, len(routing))
            for key in sorted(routing):
                entry = routing[key]
                counts = [int(value) for value in entry["counts"]]
                pairs = sum(counts)
                capacity = reference.capacity(pairs // k, k, len(counts),
                                              modeling.CAPACITY_FACTOR)
                dropped = int(entry["dropped_pairs"])
                entries += 1
                entries_with_overflow += 1 if dropped else 0
                pairs_total += pairs
                dropped_total += dropped
                fraction = dropped / pairs
                if fraction > worst_fraction:
                    worst_fraction = fraction
                    worst_step = {"step": record["step"], "layer": key,
                                  "counts": counts, "capacity": capacity,
                                  "dropped": dropped, "pairs": pairs,
                                  "dropped_fraction": fraction,
                                  "dropped_percent": round(100.0 * fraction, 2)}
                per_sample.append((counts, capacity, record["step"]))
    models = [misassignment(counts, capacity, draws, seed + index)
              for index, (counts, capacity, _step) in enumerate(per_sample)]
    models = [model for model in models if model.get("draws")]
    return {
        "log": os.path.relpath(log_path, REPO),
        "log_sha256": sha256_file(log_path),
        "sampled_forwards": samples,
        "layers_per_sample": layers,
        "layer_samples": entries,
        "layer_samples_with_overflow": entries_with_overflow,
        "pairs": pairs_total,
        "dropped_pairs": dropped_total,
        "dropped_fraction": (dropped_total / pairs_total) if pairs_total else 0.0,
        "dropped_percent": round(100.0 * dropped_total / pairs_total, 3)
        if pairs_total else 0.0,
        "worst_sampled_layer": worst_step,
        "misassignment_model": {
            "draws_per_sample": draws,
            "seed": seed,
            "assumption": "uniform random pairing of token-major pairs to sorted "
                          "positions, conditional on the recorded expert counts",
            "retained_overflow_pairs_mean": round(
                sum(model["retained_overflow_mean"] for model in models) / len(models)
                if models else 0.0, 1),
            "silently_dropped_pairs_mean": round(
                sum(model["silently_dropped_mean"] for model in models) / len(models)
                if models else 0.0, 1),
            "misassigned_fraction_mean": (
                sum(model["misassigned_fraction_mean"] for model in models) / len(models)
                if models else 0.0),
            "misassigned_percent": round(
                100.0 * sum(model["misassigned_fraction_mean"] for model in models)
                / len(models), 3) if models else 0.0,
            "misassigned_fraction_analytic": (
                sum(model["misassigned_fraction_analytic"] for model in models) / len(models)
                if models else 0.0),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t2-dir", default="results/t2")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261_115)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args(argv)

    cases = [toy_case(case) for case in CASES]
    headline = headline_case()
    agreements = {
        "pair_placement_matches_reference": all(
            case["placements"]["pair"]["agrees_with_reference"] for case in cases),
        "legacy_placement_matches_reference_sorted": all(
            case["placements"]["sorted"]["agrees_with_reference"] for case in cases),
        "dispatch_halves_bitwise_equal": all(
            case["dispatch_halves_bitwise_equal"] for case in cases),
        "placements_agree_without_overflow": all(
            case["both_placements_agree"] for case in cases
            if case["capacity_factor"] >= 1.0 or not case["dropped"]),
        "placements_separate_under_overflow": sum(
            1 for case in cases if not case["both_placements_agree"]),
    }

    pilots = {}
    for path in sorted(glob.glob(os.path.join(REPO, args.t2_dir, "*", "train.jsonl"))):
        arm = os.path.basename(os.path.dirname(path))
        if arm in ("D", "M"):  # no routed bank
            continue
        pilots[arm] = pilot_arm(path, args.k, args.draws, args.seed)

    receipt = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "created_utc": utc_now(),
        "verdict": {
            "is_a_defect": True,
            "summary": "the shipped combine applies the sorted-order validity mask "
                       "to token-major positions; with any expert overflow it drops "
                       "pairs that were kept and keeps pairs that overflowed, the "
                       "latter reading a clamped buffer row that can hold another "
                       "token's activation",
            "default_mask_order": modeling.MASK_ORDER_LEGACY,
            "corrected_mask_order": modeling.MASK_ORDER_PAIR,
            "corrected_default": False,
        },
        "toy": {"cases": cases, "agreements": agreements, "headline": headline,
                "reference": "oracle/train/reference_dispatch.py"},
        "pilot": pilots,
        "decode_exposure": {
            "note": "capacity per expert at a single-token decode forward is "
                    "max(1, ceil(1*4/8*1.25)) = 1, so no overflow is reachable; a "
                    "batched prefill of W tokens has capacity "
                    "ceil(W*4/8*1.25) = 0.625*W against a mean of 0.5*W per expert, "
                    "so overflow is reachable there and is not recorded by the "
                    "evaluation receipts written before this audit",
            "single_token_capacity": reference.capacity(1, 4, 8, 1.25),
            "batched_window_capacity_1024": reference.capacity(1024, 4, 8, 1.25),
            "batched_mean_per_expert_1024": 512,
        },
        "commands": {
            "tests": "scripts/court oracle -- python3 -m pytest "
                     "oracle/tests/test_p12_1_dispatch.py -q",
            "audit": "scripts/court oracle -- python3 oracle/eval/dispatch_audit.py "
                     "--out results/t2/dispatch-audit-<stamp>.json",
        },
    }
    write_json_atomic(os.path.join(REPO, args.out), receipt)
    print(json.dumps({
        "verdict": receipt["verdict"]["summary"],
        "agreements": agreements,
        "headline": headline,
        "pilot": {arm: {"dropped_fraction": round(value["dropped_fraction"], 6),
                        "samples": value["sampled_forwards"],
                        "misassigned_fraction_mean": round(
                            value["misassignment_model"]["misassigned_fraction_mean"], 6)}
                  for arm, value in pilots.items()},
        "out": args.out,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
