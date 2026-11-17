#!/usr/bin/env python3
"""S46: summarise the oracle-gate bounds into published, generated numbers.

The bounds receipt (`scripts/s46_oracle_gate.py --bounds`) prints point values
for three gating upper bounds. This script reads that receipt *and* the forward
receipt it names, re-derives every quantity from the stored per-position rows,
and adds the two things a point value cannot carry:

* **a prompt-clustered interval.** Positions inside one prompt share a
  trajectory and are not independent, so the bootstrap resamples whole prompts
  (documents), not tokens. The gate parameters are held at their fitted values
  inside the bootstrap, so the interval is conditional on the fit; it measures
  sampling error, not refit variance.
* **a gain measured against the free policy.** `recovery_fraction` compares a
  gate with memory-always-on. The decision "is a gate worth building" compares
  it with memory-always-*off*, which costs nothing, so every row also reports
  `all_off - value` with its interval. A gate whose gain interval includes zero
  is no better than disabling the memory.

It also folds in the optional generation receipt: the answer-level accuracy of
each policy on the same selected prompts, paired by prompt against `base`.

Outputs a JSON receipt and, with `--write`, the `report-s46-gate-headroom`
region of `results/REPORT.md`. No torch, no network.

Usage
-----
    scripts/court python -- python3 scripts/s46_headroom_summary.py --write
    scripts/court python -- python3 scripts/s46_headroom_summary.py --check
    scripts/court python -- python3 scripts/s46_headroom_summary.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import s46_oracle_gate as gate  # noqa: E402

SCHEMA = "selvedge/s46-headroom-summary/v1"
REPORT = "results/REPORT.md"
SENTINEL = "report-s46-gate-headroom"
OUT_DIR = "results/stage2"
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 46
SIGNALS = ("ce", "kl")
POLICIES = ("per_token", "class", "base_entropy", "memory_residual_norm",
            "position")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: str) -> dict:
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    full = os.path.join(REPO, path)
    os.makedirs(os.path.dirname(os.path.abspath(full)), exist_ok=True)
    tmp = full + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, full)


STAMPED = re.compile(r"-\d{8}T\d{6}Z\.json$")


def latest(pattern: str) -> str:
    found = sorted(name for name in os.listdir(os.path.join(REPO, OUT_DIR))
                   if name.startswith(pattern) and STAMPED.search(name))
    if not found:
        raise SystemExit(f"s46: no receipt matching {pattern}* in {OUT_DIR}")
    return os.path.join(OUT_DIR, found[-1])


# ---------------------------------------------------------------------------
# the policies, re-applied from the stored rows and fitted parameters


def policy_values(rows: list[dict], signal: str, policy: str,
                  folds: dict) -> list[float]:
    """One value per row: the branch the policy selects for that position."""
    picked = []
    for row in rows:
        if policy == "per_token":
            on = row[f"{signal}_memory"] < row[f"{signal}_base"]
            picked.append(row[f"{signal}_memory"] if on else row[f"{signal}_base"])
            continue
        fold = folds[row["split"]]
        if policy == "class":
            on = row["class"] in (fold.get("on_classes") or [])
        else:
            threshold = fold.get("threshold")
            direction = fold.get("direction")
            value = row[policy]
            if threshold is None or value is None:
                on = False
            elif direction == "high":
                on = value >= threshold
            else:
                on = value <= threshold
        picked.append(row[f"{signal}_memory"] if on else row[f"{signal}_base"])
    return picked


def policy_folds(bounds: dict, signal: str, policy: str) -> dict[str, dict]:
    result = bounds["results"][signal]
    if policy == "class":
        entry = result["class_oracle"]["cross_fitted"]["folds"]
    elif policy == "per_token":
        entry = None
    else:
        entry = result["feature_oracle"][policy]["folds"]
    folds = {}
    for pair, fit in (entry or {}).items():
        fitted_on, eval_on = pair.split("->")
        folds[eval_on] = fit
    return folds


def by_doc(rows: list[dict], values: list[float]) -> dict[int, list[float]]:
    groups: dict[int, list[float]] = {}
    for row, value in zip(rows, values):
        groups.setdefault(row["doc_id"], []).append(value)
    return groups


def bootstrap_gain(rows: list[dict], values: list[float], signal: str,
                   resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> dict:
    """Prompt-clustered percentile interval on (all_off - policy)."""
    chosen = by_doc(rows, values)
    base = by_doc(rows, [row[f"{signal}_base"] for row in rows])
    docs = sorted(chosen)
    if not docs:
        return {"point": None, "ci95": None, "resamples": 0}
    point = (statistics.fmean(v for doc in docs for v in base[doc])
             - statistics.fmean(v for doc in docs for v in chosen[doc]))
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [rng.choice(docs) for _ in docs]
        off = statistics.fmean(v for doc in sample for v in base[doc])
        value = statistics.fmean(v for doc in sample for v in chosen[doc])
        draws.append(off - value)
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"point": point, "ci95": [low, high], "resamples": resamples,
            "clusters": len(docs),
            "reading": "all_off - value; positive means the gate beats simply "
                       "disabling the memory"}


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided McNemar/binomial p on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def summarise(bounds: dict, forward: dict, generation: dict | None) -> dict:
    items = [item for item in forward["items"] if not item.get("control")]
    rows = gate._flatten(items, damage_only=True)
    receipt = {
        "schema": SCHEMA,
        "kind": "oracle_gate_headroom_summary",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": gate.revision(),
        "inputs": {
            "bounds": {"path": bounds.get("_path"),
                       "stamp": bounds.get("stamp"),
                       "sha256": gate.s45.sha256_file(
                           os.path.join(REPO, bounds["_path"]))},
            "forward": {"path": bounds["input"]["path"],
                        "stamp": bounds["input"].get("stamp"),
                        "sha256": bounds["input"].get("sha256")},
        },
        "generation": None,
        "sample": forward.get("sample"),
        "sample_size": forward.get("sample_size"),
        "positions": len(rows),
        "prompts": len({row["doc_id"] for row in rows}),
        "method": {
            "interval": "prompt-clustered percentile bootstrap over documents",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "conditional_on": "the gate threshold/direction and class set are "
                              "held at the values fitted on the complementary "
                              "fold; the interval does not include refit variance",
            "gain": "all_off - value, so a positive number is a real "
                    "improvement over memory-always-off",
            "per_token_is_non_causal": "the per-token oracle chooses the branch "
                                       "with the target token visible; it is an "
                                       "upper bound, not a usable gate",
            "class_oracle_is_non_causal": "the class labels are those of the "
                                          "token being predicted, so the class "
                                          "bound is looser than any causal class "
                                          "gate; the generation receipt's class "
                                          "policy is the causal one",
            "not_the_benchmark": "selected damage sample (base correct, "
                                 "memory-2m wrong); not an anchored score",
        },
        "results": {},
    }
    if generation is not None:
        receipt["generation"] = generation
    for signal in SIGNALS:
        block = {}
        off = [row[f"{signal}_base"] for row in rows]
        on = [row[f"{signal}_memory"] for row in rows]
        block["level"] = {
            "all_off": statistics.fmean(off),
            "all_on": statistics.fmean(on),
            "damage": statistics.fmean(on) - statistics.fmean(off),
            "positions_memory_helps": sum(
                1 for row in rows
                if row[f"{signal}_memory"] < row[f"{signal}_base"]),
            "positions": len(rows),
        }
        per_fold = {}
        for split in sorted({row["split"] for row in rows}):
            subset = [row for row in rows if row["split"] == split]
            per_fold[split] = {
                "positions": len(subset),
                "all_off": statistics.fmean(row[f"{signal}_base"]
                                            for row in subset),
                "all_on": statistics.fmean(row[f"{signal}_memory"]
                                           for row in subset),
            }
        block["per_fold_level"] = per_fold
        block["policies"] = {}
        for policy in POLICIES:
            folds = policy_folds(bounds, signal, policy)
            values = policy_values(rows, signal, policy, folds)
            value = statistics.fmean(values)
            fold_rows = {}
            for split in sorted({row["split"] for row in rows}):
                indices = [index for index, row in enumerate(rows)
                           if row["split"] == split]
                if not indices:
                    continue
                stack = [values[index] for index in indices]
                fit = folds.get(split) or {}
                fold_rows[split] = {
                    "positions": len(indices),
                    "value": statistics.fmean(stack),
                    "all_off": per_fold[split]["all_off"],
                    "all_on": per_fold[split]["all_on"],
                    "gain_vs_all_off": per_fold[split]["all_off"]
                                       - statistics.fmean(stack),
                    "on_classes": fit.get("on_classes"),
                    "direction": fit.get("direction"),
                    "threshold": fit.get("threshold"),
                    "gate_on_fraction": fit.get("gate_on_fraction"),
                }
            block["policies"][policy] = {
                "value": value,
                "recovered_nats_vs_all_on": block["level"]["all_on"] - value,
                "gain_vs_all_off": block["level"]["all_off"] - value,
                "recovery_fraction": (
                    None if block["level"]["damage"] <= 0
                    else (block["level"]["all_on"] - value)
                         / block["level"]["damage"]),
                "bootstrap": bootstrap_gain(rows, values, signal),
                "folds": fold_rows,
            }
        receipt["results"][signal] = block
    if generation is not None:
        receipt["answer_level"] = generation["answer_level"]
    return receipt


# ---------------------------------------------------------------------------
# rendering


def _num(value, spec=".4f", scale=1.0):
    return "n/a" if value is None else format(scale * value, spec)


def _gain(entry: dict) -> str:
    point = entry["bootstrap"]["point"]
    low, high = entry["bootstrap"]["ci95"]
    return (f"{1000 * point:+.2f} [{1000 * low:+.2f}, {1000 * high:+.2f}]")


def render(receipt: dict) -> str:
    ce = receipt["results"]["ce"]
    kl = receipt["results"]["kl"]
    cheap = POLICIES[1:]
    best_upper = max(ce["policies"][name]["bootstrap"]["ci95"][1]
                     for name in cheap)
    beats = [name for name in cheap
             if ce["policies"][name]["bootstrap"]["ci95"][0] > 0]
    loses = [name for name in cheap
             if ce["policies"][name]["bootstrap"]["ci95"][1] < 0]
    level = receipt["sample_size"]
    lines = [
        "### The gate headroom is real, and no cheap one-threshold gate reaches it",
        "",
        "Read this as a **bound**, not a score. The sample is "
        f"{level['damage']} GSM8K prompts where the base is correct and "
        "memory-2m is wrong, plus "
        f"{level['controls']} control prompts that are excluded here; the "
        "scored text is the arm's own stored generation, "
        f"{receipt['positions']} completion positions in total. It is a "
        "selected damage sample, so every number below describes *where the "
        "memory hurts*, not a benchmark.",
        "",
        "| signal | all off | all on | damage | per-token oracle | gain vs off "
        "(m nats/token, 95% CI) | class gate (held-out) | entropy gate "
        "(held-out) | residual-norm gate | position gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in (("CE", ce), ("KL", kl)):
        policies = block["policies"]
        lines.append(
            f"| {name} | {_num(block['level']['all_off'])} | "
            f"{_num(block['level']['all_on'])} | "
            f"{_num(block['level']['damage'])} | "
            f"{_num(policies['per_token']['value'])} | "
            f"{_gain(policies['per_token'])} | "
            f"{_num(policies['class']['value'])} | "
            f"{_num(policies['base_entropy']['value'])} | "
            f"{_num(policies['memory_residual_norm']['value'])} | "
            f"{_num(policies['position']['value'])} |")
    lines += [
        "",
        "All cells are generated by `scripts/s46_headroom_summary.py` from "
        f"`{receipt['inputs']['bounds']['path']}` (stamp "
        f"`{receipt['inputs']['bounds']['stamp']}`) and the forward receipt it "
        f"names (stamp `{receipt['inputs']['forward']['stamp']}`). The gain "
        "column is the prompt-clustered bootstrap mean of "
        "`all_off - value` in millinats per token; the other policy columns are "
        "their held-out values (a class set or threshold fitted on one half of "
        "the prompts, applied to the other half, pooled).",
        "",
        "**What the bound says.** On this damage sample memory-always-on costs "
        f"{_num(ce['level']['damage'])} nats CE per token over memory-always-off. "
        "A per-token oracle, which sees the target token and may choose the "
        "branch position by position, reaches "
        f"{_num(ce['policies']['per_token']['value'])} nats -- "
        f"{_num(ce['policies']['per_token']['gain_vs_all_off'], '+.4f')} "
        f"better than switching the memory off entirely "
        f"({_num(ce['policies']['per_token']['recovery_fraction'], '.2f')}x "
        "the net damage). That is the headroom: a gate worth its inference cost "
        "has to capture part of this improvement over the *free* policy, not "
        "merely reproduce it.",
        "",
        "**What the cheap gates actually do.** Every held-out gate here lands "
        f"within {_num(max(abs(ce['policies'][name]['value'] - ce['level']['all_off']) for name in POLICIES[1:]))} "
        f"nats of all-off ({_num(ce['level']['all_off'])}): class "
        f"{_num(ce['policies']['class']['value'])}, entropy "
        f"{_num(ce['policies']['base_entropy']['value'])}, residual norm "
        f"{_num(ce['policies']['memory_residual_norm']['value'])}, position "
        f"{_num(ce['policies']['position']['value'])}. Their gains versus "
        "all-off are "
        + ", ".join(
            f"{name} {1000 * ce['policies'][name]['gain_vs_all_off']:+.2f} "
            f"[{1000 * ce['policies'][name]['bootstrap']['ci95'][0]:+.2f}, "
            f"{1000 * ce['policies'][name]['bootstrap']['ci95'][1]:+.2f}]"
            for name in POLICIES[1:])
        + f" millinats per token. The best upper bound of the four is "
        f"{1000 * best_upper:+.2f}, against the per-token oracle's lower bound of "
        f"{1000 * ce['policies']['per_token']['bootstrap']['ci95'][0]:+.2f}. "
        + ("None of the four intervals lies above zero. "
           if not beats else
           "Above zero: " + ", ".join(beats) + ". ")
        + ("None lies entirely below zero either." if not loses else
           ("Entirely below zero: " + ", ".join(loses) +
            ("." if len(loses) > 1 else ".")))
        + " In words: a one-threshold gate on a single cheap feature "
        "recovers the damage by turning the memory off and captures essentially "
        "none of the per-token oracle's gain. The class gate is also unstable "
        "across folds: fitted on A it turns on "
        f"{', '.join(ce['policies']['class']['folds']['B']['on_classes'] or ['nothing'])} "
        f"and reaches {_num(ce['policies']['class']['folds']['B']['value'])} on "
        f"fold B's {ce['policies']['class']['folds']['B']['positions']} "
        f"positions, against {_num(ce['policies']['class']['folds']['B']['all_on'])} "
        f"for memory-always-on and {_num(ce['policies']['class']['folds']['B']['all_off'])} "
        f"for memory-off; fitted on B it turns on "
        f"{', '.join(ce['policies']['class']['folds']['A']['on_classes'] or ['nothing'])} "
        f"and reaches {_num(ce['policies']['class']['folds']['A']['value'])} on "
        f"fold A's {ce['policies']['class']['folds']['A']['positions']} positions. "
        "On both folds the gate lands within "
        f"{_num(max(abs(ce['policies']['class']['folds'][split]['value'] - ce['policies']['class']['folds'][split]['all_off']) for split in ('A', 'B')))} "
        f"of memory-off ({_num(ce['policies']['class']['folds']['A']['value'])} "
        f"against {_num(ce['policies']['class']['folds']['A']['all_off'])} on A; "
        f"{_num(ce['policies']['class']['folds']['B']['value'])} against "
        f"{_num(ce['policies']['class']['folds']['B']['all_off'])} on B), so the "
        "pooled number is the 'turn it off' policy, not a gate.",
        "",
        "**Reading for the next mechanism.** The headroom is not in the class or "
        "in a single feature threshold; it is in the per-position choice, and "
        "the per-token bound is non-causal. A learned gate therefore has to be "
        "trained to predict the *sign* of the residual's local effect from "
        "causally available state, and it has to beat all-off on held-out "
        "prompts before it can be called a mechanism. KL tells the same story "
        f"({_num(kl['level']['damage'])} nats damage; oracle gain "
        f"{_num(kl['policies']['per_token']['gain_vs_all_off'], '+.4f')}), so "
        "the conclusion does not depend on the signal.",
    ]
    if receipt.get("answer_level"):
        answer = receipt["answer_level"]
        lines += [
            "",
            "**Answer level (selected prompts, greedy decode).** "
            f"{answer['prompts']} damage prompts, {answer['max_new_tokens']} "
            "new tokens at most, one greedy decode per policy, graded by the "
            "RFT GSM8K verifier against the reference answer: "
            + ", ".join(
                f"`{name}` {entry['correct']}/{entry['n']}"
                for name, entry in answer["policies"].items())
            + ". Paired against `base` by prompt: "
            + "; ".join(
                f"`{name}` +{entry['wins_vs_base']} wins / "
                f"{entry['losses_vs_base']} losses "
                f"(exact p {entry['p_value']:.3g})"
                for name, entry in answer["policies"].items()
                if name != "base")
            + ". This is a small, selected sample and its interval is wide; it "
            "is reported because it is the only answer-level check, not because "
            "it decides the mechanism.",
        ]
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# regions


def _scan(text: str, where: str):
    begin = re.compile(r"^\s*<!--\s*BEGIN GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    end = re.compile(r"^\s*<!--\s*END GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    lines = text.splitlines()
    regions = {}
    open_id = None
    start = None
    for index, line in enumerate(lines):
        match = begin.match(line)
        if match:
            if open_id is not None:
                raise ValueError(f"{where}:{index + 1}: nested BEGIN")
            open_id, start = match.group(1), index
            continue
        match = end.match(line)
        if match:
            if open_id != match.group(1):
                raise ValueError(f"{where}:{index + 1}: mismatched END")
            regions[open_id] = (start, index)
            open_id = None
    if open_id is not None:
        raise ValueError(f"{where}: BEGIN {open_id} never closed")
    return lines, regions


def _body(lines: list[str], span) -> str:
    return "\n".join(lines[span[0] + 1:span[1]])


def write_region(receipt: dict) -> None:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    if SENTINEL not in regions:
        raise SystemExit(f"s46: {REPORT} has no BEGIN GENERATED: {SENTINEL}")
    span = regions[SENTINEL]
    body = render(receipt)
    lines = lines[:span[0] + 1] + body.splitlines() + lines[span[1]:]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def check_region(receipt: dict) -> int:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    if SENTINEL not in regions:
        print(f"s46: {REPORT}: missing region {SENTINEL}", file=sys.stderr)
        return 1
    if _body(lines, regions[SENTINEL]) != render(receipt):
        print(f"s46: {REPORT}: region {SENTINEL} is stale for the receipt",
              file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# answer-level fold-in


def answer_summary(generation: dict) -> dict:
    policies = generation["policies"]
    base_docs = {row["doc_id"]: row for row in policies["base"]["items"]
                 if "verdict" in row}

    def correct(row) -> bool:
        return row.get("verdict") == "correct"

    folded = {}
    for name, entry in policies.items():
        rows = [row for row in entry["items"] if "verdict" in row]
        wins = losses = 0
        for row in rows:
            other = base_docs.get(row["doc_id"])
            if other is None:
                continue
            if correct(row) and not correct(other):
                wins += 1
            elif correct(other) and not correct(row):
                losses += 1
        folded[name] = {
            "n": len(rows),
            "correct": sum(1 for row in rows if correct(row)),
            "wins_vs_base": wins,
            "losses_vs_base": losses,
            "p_value": sign_test(wins, losses),
            "gate_on_fraction_mean": entry.get("gate_on_fraction_mean"),
        }
    return {
        "prompts": generation["decode"]["prompts"],
        "max_new_tokens": generation["decode"]["max_new_tokens"],
        "decode_tokens": generation.get("decode_tokens"),
        "ms_per_decode_token": generation.get("ms_per_decode_token"),
        "policies": folded,
    }


def self_test() -> int:
    assert sign_test(0, 0) == 1.0
    assert abs(sign_test(5, 5) - 1.0) < 1e-12
    assert sign_test(6, 0) < 0.05
    rows = [
        {"doc_id": 1, "split": "A", "class": "digit", "position": 0,
         "ce_base": 1.0, "ce_memory": 0.5, "kl_base": 1.0, "kl_memory": 0.5,
         "base_entropy": 1.0, "memory_residual_norm": 1.0},
        {"doc_id": 2, "split": "B", "class": "letter", "position": 0,
         "ce_base": 1.0, "ce_memory": 2.0, "kl_base": 1.0, "kl_memory": 2.0,
         "base_entropy": 2.0, "memory_residual_norm": 2.0},
    ]
    values = policy_values(rows, "ce", "per_token", {})
    assert values == [0.5, 1.0]
    folds = {"A": {}, "B": {}}
    assert policy_values(rows, "ce", "class", folds) == [1.0, 1.0]
    folds = {"A": {"on_classes": ["digit"]}, "B": {"on_classes": ["digit"]}}
    assert policy_values(rows, "ce", "class", folds) == [0.5, 1.0]
    folds = {"A": {"threshold": 1.5, "direction": "high"},
             "B": {"threshold": 1.5, "direction": "high"}}
    assert policy_values(rows, "ce", "base_entropy", folds) == [1.0, 2.0]
    first = bootstrap_gain(rows, values, "ce", resamples=50, seed=1)
    second = bootstrap_gain(rows, values, "ce", resamples=50, seed=1)
    assert first == second, "the bootstrap is not deterministic"
    assert abs(first["point"] - (1.0 - 0.75)) < 1e-12
    print("self-test: sign test, policy re-application and bootstrap PASS")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="write the receipt and print the region body; "
                             "do not touch or check the document")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--bounds", default=None)
    parser.add_argument("--generation", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if sum(bool(flag) for flag in (args.write, args.check, args.summary)) != 1:
        raise SystemExit("choose exactly one of --write, --check, --summary")
    bounds_path = args.bounds or latest("s46-oracle-bounds-")
    bounds = read_json(bounds_path)
    bounds["_path"] = bounds_path
    forward = read_json(bounds["input"]["path"])
    generation = None
    generation_path = args.generation
    if generation_path is None:
        try:
            generation_path = latest("s46-oracle-generation-")
        except SystemExit:
            generation_path = None
    if generation_path:
        payload = read_json(generation_path)
        generation = {
            "path": generation_path,
            "stamp": payload.get("stamp"),
            "answer_level": answer_summary(payload),
        }
    summary = summarise(bounds, forward, generation)
    out = args.out or os.path.join(OUT_DIR, f"s46-headroom-summary-{stamp()}.json")
    write_json(out, summary)
    print(f"s46: receipt {out}")
    if args.summary:
        print(render(summary))
        return 0
    if args.write:
        write_region(summary)
        print(f"s46: wrote {REPORT}:{SENTINEL}")
        return 0
    return check_region(summary)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
