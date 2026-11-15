#!/usr/bin/env python3
"""Render `results/T2-PILOT.md` from the T2 decision receipt.

Directive 12: no performance, storage, latency, throughput, resource or
verification count is hand-written in a tracked document. This script is the
single writer for the T2 pilot report, and it enforces that in two independent
ways:

1. **The whole document is a render.** `--check` re-renders from the receipt and
   compares the file byte for byte, so a hand-edited table cell fails the check.
   The tables therefore carry no hand-written number at all.
2. **The verdict-bearing numbers are also published as `BEGIN GENERATED`
   regions**, sourced by JSON pointer and checked by `scripts/court
   perf_numbers` through `docs/perf-numbers.json`. That court resolves the same
   numbers through a different code path, so a pointer or formatting bug in this
   script cannot pass unnoticed.

The split is deliberate and stated in the document itself: a markdown table
cannot contain one generated region per cell, so the tables are covered by (1)
and the decision-relevant scalars by (2).

Usage:

  python3 scripts/t2_report.py --write     # write the doc and the manifest
  python3 scripts/t2_report.py --check     # re-render and compare, write nothing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import perf_numbers  # noqa: E402

DOC_REL = "results/T2-PILOT.md"
MANIFEST_REL = "docs/perf-numbers.json"
RECEIPT_GLOB = "results/t2/decision-*.json"
ID_PREFIX = "t2pilot-"
ARMS = ("D0", "D", "M", "X", "H", "H*")
TRAINED = ("D", "M", "X", "H", "H*")
COMPARISONS = (
    ("H*-H", "K15", "`H*` vs `H` — error direction"),
    ("H*-D", "K16", "`H*` vs `D` — hybrid vs dense capacity"),
    ("X-D", "K18", "`X` vs `D` — routed specialists (T1 K13 replication)"),
    ("M-D0", "K17", "`M` vs `D0` — memory path at equal active parameters"),
)


def latest_receipt() -> str:
    matches = sorted(glob.glob(os.path.join(REPO, RECEIPT_GLOB)))
    if not matches:
        raise SystemExit("t2_report: no decision receipt yet; run the pilot first")
    return matches[-1]


class Renderer:
    """Resolves numbers from the receipt and records the published regions."""

    def __init__(self, receipt: dict, receipt_rel: str, doc_rel: str):
        self.receipt = receipt
        self.receipt_rel = receipt_rel
        self.doc_rel = doc_rel
        self.stamp = receipt["stamp"]
        self.revision = receipt["revision"]
        self.blocks: list[dict] = []
        self.published: dict[str, str] = {}

    def value(self, pointer: str, fmt: str) -> str:
        return perf_numbers.render_value(
            perf_numbers.resolve_pointer(self.receipt, pointer, pointer), fmt, pointer)

    def region(self, name: str, pointer: str, fmt: str, label: str) -> str:
        """A published `BEGIN GENERATED` region (checked by the perf-numbers court)."""
        block_id = ID_PREFIX + name
        rendered = self.value(pointer, fmt)
        self.blocks.append({
            "id": block_id,
            "doc": self.doc_rel,
            "sentinel": block_id,
            "source": {"receipt": self.receipt_rel, "pointer": pointer},
            "format": fmt,
            "declared_receipt_stamp": self.stamp,
            "declared_revision": self.revision,
        })
        self.published[name] = rendered
        return "\n".join([
            f"{label}:",
            f"<!-- BEGIN GENERATED: {block_id} -->",
            rendered,
            f"<!-- END GENERATED: {block_id} -->",
        ])


def define_regions(r: Renderer) -> dict[str, str]:
    """The ledger: every number the verdict depends on, as a generated region."""
    out: dict[str, str] = {}
    for label, condition, title in COMPARISONS:
        slug = label.replace("*", "star")
        base = f"/comparisons/{label}"
        out[slug] = "\n".join([
            f"**{title}** (`{condition}`)",
            "",
            r.region(f"{slug}-decisive", f"{base}/decisive_statistic", "{value}",
                     "decisive statistic applied"),
            "",
            r.region(f"{slug}-pc-delta", f"{base}/pc/difference/mean_difference",
                     "{value:+.4f}", "paired PC difference (arm - comparator)"),
            r.region(f"{slug}-pc-lo", f"{base}/pc/difference/ci_low", "{value:+.4f}",
                     "paired PC difference, 95% CI low"),
            r.region(f"{slug}-pc-hi", f"{base}/pc/difference/ci_high", "{value:+.4f}",
                     "paired PC difference, 95% CI high"),
            r.region(f"{slug}-pc-rel", f"{base}/pc/relative_improvement", "{value:+.4f}",
                     "PC relative improvement"),
            r.region(f"{slug}-kl-delta", f"{base}/kl/difference/mean_difference",
                     "{value:+.4f}", "paired KL difference, lower is better"),
            r.region(f"{slug}-kl-lo", f"{base}/kl/difference/ci_low", "{value:+.4f}",
                     "paired KL difference, 95% CI low"),
            r.region(f"{slug}-kl-hi", f"{base}/kl/difference/ci_high", "{value:+.4f}",
                     "paired KL difference, 95% CI high"),
            r.region(f"{slug}-kl-rel", f"{base}/kl/relative_improvement", "{value:+.4f}",
                     "KL relative improvement"),
            r.region(f"{slug}-verdict", f"{base}/verdict", "{value}", "verdict"),
            "",
        ])
    for condition in ("K15", "K16", "K17", "K18", "K19"):
        out[condition] = r.region(condition.lower(), f"/kill_conditions/{condition}/verdict",
                                  "{value}", f"{condition} verdict")
    out["k17-gap"] = "\n".join([
        r.region("k17-gap-fraction", "/kill_conditions/K17/clause_2/fraction",
                 "{value:.3f}", "K17 clause 2, fraction of the D0 -> D gap closed"),
        r.region("k17-gap-numerator", "/kill_conditions/K17/clause_2/numerator",
                 "{value:+.4f}", "K17 clause 2 numerator"),
        r.region("k17-gap-denominator", "/kill_conditions/K17/clause_2/denominator",
                 "{value:+.4f}", "K17 clause 2 denominator"),
    ])
    out["one-seed"] = "\n".join([
        r.region("hh-half-width", "/comparisons/H*-H/one_seed_rule/ci_half_width",
                 "{value:.4f}", "H*-H item-level CI half-width"),
        r.region("hh-twice-half-width",
                 "/comparisons/H*-H/one_seed_rule/twice_half_width", "{value:.4f}",
                 "twice that half-width (the T2.8 gate)"),
        r.region("hh-separable", "/comparisons/H*-H/one_seed_rule/separable_text",
                 "{value}", "the H*-H difference against the gate"),
    ])
    for arm in ARMS:
        slug = arm.replace("*", "star")
        out[f"pc-{slug}"] = "\n".join([
            r.region(f"{slug}-pc", f"/arms/{arm}/pc/mean", "{value:.4f}", f"{arm} PC"),
            r.region(f"{slug}-kl", f"/arms/{arm}/kl_mixed/mean", "{value:.4f}",
                     f"{arm} P1 KL mixed"),
        ])
    return out


def render_document(r: Renderer, regions: dict[str, str]) -> str:
    receipt = r.receipt
    v = r.value
    lines: list[str] = []
    add = lines.append

    add("# T2 pilot — the five-arm error-directed capacity matrix, measured")
    add("")
    add("Status: **measured.** The registration is `docs/EXPERIMENT-CONTRACT.md` T2")
    add("with amendment A8 (budget and execution decisions) and A9 (the P2.4")
    add("instrument decision). Nothing in this document moves a threshold, an arm or")
    add("a kill condition.")
    add("")
    add("This file is generated by `scripts/t2_report.py` from")
    add(f"`{r.receipt_rel}` and must not be edited by hand. Two independent checks")
    add("cover it: `scripts/t2_report.py --check` re-renders the whole document and")
    add("compares it byte for byte, and every number the verdict depends on is also")
    add("published as a `BEGIN GENERATED` region below, which `scripts/court")
    add("perf_numbers` re-resolves through `docs/perf-numbers.json`. A markdown table")
    add("cannot hold one generated region per cell, so the tables are covered by the")
    add("first check and the ledger by the second.")
    add("")
    add("What the pilot ran, exactly: `results/t2/pilot-state.json` (status,")
    add("timestamps, exit codes, per-arm argv), `results/t2/<arm>/run.json` (git SHA,")
    add("config hash, seed, data-order hash, admission record, VRAM peak, tokens/s,")
    add("wall-clock, checkpoint hash) and the court receipts it names.")
    add("")
    add("## 1. Quality on the registered evaluation suite")
    add("")
    add("`PC` is the primary composite, `(HumanEval pass@1 + GSM8K exact match +")
    add("TruthfulQA MC1) / 3`; `P1` is the pooled teacher KL on the fixed mixed")
    add("interleaving, lower is better. Per-domain KL is reported so a domain gain")
    add("cannot hide inside the pooled number. `D0` is the frozen equal-active dense")
    add("reference and is never trained.")
    add("")
    add("<!-- BEGIN T2 TABLE: quality -->")
    add("| arm | PC | HumanEval pass@1 | GSM8K exact match | TruthfulQA MC1 |"
        " P1 KL mixed | KL code | KL general | KL math |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in ARMS:
        base = f"/arms/{arm}"
        add(f"| `{arm}` | {v(base + '/pc/mean', '{value:.4f}')} | "
            f"{v(base + '/per_task/humaneval', '{value:.4f}')} | "
            f"{v(base + '/per_task/gsm8k', '{value:.4f}')} | "
            f"{v(base + '/per_task/truthfulqa', '{value:.4f}')} | "
            f"{v(base + '/kl_mixed/mean', '{value:.4f}')} | "
            f"{v(base + '/per_domain/code/kl', '{value:.4f}')} | "
            f"{v(base + '/per_domain/general/kl', '{value:.4f}')} | "
            f"{v(base + '/per_domain/math/kl', '{value:.4f}')} |")
    add("<!-- END T2 TABLE: quality -->")
    add("")
    add("## 2. Capacity, compute and resources per trained arm")
    add("")
    add("Active and accessible counts are the Rust admission records; tokens,")
    add("wall-clock and VRAM are measured by the training loop; `bytes/token` is the")
    add("**modelled** lower bound on distinct added-module weight bytes read per token")
    add("at the declared bf16 deployment dtype. This instrument counts no DRAM or")
    add("PCIe traffic, so no measured bytes/token exists and none is claimed.")
    add("")
    add("<!-- BEGIN T2 TABLE: resources -->")
    add("| arm | active unique | total accessible | tokens | declared FLOPs/token |"
        " wall-clock (h) | tokens/s | VRAM peak (GiB) | added weight bytes/token"
        " (modelled) | checkpoint bytes |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in TRAINED:
        base = f"/training/{arm}"
        add(f"| `{arm}` | {v(base + '/active_unique_params', '{value:,}')} | "
            f"{v(base + '/total_accessible_params', '{value:,}')} | "
            f"{v(base + '/tokens', '{value:,}')} | "
            f"{v(base + '/declared_total_flops_per_token', '{value:,}')} | "
            f"{v(base + '/wall_hours', '{value:.2f}')} | "
            f"{v(base + '/tokens_per_second', '{value:.0f}')} | "
            f"{v(base + '/vram_peak_gib', '{value:.2f}')} | "
            f"{v(f'/model_resources/{arm}/added_weight_bytes_per_token', '{value:,}')} | "
            f"{v(base + '/checkpoint_bytes', '{value:,}')} |")
    add("<!-- END T2 TABLE: resources -->")
    add("")
    add("Declared FLOPs per token is the admission record's own count, not a")
    add("counter; the model reads only `k = 4` of the eight specialists per layer,")
    add("so the modelled expert read is the k-read and the whole-bank figure is")
    add("reported separately in the receipt.")
    add("")
    add("## 3. The registered decision")
    add("")
    add("`PC` is the primary statistic. T2.9's fixed order makes `P1` decisive for a")
    add("comparison whose paired `PC` interval contains zero; the `decisive` column")
    add("records which statistic was actually applied, not which was preferred.")
    add("Differences are `arm - comparator`, so a positive `PC` difference and a")
    add("negative KL difference both favour the arm. Intervals are paired percentile")
    add("bootstraps: items within each task for `PC`, documents within each domain")
    add("for KL, equal weight per domain.")
    add("")
    add("<!-- BEGIN T2 TABLE: decision -->")
    add("| comparison | kill condition | decisive | PC difference [95% CI] |"
        " PC relative | KL difference [95% CI] | KL relative | verdict |")
    add("| --- | --- | --- | --- | ---: | --- | ---: | --- |")
    for label, condition, title in COMPARISONS:
        base = f"/comparisons/{label}"
        add(f"| {title} | {condition} | `{v(base + '/decisive_statistic', '{value}')}` | "
            f"{v(base + '/pc/difference/mean_difference', '{value:+.4f}')} "
            f"[{v(base + '/pc/difference/ci_low', '{value:+.4f}')}, "
            f"{v(base + '/pc/difference/ci_high', '{value:+.4f}')}] | "
            f"{v(base + '/pc/relative_improvement', '{value:+.4f}')} | "
            f"{v(base + '/kl/difference/mean_difference', '{value:+.4f}')} "
            f"[{v(base + '/kl/difference/ci_low', '{value:+.4f}')}, "
            f"{v(base + '/kl/difference/ci_high', '{value:+.4f}')}] | "
            f"{v(base + '/kl/relative_improvement', '{value:+.4f}')} | "
            f"**{v(base + '/verdict', '{value}')}** |")
    add("<!-- END T2 TABLE: decision -->")
    add("")
    add("### 3.1 The outcome in words")
    add("")
    add(f"Registered outcome: **`{receipt.get('outcome', 'unknown')}`**.")
    add("")
    add("> " + receipt.get("interpretation", "").replace("\n", "\n> "))
    add("")
    add("### 3.2 What was not measured, and what would change the reading")
    add("")
    add("- **One seed per arm.** No seed-level CI is claimed. The replication in")
    add("  A8's arithmetic was not run, so the `H*`-`H` difference is held to T2.8's")
    add("  twice-the-item-level-half-width gate, applied in the ledger below.")
    add("- **The compact training target.** Evaluation reports the same top-16 + tail")
    add("  target the arms trained on. The full-vocabulary comparison is the P2.4")
    add("  instrument check (A9) and is not a second quality endpoint.")
    add("- **Measured bytes/token.** No I/O counter exists on this instrument; every")
    add("  byte figure is modelled and labelled as such.")
    add("- **The deferred arms.** `D_M` (equal-total-capacity dense, K17) and `HL`")
    add("  (hard-label control, A9) are registered and not run.")
    add("- **One GPU, one data order.** Every arm consumed the same token stream in")
    add("  the same order at the same budget, which is what makes the pairing valid.")
    add("")
    add("## 4. Generated ledger — the numbers the verdict depends on")
    add("")
    add("Each value below is a `BEGIN GENERATED` region checked by")
    add("`scripts/court perf_numbers` against the decision receipt. The relative")
    add("margins are the registered ones (3% for K15 and K18, 5% for K16 and K17),")
    add("with the absolute `+0.01` rule where a comparator's `PC` is below 0.02.")
    add("")
    for label, condition, _title in COMPARISONS:
        add(regions[label.replace("*", "star")])
    add("### 4.1 Kill conditions")
    add("")
    for condition in ("K15", "K16", "K17", "K18", "K19"):
        add(regions[condition])
        add("")
    add("")
    add("### 4.2 K17 clause 2 — how much of the `D0 -> D` gap the memory arm closes")
    add("")
    add(regions["k17-gap"])
    add("")
    add("### 4.3 The one-seed gate (T2.8)")
    add("")
    add(regions["one-seed"])
    add("")
    add("### 4.4 The two headline endpoints per arm")
    add("")
    for arm in ARMS:
        add(regions[f"pc-{arm.replace('*', 'star')}"])
        add("")
    add("")
    add("## 5. Reproduce and monitor")
    add("")
    add("```sh")
    add("scripts/run-t2-pilot.sh --dry-run       # the exact registered argv per arm")
    add("scripts/run-t2-pilot.sh --resume        # resume from results/t2/pilot-state.json")
    add("scripts/poll-t2-pilot.sh --follow       # progress, ETA and what remains")
    add("python3 scripts/t2_report.py --check    # prove this document")
    add("scripts/court perf_numbers              # prove the ledger")
    add("```")
    add("")
    return "\n".join(lines)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def merge_manifest(blocks: list[dict], manifest_rel: str) -> None:
    path = os.path.join(REPO, manifest_rel)
    manifest = perf_numbers.load_manifest(path)
    kept = [block for block in manifest["blocks"] if not block["id"].startswith(ID_PREFIX)]
    merged = kept + blocks
    merged.sort(key=lambda block: block["id"])
    manifest["blocks"] = merged
    write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--doc", default=DOC_REL)
    parser.add_argument("--manifest", default=MANIFEST_REL)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    receipt_path = args.receipt or latest_receipt()
    receipt_rel = os.path.relpath(receipt_path, REPO)
    with open(receipt_path, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)

    renderer = Renderer(receipt, receipt_rel, args.doc)
    regions = define_regions(renderer)
    text = render_document(renderer, regions) + "\n"

    if args.check:
        doc_path = os.path.join(REPO, args.doc)
        if not os.path.exists(doc_path):
            print(f"t2_report: {args.doc} does not exist", file=sys.stderr)
            return 1
        with open(doc_path, "r", encoding="utf-8") as handle:
            existing = handle.read()
        if existing != text:
            print(f"t2_report: {args.doc} is stale against {receipt_rel}; "
                  f"run --write", file=sys.stderr)
            return 1
        print(f"t2_report: {args.doc} matches {receipt_rel} "
              f"({len(renderer.blocks)} published region(s))")
        return 0

    if args.write:
        write_text(os.path.join(REPO, args.doc), text)
        merge_manifest(renderer.blocks, args.manifest)
        print(f"t2_report: wrote {args.doc} with {len(renderer.blocks)} published "
              f"region(s) from {receipt_rel}")
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
