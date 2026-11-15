#!/usr/bin/env python3
"""Render `results/T2-PILOT.md` and its manifest blocks from the T2 receipts.

Directive 12: no performance, storage, latency, throughput, resource or
verification count is hand-written in a tracked document. This script is the
writer for the T2 pilot report: every number in the rendered document is a
generated region sourced by JSON pointer from

  * `results/t2/decision-<stamp>.json` — the decision receipt, which carries the
    per-arm quality endpoints, the paired comparisons, the kill-condition
    verdicts, the training measurements (copied from each arm's `run.json`) and
    the bytes/token arithmetic; and
  * the per-arm `results/t2/<arm>/run.json` and `eval.json` receipts, which the
    decision receipt embeds.

`scripts/perf_numbers.py --check` independently re-resolves every block and
fails closed on a stale or mismatched region, so this script and the court
cannot drift.

Usage:

  python3 scripts/t2_report.py --write     # write the doc and the manifest
  python3 scripts/t2_report.py --check     # resolve and compare, write nothing
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
    ("H*-H", "K15", "H\\* vs H (error direction)"),
    ("H*-D", "K16", "H\\* vs D (hybrid vs dense capacity)"),
    ("X-D", "K18", "X vs D (routed specialists, T1 K13 replication)"),
    ("M-D0", "K17", "M vs D0 (memory path, equal active parameters)"),
)


def latest_receipt() -> str:
    matches = sorted(glob.glob(os.path.join(REPO, RECEIPT_GLOB)))
    if not matches:
        raise SystemExit("t2_report: no decision receipt yet; run the pilot first")
    return matches[-1]


class Blocks:
    def __init__(self, receipt: dict, receipt_rel: str):
        self.receipt = receipt
        self.receipt_rel = receipt_rel
        self.stamp = receipt["stamp"]
        self.revision = receipt["revision"]
        self.blocks: list[dict] = []
        self.values: dict[str, str] = {}

    def render(self, name: str, pointer: str, fmt: str) -> str:
        block_id = ID_PREFIX + name
        value = perf_numbers.resolve_pointer(self.receipt, pointer, block_id)
        rendered = perf_numbers.render_value(value, fmt, block_id)
        self.blocks.append({
            "id": block_id,
            "doc": DOC_REL,
            "sentinel": block_id,
            "source": {"receipt": self.receipt_rel, "pointer": pointer},
            "format": fmt,
            "declared_receipt_stamp": self.stamp,
            "declared_revision": self.revision,
        })
        self.values[name] = rendered
        return rendered

    def get(self, name: str) -> str:
        return self.values[name]


def define(blocks: Blocks) -> None:
    for arm in ARMS:
        slug = arm.replace("*", "star")
        base = f"/arms/{arm}"
        blocks.render(f"{slug}-pc", f"{base}/pc/mean", "{value:.4f}")
        blocks.render(f"{slug}-he", f"{base}/per_task/humaneval", "{value:.4f}")
        blocks.render(f"{slug}-gsm8k", f"{base}/per_task/gsm8k", "{value:.4f}")
        blocks.render(f"{slug}-mc1", f"{base}/per_task/truthfulqa", "{value:.4f}")
        blocks.render(f"{slug}-kl", f"{base}/kl_mixed/mean", "{value:.4f}")
        for domain in ("code", "general", "math"):
            blocks.render(f"{slug}-kl-{domain}",
                          f"{base}/per_domain/{domain}/kl", "{value:.4f}")
    for arm in TRAINED:
        slug = arm.replace("*", "star")
        base = f"/training/{arm}"
        blocks.render(f"{slug}-active", f"{base}/active_unique_params", "{value:,}")
        blocks.render(f"{slug}-accessible", f"{base}/total_accessible_params", "{value:,}")
        blocks.render(f"{slug}-tokens", f"{base}/tokens", "{value:,}")
        blocks.render(f"{slug}-flops-token",
                      f"{base}/declared_total_flops_per_token", "{value:,}")
        blocks.render(f"{slug}-flops-trained", f"{base}/declared_flops_trained", "{value:.4e}")
        blocks.render(f"{slug}-wall-hours", f"{base}/wall_hours", "{value:.2f}")
        blocks.render(f"{slug}-tok-s", f"{base}/tokens_per_second", "{value:.0f}")
        blocks.render(f"{slug}-vram", f"{base}/vram_peak_gib", "{value:.2f}")
        blocks.render(f"{slug}-ckpt-bytes", f"{base}/checkpoint_bytes", "{value:,}")
        blocks.render(f"{slug}-bytes-token",
                      f"/model_resources/{arm}/added_weight_bytes_per_token", "{value:,}")
        blocks.render(f"{slug}-mem-table-bf16",
                      f"/model_resources/{arm}/memory_table_resident_bytes_bf16", "{value:,}")
    for label, condition, _title in COMPARISONS:
        slug = label.replace("*", "star")
        base = f"/comparisons/{label}"
        blocks.render(f"{slug}-decisive", f"{base}/decisive_statistic", "{value}")
        blocks.render(f"{slug}-pc-delta",
                      f"{base}/pc/difference/mean_difference", "{value:+.4f}")
        blocks.render(f"{slug}-pc-lo", f"{base}/pc/difference/ci_low", "{value:+.4f}")
        blocks.render(f"{slug}-pc-hi", f"{base}/pc/difference/ci_high", "{value:+.4f}")
        blocks.render(f"{slug}-pc-rel", f"{base}/pc/relative_improvement", "{value:+.4f}")
        blocks.render(f"{slug}-kl-delta",
                      f"{base}/kl/difference/mean_difference", "{value:+.4f}")
        blocks.render(f"{slug}-kl-lo", f"{base}/kl/difference/ci_low", "{value:+.4f}")
        blocks.render(f"{slug}-kl-hi", f"{base}/kl/difference/ci_high", "{value:+.4f}")
        blocks.render(f"{slug}-kl-rel", f"{base}/kl/relative_improvement", "{value:+.4f}")
        blocks.render(f"{slug}-verdict", f"{base}/verdict", "{value}")
    for condition in ("K15", "K16", "K17", "K18", "K19"):
        blocks.render(f"{condition.lower()}-verdict",
                      f"/kill_conditions/{condition}/verdict", "{value}")
    blocks.render("k17-gap-fraction",
                  "/kill_conditions/K17/clause_2/fraction", "{value:.3f}")
    blocks.render("k17-gap-numerator",
                  "/kill_conditions/K17/clause_2/numerator", "{value:+.4f}")
    blocks.render("k17-gap-denominator",
                  "/kill_conditions/K17/clause_2/denominator", "{value:+.4f}")
    blocks.render("hh-half-width",
                  "/comparisons/H*-H/one_seed_rule/ci_half_width", "{value:.4f}")
    blocks.render("hh-twice-half-width",
                  "/comparisons/H*-H/one_seed_rule/twice_half_width", "{value:.4f}")
    blocks.render("hh-separable",
                  "/comparisons/H*-H/one_seed_rule/separable_text", "{value}")


def render_document(blocks: Blocks) -> str:
    receipt = blocks.receipt
    get = blocks.get
    lines: list[str] = []
    add = lines.append

    add("# T2 pilot — the five-arm error-directed capacity matrix, measured")
    add("")
    add("Status: **measured, not yet reviewed.** Every number in this document is a")
    add("generated region: `scripts/t2_report.py --write` renders it from the decision")
    add("receipt, and `scripts/court perf_numbers` fails closed if any region is stale")
    add("or no longer matches its source. The registration is")
    add("`docs/EXPERIMENT-CONTRACT.md` T2 with amendments A8 (budget and execution")
    add("decisions) and A9 (the P2.4 instrument decision); nothing here changes a")
    add("threshold, an arm or a kill condition.")
    add("")
    add(f"Pilot state, exact argv per arm, exit codes and the run receipts are in")
    add(f"`results/t2/pilot-state.json` and `results/t2/<arm>/run.json`; the decision")
    add(f"receipt behind this document is `{blocks.receipt_rel}`.")
    add("")
    add("## 1. Quality on the registered evaluation suite")
    add("")
    add("`PC` is the primary composite, `(HumanEval pass@1 + GSM8K exact match +")
    add("TruthfulQA MC1) / 3`; `P1` is the pooled teacher KL on the fixed mixed")
    add("interleaving (lower is better). Per-domain KL is reported so a domain gain")
    add("cannot hide inside the pooled number. `D0` is the frozen equal-active dense")
    add("reference; it is never trained.")
    add("")
    add("| arm | PC | HumanEval pass@1 | GSM8K exact match | TruthfulQA MC1 | P1 KL mixed | KL code | KL general | KL math |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in ARMS:
        slug = arm.replace("*", "star")
        add(f"| `{arm}` | {get(slug + '-pc')} | {get(slug + '-he')} | "
            f"{get(slug + '-gsm8k')} | {get(slug + '-mc1')} | {get(slug + '-kl')} | "
            f"{get(slug + '-kl-code')} | {get(slug + '-kl-general')} | "
            f"{get(slug + '-kl-math')} |")
    add("")
    add("## 2. Capacity, compute and resources per trained arm")
    add("")
    add("Active and accessible counts are the Rust admission records; tokens, FLOPs")
    add("and wall-clock are measured by the training loop; `bytes/token` is the")
    add("**modelled** lower bound on distinct added-module weight bytes read per token")
    add("at the declared bf16 deployment dtype. This instrument counts no DRAM or PCIe")
    add("traffic, so no measured bytes/token exists and none is reported.")
    add("")
    add("| arm | active unique | total accessible | tokens | declared FLOPs/token | FLOPs trained | wall-clock (h) | tokens/s | VRAM peak (GiB) | added weight bytes/token (modelled) | checkpoint bytes |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in TRAINED:
        slug = arm.replace("*", "star")
        add(f"| `{arm}` | {get(slug + '-active')} | {get(slug + '-accessible')} | "
            f"{get(slug + '-tokens')} | {get(slug + '-flops-token')} | "
            f"{get(slug + '-flops-trained')} | {get(slug + '-wall-hours')} | "
            f"{get(slug + '-tok-s')} | {get(slug + '-vram')} | "
            f"{get(slug + '-bytes-token')} | {get(slug + '-ckpt-bytes')} |")
    add("")
    add("## 3. The registered decision")
    add("")
    add("The primary statistic is `PC`. T2.9's fixed order makes `P1` decisive for a")
    add("comparison whose paired `PC` interval contains zero; the `decisive` column")
    add("records which statistic was actually applied. Differences are")
    add("`arm - comparator`, so a positive `PC` difference and a negative KL")
    add("difference both favour the arm. Intervals are paired percentile bootstraps")
    add("with 10,000 resamples at seed `20261115`.")
    add("")
    add("| comparison | kill condition | decisive | PC difference [95% CI] | PC relative | KL difference [95% CI] | KL relative | verdict |")
    add("| --- | --- | --- | --- | ---: | --- | ---: | --- |")
    for label, condition, title in COMPARISONS:
        slug = label.replace("*", "star")
        add(f"| {title} | {condition} | `{get(slug + '-decisive')}` | "
            f"{get(slug + '-pc-delta')} [{get(slug + '-pc-lo')}, {get(slug + '-pc-hi')}] | "
            f"{get(slug + '-pc-rel')} | "
            f"{get(slug + '-kl-delta')} [{get(slug + '-kl-lo')}, {get(slug + '-kl-hi')}] | "
            f"{get(slug + '-kl-rel')} | **{get(slug + '-verdict')}** |")
    add("")
    add("Kill conditions, applied exactly as registered:")
    add("")
    add(f"- **K15 (error direction)** — {get('k15-verdict')}.")
    add(f"- **K16 (hybrid vs dense capacity)** — {get('k16-verdict')}.")
    add(f"- **K17 (memory path)** — {get('k17-verdict')}. The `D0 -> D` gap closure is")
    add(f"  {get('k17-gap-fraction')} = {get('k17-gap-numerator')} / "
        f"{get('k17-gap-denominator')} on the decisive statistic.")
    add(f"- **K18 (expert path)** — {get('k18-verdict')}.")
    add(f"- **K19 (aggregate scaling gate)** — {get('k19-verdict')}.")
    add("")
    add("The one-seed rule (T2.8): the `H* - H` difference must exceed twice the")
    add(f"item-level CI half-width to count. The half-width is {get('hh-half-width')},")
    add(f"twice it is {get('hh-twice-half-width')}, and `separable_from_noise` is")
    add(f"{get('hh-separable')}.")
    add("")
    add("### 3.1 What the result means")
    add("")
    outcome = receipt.get("outcome", "unknown")
    add(f"Registered outcome: **`{outcome}`**.")
    add("")
    add("> " + receipt.get("interpretation", "").replace("\n", "\n> "))
    add("")
    add("### 3.2 What was not measured")
    add("")
    add("- **One seed.** No seed-level CI is claimed; the replication in A8's")
    add("  arithmetic was not run inside this pilot's envelope.")
    add("- **The compact training target.** The evaluation reports the same top-16 +")
    add("  tail target the arms trained on; the full-vocabulary teacher comparison is")
    add("  the P2.4 instrument check (A9), not a second quality endpoint.")
    add("- **Measured bytes/token.** No I/O counter exists on this instrument; every")
    add("  byte figure is modelled and labelled `modelled`.")
    add("- **The deferred arms.** `D_M` (equal-total-capacity dense, K17) and `HL`")
    add("  (hard-label control, A9) are registered and not run.")
    add("")
    add("## 4. Reproduce and monitor")
    add("")
    add("```sh")
    add("scripts/run-t2-pilot.sh --dry-run       # the exact registered argv per arm")
    add("scripts/run-t2-pilot.sh --resume        # resume from results/t2/pilot-state.json")
    add("scripts/poll-t2-pilot.sh --follow       # progress, ETA and what remains")
    add("python3 scripts/t2_report.py --write    # re-render this document")
    add("scripts/court perf_numbers              # prove every number here")
    add("```")
    add("")
    return "\n".join(lines)


def write_document(text: str) -> None:
    path = os.path.join(REPO, DOC_REL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def merge_manifest(blocks: Blocks) -> None:
    path = os.path.join(REPO, MANIFEST_REL)
    manifest = perf_numbers.load_manifest(path)
    kept = [block for block in manifest["blocks"]
            if not block["id"].startswith(ID_PREFIX)]
    merged = kept + blocks.blocks
    merged.sort(key=lambda block: block["id"])
    manifest["blocks"] = merged
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    receipt_path = args.receipt or latest_receipt()
    receipt_rel = os.path.relpath(receipt_path, REPO)
    with open(receipt_path, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)

    blocks = Blocks(receipt, receipt_rel)
    define(blocks)
    text = render_document(blocks)

    if args.check:
        doc_path = os.path.join(REPO, DOC_REL)
        if not os.path.exists(doc_path):
            print(f"t2_report: {DOC_REL} does not exist", file=sys.stderr)
            return 1
        with open(doc_path, "r", encoding="utf-8") as handle:
            existing = handle.read()
        expected = text + "\n"
        if existing != expected:
            print("t2_report: the document is stale; run --write", file=sys.stderr)
            return 1
        print(f"t2_report: {DOC_REL} matches {receipt_rel} "
              f"({len(blocks.blocks)} generated regions)")
        return 0

    if args.write:
        write_document(text)
        merge_manifest(blocks)
        print(f"t2_report: wrote {DOC_REL} and {len(blocks.blocks)} manifest block(s) "
              f"from {receipt_rel}")
        return 0

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
