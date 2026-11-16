#!/usr/bin/env python3
"""Render the WS19 anchor scoreboard tables, and refuse a document that drifts.

AGENTS.md directive 12: no scoreboard number is typed into a tracked document.
The tables below are rendered from receipts and nothing else:

    results/frontier/anchor-0.8b-<stamp>.json   the anchor run  (S36)
    results/frontier/anchor-2b-<stamp>.json     the gap upper end (S36)
    results/frontier/anchor-gap-<stamp>.json    per-task and per-family gaps

Three regions are owned here and declared in `docs/perf-numbers.json` under
`external_regions`, so the perf-numbers court refuses a generated region nobody
owns and refuses a region this script does not print:

    docs/FRONTIER-TARGET.md   frontier-anchor-scoreboard, frontier-anchor-gap
    results/REPORT.md         report-anchor-scoreboard

`--check` fails closed when a block body differs from what the receipts render
now, so a score cannot be edited by hand and a new receipt cannot leave the
table stale.

The upper-end columns are rendered as `pending` when no 2B run receipt exists.
That is a statement about the receipts on disk, not a placeholder: when the 2B
receipt lands, `--check` fails until the table is regenerated.

Usage:
    scripts/court python scripts/s36_scoreboard.py --check
    python3 scripts/s36_scoreboard.py --write
    python3 scripts/s36_scoreboard.py --list
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import perf_numbers as pn  # noqa: E402  (sentinel parsing is shared, never copied)

PREREG_REL = "docs/anchor-preregistration-v1.json"

REGIONS = (
    ("frontier-anchor-scoreboard", "docs/FRONTIER-TARGET.md"),
    ("frontier-anchor-gap", "docs/FRONTIER-TARGET.md"),
    ("frontier-anchor-launch", "docs/FRONTIER-TARGET.md"),
    ("report-anchor-scoreboard", "results/REPORT.md"),
)

ANCHOR_GLOB = "results/frontier/anchor-0.8b-*.json"
UPPER_GLOB = "results/frontier/anchor-2b-*.json"
GAP_GLOB = "results/frontier/anchor-gap-*.json"


class Failure(Exception):
    pass


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest(repo: str, pattern: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(repo, pattern)))
    if not matches:
        return None
    if len(matches) > 1:
        raise Failure(
            f"{pattern} matches {len(matches)} receipts ({', '.join(os.path.basename(m) for m in matches)}); "
            "scoreboard rendering needs exactly one run per model so the table cannot mix runs"
        )
    return matches[0]


def load(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def task_entry(receipt: dict | None, task: str) -> dict | None:
    if receipt is None:
        return None
    for entry in receipt["tasks"]:
        if entry["task"] == task:
            return entry
    return None


def pct(value: float | None) -> str:
    return "pending" if value is None else f"{value * 100:.2f}"


def se(entry: dict | None) -> str:
    if entry is None:
        return "pending"
    value = entry.get("stderr")
    return "n/a" if value is None else f"{value * 100:.2f}"


def n_of(entry: dict | None) -> str:
    if entry is None:
        return "pending"
    value = (entry.get("n_samples") or {}).get("sample_len")
    return "n/a" if value is None else str(value)


def metric_of(entry: dict | None, fallback: str) -> str:
    if entry is None:
        return fallback
    return entry.get("primary_metric", fallback)


def render_scoreboard(repo: str, prereg: dict, anchor: dict | None, upper: dict | None) -> list[str]:
    lines = [
        "| task | metric | n | few-shot | batch | anchor | SE | 2B | SE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in prereg["tasks"]:
        name = task["name"]
        low = task_entry(anchor, name)
        high = task_entry(upper, name)
        metric = metric_of(low, task["primary_metric"])
        lines.append(
            f"| `{name}` | `{metric}` | {n_of(low)} | {task['num_fewshot']} | {task['batch_size']} | "
            f"{pct(low['score'] if low else None)} | "
            f"{se(low)} | {pct(high['score'] if high else None)} | {se(high)} |"
        )
    return lines


def render_launch(repo: str, prereg: dict, anchor: dict | None, upper: dict | None) -> list[str]:
    lines = [
        "| run | model | container limit | container peak (B) | device peak (GiB) | device total (GiB) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for label, receipt in (("anchor", anchor), ("gap reference", upper)):
        if receipt is None:
            lines.append(f"| {label} | pending | pending | pending | pending | pending |")
            continue
        peaks = [
            entry.get("peak_memory_allocated_bytes")
            for entry in receipt["tasks"]
            if entry.get("peak_memory_allocated_bytes")
        ]
        device_peak = max(peaks) / 2**30 if peaks else None
        total = (receipt.get("device") or {}).get("total_memory_bytes")
        cgroup = receipt.get("cgroup") or {}
        cgroup_peak = cgroup.get("memory.peak")
        limit = (receipt.get("court") or {}).get("memory_limit", "unknown")
        lines.append(
            f"| {label} | `{receipt['model']['id']}` | {limit} | "
            f"{cgroup_peak if cgroup_peak not in (None, '') else 'n/a (assembled)'} | "
            f"{f'{device_peak:.2f}' if device_peak else 'n/a'} | "
            f"{f'{total / 2**30:.2f}' if total else 'n/a'} |"
        )
    return lines


def render_gap(repo: str, prereg: dict, gap: dict | None) -> list[str]:
    if gap is None:
        return ["No gap receipt exists yet: the upper-end run has not been scored, so no gap is claimed."]
    lines = [
        "| task | family | anchor | 2B | gap (pp) | half gap (pp) | target = anchor + half |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gap["tasks"]:
        lines.append(
            f"| `{row['task']}` | {row['family'] or '—'} | {row['anchor_score'] * 100:.2f} | "
            f"{row['upper_score'] * 100:.2f} | {row['gap_pp']:.2f} | {row['half_gap_pp']:.2f} | "
            f"{row['anchor_plus_half_gap_pp']:.2f} |"
        )
    lines += [
        "",
        "| family | members | anchor mean | 2B mean | gap (pp) | half gap (pp) | target |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gap["families"]:
        lines.append(
            f"| {row['family']} | {', '.join('`' + member + '`' for member in row['members'])} | "
            f"{row['anchor_mean'] * 100:.2f} | {row['upper_mean'] * 100:.2f} | {row['gap_pp']:.2f} | "
            f"{row['half_gap_pp']:.2f} | {row['anchor_plus_half_gap'] * 100:.2f} |"
        )
    return lines


def render(region_id: str, repo: str, prereg: dict, anchor: dict | None, upper: dict | None, gap: dict | None) -> str:
    if region_id == "frontier-anchor-scoreboard":
        return "\n".join(render_scoreboard(repo, prereg, anchor, upper))
    if region_id == "report-anchor-scoreboard":
        lines = list(render_scoreboard(repo, prereg, anchor, upper))
        lines += ["", *render_gap(repo, prereg, gap)]
        return "\n".join(lines)
    if region_id == "frontier-anchor-gap":
        return "\n".join(render_gap(repo, prereg, gap))
    if region_id == "frontier-anchor-launch":
        return "\n".join(render_launch(repo, prereg, anchor, upper))
    raise Failure(f"unknown region {region_id!r}")


def collect(repo: str, args) -> tuple[dict, dict | None, dict | None, dict | None]:
    prereg = load(os.path.join(repo, PREREG_REL))
    anchor_path = args.anchor or newest(repo, ANCHOR_GLOB)
    upper_path = args.upper or newest(repo, UPPER_GLOB)
    gap_path = args.gap or newest(repo, GAP_GLOB)
    anchor = load(anchor_path if anchor_path else None)
    upper = load(upper_path if upper_path else None)
    gap = load(gap_path if gap_path else None)
    if anchor is not None and anchor.get("smoke"):
        raise Failure(f"{anchor_path}: a smoke receipt is not a scoreboard")
    if upper is not None and upper.get("smoke"):
        raise Failure(f"{upper_path}: a smoke receipt is not a scoreboard")
    if gap is not None and upper is not None:
        if gap["upper"]["receipt"] and os.path.basename(gap["upper"]["receipt"]) != os.path.basename(str(upper_path)):
            raise Failure("the gap receipt was computed from a different 2B run than the one being rendered")
    return prereg, anchor, upper, gap


def body_for(region_id: str, repo: str, payload) -> list[str]:
    prereg, anchor, upper, gap = payload
    text = render(region_id, repo, prereg, anchor, upper, gap)
    return text.split("\n")


def check(repo: str, args) -> int:
    payload = collect(repo, args)
    failures = 0
    for region_id, rel in REGIONS:
        path = os.path.join(repo, rel)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        lines, regions = pn.scan_regions(text, rel)
        by_id = {region.id: region for region in regions}
        if region_id not in by_id:
            raise Failure(f"{rel} has no generated region {region_id!r}")
        actual = pn.region_body(lines, by_id[region_id])
        expected = "\n".join(body_for(region_id, repo, payload))
        if actual != expected:
            failures += 1
            print(f"scoreboard: STALE {rel}:{region_id} — run --write", file=sys.stderr)
    if failures:
        return 1
    print(f"scoreboard: all {len(REGIONS)} regions match their receipts")
    return 0


def write(repo: str, args) -> int:
    payload = collect(repo, args)
    for region_id, rel in REGIONS:
        path = os.path.join(repo, rel)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        lines, regions = pn.scan_regions(text, rel)
        by_id = {region.id: region for region in regions}
        if region_id not in by_id:
            raise Failure(f"{rel} has no generated region {region_id!r}")
        region = by_id[region_id]
        rendered = "\n".join(body_for(region_id, repo, payload))
        lines = text.split("\n")
        lines = pn.replace_region(lines, region, rendered)
        directory = os.path.dirname(path) or "."
        handle_fd, tmp = tempfile.mkstemp(dir=directory, prefix=".scoreboard-")
        with os.fdopen(handle_fd, "w", encoding="utf-8") as out:
            out.write("\n".join(lines))
        os.replace(tmp, path)
        print(f"scoreboard: wrote {rel}:{region_id}")
    return 0


def list_regions(repo: str, args) -> int:
    payload = collect(repo, args)
    for region_id, rel in REGIONS:
        print(f"{region_id} -> {rel} ({len(body_for(region_id, repo, payload))} lines)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--list", action="store_true")
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--upper", default=None)
    parser.add_argument("--gap", default=None)
    parser.add_argument("--repo", default=repo_root())
    args = parser.parse_args(argv)
    repo = os.path.abspath(args.repo)
    try:
        if args.check:
            return check(repo, args)
        if args.write:
            return write(repo, args)
        return list_regions(repo, args)
    except Failure as exc:
        print(f"scoreboard: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
