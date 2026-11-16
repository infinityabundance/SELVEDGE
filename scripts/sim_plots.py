#!/usr/bin/env python3
"""Render the simulator's Pareto and placement receipts as SVG.

Standard library only: the python court image has no plotting library, and the
program does not add one for two charts. The charts are generated from the
receipts, never hand-drawn.

Usage:

    python3 scripts/sim_plots.py \
        --pareto results/sim/pareto-m2-17b.json \
        --placement results/sim/placement-synthetic-alternating.json \
        --out-dir results/sim
"""

from __future__ import annotations

import argparse
import json
import os


POLICY_COLOURS = {
    "none": "#666666",
    "lru": "#1f6feb",
    "lfu": "#bf3989",
    "least_stale": "#2da44e",
    "pinned": "#9a6700",
}


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pareto_svg(report: dict) -> str:
    rows = report["rows"]
    width, height = 900, 520
    left, right, top, bottom = 90, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_bytes = max(row["bytes_per_record"] for row in rows) or 1.0
    max_stall = max(row["p95_stall_ns"] for row in rows) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="24" y="30" font-family="sans-serif" font-size="16" fill="#111">{report["trace_id"]} — bytes per record against p95 stall</text>')
    parts.append('<text x="24" y="50" font-family="sans-serif" font-size="11" fill="#555">simulated; quality is fixed because every arm executes the trace selection exactly</text>')
    # axes
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111"/>')
    for tick in range(0, 101, 25):
        y = top + plot_h - plot_h * tick / 100.0
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        value = max_stall * tick / 100.0 / 1e6
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="monospace" font-size="10" fill="#555">{value:.1f}</text>')
    for tick in range(0, 101, 25):
        x = left + plot_w * tick / 100.0
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f0f0f0"/>')
        value = max_bytes * tick / 100.0 / 1e6
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 16}" text-anchor="middle" font-family="monospace" font-size="10" fill="#555">{value:.1f}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#111">device bytes per record (MB)</text>')
    parts.append(f'<text x="20" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2:.1f})" font-family="sans-serif" font-size="12" fill="#111">p95 stall (ms)</text>')
    for row in rows:
        x = left + plot_w * row["bytes_per_record"] / max_bytes
        y = top + plot_h - plot_h * row["p95_stall_ns"] / max_stall
        colour = POLICY_COLOURS.get(row["policy"], "#333333")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colour}" fill-opacity="0.85"/>')
    # legend
    legend_y = top + 8
    for policy, colour in POLICY_COLOURS.items():
        if not any(row["policy"] == policy for row in rows):
            continue
        parts.append(f'<circle cx="{left + plot_w - 130}" cy="{legend_y}" r="4" fill="{colour}"/>')
        parts.append(f'<text x="{left + plot_w - 120}" y="{legend_y + 4}" font-family="monospace" font-size="11" fill="#111">{policy}</text>')
        legend_y += 16
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def placement_svg(report: dict) -> str:
    rows = report["rows"]
    width = 900
    bar_h = 34
    top = 80
    height = top + bar_h * len(rows) + 60
    max_stall = max(row["p95_stall_ns"] for row in rows) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="24" y="30" font-family="sans-serif" font-size="16" fill="#111">{report["trace_id"]} — p95 stall by physical layout</text>')
    parts.append('<text x="24" y="50" font-family="sans-serif" font-size="11" fill="#555">simulated; bar length is p95, label carries amplification and bytes per record</text>')
    for index, row in enumerate(rows):
        y = top + index * bar_h
        length = 520.0 * row["p95_stall_ns"] / max_stall
        parts.append(f'<text x="24" y="{y + 20}" font-family="monospace" font-size="11" fill="#111">{row["kind"]}</text>')
        parts.append(f'<rect x="220" y="{y + 6}" width="{length:.2f}" height="18" fill="#1f6feb" fill-opacity="0.75"/>')
        parts.append(f'<text x="{230 + length:.2f}" y="{y + 20}" font-family="monospace" font-size="11" fill="#111">p95 {row["p95_stall_ns"] / 1e6:.3f} ms · amp {row["amplification"]:.3f} · {row["bytes_per_record"] / 1e6:.3f} MB/record</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pareto = load(args.pareto)
    placement = load(args.placement)
    pareto_path = os.path.join(args.out_dir, "pareto-m2-17b.svg")
    placement_path = os.path.join(args.out_dir, "placement-synthetic-alternating.svg")
    with open(pareto_path, "w", encoding="utf-8") as handle:
        handle.write(pareto_svg(pareto))
    with open(placement_path, "w", encoding="utf-8") as handle:
        handle.write(placement_svg(placement))
    print(f"wrote {pareto_path}")
    print(f"wrote {placement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
