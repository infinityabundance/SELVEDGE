#!/usr/bin/env python3
"""Court E (real model): assert bitwise parity across streaming variants.

Reads the ``selvedge run --json`` receipts written by
``scripts/runtime-courts.sh court-e`` and fails closed unless every variant
produced the same per-token logits digests, the same sequence digest, the same
generated tokens and a proven exact-mode classification.

The receipt it writes is the evidence: it names every variant's streaming
configuration, its device bytes and its memory peak, so a reader can see both
that parity held and that the variants really were different.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "selvedge/runtime-court-e/v1"


def load(receipt: Path) -> dict:
    document = json.loads(receipt.read_text(encoding="utf-8"))
    if document.get("schema") != "selvedge/runtime-run/v1":
        raise SystemExit(f"{receipt} is not a runtime run receipt")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--dir", default="results/runtime")
    parser.add_argument("--names", required=True, help="comma-separated receipt names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.dir)
    names = [name for name in args.names.split(",") if name]
    variants: list[dict] = []
    for name in names:
        receipt = out_dir / f"{name}-{args.stamp}.json"
        run = load(receipt)
        peak_path = out_dir / f"{name}-{args.stamp}.peak"
        peak = int(peak_path.read_text().strip()) if peak_path.exists() else None
        variants.append(
            {
                "name": name,
                "receipt": str(receipt),
                "stream": run["stream"],
                "mode": run["mode"],
                "exactness": run["exactness"]["status"],
                "generated_tokens": run["generated_tokens"],
                "sequence_digest": run["sequence_digest"],
                "logits_digests": run["logits_digests"],
                "device_bytes": run["environment"]["store"]["device_bytes"],
                "bytes_device_per_token": run["totals"]["bytes_device"] // max(1, run["totals"]["tokens"]),
                "active_unique_max": run["totals"]["active_unique_max"],
                "prefill_p50_ns": run["prefill"]["p50_ns"],
                "decode_p50_ns": run["decode"]["p50_ns"],
                "memory_peak_bytes": peak,
                "threads": run["environment"]["threads"],
            }
        )

    baseline = variants[0]
    failures: list[str] = []
    for variant in variants:
        if variant["sequence_digest"] != baseline["sequence_digest"]:
            failures.append(
                f"{variant['name']}: sequence digest {variant['sequence_digest']} "
                f"!= {baseline['sequence_digest']} ({baseline['name']})"
            )
        if variant["logits_digests"] != baseline["logits_digests"]:
            failures.append(f"{variant['name']}: per-token logits digests differ")
        if variant["generated_tokens"] != baseline["generated_tokens"]:
            failures.append(f"{variant['name']}: generated tokens differ")
        if variant["active_unique_max"] != baseline["active_unique_max"]:
            failures.append(f"{variant['name']}: a different active set executed")
        if variant["exactness"] != "proven":
            failures.append(f"{variant['name']}: exactness is {variant['exactness']}")

    # The variants must differ physically, or the parity claim proves nothing.
    cold = next((v for v in variants if v["name"].endswith("-cold")), None)
    warm = next((v for v in variants if v["name"].endswith("-warm")), None)
    if cold and warm:
        if cold["bytes_device_per_token"] <= warm["bytes_device_per_token"]:
            failures.append(
                "the cold variant did not move more device bytes per token than the warm one"
            )
    else:
        failures.append("court E needs a -cold and a -warm variant")

    document = {
        "schema": SCHEMA,
        "stamp": args.stamp,
        "revision": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "ok": not failures,
        "failures": failures,
        "claim": (
            "exact-mode logits are bitwise identical under different cache capacity, "
            "cache policy, prefetch setting, fetch order, injected device delay and "
            "kernel thread count"
        ),
        "parity_basis": "SHA-256 over each token's exact f32 logits, plus the sequence digest",
        "variants": variants,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in document.items() if k != "variants"}, indent=2))
    for variant in variants:
        print(
            f"{variant['name']:<18} threads={variant['threads']} "
            f"device_bytes/token={variant['bytes_device_per_token']} "
            f"decode_p50={variant['decode_p50_ns']} ns "
            f"peak={variant['memory_peak_bytes']} digest={variant['sequence_digest'][:16]}"
        )
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
