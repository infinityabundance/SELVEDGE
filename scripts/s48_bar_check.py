#!/usr/bin/env python3
"""S48: verify the 20M bar's receipt, checkpoint, config hash and token count.

The bar's own receipt is written by the process it certifies. This script
re-derives the claims that a reader can check without trusting that process:

* the receipt's `config` hashes to its recorded `config_sha256`, with the same
  canonicalisation `oracle/frontier/train.py` uses (`json.dumps(sort_keys=True)`);
* the final checkpoint exists, its SHA-256 is the receipt's, and the payload
  inside carries the same step, tokens seen, config hash and stream hash;
* the token arithmetic is exact: steps x tokens-per-step equals the tokens seen;
* the stream manifest's SHA-256 is the file's SHA-256 and the receipt's;
* the registered knobs (objective, KL weight, temperature, lr, seed, schedule)
  are the ones the launch receipt recorded, not a drift.

It writes `results/stage2/s48-bar-verify-<stamp>.json` and fails closed on any
mismatch. CPU only; it reads the checkpoint with `weights_only=False` because the
payload carries RNG states, and never loads the model.

Usage:
    scripts/court oracle -- python3 scripts/s48_bar_check.py \
        --run-dir results/stage2/runs/full-20m-math
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCHEMA = "selvedge/s48-bar-verify/v1"
EXPECTED = {
    "arm": "full",
    "tokens": 20_000_000,
    "tokens_per_step": 2048,
    "steps": 9765,
    "kl_weight": 0.5,
    "temperature": 2.0,
    "lr": 1e-4,
    "seed": 0,
    "schedule": "cosine",
    "optimizer": "adam8bit",
    "seq_len": 512,
    "micro_batch": 1,
    "grad_accum": 4,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def main(argv: list[str]) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/stage2/runs/full-20m-math")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = os.path.join(REPO, args.run_dir)
    receipt_path = os.path.join(run_dir, "receipt.json")
    if not os.path.exists(receipt_path):
        raise SystemExit(f"s48-bar: no receipt at {args.run_dir}/receipt.json")
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    config = receipt["config"]
    launch_path = os.path.join(run_dir, "launch.json")
    launch = (json.load(open(launch_path, encoding="utf-8"))
              if os.path.exists(launch_path) else {})

    checks: list[dict] = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    recomputed = config_hash(config)
    check("receipt schema", receipt.get("schema") == "selvedge/stage2-train/v1",
          receipt.get("schema"))
    check("config hash recomputes", recomputed == config.get("config_sha256"),
          {"recorded": config.get("config_sha256"), "recomputed": recomputed})
    for key, expected in EXPECTED.items():
        actual = config.get(key)
        check(f"config {key}", actual == expected, {"expected": expected,
                                                    "actual": actual})

    token_arithmetic = int(config["steps"]) * int(config["tokens_per_step"])
    check("steps x tokens_per_step == tokens seen",
          token_arithmetic == int(receipt.get("tokens_seen_total") or -1)
          == int(receipt.get("tokens_trained") or -1),
          {"product": token_arithmetic,
           "tokens_seen_total": receipt.get("tokens_seen_total"),
           "tokens_trained": receipt.get("tokens_trained")})
    check("realised tokens do not exceed the declared budget",
          token_arithmetic <= int(config["tokens"]),
          {"product": token_arithmetic, "declared": config["tokens"]})
    check("objective is the registered CE+KL",
          receipt.get("objective") == "ce_kl", receipt.get("objective"))

    final_path = receipt.get("final_checkpoint")
    full_path = os.path.join(REPO, final_path) if final_path else None
    exists = bool(full_path) and os.path.exists(full_path)
    check("final checkpoint exists", exists, final_path)
    file_sha = sha256_file(full_path) if exists else None
    check("checkpoint SHA-256 matches the receipt",
          file_sha is not None
          and file_sha == receipt.get("final_checkpoint_sha256"),
          {"receipt": receipt.get("final_checkpoint_sha256"), "file": file_sha})
    latest_path = os.path.join(run_dir, "latest.pt")
    latest_sha = (sha256_file(latest_path) if os.path.exists(latest_path)
                  else None)
    check("latest.pt is the final checkpoint", latest_sha is not None
          and latest_sha == file_sha,
          {"latest": latest_sha, "final": file_sha})

    payload = (torch.load(full_path, map_location="cpu", weights_only=False)
               if exists else {})
    check("payload step", payload.get("step") == config["steps"],
          {"payload": payload.get("step"), "config": config["steps"]})
    check("payload tokens_seen == tokens_trained",
          payload.get("tokens_seen") == receipt.get("tokens_seen_total"),
          {"payload": payload.get("tokens_seen"),
           "receipt": receipt.get("tokens_seen_total")})
    check("payload config hash", payload.get("config_sha256")
          == config.get("config_sha256"),
          {"payload": payload.get("config_sha256"),
           "config": config.get("config_sha256")})
    check("payload stream hash == config stream hash",
          payload.get("stream_sha256") == config.get("stream_sha256"),
          {"payload": payload.get("stream_sha256"),
           "config": config.get("stream_sha256")})
    check("payload arm", payload.get("arm") == "full", payload.get("arm"))

    manifest_path = os.path.join(REPO,
                                 launch.get("stream", {}).get("manifest", ""))
    manifest_sha = (sha256_file(manifest_path)
                    if manifest_path and os.path.exists(manifest_path) else None)
    check("stream manifest SHA-256",
          manifest_sha is not None
          and manifest_sha == launch.get("stream", {}).get("sha256"),
          {"file": manifest_sha, "launch": launch.get("stream", {}).get("sha256")})
    stream_file = None
    if manifest_path and os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        stream_file = os.path.join(REPO, manifest["stream_path"])
        stream_sha = (sha256_file(stream_file)
                      if os.path.exists(stream_file) else None)
        check("stream file SHA-256 == manifest",
              stream_sha == manifest.get("sha256"),
              {"file": stream_sha, "manifest": manifest.get("sha256")})
        check("stream file SHA-256 == config", stream_sha == config.get("stream_sha256"),
              {"file": stream_sha, "config": config.get("stream_sha256")})

    log_path = os.path.join(REPO, receipt.get("log", ""))
    last_step = None
    lines = 0
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                lines += 1
                last_step = json.loads(line).get("step")
    check("log is complete", lines == config["steps"] and last_step == config["steps"],
          {"lines": lines, "last_step": last_step, "steps": config["steps"]})

    ok = all(item["ok"] for item in checks)
    report = {
        "schema": SCHEMA,
        "kind": "bar_verification",
        "stamp": stamp(),
        "created_utc": utcnow(),
        "run_dir": args.run_dir,
        "receipt": {"path": os.path.join(args.run_dir, "receipt.json"),
                    "revision": receipt.get("revision"),
                    "created_utc": receipt.get("created_utc")},
        "checks": checks,
        "ok": ok,
        "tokens_trained": receipt.get("tokens_trained"),
        "tokens_seen_total": receipt.get("tokens_seen_total"),
        "wall_seconds": receipt.get("wall_seconds"),
        "tokens_per_second": receipt.get("tokens_per_second"),
        "peak_allocated_gib": receipt.get("peak_allocated_gib"),
        "final_checkpoint": final_path,
        "final_checkpoint_sha256": file_sha,
        "config_sha256": config.get("config_sha256"),
        "stream_sha256": config.get("stream_sha256"),
        "optimizer_numerics": receipt.get("optimizer_numerics"),
        "container_memory": receipt.get("container_memory"),
    }
    out = args.out or os.path.join("results/stage2",
                                   f"s48-bar-verify-{report['stamp']}.json")
    full_out = os.path.join(REPO, out)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    with open(full_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for item in checks:
        print(f"{'ok  ' if item['ok'] else 'FAIL'} {item['check']}: {item['detail']}")
    print(f"s48-bar: {'VERIFIED' if ok else 'MISMATCH'} receipt={out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
