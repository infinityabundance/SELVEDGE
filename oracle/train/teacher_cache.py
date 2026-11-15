#!/usr/bin/env python3
"""Precompute compact teacher targets once for every T1 arm.

Per position the cache stores the top-16 teacher logits (bf16), their vocabulary
indices (int32) and the teacher log-sum-exp (fp32): exactly 100 bytes. Records
are packed in token order in one file, so a position's record lives at
`offset = token_index * 100`. The manifest splits the file into checksummed
chunks and records the teacher revision, the token budget and the source stream
hash.

`verify` recomputes a registered sample of chunks with a live teacher forward and
compares the top-16 index sets, the logits and the log-sum-exp. A failure
invalidates the cache (K10).

Subcommands: build | verify | show
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

RECORD_BYTES = 16 * 2 + 16 * 4 + 4  # 100
CHUNK_TOKENS = 65_536
TOP_K = 16
WINDOW = 1024
BATCH_WINDOWS = 2
VERIFY_CHUNKS = 16


class CacheError(Exception):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_region(path: str, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise CacheError(f"{path}: short read at {start}+{length}")
            digest.update(block)
            remaining -= len(block)
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


def model_path(hf_root: str, size: str) -> str:
    revisions = {
        "0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
        "1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    }
    directory = {"0.6b": "Qwen3-0.6B", "1.7b": "Qwen3-1.7B"}[size]
    path = os.path.join(hf_root, "models", "Qwen", directory, revisions[size])
    if not os.path.isdir(path):
        raise CacheError(f"model path {path} is absent; fetch models first")
    return path


def load_teacher(hf_root: str, size: str = "1.7b"):
    import torch
    from transformers import AutoModelForCausalLM

    path = model_path(hf_root, size)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def teacher_records(teacher, tokens, torch):
    """Top-k logits/ids and logsumexp for one [n_windows, WINDOW] token block."""
    device = next(teacher.parameters()).device
    with torch.inference_mode():
        logits = teacher(input_ids=tokens.to(device), use_cache=False).logits
        values, indices = torch.topk(logits, TOP_K, dim=-1)
        logsumexp = torch.logsumexp(logits.float(), dim=-1)
    values = values.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)
    indices = indices.to(torch.int32).cpu().numpy()
    logsumexp = logsumexp.cpu().numpy().astype(np.float32)
    return values, indices, logsumexp


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """Exact bf16 (raw uint16) to float32 widening."""
    widened = bits.astype(np.uint32) << 16
    return widened.view(np.float32)


def pack(values: np.ndarray, indices: np.ndarray, logsumexp: np.ndarray) -> bytes:
    positions = values.shape[0] * values.shape[1]
    record = np.zeros((positions, RECORD_BYTES), dtype=np.uint8)
    values = values.reshape(positions, TOP_K)
    indices = indices.reshape(positions, TOP_K)
    logsumexp = logsumexp.reshape(positions)
    record[:, 0:32] = values.view(np.uint8).reshape(positions, 32)
    record[:, 32:96] = indices.view(np.uint8).reshape(positions, 64)
    record[:, 96:100] = logsumexp.view(np.uint8).reshape(positions, 4)
    return record.tobytes()


def unpack(record: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = np.asarray(record, dtype=np.uint8).reshape(-1, RECORD_BYTES)
    bits = record[:, 0:32].copy().view(np.uint16)
    values = bf16_to_float32(bits)
    indices = record[:, 32:96].copy().view(np.int32)
    logsumexp = record[:, 96:100].copy().view(np.float32).reshape(-1)
    return values, indices, logsumexp


def build(hf_root: str, artifact_dir: str, receipt_dir: str, tokens: int | None,
          teacher_size: str) -> int:
    import torch

    if not torch.cuda.is_available():
        raise CacheError("teacher cache build requires the GPU court (--gpu)")
    stream_index = json.load(open(os.path.join(artifact_dir, "train.index.json"),
                                  encoding="utf-8"))
    stream_path = os.path.join(artifact_dir, "train.bin")
    available = stream_index["tokens"]
    budget = tokens or stream_index["registered_budget_tokens"]
    if budget > available:
        raise CacheError(f"requested {budget} tokens but the stream has {available}")
    tokens_per_step = WINDOW * BATCH_WINDOWS
    if budget % tokens_per_step:
        raise CacheError(f"budget {budget} is not a multiple of {tokens_per_step}")

    teacher = load_teacher(hf_root, teacher_size)
    revision = teaching_revision(hf_root, teacher_size)
    stream = np.memmap(stream_path, dtype="<u4", mode="r", shape=(available,))
    out_path = os.path.join(artifact_dir, "teacher-targets.bin")
    tmp_path = out_path + ".tmp"
    chunk_bytes = CHUNK_TOKENS * RECORD_BYTES
    chunks = []
    started = utc_now()
    import time

    clock = time.time()
    with open(tmp_path, "wb") as handle:
        for chunk_start in range(0, budget, CHUNK_TOKENS):
            chunk_end = min(chunk_start + CHUNK_TOKENS, budget)
            chunk_buffer = bytearray((chunk_end - chunk_start) * RECORD_BYTES)
            view = memoryview(chunk_buffer)
            position = 0
            for step in range(chunk_start, chunk_end, tokens_per_step):
                block = np.asarray(stream[step:step + tokens_per_step], dtype=np.int64)
                ids = torch.from_numpy(block.reshape(BATCH_WINDOWS, WINDOW))
                values, indices, logsumexp = teacher_records(teacher, ids, torch)
                raw = pack(values, indices, logsumexp)
                view[position:position + len(raw)] = raw
                position += len(raw)
            offset = handle.tell()
            handle.write(chunk_buffer)
            handle.flush()
            os.fsync(handle.fileno())
            chunks.append({
                "index": len(chunks),
                "start_token": chunk_start,
                "end_token": chunk_end,
                "offset": offset,
                "bytes": len(chunk_buffer),
                "sha256": hashlib.sha256(chunk_buffer).hexdigest(),
            })
            if len(chunks) % 8 == 0:
                elapsed = time.time() - clock
                done = chunk_end
                print(f"targets: {done}/{budget} tokens "
                      f"({done / max(elapsed, 1e-9):.0f} tokens/s)", flush=True)
    os.replace(tmp_path, out_path)
    elapsed = time.time() - clock

    manifest = {
        "schema": "selvedge/teacher-targets/v1",
        "teacher_size": teacher_size,
        "teacher_revision": revision,
        "stream_sha256": stream_index["stream_sha256"],
        "stream_path": os.path.relpath(stream_path, os.path.dirname(artifact_dir) or "."),
        "targets_file": os.path.relpath(out_path, os.path.dirname(artifact_dir) or "."),
        "record_bytes": RECORD_BYTES,
        "top_k": TOP_K,
        "tokens": budget,
        "bytes": os.path.getsize(out_path),
        "chunk_tokens": CHUNK_TOKENS,
        "chunks": chunks,
        "built_utc": utc_now(),
        "build_seconds": round(elapsed, 3),
        "tokens_per_second": round(budget / max(elapsed, 1e-9), 3),
    }
    manifest_path = os.path.join(artifact_dir, "teacher-targets.json")
    write_json_atomic(manifest_path, manifest)
    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/teacher-targets-receipt/v1",
        "kind": "build",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "started_utc": started,
        "manifest": manifest_path,
        "manifest_sha256": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
        "tokens": budget,
        "bytes": manifest["bytes"],
        "teacher_revision": revision,
        "stream_sha256": manifest["stream_sha256"],
        "build_seconds": manifest["build_seconds"],
        "tokens_per_second": manifest["tokens_per_second"],
        "gpu": gpu_name(torch),
    }
    out = os.path.join(receipt_dir, f"targets-{stamp}.json")
    write_json_atomic(out, receipt)
    print(json.dumps({key: receipt[key] for key in
                      ("stamp", "tokens", "bytes", "tokens_per_second", "gpu")}, indent=2))
    print(f"targets: manifest={manifest_path} receipt={out}")
    return 0


def teaching_revision(hf_root: str, size: str) -> str:
    return os.path.basename(model_path(hf_root, size))


def gpu_name(torch) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def verify(hf_root: str, artifact_dir: str, receipt_dir: str, teacher_size: str) -> int:
    import torch

    manifest = json.load(open(os.path.join(artifact_dir, "teacher-targets.json"),
                              encoding="utf-8"))
    path = os.path.join(artifact_dir, "teacher-targets.bin")
    if os.path.getsize(path) != manifest["bytes"]:
        raise CacheError("targets file size does not match the manifest")
    for chunk in manifest["chunks"]:
        got = sha256_region(path, chunk["offset"], chunk["bytes"])
        if got != chunk["sha256"]:
            raise CacheError(f"chunk {chunk['index']}: sha256 {got} != {chunk['sha256']}")

    stream_index = json.load(open(os.path.join(artifact_dir, "train.index.json"),
                                  encoding="utf-8"))
    stream = np.memmap(os.path.join(artifact_dir, "train.bin"), dtype="<u4", mode="r",
                       shape=(stream_index["tokens"],))
    teacher = load_teacher(hf_root, teacher_size)
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(manifest["bytes"],))
    sample = np.linspace(0, len(manifest["chunks"]) - 1, VERIFY_CHUNKS, dtype=int)
    worst_logit = 0.0
    worst_lse = 0.0
    mismatched = 0
    compared = 0
    for index in sorted(set(int(item) for item in sample)):
        chunk = manifest["chunks"][index]
        start = chunk["start_token"]
        for step in range(start, chunk["end_token"], WINDOW * BATCH_WINDOWS):
            block = np.asarray(stream[step:step + WINDOW * BATCH_WINDOWS], dtype=np.int64)
            ids = torch.from_numpy(block.reshape(BATCH_WINDOWS, WINDOW))
            values, indices, logsumexp = teacher_records(teacher, ids, torch)
            live_values = bf16_to_float32(values.reshape(-1, TOP_K))
            live_lse = logsumexp.reshape(-1)
            offset = step * RECORD_BYTES
            count = live_values.shape[0]
            cached = np.asarray(raw[offset:offset + count * RECORD_BYTES])
            cached_values, cached_ids, cached_lse = unpack(cached)
            cached_values = cached_values.astype(np.float32)
            live_ids = indices.reshape(-1, TOP_K)
            if not np.array_equal(live_ids, cached_ids):
                mismatched += int((live_ids != cached_ids).any(axis=1).sum())
            worst_logit = max(worst_logit, float(np.max(np.abs(live_values - cached_values))))
            worst_lse = max(worst_lse, float(np.max(np.abs(live_lse - cached_lse))))
            compared += count
    ok = mismatched == 0 and worst_logit <= 1e-2 and worst_lse <= 1e-3
    stamp = stamp_now()
    receipt = {
        "schema": "selvedge/teacher-targets-receipt/v1",
        "kind": "verify",
        "stamp": stamp,
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "chunks_checked": len(set(int(item) for item in sample)),
        "positions_compared": compared,
        "mismatched_positions": mismatched,
        "max_abs_logit_delta": worst_logit,
        "max_abs_logsumexp_delta": worst_lse,
        "logit_tolerance": 1e-2,
        "logsumexp_tolerance": 1e-3,
        "ok": bool(ok),
        "gpu": gpu_name(torch),
    }
    out = os.path.join(receipt_dir, f"targets-verify-{stamp}.json")
    write_json_atomic(out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not ok:
        raise CacheError("teacher target verification failed (K10)")
    print(f"targets: verified {compared} positions; receipt={out}")
    return 0


def show(artifact_dir: str) -> int:
    manifest = json.load(open(os.path.join(artifact_dir, "teacher-targets.json"),
                              encoding="utf-8"))
    summary = {key: manifest[key] for key in
               ("teacher_revision", "tokens", "bytes", "top_k", "record_bytes",
                "chunk_tokens", "built_utc", "build_seconds", "tokens_per_second")}
    summary["chunks"] = len(manifest["chunks"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify", "show"])
    parser.add_argument("--hf-root", default=os.environ.get("HF_HOME", "/hf-cache"))
    parser.add_argument("--artifact-dir", default="results/artifacts/train")
    parser.add_argument("--receipt-dir", default="results/train")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--teacher-size", default="1.7b", choices=["0.6b", "1.7b"])
    args = parser.parse_args(argv)
    if args.command == "build":
        return build(args.hf_root, args.artifact_dir, args.receipt_dir, args.tokens,
                     args.teacher_size)
    if args.command == "verify":
        return verify(args.hf_root, args.artifact_dir, args.receipt_dir, args.teacher_size)
    return show(args.artifact_dir)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CacheError as error:
        print(f"targets: {error}", file=sys.stderr)
        sys.exit(1)
