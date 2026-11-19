#!/usr/bin/env python3
"""Verify that the model directory a run is about to train is the registered one.

Stage 2a and stage 2b both train the **post-trained** `Qwen/Qwen3.5-0.8B`
(`2fc06364…`), while `train.py`'s historical default was the Base checkpoint.
A path that is merely not the Base path is not proof of identity: a re-fetched
snapshot, a different revision, or an accidentally mutated shard would all pass
that test and silently void every downstream comparison.

This module checks three independent things against
`models/FRONTIER-POST-TRAINED.lock.json` before the model is loaded:

1. the requested model id and revision match the lock entry;
2. the directory name is the pinned revision (this cache layout stores
   snapshots at `models/<org>/<repo>/<revision>/`);
3. the weight bundle hashes match the lock's registered `weight_sha256`
   (the first shard in sorted order, the same rule the S54 eval driver uses)
   and the licence file hash when the lock records one.

Every mismatch raises :class:`IdentityMismatch`. There is no warning-only path:
a run that cannot prove its student is refused.

CPU-only; no model is loaded here.
"""

from __future__ import annotations

import hashlib
import json
import os


class IdentityMismatch(RuntimeError):
    """Raised when a model directory does not match the registered identity."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_files(model_dir: str) -> list[str]:
    """Weight shards in the order the S54 driver hashes them."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            payload = json.load(handle)
        shards = sorted(set((payload.get("weight_map") or {}).values()))
        return [os.path.join(model_dir, name) for name in shards]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    return [os.path.join(model_dir, name)
            for name in sorted(os.listdir(model_dir))
            if name.endswith(".safetensors")]


def weight_sha256(model_dir: str) -> str | None:
    """The registered rule: hash the first sorted shard."""
    shards = shard_files(model_dir)
    return sha256_file(shards[0]) if shards else None


def lock_entry(lock: dict, model_id: str) -> dict | None:
    """Find a lock entry by model id across the top-level sections."""
    for key in ("target", "stage_2_teacher"):
        entry = lock.get(key)
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return entry
    for entry in (lock.get("controls") or {}).values():
        if isinstance(entry, dict) and entry.get("id") == model_id:
            return entry
    return None


def verify_student(student_dir: str, lock_path: str, model_id: str,
                   revision: str) -> dict:
    """Fail closed unless `student_dir` is the lock's pinned identity.

    Returns a report suitable for a manifest. The report is only produced after
    every check passes; a partial report is never returned.
    """
    if not os.path.isdir(student_dir):
        raise IdentityMismatch(f"student directory does not exist: {student_dir}")
    if not os.path.isfile(lock_path):
        raise IdentityMismatch(f"identity lock does not exist: {lock_path}")
    with open(lock_path, encoding="utf-8") as handle:
        lock = json.load(handle)
    entry = lock_entry(lock, model_id)
    if entry is None:
        raise IdentityMismatch(
            f"{lock_path}: no entry for model id {model_id!r}")
    if entry.get("revision") != revision:
        raise IdentityMismatch(
            f"{model_id}: lock revision {entry.get('revision')!r} != requested "
            f"{revision!r}")
    if os.path.basename(os.path.normpath(student_dir)) != revision:
        raise IdentityMismatch(
            f"student path {student_dir!r} does not end in the pinned revision "
            f"{revision!r}; the snapshot is not provably the registered one")
    config_path = os.path.join(student_dir, "config.json")
    if not os.path.isfile(config_path):
        raise IdentityMismatch(f"{student_dir}: config.json is missing")

    shards = shard_files(student_dir)
    if not shards:
        raise IdentityMismatch(f"{student_dir}: no safetensors weight file")
    shard_hashes = {os.path.basename(path): {
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path)} for path in shards}
    expected_weight = entry.get("weight_sha256")
    actual_weight = shard_hashes[os.path.basename(shards[0])]["sha256"]
    if expected_weight and actual_weight != expected_weight:
        raise IdentityMismatch(
            f"{model_id}@{revision}: first-shard sha256 {actual_weight} != "
            f"registered {expected_weight}")

    licence = None
    licence_path = os.path.join(student_dir, "LICENSE")
    expected_licence = entry.get("licence_file_sha256")
    if expected_licence and os.path.isfile(licence_path):
        licence = sha256_file(licence_path)
        if licence != expected_licence:
            raise IdentityMismatch(
                f"{model_id}@{revision}: LICENSE sha256 {licence} != registered "
                f"{expected_licence}")

    return {
        "verified": True,
        "model_id": model_id,
        "revision": revision,
        "directory": student_dir,
        "lock": lock_path,
        "lock_sha256": sha256_file(lock_path),
        "config_sha256": sha256_file(config_path),
        "weight_sha256": actual_weight,
        "weight_basis": "first sorted safetensors shard, the S54 driver's rule",
        "shards": shard_hashes,
        "licence_sha256": licence,
        "licence_checked": bool(expected_licence),
    }


__all__ = ["IdentityMismatch", "sha256_file", "shard_files", "weight_sha256",
           "lock_entry", "verify_student"]
