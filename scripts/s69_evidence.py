#!/usr/bin/env python3
"""Pack and verify the S69 corpus evidence chain.

The S69 rollout and preference artifacts were produced under the gitignored
``results/artifacts/`` tree, so a reader checking out the repository could not
inspect the corpus that ``docs/S69-RESULT.md`` cites.  This tool makes the chain
checkable from tracked files alone:

    scripts/s69_evidence.py pack      # regenerate the .gz files and INDEX.json
    scripts/s69_evidence.py verify    # recompute everything, fail closed

``pack`` gzips (level 9, zero timestamp, no stored filename) each text artifact
into ``results/stage2/<group>/raw/`` and writes ``results/stage2/<group>/INDEX.json``
with, per file, the uncompressed SHA-256 and byte count, the row count for
JSONL, the compressed byte count and SHA-256, and every receipt field that
records the same digest.  A digest that disagrees with a receipt is a defect:
``pack`` records ``matches: false`` and exits non-zero, and ``verify`` refuses.

``verify`` reads only the tracked ``.gz`` archives, the index and the receipts.
It never needs the gitignored source tree, and it never writes.  Every digest
in the index is recomputed from the decompressed bytes and compared with the
index and with the digest the cited receipt records, so a tampered archive or a
stale receipt fails.

Standard library only; reads and writes files, never the network.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile

SCHEMA = "selvedge/s69-evidence-index/v1"

#: Each artifact names its group, source path under the gitignored artifacts
#: tree, and the receipt pointers that record its SHA-256.  ``kind`` is "rows"
#: for line-delimited JSON and "receipt" for a receipt JSON.
ARTIFACTS = [
    # --- s69-rollouts -----------------------------------------------------
    {
        "group": "s69-rollouts",
        "name": "soup-prompts.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-rollouts/raw/soup-receipt.json.gz",
             "/prompt_manifest"),
        ],
    },
    {
        "group": "s69-rollouts",
        "name": "soup-accepted.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-rollouts/raw/soup-receipt.json.gz",
             "/accepted"),
        ],
    },
    {
        "group": "s69-rollouts",
        "name": "soup-ledger.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-rollouts/raw/soup-receipt.json.gz",
             "/ledger"),
        ],
    },
    {
        "group": "s69-rollouts",
        "name": "soup-rejected.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-rollouts/raw/soup-receipt.json.gz",
             "/preference_data"),
        ],
    },
    {
        "group": "s69-rollouts",
        "name": "soup-receipt.json",
        "kind": "receipt",
        "citations": [],
    },
    # --- s69-preference ---------------------------------------------------
    {
        "group": "s69-preference",
        "name": "soup-pairs.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-preference/raw/soup-receipt.json.gz",
             "/pairs"),
            ("results/stage2/s69-preference/raw/"
             "pairs-with-reference-receipt.json.gz", "/inputs/0"),
        ],
    },
    {
        "group": "s69-preference",
        "name": "soup-recovery.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-preference/raw/soup-receipt.json.gz",
             "/recovery"),
            ("results/stage2/s69-preference/raw/"
             "recovery-with-reference-receipt.json.gz", "/inputs/0"),
        ],
    },
    {
        "group": "s69-preference",
        "name": "pairs-with-reference.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-preference/raw/"
             "pairs-with-reference-receipt.json.gz", "/output"),
            ("results/stage2/s69-preference/preference/receipt-preference.json",
             "/pairs"),
            ("results/stage2/s69-preference/matched-sft/receipt-matched-sft.json",
             "/pairs"),
            ("results/stage2/s69-preference/matched-sft-weighted/"
             "receipt-matched-sft-weighted.json", "/pairs"),
        ],
    },
    {
        "group": "s69-preference",
        "name": "recovery-with-reference.jsonl",
        "kind": "rows",
        "citations": [
            ("results/stage2/s69-preference/raw/"
             "recovery-with-reference-receipt.json.gz", "/output"),
            ("results/stage2/s69-preference/recovery/receipt-recovery.json",
             "/pairs"),
        ],
    },
    {
        "group": "s69-preference",
        "name": "soup-receipt.json",
        "kind": "receipt",
        "citations": [],
    },
    {
        "group": "s69-preference",
        "name": "pairs-with-reference-receipt.json",
        "kind": "receipt",
        "citations": [],
    },
    {
        "group": "s69-preference",
        "name": "recovery-with-reference-receipt.json",
        "kind": "receipt",
        "citations": [],
    },
]

GROUPS = {
    "s69-rollouts": {
        "source_dir": "results/artifacts/stage2/s69-rollouts",
        "stored_dir": "results/stage2/s69-rollouts/raw",
        "index": "results/stage2/s69-rollouts/INDEX.json",
    },
    "s69-preference": {
        "source_dir": "results/artifacts/stage2/s69-preference",
        "stored_dir": "results/stage2/s69-preference/raw",
        "index": "results/stage2/s69-preference/INDEX.json",
    },
}

NOTE = (
    "The uncompressed artifacts live under the gitignored "
    "results/artifacts/ tree. These gzip -9 archives are lossless; every digest "
    "here is the SHA-256 of the uncompressed bytes and is compared with the "
    "digest the cited receipt records. `python3 scripts/s69_evidence.py verify` "
    "recomputes the whole chain from tracked files."
)


class Failure(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def read_maybe_gzip(path: str) -> bytes:
    data = read_bytes(path)
    if path.endswith(".gz"):
        return gzip.decompress(data)
    return data


def count_rows(data: bytes) -> int:
    if not data:
        return 0
    rows = data.count(b"\n")
    if not data.endswith(b"\n"):
        rows += 1
    return rows


def resolve_pointer(document: object, pointer: str, where: str) -> object:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise Failure(f"{where}: pointer {pointer!r} must start with '/'")
    current = document
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                raise Failure(f"{where}: pointer {pointer!r} has no list index {token!r}")
        elif isinstance(current, dict):
            if token not in current:
                raise Failure(f"{where}: pointer {pointer!r} has no key {token!r}")
            current = current[token]
        else:
            raise Failure(
                f"{where}: pointer {pointer!r} descends into a {type(current).__name__}")
    return current


def recorded_digest(root: str, receipt: str, pointer: str) -> tuple[str | None, int | None]:
    path = os.path.join(root, receipt)
    if not os.path.exists(path):
        raise Failure(f"cited receipt {receipt} is missing")
    document = json.loads(read_maybe_gzip(path))
    node = resolve_pointer(document, pointer, f"{receipt}#{pointer}")
    if not isinstance(node, dict):
        raise Failure(f"{receipt}#{pointer} is not an object")
    digest = node.get("sha256")
    rows = node.get("rows")
    if digest is None:
        # /output in the reference receipts has no rows but does have sha256.
        raise Failure(f"{receipt}#{pointer} records no sha256")
    return digest, rows


def write_atomic(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=directory, prefix=".s69-evidence-", delete=False)
    try:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# pack


def pack(root: str) -> int:
    problems: list[str] = []
    by_group: dict[str, list[dict]] = {name: [] for name in GROUPS}

    # Phase 1: write every archive, so a receipt packed later in the same run is
    # on disk before any citation is resolved (a receipt can cite a sibling).
    for spec in ARTIFACTS:
        group = GROUPS[spec["group"]]
        name = spec["name"]
        source = os.path.join(root, group["source_dir"], name)
        stored_rel = f"{group['stored_dir']}/{name}.gz"
        stored = os.path.join(root, stored_rel)
        if not os.path.exists(source):
            problems.append(f"{group['source_dir']}/{name}: source is missing")
            continue
        raw = read_bytes(source)
        digest = sha256_bytes(raw)
        rows = count_rows(raw) if spec["kind"] == "rows" else None
        # gzip level 9, zero mtime, no filename: deterministic.
        packed = gzip.compress(raw, compresslevel=9, mtime=0)
        write_atomic(stored, packed)
        by_group[spec["group"]].append({
            "_spec": spec,
            "source": f"{group['source_dir']}/{name}",
            "stored": stored_rel,
            "kind": spec["kind"],
            "bytes": len(raw),
            "rows": rows,
            "sha256": digest,
            "gz_bytes": len(packed),
            "gz_sha256": sha256_bytes(packed),
            "citations": [],
        })

    # Phase 2: resolve every receipt citation against the digest computed above.
    for entries in by_group.values():
        for entry in entries:
            name = os.path.basename(entry["source"])
            for receipt, pointer in entry["_spec"]["citations"]:
                want, want_rows = recorded_digest(root, receipt, pointer)
                entry["citations"].append({
                    "receipt": receipt,
                    "pointer": pointer,
                    "recorded_sha256": want,
                    "recorded_rows": want_rows,
                    "matches": want == entry["sha256"],
                })
                if want != entry["sha256"]:
                    problems.append(
                        f"{name}: receipt {receipt}#{pointer} records {want}, "
                        f"uncompressed bytes hash to {entry['sha256']}")
                if (want_rows is not None and entry["rows"] is not None
                        and want_rows != entry["rows"]):
                    problems.append(
                        f"{name}: receipt {receipt}#{pointer} records {want_rows} "
                        f"rows, file has {entry['rows']}")

    for entries in by_group.values():
        for entry in entries:
            del entry["_spec"]

    for group_name, group in GROUPS.items():
        entries = by_group[group_name]
        entries.sort(key=lambda item: item["stored"])
        index = {
            "schema": SCHEMA,
            "group": group_name,
            "source_dir": group["source_dir"],
            "stored_dir": group["stored_dir"],
            "generated_by": "scripts/s69_evidence.py pack",
            "note": NOTE,
            "files": {
                os.path.basename(entry["source"]): entry for entry in entries
            },
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "total_gz_bytes": sum(entry["gz_bytes"] for entry in entries),
        }
        write_atomic(os.path.join(root, group["index"]),
                     (json.dumps(index, indent=2, sort_keys=True) + "\n").encode())

    if problems:
        for problem in problems:
            print(f"s69_evidence: DEFECT — {problem}", file=sys.stderr)
        print(f"s69_evidence: {len(problems)} defect(s); the index records them",
              file=sys.stderr)
        return 1
    files = sum(len(entries) for entries in by_group.values())
    gz_bytes = sum(
        entry["gz_bytes"] for entries in by_group.values() for entry in entries)
    print(
        f"s69_evidence: packed {files} artifact(s) into "
        f"{len(GROUPS)} group(s), {gz_bytes} compressed bytes; every recorded "
        "digest matches its receipt")
    return 0


# ---------------------------------------------------------------------------
# verify


def verify(root: str) -> int:
    problems: list[str] = []
    files = 0
    rows = 0
    citations = 0
    gz_bytes = 0
    for group_name, group in GROUPS.items():
        index_path = os.path.join(root, group["index"])
        if not os.path.exists(index_path):
            problems.append(f"{group['index']}: index is missing")
            continue
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if index.get("schema") != SCHEMA:
            problems.append(f"{group['index']}: schema is {index.get('schema')!r}, "
                            f"expected {SCHEMA!r}")
            continue
        expected_names = {
            spec["name"] for spec in ARTIFACTS if spec["group"] == group_name
        }
        if set(index.get("files", {})) != expected_names:
            problems.append(
                f"{group['index']}: files {sorted(index.get('files', {}))} do not "
                f"match the expected set {sorted(expected_names)}")
        for name, entry in sorted(index.get("files", {}).items()):
            stored = os.path.join(root, entry["stored"])
            if not os.path.exists(stored):
                problems.append(f"{entry['stored']}: stored archive is missing")
                continue
            packed = read_bytes(stored)
            raw = gzip.decompress(packed) if stored.endswith(".gz") else packed
            digest = sha256_bytes(raw)
            file_rows = count_rows(raw) if entry["kind"] == "rows" else None
            files += 1
            rows += file_rows or 0
            gz_bytes += len(packed)
            if digest != entry["sha256"]:
                problems.append(
                    f"{entry['stored']}: uncompressed sha256 {digest} != index "
                    f"{entry['sha256']}")
            if len(raw) != entry["bytes"]:
                problems.append(
                    f"{entry['stored']}: uncompressed bytes {len(raw)} != index "
                    f"{entry['bytes']}")
            if file_rows != entry["rows"]:
                problems.append(
                    f"{entry['stored']}: rows {file_rows} != index {entry['rows']}")
            if len(packed) != entry["gz_bytes"]:
                problems.append(
                    f"{entry['stored']}: compressed bytes {len(packed)} != index "
                    f"{entry['gz_bytes']}")
            if sha256_bytes(packed) != entry["gz_sha256"]:
                problems.append(f"{entry['stored']}: compressed sha256 does not "
                                "match the index")
            for citation in entry["citations"]:
                citations += 1
                receipt = citation["receipt"]
                pointer = citation["pointer"]
                try:
                    want, want_rows = recorded_digest(root, receipt, pointer)
                except Failure as exc:
                    problems.append(str(exc))
                    continue
                if want != digest:
                    problems.append(
                        f"{name}: receipt {receipt}#{pointer} records {want} but "
                        f"the tracked archive hashes to {digest}")
                if want != citation["recorded_sha256"]:
                    problems.append(
                        f"{name}: index recorded {citation['recorded_sha256']} for "
                        f"{receipt}#{pointer}, receipt now records {want}")
                if want_rows is not None and file_rows is not None and want_rows != file_rows:
                    problems.append(
                        f"{name}: receipt {receipt}#{pointer} records {want_rows} "
                        f"rows, archive has {file_rows}")
    if problems:
        for problem in problems:
            print(f"s69_evidence: FAIL — {problem}", file=sys.stderr)
        print(f"s69_evidence: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(
        f"s69_evidence: OK — {files} artifact(s), {rows} JSONL rows, "
        f"{citations} receipt citation(s), {gz_bytes} compressed bytes verified")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "verify"))
    parser.add_argument("--repo", default=repo_root())
    args = parser.parse_args(argv)
    root = os.path.abspath(args.repo)
    try:
        return pack(root) if args.action == "pack" else verify(root)
    except Failure as exc:
        print(f"s69_evidence: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
