#!/usr/bin/env python3
"""Generate and check every published number from its receipt.

AGENTS.md directive 12: no performance, storage, latency, throughput, resource
or verification count is hand-written in a tracked document. Each number lives
between sentinels emitted here, is sourced from a receipt by JSON pointer, and
carries the receipt stamp and revision that produced it.

Sentinels:

    <!-- BEGIN GENERATED: <id> -->
    <rendered value>
    <!-- END GENERATED: <id> -->

Manifest: `docs/perf-numbers.json`. Blocks:

    {
      "id": "cpu-logical-cores",              # sentinel id, unique
      "doc": "docs/PERF-NUMBERS.md",          # tracked doc holding the sentinel
      "sentinel": "cpu-logical-cores",        # sentinel id in the doc
      "source": {
        "receipt": "results/hardware/probe-....json",
        "pointer": "/cpu/logical_cores"       # RFC 6901 JSON pointer
      },
      "format": "{value} logical cores",      # must contain {value}
      "declared_receipt_stamp": "20261114T...Z",
      "declared_revision": "<git sha of the receipt>"
    }

`--check` fails closed on: a stale region, a missing receipt, a declared stamp
or revision that does not match the receipt, a manifest number no block prints,
a sentinel with no manifest block, an untracked receipt, or a receipt whose
revision is not a real commit. `--write` regenerates regions atomically.

External regions. A generated region whose content is not a single receipt
number (a status block, a per-arm table) is owned by its own generator, not by
this one. Such regions are declared under the manifest's optional
`external_regions` list:

    {
      "owner": "scripts/status_consistency.py",
      "doc": "README.md",
      "sentinels": ["readme-status"]
    }

This checker then refuses an undeclared region as before, and additionally
refuses a declared external region that the document does not contain, so an
owner cannot claim a block it does not print. It never renders or overwrites an
external region: the owner's own court does that.

Standard library only; reads files, never the network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

MANIFEST_REL = "docs/perf-numbers.json"
SCHEMA = "selvedge/perf-numbers/v1"

BEGIN_RE = re.compile(
    r"^\s*<!--\s*BEGIN GENERATED:\s*(?P<id>[A-Za-z0-9_.:-]+)\s*(?:\([^)]*\))?\s*-->\s*$"
)
END_RE = re.compile(
    r"^\s*<!--\s*END GENERATED:\s*(?P<id>[A-Za-z0-9_.:-]+)\s*-->\s*$"
)
NUMBER_RE = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")


class Failure(Exception):
    pass


# ---------------------------------------------------------------------------
# repository helpers


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def git_tracked(repo: str, rel_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", repo, "ls-files", "--error-unmatch", "--", rel_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise Failure(f"cannot run git: {exc}")
    return result.returncode == 0


def git_commit_exists(repo: str, rev: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{7,64}", rev or ""):
        return False
    result = subprocess.run(
        ["git", "-C", repo, "cat-file", "-e", f"{rev}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def is_git_repo(repo: str) -> bool:
    result = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--git-dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# JSON pointer


def resolve_pointer(document: object, pointer: str, where: str) -> object:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise Failure(f"{where}: JSON pointer {pointer!r} must start with '/'")
    current = document
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise Failure(f"{where}: pointer {pointer!r} missing key {part!r}")
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit():
                raise Failure(f"{where}: pointer {pointer!r} has non-numeric index {part!r}")
            index = int(part)
            if index >= len(current):
                raise Failure(f"{where}: pointer {pointer!r} index {index} out of range")
            current = current[index]
        else:
            raise Failure(f"{where}: pointer {pointer!r} walks into a scalar at {part!r}")
    return current


# ---------------------------------------------------------------------------
# formatting and rendering


def render_value(value: object, fmt: str, where: str) -> str:
    if not isinstance(fmt, str) or "{value" not in fmt:
        raise Failure(f"{where}: format {fmt!r} must contain '{{value' so the number is printed")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise Failure(f"{where}: source value {value!r} is not a number or string")
    try:
        rendered = fmt.format(value=value)
    except (KeyError, IndexError, ValueError) as exc:
        raise Failure(f"{where}: format {fmt!r} cannot render {value!r}: {exc}")
    if "\n" in rendered:
        raise Failure(f"{where}: rendered value must be a single line")
    if not rendered.strip():
        raise Failure(f"{where}: rendered value is empty")
    if isinstance(value, (int, float)):
        if not NUMBER_RE.search(rendered):
            raise Failure(f"{where}: numeric source {value!r} produced no digit in {rendered!r}")
    return rendered


# ---------------------------------------------------------------------------
# document regions


class Region:
    __slots__ = ("id", "begin", "end")

    def __init__(self, region_id: str, begin: int, end: int) -> None:
        self.id = region_id
        self.begin = begin
        self.end = end


def scan_regions(text: str, where: str) -> tuple[list[str], list[Region]]:
    lines = text.splitlines()
    regions: list[Region] = []
    open_begin: tuple[str, int] | None = None
    seen: set[str] = set()
    for index, line in enumerate(lines):
        begin_match = BEGIN_RE.match(line)
        if begin_match:
            if open_begin is not None:
                raise Failure(
                    f"{where}:{index + 1}: nested BEGIN GENERATED:{begin_match.group('id')} "
                    f"inside open region {open_begin[0]!r}"
                )
            region_id = begin_match.group("id")
            if region_id in seen:
                raise Failure(f"{where}:{index + 1}: duplicate generated region {region_id!r}")
            seen.add(region_id)
            open_begin = (region_id, index)
            continue
        end_match = END_RE.match(line)
        if end_match:
            region_id = end_match.group("id")
            if open_begin is None:
                raise Failure(f"{where}:{index + 1}: END GENERATED:{region_id} without BEGIN")
            if open_begin[0] != region_id:
                raise Failure(
                    f"{where}:{index + 1}: END GENERATED:{region_id} closes "
                    f"BEGIN GENERATED:{open_begin[0]}"
                )
            regions.append(Region(region_id, open_begin[1], index))
            open_begin = None
    if open_begin is not None:
        raise Failure(f"{where}:{open_begin[1] + 1}: BEGIN GENERATED:{open_begin[0]} is never closed")
    return lines, regions


def region_body(lines: list[str], region: Region) -> str:
    return "\n".join(lines[region.begin + 1 : region.end]).strip()


def replace_region(lines: list[str], region: Region, rendered: str) -> list[str]:
    return lines[: region.begin + 1] + [rendered] + lines[region.end :]


# ---------------------------------------------------------------------------
# manifest


def load_manifest(path: str) -> dict:
    if not os.path.exists(path):
        raise Failure(f"manifest {path} does not exist")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        raise Failure(f"manifest {path} is not valid JSON: {exc}")
    if not isinstance(manifest, dict):
        raise Failure("manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise Failure(f"manifest schema must be {SCHEMA!r}, got {manifest.get('schema')!r}")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise Failure("manifest 'blocks' must be a non-empty array")
    seen: set[str] = set()
    for index, block in enumerate(blocks):
        where = f"block[{index}]"
        if not isinstance(block, dict):
            raise Failure(f"{where}: must be an object")
        for key in (
            "id",
            "doc",
            "sentinel",
            "source",
            "format",
            "declared_receipt_stamp",
            "declared_revision",
        ):
            if key not in block:
                raise Failure(f"{where}: missing required key {key!r}")
        if not isinstance(block["id"], str) or not block["id"]:
            raise Failure(f"{where}: id must be a non-empty string")
        if block["id"] in seen:
            raise Failure(f"{where}: duplicate block id {block['id']!r}")
        seen.add(block["id"])
        source = block["source"]
        if not isinstance(source, dict) or "receipt" not in source or "pointer" not in source:
            raise Failure(f"{where}: source must be an object with 'receipt' and 'pointer'")
    external = manifest.get("external_regions")
    if external is not None:
        if not isinstance(external, list) or not external:
            raise Failure("manifest 'external_regions' must be a non-empty array when present")
        for index, entry in enumerate(external):
            where = f"external_regions[{index}]"
            if not isinstance(entry, dict):
                raise Failure(f"{where}: must be an object")
            for key in ("owner", "doc", "sentinels"):
                if key not in entry:
                    raise Failure(f"{where}: missing required key {key!r}")
            if not isinstance(entry["owner"], str) or not entry["owner"]:
                raise Failure(f"{where}: owner must be a non-empty string")
            if not isinstance(entry["doc"], str) or not entry["doc"]:
                raise Failure(f"{where}: doc must be a non-empty string")
            sentinels = entry["sentinels"]
            if not isinstance(sentinels, list) or not sentinels:
                raise Failure(f"{where}: sentinels must be a non-empty array")
            for sentinel in sentinels:
                if not isinstance(sentinel, str) or not sentinel:
                    raise Failure(f"{where}: every sentinel must be a non-empty string")
    return manifest


def collect_numbers(value: object, path: str = "") -> list[tuple[str, int | float]]:
    found: list[tuple[str, int | float]] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.append((path or "/", value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(collect_numbers(item, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(collect_numbers(item, f"{path}/{index}"))
    return found


# ---------------------------------------------------------------------------
# resolution


def resolve_block(repo: str, block: dict, require_tracked: bool) -> dict:
    where = f"block {block['id']!r}"
    receipt_rel = block["source"]["receipt"]
    receipt_path = os.path.join(repo, receipt_rel)
    if not os.path.exists(receipt_path):
        raise Failure(f"{where}: receipt {receipt_rel} is missing")
    if require_tracked and not git_tracked(repo, receipt_rel):
        raise Failure(f"{where}: receipt {receipt_rel} is not tracked by git")
    try:
        with open(receipt_path, "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except ValueError as exc:
        raise Failure(f"{where}: receipt {receipt_rel} is not valid JSON: {exc}")

    stamp = receipt.get("stamp")
    revision = receipt.get("revision")
    if not isinstance(stamp, str) or not stamp:
        raise Failure(f"{where}: receipt {receipt_rel} has no 'stamp' field")
    if not isinstance(revision, str) or not revision:
        raise Failure(f"{where}: receipt {receipt_rel} has no 'revision' field")
    if stamp != block["declared_receipt_stamp"]:
        raise Failure(
            f"{where}: declared stamp {block['declared_receipt_stamp']!r} does not match "
            f"receipt stamp {stamp!r} in {receipt_rel}"
        )
    if revision != block["declared_revision"]:
        raise Failure(
            f"{where}: declared revision {block['declared_revision']!r} does not match "
            f"receipt revision {revision!r} in {receipt_rel}"
        )
    if require_tracked and not git_commit_exists(repo, revision):
        raise Failure(
            f"{where}: receipt revision {revision!r} is not a commit in this repository"
        )

    value = resolve_pointer(receipt, block["source"]["pointer"], f"{where}: {receipt_rel}")
    rendered = render_value(value, block["format"], where)
    return {
        "id": block["id"],
        "doc": block["doc"],
        "sentinel": block["sentinel"],
        "receipt": receipt_rel,
        "pointer": block["source"]["pointer"],
        "value": value,
        "rendered": rendered,
    }


def markdown_files(repo: str) -> list[str]:
    paths = ["README.md"]
    docs_dir = os.path.join(repo, "docs")
    for root, _dirs, files in os.walk(docs_dir):
        for name in sorted(files):
            if name.endswith(".md"):
                paths.append(os.path.relpath(os.path.join(root, name), repo))
    return [path for path in paths if os.path.exists(os.path.join(repo, path))]


# ---------------------------------------------------------------------------
# check / write


def check(repo: str, manifest_rel: str) -> int:
    manifest_path = os.path.join(repo, manifest_rel)
    manifest = load_manifest(manifest_path)
    tracked = is_git_repo(repo)
    blocks = manifest["blocks"]

    resolved = [resolve_block(repo, block, tracked) for block in blocks]
    by_sentinel = {entry["sentinel"]: entry for entry in resolved}
    if len(by_sentinel) != len(resolved):
        raise Failure("two blocks declare the same sentinel id")

    # every declared doc must be tracked too: the doc is the published artifact
    if tracked:
        for entry in resolved:
            if not git_tracked(repo, entry["doc"]):
                raise Failure(f"block {entry['id']!r}: doc {entry['doc']} is not tracked by git")

    doc_lines: dict[str, list[str]] = {}
    doc_regions: dict[str, dict[str, Region]] = {}
    for doc in sorted({entry["doc"] for entry in resolved}):
        doc_path = os.path.join(repo, doc)
        if not os.path.exists(doc_path):
            raise Failure(f"doc {doc} does not exist")
        with open(doc_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        lines, regions = scan_regions(text, doc)
        doc_lines[doc] = lines
        doc_regions[doc] = {region.id: region for region in regions}

    problems: list[str] = []
    for entry in resolved:
        regions = doc_regions[entry["doc"]]
        if entry["sentinel"] not in regions:
            problems.append(
                f"block {entry['id']!r}: sentinel {entry['sentinel']!r} not found in {entry['doc']}"
            )
            continue
        body = region_body(doc_lines[entry["doc"]], regions[entry["sentinel"]])
        if body != entry["rendered"]:
            problems.append(
                f"block {entry['id']!r}: stale region in {entry['doc']} "
                f"(doc has {body!r}, source renders {entry['rendered']!r})"
            )

    # external regions are declared, not derived: they must exist as declared
    external_declared = {
        (entry["doc"], sentinel)
        for entry in manifest.get("external_regions", [])
        for sentinel in entry["sentinels"]
    }
    external_seen: set[tuple[str, str]] = set()

    # (e) a sentinel with no manifest block
    for doc in markdown_files(repo):
        doc_path = os.path.join(repo, doc)
        with open(doc_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            _lines, regions = scan_regions(text, doc)
        except Failure as exc:
            problems.append(str(exc))
            continue
        for region in regions:
            if (doc, region.id) in external_declared:
                external_seen.add((doc, region.id))
                continue
            if region.id not in by_sentinel:
                problems.append(
                    f"{doc}: generated region {region.id!r} has no manifest block"
                )
            elif by_sentinel[region.id]["doc"] != doc:
                problems.append(
                    f"{doc}: region {region.id!r} is declared for {by_sentinel[region.id]['doc']}"
                )

    # declared external regions that the document does not print
    for doc, sentinel in sorted(external_declared):
        if (doc, sentinel) in external_seen:
            continue
        doc_path = os.path.join(repo, doc)
        if not os.path.exists(doc_path):
            problems.append(f"external region {sentinel!r}: doc {doc} does not exist")
            continue
        with open(doc_path, "r", encoding="utf-8") as handle:
            _lines, regions = scan_regions(handle.read(), doc)
        if sentinel not in {region.id for region in regions}:
            problems.append(
                f"external region {sentinel!r} is declared for {doc} but the document "
                "does not contain it"
            )

    # (d) a manifest number that no block prints
    all_regions = "\n".join(entry["rendered"] for entry in resolved)
    for pointer, number in collect_numbers(manifest):
        needle = str(int(number)) if isinstance(number, float) and number.is_integer() else str(number)
        if needle not in all_regions:
            problems.append(
                f"manifest number {number!r} at {pointer} is not printed by any block"
            )

    if problems:
        for problem in problems:
            print(f"perf_numbers: {problem}", file=sys.stderr)
        print(
            f"perf_numbers: {len(problems)} problem(s); run --write after fixing the sources",
            file=sys.stderr,
        )
        return 1

    print(
        f"perf_numbers: OK — {len(resolved)} block(s) verified against receipts "
        f"({', '.join(sorted(by_sentinel))})"
    )
    return 0


def write(repo: str, manifest_rel: str) -> int:
    manifest_path = os.path.join(repo, manifest_rel)
    manifest = load_manifest(manifest_path)
    tracked = is_git_repo(repo)
    resolved = [resolve_block(repo, block, tracked) for block in manifest["blocks"]]

    updates: dict[str, list[str]] = {}
    changed: list[str] = []
    for doc in sorted({entry["doc"] for entry in resolved}):
        doc_path = os.path.join(repo, doc)
        if not os.path.exists(doc_path):
            raise Failure(f"doc {doc} does not exist")
        with open(doc_path, "r", encoding="utf-8") as handle:
            original = handle.read()
        lines, regions = scan_regions(original, doc)
        by_id = {region.id: region for region in regions}
        entries = [entry for entry in resolved if entry["doc"] == doc]
        for entry in entries:
            if entry["sentinel"] not in by_id:
                raise Failure(
                    f"block {entry['id']!r}: sentinel {entry['sentinel']!r} not found in {doc}"
                )
        # replace from the end so earlier line indices stay valid
        working = list(lines)
        for entry in sorted(entries, key=lambda item: by_id[item["sentinel"]].begin, reverse=True):
            region = by_id[entry["sentinel"]]
            current = region_body(working, region)
            if current != entry["rendered"]:
                changed.append(f"{entry['doc']}:{entry['sentinel']}")
            working = replace_region(working, region, entry["rendered"])
        new_text = "\n".join(working) + ("\n" if original.endswith("\n") else "")
        if new_text != original:
            updates[doc] = working

    for doc, working in updates.items():
        doc_path = os.path.join(repo, doc)
        directory = os.path.dirname(doc_path) or "."
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".perf-", delete=False
        )
        try:
            handle.write("\n".join(working) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(handle.name, doc_path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise

    if changed:
        print(f"perf_numbers: updated {len(changed)} region(s): {', '.join(changed)}")
    else:
        print(f"perf_numbers: {len(resolved)} region(s) already current")
    return 0


def list_blocks(repo: str, manifest_rel: str) -> int:
    manifest = load_manifest(os.path.join(repo, manifest_rel))
    tracked = is_git_repo(repo)
    width = max(len(block["id"]) for block in manifest["blocks"])
    for block in manifest["blocks"]:
        entry = resolve_block(repo, block, tracked)
        print(
            f"{entry['id']:<{width}}  {entry['receipt']}  {entry['pointer']}  -> {entry['rendered']}"
        )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate and check published numbers")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify, never mutate")
    group.add_argument("--write", action="store_true", help="regenerate regions atomically")
    group.add_argument("--list", action="store_true", help="print id -> source -> value")
    parser.add_argument("--manifest", default=MANIFEST_REL)
    parser.add_argument("--repo", default=repo_root())
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    try:
        if args.check:
            return check(repo, args.manifest)
        if args.write:
            return write(repo, args.manifest)
        return list_blocks(repo, args.manifest)
    except Failure as exc:
        print(f"perf_numbers: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
