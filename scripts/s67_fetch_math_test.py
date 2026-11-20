#!/usr/bin/env python3
"""S67: fetch and pin the full Hendrycks MATH test split (5,000 problems).

The resolution instrument registered for S67 is the canonical MATH **test**
split, not a 500-item subsample. Only the test parquet files are downloaded;
the train split is already pinned separately as the calibration source. Every
file is pinned by the dataset revision and by the SHA-256 declared in the
Hugging Face LFS pointer, and a mismatch fails closed.

Output (all under the gitignored artifact cache):

* ``results/artifacts/math-test/<subject>.parquet``  the seven test files
* ``results/artifacts/math-test/math-test-5000.jsonl``
  one row per problem: ``problem_id``, ``subject``, ``level``, ``question``,
  ``solution``, ``gold`` (the boxed answer parsed from the reference solution)
* a tracked receipt under ``results/datasets/`` naming every file, size,
  sha256, revision and the row count per subject.

Usage (network, CPU; oracle court):
    SELVEDGE_MEM=4g scripts/court --net oracle -- \
      python3 scripts/s67_fetch_math_test.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "oracle"))

REPO = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
OUT_DIR = "results/artifacts/math-test"
CORPUS = "results/artifacts/math-test/math-test-5000.jsonl"
RECEIPT = "results/datasets/math-test-fetch.json"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree() -> dict[str, dict]:
    url = (f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
           "?recursive=true")
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    return {entry["path"]: entry for entry in entries
            if entry.get("type") == "file"}


def download(path: str, lfs_sha256: str, destination: str) -> dict:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response, \
            open(destination, "wb") as handle:
        for block in iter(lambda: response.read(1 << 20), b""):
            handle.write(block)
    actual = sha256_file(destination)
    if actual != lfs_sha256:
        raise SystemExit(
            f"s67-math-test: {path} sha256 mismatch: {actual} != {lfs_sha256}")
    return {"path": path, "file": os.path.relpath(destination, ROOT),
            "bytes": os.path.getsize(destination), "sha256": actual}


def last_boxed(solution: str) -> str | None:
    """The final balanced ``\\boxed{}`` of a reference solution.

    The canonical MATH reference solution may box intermediate results; the
    final box is the answer. Model outputs keep the stricter
    ``rft.extract(..., 'math')`` rule (two distinct boxes is ambiguous), which
    is why this is a separate gold rule for the *reference*, not a relaxation
    of scoring.
    """
    from frontier import rft

    answers = rft._boxed_answers(solution)
    return answers[-1] if answers else None


def validate_against_math500(rows: list[dict]) -> dict:
    """Check the gold rule on the 500 problems the program already scored.

    MATH-500 is a subset of the 5,000-problem test split. Its rows carry an
    explicit ``answer`` field, so it is an independent check that "last boxed
    answer in the reference solution" recovers the canonical gold.
    """
    import pyarrow.ipc as ipc

    from frontier import rft

    cache = os.environ.get("HF_HOME", "/hf-cache")
    path = os.path.join(
        cache, "datasets", "HuggingFaceH4___math-500", "default", "0.0.0",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "math-500-test.arrow")
    if not os.path.isfile(path):
        return {"available": False, "path": path}
    reference = ipc.open_stream(path).read_all().to_pylist()

    def normalise(text: str) -> str:
        return " ".join(str(text).split())

    by_question = {normalise(row["question"]): row for row in rows}
    matched = 0
    disagreements = []
    missing = []
    for row in reference:
        target = by_question.get(normalise(row["problem"]))
        if target is None:
            missing.append(row.get("unique_id"))
            continue
        matched += 1
        gold = str(row.get("answer"))
        if not rft.answers_equal(target["gold"], gold):
            disagreements.append({"unique_id": row.get("unique_id"),
                                  "expected": gold,
                                  "extracted": target["gold"]})
    return {"available": True, "path": os.path.relpath(path, ROOT),
            "items": len(reference), "matched": matched,
            "missing": missing[:20], "missing_count": len(missing),
            "disagreements": disagreements[:20],
            "disagreement_count": len(disagreements)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--receipt", default=RECEIPT)
    args = parser.parse_args(argv)

    import pyarrow.parquet as pq
    from frontier import rft

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    entries = tree()

    files = []
    rows = []
    per_subject = {}
    for subject in SUBJECTS:
        path = f"{subject}/test-00000-of-00001.parquet"
        entry = entries.get(path)
        if entry is None:
            raise SystemExit(f"s67-math-test: {path} is absent at {REVISION}")
        lfs = (entry.get("lfs") or {}).get("oid")
        if not lfs:
            raise SystemExit(f"s67-math-test: {path} has no LFS sha256")
        destination = os.path.join(out_dir, f"{subject}.parquet")
        files.append(download(path, lfs, destination))
        table = pq.read_table(destination).to_pylist()
        kept = 0
        for index, row in enumerate(table):
            gold = last_boxed(row.get("solution", ""))
            if gold is None:
                raise SystemExit(
                    f"s67-math-test: {subject}[{index}] has no boxed answer in "
                    "the reference solution; refusing to score a partial "
                    "instrument")
            rows.append({
                "problem_id": f"math-test:{subject}:{index}",
                "subject": subject,
                "level": row.get("level"),
                "question": row["problem"],
                "solution": row["solution"],
                "gold": gold,
            })
            kept += 1
        per_subject[subject] = kept

    total = sum(per_subject.values())
    if total != 5000:
        raise SystemExit(
            f"s67-math-test: expected 5000 test problems, found {total}")

    validation = validate_against_math500(rows)
    if validation.get("available"):
        if validation["matched"] != 500 or validation["disagreement_count"]:
            raise SystemExit(
                "s67-math-test: the last-boxed gold rule does not reproduce "
                f"MATH-500 ({validation['matched']}/500 matched, "
                f"{validation['disagreement_count']} disagreements)")

    corpus_path = os.path.join(out_dir, "math-test-5000.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    payload = {
        "schema": "selvedge/s67-math-test-fetch/v1",
        "kind": "math-test-corpus",
        "created_utc": utcnow(),
        "repo_id": REPO,
        "revision": REVISION,
        "split": "test",
        "subjects": per_subject,
        "items": total,
        "files": files,
        "corpus": {
            "path": os.path.relpath(corpus_path, ROOT),
            "bytes": os.path.getsize(corpus_path),
            "sha256": sha256_file(corpus_path),
        },
        "gold_rule": "last balanced boxed answer in the reference solution; "
                     "validated against the MATH-500 answer field on the 500 "
                     "overlapping problems; a missing box is a hard error, "
                     "never a dropped problem",
        "overlap_validation": validation,
        "claims": {
            "train_split_not_downloaded": True,
            "test_split_is_the_canonical_hendrycks_math_test": True,
            "no_approximate_decoder": True,
        },
    }
    receipt_path = os.path.join(ROOT, args.receipt)
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"s67-math-test: {total} problems -> {args.out_dir}")
    print(f"s67-math-test: receipt -> {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
