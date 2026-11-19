"""S60 — the corpus gate re-runs the verifier and checks the examples hash.

A row count is not a verification. The accepted file's stored `extracted` and
`verification_method` are metadata from the generator; the gate must recompute
the verdict with `frontier.rft.verify` on every accepted solution, and it must
refuse a built examples file whose bytes differ from the SHA-256 its own receipt
declares.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _load_module():
    path = os.path.join(ROOT, "scripts", "s57_verify_stage2a.py")
    spec = importlib.util.spec_from_file_location("s57_verify_stage2a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sv = _load_module()


def _row(identifier: str, dataset: str, solution: str, target: str) -> dict:
    return {"id": identifier, "dataset": dataset, "solution": solution,
            "target": target, "prompt": "", "prompt_ids": [1],
            "solution_ids": [2, 3], "extracted": target,
            "max_new_tokens": 2048}


def test_reverification_accepts_every_correct_solution():
    rows = [
        _row("gsm8k-train:1", "gsm8k", "The answer is 72.\n#### 72", "72"),
        _row("math-train:1", "math", "so \\boxed{\\frac{1}{2}}",
             "\\frac{1}{2}"),
    ]
    result = sv.reverify(rows)
    assert result["all_correct"] is True
    assert result["verdicts"] == {"correct": 2}


def test_reverification_refuses_a_wrong_stored_label():
    # The stored `extracted` field says 72, but the text says 71: the
    # re-verification must notice and fail the corpus.
    rows = [_row("gsm8k-train:2", "gsm8k", "#### 71", "72")]
    result = sv.reverify(rows)
    assert result["all_correct"] is False
    assert result["verdicts"] == {"wrong": 1}
    assert result["failures"][0]["id"] == "gsm8k-train:2"


def test_examples_hash_must_match_the_receipt(tmp_path, monkeypatch):
    root = tmp_path
    examples = root / "examples.jsonl"
    examples.write_text('{"id": "a"}\n')
    accepted = root / "accepted.jsonl"
    accepted.write_text('{"id": "a"}\n')
    digest = hashlib.sha256(examples.read_bytes()).hexdigest()
    accepted_digest = hashlib.sha256(accepted.read_bytes()).hexdigest()
    receipt = root / "receipt.json"
    receipt.write_text(json.dumps({
        "examples": {"sha256": digest},
        "inputs": ["accepted.jsonl"],
    }))
    monkeypatch.setattr(sv, "ROOT", str(root))
    block = sv.examples_check("examples.jsonl", "receipt.json",
                              "accepted.jsonl", accepted_digest)
    assert block["matches"] is True
    assert block["accepted_input_matches"] is True

    examples.write_text('{"id": "tampered"}\n')
    block = sv.examples_check("examples.jsonl", "receipt.json",
                              "accepted.jsonl", accepted_digest)
    assert block["matches"] is False


def test_a_missing_examples_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    try:
        sv.examples_check("absent.jsonl", None, "accepted.jsonl", "0" * 64)
    except SystemExit as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("a missing examples file must fail closed")


def test_declared_inputs_are_read_from_a_stage2b_corpus_receipt():
    built = {
        "sources": {
            "student": {
                "rows": 3,
                "sources": [
                    {"source": "results/a.jsonl", "rows": 2},
                    {"source": "results/b.jsonl", "rows": 1},
                ],
            },
            "teacher": None,
        },
        "examples": {"sha256": "0" * 64},
    }
    assert sv._declared_inputs(built) == ["results/a.jsonl", "results/b.jsonl"]


def test_acceptance_is_reconstructed_from_the_named_generation_receipts(
        tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"acceptance": {
        "by_dataset": {"gsm8k": {"sampled": 8, "correct": 4, "kept": 4,
                                  "truncated": 3, "loop_cut": 1,
                                  "verdicts": {"correct": 4, "wrong": 1}}}},
        "totals": {"sampled": 8, "correct": 4, "kept": 4,
                   "truncated": 3, "loop_cut": 1}}))
    second.write_text(json.dumps({"acceptance": {
        "by_dataset": {"math": {"sampled": 6, "correct": 2, "kept": 2,
                                 "truncated": 4, "loop_cut": 0,
                                 "verdicts": {"correct": 2, "wrong": 2}}}},
        "totals": {"sampled": 6, "correct": 2, "kept": 2,
                   "truncated": 4, "loop_cut": 0}}))
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    block = sv.acceptance_block({"sources": {"generation_receipts": [
        {"path": "first.json"}, {"path": "second.json"}]}})
    assert block["totals"]["sampled"] == 14
    assert block["totals"]["kept"] == 6
    assert block["totals"]["loop_cut"] == 1
    assert abs(block["kept_rate"] - 6 / 14) < 1e-12
    assert abs(block["truncation_rate"] - 7 / 14) < 1e-12
    assert block["by_dataset"]["gsm8k"]["verdicts"]["correct"] == 4
