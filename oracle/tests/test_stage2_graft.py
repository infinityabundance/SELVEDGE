"""CPU tests for the stage-2 stream guard, graft linear algebra and smoke math.

The GPU receipts prove what ran; these prove that the pieces which decide what
*should* run are correct: the leakage index catches a copied document, the ridge
map recovers a known linear map, PCA recovers a known low-rank subspace, row
grafting stores means per address with the right counts, and the smoke delta
arithmetic uses independent standard errors.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import graft, memory, smoke, stream  # noqa: E402


def test_shingle_hashes_are_stable_and_width_aware():
    text = "the quick brown fox jumps over the lazy dog"
    a = stream.shingle_hashes(stream.words(text))
    b = stream.shingle_hashes(stream.words(text))
    assert a == b
    assert len(a) == len(stream.words(text)) - stream.SHINGLE_WIDTH + 1
    assert stream.shingle_hashes(["too", "short"]) == set()


def test_leakage_index_catches_a_copy_and_passes_an_unrelated_text():
    item = ("eval.arrow", "alpha beta gamma delta epsilon zeta eta theta iota kappa")
    index = stream.LeakageIndex([item])
    copied, source = index.worst("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    assert copied == pytest.approx(1.0)
    assert source == "eval.arrow"
    unrelated, _ = index.worst("completely different words that share nothing at all here")
    assert unrelated == 0.0


def test_ridge_map_recovers_a_known_linear_map():
    torch.manual_seed(0)
    teacher = torch.randn(4000, 6)
    true_w = torch.randn(6, 4)
    student = teacher @ true_w
    learned, report = graft.fit_ridge(teacher, student, ridge=1e-6, device="cpu",
                                      chunk=1024)
    relative = float((teacher @ learned - student).norm() / student.norm())
    assert relative < 1e-3
    assert report["heldout_relative_frobenius_error"] < 1e-3
    assert report["fit_tokens"] == 3200


def test_pca_recovers_a_known_low_rank_subspace():
    torch.manual_seed(1)
    rank = 8
    basis_true = torch.linalg.qr(torch.randn(32, rank))[0]
    coefficients = torch.randn(5000, rank)
    mean_true = torch.randn(32)
    targets = coefficients @ basis_true.T + mean_true
    mean, basis, report = graft.fit_pca(targets, rank, device="cpu", chunk=1024)
    sample_mean = targets.mean(0)
    assert torch.allclose(mean, sample_mean, atol=1e-4)
    assert report["explained_variance"] > 0.999
    # the reconstructed subspace is the same, up to sign and rotation
    centered = targets - sample_mean
    projection_error = float(
        (centered - (centered @ basis @ basis.T)).norm() / centered.norm())
    assert projection_error < 1e-3


def test_graft_rows_stores_means_and_counts():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    coefficients = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    rows, counts = graft.graft_rows(tokens, coefficients, rows=8, site=2,
                                    device="cpu")
    assert int(counts.sum()) == tokens.numel() * memory.HEADS
    # 48 reads into 8 rows: at least one row is hit more than once
    assert int((counts > 1).sum()) >= 1
    # a row aggregates every occurrence that hashes to it, across all heads
    sums = torch.zeros(8, 4, dtype=torch.float64)
    hits = torch.zeros(8, dtype=torch.int64)
    for head, order in enumerate(memory.ORDERS):
        addresses = memory.address(tokens, 2, order, head, 8)[0]
        for position, address in enumerate(addresses.tolist()):
            sums[address] += coefficients[position].double()
            hits[address] += 1
    expected = sums[hits > 0] / hits[hits > 0, None].double()
    assert torch.allclose(rows[hits > 0], expected.float(), atol=1e-6)
    assert torch.equal(counts, hits)


def test_graft_rows_keeps_empty_rows_zero():
    tokens = torch.tensor([[7]], dtype=torch.int64)
    coefficients = torch.ones(1, 4)
    rows, counts = graft.graft_rows(tokens, coefficients, rows=16, site=0,
                                    device="cpu")
    assert int((counts == 0).sum()) == 16 - int((counts > 0).sum())
    assert torch.equal(rows[counts == 0], torch.zeros_like(rows[counts == 0]))


def test_smoke_delta_arithmetic_uses_independent_errors():
    assert smoke.binomial_stderr(0.5, 100) == pytest.approx(0.05)
    assert smoke.binomial_stderr(0.5, None) is None
    combined = smoke.combined_stderr(1.0, 2.0)
    assert combined == pytest.approx(math.sqrt(5.0))


def test_smoke_score_map_uses_binomial_when_the_harness_reports_na():
    receipt = {"tasks": [{"task": "mmlu_pro", "score": 0.25, "stderr": None,
                          "n_samples": 100, "primary_metric": "exact_match"}]}
    scores = smoke.score_map(receipt)
    assert scores["mmlu_pro"]["stderr"] == pytest.approx(
        math.sqrt(0.25 * 0.75 / 100))
