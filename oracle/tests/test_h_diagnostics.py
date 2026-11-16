"""Tests for the H-diagnostics estimators, on synthetic ground truth.

Every case has an answer known by construction, so the tests check that the
estimators compute what `oracle/eval/h_diagnostics.py` claims rather than
reproducing a stored number:

* a router whose expert directions all point the same way must report a
  near-one mean absolute cosine and a near-one top-1 energy share, while an
  orthogonal router must report near-zero cosine and an effective rank of `m`;
* an expert whose direction is a hundredth of the median must be flagged as a
  dead-expert candidate, and a normal one must not;
* uniform router logits must give full usage entropy, and a router that always
  selects the same four experts must report one modal top-4 set and four dead
  experts;
* row usage must count distinct rows, collisions and coverage exactly;
* mutual information must be zero for independent codes, `H(y)` bits for a
  copied code, and the permutation test must call a copy significant and two
  independent draws not significant;
* the outcome partition must reproduce cell counts, per-cell accuracies and the
  task-weighted composite exactly;
* path magnitudes must be zero when the projection is zero, and the memory
  share of the path norm must follow the two modules' own norms;
* the initialiser reconstruction must be the draw the checkpoint code makes,
  which is what makes displacement measurable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import h_diagnostics as diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# initialisation


def test_init_block_reproduces_the_constructor_draw():
    shape = (8, 16)
    first = diagnostics.init_block(1001, 2000, 3, shape)
    again = diagnostics.init_block(1001, 2000, 3, shape)
    other = diagnostics.init_block(1001, 2000, 4, shape)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert 0.015 < float(first.std()) < 0.025


def test_router_stats_orthogonal_router_has_full_rank_and_identity_displacement():
    weight = torch.eye(8)
    init = weight.clone()
    stats = diagnostics.router_stats(weight, init)
    assert stats["effective_rank"] == pytest.approx(8.0, abs=1e-6)
    assert stats["top1_energy_share"] == pytest.approx(1.0 / 8.0, abs=1e-6)
    assert stats["mean_absolute_pairwise_cosine"] == pytest.approx(0.0, abs=1e-9)
    assert stats["displacement_relative"] == pytest.approx(0.0, abs=1e-12)
    assert stats["cosine_to_init"] == pytest.approx(1.0, abs=1e-12)
    assert stats["dead_expert_candidates"] == []


def test_router_stats_collapsed_router_reports_one_direction():
    direction = torch.arange(1.0, 9.0).reshape(1, 8).repeat(8, 1)
    weight = direction.clone()
    stats = diagnostics.router_stats(weight, torch.ones_like(weight))
    assert stats["top1_energy_share"] == pytest.approx(1.0, abs=1e-9)
    assert stats["mean_absolute_pairwise_cosine"] == pytest.approx(1.0, abs=1e-9)
    assert stats["effective_rank"] == pytest.approx(1.0, abs=1e-6)


def test_router_stats_flags_a_negligible_expert_direction():
    weight = torch.ones(8, 4)
    init = torch.ones(8, 4)
    weight[5] = weight[5] * 0.01
    stats = diagnostics.router_stats(weight, init)
    assert stats["dead_expert_candidates"] == [5]
    assert stats["norm_per_expert"][5] < stats["norm_per_expert"][0]


# ---------------------------------------------------------------------------
# usage


def test_usage_stats_uniform_choice_is_full_entropy():
    # Each expert wins an equal block, so the marginal usage is exactly uniform.
    logits = torch.zeros(800, 8)
    for expert in range(8):
        logits[expert * 100:(expert + 1) * 100, expert] = 10.0
    stats = diagnostics.usage_stats(logits, k=1)
    assert stats["normalised_entropy"] == pytest.approx(1.0, abs=1e-9)
    assert stats["kl_to_uniform"] == pytest.approx(0.0, abs=1e-9)
    assert stats["coefficient_of_variation"] == pytest.approx(0.0, abs=1e-9)
    assert stats["dead_experts_used"] == []
    assert stats["counts"] == [100] * 8
    assert stats["distinct_topk_sets"] == 8


def test_usage_stats_tied_logits_break_to_the_first_experts():
    # Under `topk` all-zero logits are tied, so every position gets one set; the
    # estimator must report the set it actually sees, whatever `topk`'s order.
    stats = diagnostics.usage_stats(torch.zeros(600, 8))
    assert stats["distinct_topk_sets"] == 1
    assert len(stats["modal_set"]) == diagnostics.K
    assert stats["modal_set_share"] == pytest.approx(1.0)
    # The four experts the router never selects are exactly one dead-expert set.
    assert sorted(stats["modal_set"] + stats["dead_experts_used"]) == list(range(8))


def test_usage_stats_constant_choice_is_one_set_and_four_dead_experts():
    logits = torch.zeros(500, 8)
    logits[:, :4] = 5.0
    logits[:, 4:] = -5.0
    stats = diagnostics.usage_stats(logits)
    assert stats["distinct_topk_sets"] == 1
    assert stats["modal_set_share"] == pytest.approx(1.0, abs=1e-12)
    assert sorted(stats["modal_set"]) == [0, 1, 2, 3]
    assert sorted(stats["dead_experts_used"]) == [4, 5, 6, 7]
    assert stats["max_over_mean"] == pytest.approx(2.0, abs=1e-9)


def test_usage_stats_records_the_expert_counts_of_a_known_router():
    logits = torch.zeros(8, 8)
    for expert in range(8):
        logits[expert:expert + 1, expert] = 10.0
    stats = diagnostics.usage_stats(logits, k=1)
    assert stats["counts"] == [1] * 8
    assert stats["normalised_entropy"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# row usage and displacement


def test_row_usage_counts_distinct_rows_and_collisions():
    indices = torch.zeros(10, 2, dtype=torch.int64)
    indices[5:, 0] = 7
    indices[:, 1] = torch.arange(10)
    result = diagnostics.row_usage(indices, rows=1024)
    assert result["reads"] == 20
    assert result["distinct_rows_read"] == 10  # rows 0..9; the extra 7s collide
    assert result["collision_reads"] == 10
    assert result["coverage_of_table"] == pytest.approx(10 / 1024)
    assert result["two_head_same_row_share"] == pytest.approx(0.2)  # positions 0 and 7
    assert result["max_row_reads"] == 6  # row 7: head two once, head one five times


def test_row_usage_reports_the_two_head_collision():
    indices = torch.zeros(4, 2, dtype=torch.int64)
    result = diagnostics.row_usage(indices, rows=64)
    assert result["two_head_same_row_share"] == pytest.approx(1.0)
    assert result["distinct_rows_read"] == 1


def test_displacement_stats_identifies_updated_rows():
    init = torch.ones(8, 4, dtype=torch.bfloat16)
    table = init.clone()
    table[2] = 2.0
    table[6] = -3.0
    stats = diagnostics.displacement_stats(table, init)
    assert stats["rows"] == 8
    assert stats["rows_bitwise_unchanged"] == 6
    assert stats["total_displacement"] > 0
    # The moved row 6 carries 8 of the 10 units of displacement norm.
    assert stats["top_1pct_displacement_share"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# mutual information


def test_mutual_information_is_zero_for_independent_codes():
    generator = np.random.default_rng(7)
    first = generator.integers(0, 4, size=4000)
    second = generator.integers(0, 4, size=4000)
    assert diagnostics.mutual_information(first, second) < 0.02


def test_mutual_information_is_the_entropy_of_a_copied_code():
    generator = np.random.default_rng(11)
    first = generator.integers(0, 4, size=4000)
    values = diagnostics.mutual_information(first, first.copy())
    counts = np.bincount(first, minlength=4) / first.size
    expected = float(-(counts * np.log2(counts)).sum())
    assert values == pytest.approx(expected, abs=1e-9)


def test_permutation_test_separates_a_copy_from_independence():
    generator = np.random.default_rng(13)
    first = generator.integers(0, 4, size=500)
    independent = generator.integers(0, 4, size=500)
    copied = diagnostics.mi_permutation_p(first, first.copy(), 200, 5)
    unrelated = diagnostics.mi_permutation_p(first, independent, 200, 5)
    assert copied["p_value"] < 0.05
    assert unrelated["p_value"] > 0.05


def test_popularity_codes_bucket_by_row_frequency():
    # Row 1 read once, row 2 twice, row 3 three times: codes must follow.
    indices = torch.zeros(6, 2, dtype=torch.int64)
    indices[0, 0] = 1
    indices[1:3, 0] = 2
    indices[3:6, 0] = 3
    codes = diagnostics.popularity_codes(indices, levels=2)
    assert codes[0] == 0
    assert codes[1] == codes[2]
    assert codes[3] == codes[5]
    assert codes[1] != codes[3]


def test_popularity_codes_are_constant_when_every_row_is_equally_popular():
    indices = torch.zeros(10, 2, dtype=torch.int64)
    indices[:5, 0] = 1
    indices[5:, 0] = 2
    codes = diagnostics.popularity_codes(indices, levels=2)
    assert len(set(codes.tolist())) == 1


# ---------------------------------------------------------------------------
# outcomes


def _items(first: dict, second: dict, reference: dict) -> dict[str, dict]:
    """Build three arms with the requested success lists per task."""
    arms = {"left": {}, "right": {}, "reference": {}}
    for task, _field in diagnostics.TASKS:
        size = len(first[task])
        arms["left"][task] = {f"{task}-{index}": bool(value)
                              for index, value in enumerate(first[task])}
        arms["right"][task] = {f"{task}-{index}": bool(value)
                               for index, value in enumerate(second[task])}
        arms["reference"][task] = {f"{task}-{index}": bool(value)
                                   for index, value in enumerate(reference[task])}
        assert size == len(second[task]) == len(reference[task])
    return arms


def test_outcome_cells_reproduce_counts_and_cell_accuracies():
    # One task of four items: both, left-only, right-only, neither.
    first = {"humaneval": [1, 1, 0, 0], "gsm8k": [1, 1, 0, 0],
             "truthfulqa": [1, 1, 0, 0]}
    second = {"humaneval": [1, 0, 1, 0], "gsm8k": [1, 0, 1, 0],
              "truthfulqa": [1, 0, 1, 0]}
    reference = {"humaneval": [1, 1, 0, 0], "gsm8k": [1, 1, 0, 0],
                 "truthfulqa": [1, 1, 0, 0]}
    arms = _items(first, second, reference)
    cells = diagnostics.outcome_cells(arms, "left", "right", "reference")
    table = cells["per_task"]["humaneval"]
    assert table["both"]["n"] == 1
    assert table["left_only"]["n"] == 1
    assert table["right_only"]["n"] == 1
    assert table["neither"]["n"] == 1
    assert table["both"]["reference_accuracy"] == pytest.approx(1.0)
    assert table["left_only"]["reference_accuracy"] == pytest.approx(1.0)
    assert table["right_only"]["reference_accuracy"] == pytest.approx(0.0)
    assert table["neither"]["reference_accuracy"] == pytest.approx(0.0)
    pooled = cells["pooled"]
    assert pooled["both"]["n"] == 3
    assert pooled["both"]["reference_composite"] == pytest.approx(1.0)
    assert pooled["union"]["n"] == 9
    assert pooled["disjoint"]["n"] == 6
    # The disjoint cell holds one reference-correct item per task (the
    # left-only one), so its composite is exactly one half.
    assert pooled["disjoint"]["reference_composite"] == pytest.approx(0.5)


def test_outcome_cells_composite_weights_tasks_equally():
    # `humaneval` is solved everywhere, `gsm8k` nowhere: the composite must be
    # the task mean, 0.5, not the item mean.
    first = {"humaneval": [1, 1, 1, 1, 1, 1], "gsm8k": [1, 1, 1, 1, 1, 1],
             "truthfulqa": [1, 1, 1, 1, 1, 1]}
    second = {"humaneval": [1, 1, 1, 1, 1, 1], "gsm8k": [0, 0, 0, 0, 0, 0],
              "truthfulqa": [1, 1, 1, 1, 1, 1]}
    reference = {"humaneval": [1] * 6, "gsm8k": [0] * 6, "truthfulqa": [1] * 6}
    arms = _items(first, second, reference)
    cells = diagnostics.outcome_cells(arms, "left", "right", "reference")
    assert cells["pooled"]["union"]["reference_composite"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# memory path


def _memory_state(hidden: int = 4, rows: int = 8, value_dim: int = 2,
                  reads: int = 2, gate_scale: float = 1.0,
                  projection_scale: float = 1.0) -> dict:
    table = torch.arange(rows * value_dim, dtype=torch.float32).reshape(rows, value_dim)
    proj = torch.eye(hidden, reads * value_dim) * projection_scale
    gate = torch.zeros(hidden) if gate_scale == 0 else torch.full((hidden,), gate_scale)
    return {"table": table, "proj": proj, "gate": gate}


def test_path_magnitudes_are_zero_when_the_projection_is_zero():
    features = torch.ones(16, 4)
    indices = torch.zeros(16, 2, dtype=torch.int64)
    state = _memory_state(gate_scale=0.0, projection_scale=0.0)
    result = diagnostics.path_magnitudes(features, None, state, indices)
    assert result["memory_output_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result["gate"]["mean"] == pytest.approx(0.5, abs=1e-9)


def test_path_magnitudes_scale_with_the_projection():
    features = torch.ones(16, 4)
    indices = torch.zeros(16, 2, dtype=torch.int64)
    small = diagnostics.path_magnitudes(features, None,
                                        _memory_state(projection_scale=0.1), indices)
    large = diagnostics.path_magnitudes(features, None,
                                        _memory_state(projection_scale=1.0), indices)
    assert large["memory_output_rms"] == pytest.approx(
        10.0 * small["memory_output_rms"], rel=1e-6)
    assert large["memory_output_over_input_rms"] > small["memory_output_over_input_rms"]


def test_path_magnitudes_gate_opens_with_a_positive_read_direction():
    features = torch.ones(16, 4)
    indices = torch.zeros(16, 2, dtype=torch.int64)
    state = _memory_state(gate_scale=0.0)
    state["gate"] = torch.full((4,), 3.0)
    result = diagnostics.path_magnitudes(features, None, state, indices)
    assert result["gate"]["mean"] > 0.99


def test_path_magnitudes_memory_share_tracks_the_two_paths():
    features = torch.randn(64, 4, generator=torch.Generator().manual_seed(2))
    indices = torch.zeros(64, 2, dtype=torch.int64)
    state = _memory_state()
    bank = {"router": torch.zeros(8, 4), "gate": torch.zeros(8, 2, 4),
            "up": torch.zeros(8, 2, 4), "down": torch.zeros(8, 4, 2)}
    result = diagnostics.path_magnitudes(features, bank, state, indices)
    # A zero bank contributes nothing, so the memory is the whole path.
    assert result["bank_output_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result["memory_share_of_path_norm"]["median"] == pytest.approx(1.0)


def test_gate_router_alignment_finds_the_aligned_expert():
    gate = torch.tensor([0.0, 0.0, 1.0, 0.0])
    router = torch.eye(4)
    alignment = diagnostics.gate_router_alignment(gate, router)
    assert alignment["argmax_absolute_expert"] == 2
    assert alignment["per_expert"][2] == pytest.approx(1.0)
    assert alignment["per_expert"][0] == pytest.approx(0.0)


def test_gate_stats_reports_the_known_fractions():
    values = torch.tensor([0.05, 0.2, 0.6, 0.95])
    stats = diagnostics.gate_stats(values)
    assert stats["fraction_above_half"] == pytest.approx(0.5)
    assert stats["fraction_below_tenth"] == pytest.approx(0.25)
    assert stats["n"] == 4


# ---------------------------------------------------------------------------
# coverage scan


def test_coverage_scan_matches_a_direct_union_of_indices():
    import memory as memory_module

    tokens = torch.arange(1, 1 + 64, dtype=torch.int64)
    result = diagnostics.coverage_scan(tokens, layer=3, rows=1 << 12, block=32)
    seen = set()
    for start in range(0, 64, 32):
        window = tokens[start:start + 32].reshape(1, 32)
        for order in diagnostics.ORDERS:
            seen.update(memory_module.ngram_index(window, 3, order,
                                                  1 << 12).reshape(-1).tolist())
    assert result["tokens"] == 64
    assert result["distinct_rows"] == len(seen)
    assert result["reads"] == 64 * len(diagnostics.ORDERS)


def test_coverage_scan_is_empty_for_an_empty_stream():
    result = diagnostics.coverage_scan(torch.zeros(0, dtype=torch.int64), 3, 16)
    assert result["distinct_rows"] == 0


# ---------------------------------------------------------------------------
# checkpoint discovery and rendering


def test_hstar_checkpoints_sorts_by_step(tmp_path):
    for step in (200, 400, 300):
        (tmp_path / f"step-{step:06d}.pt").write_bytes(b"")
    (tmp_path / "run-meta.json").write_text("{}")
    found = diagnostics.hstar_checkpoints(str(tmp_path))
    assert [step for step, _path in found] == [200, 300, 400]


def test_render_markdown_runs_on_a_minimal_receipt():
    receipt = {
        "schema": diagnostics.SCHEMA,
        "stamp": "20260101T000000Z",
        "revision": "deadbeef",
        "created_utc": "2026-01-01T00:00:00Z",
        "scope": {"forward_passes_run": 0, "gpu_used": False},
        "inputs": {"labels": {"path": "x", "sha256": "ab" * 32}},
        "outcomes": {},
        "routers": {},
        "usage": {},
        "memory": {},
        "interaction": {},
        "deferred": [],
        "limits": ["a limit"],
    }
    text = diagnostics.render_markdown(receipt)
    assert "# H-diagnostics" in text
    assert "does not establish" in text
    assert "a limit" in text
