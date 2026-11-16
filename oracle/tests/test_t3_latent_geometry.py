"""T3 (P10.2) — CPU tests for the latent-geometry court.

Every assertion is arithmetic or a measured property of a tiny layer: the sweep
budget, the shared-projection accounting, zero-init parity, determinism, the
Jacobian rank proxy against its analytic bound, the utility-diversity estimator
against known structure, and the admission mirror. No quality or timing number
is claimed here; those need the GPU court (`--train` / `--report`).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ORACLE, "train"))

import latent_geometry as geometry  # noqa: E402


def tiny_spec(**overrides) -> geometry.LatentSpec:
    values = dict(d_model=16, d_latent=4, experts=4, k=2, expert_width=4,
                  layers=2)
    values.update(overrides)
    return geometry.LatentSpec(**values)


# ---------------------------------------------------------------------------
# the sweep arithmetic


def test_every_sweep_point_is_square_matched_and_under_the_ceiling():
    report = geometry.sweep_report()
    budget = report["budget_per_layer"]
    assert [point["d_latent"] for point in report["points"]] == list(
        geometry.LATENT_DIMS)
    for point in report["points"]:
        assert point["expert_width"] == point["d_latent"]
        assert point["near_square"] is True
        assert point["experts"] == geometry.BANK_FACTOR * point["k"]
        assert point["active_per_layer"] <= budget
        assert point["active_utilisation"] > 0.9
        assert point["within_ceiling"] is True
        # `k` is the largest that fits: one more active expert would exceed it
        candidate = geometry.latent_spec(
            point["d_latent"], budget_per_layer=budget - 3 * point["d_latent"] ** 2,
            d_model=1024)
        assert candidate.k <= point["k"]
    # a smaller latent buys more (tiny) experts; a wider latent buys fewer
    ks = [point["k"] for point in report["points"]]
    assert ks == sorted(ks, reverse=True)
    # the active cost is matched, so modelled FLOPs are matched too
    for point in report["points"]:
        assert point["decode_overhead"]["flops_ratio_vs_reference"] <= 1.0


def test_the_reference_point_is_the_t2_x_arithmetic():
    reference = geometry.reference_spec()
    # 28 x 4 x 3 x 512 x 1024 + 28 x 1024 x 8
    assert reference.active_total == 176_390_144
    assert reference.active_per_layer == 6_299_648
    assert reference.latent is False
    assert reference.shared_elements == 0
    assert reference.jacobian_rank_bound == 1024
    # 28 x 8 x 3 x 512 x 1024 + 28 x 1024 x 8
    assert reference.accessible_total == 948_600_832 - 596_049_920


def test_the_shared_projection_is_counted_once_per_layer_not_per_expert():
    spec = geometry.latent_spec(256)
    assert spec.shared_elements == 2 * 1024 * 256
    assert spec.active_per_layer == (spec.shared_elements
                                     + spec.k * 3 * 256 * 256
                                     + 1024 * spec.experts)
    assert spec.accessible_per_layer == (spec.shared_elements
                                         + spec.experts * 3 * 256 * 256
                                         + 1024 * spec.experts)
    # the accessible set is the bank, not the k read
    assert spec.accessible_per_layer > spec.active_per_layer


def test_the_local_rank_bound_distinguishes_latent_from_reference():
    for dim in geometry.LATENT_DIMS:
        spec = geometry.latent_spec(dim)
        assert spec.jacobian_rank_bound == dim
        assert spec.expert_rank_bound == dim
    reference = geometry.reference_spec()
    assert reference.jacobian_rank_bound == reference.d_model
    assert reference.expert_rank_bound == 512


# ---------------------------------------------------------------------------
# the layer


def test_the_layer_is_zero_at_init_and_deterministic():
    spec = tiny_spec()
    first = geometry.LatentGeometryLayer(spec, seed=5)
    second = geometry.LatentGeometryLayer(spec, seed=5)
    hidden = torch.randn(2, 5, spec.d_model,
                         generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        assert torch.equal(first(hidden), torch.zeros_like(hidden))
        assert torch.equal(second(hidden), torch.zeros_like(hidden))
    assert all(torch.equal(left.detach(), right.detach())
               for left, right in zip(first.parameters(), second.parameters()))
    with torch.no_grad():
        first.down.normal_(std=0.1)
    output = first(hidden)
    assert torch.equal(output, first(hidden))
    assert float(output.detach().abs().max()) > 0.0


def test_parameter_count_is_the_documented_arithmetic():
    spec = tiny_spec()
    layer = geometry.LatentGeometryLayer(spec, seed=1)
    expected = (2 * spec.d_model * spec.d_latent
                + spec.experts * 3 * spec.expert_width * spec.d_latent
                + spec.d_model * spec.experts)
    assert sum(parameter.numel() for parameter in layer.parameters()) == expected
    assert layer.active_elements() == spec.active_per_layer


def _reference_dispatch(layer, hidden, selected):
    """A direct per-token loop: same maths, a different execution order."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    latent = layer.project(flat) if layer.project is not None else flat
    output = torch.zeros_like(latent)
    for token in range(flat.shape[0]):
        for expert in selected[token].tolist():
            output[token] = output[token] + layer._expert(latent[token:token + 1],
                                                          expert)[0]
    result = layer.unproject(output) if layer.unproject is not None else output
    return result.reshape(hidden.shape)


def test_the_dispatch_matches_a_direct_loop_and_uses_exactly_k_slots():
    spec = tiny_spec(d_latent=4, experts=6, k=3, expert_width=4)
    layer = geometry.LatentGeometryLayer(spec, seed=2)
    with torch.no_grad():
        layer.down.normal_(std=0.1)
    hidden = torch.randn(1, 4, spec.d_model,
                         generator=torch.Generator().manual_seed(3))
    selected = torch.topk(layer.router(hidden.reshape(-1, spec.d_model)),
                          spec.k, dim=-1).indices
    ours = layer(hidden)
    reference = _reference_dispatch(layer, hidden, selected)
    assert torch.allclose(ours, reference, atol=1e-6, rtol=1e-5)
    # excluding one selected expert changes the output but keeps every other slot
    excluded = layer(hidden, exclude=int(selected[0, 0]))
    assert not torch.equal(ours, excluded)


# ---------------------------------------------------------------------------
# the Jacobian rank proxy


def test_the_jacobian_rank_proxy_measures_the_documented_bound():
    spec = tiny_spec(d_model=16, d_latent=4, experts=4, k=2, expert_width=4)
    layer = geometry.LatentGeometryLayer(spec, seed=4)
    with torch.no_grad():
        layer.down.normal_(std=0.5)
    report = geometry.jacobian_rank_proxy(layer, torch.randn(16))
    assert report["bound"] == spec.jacobian_rank_bound == 4
    assert 0 < report["rank_at_tolerance"] <= 4
    assert report["effective_rank"] <= 4 + 1e-6
    expert = geometry.expert_jacobian_rank(layer, torch.randn(4), 0)
    assert expert["bound"] == 4
    assert 0 < expert["rank_at_tolerance"] <= 4


def test_the_reference_layer_reaches_full_rank_at_the_same_small_scale():
    spec = geometry.reference_spec(d_model=16, width=8, k=2, experts=4, layers=1)
    layer = geometry.LatentGeometryLayer(spec, seed=4)
    with torch.no_grad():
        layer.down.normal_(std=0.2)
    report = geometry.jacobian_rank_proxy(layer, torch.randn(16))
    assert report["bound"] == 16
    assert report["rank_at_tolerance"] > 4


# ---------------------------------------------------------------------------
# utility diversity


def test_utility_diversity_recovers_known_structure():
    tokens = 64
    generator = torch.Generator().manual_seed(9)
    signal = torch.randn(tokens, 1, generator=generator)
    identical = signal.repeat(1, 4)
    report = geometry.utility_diversity(identical)
    assert report["mean_pairwise_correlation"] == pytest.approx(1.0, abs=1e-5)
    # the mean-utility share is uniform, so the effective expert count is m;
    # this estimator measures the spread of the mean, not co-movement
    assert report["effective_experts"] == pytest.approx(4.0, abs=1e-4)
    assert report["mean_absolute_pairwise_correlation"] == pytest.approx(1.0,
                                                                        abs=1e-5)

    independent = torch.randn(tokens, 4, generator=generator)
    report = geometry.utility_diversity(independent)
    assert abs(report["mean_pairwise_correlation"]) < 0.1
    assert report["effective_experts"] > 2.0
    # one expert dominating the mean utility raises the top-1 share
    dominated = independent.clone()
    dominated[:, 0] += 5.0
    assert geometry.utility_diversity(dominated)["top1_share"] > \
        geometry.utility_diversity(independent)["top1_share"]


def test_utility_diversity_treats_masked_entries_as_missing():
    utilities = torch.tensor([[1.0, 2.0, float("nan")],
                              [1.5, 2.5, float("nan")],
                              [2.0, 3.0, float("nan")]])
    mask = torch.tensor([[True, True, False]] * 3)
    report = geometry.utility_diversity(utilities, mask)
    assert report["pairs"] == 1
    assert report["mean_pairwise_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert report["tokens"] == 3 and report["experts"] == 3


# ---------------------------------------------------------------------------
# decode overhead and the admission mirror


def test_decode_overhead_decomposes_the_shared_projection():
    spec = geometry.latent_spec(256)
    report = geometry.decode_overhead(spec)
    assert report["experts_per_token"] == spec.k
    assert 0.0 < report["shared_projection_fraction"] < 1.0
    assert 0.0 < report["routing_overhead_fraction"] < 1.0
    assert report["per_layer_read_bytes_bf16"] == 2 * spec.active_per_layer
    assert report["flops_ratio_vs_reference"] <= 1.0
    reference = geometry.reference_spec()
    assert geometry.decode_overhead(reference)["shared_projection_fraction"] == 0.0


def test_the_admission_config_mirrors_the_spec_and_stays_under_the_ceiling():
    for spec in geometry.sweep_specs():
        config = geometry.latent_admission_config(spec)
        mirror = geometry.python_mirror(config)
        assert config["arm"] == f"T3-L{spec.d_latent}"
        assert mirror["within_budget"] is True
        # the core's active count plus the latent arm's added active elements
        assert mirror["active_unique_params"] == 596_049_920 + spec.active_total
        assert config["total_accessible"] == (596_049_920 + spec.accessible_total)
        assert config["total_accessible"] <= geometry.CEILING
        # the shared P/Q run for every token, so they are execution slices
        projected = [use for use in config["execution_uses"]
                     if ".latent.project." in use["identity"]["tensor"]]
        assert len(projected) == spec.layers


def test_a_spec_round_trips_through_the_emitted_json(tmp_path):
    spec = geometry.latent_spec(256)
    relative = str(tmp_path / "L256.json")
    payload = {"schema": geometry.SCHEMA_SPEC, **spec.report(), **spec.as_dict()}
    (tmp_path / "L256.json").write_text(__import__("json").dumps(payload))
    loaded = geometry.LatentSpec.load(relative)
    assert loaded == spec
    assert loaded.report() == spec.report()


def test_the_sweep_float_arithmetic_is_deterministic():
    first = geometry.sweep_report()
    second = geometry.sweep_report()
    assert first == second
    assert geometry.latent_spec(128) == geometry.latent_spec(128)
