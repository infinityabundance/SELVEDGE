"""P12.2 — tests for the `H*` target diagnostics, on synthetic ground truth.

Every case below has a known answer, so the tests check the *estimators* rather
than a stored number:

* a utility matrix that is exactly an expert-identity constant must be fully
  explained by identity (share 1, margin over noise large);
* a utility matrix whose token variation is one latent factor must show a
  first-component energy share near 1 and a near-unity mean correlation;
* independent noise must show a first-component share near `1/m` and a mean
  correlation near 0;
* two experts that differ by less than one label standard deviation must be
  reported as indistinguishable (`margin_over_noise < 1`);
* the router split `z = w.mean(h) + w.(h - mean(h))` must be exact, and an
  identity-dominated router must be reported as such.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

import t2_diagnostics as diagnostics  # noqa: E402


def _identity_utilities(positions: int = 400, experts: int = 8,
                        spread: float = 1.0, noise: float = 0.01,
                        seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mean = torch.linspace(0.0, spread, experts)
    jitter = torch.randn(positions, experts, generator=generator) * noise
    return mean.reshape(1, -1) + jitter


def _rank_one_utilities(positions: int = 400, experts: int = 8,
                        seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(positions, 1, generator=generator)
    loadings = torch.linspace(-1.0, 1.0, experts).reshape(1, -1)
    return latent @ loadings + 0.001 * torch.randn(positions, experts,
                                                   generator=generator)


def _independent_utilities(positions: int = 4000, experts: int = 8,
                           seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(positions, experts, generator=generator)


def _router(experts: int = 8, hidden: int = 16, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(experts, hidden, generator=generator) * 0.02


def _features(positions: int = 400, hidden: int = 16, seed: int = 13) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(positions, hidden, generator=generator)


# ---------------------------------------------------------------------------
# identity versus token structure in the labels


def test_identity_dominated_labels_are_explained_by_identity():
    utilities = _identity_utilities()
    result = diagnostics.analyse_layer(_features(utilities.shape[0]), utilities,
                                       _router())
    assert result["identity_argmax_share"] > 0.99
    assert result["margin_over_noise"] > 5.0
    # the columns do not co-move: all of the structure is the per-expert level,
    # which correlation (invariant to location) cannot see.
    assert abs(result["mean_pairwise_utility_correlation"]) < 0.1
    # and a nearly constant per-expert level leaves little token structure
    assert result["first_component_energy_share"] < 0.2


def test_rank_one_labels_are_dominated_by_one_common_factor():
    utilities = _rank_one_utilities()
    result = diagnostics.analyse_layer(_features(utilities.shape[0]), utilities,
                                       _router())
    assert result["first_component_energy_share"] > 0.95
    # every column is the same latent factor up to its loadings' sign
    assert result["mean_absolute_pairwise_utility_correlation"] > 0.9
    assert result["margin_over_noise"] < 1.0


def test_independent_labels_show_no_common_factor():
    utilities = _independent_utilities()
    result = diagnostics.analyse_layer(_features(utilities.shape[0]), utilities,
                                       _router())
    assert result["first_component_energy_share"] < 0.5
    assert abs(result["mean_pairwise_utility_correlation"]) < 0.1
    assert result["mean_absolute_pairwise_utility_correlation"] < 0.1


# ---------------------------------------------------------------------------
# per-expert standardisation


def test_per_expert_standardisation_changes_the_target_when_offsets_dominate():
    utilities = _identity_utilities(spread=5.0, noise=0.05)
    result = diagnostics.analyse_layer(_features(utilities.shape[0]), utilities,
                                       _router())
    # removing each expert's own mean and spread is a large change to the
    # distribution the router is fitted to.
    assert result["kl_absolute_vs_relative_mean"] > 0.1
    # the absolute means themselves are far apart, which is what the
    # standardised target deletes.
    means = torch.tensor(result["label_mean"])
    assert float(means.max() - means.min()) > 4.0


def test_standardisation_is_a_no_op_when_all_labels_share_one_scale():
    generator = torch.Generator().manual_seed(17)
    utilities = torch.randn(600, 4, generator=generator)
    # give every expert the same mean and spread: per-expert standardisation
    # then equals a global affine map, and the softmax target cannot change.
    utilities = (utilities - utilities.mean(dim=0)) / utilities.std(dim=0, unbiased=False)
    result = diagnostics.analyse_layer(_features(600), utilities, _router(experts=4))
    assert result["kl_absolute_vs_relative_mean"] < 1e-9


# ---------------------------------------------------------------------------
# margin against noise


def test_margin_over_noise_separates_indistinguishable_from_clearly_better():
    positions, experts = 600, 8
    generator = torch.Generator().manual_seed(19)
    noise = torch.randn(positions, experts, generator=generator)
    # four experts sit 0.1 standard deviations above the rest: inside the noise.
    close = noise.clone()
    close[:, :4] += 0.1
    result_close = diagnostics.analyse_layer(_features(positions), close, _router())
    assert result_close["margin_over_noise"] < 1.0
    # the same four sit 5 standard deviations above: outside the noise.
    far = noise.clone()
    far[:, :4] += 5.0
    result_far = diagnostics.analyse_layer(_features(positions), far, _router())
    assert result_far["margin_over_noise"] > 1.0
    assert result_far["margin_over_noise"] > result_close["margin_over_noise"]


# ---------------------------------------------------------------------------
# the router decomposition


def test_router_split_is_exact_and_reports_identity_dominance():
    router = torch.zeros(8, 16)
    for expert in range(8):
        router[expert, expert] = float(expert + 1)
    # features constant in the router's span and varying outside it: the router's
    # logits are then a pure expert constant with no token term.
    features = torch.cat([torch.ones(300, 8), _features(300, 8, seed=29)], dim=1)
    result = diagnostics.analyse_layer(features, _independent_utilities(300), router)
    assert result["router_identity_bias_share"] > 0.999
    assert result["router_top1_is_identity_winner"] == 1.0
    assert result["router_top4_overlap_with_identity"] == 1.0
    # a genuinely varying router is not identity dominated
    varying = _features(300, 16, seed=31)
    other = diagnostics.analyse_layer(varying, _independent_utilities(300), _router())
    assert other["router_identity_bias_share"] < 0.9
    assert 0.0 <= other["router_top1_is_identity_winner"] <= 1.0


def test_router_that_matches_the_utility_oracle_captures_everything():
    positions, experts, hidden = 200, 8, 16
    generator = torch.Generator().manual_seed(37)
    features = torch.randn(positions, hidden, generator=generator)
    # utilities whose ranking is exactly the router's ranking
    router = torch.randn(experts, hidden, generator=generator)
    logits = features @ router.T
    utilities = logits + torch.randn(positions, experts, generator=generator) * 1e-6
    result = diagnostics.analyse_layer(features, utilities, router, k=4)
    assert result["router_top4_overlap_with_utility"] == 1.0
    assert abs(result["router_additive_gap_mean"]) < 1e-4
    # the chosen set contains a negative marginal exactly when the fourth largest
    # utility is negative: the "least harmful omission" quantity, checked
    # against a direct computation.
    ranked = utilities.sort(dim=-1, descending=True).values
    expected_token = float((ranked[:, 3] < 0).double().mean())
    expected_pair = float((ranked[:, :4] < 0).double().mean())
    assert abs(result["router_chosen_negative_marginal_pair_share"]
               - expected_pair) < 1e-9
    assert abs(result["router_least_harmful_share"] + expected_token - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# determinism, shapes and fail-closed behaviour


def test_the_bootstrap_is_deterministic():
    values = torch.randn(500, generator=torch.Generator().manual_seed(41))
    first = diagnostics.bootstrap_mean_ci(values)
    second = diagnostics.bootstrap_mean_ci(values)
    assert first == second
    assert first["low"] <= first["mean"] <= first["high"]


def test_analyse_covers_every_layer_and_refuses_a_missing_router():
    payload = {
        "features": {0: _features(120, 8, seed=43), 1: _features(120, 8, seed=47)},
        "utilities": {0: _independent_utilities(120, 4),
                      1: _independent_utilities(120, 4)},
        "manifest": {"arm": "H*"},
    }
    routers = {0: _router(experts=4, hidden=8), 1: _router(experts=4, hidden=8)}
    result = diagnostics.analyse(payload, routers, k=2)
    assert result["layers"] == 2
    assert set(result["per_layer"]) == {"0", "1"}
    assert result["summary"]["router_identity_bias_share"]["mean"] is not None
    with pytest.raises(SystemExit):
        diagnostics.analyse(payload, {0: routers[0]}, k=2)


def test_missing_label_artifact_exits_non_zero(tmp_path):
    out = tmp_path / "receipt.json"
    code = diagnostics.main(["--labels", str(tmp_path / "absent.pt"),
                             "--routers", str(tmp_path / "absent-2.pt"),
                             "--out", str(out)])
    assert code == 66
    assert not out.exists()
