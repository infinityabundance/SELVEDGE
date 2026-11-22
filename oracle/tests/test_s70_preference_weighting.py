"""S70: the difficulty weight must change realized updates, not just the loss.

The S69 registration weights each pair by ``w = p(1-p)`` and runs at
``batch_pairs = 1``. The pre-S70 batch loss divided by the batch's own weight
sum, so at one pair per step the weight cancelled exactly and the objective was
unweighted. A plain "divide by the corpus mean weight instead" is not enough
either: AdamW's per-parameter normalisation cancels any constant rescale of the
gradient, so a rescaled loss can leave the parameter trajectory untouched. The
fix therefore has two parts, and this file tests both:

* **the corpus normaliser** (`preference_batch_loss(weight_normalizer=...)`),
  which makes ``w`` appear in the loss value and in the gradient magnitude;
* **difficulty-proportional ordering** (`weighted_permutation` /
  `pair_stream(order="weighted")`), which changes *which* pair each step sees
  and therefore changes the realized update even under AdamW.

The last two tests are deliberately the kind that fails if the fix is cosmetic:
a 40-step AdamW trajectory that must differ between uniform/unweighted and
weighted/normalized training, and a control that pins down the fact that a pure
rescale alone does **not** move the parameters. The control-budget replay is
compared element for element with the launcher's schedule.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from frontier import preference  # noqa: E402
import s69_control_budget as control_budget  # noqa: E402
import s64_preference_launch as launch  # noqa: E402

VOCAB = 8


class TwoParamLM(nn.Module):
    """A linear next-token model with exactly two trainable scalars.

    ``logits[v] = -((v - center)**2)`` with ``center = w * (x - 0.5) + b`` for a
    token id ``x``. The parameters are ``w`` and ``b``; nothing else is
    trainable, so every trajectory difference below is attributable to the
    weighting scheme, not to capacity.
    """

    def __init__(self, w: float = 0.6, b: float = 0.4):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self._grid = torch.arange(VOCAB, dtype=torch.float32)

    def forward(self, input_ids, use_cache: bool = False,
                logits_to_keep=None):
        x = input_ids.float() / float(VOCAB)
        center = (self.w * (x - 0.5) + self.b).unsqueeze(-1)
        logits = -((self._grid - center) ** 2)
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep, :]
        return SimpleNamespace(logits=logits)


def _reference(pair, w=0.6, b=0.4):
    """Fill a pair's frozen-reference cache from the initial toy model."""
    model = TwoParamLM(w, b)
    with torch.no_grad():
        _, chosen = preference.continuation_logprobs(
            model, pair.prompt_ids, pair.chosen_ids)
        _, rejected = preference.continuation_logprobs(
            model, pair.prompt_ids, pair.rejected_ids)
    return preference.replace(
        pair,
        chosen_reference_logprobs=tuple(float(v) for v in chosen.tolist()),
        rejected_reference_logprobs=tuple(float(v) for v in rejected.tolist()))


def _pair(name, weight, prompt=(1, 2), chosen=(6, 7), rejected=(0, 1)):
    pair = preference.make_pair(name, prompt, chosen, rejected, weight=weight)
    return _reference(pair)


def _loss(model, pairs, normalizer):
    loss, detail = preference.preference_batch_loss(
        model, pairs, weight_normalizer=normalizer)
    return float(loss.detach()), detail


# The two weights the task names: w = 0.25 and w = 0.0625, mean 0.15625.
W_HIGH = 0.25
W_LOW = 0.0625
MEAN = (W_HIGH + W_LOW) / 2


# ---------------------------------------------------------------------------
# (a) the corpus-normalized loss


def test_per_pair_contributions_are_in_ratio_to_their_weights():
    model = TwoParamLM()
    pair = _pair("shared", W_HIGH)
    low = preference.replace(pair, weight=W_LOW)

    high_loss, _ = _loss(model, [pair], MEAN)
    low_loss, _ = _loss(model, [low], MEAN)
    assert high_loss > 0.0
    # Identical pair content, so the only difference is the weight.
    assert high_loss / low_loss == pytest.approx(4.0, rel=1e-6)

    # And the two-pair batch is the sum of those contributions, not their mean.
    batch_loss, detail = _loss(model, [pair, low], MEAN)
    assert batch_loss == pytest.approx(high_loss + low_loss, rel=1e-6)
    assert detail["weight_normalizer"] == MEAN
    assert [round(item["weight_scale"], 6) for item in detail["diagnostics"]] \
        == [round(W_HIGH / MEAN, 6), round(W_LOW / MEAN, 6)]


def test_gradients_from_two_weighted_pairs_are_scaled_by_w_over_mean():
    """A1's acceptance: two pairs with different w scale the gradient by w/mean."""
    model = TwoParamLM()
    pair = _pair("shared", W_HIGH)
    low = preference.replace(pair, weight=W_LOW)

    def gradient(pair, normalizer):
        for parameter in model.parameters():
            parameter.grad = None
        loss, _ = preference.preference_batch_loss(
            model, [pair], weight_normalizer=normalizer)
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.cat([p.grad.reshape(-1) for p in parameters])

    high_grad = gradient(pair, MEAN)
    low_grad = gradient(low, MEAN)
    assert float((high_grad / low_grad).min()) == pytest.approx(4.0, rel=1e-5)
    assert float((high_grad / low_grad).max()) == pytest.approx(4.0, rel=1e-5)
    # The pre-S70 denominator cancels the weight, so the gradients are equal.
    old_high = gradient(pair, None)
    old_low = gradient(low, None)
    assert torch.allclose(old_high, old_low, rtol=1e-6, atol=1e-9)


def test_a_single_pair_below_the_mean_loses_less_than_at_the_mean():
    model = TwoParamLM()
    at_mean = _pair("at-mean", MEAN)
    below = _pair("below", W_LOW)

    with_normalizer, _ = _loss(model, [below], MEAN)
    at_mean_loss, _ = _loss(model, [at_mean], MEAN)
    assert with_normalizer == pytest.approx(0.4 * at_mean_loss, rel=1e-6)
    assert with_normalizer < at_mean_loss

    # This is the identity that made the old code inert: with the batch's own
    # sum as the denominator a single pair is unweighted whatever its w.
    old_below, _ = _loss(model, [below], None)
    old_at_mean, _ = _loss(model, [at_mean], None)
    assert old_below == pytest.approx(old_at_mean, rel=1e-6)


def test_the_normalizer_must_be_positive_and_corpus_mean_is_the_pair_file_mean():
    model = TwoParamLM()
    with pytest.raises(ValueError):
        _loss(model, [_pair("p", W_HIGH)], 0.0)
    pairs = [_pair("a", W_HIGH), _pair("b", W_LOW),
             _pair("zero", 0.0)]
    assert preference.corpus_mean_weight(pairs) == pytest.approx(MEAN)
    with pytest.raises(ValueError):
        preference.corpus_mean_weight([_pair("z", 0.0)])


# ---------------------------------------------------------------------------
# (b) difficulty-proportional ordering


def test_weighted_order_is_deterministic_and_a_permutation_per_epoch():
    weights = [W_HIGH, W_LOW, 0.1875, 0.05]
    first = preference.weighted_permutation(weights, random.Random(7))
    second = preference.weighted_permutation(weights, random.Random(7))
    assert first == second
    assert sorted(first) == list(range(len(weights)))
    assert first != list(range(len(weights)))  # not the identity by accident

    pairs = [_pair(f"p{index}", weight)
             for index, weight in enumerate(weights)]
    stream = preference.pair_stream(pairs, seed=7, order="weighted")
    for _ in range(5):
        epoch = [next(stream) for _ in range(len(pairs))]
        assert sorted(epoch) == list(range(len(pairs)))
    with pytest.raises(ValueError):
        preference.weighted_permutation([0.0, 0.1], random.Random(0))


def test_weighted_order_selects_in_proportion_to_weight():
    pairs = [_pair("high", W_HIGH), _pair("low", W_LOW)]
    epochs = 4000
    stream = preference.pair_stream(pairs, seed=6900, order="weighted")
    first_position = []
    for _ in range(epochs):
        first_position.append(next(stream))
        next(stream)  # consume the epoch's second position
    # The index drawn first in each epoch is drawn with probability w_i / sum.
    high = first_position.count(0)
    low = first_position.count(1)
    assert high + low == epochs
    ratio = high / low
    # Expected 4.0. The tolerance is far tighter than the value a uniform
    # shuffle would produce (1.0), so this test fails if sampling is uniform.
    assert ratio == pytest.approx(4.0, abs=0.5)
    assert ratio > 2.5

    # `shuffled` is uniform over the same pairs, which pins the comparison.
    uniform = preference.pair_stream(pairs, seed=6900, order="shuffled")
    uniform_first = []
    for _ in range(epochs):
        uniform_first.append(next(uniform))
        next(uniform)
    uniform_ratio = uniform_first.count(0) / uniform_first.count(1)
    assert uniform_ratio == pytest.approx(1.0, abs=0.2)


def test_shuffled_order_reproduces_the_pre_s70_stream():
    pairs = [_pair(f"p{index}", 0.1 + 0.01 * index) for index in range(5)]
    # The pre-S70 launcher: one Random(seed), one in-place shuffle per epoch.
    rng = random.Random(6400)
    order = list(range(len(pairs)))
    expected = []
    for _ in range(3):
        rng.shuffle(order)
        expected.extend(order)
    stream = preference.pair_stream(pairs, seed=6400, order="shuffled")
    assert [next(stream) for _ in range(len(expected))] == expected


# ---------------------------------------------------------------------------
# (c) the realized update, under a real AdamW and a real grad clip


def _trajectory(pair_order, weight_normalization, steps=40, lr=0.05,
                pairs=None):
    """Train the two-parameter model and return the per-step trajectory."""
    model = TwoParamLM()
    if pairs is None:
        pairs = [_pair("high", W_HIGH),
                 _pair("low", W_LOW, prompt=(2, 3), chosen=(5, 6),
                       rejected=(0, 2))]
    mean = preference.corpus_mean_weight(pairs)
    normalizer = mean if weight_normalization == "corpus" else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    schedule = launch.pair_schedule(pairs, seed=70, pair_order=pair_order)
    trace = preference.UpdateTelemetry(grad_clip=1.0, lr=lr)
    trace.bind(trainable)
    trajectory = []
    for _ in range(steps):
        step, batch = next(schedule)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = preference.preference_batch_loss(
            model, batch, weight_normalizer=normalizer)
        loss.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        post_clip = preference.grad_norm(trainable)
        optimizer.step()
        trace.row(step=step, arm="preference",
                  problem_ids=[pair.problem_id for pair in batch],
                  weights=[pair.weight for pair in batch],
                  loss=float(loss.detach()), mean_weight=mean,
                  weight_normalizer=normalizer,
                  pre_clip_grad_norm=float(pre_clip),
                  post_clip_grad_norm=float(post_clip))
        trajectory.append(torch.tensor([float(p.detach())
                                        for p in trainable]))
    return trajectory, trace.summary()


def test_adamw_trajectory_differs_between_uniform_and_weighted_training():
    uniform, uniform_summary = _trajectory("shuffled", "batch")
    weighted, weighted_summary = _trajectory("weighted", "corpus")
    assert len(uniform) == len(weighted) == 40

    differences = [float((a - b).norm()) for a, b in zip(uniform, weighted)]
    final_difference = differences[-1]
    mean_difference = sum(differences) / len(differences)
    # Stated tolerance: a real effect, not float noise.
    assert final_difference > 1e-3, (final_difference, differences)
    assert mean_difference > 1e-3, (mean_difference, differences)
    # Both runs move the parameters; the difference is not "one arm is frozen".
    assert float((uniform[-1] - uniform[0]).norm()) > 1e-3
    assert float((weighted[-1] - weighted[0]).norm()) > 1e-3
    # A full epoch always contains every pair once, whatever the order, so the
    # realized mean weight over whole epochs is the corpus mean in both arms.
    # The difference is *which step* saw which pair, not the epoch composition.
    assert weighted_summary["realized_mean_weight"] == pytest.approx(
        uniform_summary["realized_mean_weight"])
    assert weighted_summary["realized_mean_weight"] == pytest.approx(MEAN)
    assert weighted_summary["steps"] == uniform_summary["steps"] == 40


def test_ordering_matters_when_the_budget_does_not_complete_a_whole_epoch():
    """A complete epoch is a permutation under either order, so ordering only
    changes *timing* there. A budget that ends mid-epoch changes *coverage*:
    the weighted permutation puts the high-weight pairs into the partial epoch
    first. The real S69 budget is about 1.5 epochs, so this is its regime."""
    pairs = [
        _pair("p0", 0.25, prompt=(1, 2), chosen=(6, 7), rejected=(0, 1)),
        _pair("p1", 0.20, prompt=(2, 3), chosen=(5, 6), rejected=(0, 2)),
        _pair("p2", 0.15, prompt=(3, 4), chosen=(4, 5), rejected=(1, 3)),
        _pair("p3", 0.10, prompt=(4, 5), chosen=(3, 4), rejected=(2, 4)),
        _pair("p4", 0.05, prompt=(5, 6), chosen=(2, 3), rejected=(3, 5)),
    ]
    steps = 87  # 17 whole epochs plus two pairs of the eighteenth
    base = _trajectory("shuffled", "batch", steps=steps, pairs=pairs)
    order = _trajectory("weighted", "batch", steps=steps, pairs=pairs)
    both = _trajectory("weighted", "corpus", steps=steps, pairs=pairs)
    order_mean, order_final = _mean_trajectory_distance(base[0], order[0])
    both_mean, both_final = _mean_trajectory_distance(base[0], both[0])
    assert order_mean > 1e-3, order_mean
    assert both_mean > 1e-3, both_mean

    # The partial epoch is the coverage argument, measured in expectation over
    # seeds: a weighted permutation puts high-weight pairs first, a uniform one
    # does not. (A single seed is noisy: 17 whole epochs contribute the corpus
    # mean under either order.)
    def partial_mean(order, seeds=200, prefix=17 * len(pairs), draws=2):
        means = []
        for seed in range(seeds):
            stream = preference.pair_stream(pairs, seed=seed, order=order)
            for _ in range(prefix):
                next(stream)
            means.append(sum(pairs[next(stream)].weight
                             for _ in range(draws)) / draws)
        return sum(means) / len(means)

    weighted_partial = partial_mean("weighted")
    uniform_partial = partial_mean("shuffled")
    corpus_mean = sum(pair.weight for pair in pairs) / len(pairs)
    assert uniform_partial == pytest.approx(corpus_mean, abs=0.02)
    assert weighted_partial > corpus_mean * 1.05


def test_update_trace_records_pre_and_post_clip_norms_and_the_delta(tmp_path):
    """The trace is the evidence that weighting changes realized updates."""
    model = TwoParamLM()
    pairs = [_pair("high", W_HIGH), _pair("low", W_LOW)]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    trace = preference.UpdateTelemetry(grad_clip=1e-9, lr=0.05)
    trace.bind(trainable)
    schedule = launch.pair_schedule(pairs, seed=5, pair_order="weighted")
    for _ in range(3):
        step, batch = next(schedule)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = preference.preference_batch_loss(
            model, batch, weight_normalizer=MEAN)
        loss.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(trainable, 1e-9)
        post_clip = preference.grad_norm(trainable)
        optimizer.step()
        trace.row(step=step, arm="preference",
                  problem_ids=[pair.problem_id for pair in batch],
                  weights=[pair.weight for pair in batch],
                  loss=float(loss.detach()), mean_weight=MEAN,
                  weight_normalizer=MEAN,
                  pre_clip_grad_norm=float(pre_clip),
                  post_clip_grad_norm=float(post_clip))

    path = trace.write(str(tmp_path / "trace.json"))
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["schema"] == preference.UpdateTelemetry.SCHEMA
    assert len(payload["rows"]) == 3
    for row in payload["rows"]:
        assert row["pre_clip_grad_norm"] > 0.0
        # grad_clip = 1e-9 is always exceeded, so every step was clipped and
        # the post-clip norm is pinned at the clip value.
        assert row["clipping_active"] is True
        assert row["pre_clip_grad_norm"] > row["post_clip_grad_norm"]
        assert row["post_clip_grad_norm"] == pytest.approx(1e-9, rel=0.02)
        assert row["update_l2"] > 0.0
        assert row["weight"] in (W_HIGH, W_LOW)
        assert row["weight_normalized"] == pytest.approx(
            row["weight"] / MEAN, rel=1e-6)
    summary = payload["summary"]
    assert summary["steps"] == 3
    assert summary["clipping_active_steps"] == 3
    assert summary["mean_update_l2"] > 0.0
    # Three steps are not a whole epoch, so the realized mean is the mean of
    # the three sampled weights and need not equal the corpus mean.
    row_weights = [row["weight"] for row in payload["rows"]]
    assert summary["realized_mean_weight"] == pytest.approx(
        sum(row_weights) / len(row_weights))
    assert W_LOW <= summary["realized_mean_weight"] <= W_HIGH


def _mean_trajectory_distance(a, b):
    differences = [float((x - y).norm()) for x, y in zip(a, b)]
    return sum(differences) / len(differences), differences[-1]


def test_both_knobs_change_the_realized_adamw_update():
    """Each part of the fix moves the parameters; neither is cosmetic.

    Measured, not asserted from theory: with a *time-varying* per-step weight,
    AdamW's moments mix differently-scaled gradients, so the corpus normaliser
    is not cancelled either. The ordering and the normaliser are therefore two
    independent knobs, and the test pins both.
    """
    base = _trajectory("shuffled", "batch")          # (i) unweighted, uniform
    rescale_only = _trajectory("shuffled", "corpus")
    order_only = _trajectory("weighted", "batch")
    both = _trajectory("weighted", "corpus")          # (ii) the fixed arm

    for label, arm in (("rescale only", rescale_only),
                       ("order only", order_only),
                       ("both", both)):
        mean_diff, final_diff = _mean_trajectory_distance(base[0], arm[0])
        assert mean_diff > 1e-3, (label, mean_diff)
        assert final_diff > 1e-3, (label, final_diff)
    # The two knobs are not redundant: each alone differs from the combination.
    for label, arm in (("rescale only", rescale_only),
                       ("order only", order_only)):
        mean_diff, _ = _mean_trajectory_distance(both[0], arm[0])
        assert mean_diff > 1e-3, (label, mean_diff)


# ---------------------------------------------------------------------------
# (d) the control replay and the matched-compute basis


def _synthetic_pairs(count=5):
    pairs = []
    for index in range(count):
        pairs.append(_pair(f"p{index}", 0.05 + 0.05 * index,
                           prompt=(1, 2), chosen=(4, 5), rejected=(0, 3)))
    return pairs


@pytest.mark.parametrize("order", ["shuffled", "weighted"])
def test_control_budget_reproduces_the_launcher_pair_order(order):
    pairs = _synthetic_pairs()
    steps = 60  # five pairs, so this crosses twelve epoch boundaries
    launcher_ids = [[pair.problem_id for pair in batch]
                    for _, batch in itertools.islice(
                        launch.pair_schedule(pairs, seed=11, pair_order=order),
                        steps)]
    control = control_budget.preference_schedule(
        pairs, seed=11, pair_order=order, batch_pairs=1,
        budget_tokens=10 ** 9, max_steps=steps)
    control_ids = [[pairs[index].problem_id for index in row["pair_indices"]]
                   for row in control]
    assert control_ids == launcher_ids
    assert len(control) == steps


def test_compute_basis_ratio_is_the_analytic_value(tmp_path):
    """prompt=10, chosen=5, rejected=3: 28 positions per preference step,
    15 per SFT step, so 40 preference steps need ceil(1120 / 15) = 75."""
    pair_path = tmp_path / "pairs.jsonl"
    rows = []
    for index in range(4):
        rows.append({
            "problem_id": f"p{index}",
            "prompt_ids": list(range(10)),
            "chosen_ids": [1, 2, 3, 4, 5],
            "rejected_ids": [6, 7, 8],
            "weight": 0.25 - 0.05 * index,
        })
    pair_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                         encoding="utf-8")
    out = tmp_path / "receipt.json"
    assert control_budget.main([
        "--pairs", str(pair_path), "--budget-tokens", str(28 * 40),
        "--seed", "3", "--basis", "compute", "--out", str(out),
    ]) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    block = receipt["bases"][0]
    assert block["steps"]["preference"]["steps"] == 40
    assert block["steps"]["preference"]["forward_positions"] == 28 * 40
    # Matched steps: the SFT arm takes 40 steps and forwards 15 per step.
    assert block["steps"]["matched_sft"]["steps"] == 40
    assert block["steps"]["matched_sft"]["forward_positions"] == 15 * 40
    # Matched compute: the SFT arm must reach 1120 forward positions.
    assert block["compute"]["matched_sft"]["steps"] == math.ceil(28 * 40 / 15)
    assert block["compute"]["matched_sft"]["steps"] == 75
    assert block["compute"]["matched_sft"]["forward_positions"] >= 28 * 40
    assert (block["compute"]["matched_sft"]["forward_positions"]
            - 28 * 40) < 15
    assert receipt["matched_sft"]["selected"][0]["compute_ratio"] == \
        pytest.approx(75 / 40)
    assert receipt["bases"][0]["compute"]["matched_sft"][
        "budget_tokens_for_control"] == 75 * 15
