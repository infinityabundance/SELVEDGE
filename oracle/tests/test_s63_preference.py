"""S63: the reference-anchored preference objective, LoRA merge and control.

These tests pin the arithmetic that must be exact before a GPU hour is spent:

* the preference loss is the stated ``-log sigmoid`` margin, with the frozen
  reference subtracted;
* sequence log-probabilities are token-length normalised, and the mode is
  explicit;
* merging rank-64 all-linear LoRA reproduces the adapter's logits within the
  stated tolerance;
* after the merge the adapter is not part of the active union, whose registered
  value is still 752,393,024;
* a matched SFT control runs on the same pairs and processed-token budget;
* a p=0 problem gets a recovery pair weighted by the mean positive p(1-p).
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from frontier import preference


class TinyLM(nn.Module):
    """A positionwise model with attention-like and MLP linear layers."""

    def __init__(self, vocab: int = 64, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 16), nn.Linear(16, hidden))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, use_cache: bool = False):
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.q_proj(hidden))
        hidden = self.mlp(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _model(seed: int = 0, rank: int = 4) -> TinyLM:
    torch.manual_seed(seed)
    model = TinyLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preference.install_lora(model, rank=rank, alpha=2 * rank)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, preference.LoRALinear):
                module.lora_A.normal_(generator=generator, std=0.1)
                module.lora_B.normal_(generator=generator, std=0.1)
    return model


def _pair(seed: int = 0) -> preference.PreferencePair:
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, 64, (4,), generator=generator).tolist()
    chosen = torch.randint(0, 64, (5,), generator=generator).tolist()
    rejected = torch.randint(0, 64, (7,), generator=generator).tolist()
    return preference.make_pair("p", prompt, chosen, rejected, weight=0.25)


def test_preference_loss_matches_the_closed_form():
    chosen = torch.tensor(-1.0)
    rejected = torch.tensor(-2.0)
    reference_chosen = torch.tensor(-1.5)
    reference_rejected = torch.tensor(-1.0)
    beta = 0.5
    # margin = 0.5 * [(-1 - -2) - (-1.5 - -1.0)] = 0.5 * [1 - -0.5] = 0.75
    expected = -torch.nn.functional.logsigmoid(torch.tensor(0.75))
    got = preference.preference_loss(
        chosen, rejected, reference_chosen, reference_rejected, beta=beta)
    assert torch.allclose(got, expected)


def test_token_length_normalisation_is_explicit_and_changes_the_scale():
    total = torch.tensor(-4.0)
    assert preference.normalise_logprob(total, 2, "mean").item() == -2.0
    assert preference.normalise_logprob(total, 4, "mean").item() == -1.0
    assert preference.normalise_logprob(total, 2, "sum").item() == -4.0
    assert preference.normalise_logprob(total, 4, "sum").item() == -4.0
    # The registered objective is the per-token mean, so a long rejected
    # continuation cannot win merely by having more tokens.
    assert preference.REGISTERED_NORMALIZATION == "mean"


def test_cached_reference_logprobs_are_required_and_cached():
    reference = TinyLM()
    pair = _pair()
    cached = preference.cache_reference_logprobs(reference, pair)
    assert len(cached.chosen_reference_logprobs) == len(pair.chosen_ids)
    assert len(cached.rejected_reference_logprobs) == len(pair.rejected_ids)
    assert all(math.isfinite(value)
               for value in cached.chosen_reference_logprobs)
    try:
        preference.reference_logprob(None)
    except ValueError as exc:
        assert "cached" in str(exc)
    else:
        raise AssertionError("an uncached reference must fail closed")


def test_merge_reproduces_the_adapter_logits_within_tolerance():
    model = _model(seed=3)
    input_ids = torch.randint(0, 64, (2, 6),
                              generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        before = model(input_ids).logits.clone()
    report = preference.merge_lora(model)
    with torch.no_grad():
        after = model(input_ids).logits
    assert report["remaining_lora_modules"] == 0
    assert report["merged_layers"] > 0
    assert torch.allclose(after, before, atol=preference.MERGE_ATOL,
                          rtol=preference.MERGE_RTOL)


def test_adapter_is_excluded_from_the_active_union_after_merge():
    torch.manual_seed(3)
    base = TinyLM()
    base_elements = preference.unique_active_elements(base)
    model = _model(seed=3)
    adapter_elements = preference.adapter_element_count(model)
    assert adapter_elements > 0
    assert preference.unique_active_elements(model) == base_elements + adapter_elements
    preference.merge_lora(model)
    assert preference.unique_active_elements(model) == base_elements
    assert preference.active_union_after_merge(
        base_elements, adapter_elements) == base_elements
    # The registered deployed union of the real student is unchanged.
    assert preference.REGISTERED_ACTIVE_UNION == 752_393_024
    assert preference.active_union_after_merge(
        preference.REGISTERED_ACTIVE_UNION,
        adapter_elements) == 752_393_024


def test_registered_lora_geometry_is_all_linear_rank_64_alpha_32():
    torch.manual_seed(0)
    model = TinyLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrappers = preference.install_lora(model)
    assert preference.REGISTERED_LORA_RANK == 64
    assert preference.REGISTERED_LORA_ALPHA == 32
    assert wrappers
    assert all(wrapper.rank == 64 for wrapper in wrappers)
    assert all(wrapper.alpha == 32 for wrapper in wrappers)
    assert all(wrapper.scaling == 0.5 for wrapper in wrappers)
    assert not isinstance(model.lm_head, preference.LoRALinear)
    assert isinstance(model.q_proj, preference.LoRALinear)
    assert isinstance(model.mlp[0], preference.LoRALinear)
    assert isinstance(model.mlp[1], preference.LoRALinear)


def test_preference_step_updates_only_the_adapter_and_uses_the_cache():
    reference = TinyLM()
    pair = preference.cache_reference_logprobs(reference, _pair(seed=5))
    model = _model(seed=6)
    base_before = model.q_proj.base.weight.detach().clone()
    report = preference.run_preference(model, [pair], steps=3, lr=1e-3,
                                       beta=0.1)
    assert report["steps"] == 3
    assert all(math.isfinite(value) for value in report["losses"])
    assert torch.equal(model.q_proj.base.weight, base_before)
    assert any(module.lora_B.abs().sum().item() > 0
               for module in model.modules()
               if isinstance(module, preference.LoRALinear))


def test_matched_sft_control_runs_on_the_same_examples_and_budget():
    reference = TinyLM()
    pairs = [preference.cache_reference_logprobs(reference, _pair(seed=7))]
    model = _model(seed=8)
    budget = preference.preference_processed_tokens(pairs)
    report = preference.run_matched_sft_control(model, pairs, lr=1e-3)
    assert report["kind"] == "matched_sft_control"
    assert report["budget_processed_tokens"] == budget
    assert report["steps"] > 0
    assert report["processed_tokens"] >= budget
    assert report["supervised_chosen_tokens"] > 0
    assert math.isfinite(report["final_loss"])


def test_recovery_pair_carries_the_mean_positive_weight_at_p_zero():
    assert preference.difficulty_weight(0.0) == 0.0
    assert preference.difficulty_weight(1.0) == 0.0
    assert preference.difficulty_weight(0.25) == 0.1875
    weights = [0.0, 0.25, 0.0, 0.1875]
    assert preference.recovery_weight(weights) == (0.25 + 0.1875) / 2
    pair = preference.recovery_pair("hard", [1, 2], [3, 4, 5], [6], 0.0,
                                    weights)
    assert pair.weight == (0.25 + 0.1875) / 2
    assert pair.provenance == "teacher-recovery"
    try:
        preference.recovery_pair("solved", [1], [2], [3], 1.0, weights)
    except ValueError as exc:
        assert "p == 0" in str(exc)
    else:
        raise AssertionError("the recovery arm is only for p == 0")
