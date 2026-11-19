"""S64: end-to-end CPU test of the preference pipeline.

Everything here runs on tiny float32 tensors and a tiny model. It covers the
parts the GPU jobs depend on and that a GPU hour cannot cheaply re-check:

* the teacher pass builds pairs only on a shared prefix, re-verifies the
  teacher continuation, and emits the p=0 recovery pair with the matched
  weight;
* the frozen-reference cache is per-token, required, and survives the pairs
  round trip;
* the launcher installs rank-64 / alpha-32 LoRA on every linear layer except
  the output head, freezes the base, and its merged checkpoint reproduces the
  adapted logits within the registered tolerance;
* after the merge the adapter is not part of the active union and the
  registered deployed union is still 752,393,024;
* the matched SFT control consumes the same processed-token budget on the
  chosen continuations, and the recovery arm trains on the recovery pairs.
"""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from frontier import preference  # noqa: E402
import s64_preference_data as data_pass  # noqa: E402
import s64_preference_launch as launch  # noqa: E402
import s64_shard_window as window  # noqa: E402


class TinyLM(nn.Module):
    """A positionwise causal-ish model with attention-like and MLP linears."""

    def __init__(self, vocab: int = 64, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 16), nn.Linear(16, hidden))
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, use_cache: bool = False):
        hidden = self.embed(input_ids)
        hidden = torch.tanh(self.q_proj(hidden))
        hidden = self.mlp(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _base(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    model = TinyLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _adapted(seed: int = 0, rank: int = 4) -> TinyLM:
    model = _base(seed)
    preference.install_lora(model, rank=rank, alpha=2 * rank)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, preference.LoRALinear):
                module.lora_A.normal_(generator=generator, std=0.05)
                module.lora_B.normal_(generator=generator, std=0.05)
    return model


# ---------------------------------------------------------------------------
# teacher pass


def _fake_decode(ids):
    return "CHOSEN" if list(ids) == [30, 31] else "WRONG"


def _fake_verify(text, target, dataset):
    return {"verdict": "correct" if text == target else "wrong"}


def _ledger_row(problem_id, prompt, p, target="CHOSEN"):
    return {"id": problem_id, "dataset": "gsm8k", "target": target,
            "prompt_ids": list(prompt), "p": p,
            "weight_p1mp": p * (1.0 - p)}


def _teacher(problem_id, prompt, ids=(30, 31), kind="teacher"):
    return {"id": problem_id, "dataset": "gsm8k", "kind": kind,
            "prompt_ids": list(prompt), "solution_ids": list(ids)}


def _rejected(problem_id, prompt, ids):
    return {"id": problem_id, "dataset": "gsm8k", "prompt_ids": list(prompt),
            "solution_ids": list(ids), "truncated": False}


def test_teacher_pass_builds_shared_prefix_pairs_and_official_recovery():
    ledger = {
        "a": _ledger_row("a", [1, 2, 3], 0.25),
        "b": _ledger_row("b", [4, 5], 0.0),
        "c": _ledger_row("c", [6, 7], 0.5),
        "d": _ledger_row("d", [8, 9], 0.5),
        "e": _ledger_row("e", [10, 11], 0.5),
        "f": _ledger_row("f", [12, 13], 1.0),
    }
    rejected = {
        "a": _rejected("a", [1, 2, 3], [40, 41]),
        "b": _rejected("b", [4, 5], [50]),
        "c": _rejected("c", [6, 7], [60]),
        "d": _rejected("d", [8, 9], [70]),
        "e": _rejected("e", [10, 11], [30, 31]),  # identity pair
        "g": _rejected("g", [99], [1]),  # no ledger row
    }
    corpus = {
        "a": [_teacher("a", [1, 2, 3])],
        "b": [_teacher("b", [4, 5])],
        "c": [_teacher("c", [98, 99])],  # prefix mismatch
        "d": [_teacher("d", [8, 9], ids=(7, 8))],  # unverified
        "e": [_teacher("e", [10, 11])],
    }
    on_policy, recovery, diagnostics = data_pass.build_pairs(
        ledger, rejected, corpus, decode_fn=_fake_decode,
        verify_fn=_fake_verify)
    by_id = {pair.problem_id: pair for pair in on_policy}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].chosen_ids == (30, 31)
    assert by_id["a"].rejected_ids == (40, 41)
    assert by_id["a"].weight == pytest.approx(0.1875)
    assert by_id["a"].provenance == "on-policy"
    assert by_id["b"].weight == 0.0
    # p=0 gets one recovery pair with the mean positive on-policy weight.
    assert len(recovery) == 1
    assert recovery[0].problem_id == "b"
    assert recovery[0].provenance == "teacher-recovery"
    expected_weight = 0.1875  # the only positive on-policy pair is a
    assert recovery[0].weight == pytest.approx(expected_weight)
    assert diagnostics["teacher_from_corpus"] == ["a", "b"]
    assert "c" in diagnostics["prefix_mismatch_teacher"]
    assert "c" in diagnostics["missing_teacher"]
    assert "d" in diagnostics["missing_teacher"]
    assert "e" in diagnostics["identity_pair"]
    assert "f" in diagnostics["p1_without_failure"]


def test_teacher_pass_fails_closed_on_a_rejected_prefix_mismatch():
    ledger = {"a": _ledger_row("a", [1, 2, 3], 0.5)}
    rejected = {"a": _rejected("a", [9, 9, 9], [40])}
    corpus = {"a": [_teacher("a", [1, 2, 3])]}
    on_policy, recovery, diagnostics = data_pass.build_pairs(
        ledger, rejected, corpus, decode_fn=_fake_decode,
        verify_fn=_fake_verify)
    assert on_policy == [] and recovery == []
    assert diagnostics["prefix_mismatch_rejected"] == ["a"]


def test_teacher_pass_generates_only_when_the_corpus_is_silent():
    ledger = {"a": _ledger_row("a", [1, 2, 3], 0.5)}
    rejected = {"a": _rejected("a", [1, 2, 3], [40])}
    calls = []

    def generate(prompt_ids):
        calls.append(list(prompt_ids))
        return [30, 31], True, False

    on_policy, _, diagnostics = data_pass.build_pairs(
        ledger, rejected, {}, decode_fn=_fake_decode,
        verify_fn=_fake_verify, generate_fn=generate)
    assert calls == [[1, 2, 3]]
    assert len(on_policy) == 1
    assert diagnostics["teacher_generated"] == ["a"]


def test_teacher_pass_rejects_a_truncated_generated_teacher():
    ledger = {"a": _ledger_row("a", [1, 2, 3], 0.5)}
    rejected = {"a": _rejected("a", [1, 2, 3], [40])}

    def generate(prompt_ids):
        return [30, 31], False, True

    on_policy, _, diagnostics = data_pass.build_pairs(
        ledger, rejected, {}, decode_fn=_fake_decode,
        verify_fn=_fake_verify, generate_fn=generate)
    assert on_policy == []
    assert "a" in diagnostics["teacher_generation_failed"]


# ---------------------------------------------------------------------------
# reference cache and pair round trip


def test_reference_cache_is_per_token_and_survives_the_round_trip(tmp_path):
    pair = preference.make_pair("a", [1, 2, 3], [4, 5], [6, 7, 8], weight=0.2)
    reference = _base(seed=5)
    cached = preference.cache_reference_logprobs(reference, pair)
    assert len(cached.chosen_reference_logprobs) == 2
    assert len(cached.rejected_reference_logprobs) == 3
    assert all(math.isfinite(value)
               for value in cached.chosen_reference_logprobs)

    path = tmp_path / "pairs.jsonl"
    data_pass.write_pairs(str(path), [cached], reference=True)
    row = json.loads(path.read_text().strip())
    assert row["chosen_tokens"] == 2 and row["rejected_tokens"] == 3
    assert row["chosen_reference_mean"] == pytest.approx(
        sum(row["chosen_reference_logprobs"]) / 2)
    assert row["chosen_reference_sum"] == pytest.approx(
        sum(row["chosen_reference_logprobs"]))
    loaded = data_pass.pair_from_row(row)
    assert loaded.chosen_reference_logprobs == cached.chosen_reference_logprobs
    assert torch.allclose(
        preference.reference_logprob(loaded.chosen_reference_logprobs),
        preference.reference_logprob(cached.chosen_reference_logprobs))


# ---------------------------------------------------------------------------
# launcher geometry and merge


def test_launcher_installs_registered_rank_64_all_linear():
    model = _base(seed=1)
    geometry = launch.prepare_registered_lora(model)
    assert geometry["lora_rank"] == 64
    assert geometry["lora_alpha"] == 32
    assert geometry["scaling"] == 0.5
    assert geometry["wrapped_layers"] > 0
    assert not isinstance(model.lm_head, preference.LoRALinear)
    assert isinstance(model.q_proj, preference.LoRALinear)
    assert geometry["active_union_before_merge"] == (
        geometry["base_text_elements"] + geometry["adapter_elements"])
    assert geometry["active_union_after_merge"] == (
        geometry["base_text_elements"])


def test_merge_parity_and_the_registered_active_union():
    base = _base(seed=2)
    base_elements = preference.unique_active_elements(base)
    model = _adapted(seed=2)
    adapter_elements = preference.adapter_element_count(model)
    input_ids = torch.randint(0, 64, (2, 6),
                              generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        adapted_logits = model(input_ids).logits.clone()
    state, report = launch.snapshot_merged_full_state(model)
    assert report["remaining_lora_modules"] == 0
    assert report["merged_text_elements"] == base_elements
    # TinyLM exposes get_input_embeddings, so the embedding is the frozen
    # identity and is excluded from the exported full-arm state.
    assert report["frozen_identity_excluded"]
    assert report["exported_state_elements"] < base_elements

    merged = TinyLM()
    source = model.state_dict()
    target = merged.state_dict()
    for key in target:
        if key in source:
            target[key] = source[key].clone()
    merged.load_state_dict(target)
    current = merged.state_dict()
    current.update({key: value for key, value in state.items()})
    merged.load_state_dict(current)
    with torch.no_grad():
        merged_logits = merged(input_ids).logits
    assert torch.allclose(merged_logits, adapted_logits,
                          atol=preference.MERGE_ATOL,
                          rtol=preference.MERGE_RTOL)
    assert preference.active_union_after_merge(
        preference.REGISTERED_ACTIVE_UNION,
        adapter_elements) == 752_393_024


def test_snapshot_excludes_the_frozen_tied_embedding_from_the_full_arm_state():
    model = _adapted(seed=11)
    embedding = model.get_input_embeddings().weight
    pointer = embedding.data_ptr()
    names = {name for name, parameter in model.named_parameters()
             if parameter.data_ptr() == pointer}
    assert names
    state, report = launch.snapshot_merged_full_state(model)
    assert set(report["frozen_identity_excluded"]) == names
    assert not (set(state) & names)


# ---------------------------------------------------------------------------
# training arms


def _cached_pairs(n=2, seed=3):
    reference = _base(seed=seed)
    pairs = []
    for index in range(n):
        raw = preference.make_pair(
            f"p{index}", [1, 2, 3], [4 + index, 5], [6 + index, 7, 8],
            weight=0.25 + 0.1 * index)
        pairs.append(preference.cache_reference_logprobs(reference, raw))
    return pairs


def test_preference_arm_emits_a_merged_checkpoint_at_the_cadence(tmp_path):
    pairs = _cached_pairs()
    model = _base(seed=4)
    budget = 2 * preference.preference_processed_tokens(pairs)
    report = launch.run_arm(
        model, pairs, arm="preference", budget_tokens=budget,
        cadence=(preference.preference_processed_tokens(pairs),),
        seed=7, checkpoint_dir=str(tmp_path), expected_active_union=None)
    assert report["processed_tokens"] >= budget
    assert report["steps"] > 0
    assert all(math.isfinite(value) for value in report["loss_trace"])
    assert report["geometry"]["lora_rank"] == 64
    assert len(report["checkpoints"]) >= 1
    checkpoint = torch.load(report["checkpoints"][0]["path"],
                            map_location="cpu", weights_only=False)
    assert checkpoint["arm"] == "full"
    assert checkpoint["schema"] == "selvedge/stage2-ckpt/v1"
    assert checkpoint["step"] == report["checkpoints"][0]["step"]
    # The merged state is the text tower minus the frozen embedding.
    canonical = _base(seed=5)
    embedding_pointer = canonical.get_input_embeddings().weight.data_ptr()
    expected_names = {name for name, parameter in canonical.named_parameters()
                      if parameter.data_ptr() != embedding_pointer}
    assert set(checkpoint["arm_state"]) == expected_names
    assert checkpoint["merge"]["remaining_lora_modules"] == 0


def test_matched_sft_control_uses_the_same_pairs_and_budget(tmp_path):
    pairs = _cached_pairs()
    model = _base(seed=6)
    budget = preference.preference_processed_tokens(pairs)
    report = launch.run_arm(
        model, pairs, arm="matched-sft", budget_tokens=budget,
        cadence=(budget,), seed=8, checkpoint_dir=str(tmp_path))
    assert report["kind"] == "matched-sft"
    assert report["budget_tokens"] == budget
    assert report["processed_tokens"] >= budget
    assert report["supervised_chosen_tokens"] > 0
    assert report["final_loss"] is not None
    # The control is completion CE on the chosen side only.
    assert report["beta"] is None


def test_recovery_arm_trains_on_the_recovery_pairs(tmp_path):
    ledger = {"b": _ledger_row("b", [4, 5], 0.0)}
    rejected = {"b": _rejected("b", [4, 5], [50])}
    corpus = {"b": [_teacher("b", [4, 5])]}
    _, recovery, _ = data_pass.build_pairs(
        ledger, rejected, corpus, decode_fn=_fake_decode,
        verify_fn=_fake_verify)
    assert len(recovery) == 1
    reference = _base(seed=9)
    cached = [preference.cache_reference_logprobs(reference, pair)
              for pair in recovery]
    assert cached[0].provenance == "teacher-recovery"
    model = _base(seed=10)
    budget = preference.preference_processed_tokens(cached)
    report = launch.run_arm(
        model, cached, arm="recovery", budget_tokens=budget,
        cadence=(budget,), seed=10, checkpoint_dir=str(tmp_path))
    assert report["processed_tokens"] >= budget
    assert report["steps"] > 0
    assert report["checkpoints"]


# ---------------------------------------------------------------------------
# the registered shard-window projection


def test_shard_window_projects_and_refuses_from_the_pilot_only():
    pilot = {
        "generation": {"tokens_per_second": 500.0},
        "acceptance": {"by_dataset": {
            "gsm8k": {"sampled": 240, "generated_tokens": 240 * 500},
            "math": {"sampled": 240, "generated_tokens": 240 * 1000},
        }},
    }
    fits = window.project(pilot, bound_hours=4.5)
    assert fits["all_fit"]
    by_label = {row["label"]: row for row in fits["shards"]}
    assert by_label["gsm8k-shard0"]["prompts"] == 600
    assert by_label["gsm8k-shard0"]["samples"] == 4800
    assert by_label["math-shard0"]["prompts"] == 300
    assert by_label["math-shard0"]["samples"] == 2400
    # 4800 samples * 500 tok/sample / 500 tok/s = 4800 s = 1.333 h
    assert by_label["gsm8k-shard0"]["projected_hours"] == pytest.approx(4 / 3)
    refused = window.project(pilot, bound_hours=1.0)
    assert refused["all_fit"] is False
    assert all(not row["fits"] for row in refused["shards"])


def test_shard_commands_carry_k8_shards_and_the_rejected_sidecar():
    commands = [window.command(label, dataset, select_flag, selected,
                               shards_count, shard_index, k, cap)
                for (label, dataset, select_flag, selected, shards_count,
                     shard_index, k, cap) in window.PLAN]
    assert [cmd[cmd.index("--label") + 1] for cmd in commands] == [
        row[0] for row in window.PLAN]
    for cmd in commands:
        k_flag = "--k-gsm8k" if "--k-gsm8k" in cmd else "--k-math"
        assert cmd[cmd.index(k_flag) + 1] == "8"
        assert "--rejected-out" in cmd
        assert cmd[cmd.index("--shards") + 1] == "2"
