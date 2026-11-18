"""S52: the P19.9 runner's CPU contract.

The runner is tested with fake tensors, fake rollouts and a fake store. No model
is loaded and no generation runs. The pinned claims are the ones the S50 queue
entry makes gating conditions:

* the feature allow-list (causal hidden state and registered metadata only);
* mask enforcement (zero contribution and zero requests for unselected tokens);
* the sign of `Delta_outcome` on hand-built paired rollouts;
* deterministic training and deterministic greedy rollouts;
* physical read accounting (unique rows, tier bytes, cache hits);
* a refusal path when the paired rollouts are missing.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import routing_runner as runner  # noqa: E402
from frontier import utility_routing as routing  # noqa: E402


SPEC = runner.RolloutSpec(**runner.DEFAULT_SPEC)


def _item(prompt_id: str, split: str = "fit", answer: str = "7",
          question: str | None = None) -> runner.PromptItem:
    return runner.PromptItem(
        prompt_id=prompt_id, dataset="gsm8k",
        prompt=question or f"Q {prompt_id}: what is 3 + 4?", answer=answer,
        split=split)


class FakeGenerator:
    """Deterministic stand-in for the GPU generator."""

    def __init__(self, completion_of):
        self.completion_of = completion_of
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, condition, item, sample):
        self.calls.append((condition, item.prompt_id, sample["sample_index"]))
        completion = self.completion_of(condition, item, sample)
        return {
            "completion": completion,
            "ce": 1.0 if condition == "base" else 0.75,
            "token_ce": [1.0, 1.0] if condition == "base" else [0.75, 0.75],
        }


def exact_verifier(item, completion):
    return {"verdict": "correct" if completion.strip() == item.answer else "wrong"}


def _prompt_payload(items, sampling):
    return {
        "schema": runner.PROMPT_SET_SCHEMA,
        "sampling": sampling,
        "prompts": [
            {"prompt_id": item.prompt_id, "dataset": item.dataset,
             "prompt": item.prompt, "answer": item.answer,
             "split": item.split}
            for item in items],
    }


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_prompt_set_refuses_split_leakage_and_sampling_drift(tmp_path):
    items = [_item("a", "fit"), _item("b", "calib"), _item("c", "holdout")]
    path = _write(tmp_path, "set.json", _prompt_payload(items, SPEC.mapping()))
    loaded, meta = runner.load_prompt_set(path, SPEC)
    assert [row.prompt_id for row in loaded] == ["a", "b", "c"]
    assert meta["counts"] == {"fit": 1, "calib": 1, "holdout": 1}

    # the same question text under another id in another split is leakage
    leaked = items + [_item("d", "holdout", question=items[0].prompt)]
    path = _write(tmp_path, "leak.json", _prompt_payload(leaked, SPEC.mapping()))
    with pytest.raises(SystemExit, match="both"):
        runner.load_prompt_set(path, SPEC)

    # a prompt set registered for another sampling plan cannot be reused
    drift = dict(SPEC.mapping(), temperature=0.7)
    path = _write(tmp_path, "drift.json", _prompt_payload(items, drift))
    with pytest.raises(SystemExit, match="sampling"):
        runner.load_prompt_set(path, SPEC)

    duplicated = items + [_item("a", "holdout", question="different question")]
    path = _write(tmp_path, "dup.json",
                  _prompt_payload(duplicated, SPEC.mapping()))
    with pytest.raises(SystemExit, match="duplicate"):
        runner.load_prompt_set(path, SPEC)


def test_delta_outcome_sign_is_correct_on_hand_built_rollouts():
    # memory always right, base always wrong: Delta = +1
    helped = runner.run_paired_rollouts(
        [_item("h")],
        FakeGenerator(lambda condition, item, sample:
                      item.answer if condition == "memory" else "5"),
        exact_verifier, SPEC)
    assert helped[0]["delta_outcome"] == pytest.approx(1.0)
    assert helped[0]["outcome"]["memory_successes"] == SPEC.k
    assert helped[0]["outcome"]["base_successes"] == 0

    # memory always wrong, base always right: Delta = -1
    hurt = runner.run_paired_rollouts(
        [_item("u")],
        FakeGenerator(lambda condition, item, sample:
                      "5" if condition == "memory" else item.answer),
        exact_verifier, SPEC)
    assert hurt[0]["delta_outcome"] == pytest.approx(-1.0)

    # half the pairs helped, half unchanged: Delta = +0.5
    def mixed(condition, item, sample):
        if condition == "memory":
            return item.answer
        return item.answer if sample["sample_index"] % 2 else "5"

    mixed_rows = runner.run_paired_rollouts(
        [_item("m")], FakeGenerator(mixed), exact_verifier, SPEC)
    assert mixed_rows[0]["delta_outcome"] == pytest.approx(0.5)

    # each side is verified separately: 2K verifier calls per prompt
    calls = {"n": 0}

    def counting_verifier(item, completion):
        calls["n"] += 1
        return exact_verifier(item, completion)

    runner.run_paired_rollouts(
        [_item("v")],
        FakeGenerator(lambda condition, item, sample: item.answer),
        counting_verifier, SPEC)
    assert calls["n"] == 2 * SPEC.k


def test_greedy_mode_is_deterministic_and_rejects_sampled_specs():
    greedy = runner.RolloutSpec(k=3, temperature=0.0, top_p=1.0,
                                mode="greedy", max_new_tokens=64, seed=0)
    plan = runner.sampling_plan(greedy, "p")
    assert len(plan) == 3 and all(sample["greedy"] for sample in plan)
    rows = runner.run_paired_rollouts(
        [_item("g")],
        FakeGenerator(lambda condition, item, sample: item.answer),
        exact_verifier, greedy)
    assert set(rows[0]["completions"]["base"]) == {"7"}
    assert rows[0]["delta_outcome"] == pytest.approx(0.0)

    # a non-deterministic greedy generator is a failed determinism check
    def nondeterministic(condition, item, sample):
        return item.answer if sample["sample_index"] % 2 == 0 else "5"

    with pytest.raises(ValueError, match="not deterministic"):
        runner.run_paired_rollouts([_item("n")],
                                   FakeGenerator(nondeterministic),
                                   exact_verifier, greedy)

    with pytest.raises(ValueError):
        runner.RolloutSpec(k=1, temperature=0.8, top_p=0.95,
                           mode="greedy", max_new_tokens=64)
    with pytest.raises(ValueError):
        runner.RolloutSpec(k=1, temperature=0.0, top_p=0.95,
                           mode="sampled", max_new_tokens=64)


def _labels(results, source="counterfactual-outcome"):
    return runner.labels_by_split(results, source)


def test_feature_allow_list_rejects_post_read_and_label_inputs():
    router = routing.PreReadUtilityRouter(hidden=5, tau=1.0)
    hidden = torch.randn(4, 5)
    metadata = runner.metadata_for_prompt("p", 4)
    labels = {"fit": [{"prompt_id": "p", "split": "fit", "label": 0.5}],
              "calib": [{"prompt_id": "c", "split": "calib", "label": -0.5}],
              "holdout": []}
    examples = runner.assemble_examples(router, {"p": hidden, "c": hidden},
                                        {"p": metadata, "c": metadata}, labels,
                                        "counterfactual-outcome")
    assert examples[0].features.shape == (4, router.feature_dim)

    for bad in ("r_t", "read_norm", "gate_t", "target", "label", "delta",
                "base_loss", "memory_loss"):
        poisoned = dict(metadata)
        poisoned[bad] = torch.zeros(4)
        with pytest.raises(ValueError):
            runner.assemble_examples(
                router, {"p": hidden}, {"p": poisoned}, labels,
                "counterfactual-outcome")

    # a label key in `features` is the registered leakage rejection
    record = [{"prompt_id": "p", "split": "fit", "verified": True,
               "features": {"h_t": [0.0], "label": 1.0}}]
    with pytest.raises(ValueError):
        routing.labels_from_records(record, "verified-outcome")


def test_labels_keep_splits_disjoint_and_follow_the_paired_delta():
    results = runner.run_paired_rollouts(
        [_item("a", "fit"), _item("b", "calib"), _item("c", "holdout")],
        FakeGenerator(lambda condition, item, sample:
                      item.answer if condition == "memory" else "5"),
        exact_verifier, SPEC)
    labels = runner.labels_by_split(results, "counterfactual-outcome")
    assert labels["fit"][0]["label"] == pytest.approx(1.0)
    assert labels["calib"][0]["split"] == "calib"
    assert labels["holdout"][0]["split"] == "holdout"

    leaked = [dict(row) for row in results]
    leaked[2]["prompt_id"] = "a"
    with pytest.raises(ValueError, match="both"):
        runner.labels_by_split(leaked, "counterfactual-outcome")

    # the CE source needs both sides' CE, and its label is base - memory
    ce_labels = runner.labels_by_split(results, "counterfactual-token-loss")
    assert ce_labels["fit"][0]["label"] == pytest.approx(0.25)
    missing = [dict(row) for row in results]
    missing[0]["memory_ce"] = None
    with pytest.raises(ValueError, match="no CE"):
        runner.labels_by_split(missing, "counterfactual-token-loss")


def test_training_is_deterministic_and_selects_the_calibration_epoch():
    hidden = torch.randn(24, 6)
    metadata = runner.metadata_for_prompt("p", 24)
    # a learnable rule: Delta is positive where feature 0 is positive
    label = float(hidden[:, 0].mean() > 0.0)
    labels = {
        "fit": [{"prompt_id": "p", "split": "fit", "label": label}],
        "calib": [{"prompt_id": "c", "split": "calib", "label": label}],
        "holdout": [],
    }
    hidden_map = {"p": hidden, "c": hidden}
    metadata_map = {"p": metadata, "c": metadata}

    def build():
        router = runner.build_router(6, tau=1.0, seed=11)
        examples = runner.assemble_examples(
            router, hidden_map, metadata_map, labels,
            "counterfactual-outcome")
        fit = [row for row in examples if row.split == "fit"]
        calib = [row for row in examples if row.split == "calib"]
        report = runner.train_router(router, fit, calib, epochs=8, lr=1e-2,
                                     seed=3)
        return router, report

    first_router, first = build()
    second_router, second = build()
    assert first["parameter_hash"] == second["parameter_hash"]
    assert first["chosen_epoch"] == second["chosen_epoch"]
    assert 1 <= first["chosen_epoch"] <= 8
    for left, right in zip(first_router.state_dict().values(),
                           second_router.state_dict().values()):
        assert torch.equal(left, right)

    # training refuses an empty split
    with pytest.raises(ValueError):
        runner.train_router(first_router, [], [])


def _partial_router(hidden: int = 4):
    router = routing.TwoStageUtilityRouter(hidden=hidden, use_post_read=False)
    with torch.no_grad():
        router.pre.predictor[0].weight.zero_()
        router.pre.predictor[0].weight[:, 0] = 1.0
        router.pre.predictor[2].weight.fill_(1.0)
        router.pre.predictor[2].bias.fill_(0.0)
    return router


def test_mask_enforcement_zero_contribution_and_zero_requests():
    router = _partial_router()
    hidden = torch.tensor([[-2.0, 0.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0, 0.0]]).unsqueeze(0)
    metadata = runner.metadata_for_prompt("p", 5)
    store = torch.arange(20, dtype=torch.float32).view(5, 4)
    requested: list[torch.Tensor] = []

    def read_path(indices):
        requested.append(indices.clone())
        return store[indices]

    execution = router.execute(hidden, metadata, read_path, cost=0.0,
                               threshold=0.5)
    mask = execution.selection_mask
    assert mask.tolist() == [[False, False, True, True, True]]
    assert len(requested) == 1
    assert torch.equal(requested[0], mask.reshape(-1).nonzero().squeeze(-1))
    # the store was never asked for an unselected token
    assert not bool(set(requested[0].tolist()) & set((~mask.reshape(-1))
                                                     .nonzero().squeeze(-1)
                                                     .tolist()))
    outside = execution.contribution[~mask]
    assert outside.shape[0] == 2
    assert float(outside.detach().abs().max()) == 0.0
    assert execution.accounting.requests == 3
    assert execution.accounting.tokens == 5

    # forced rejection fetches nothing at all
    with torch.no_grad():
        router.pre.predictor[2].bias.fill_(-100.0)
    seen: list[torch.Tensor] = []

    def rejecting(indices):
        seen.append(indices.clone())
        return store[indices]

    rejected = router.execute(hidden, metadata, rejecting)
    assert not bool(rejected.selection_mask.any())
    assert seen[0].numel() == 0
    assert float(rejected.contribution.detach().abs().max()) == 0.0


def test_physical_read_accounting_counts_unique_rows_and_tier_bytes():
    mask = torch.tensor([True, False, True, True])
    addresses = torch.tensor([0, 0, 0, 1])
    accounting = routing.physical_read_accounting(
        mask, row_addresses=addresses, row_bytes_by_tier={"table": 16},
        cached_addresses={0})
    assert accounting.tokens == 4
    assert accounting.selected_tokens == 3
    assert accounting.requests == 3
    assert accounting.unique_rows_requested == 2
    assert accounting.bytes_by_tier == {"table": 16}  # address 1 was fetched
    assert accounting.bytes_from_cache == 16           # address 0 hit
    assert accounting.cache_hits == 1
    assert accounting.cache_misses == 1

    with pytest.raises(ValueError):
        routing.physical_read_accounting(mask, cached_addresses={-1})
    with pytest.raises(ValueError):
        routing.physical_read_accounting(mask, row_bytes_by_tier={"table": -1})
    # a float mask would let a rejected token receive a scaled contribution
    with pytest.raises(ValueError):
        routing.physical_read_accounting(mask.float())

    cost = runner.cost_from_accounting(accounting)
    assert cost["source"] == "physical_read_accounting"
    assert cost["surrogate_used"] is False
    # three requests at 16 bytes each: two cache hits and one fetch. The old
    # global-unique ledger said 32; the physical count is 48.
    assert cost["bytes_total"] == 48
    assert cost["bytes_per_token"] == pytest.approx(48 / 4)
    assert cost["bytes_unique_modelled"] == 32
    assert cost["cache_model"] == "call-scoped-address-cache"
    with pytest.raises(TypeError):
        runner.cost_from_accounting(None)
    with pytest.raises(TypeError):
        runner.cost_from_accounting(torch.zeros(1))

    # the store ledger measures every request from the table's own row size
    ledger = runner.StoreLedger()
    table = torch.arange(8, dtype=torch.float32).view(8, 1)
    ledger.mark_generation()
    ledger.gather(table, torch.tensor([1, 2, 2]))
    ledger.mark_generation()
    ledger.gather(table, torch.tensor([2, 3]))
    report = ledger.report()
    assert report["calls"] == 2
    assert report["generations"] == 2
    assert report["requested_rows"] == 5
    assert report["unique_rows"] == 3
    assert report["repeated_rows"] == 2
    # row_bytes = element_size 4 * width 1 = 4; physical = 5 * 4
    assert report["row_bytes"] == 4
    assert report["bytes"] == 20
    assert report["bytes_from_source"] == 20
    assert report["unique_bytes_modelled"] == 12
    assert report["cache_model"] == "none"
    assert report["per_call"] == [
        {"call": 1, "generation": 0, "requested": 3, "unique": 2,
         "repeated": 1, "cumulative_unique": 2, "cache_hits": 0},
        {"call": 2, "generation": 1, "requested": 2, "unique": 2,
         "repeated": 0, "cumulative_unique": 3, "cache_hits": 0},
    ]
    assert report["per_generation"] == [
        {"generation": 0, "calls": 1, "requested": 3, "unique": 2,
         "repeated": 1, "cache_hits": 0},
        {"generation": 1, "calls": 1, "requested": 2, "unique": 2,
         "repeated": 0, "cache_hits": 0},
    ]


def test_selection_report_aggregates_gate_stats_and_reads():
    router = _partial_router()
    example = runner.RouterExample(
        prompt_id="p", split="holdout",
        hidden=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0, 0.0]]).unsqueeze(0),
        metadata=runner.metadata_for_prompt("p", 2),
        features=torch.zeros(2, 9), targets=torch.zeros(2),
        weights=torch.full((2,), 0.5))
    report = runner.selection_report(router, [example], threshold=0.5,
                                     row_bytes_by_tier={"table": 4})
    assert report["tokens"] == 2
    assert report["selected_tokens"] == 1
    assert report["unselected_tokens"] == 1
    assert report["requests_outside_mask"] == 0
    assert report["accounting"]["requests"] == 1
    assert report["accounting"]["bytes_by_tier"] == {"table": 4}
    assert report["gate_min"] <= report["gate_mean"] <= report["gate_max"]
    assert report["cost"]["bytes_per_token"] == pytest.approx(2.0)


def test_runner_refuses_missing_or_unpaired_rollouts(tmp_path):
    with pytest.raises(SystemExit, match="rollouts are required"):
        runner.require_rollouts("")
    with pytest.raises(SystemExit, match="missing rollout receipt"):
        runner.require_rollouts(str(tmp_path / "absent.json"))

    empty = _write(tmp_path, "empty.json",
                   {"schema": runner.ROLLOUT_RECEIPT_SCHEMA, "prompts": []})
    with pytest.raises(SystemExit, match="no prompts"):
        runner.require_rollouts(empty)

    unpaired = {
        "schema": runner.ROLLOUT_RECEIPT_SCHEMA,
        "prompts": [{"prompt_id": "p", "split": "fit", "paired_rollouts": [
            {"base": {"verified": True}}]}],
    }
    path = _write(tmp_path, "unpaired.json", unpaired)
    with pytest.raises(SystemExit, match="memory.verified"):
        runner.require_rollouts(path)

    good = runner.rollout_receipt(
        runner.run_paired_rollouts(
            [_item("p")],
            FakeGenerator(lambda condition, item, sample: item.answer),
            exact_verifier, SPEC),
        SPEC, {"sha256": "deadbeef"})
    path = _write(tmp_path, "good.json", good)
    assert runner.require_rollouts(path)["prompt_count"] == 1


def test_routed_arm_wraps_a_pre_read_router_and_fetches_only_selected():
    from frontier import memory as mem

    core = mem.ConditionalMemory(hidden=4, value_dim=8, rows=8, sites=(0,),
                                 table_dtype=torch.float32)
    core.context.ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 4)

    # zero-init router, positive cost: sigmoid(-0.7) < 0.5, nothing is fetched
    rejecting = runner.RoutedMemoryArm(
        core, router=runner.build_router(4, seed=0), threshold=0.5, cost=0.35)
    out = rejecting(hidden)
    assert out.shape == hidden.shape
    assert float(out.detach().abs().max()) == 0.0
    assert rejecting.ledger.report()["calls"] == 0
    assert rejecting.ledger.report()["requested_rows"] == 0

    # zero cost: the gate is exactly 0.5 and every token is selected
    dense = runner.RoutedMemoryArm(
        core, router=runner.build_router(4, seed=0), threshold=0.5, cost=0.0)
    dense(hidden)
    report = dense.ledger.report()
    assert report["calls"] == 1
    assert report["requested_rows"] == 3 * 8
    assert report["unique_rows"] > 0

    # no router is the dense memory condition
    plain = runner.RoutedMemoryArm(core)
    plain(hidden)
    assert plain.ledger.report()["requested_rows"] == 3 * 8


def test_win_bar_needs_both_controls_and_fewer_bytes():
    arms = {
        "base": {"correct": 10, "items": 50, "read_requests": 0,
                 "read_bytes": 0},
        # gated-dense is the accuracy control only. Its read numbers are null;
        # in this test it carries deliberately wrong values to prove they are
        # never used as the byte control.
        "gated-dense": {"correct": 10, "items": 50, "read_requests": 999999,
                        "read_bytes": 999999},
        "memory": {"correct": 10, "items": 50, "read_requests": 500,
                   "read_bytes": 1000},
        "utility-outcome": {"correct": 20, "items": 50, "read_requests": 200,
                            "read_bytes": 400},
        "utility-ce": {"correct": 20, "items": 50, "read_requests": 500,
                       "read_bytes": 1000},
        "no-memory-gate": {"correct": 10, "items": 50, "read_requests": 200,
                           "read_bytes": 0},
    }
    report = runner.evaluate_arms(arms, 50, threshold=0.5)
    assert report["verdict"]["accuracy_control"] == "gated-dense"
    assert report["verdict"]["byte_control"] == "memory"
    assert report["arms"]["utility-outcome"]["win"] is True
    # equal accuracy to the outcome arm, but no byte saving vs the dense memory
    assert report["arms"]["utility-ce"]["beats_base"] is True
    assert report["arms"]["utility-ce"]["fewer_bytes_than_dense"] is False
    assert report["arms"]["utility-ce"]["win"] is False
    # the gated-dense numbers are not the byte control
    assert report["arms"]["utility-outcome"]["fewer_bytes_than_dense"] is True
    assert report["verdict"]["chosen"] == "utility-outcome"
    assert report["verdict"]["no_memory_gate_parity"] is True

    # without a dense-memory arm there is no byte claim and no win
    no_dense = {key: value for key, value in arms.items()
                if key != "memory"}
    no_dense_report = runner.evaluate_arms(no_dense, 50)
    assert no_dense_report["verdict"]["byte_control"] is None
    assert (no_dense_report["arms"]["utility-outcome"]
            ["fewer_bytes_than_dense"] is None)
    assert no_dense_report["arms"]["utility-outcome"]["win"] is False

    tie = dict(arms)
    tie["utility-outcome"] = {"correct": 11, "items": 50,
                              "read_requests": 200, "read_bytes": 400}
    tied = runner.evaluate_arms(tie, 50)
    assert tied["arms"]["utility-outcome"]["beats_base"] is False
    assert tied["verdict"]["chosen"] is None


def test_prompt_level_bootstrap_clusters_by_problem():
    # 10 problems, each with 16 continuations. Outcome is always 1, base 0, so
    # the prompt-level interval excludes zero; a continuation-level bootstrap
    # would have 160 independent units and a much narrower interval.
    per_prompt = [{"prompt_id": f"p{index}",
                   "score": {"utility-outcome": 1.0, "base": 0.0,
                             "gated-dense": 0.0, "memory": 0.0}}
                  for index in range(10)]
    arms = {
        "base": {"correct": 0, "items": 160, "read_bytes": 0},
        "gated-dense": {"correct": 0, "items": 160, "read_bytes": None},
        "memory": {"correct": 0, "items": 160, "read_bytes": 1000},
        "utility-outcome": {"correct": 160, "items": 160,
                            "read_bytes": 400},
    }
    report = runner.evaluate_arms(arms, 160, per_prompt=per_prompt)
    row = report["arms"]["utility-outcome"]
    assert row["prompt_level"]["distinct_problems"] == 10
    assert row["prompt_level"]["vs_base"]["problems"] == 10
    assert row["prompt_level"]["vs_base"]["ci95"][0] > 0.0
    assert report["verdict"]["distinct_problems"] == 10
    assert row["win"] is True

    # one problem is not enough to establish a positive interval
    row_ci = runner.prompt_paired_bootstrap(
        [{"prompt_id": "p0", "score": {"arm": 1.0, "base": 0.0}}],
        "arm", "base", resamples=100)
    assert row_ci["problems"] == 1
    assert row_ci["ci95"] == [1.0, 1.0]
