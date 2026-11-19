"""S59: CPU tests for the stage-2b rollout and corpus machinery.

The GPU parts (generation, training) are receipted by their own runs; these
tests cover the rules whose arithmetic must be exact before any GPU hour is
spent:

* the registered loop cut (two rules, the cut position, and the non-cut cases);
* the whole-row supervised-token matching used to build the matched control and
  the 50/50 mixture;
* the corpus loader's hold-out exclusion and within-prompt exact dedup.
"""

from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from s59_student_rollouts import (  # noqa: E402
    loop_cut_index, ledger_entry, rejected_row, summarise_ledger,
    write_ledger_entry)
from s59_build_corpus import (  # noqa: E402
    load_rows, plan_mixture, read_budget_receipt, subsample_to_tokens,
    write_examples)


def test_no_cut_on_short_sequence():
    assert loop_cut_index([]) is None
    assert loop_cut_index([1, 2, 3]) is None


def test_no_cut_on_nonrepeating_tail():
    ids = list(range(200))
    assert loop_cut_index(ids) is None


def test_rule_a_sixteen_token_block_four_times():
    block = list(range(100, 116))
    ids = list(range(50)) + block * 4
    # The first copy of the block starts at index 50.
    assert loop_cut_index(ids) == 50


def test_rule_b_forty_eight_token_block_three_times():
    block = list(range(1000, 1048))
    ids = list(range(10)) + block * 3
    assert loop_cut_index(ids) == 10


def test_rule_a_needs_four_copies_not_three():
    block = list(range(100, 116))
    ids = list(range(50)) + block * 3
    assert loop_cut_index(ids) is None


def test_rule_b_takes_precedence_for_a_long_block():
    block = list(range(2000, 2048))
    ids = list(range(7)) + block * 3
    assert loop_cut_index(ids) == 7


def test_cut_leaves_the_prefix_before_the_repetition():
    block = [9] * 16
    ids = [1, 2, 3] + block * 4
    cut = loop_cut_index(ids)
    assert cut == 3
    assert ids[:cut] == [1, 2, 3]


def _row(identifier: str, dataset: str, tokens: int, sample: int = 0) -> dict:
    return {"id": identifier, "dataset": dataset,
            "prompt_ids": [1, 2, 3],
            "solution_ids": list(range(tokens)),
            "sample_index": sample,
            "solution": "x" * tokens, "prompt": "p"}


def test_subsample_respects_the_target_within_one_row():
    rows = [_row(f"id{i}", "gsm8k", 100) for i in range(20)]
    selected, report = subsample_to_tokens(rows, 950, seed=7)
    total = sum(len(row["solution_ids"]) for row in selected)
    assert abs(total - 950) <= 100
    assert report["achieved"] == total
    assert report["target"] == 950


def test_subsample_is_deterministic():
    rows = [_row(f"id{i}", "math", 10 + i) for i in range(30)]
    first, _ = subsample_to_tokens(rows, 200, seed=11)
    second, _ = subsample_to_tokens(rows, 200, seed=11)
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_subsample_keeps_everything_below_the_target():
    rows = [_row(f"id{i}", "math", 10) for i in range(5)]
    selected, report = subsample_to_tokens(rows, 10_000, seed=3)
    assert len(selected) == 5
    assert report["achieved"] == 50


def test_load_rows_excludes_the_holdout_and_dedups_within_a_prompt(tmp_path):
    path = tmp_path / "accepted.jsonl"
    import json
    rows = [
        {"id": "a", "dataset": "gsm8k", "prompt_ids": [1],
         "solution_ids": [7, 8], "solution": "s", "prompt": "p"},
        {"id": "a", "dataset": "gsm8k", "prompt_ids": [1],
         "solution_ids": [7, 8], "solution": "s", "prompt": "p"},
        {"id": "b", "dataset": "gsm8k", "prompt_ids": [1],
         "solution_ids": [9], "solution": "t", "prompt": "p"},
        {"id": "held", "dataset": "math", "prompt_ids": [1],
         "solution_ids": [5], "solution": "u", "prompt": "p"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    loaded, report = load_rows(str(path), {"held"}, "student")
    assert [row["id"] for row in loaded] == ["a", "b"]
    assert report["excluded_holdout_rows"] == 1
    assert report["duplicate_rows_removed"] == 1
    assert all(row["kind"] == "student" for row in loaded)


def test_load_rows_dedups_exactly_across_shards_but_keeps_k_samples(tmp_path):
    import json
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    row = {"id": "p", "dataset": "gsm8k", "prompt_ids": [1],
           "solution_ids": [7, 8], "solution": "s", "prompt": "q"}
    other = {"id": "p", "dataset": "gsm8k", "prompt_ids": [1],
             "solution_ids": [9], "solution": "t", "prompt": "q"}
    shard_a.write_text(json.dumps(row) + "\n", encoding="utf-8")
    shard_b.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n",
                       encoding="utf-8")
    seen_global, seen_prompt = set(), {}
    first, report_a = load_rows(str(shard_a), set(), "student",
                                seen_global, seen_prompt)
    second, report_b = load_rows(str(shard_b), set(), "student",
                                 seen_global, seen_prompt)
    # the duplicate exact sequence is removed across shards; the second distinct
    # sample for the same prompt is a legitimate K>1 row and is kept.
    assert [r["solution_ids"] for r in first] == [[7, 8]]
    assert [r["solution_ids"] for r in second] == [[9]]
    assert report_a["duplicate_rows_removed"] == 0
    assert report_b["duplicate_rows_removed"] == 1


def test_write_examples_preserves_the_masked_synthetic_eos(tmp_path):
    import json
    path = tmp_path / "examples.jsonl"
    rows = [
        {"id": "a", "dataset": "gsm8k", "prompt_ids": [1],
         "solution_ids": [2, 3]},
        {"id": "replay", "kind": "replay", "prompt_ids": [],
         "solution_ids": [4, 5], "supervise_eos": False},
    ]
    block = write_examples(str(path), rows)
    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert block["rows"] == 2
    assert "supervise_eos" not in written[0]
    assert written[1]["supervise_eos"] is False
    assert written[1]["kind"] == "replay"


# ---------------------------------------------------------------------------
# S63: per-row stopping and the per-problem ledger


def _scripted_model():
    """A tiny real `PreTrainedModel` whose next token is scripted per row.

    Row 0 emits a 16-token block four times (a registered loop); row 1 emits
    distinct filler tokens and then EOS. Both prompts are one token, so the
    criterion's `prompt_width` is 1. The forward is positionwise and ignores
    the weights: this tests generation control, not a language model.
    """
    import torch
    import torch.nn as nn
    from transformers import (GenerationMixin, PretrainedConfig,
                              PreTrainedModel)
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class ScriptedConfig(PretrainedConfig):
        model_type = "s63_scripted"

        def __init__(self, vocab_size=256, **kwargs):
            self.vocab_size = int(vocab_size)
            # Some transformers generation paths read these even for a scripted
            # model; one layer of width 8 is enough for the control test.
            self.num_hidden_layers = 1
            self.num_attention_heads = 1
            self.hidden_size = 8
            super().__init__(pad_token_id=0, eos_token_id=1, bos_token_id=2,
                             **kwargs)

    class ScriptedLM(PreTrainedModel, GenerationMixin):
        config_class = ScriptedConfig

        def __init__(self, config):
            super().__init__(config)
            self.embed = nn.Embedding(config.vocab_size, 8)
            self.head = nn.Linear(8, config.vocab_size, bias=False)
            self.script = {}
            self.prompt_width = None

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            rows, width = input_ids.shape
            token = torch.full((rows,), self.config.pad_token_id,
                               dtype=torch.long, device=input_ids.device)
            for row in range(rows):
                marker = int(input_ids[row, 0].item())
                generated = width - self.prompt_width
                token[row] = self.script[marker](generated)
            logits = torch.zeros(rows, width, self.config.vocab_size,
                                 device=input_ids.device)
            logits[:, -1, :].scatter_(1, token[:, None], 50.0)
            return CausalLMOutputWithPast(logits=logits)

        def prepare_inputs_for_generation(self, input_ids, attention_mask=None,
                                          **kwargs):
            return {"input_ids": input_ids,
                    "attention_mask": attention_mask}

    return ScriptedConfig(), ScriptedLM


def test_stopping_criterion_returns_a_per_row_booltensor_and_stops_one_row():
    """The old whole-batch bool could not stop row 0 until row 1 finished."""
    import torch

    from s59_student_rollouts import LoopCutStopping

    config, ScriptedLM = _scripted_model()
    model = ScriptedLM(config).eval()
    block = list(range(20, 36))
    model.script = {
        10: lambda n: block[n % 16],          # row 0: four block copies
        11: lambda n: 1 if n >= 80 else 100 + n,  # row 1: EOS at step 80
    }
    model.prompt_width = 1
    input_ids = torch.tensor([[10], [11]], dtype=torch.long)
    criterion = LoopCutStopping(prompt_width=1, eos=1, pad=0)
    output = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=100, do_sample=False, num_return_sequences=1,
        pad_token_id=0, eos_token_id=1, use_cache=False,
        stopping_criteria=[criterion])

    # Row 0 stopped by the loop rule alone; row 1 emitted EOS.
    assert criterion.looped == {0}
    assert criterion.eos_seen == {1}
    assert criterion.cut_at[0] == 0
    assert 0 in criterion.finished and 1 in criterion.finished

    reference = criterion(input_ids, None)
    assert reference.dtype == torch.bool
    assert reference.shape == (2,)
    assert bool(reference.all())

    generated = output[:, 1:]
    row0 = int((generated[0] != 0).sum().item())
    row1 = int((generated[1] != 0).sum().item())
    # Row 0 was cut at 64 tokens; row 1 continued to 80 filler tokens + EOS.
    assert row0 == 64
    assert row1 == 81
    assert row1 != row0


def test_generate_reports_textual_replay_tokens_not_compute_saved():
    """The cut's token effect is measured, and labelled as textual replay."""
    from s59_student_rollouts import generate

    config, ScriptedLM = _scripted_model()
    model = ScriptedLM(config).eval()
    block = list(range(20, 36))
    model.script = {
        10: lambda n: block[n % 16],
        11: lambda n: 1 if n >= 80 else 100 + n,
    }
    model.prompt_width = 1

    class FakeTokenizer:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=None,
                     add_special_tokens=None):
            import torch
            ids = torch.tensor([[10], [11]], dtype=torch.long)
            return {"input_ids": ids,
                    "attention_mask": torch.ones_like(ids)}

        def decode(self, ids, skip_special_tokens=True):
            return "x"

    rows, stats = generate(
        model, FakeTokenizer(), ["a", "b"], samples=1,
        max_new_tokens=100, temperature=0.8, top_p=0.95, device="cpu")
    # Row 0 emitted a 64-token loop, cut at its first token (0 -> 64 cut).
    # Row 1 emitted 80 filler tokens plus EOS (81).
    assert stats["loop_cut"] == 1
    assert stats["pre_cut_generated_tokens"] == 64 + 81
    assert stats["textual_replay_tokens_cut"] == 64
    assert stats["generated_tokens"] == 81
    assert rows[0] == []
    assert len(rows[1]) == 81
    assert stats["compute_saved"] is False
    assert "textual replay" in stats["token_savings_basis"]


def test_ledger_p_is_successes_over_attempts_not_accepted_rows():
    task = {"id": "p", "dataset": "gsm8k", "samples": 4,
            "max_new_tokens": 2048}
    entry = ledger_entry(task)
    entry.update({"attempts": 4, "verified_correct": 2, "wrong": 1,
                  "truncated": 1, "duplicates": 1, "accepted_rows": 1,
                  "accepted_solution_tokens": 100})
    # p from attempts: 2/4. A p from accepted rows would be 1/1 = 1.0 and the
    # p(1-p) weight would be zero, exactly the defect this ledger removes.
    finalised = write_ledger_entry(io.StringIO(), entry)
    assert finalised["p"] == 0.5
    assert finalised["weight_p1mp"] == 0.25
    assert "duplicates" in finalised["p_basis"]
    summary = summarise_ledger([finalised])
    assert summary["p_basis"].startswith("verified_correct / attempts")
    assert summary["attempts"] == 4


def test_rejected_row_keeps_token_ids_for_the_preference_pass():
    task = {"id": "p", "dataset": "math", "target": "7",
            "samples": 2, "max_new_tokens": 2048}
    entry = ledger_entry(task, [1, 2])
    entry.update({"attempts": 2, "verified_correct": 0, "wrong": 2})
    write_ledger_entry(io.StringIO(), entry)
    row = rejected_row(task, [1, 2], 1, [9, 8, 7], "wrong", False, False,
                       entry)
    assert row["prompt_ids"] == [1, 2]
    assert row["solution_ids"] == [9, 8, 7]
    assert row["verdict"] == "wrong"
    assert row["provenance"] == "student-failed-continuation"
    assert row["p"] == 0.0


def test_ledger_entry_carries_the_prefix_needed_for_recovery():
    task = {"id": "p", "dataset": "gsm8k", "target": "4",
            "samples": 2, "max_new_tokens": 10}
    entry = ledger_entry(task, [11, 12, 13])
    # A p=0 problem has no accepted row, so the ledger is the only place the
    # shared prefix survives for the teacher-guided recovery pair.
    assert entry["prompt_ids"] == [11, 12, 13]
    assert entry["target"] == "4"


def test_truncated_correct_attempt_counts_for_p_but_not_accepted():
    task = {"id": "cap", "dataset": "math", "samples": 2,
            "max_new_tokens": 2048}
    entry = ledger_entry(task)
    # One attempt boxed the correct answer before the cap but is truncated, so
    # it is a verified success for p and not an accepted training row.
    entry.update({"attempts": 2, "verified_correct": 1, "truncated": 1,
                  "accepted_rows": 0})
    write_ledger_entry(io.StringIO(), entry)
    assert entry["p"] == 0.5
    assert entry["weight_p1mp"] == 0.25
    assert entry["accepted_rows"] == 0


def test_ledger_row_is_written_with_zero_accepted_rows():
    task = {"id": "hard", "dataset": "math", "samples": 2,
            "max_new_tokens": 2048}
    entry = ledger_entry(task)
    entry.update({"attempts": 2, "wrong": 2, "accepted_rows": 0})
    handle = io.StringIO()
    write_ledger_entry(handle, entry)
    line = handle.getvalue().strip()
    assert line
    import json
    row = json.loads(line)
    assert row["id"] == "hard"
    assert row["accepted_rows"] == 0
    assert row["p"] == 0.0
    assert row["weight_p1mp"] == 0.0


# ---------------------------------------------------------------------------
# S63: the control-teacher budget comes from a separate corpus receipt


def _receipt(tmp_path, budget=1000, arm="rft-only"):
    import json
    path = tmp_path / "rft-only-corpus-receipt.json"
    path.write_text(json.dumps({
        "schema": "selvedge/s59-stage2b-corpus/v1",
        "arm": arm,
        "supervised_solution_tokens": budget,
        "revision": "testrevision",
        "stamp": "20261119T000000Z",
    }), encoding="utf-8")
    return path


def test_all_three_arms_get_matched_supervised_token_budgets(tmp_path):
    """The control-teacher arm must not be subsampled to a zero budget.

    This is the defect: `control-teacher` had no student rows of its own, so
    `student_tokens` was 0 and `subsample_to_tokens(teacher, 0)` produced an
    empty control. The budget now comes from the separately receipted student
    corpus, and all three arms are built from the same planner.
    """
    receipt = read_budget_receipt(str(_receipt(tmp_path, budget=1000)))
    budget = receipt["supervised_solution_tokens"]
    assert budget == 1000

    student = [_row(f"s{i}", "gsm8k", 100, sample=i) for i in range(10)]
    teacher = [_row(f"t{i}", "gsm8k", 100, sample=i) for i in range(20)]

    rft, rft_report = plan_mixture("rft-only", student, teacher, budget, 5900)
    control, control_report = plan_mixture(
        "control-teacher", [], teacher, budget, 5900)
    mixed, mixed_report = plan_mixture(
        "rft-teacher-5050", student, teacher, sum(
            len(row["solution_ids"]) for row in student), 5900)

    def tokens(rows):
        return sum(len(row["solution_ids"]) for row in rows)

    assert tokens(rft) == 1000
    assert tokens(control) > 0, "zero-token control is the defect"
    assert abs(tokens(control) - budget) <= 100
    assert control_report["teacher_tokens"] == tokens(control)
    # The 50/50 arm matches its two halves to each other by whole rows.
    assert mixed_report["student_tokens"] == mixed_report["teacher_tokens"]
    assert abs(tokens(mixed) - 2000) <= 200
    assert len(mixed) > 0


def test_control_teacher_refuses_a_zero_token_budget_receipt(tmp_path):
    try:
        read_budget_receipt(str(_receipt(tmp_path, budget=0)))
    except SystemExit as exc:
        assert "positive budget" in str(exc)
    else:
        raise AssertionError("a zero-token budget receipt must fail closed")


def test_budget_receipt_must_name_a_student_arm(tmp_path):
    try:
        read_budget_receipt(str(_receipt(tmp_path, budget=1000,
                                        arm="control-teacher")))
    except SystemExit as exc:
        assert "student arm" in str(exc)
    else:
        raise AssertionError("a control-teacher receipt is not a budget source")


def test_budget_receipt_checks_the_recorded_examples_hash(tmp_path):
    import hashlib
    import json

    examples = tmp_path / "examples.jsonl"
    examples.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(examples.read_bytes()).hexdigest()
    path = _receipt(tmp_path, budget=1000)
    payload = json.loads(path.read_text())
    payload["examples"] = {"path": str(examples), "sha256": digest}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_budget_receipt(str(path))["supervised_solution_tokens"] == 1000

    examples.write_text("{}\n{}\n", encoding="utf-8")
    try:
        read_budget_receipt(str(path))
    except SystemExit as exc:
        assert "sha256" in str(exc)
    else:
        raise AssertionError("a stale examples hash must fail closed")
