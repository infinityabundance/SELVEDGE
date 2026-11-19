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

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from s59_student_rollouts import loop_cut_index  # noqa: E402
from s59_build_corpus import (  # noqa: E402
    load_rows, subsample_to_tokens, write_examples)


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
