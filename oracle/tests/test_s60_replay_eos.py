"""S60 — a synthetic EOS must never be supervised.

A FineWeb replay document cut at `--replay-max-tokens` is followed by an EOS
that never occurred in the text. `completion_example` used to supervise it, so
the model was trained to stop mid-document. The examples builder now marks such
rows with `supervise_eos=False`, which keeps the EOS token in the row (the
length is unchanged) but masks its label. Complete documents and verified
solutions keep the EOS supervised.
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "oracle"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from frontier import sft_data  # noqa: E402

EOS = 7


def _load_examples_builder():
    path = os.path.join(ROOT, "scripts", "s55_stage2a_examples.py")
    spec = importlib.util.spec_from_file_location("s55_stage2a_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_complete_solution_supervises_its_eos():
    tokens, labels = sft_data.completion_example([1, 2], [3, 4], EOS)
    assert tokens[-1] == EOS
    assert labels[-1] == EOS


def test_a_truncated_document_masks_the_synthetic_eos():
    tokens, labels = sft_data.completion_example([], [3, 4], EOS,
                                                 supervise_eos=False)
    assert tokens[-1] == EOS            # the row length is unchanged
    assert labels[-1] == sft_data.IGNORE_INDEX
    assert labels[:1] == [3]            # the document tokens stay supervised
    assert sum(1 for value in labels if value != sft_data.IGNORE_INDEX) == 2


def test_packing_counts_and_masks_the_synthetic_eos():
    packed = sft_data.pack_examples(
        [{"prompt_ids": [], "solution_ids": [3, 4], "supervise_eos": False},
         {"prompt_ids": [1], "solution_ids": [5], "supervise_eos": True}],
        seq_len=8, pad_id=0, eos_id=EOS)
    assert packed.report["masked_synthetic_eos"] == 1
    # first row: 2 supervised document tokens, EOS masked; second: 1 + EOS.
    assert packed.report["supervised_tokens"] == 2 + 2
    assert packed.labels[0, 2].item() == sft_data.IGNORE_INDEX


def test_replay_example_marks_truncation():
    builder = _load_examples_builder()
    truncated = builder.replay_example([1, 2, 3], True, 5)
    complete = builder.replay_example([4, 5], False, 6)
    assert truncated["supervise_eos"] is False
    assert complete["supervise_eos"] is True
    assert truncated["truncated"] is True


def test_overflow_profile_separates_dataset_and_difficulty():
    builder = _load_examples_builder()
    rows = [
        {"dataset": "math", "difficulty": 5,
         "prompt_ids": [1] * 10, "solution_ids": [2] * 90},   # 101 tokens
        {"dataset": "math", "difficulty": 2,
         "prompt_ids": [1] * 2, "solution_ids": [2] * 8},     # 11 tokens
        {"dataset": "gsm8k", "difficulty": 3,
         "prompt_ids": [1], "solution_ids": [2]},             # 3 tokens
    ]
    profile = builder.overflow_profile(rows, [50, 200])
    assert profile["50"]["dropped"] == 1
    assert profile["50"]["supervised_tokens_dropped"] == 91
    assert profile["50"]["by_dataset"]["math"]["examples"] == 1
    assert profile["50"]["by_dataset_difficulty"]["math|5"]["supervised_tokens"] == 91
    assert profile["200"]["dropped"] == 0
