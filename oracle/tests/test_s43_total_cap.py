"""CPU tests for the S43 total-cap allocation (pure function, no harness load).

The group-size check needs the evaluation harness and its dataset cache and is
the `--self-test` path of `scripts/s43_score_total_cap.py`, run in the eval
court. These tests pin the allocation arithmetic the driver relies on.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import s43_score_total_cap as cap  # noqa: E402

SIZES = [("mmlu_a", 100), ("mmlu_b", 300), ("mmlu_c", 600)]


def test_allocation_is_exact_and_deterministic():
    first = cap.allocate(200, SIZES)
    second = cap.allocate(200, SIZES)
    assert first == second
    assert sum(count for _, count in first) == 200
    assert [name for name, _ in first] == [name for name, _ in SIZES]
    assert sum(count for _, count in cap.allocate(557, SIZES)) == 557


def test_cap_above_the_available_total_keeps_every_item():
    plan = cap.allocate(10_000, SIZES)
    assert plan == SIZES
    assert sum(count for _, count in plan) == 1000


def test_cap_below_the_leaf_count_fails_closed():
    with pytest.raises(SystemExit):
        cap.allocate(2, SIZES)


def test_every_leaf_gets_at_least_one_item():
    sizes = [(f"leaf_{index}", 1) for index in range(10)]
    plan = cap.allocate(10, sizes)
    assert all(count >= 1 for _, count in plan)
    assert sum(count for _, count in plan) == 10


def test_caps_are_parsed_strictly():
    declared = {"mmlu": {}, "gsm8k": {}}
    assert cap.parse_caps("mmlu=2000", declared, ["mmlu"]) == {"mmlu": 2000}
    with pytest.raises(SystemExit):
        cap.parse_caps("mmlu=0", declared, ["mmlu"])
    with pytest.raises(SystemExit):
        cap.parse_caps("magic=1", declared, ["mmlu"])
    with pytest.raises(SystemExit):
        cap.parse_caps("gsm8k=10", declared, ["mmlu"])
