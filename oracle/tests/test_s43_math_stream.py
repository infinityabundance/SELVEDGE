"""CPU tests for the S43 math-weighted stream plan.

The build itself is measured by `math_stream.py build` and its receipt. These
tests pin the registered arithmetic and the contamination policy so a later
edit cannot silently change the mixture or relax the guard.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import math_stream  # noqa: E402


def test_cycle_is_exact_and_the_totals_are_the_registered_ones():
    cycle = sum(size for _, size in math_stream.CYCLE)
    assert cycle == 10_000
    assert math_stream.CYCLES_PER_SHARD == 20
    assert math_stream.SHARD_TOKENS == 200_000
    assert math_stream.SHARDS == 100
    assert math_stream.TOTAL_TOKENS == 20_000_000
    per_source = {name: size * math_stream.CYCLES_PER_SHARD * math_stream.SHARDS
                  for name, size in math_stream.CYCLE}
    assert per_source == {"numina": 8_000_000, "math": 2_200_000,
                          "gsm8k": 1_200_000, "fineweb": 8_600_000}
    assert sum(per_source.values()) == math_stream.TOTAL_TOKENS
    assert per_source["numina"] + per_source["math"] + per_source["gsm8k"] \
        == 11_400_000
    assert math_stream.MATH_FRACTION == 0.57


def test_every_shard_has_the_same_composition():
    per_shard = {name: size * math_stream.CYCLES_PER_SHARD
                 for name, size in math_stream.CYCLE}
    assert sum(per_shard.values()) == math_stream.SHARD_TOKENS
    assert per_shard == {"numina": 80_000, "math": 22_000,
                         "gsm8k": 12_000, "fineweb": 86_000}


def test_contamination_policy_is_fail_closed_at_the_discard_threshold():
    assert math_stream.DISCARD == 0.2
    assert math_stream.FAIL_CLOSED == 0.5
    assert math_stream.SHINGLE_WIDTH == 5
    # the build asserts that no retained document reaches DISCARD; a near
    # verbatim copy at FAIL_CLOSED is excluded and flagged
    assert math_stream.DISCARD < math_stream.FAIL_CLOSED


def test_sealed_holdback_is_in_the_evaluation_index():
    items = math_stream.sealed_items()
    assert len(items) >= 20
    assert any("SEALED.json" in name for name, _ in items)
    assert any("sealed-" in name for name, _ in items)


def test_every_source_has_a_pinned_revision_and_licence():
    for name, spec in math_stream.SOURCES.items():
        assert len(spec["revision"]) == 40, name
        assert spec["license"], name
        assert spec["repo_id"], name
    assert math_stream.SOURCES["math"]["license"] == "MIT"
    assert math_stream.SOURCES["gsm8k"]["license"] == "MIT"
    assert math_stream.SOURCES["numina"]["license"] == "Apache-2.0"
    assert math_stream.SOURCES["fineweb"]["license"] == "ODC-By-1.0"
    assert math_stream.SOURCES["numina"]["lfs_oid"].startswith("25c0fa2c")
