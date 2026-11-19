"""S60 — the interpolation diagnostic's arithmetic and selection rule.

The diagnostic interpolates the post-trained weights and the S58 best
checkpoint, evaluates on a train-split calibration set and chooses a
coefficient on calibration only. These tests pin the two pure pieces: the
interpolation must be exact at both endpoints and reject a mismatched state
dict, and the selection rule must apply the MMLU retention bound without ever
looking at a test number.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load_module():
    path = os.path.join(ROOT, "scripts", "s60_interpolate.py")
    spec = importlib.util.spec_from_file_location("s60_interpolate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s60 = _load_module()


def test_endpoints_are_exact_and_the_midpoint_is_the_average():
    base = {"w": torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)}
    candidate = {"w": torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)}
    assert torch.equal(s60.interpolate_state_dict(base, candidate, 0.0)["w"],
                       base["w"])
    assert torch.equal(s60.interpolate_state_dict(base, candidate, 1.0)["w"],
                       candidate["w"])
    half = s60.interpolate_state_dict(base, candidate, 0.5)["w"]
    assert torch.allclose(half, torch.tensor([1.0, 3.0, 6.0]))


def test_dtype_is_preserved():
    base = {"w": torch.zeros(3, dtype=torch.bfloat16)}
    candidate = {"w": torch.ones(3, dtype=torch.bfloat16)}
    mixed = s60.interpolate_state_dict(base, candidate, 0.25)["w"]
    assert mixed.dtype == torch.bfloat16


def test_a_key_mismatch_fails_closed():
    with torch.no_grad():
        try:
            s60.interpolate_state_dict({"a": torch.zeros(1)},
                                       {"b": torch.zeros(1)}, 0.5)
        except ValueError as exc:
            assert "keys differ" in str(exc)
        else:
            raise AssertionError("a key mismatch must fail closed")


def test_a_shape_mismatch_fails_closed():
    try:
        s60.interpolate_state_dict({"a": torch.zeros(2)},
                                   {"a": torch.zeros(3)}, 0.5)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("a shape mismatch must fail closed")


def test_selection_uses_calibration_and_the_retention_bound():
    rows = [
        {"coefficient": 0.0, "gsm8k_cal": 0.50, "math_cal": 0.38, "mmlu_cal": 0.50},
        {"coefficient": 0.25, "gsm8k_cal": 0.52, "math_cal": 0.40, "mmlu_cal": 0.50},
        {"coefficient": 0.50, "gsm8k_cal": 0.57, "math_cal": 0.27, "mmlu_cal": 0.48},
        {"coefficient": 1.00, "gsm8k_cal": 0.57, "math_cal": 0.27, "mmlu_cal": 0.46},
    ]
    selection = s60.choose_coefficient(rows)
    # 0.5 and 1.0 are excluded by the 1.0-point retention bound; 0.25 gains on
    # both calibration tasks.
    assert selection["coefficient"] == 0.25
    assert selection["eligible"] is True
    assert selection["mmlu_present"] is True


def test_selection_ties_choose_the_smaller_coefficient():
    rows = [
        {"coefficient": 0.0, "gsm8k_cal": 0.50, "math_cal": 0.40},
        {"coefficient": 0.25, "gsm8k_cal": 0.51, "math_cal": 0.41},
        {"coefficient": 0.5, "gsm8k_cal": 0.51, "math_cal": 0.41},
    ]
    selection = s60.choose_coefficient(rows)
    assert selection["coefficient"] == 0.25
    assert selection["mmlu_present"] is False
