"""S51: CPU tests for the real-gradient first-moment validation harness.

The GPU capture (`scripts/s50_capture_real_gradients.py`) is queued; these tests
pin the CPU harness that will consume it. They build a fixture with the same
schema and a heterogeneous gradient spectrum (the regime the synthetic S50
receipt identified), then check the comparison, the decision rule and the
bfloat16-first-moment switch.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "oracle"))


def _load():
    path = os.path.join(ROOT, "scripts", "s50_real_gradient_validation.py")
    spec = importlib.util.spec_from_file_location(
        "s50_real_gradient_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load()


def _fixture(tensors: int = 3, elements: int = 1024, steps: int = 4,
             sigma: float = 1.0, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    payload = {}
    for index in range(tensors):
        initial = torch.randn(elements, generator=generator) * 0.02
        scales = torch.exp(sigma * torch.randn(elements, generator=generator))
        directions = torch.randn(steps, elements, generator=generator)
        payload[f"tensor-{index}"] = {
            "parameter": initial,
            "grads": [row * scales for row in directions],
        }
    return {
        "schema": validation.FIXTURE_SCHEMA,
        "metadata": {"steps_captured": steps, "seq_len": 512},
        "tensors": payload,
    }


def test_fixture_schema_is_checked():
    good = _fixture()
    assert set(good["tensors"])
    bad = dict(good, schema="something/else")
    path = os.path.join(ROOT, "results", "artifacts", ".s51-bad-fixture.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        torch.save(bad, path)
        with pytest.raises(SystemExit):
            validation.load_fixture(path)
        malformed = _fixture()
        next(iter(malformed["tensors"].values()))["grads"] = []
        torch.save(malformed, path)
        with pytest.raises(SystemExit):
            validation.load_fixture(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_first_step_is_exact_for_both_storages():
    fixture = _fixture(steps=2)
    for storage in validation.STORAGES:
        arm = validation._evaluate_arm(
            fixture["tensors"], 2, storage, validation.DEFAULT_LR,
            validation.BETAS, validation.EPS, validation.BLOCK)
        # At step 1 both moments are the exact fp32 sums in both arms.
        assert arm["per_step_relative_l2"][0] < 1e-6, storage
        assert arm["per_step_relative_l2"][1] > 1e-5, storage


def test_bfloat16_first_moment_is_closer_to_adamw_than_int8():
    fixture = _fixture(tensors=2, elements=1024, steps=3, sigma=1.0)
    int8 = validation._evaluate_arm(
        fixture["tensors"], 3, "int8", validation.DEFAULT_LR,
        validation.BETAS, validation.EPS, validation.BLOCK)
    bf16 = validation._evaluate_arm(
        fixture["tensors"], 3, "bfloat16", validation.DEFAULT_LR,
        validation.BETAS, validation.EPS, validation.BLOCK)
    assert bf16["worst_relative_l2"] < int8["worst_relative_l2"]
    assert bf16["final_parameter_relative_error"] <= int8["final_parameter_relative_error"]
    assert bf16["moment_bytes_per_element"] > int8["moment_bytes_per_element"]
    assert bf16["moment_bytes_per_element"] == pytest.approx(4.0)
    assert int8["moment_bytes_per_element"] == pytest.approx(3.015625)


def test_decision_retains_int8_when_it_is_within_tolerance():
    arms = {
        "int8": {**{key: 0.0 for key in validation.TOLERANCE},
                 "max_update_ratio": 1.0},
        "bfloat16": {key: 0.0 for key in validation.TOLERANCE},
    }
    decision = validation.decide(arms)
    assert decision["status"] == "retain-int8-first-moment"
    assert decision["int8_failures"] == []


def test_decision_switches_when_int8_fails_and_bfloat16_passes():
    arms = {
        "int8": {"worst_relative_l2": 0.55, "max_update_ratio": 1.2,
                 "final_parameter_relative_error": 0.005},
        "bfloat16": {"worst_relative_l2": 0.01, "max_update_ratio": 1.1,
                     "final_parameter_relative_error": 0.0001},
    }
    decision = validation.decide(arms)
    assert decision["status"] == "switch-first-moment-to-bfloat16"
    assert any("worst_relative_l2" in item for item in decision["int8_failures"])
    assert decision["bfloat16_failures"] == []


def test_decision_does_not_switch_when_both_layouts_fail():
    arms = {
        "int8": {"worst_relative_l2": 0.55, "max_update_ratio": 5.0,
                 "final_parameter_relative_error": 0.1},
        "bfloat16": {"worst_relative_l2": 0.45, "max_update_ratio": 4.0,
                     "final_parameter_relative_error": 0.09},
    }
    decision = validation.decide(arms)
    assert decision["status"] == "first-moment-storage-not-the-cause"
    assert decision["int8_failures"] and decision["bfloat16_failures"]


def test_evaluate_is_deterministic_and_reports_the_memory_projection():
    fixture = _fixture(tensors=1, elements=1024, steps=3)
    first = validation.evaluate(fixture, steps=3)
    second = validation.evaluate(fixture, steps=3)
    assert first["arms"] == second["arms"]
    assert first["decision"] == second["decision"]
    memory = first["memory"]
    assert memory["bfloat16_projected_peak_gib"] > memory["measured_int8_projected_peak_gib"]
    assert memory["bfloat16_projected_peak_gib"] < memory["vram_limit_gib"]
    assert memory["bfloat16_fits"] is True
    assert 0.0 <= first["arms"]["int8"]["worst_relative_l2"]
    assert first["arms"]["int8"]["worst_step"] >= 1
