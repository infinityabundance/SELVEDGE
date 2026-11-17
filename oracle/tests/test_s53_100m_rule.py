"""CPU contract for the S53 fixed-batch 100M-extension rule.

The rule replaced the three-harness-point OLS slope. These tests pin the
arithmetic: the fitted trend, the last-interval share, the KL preservation
check, the harness confirmation, and the fail-closed checkpoint/hash checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRIPT = os.path.join(ROOT, "scripts", "s50_bar_curve.py")

spec = importlib.util.spec_from_file_location("s50_bar_curve", SCRIPT)
curve = importlib.util.module_from_spec(spec)
sys.modules["s50_bar_curve"] = curve
spec.loader.exec_module(curve)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _fixture(tmp_path, ce=(1.0, 0.9, 0.88), kl=(0.30, 0.28, 0.27)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = [{"checkpoint": "init", "ce": 1.2, "kl": 0.5, "sha256": None,
                "batches": 8, "positions": 4096}]
    for index, (ce_value, kl_value) in enumerate(zip(ce, kl), start=1):
        path = run_dir / f"ckpt-{index * 100:06d}.pt"
        path.write_bytes(f"checkpoint-{index}".encode())
        results.append({"checkpoint": path.name, "ce": ce_value,
                        "kl": kl_value, "sha256": _sha(str(path)),
                        "batches": 8, "positions": 4096})
    receipt = {
        "schema": "selvedge/s50-fixed-batch-kl/v1",
        "run_dir": str(run_dir),
        "start_token": 2_000_000,
        "seq_len": 512,
        "batch_count": 8,
        "temperature": 2.0,
        "results": results,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return str(receipt_path), str(run_dir)


def test_checkpoint_step_parses_only_ckpt_labels():
    assert curve._checkpoint_step("init") is None
    assert curve._checkpoint_step("ckpt-002441.pt") == 2441
    with pytest.raises(SystemExit):
        curve._checkpoint_step("latest.pt")


def test_fixed_batch_trend_uses_every_saved_checkpoint(tmp_path):
    receipt, run_dir = _fixture(tmp_path)
    trend = curve._fixed_batch_trend(receipt, run_dir, tokens_per_step=2048)
    assert [point["step"] for point in trend["checkpoints"]] == [100, 200, 300]
    assert [point["tokens_seen"] for point in trend["checkpoints"]] == [
        100 * 2048, 200 * 2048, 300 * 2048]
    assert trend["ce_slope_per_million_tokens"] < 0.0
    assert trend["kl_slope_per_million_tokens"] < 0.0
    assert trend["ce_last_interval"] == pytest.approx(0.02)
    assert trend["ce_total_drop_after_first_checkpoint"] == pytest.approx(0.12)
    assert trend["ce_last_share"] == pytest.approx(0.02 / 0.12)
    assert trend["init"]["ce"] == pytest.approx(1.2)


def test_rule_requires_slope_last_interval_share_and_kl(tmp_path):
    receipt, run_dir = _fixture(tmp_path)
    trend = curve._fixed_batch_trend(receipt, run_dir, tokens_per_step=2048)
    decision = curve._extension_decision(
        trend, final_math=0.40, previous_math=0.39, previous_error=0.01,
        base_math=0.30)
    assert decision["still_paying_off"] is True
    assert decision["harness_confirms"] is True
    assert decision["justified"] is True

    # a CE that rose in the last interval is not "still paying off"
    receipt2, run_dir2 = _fixture(tmp_path / "rising", ce=(1.0, 0.9, 0.91))
    trend2 = curve._fixed_batch_trend(receipt2, run_dir2, tokens_per_step=2048)
    assert curve._extension_decision(
        trend2, 0.40, 0.39, 0.01, 0.30)["still_paying_off"] is False

    # a flat tail (last interval carries < 10% of the total drop) fails
    receipt3, run_dir3 = _fixture(tmp_path / "flat", ce=(1.0, 0.80, 0.799))
    trend3 = curve._fixed_batch_trend(receipt3, run_dir3, tokens_per_step=2048)
    assert trend3["ce_last_share"] < curve.CE_LAST_SHARE_MIN
    assert curve._extension_decision(
        trend3, 0.40, 0.39, 0.01, 0.30)["still_paying_off"] is False

    # a rising teacher KL fails the preservation check
    receipt4, run_dir4 = _fixture(tmp_path / "kl", kl=(0.30, 0.28, 0.29))
    trend4 = curve._fixed_batch_trend(receipt4, run_dir4, tokens_per_step=2048)
    assert curve._extension_decision(
        trend4, 0.40, 0.39, 0.01, 0.30)["still_paying_off"] is False


def test_harness_endpoints_can_contradict_the_fixed_batch_reading(tmp_path):
    receipt, run_dir = _fixture(tmp_path)
    trend = curve._fixed_batch_trend(receipt, run_dir, tokens_per_step=2048)
    decision = curve._extension_decision(
        trend, final_math=0.30, previous_math=0.39, previous_error=0.01,
        base_math=0.25)
    assert decision["still_paying_off"] is True
    assert decision["harness_confirms"] is False
    assert decision["justified"] is False

    # final below base is not a confirmation even when the slope is negative
    decision2 = curve._extension_decision(
        trend, final_math=0.20, previous_math=0.39, previous_error=0.01,
        base_math=0.25)
    assert decision2["harness_confirms"] is False


def test_fixed_batch_trend_fails_closed_on_missing_or_mismatched(tmp_path):
    receipt, run_dir = _fixture(tmp_path)
    open(os.path.join(run_dir, "ckpt-000400.pt"), "wb").write(b"extra")
    with pytest.raises(SystemExit):
        curve._fixed_batch_trend(receipt, run_dir, tokens_per_step=2048)

    receipt2, run_dir2 = _fixture(tmp_path / "mismatch")
    with open(os.path.join(run_dir2, "ckpt-000100.pt"), "wb") as handle:
        handle.write(b"changed")
    with pytest.raises(SystemExit):
        curve._fixed_batch_trend(receipt2, run_dir2, tokens_per_step=2048)


def test_fixed_batch_trend_needs_two_checkpoints(tmp_path):
    run_dir = tmp_path / "one"
    run_dir.mkdir()
    path = run_dir / "ckpt-000100.pt"
    path.write_bytes(b"only")
    receipt = {
        "schema": "selvedge/s50-fixed-batch-kl/v1",
        "seq_len": 512, "batch_count": 8,
        "results": [{"checkpoint": "init", "ce": 1.0, "kl": 0.5, "sha256": None},
                    {"checkpoint": path.name, "ce": 0.9, "kl": 0.4,
                     "sha256": _sha(str(path))}],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit):
        curve._fixed_batch_trend(str(receipt_path), str(run_dir),
                                 tokens_per_step=2048)
