"""The committed tiny-fixture reference logits are reproducible and unedited.

`fixtures/tiny-qwen3/reference-logits.json` is the parity oracle for the Rust
runtime. This test regenerates the logits from the committed checkpoint with
`oracle/qwen3_numpy_reference.py` and requires them to match the committed file
exactly, so the reference cannot drift away from the fixture it describes and
cannot be hand-edited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ORACLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORACLE_DIR))

from qwen3_numpy_reference import Qwen3Reference, load_safetensors  # noqa: E402

FIXTURE = ORACLE_DIR.parent / "fixtures" / "tiny-qwen3"


def test_reference_logits_are_reproducible() -> None:
    config_path = FIXTURE / "config.json"
    weights_path = FIXTURE / "model.safetensors"
    reference_path = FIXTURE / "reference-logits.json"
    if not (config_path.exists() and weights_path.exists() and reference_path.exists()):
        pytest.skip("tiny fixture is not present")

    committed = json.loads(reference_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = Qwen3Reference(config, load_safetensors(weights_path))

    cache: dict = {}
    logits: list[list[float]] = []
    generated: list[int] = []
    position = 0
    for token in committed["prompt_ids"]:
        scores = model.forward(token, position, cache)
        logits.append([float(value) for value in scores])
        position += 1
    next_token = int(np.argmax(np.asarray(logits[-1], dtype=np.float32)))
    for _ in range(len(committed["generated_ids"])):
        generated.append(next_token)
        scores = model.forward(next_token, position, cache)
        logits.append([float(value) for value in scores])
        position += 1
        next_token = int(np.argmax(scores))

    assert generated == committed["generated_ids"]
    assert len(logits) == committed["positions"]
    for position, (produced, expected) in enumerate(zip(logits, committed["logits"])):
        assert len(produced) == len(expected)
        for index, (left, right) in enumerate(zip(produced, expected)):
            assert left == right, f"position {position} index {index}: {left} != {right}"
