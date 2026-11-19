"""S60 — the student identity is verified before a run, not assumed from a path.

Stage 2a/2b train the post-trained 0.8B (`2fc06364…`), while `train.py`'s
default was the Base checkpoint. The launcher now passes the post-trained path
*and* the trainer verifies it against the identity lock: the model id, the
revision, the directory name, the first-shard weight hash and the licence hash.
A mismatch is fatal, never a warning.

The launcher text is also pinned here, because a command that merely fails to
mention the Base path is not proof that it names the post-trained one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "oracle"))

from frontier import identity  # noqa: E402

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_ID = "Qwen/Qwen3.5-0.8B"


def _model_dir(tmp_path) -> str:
    directory = tmp_path / REVISION
    directory.mkdir()
    (directory / "config.json").write_text("{\"model_type\": \"qwen3\"}\n")
    (directory / "LICENSE").write_text("Apache-2.0\n")
    shard = (directory / "model.safetensors-00001-of-00001.safetensors")
    shard.write_bytes(b"weights")
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": shard.name, "b": shard.name}}))
    return str(directory)


def _lock(tmp_path, directory: str, **overrides) -> str:
    shard = os.path.join(directory,
                         "model.safetensors-00001-of-00001.safetensors")
    entry = {
        "id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": hashlib.sha256(open(shard, "rb").read()).hexdigest(),
        "licence_file_sha256": hashlib.sha256(b"Apache-2.0\n").hexdigest(),
    }
    entry.update(overrides)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"schema": "test", "target": entry}))
    return str(path)


def test_a_matching_snapshot_verifies(tmp_path):
    directory = _model_dir(tmp_path)
    report = identity.verify_student(directory, _lock(tmp_path, directory),
                                     MODEL_ID, REVISION)
    assert report["verified"] is True
    assert report["weight_sha256"] == report["shards"][
        "model.safetensors-00001-of-00001.safetensors"]["sha256"]


def test_a_mutated_shard_fails_closed(tmp_path):
    directory = _model_dir(tmp_path)
    lock = _lock(tmp_path, directory)
    with open(os.path.join(
            directory, "model.safetensors-00001-of-00001.safetensors"), "wb") as f:
        f.write(b"tampered")
    with pytest.raises(identity.IdentityMismatch, match="sha256"):
        identity.verify_student(directory, lock, MODEL_ID, REVISION)


def test_a_wrong_revision_fails_closed(tmp_path):
    directory = _model_dir(tmp_path)
    with pytest.raises(identity.IdentityMismatch, match="revision"):
        identity.verify_student(directory, _lock(tmp_path, directory),
                                MODEL_ID, "0" * 40)


def test_a_directory_named_for_another_revision_fails_closed(tmp_path):
    directory = _model_dir(tmp_path)
    lock = _lock(tmp_path, directory)
    other = tmp_path / ("1" * 40)
    other.mkdir()
    for name in os.listdir(directory):
        os.replace(os.path.join(directory, name), other / name)
    with pytest.raises(identity.IdentityMismatch, match="pinned revision"):
        identity.verify_student(str(other), lock, MODEL_ID, REVISION)


def test_the_launcher_names_the_post_trained_snapshot():
    script = open(os.path.join(ROOT, "scripts", "s57_stage2a_train.sh"),
                  encoding="utf-8").read()
    pinned = re.search(r'^student_revision="([0-9a-f]+)"', script,
                       re.MULTILINE)
    assert pinned, "the launcher does not pin a student revision"
    assert pinned.group(1) == REVISION
    default = re.search(r'^student_dir="([^"]+)"', script, re.MULTILINE)
    assert default, "the launcher has no default student_dir"
    assert "Qwen3.5-0.8B/$student_revision" in default.group(1)
    assert "Qwen3.5-0.8B-Base" not in default.group(1)
    assert '--student-dir "$student_dir"' in script
    assert "--student-lock" in script
    assert "--student-revision" in script
    assert "Qwen3.5-0.8B-Base" in script  # the fail-closed Base guard
    assert "--loss-unit token" in script


def test_the_stage2b_launcher_also_pins_the_identity():
    script = open(os.path.join(ROOT, "scripts", "s59_stage2b_train.sh"),
                  encoding="utf-8").read()
    assert '--student-dir "$student_dir"' in script
    assert "--student-lock" in script
    assert "--student-revision" in script
    assert "--loss-unit token" in script
