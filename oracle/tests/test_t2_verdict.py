"""CPU tests for the guarded T2 verdict reader (`scripts/t2_verdict.py`).

The receipts are synthetic and deliberately small: every task has 40 items and
the bootstrap uses 200 resamples, because these tests are about the *gate* and
the decision rule, not about the registered sample size. The fixture tree mirrors
the real layout (`results/t2/<arm>/run.json`, `results/eval/t2-<arm>.json`,
checkpoint sidecars, admission and config files), so the reader exercises the
same paths it will use on the real artifacts.

Every refusal in the reader has a named test here; the positive tests pin the
registered outcome, the K17 failure of a cheaper-but-worse arm, and
determinism.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import t2_verdict  # noqa: E402

STEPS = 1024
TOKENS = STEPS * 65536
ITEMS = 40
DOMAINS = {"code": 10, "prose": 10}
KL = {"D0": 1.95, "D": 1.70, "M": 1.94, "X": 1.72, "H": 1.60, "H*": 1.50}
ACTIVE = {"D0": 596049920, "D": 772210688, "M": 598153216, "X": 772440064,
          "H": 774543360, "H*": 774543360}


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dump(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def items(accuracy: float, key: str) -> list:
    passes = int(round(ITEMS * accuracy))
    return [{"id": f"item-{index:03d}", key: index < passes} for index in range(ITEMS)]


def task_block(accuracy: float, key: str) -> dict:
    rows = items(accuracy, key)
    return {"items": rows,
            "accuracy": sum(1.0 for row in rows if row[key]) / len(rows)}


def distributional(kl: float) -> dict:
    domains = {}
    for name, count in DOMAINS.items():
        documents = []
        for index in range(count):
            jitter = (index - count / 2) * 0.002
            documents.append({"doc_id": f"{name}-{index}", "kl": kl + jitter,
                              "ppl": 10.0 + jitter, "top1": 0.5 + jitter,
                              "windows": 1})
        domains[name] = {"documents": documents, "window_count": count}
    return {"domains": domains,
            "mixed": {"kl": {"mean": kl, "ci_low": kl - 0.01,
                             "ci_high": kl + 0.01, "n": sum(DOMAINS.values())}}}


def build_fixture(root: str, accuracy: dict, *, routed_mask_in_receipt: bool = False,
                  amendment: bool = True) -> dict:
    paths = {"root": root}
    for arm in ("D", "M", "X", "H", "H*"):
        run_name = "Hstar" if arm == "H*" else arm
        config = os.path.join(root, "results/train/admission-configs/t2",
                              f"{arm}.json")
        admission = os.path.join(root, "results/train/admission/t2", f"{arm}.json")
        dump(config, {"arm": arm, "total_accessible": ACTIVE[arm]})
        dump(admission, {"arm": arm, "admitted": True,
                         "active_unique_params": ACTIVE[arm],
                         "total_accessible": ACTIVE[arm]})
        checkpoint = os.path.join(root, "results/artifacts/train/t2", run_name,
                                  f"step-{STEPS:06d}.pt")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        with open(checkpoint, "wb") as handle:
            handle.write(b"checkpoint-" + arm.encode())
        digest = sha256(checkpoint)
        dump(checkpoint + ".sha256", {"sha256": digest, "bytes": 10})
        dispatch = ("legacy" if (arm in ("D", "M", "H*") or routed_mask_in_receipt)
                    else None)
        dump(os.path.join(root, "results/t2", arm, "run.json"), {
            "schema": "selvedge/t2-arm-receipt/v1",
            "arm": arm,
            "seed": 1001,
            "revision": "a" * 40,
            "config": {"path": f"results/train/admission-configs/t2/{arm}.json",
                       "sha256": sha256(config)},
            "admission": {"path": f"results/train/admission/t2/{arm}.json",
                          "active_unique_params": ACTIVE[arm]},
            "checkpoint": {"path": f"results/artifacts/train/t2/{run_name}/"
                                   f"step-{STEPS:06d}.pt",
                           "name": f"step-{STEPS:06d}.pt",
                           "sha256": digest, "sha256_source": "sidecar"},
            "budget": {"steps": STEPS, "tokens_per_step": 65536,
                       "registered_budget_tokens": TOKENS},
            "measured": {"tokens": TOKENS, "tokens_per_second": 5000.0,
                         "vram_peak_gib": 10.5, "wall_seconds": 14000.0,
                         "gpu_hours": 3.9},
            "dispatch_mask_order": dispatch,
        })
        evaluation = {
            "schema": "selvedge/eval-receipt/v1",
            "arm": arm,
            "stamp": "20261116T000000Z",
            "revision": "b" * 40,
            "vram_peak_gib": 9.5,
            "model": {
                "checkpoint": f"results/artifacts/train/t2/{run_name}/"
                              f"step-{STEPS:06d}.pt",
                "checkpoint_sha256": digest,
                "config": f"results/train/admission-configs/t2/{arm}.json",
                "config_sha256": sha256(config),
                "admission": f"results/train/admission/t2/{arm}.json",
                "admission_sha256": sha256(admission),
            },
            "instrument": {"dispatch_mask_order": dispatch},
            "results": {
                "humaneval": task_block(accuracy[arm], "passed"),
                "gsm8k": task_block(accuracy[arm], "correct"),
                "truthfulqa": task_block(accuracy[arm], "correct"),
                "distributional": distributional(KL[arm]),
            },
        }
        dump(os.path.join(root, "results/eval", f"t2-{arm}.json"), evaluation)
        paths[arm] = {"config": config, "admission": admission,
                      "run": os.path.join(root, "results/t2", arm, "run.json"),
                      "eval": os.path.join(root, "results/eval", f"t2-{arm}.json"),
                      "checkpoint": checkpoint}

    d0 = {
        "schema": "selvedge/eval-receipt/v1",
        "arm": "D",
        "stamp": "20261115T000000Z",
        "revision": "b" * 40,
        "vram_peak_gib": 8.9,
        "model": {"checkpoint": "frozen", "checkpoint_sha256": None},
        "results": {
            "humaneval": task_block(accuracy["D0"], "passed"),
            "gsm8k": task_block(accuracy["D0"], "correct"),
            "truthfulqa": task_block(accuracy["D0"], "correct"),
            "distributional": distributional(KL["D0"]),
        },
    }
    dump(os.path.join(root, "results/eval/t2-D0.json"), d0)
    paths["D0"] = os.path.join(root, "results/eval/t2-D0.json")

    amendment_path = os.path.join(root, "results/t2",
                                  "PROVENANCE-AMENDMENT-X-and-H.json")
    if amendment:
        dump(amendment_path, {
            "schema": "selvedge/t2-provenance-amendment/v1",
            "decision": "CERTIFIED",
            "subject_arms": ["X", "H"],
            "inferred_mask_order": "legacy",
            "adapter_consumption": {
                "binding_sha256": {
                    "results/t2/X/run.json": sha256(paths["X"]["run"]),
                    "results/t2/H/run.json": sha256(paths["H"]["run"]),
                },
            },
        })
    paths["amendment"] = amendment_path
    return paths


WIN = {"D0": 0.40, "D": 0.45, "M": 0.35, "X": 0.42, "H": 0.40, "H*": 0.60}


def run_verdict(root: str, paths: dict, extra: list | None = None,
                out: str = "results/t2-verdict.json") -> int:
    argv = [
        "--run-root", "results/t2",
        "--eval-root", "results/eval",
        "--d0", "results/eval/t2-D0.json",
        "--amendment", paths["amendment"],
        "--resamples", "200",
        "--out", out,
        "--doc", "results/T2-VERDICT.md",
        "--no-manifest",
        "--write",
        "--quiet",
        *(extra or []),
    ]
    return t2_verdict.main(argv)


@pytest.fixture()
def fixture(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(t2_verdict, "REPO", root)
    paths = build_fixture(root, WIN)
    return root, paths


def test_the_registered_instrument_schema_is_accepted(fixture):
    """The eval instrument writes `selvedge/eval-receipt/v1`.

    This reader once required a name the instrument never emitted and refused
    every real receipt; the fixture carried the same wrong string, so the test
    agreed with the bug (S27).
    """
    root, paths = fixture
    receipt = json.load(open(paths["D"]["eval"], encoding="utf-8"))
    assert receipt["schema"] == "selvedge/eval-receipt/v1"
    assert run_verdict(root, paths) == 0


def test_an_unknown_evaluation_schema_is_refused(fixture, capsys):
    root, paths = fixture
    receipt = json.load(open(paths["D"]["eval"], encoding="utf-8"))
    receipt["schema"] = "selvedge/nonsense/v1"
    json.dump(receipt, open(paths["D"]["eval"], "w", encoding="utf-8"))
    assert run_verdict(root, paths) == 3
    assert "unknown schema" in capsys.readouterr().err
    assert not os.path.isfile(os.path.join(root, "results/t2-verdict.json"))


def test_a_complete_receipt_set_produces_the_registered_win(fixture):
    root, paths = fixture
    assert run_verdict(root, paths) == 0
    payload = json.load(open(os.path.join(root, "results/t2-verdict.json")))
    assert payload["outcome"] == "hybrid_supported"
    assert payload["kill_conditions"]["K15"]["verdict"] == "PASS"
    assert payload["kill_conditions"]["K16"]["verdict"] == "PASS"
    assert payload["kill_conditions"]["K17"]["verdict"] == "FAIL"
    assert payload["kill_conditions"]["K18"]["verdict"] == "FAIL"
    assert payload["kill_conditions"]["K19"]["verdict"] == "PASS"
    comparison = payload["comparisons"]["H*-H"]
    assert comparison["decisive_statistic"] == "PC"
    assert comparison["pc"]["difference"]["ci_low"] > 0
    for task in ("humaneval", "gsm8k", "truthfulqa"):
        assert comparison["per_task"][task]["ci_low"] > 0
    # The certified amendment is recorded, not merely required.
    assert payload["guards"]["X"]["provenance"]["provenance_amendment"]
    assert payload["guards"]["X"]["provenance"]["resolved_mask_order"] == "legacy"
    assert os.path.isfile(os.path.join(root, "results/T2-VERDICT.md"))


def test_missing_receipt_is_refused_and_writes_no_verdict(fixture, capsys):
    root, paths = fixture
    os.unlink(paths["H*"]["run"])
    assert run_verdict(root, paths) == 3
    assert not os.path.exists(os.path.join(root, "results/t2-verdict.json"))
    assert "required receipt(s) missing" in capsys.readouterr().err


def test_config_hash_mismatch_is_refused(fixture, capsys):
    root, paths = fixture
    with open(paths["X"]["config"], "w", encoding="utf-8") as handle:
        handle.write('{"arm": "X", "total_accessible": 1}\n')
    assert run_verdict(root, paths) == 3
    assert "config digest does not match" in capsys.readouterr().err


def test_a_missing_admission_digest_is_reported_not_invented(fixture):
    """Three receipts predate the instrument recording the admission digest.

    A missing digest is not a mismatch. The verdict accepts the receipt, records
    that the digest was not recorded, and still binds the arm by path, the
    admitted flag and the active-parameter count (S27).
    """
    root, paths = fixture
    receipt = json.load(open(paths["D"]["eval"], encoding="utf-8"))
    receipt["model"].pop("admission_sha256", None)
    json.dump(receipt, open(paths["D"]["eval"], "w", encoding="utf-8"))
    assert run_verdict(root, paths) == 0
    payload = json.load(open(os.path.join(root, "results/t2-verdict.json")))
    admission = payload["guards"]["D"]["admission"]
    assert admission["digest_recorded_by_instrument"] is False
    assert admission["active_unique_params"]


def test_admission_hash_mismatch_is_refused(fixture, capsys):
    root, paths = fixture
    with open(paths["H*"]["admission"], "w", encoding="utf-8") as handle:
        handle.write('{"arm": "H*", "admitted": true, "active_unique_params": 1}\n')
    assert run_verdict(root, paths) == 3
    assert "admission digest does not match" in capsys.readouterr().err


def test_checkpoint_sidecar_mismatch_is_refused(fixture, capsys):
    root, paths = fixture
    with open(paths["D"]["checkpoint"], "ab") as handle:
        handle.write(b"tampered")
    assert run_verdict(root, paths) == 3
    assert "!= sidecar" in capsys.readouterr().err


def test_missing_provenance_amendment_for_a_routed_arm_is_refused(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    root = str(tmp_path)
    monkeypatch.setattr(t2_verdict, "REPO", root)
    paths = build_fixture(root, WIN, amendment=False)
    assert run_verdict(root, paths) == 3
    assert "no provenance amendment was supplied" in capsys.readouterr().err


def test_amendment_that_does_not_bind_the_receipt_is_refused(fixture, capsys):
    root, paths = fixture
    amendment = json.load(open(paths["amendment"]))
    amendment["adapter_consumption"]["binding_sha256"]["results/t2/X/run.json"] = "0" * 64
    dump(paths["amendment"], amendment)
    assert run_verdict(root, paths) == 3
    assert "does not match the provenance amendment's binding" in \
        capsys.readouterr().err


def test_lower_compute_worse_quality_is_a_reported_failure(fixture):
    root, paths = fixture
    assert run_verdict(root, paths) == 0
    payload = json.load(open(os.path.join(root, "results/t2-verdict.json")))
    training = payload["training"]
    assert training["M"]["active_unique_params"] < training["D"]["active_unique_params"]
    assert payload["arms"]["M"]["pc"]["mean"] < payload["arms"]["D"]["pc"]["mean"]
    assert payload["kill_conditions"]["K17"]["verdict"] == "FAIL"
    assert payload["kill_conditions"]["K17"]["clause_1_passes"] is False
    assert "M-D0" in payload["comparisons"]


def test_verdict_is_deterministic_across_runs(fixture):
    root, paths = fixture
    assert run_verdict(root, paths, out="results/one.json") == 0
    assert run_verdict(root, paths, out="results/two.json") == 0
    first = json.load(open(os.path.join(root, "results/one.json")))
    second = json.load(open(os.path.join(root, "results/two.json")))
    assert first["deterministic_body_sha256"] == second["deterministic_body_sha256"]
    assert first["arms"] == second["arms"]
    assert first["comparisons"] == second["comparisons"]


def test_sensitivity_receipt_must_record_the_pair_mask(fixture, capsys):
    root, paths = fixture
    legacy = os.path.join(root, "results/sensitivity-legacy.json")
    dump(legacy, {"arm": "X", "stamp": "s", "revision": "r",
                  "instrument": {"dispatch_mask_order": "legacy"},
                  "model": {"checkpoint_sha256":
                            sha256(paths["X"]["checkpoint"])},
                  "results": {}})
    assert run_verdict(root, paths, extra=["--sensitivity", f"X={legacy}"]) == 3
    assert "registered sensitivity is 'pair'" in capsys.readouterr().err


def test_sensitivity_point_difference_is_reported_without_rethresholding(fixture):
    root, paths = fixture
    pair = os.path.join(root, "results/sensitivity-pair.json")
    base = json.load(open(paths["X"]["eval"]))
    base["instrument"]["dispatch_mask_order"] = "pair"
    base["results"]["humaneval"] = task_block(0.55, "passed")
    dump(pair, base)
    h_pair = os.path.join(root, "results/sensitivity-pair-H.json")
    base_h = json.load(open(paths["H"]["eval"]))
    base_h["arm"] = "H"
    base_h["instrument"]["dispatch_mask_order"] = "pair"
    dump(h_pair, base_h)
    assert run_verdict(root, paths, extra=[
        "--sensitivity", f"X={pair}", "--sensitivity", f"H={h_pair}"]) == 0
    payload = json.load(open(os.path.join(root, "results/t2-verdict.json")))
    assert payload["sensitivity"]["X"]["present"] is True
    assert payload["sensitivity"]["X"]["mask_order"] == "pair"
    assert payload["sensitivity"]["X"]["not_rethresholded"] is True
    assert payload["sensitivity"]["X"]["per_task"]["humaneval"]["pair"] == \
        pytest.approx(0.55)
    assert payload["outcome"] == "hybrid_supported"


def test_registered_thresholds_are_not_moved(fixture):
    root, paths = fixture
    assert run_verdict(root, paths) == 0
    payload = json.load(open(os.path.join(root, "results/t2-verdict.json")))
    assert payload["kill_conditions"]["K15"]["comparison"]["threshold_met"] is True
    assert payload["comparisons"]["H*-H"]["pc"]["margin"] == 0.03
    assert payload["comparisons"]["H*-D"]["pc"]["margin"] == 0.05
    assert payload["one_seed"]["seed_level_ci_claimed"] is False
    assert payload["resamples"] == 200
