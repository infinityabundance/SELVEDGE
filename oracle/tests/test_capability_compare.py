"""CPU tests for `scripts/capability_compare.py`.

The result receipts are synthetic but structurally identical to
`oracle/capability/court.check` output, bound to the *real* v3 pre-registration
and suite hash, so the gate is tested against the object it will see in
production. The comparator underneath is `oracle/capability/court.compare`, which
already has its own tests; these cover the registration, provenance and
conjunctive behaviour the comparator must never be allowed to skip.
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
sys.path.insert(0, os.path.join(REPO, "oracle"))

import capability_compare  # noqa: E402
from capability import court, suite as suite_module  # noqa: E402

PREREG = os.path.join(REPO, "docs", "capability-preregistration-v3.json")
SUITE_ROOT = os.path.join(REPO, "fixtures", "capability")
CATEGORIES = ("api-migration", "concurrency", "dependency-compat", "failing-test",
              "lifetime-borrow", "repo-scale", "semantic-repair", "trait-impl")
PER_CATEGORY = 9


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def values(passes: dict) -> dict:
    """`passes` maps category -> number of passing items (out of 9)."""
    return {category: [1.0 if index < passes[category] else 0.0
                       for index in range(PER_CATEGORY)]
            for category in CATEGORIES}


def result_receipt(counts: dict, *, digest: str, suite_hash: str,
                   threshold: float = 0.1, name: str = "arm",
                   provenance: dict | None = None) -> dict:
    values_by_category = values(counts)
    items = []
    by_category = {}
    for category in CATEGORIES:
        rows = values_by_category[category]
        mean = sum(rows) / len(rows)
        by_category[category] = {"mean": mean, "ci_low": mean, "ci_high": mean,
                                 "n": len(rows), "minimum_required": 8,
                                 "minimum_met": True}
        items += [{"id": f"{category}-{index:02d}", "category": category,
                   "surface": None, "status": "pass" if value else "test_failed",
                   "value": value} for index, value in enumerate(rows)]
    composite = sum(entry["mean"] for entry in by_category.values()) / len(CATEGORIES)
    return {
        "schema": "selvedge/capability-result/v1",
        "stage": "check",
        "stamp": "20261116T000000Z",
        "revision": "c" * 40,
        "suite": {"suite_sha256": suite_hash, "category_minimum_items": 8},
        "generations": {
            "adapter": {"name": f"t2-{name}", "kind": "t2-arm", "tools": False,
                        "provenance": provenance or {}},
            "decoding": {"feedback": "none", "seed": 20261115},
        },
        "preregistration": {"path": PREREG, "sha256": digest,
                            "primary_relative_threshold": threshold},
        "scores": {
            "primary_composite": composite,
            "primary_ci": {"mean": composite, "ci_low": composite,
                           "ci_high": composite, "n": 72,
                           "resamples": 10000, "seed": 20261115},
            "by_category": by_category,
            "category_minimum_items": 8,
            "category_minimum_met": True,
            "categories": list(CATEGORIES),
            "counts": {"pass": sum(int(value) for category in CATEGORIES
                                   for value in values_by_category[category]),
                       "test_failed": sum(1 - int(value) for category in CATEGORIES
                                          for value in values_by_category[category])},
            "items": 72,
            "trio": {"rust": {"mean": 1.0}, "numeric-graph": {"mean": 1.0},
                     "logic": {"mean": 1.0}},
            "trio_all_surfaces": True,
        },
        "items": items,
    }


def provenance(arm: str, amendment_digest: str) -> dict:
    payload = {"arm": arm, "dispatch_mask_order": ("legacy" if arm != "D" and
                                                   arm != "M" else None),
               "checkpoint_verification": {"checked": True}}
    if arm in ("X", "H"):
        payload["provenance_amendment"] = {"used": True,
                                           "sha256": amendment_digest,
                                           "inferred_mask_order": "legacy"}
    return payload


BASE = {category: 3 for category in CATEGORIES}
WIN = {category: 7 for category in CATEGORIES}
WORSE = {category: 2 for category in CATEGORIES}


@pytest.fixture()
def environment(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(capability_compare, "REPO", root)
    amendment_path = os.path.join(root, "PROVENANCE-AMENDMENT-X-and-H.json")
    dump(amendment_path, {
        "schema": "selvedge/t2-provenance-amendment/v1",
        "decision": "CERTIFIED",
        "subject_arms": ["X", "H"],
        "inferred_mask_order": "legacy",
    })
    digest = sha256(PREREG)
    suite_hash = suite_module.build_manifest(SUITE_ROOT)["suite_sha256"]
    receipts = {}
    specs = {"D": BASE, "M": WORSE, "X": BASE, "H": BASE, "H*": WIN}
    for arm, spec in specs.items():
        path = os.path.join(root, f"result-{arm}.json")
        dump(path, result_receipt(spec, digest=digest, suite_hash=suite_hash,
                                  name=arm, provenance=provenance(arm, sha256(amendment_path))))
        receipts[arm] = path
    return {"root": root, "paths": receipts, "amendment": amendment_path,
            "digest": digest, "suite_hash": suite_hash}


def run(environment) -> dict:
    return capability_compare.compare(environment["paths"], PREREG, SUITE_ROOT,
                                     environment["amendment"])


def test_genuine_win_passes_the_conjunctive_gate(environment):
    payload = run(environment)
    assert payload["gate"]["passes"] is True
    assert payload["outcome"] == "conditional_capacity_supported_on_verified_tasks"
    comparison = payload["comparisons"]["H*-H"]["primary"]
    assert comparison["improvement_met"] is True
    assert comparison["threshold_met"] is True
    assert comparison["difference"]["relative_difference"] >= 0.1
    for category in CATEGORIES:
        assert comparison["difference"]["per_category_difference"][category] == \
            pytest.approx(4 / 9)


def test_aggregate_win_with_a_category_regression_is_not_a_win(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(capability_compare, "REPO", root)
    amendment_path = os.path.join(root, "amendment.json")
    dump(amendment_path, {"decision": "CERTIFIED", "subject_arms": ["X", "H"],
                          "inferred_mask_order": "legacy"})
    digest = sha256(PREREG)
    suite_hash = suite_module.build_manifest(SUITE_ROOT)["suite_sha256"]
    arm = dict(WIN)
    arm["concurrency"] = 2
    paths = {"D": os.path.join(root, "D.json"), "M": os.path.join(root, "M.json"),
             "X": os.path.join(root, "X.json"), "H": os.path.join(root, "H.json"),
             "H*": os.path.join(root, "H*.json")}
    specs = {"D": BASE, "M": WORSE, "X": BASE, "H": BASE, "H*": arm}
    for name, spec in specs.items():
        dump(paths[name], result_receipt(
            spec, digest=digest, suite_hash=suite_hash, name=name,
            provenance=provenance(name, sha256(amendment_path))))
    payload = capability_compare.compare(paths, PREREG, SUITE_ROOT, amendment_path)
    primary = payload["comparisons"]["H*-H"]["primary"]
    assert primary["improvement_met"] is True
    assert primary["threshold_met"] is False
    assert "concurrency" in primary["category_regressions_beyond_margin"]
    assert payload["gate"]["passes"] is False
    assert "no win is claimed" in payload["verdict"]


def test_refuses_missing_result_for_an_arm(environment):
    environment["paths"].pop("X")
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "missing capability result(s)" in str(error.value)


def test_refuses_unregistered_result(environment):
    receipt = json.load(open(environment["paths"]["H"]))
    receipt["preregistration"]["sha256"] = None
    dump(environment["paths"]["H"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "unregistered" in str(error.value)


def test_refuses_mismatched_preregistration_digest(environment):
    receipt = json.load(open(environment["paths"]["D"]))
    receipt["preregistration"]["sha256"] = "0" * 64
    dump(environment["paths"]["D"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "does not match" in str(error.value)


def test_refuses_missing_provenance_amendment_for_a_routed_arm(environment):
    receipt = json.load(open(environment["paths"]["H"]))
    receipt["generations"]["adapter"]["provenance"].pop("provenance_amendment")
    dump(environment["paths"]["H"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "provenance amendment is required" in str(error.value)


def test_refuses_amendment_that_does_not_cover_the_arm(environment):
    receipt = json.load(open(environment["paths"]["X"]))
    receipt["generations"]["adapter"]["provenance"][
        "provenance_amendment"]["sha256"] = "f" * 64
    dump(environment["paths"]["X"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "changed after generation" in str(error.value)


def test_refuses_hstar_without_a_recorded_mask_or_amendment(environment):
    receipt = json.load(open(environment["paths"]["H*"]))
    receipt["generations"]["adapter"]["provenance"]["dispatch_mask_order"] = None
    dump(environment["paths"]["H*"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "no dispatch mask placement" in str(error.value)


def test_refuses_result_on_a_different_suite(environment):
    receipt = json.load(open(environment["paths"]["M"]))
    receipt["suite"]["suite_sha256"] = "1" * 64
    dump(environment["paths"]["M"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "measured on suite" in str(error.value)


def test_refuses_result_scored_against_another_threshold(environment):
    receipt = json.load(open(environment["paths"]["D"]))
    receipt["preregistration"]["primary_relative_threshold"] = 0.05
    dump(environment["paths"]["D"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "scored against threshold" in str(error.value)


def test_refuses_a_result_without_the_category_minimum(environment):
    receipt = json.load(open(environment["paths"]["H"]))
    receipt["scores"]["category_minimum_met"] = False
    dump(environment["paths"]["H"], receipt)
    with pytest.raises(capability_compare.Refusal) as error:
        run(environment)
    assert "per-category minimum" in str(error.value)


def test_not_measured_records_the_missing_d0_comparator(environment):
    payload = run(environment)
    not_measured = {entry["comparison"] for entry in payload["not_measured"]}
    assert not_measured == {"M-D0"}
    assert payload["arms"]["M"]["composite"] < payload["arms"]["D"]["composite"]


def test_main_returns_three_on_refusal(environment, capsys):
    environment["paths"].pop("D")
    arguments = []
    for arm, path in environment["paths"].items():
        arguments += ["--result", f"{arm}={path}"]
    arguments += ["--preregistration", PREREG,
                  "--out", os.path.join(environment["root"], "comparison.json")]
    assert capability_compare.main(arguments) == 3
    assert "REFUSED" in capsys.readouterr().err
