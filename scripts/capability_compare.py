#!/usr/bin/env python3
"""Compare the T2 arms on the verified-task capability instrument.

    python3 scripts/capability_compare.py \
        --result D=results/capability/runs/t2-D/result.json \
        --result M=results/capability/runs/t2-M/result.json \
        --result X=results/capability/runs/t2-X/result.json \
        --result H=results/capability/runs/t2-H/result.json \
        --result 'H*=results/capability/runs/t2-Hstar/result.json' \
        --preregistration docs/capability-preregistration-v4.json \
        --out results/capability/runs/t2-comparison.json

The registered Level-1 test (`docs/CAPABILITY-COURT.md` §7) is applied by
`oracle/capability/court.compare`: a paired, category-balanced bootstrap at the
pre-registered 10% relative threshold, conjunctive with "no category regresses by
more than 0.05". This script is the gate in front of that comparator, and it
refuses to produce a comparison when:

* an arm's result is missing;
* a result records no pre-registration digest, or one that does not match the
  supplied registration ("unregistered");
* a result's suite hash, recorded threshold or category minimum disagrees with
  the registration or with another arm;
* a **routed** arm (`X`, `H`, `H*`) does not prove its dispatch mask placement:
  `X` and `H` must consume the certified provenance amendment and record its
  digest, and `H*` must record the mask field directly (or carry an amendment).

A refusal is exit 3 and prints the reason. It never prints a partial verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "oracle"))

from capability import court, suite as suite_module  # noqa: E402

SCHEMA = "selvedge/capability-arm-comparison/v1"
ARMS = ("D", "M", "X", "H", "H*")
ROUTED = ("X", "H", "H*")
AMENDMENT_REQUIRED = ("X", "H")
DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
DEFAULT_AMENDMENT = "results/t2/PROVENANCE-AMENDMENT-X-and-H.json"

#: The comparison set. The first two are the conditional-capacity gate; the
#: others are the registered arm-versus-control readings. `M`'s registered
#: comparator is `D0`, which has no capability row in this instrument, so that
#: comparison is reported as not measured rather than silently substituted.
COMPARISONS = (
    ("H*", "H", "conditional-capacity gate (error direction)"),
    ("H*", "D", "conditional-capacity gate (hybrid versus dense)"),
    ("X", "D", "routed specialists versus dense"),
    ("M", "D", "memory versus the dense adapter of larger active size"),
    ("X", "H", "routed specialists versus the naive hybrid"),
    ("H", "D", "naive hybrid versus dense"),
)
GATE = (("H*", "H"), ("H*", "D"))


class Refusal(RuntimeError):
    """No comparison may be emitted, and the reason is named."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, REPO)
    except ValueError:  # pragma: no cover
        return path


def verify_registration(result: dict, arm: str, prereg_path: str, prereg: dict,
                        prereg_digest: str, manifest: dict,
                        expected_categories: tuple) -> dict:
    """Every result must have been produced against this exact registration."""
    recorded = (result.get("preregistration") or {})
    digest = recorded.get("sha256")
    if not digest:
        raise Refusal(
            f"{arm}: the result records no pre-registration digest; an unregistered "
            f"run cannot enter a registered comparison")
    if digest != prereg_digest:
        raise Refusal(
            f"{arm}: the result's pre-registration digest {digest[:16]}… does not "
            f"match {_rel(prereg_path)} ({prereg_digest[:16]}…)")
    threshold = recorded.get("primary_relative_threshold")
    registered_threshold = (prereg.get("primary") or {}).get("relative_threshold")
    endpoint_kind = recorded.get("primary_endpoint_kind") or court.ENDPOINT_RELATIVE
    registered_kind = (prereg.get("primary") or {}).get(
        "endpoint_kind", court.ENDPOINT_RELATIVE)
    if endpoint_kind != registered_kind:
        raise Refusal(
            f"{arm}: the result was scored on a {endpoint_kind!r} endpoint, the "
            f"registration fixes {registered_kind!r}")
    if endpoint_kind == court.ENDPOINT_ABSOLUTE:
        minimum = recorded.get("primary_minimum_absolute_effect")
        registered_minimum = (prereg.get("primary") or {}).get(
            "minimum_absolute_effect")
        if minimum is None:
            raise Refusal(
                f"{arm}: the result records no minimum absolute effect; re-run check()")
        if minimum != registered_minimum:
            raise Refusal(
                f"{arm}: the result was scored against a minimum absolute effect of "
                f"{minimum!r}, the registration fixes {registered_minimum!r}")
        if threshold is not None:
            raise Refusal(
                f"{arm}: an absolute-endpoint result also records a relative "
                f"threshold; the two endpoints are never pooled")
    else:
        if threshold is None:
            raise Refusal(f"{arm}: the result records no threshold; re-run check()")
        if threshold != registered_threshold or threshold != court.PRIMARY_THRESHOLD:
            raise Refusal(
                f"{arm}: the result was scored against threshold {threshold!r}, the "
                f"registration fixes {registered_threshold!r} "
                f"(court default {court.PRIMARY_THRESHOLD!r})")
    suite_hash = (result.get("suite") or {}).get("suite_sha256")
    if suite_hash != manifest["suite_sha256"]:
        raise Refusal(f"{arm}: the result was measured on suite {str(suite_hash)[:12]}…, "
                      f"the fixture is {manifest['suite_sha256'][:12]}…")
    if suite_hash != prereg.get("suite_sha256"):
        raise Refusal(f"{arm}: the registration binds a different suite")
    scores = result.get("scores") or {}
    if not scores.get("category_minimum_met"):
        raise Refusal(f"{arm}: the result does not meet the registered per-category "
                      f"minimum")
    expected = expected_categories
    present = tuple(sorted((scores.get("by_category") or {}).keys()))
    if present != expected:
        raise Refusal(f"{arm}: the result covers categories {present}, the "
                      f"fixture has {expected}")
    return {"path": _rel(recorded.get("path") or ""), "sha256": digest,
            "threshold": threshold, "suite_sha256": suite_hash}


def verify_provenance(result: dict, arm: str, amendment_path: str | None,
                      amendment_cache: dict) -> dict:
    """A routed arm must prove the dispatch mask placement it was scored with."""
    provenance = ((result.get("generations") or {}).get("adapter") or {}).get(
        "provenance") or {}
    verification = provenance.get("checkpoint_verification") or {}
    mask = provenance.get("dispatch_mask_order")
    amendment = provenance.get("provenance_amendment") or {}
    info = {"arm": arm, "dispatch_mask_order": mask,
            "checkpoint_checked": bool(verification.get("checked")),
            "provenance_amendment": None}
    if arm not in ROUTED:
        return info
    if arm in AMENDMENT_REQUIRED:
        if not amendment.get("used") or not amendment.get("sha256"):
            raise Refusal(
                f"{arm}: this routed arm trained before the dispatch-mask field, so a "
                f"certified provenance amendment is required and the result records "
                f"none (generate with --provenance-amendment)")
        if not amendment_path or not os.path.isfile(amendment_path):
            raise Refusal(f"{arm}: the certified provenance amendment "
                          f"{amendment_path!r} is not available to verify the result")
        if amendment_path not in amendment_cache:
            amendment_cache[amendment_path] = load_json(amendment_path)
        document = amendment_cache[amendment_path]
        if sha256_file(amendment_path) != amendment["sha256"]:
            raise Refusal(f"{arm}: the provenance amendment changed after generation")
        if document.get("decision") != "CERTIFIED":
            raise Refusal(f"{arm}: the provenance amendment does not certify the arm")
        if arm not in (document.get("subject_arms") or []):
            raise Refusal(f"{arm}: the provenance amendment does not cover this arm")
        if document.get("inferred_mask_order") != mask:
            raise Refusal(f"{arm}: the amendment infers "
                          f"{document.get('inferred_mask_order')!r}, the result records "
                          f"{mask!r}")
        info["provenance_amendment"] = {
            "path": _rel(amendment_path),
            "sha256": amendment["sha256"],
            "inferred_mask_order": document.get("inferred_mask_order"),
        }
    else:  # H*: trained after the field existed
        if mask not in ("legacy", "pair") and not amendment.get("used"):
            raise Refusal(f"{arm}: the result records no dispatch mask placement and "
                          f"no provenance amendment, so its dispatch semantics are "
                          f"unverified")
        if not verification.get("checked"):
            raise Refusal(f"{arm}: the checkpoint was not verified against its "
                          f"training receipt")
    if info["dispatch_mask_order"] not in ("legacy", "pair"):
        raise Refusal(f"{arm}: unknown dispatch mask placement "
                      f"{info['dispatch_mask_order']!r}")
    return info


def compare(paths: dict, preregistration: str, suite_root: str,
            amendment_path: str | None = None) -> dict:
    missing = [arm for arm in ARMS if arm not in paths]
    if missing:
        raise Refusal("missing capability result(s) for: " + ", ".join(missing))
    prereg = load_json(preregistration)
    prereg_digest = sha256_file(preregistration)
    manifest = suite_module.build_manifest(suite_root)
    problems = court.verify_preregistration(prereg, manifest)
    if problems:
        raise Refusal("the pre-registration does not match this suite: "
                      + "; ".join(problems))
    expected_categories = tuple(sorted(
        {task.category for task in suite_module.load_suite(suite_root)["suite"]}))

    results = {}
    registration = {}
    provenance = {}
    amendment_cache: dict = {}
    for arm in ARMS:
        path = paths[arm]
        if not path or not os.path.isfile(path):
            raise Refusal(f"{arm}: capability result {path!r} does not exist")
        result = load_json(path)
        if result.get("schema") != court.SCHEMA_RESULT:
            raise Refusal(f"{arm}: {path} is not a capability result receipt")
        results[arm] = result
        registration[arm] = verify_registration(
            result, arm, preregistration, prereg, prereg_digest, manifest,
            expected_categories)
        provenance[arm] = verify_provenance(result, arm, amendment_path,
                                            amendment_cache)
        registration[arm]["result_file"] = _rel(path)
        registration[arm]["result_sha256"] = sha256_file(path)

    # Every arm must have been scored on the same suite (registration above pins
    # the fixture; this pins the arms to each other as well).
    suites = {arm: registration[arm]["suite_sha256"] for arm in ARMS}
    if len(set(suites.values())) != 1:
        raise Refusal(f"the arms were scored on different suites: {suites}")

    comparisons = {}
    for left, right, title in COMPARISONS:
        label = f"{left}-{right}"
        payload = court.compare(results[left], results[right],
                                preregistration=prereg)
        payload["title"] = title
        comparisons[label] = payload

    gate = {}
    for left, right in GATE:
        label = f"{left}-{right}"
        gate[label] = bool(comparisons[label]["primary"]["threshold_met"])
    passed = all(gate.values())

    arm_summary = {}
    for arm in ARMS:
        scores = results[arm]["scores"]
        arm_summary[arm] = {
            "composite": scores.get("primary_composite"),
            "composite_ci": scores.get("primary_ci"),
            "categories": {
                category: {
                    "mean": entry.get("mean"),
                    "ci_low": entry.get("ci_low"),
                    "ci_high": entry.get("ci_high"),
                    "n": entry.get("n"),
                    "minimum_required": entry.get("minimum_required"),
                    "minimum_met": entry.get("minimum_met"),
                }
                for category, entry in sorted((scores.get("by_category") or {}).items())
            },
            "counts": scores.get("counts"),
            "items": scores.get("items"),
            "trio": scores.get("trio"),
            "trio_all_surfaces": scores.get("trio_all_surfaces"),
            "receipt": registration[arm]["result_file"],
            "receipt_sha256": registration[arm]["result_sha256"],
        }

    if passed:
        verdict = (
            "the conditional-capacity gate passes on verified task success: H* "
            "improves on both H and D at the registered threshold with no category "
            "regression beyond the margin")
        outcome = "conditional_capacity_supported_on_verified_tasks"
    else:
        failed = [label for label, ok in gate.items() if not ok]
        verdict = (
            "the conditional-capacity gate does not pass: " + ", ".join(failed)
            + " did not clear the conjunctive registered rule; no win is claimed")
        outcome = "conditional_capacity_not_supported_on_verified_tasks"

    return {
        "schema": SCHEMA,
        "stamp": court.stamp_now(),
        "created_utc": court.utc_now(),
        "revision": court.revision(REPO),
        "instrument": ("verified task success, category-balanced paired bootstrap, "
                      + ("absolute endpoint with a pre-registered minimum "
                         "(per-category regressions reported and blocking transfer)"
                         if (prereg.get("primary") or {}).get(
                             "endpoint_kind") == court.ENDPOINT_ABSOLUTE else
                         "conjunctive with the no-regression margin "
                         "(CAPABILITY-COURT §7)")),
        "preregistration": {
            "path": _rel(preregistration),
            "sha256": prereg_digest,
            "endpoint_kind": (prereg.get("primary") or {}).get(
                "endpoint_kind", court.ENDPOINT_RELATIVE),
            "threshold": (prereg.get("primary") or {}).get("relative_threshold"),
            "minimum_absolute_effect": (prereg.get("primary") or {}).get(
                "minimum_absolute_effect"),
            "category_minimum_items": (prereg.get("primary") or {}).get(
                "category_minimum_items"),
            "suite_sha256": prereg.get("suite_sha256"),
        },
        "registration": registration,
        "provenance": provenance,
        "arms": arm_summary,
        "comparisons": comparisons,
        "gate": {"comparisons": gate, "passes": passed},
        "not_measured": [
            {
                "comparison": "M-D0",
                "reason": "K17's registered comparator is the frozen D0, which has no "
                          "verified-task row in this instrument; substituting D would "
                          "be a different estimand, so the comparison is not measured "
                          "here and is recorded as such",
            }
        ],
        "outcome": outcome,
        "verdict": verdict,
        "caveats": [
            "verified task success on the fresh Rust suite is not general coding "
            "ability (CAPABILITY-COURT §8)",
            "the suite is small and its per-category instrument is coarse; the "
            "per-category intervals are reported so that is visible",
            "this comparison changes no T2 threshold and does not replace the "
            "loss- and task-based verdict in results/T2-VERDICT.md",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=None,
                        help="the repository root every relative path resolves "
                             "against (default: the parent of scripts/)")
    parser.add_argument("--result", action="append", default=[],
                        help="ARM=path to a capability result receipt (repeat)")
    parser.add_argument("--preregistration", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--amendment", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    global REPO, DEFAULT_SUITE, DEFAULT_AMENDMENT
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)
        DEFAULT_SUITE = os.path.join(REPO, "fixtures", "capability")
        DEFAULT_AMENDMENT = "results/t2/PROVENANCE-AMENDMENT-X-and-H.json"
    preregistration = args.preregistration or os.path.join(
        REPO, "docs", "capability-preregistration-v4.json")
    suite_root = args.suite or DEFAULT_SUITE
    amendment = args.amendment
    if amendment is None:
        amendment = os.path.join(REPO, DEFAULT_AMENDMENT)

    paths = {}
    for entry in args.result:
        if "=" not in entry:
            raise SystemExit(f"--result expects ARM=path, got {entry!r}")
        arm, path = entry.split("=", 1)
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")
        paths[arm] = path

    try:
        payload = compare(paths, preregistration, suite_root, amendment)
    except Refusal as error:
        print(f"capability_compare: REFUSED — {error}", file=sys.stderr)
        return 3

    write_json(args.out, payload)
    print(f"capability_compare: {payload['outcome']}")
    print(f"  {payload['verdict']}")
    print()
    print(f"{'arm':4} {'composite':>9}  receipt")
    for arm in ARMS:
        row = payload["arms"][arm]
        print(f"{arm:4} {row['composite']:>9.4f}  {row['receipt']}")
    print()
    for label, comparison in payload["comparisons"].items():
        primary = comparison["primary"]
        diff = primary["difference"]
        relative = diff["relative_difference"]
        # A zero baseline makes the relative difference undefined. Printing it is
        # not cosmetic: on this run every arm scored zero, and the reader crashed
        # instead of saying so (S27). Undefined is reported as undefined.
        shown = ("undefined (baseline is zero)" if relative is None
                 else f"{relative:+.4f}")
        print(f"{label:6} relative {shown} "
              f"[{diff['ci_low']:+.4f},{diff['ci_high']:+.4f}] "
              f"improvement={primary['improvement_met']} "
              f"-> {'PASS' if primary['threshold_met'] else 'FAIL'}")
    print()
    for entry in payload["not_measured"]:
        print(f"not measured: {entry['comparison']} — {entry['reason']}")
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
