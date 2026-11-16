"""Fixture launcher that stands in for `scripts/court` in pipeline tests.

This module is *test infrastructure*, never imported by production code. It is a
faithful stand-in for the court in the one dimension the pipeline cares about:
it executes the command it is given (for the two CPU readers it executes the real
script), writes the receipts the next stage verifies, and can be told to fail one
named call once so resumability is testable.

It does not relax any limit: the pipeline still computes and records every
stage's `SELVEDGE_MEM`, and the wrapper it replaces is the only thing that
changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "oracle"))

STEPS = 1024
TOKENS = STEPS * 65536
ITEMS = 40
DOMAINS = {"code": 10, "prose": 10}
KL = {"D0": 1.95, "D": 1.70, "M": 1.94, "X": 1.72, "H": 1.60, "H*": 1.50}
ACCURACY = {"D0": 0.40, "D": 0.45, "M": 0.35, "X": 0.42, "H": 0.40, "H*": 0.60}
ACTIVE = {"D0": 596049920, "D": 772210688, "M": 598153216, "X": 772440064,
          "H": 774543360, "H*": 774543360}
ARMS = ("D", "M", "X", "H", "H*")
CATEGORIES = ("api-migration", "concurrency", "dependency-compat", "failing-test",
              "lifetime-borrow", "repo-scale", "semantic-repair", "trait-impl")
CAPABILITY_PASSES = {"D": 3, "M": 2, "X": 3, "H": 3, "H*": 7}


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def dump(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
        documents = [{"doc_id": f"{name}-{index}", "kl": kl + (index - 5) * 0.002,
                      "ppl": 10.0, "top1": 0.5, "windows": 1}
                     for index in range(count)]
        domains[name] = {"documents": documents, "window_count": count}
    return {"domains": domains,
            "mixed": {"kl": {"mean": kl, "ci_low": kl - 0.01,
                             "ci_high": kl + 0.01}}}


def eval_receipt(arm: str, digest: str, config_sha: str, admission_sha: str,
                 mask: str | None) -> dict:
    return {
        "schema": "selvedge/t2-eval/v1",
        "arm": arm,
        "stamp": "20261116T000000Z",
        "revision": "b" * 40,
        "vram_peak_gib": 9.5,
        "model": {"checkpoint": f"results/artifacts/train/t2/"
                                f"{'Hstar' if arm == 'H*' else arm}/"
                                f"step-{STEPS:06d}.pt",
                  "checkpoint_sha256": digest,
                  "config": f"results/train/admission-configs/t2/{arm}.json",
                  "config_sha256": config_sha,
                  "admission": f"results/train/admission/t2/{arm}.json",
                  "admission_sha256": admission_sha},
        "instrument": {"dispatch_mask_order": mask},
        "results": {
            "humaneval": task_block(ACCURACY[arm], "passed"),
            "gsm8k": task_block(ACCURACY[arm], "correct"),
            "truthfulqa": task_block(ACCURACY[arm], "correct"),
            "distributional": distributional(KL[arm]),
        },
    }


def capability_result(arm: str, root: str) -> dict:
    prereg_path = os.path.join(root, "docs/capability-preregistration-v3.json")
    prereg = json.load(open(prereg_path, encoding="utf-8"))
    amendment_path = os.path.join(root, "results/t2/"
                                        "PROVENANCE-AMENDMENT-X-and-H.json")
    rows = []
    by_category = {}
    for category in CATEGORIES:
        passes = CAPABILITY_PASSES[arm]
        values = [1.0 if index < passes else 0.0 for index in range(9)]
        mean = sum(values) / len(values)
        by_category[category] = {"mean": mean, "ci_low": mean, "ci_high": mean,
                                 "n": 9, "minimum_required": 8,
                                 "minimum_met": True}
        rows += [{"id": f"{category}-{index:02d}", "category": category,
                  "surface": None, "status": "pass" if value else "test_failed",
                  "value": value} for index, value in enumerate(values)]
    composite = sum(entry["mean"] for entry in by_category.values()) / len(CATEGORIES)
    provenance = {
        "arm": arm,
        "dispatch_mask_order": "legacy" if arm in ("X", "H", "H*") else None,
        "checkpoint_verification": {"checked": True},
    }
    if arm in ("X", "H"):
        provenance["provenance_amendment"] = {
            "used": True, "sha256": sha256(amendment_path),
            "inferred_mask_order": "legacy"}
    return {
        "schema": "selvedge/capability-result/v1",
        "stage": "check",
        "stamp": "20261116T000000Z",
        "revision": "c" * 40,
        "suite": {"suite_sha256": prereg["suite_sha256"],
                  "category_minimum_items": 8},
        "generations": {
            "adapter": {"name": f"t2-{arm}", "kind": "t2-arm", "tools": False,
                        "provenance": provenance},
            "decoding": {"feedback": "none", "seed": 20261115},
        },
        "preregistration": {"path": "docs/capability-preregistration-v3.json",
                            "sha256": sha256(prereg_path),
                            "primary_relative_threshold": 0.1},
        "scores": {
            "primary_composite": composite,
            "primary_ci": {"mean": composite, "ci_low": composite,
                           "ci_high": composite, "n": 72, "resamples": 10000,
                           "seed": 20261115},
            "by_category": by_category,
            "category_minimum_items": 8,
            "category_minimum_met": True,
            "categories": list(CATEGORIES),
            "counts": {"pass": sum(CAPABILITY_PASSES[arm] for _ in CATEGORIES)},
            "items": 72,
            "trio": {"rust": {"mean": 1.0}, "numeric-graph": {"mean": 1.0},
                     "logic": {"mean": 1.0}},
            "trio_all_surfaces": True,
        },
        "items": rows,
    }


def build_repo(root: str) -> dict:
    """A miniature repository with the real registration and suite, and fake arms."""
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    shutil.copy(os.path.join(REPO, "docs/capability-preregistration-v3.json"),
                os.path.join(root, "docs/capability-preregistration-v3.json"))
    shutil.copy(os.path.join(REPO, "docs/perf-numbers.json"),
                os.path.join(root, "docs/perf-numbers.json"))
    shutil.copytree(os.path.join(REPO, "fixtures/capability"),
                    os.path.join(root, "fixtures/capability"), dirs_exist_ok=True)
    os.makedirs(os.path.join(root, "results/t2"), exist_ok=True)

    for arm in ARMS:
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
        dispatch = "legacy" if arm in ("D", "M", "H*") else None
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
        if arm != "H*":
            dump(os.path.join(root, "results/eval", f"t2-{arm}.json"),
                 eval_receipt(arm, digest, sha256(config), sha256(admission),
                              dispatch))

    # The H* training artifacts the verify stage hashes.
    hstar = os.path.join(root, "results/artifacts/train/t2/Hstar")
    checkpoint = os.path.join(hstar, f"step-{STEPS:06d}.pt")
    labels = os.path.join(hstar, "utility-labels.pt")
    with open(labels, "wb") as handle:
        handle.write(b"labels")
    dump(labels + ".sha256", {"sha256": sha256(labels), "bytes": 6})
    dump(os.path.join(hstar, "summary.json"),
         {"steps": STEPS, "tokens": TOKENS, "tokens_per_second": 4975.0,
          "vram_peak_gib": 10.81, "finished_utc": "2026-11-16T09:00:00Z"})
    dump(os.path.join(hstar, "run-meta.json"), {"seed": 1001})

    d0 = eval_receipt("D", "frozen", "0" * 64, "0" * 64, None)
    d0["arm"] = "D"
    d0["stamp"] = "20261115T000000Z"
    d0["results"]["humaneval"] = task_block(ACCURACY["D0"], "passed")
    d0["results"]["gsm8k"] = task_block(ACCURACY["D0"], "correct")
    d0["results"]["truthfulqa"] = task_block(ACCURACY["D0"], "correct")
    d0["results"]["distributional"] = distributional(KL["D0"])
    dump(os.path.join(root, "results/eval/t2-D0.json"), d0)

    amendment = os.path.join(root, "results/t2/PROVENANCE-AMENDMENT-X-and-H.json")
    dump(amendment, {
        "schema": "selvedge/t2-provenance-amendment/v1",
        "decision": "CERTIFIED",
        "subject_arms": ["X", "H"],
        "inferred_mask_order": "legacy",
        "adapter_consumption": {"binding_sha256": {
            "results/t2/X/run.json": sha256(os.path.join(root,
                                                         "results/t2/X/run.json")),
            "results/t2/H/run.json": sha256(os.path.join(root,
                                                         "results/t2/H/run.json")),
        }},
    })

    # What the GPU stages would produce, pre-built.
    fixtures = os.path.join(root, ".fake-court/fixtures")
    os.makedirs(fixtures, exist_ok=True)
    hstar_ckpt = os.path.join(hstar, f"step-{STEPS:06d}.pt")
    for arm in ARMS:
        config = os.path.join(root, "results/train/admission-configs/t2",
                              f"{arm}.json")
        admission = os.path.join(root, "results/train/admission/t2",
                                 f"{arm}.json")
        record = eval_receipt(arm, sha256(os.path.join(
            root, "results/artifacts/train/t2",
            "Hstar" if arm == "H*" else arm, f"step-{STEPS:06d}.pt")),
            sha256(config), sha256(admission), "legacy")
        dump(os.path.join(fixtures, f"eval-{arm}-legacy.json"), record)
        pair = json.loads(json.dumps(record))
        pair["instrument"]["dispatch_mask_order"] = "pair"
        dump(os.path.join(fixtures, f"eval-{arm}-pair.json"), pair)
    del hstar_ckpt
    dump(os.path.join(root, "host-probe.json"),
         {"memory": {"available_gib": 90.0, "total_gib": 125.0,
                     "source": "fixture"},
          "gpu": {"available": True, "used_mib": 0, "total_mib": 16376,
                  "compute_apps": []},
          "containers": []})

    wrapper = os.path.join(root, "fake-court")
    with open(wrapper, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\nexec " + sys.executable + " "
                     + os.path.abspath(__file__) + ' "$@"\n')
    os.chmod(wrapper, 0o755)
    return {"root": root, "court": wrapper, "amendment": amendment}


# ---------------------------------------------------------------------------
# the launcher


def _parse(argv: list[str]) -> tuple[str, list[str]]:
    tail = argv[argv.index("--") + 1:] if "--" in argv else argv
    for index, part in enumerate(tail):
        if part.endswith(".py"):
            return part, tail[index + 1:]
    raise SystemExit(f"fake-court: no script in {argv!r}")


def _option(args: list[str], name: str) -> str | None:
    if name in args:
        return args[args.index(name) + 1]
    return None


def _classify(script: str, args: list[str]) -> tuple[str, str]:
    base = os.path.basename(script)
    if base == "baseline.py":
        return "eval", _option(args, "--arm")
    if base == "instrument_check.py":
        return "instrument", ""
    if base == "capability_court.py":
        subcommand = args[0] if args else "generate"
        if subcommand == "check":
            generations = _option(args, "--generations") or ""
            parts = generations.split("/")
            for name in ("D", "M", "X", "H", "Hstar"):
                if f"t2-{name}" in parts:
                    return "check", "H*" if name == "Hstar" else name
            return "check", None
        return subcommand, _option(args, "--t2-arm")
    if base == "t2_verdict.py":
        return "verdict", ""
    if base == "capability_compare.py":
        return "compare", ""
    return base, ""


def main(argv: list[str]) -> int:
    root = os.environ.get("FAKE_COURT_ROOT")
    if not root:
        raise SystemExit("fake-court: FAKE_COURT_ROOT is unset")
    script, args = _parse(argv)
    kind, arm = _classify(script, args)
    log_dir = os.path.join(root, ".fake-court")
    os.makedirs(log_dir, exist_ok=True)
    token = f"{kind}:{arm}"
    with open(os.path.join(log_dir, "calls.jsonl"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"kind": kind, "arm": arm, "token": token,
                              "argv": argv,
                              "SELVEDGE_MEM": os.environ.get("SELVEDGE_MEM")},
                             sort_keys=True) + "\n")
    fail_once = os.path.join(log_dir, "fail-once")
    if os.path.isfile(fail_once):
        target = open(fail_once, encoding="utf-8").read().strip()
        if target == token:
            os.unlink(fail_once)
            print(f"fake-court: injected failure on {token}", file=sys.stderr)
            return 9

    cwd = os.getcwd()
    if kind == "eval":
        mask = _option(args, "--mask-order") or "legacy"
        out = _option(args, "--out")
        source = os.path.join(log_dir, "fixtures", f"eval-{arm}-{mask}.json")
        payload = json.load(open(source, encoding="utf-8"))
        dump(os.path.join(cwd, out), payload)
        return 0
    if kind == "instrument":
        dump(os.path.join(cwd, "results/train/instrument-20261116T120000Z.json"),
             {"schema": "selvedge/t2-instrument/v1", "students": list(ARMS)})
        return 0
    if kind == "generate":
        out = _option(args, "--out")
        dump(os.path.join(cwd, out), {"schema": "selvedge/capability-generations/v1",
                                      "items": []})
        return 0
    if kind == "check":
        generations = _option(args, "--generations")
        out = _option(args, "--out")
        arm_from_path = None
        parts = generations.split("/")
        for name in ("D", "M", "X", "H", "Hstar"):
            if f"t2-{name}" in parts:
                arm_from_path = "H*" if name == "Hstar" else name
        if arm_from_path is None:
            raise SystemExit(f"fake-court: cannot identify the arm in {generations!r}")
        dump(os.path.join(cwd, out), capability_result(arm_from_path, root))
        return 0
    # The CPU readers run for real against the fixture tree.
    path = script if os.path.isabs(script) else os.path.join(REPO, script)
    return subprocess.run([sys.executable, path, *args], cwd=cwd).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
