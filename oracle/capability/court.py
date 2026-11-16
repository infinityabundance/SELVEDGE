#!/usr/bin/env python3
"""The capability court: generation, checking, scoring, controls and thresholds.

Three levels, one instrument (P12.8):

1. **neural mechanism** — verified task success at matched cost against
   matched dense/MoE/memory controls (`compare`);
2. **physically executable capacity** — success, cold bytes per token and p95
   latency as accessible capacity grows (`capacity_scaling`);
3. **useful local assistant** — verified task success offline plus
   responsiveness (`score` + Court F receipts).

The instrument never pools a research-branch result with a product-branch
result and never averages away a per-category regression: the primary composite
is the equal-weight mean of the seven suite categories, and every category is
reported alongside.

Each result receipt carries the anti-gaming evidence the pre-registration
requires: prompt hashes recomputed, a structural statement that the checker did
not exist at generation time, feedback parity, and tool parity.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

from . import extract, stats, suite as suite_module

SCHEMA_GENERATIONS = "selvedge/capability-generations/v1"
SCHEMA_RESULT = "selvedge/capability-result/v1"
SCHEMA_PREREG = "selvedge/capability-preregistration/v1"

DEFAULT_MAX_NEW_TOKENS = 3072
DEFAULT_STOPS = ("\n```\n\n## ",)
DEFAULT_SEED = 20261115

LEVELS = {
    1: "neural mechanism: verified success at matched cost against controls",
    2: "physically executable capacity: success, cold bytes/token and p95 as "
       "accessible capacity grows",
    3: "useful offline local assistant: verified task success and responsiveness",
}

CONTROL_ROWS = (
    "frozen student",
    "equally trained dense adapter",
    "plain text retrieval",
    "Memory Grafting-style latent baseline (2605.20948)",
    "conventional MoE",
    "naive hybrid",
    "utility-trained joint hybrid",
    "matching library without routing optimisation",
)

#: The ten fields every control row reports (P12.10). A missing field or a null
#: without a reason fails validation; it is never silently dropped.
CONTROL_FIELDS = (
    "training_tokens",
    "optimizer_flops",
    "offline_teacher_construction_cost",
    "total_accessible_capacity",
    "active_unique_params",
    "gpu_residency_bytes",
    "host_residency_bytes",
    "measured_io",
    "task_success",
)

PRIMARY_THRESHOLD = 0.10
SECONDARY_MAX_CATEGORY_REGRESSION = 0.05
CAPACITY_ACTIVE_TOLERANCE = 0.05


class CourtError(RuntimeError):
    """The court refuses to produce a result."""


# ---------------------------------------------------------------------------
# receipts and provenance


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def revision(repo: str) -> str:
    value = os.environ.get("SELVEDGE_REVISION", "")
    if value:
        return value
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# prompts, decoding and few-shot hygiene


def task_prompt(task) -> str:
    return suite_module.render_prompt(task)


def suite_prompt_hashes(collections: dict) -> dict:
    hashes = {}
    for kind in ("suite", "trio"):
        for task in collections[kind]:
            hashes[task.id] = suite_module.prompt_sha256(task)
    return hashes


def _word_shingles(text: str, width: int = 8) -> set:
    words = " ".join(text.lower().split()).split()
    return {" ".join(words[index:index + width])
            for index in range(0, max(0, len(words) - width + 1))}


def few_shot_disjoint(few_shot_texts: list, collections: dict) -> dict:
    """No few-shot example may share an 8-word shingle with a suite item.

    This is the check that a prompt cannot smuggle a suite item (or its tests)
    into the context as an "example".
    """
    item_shingles = {}
    for kind in ("suite", "trio"):
        for task in collections[kind]:
            text = "\n".join([task.metadata.get("prompt_intro", ""),
                              *[task.read_initial(path) for path in task.initial_files],
                              *[task.read_hidden(path) for path in task.hidden_files],
                              *[task.read_reference(path) for path in task.reference_files]])
            item_shingles[task.id] = _word_shingles(text)
    offenders = {}
    for index, text in enumerate(few_shot_texts):
        shared = _word_shingles(text)
        for task_id, item in item_shingles.items():
            overlap = shared & item
            if overlap:
                offenders[f"example-{index}:{task_id}"] = len(overlap)
    return {"offending_pairs": len(offenders), "offenders": offenders,
            "few_shot_examples": len(few_shot_texts)}


# ---------------------------------------------------------------------------
# stage 1: generation


def generate(adapter, suite_root: str, out: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
             stops: tuple = DEFAULT_STOPS, seed: int = DEFAULT_SEED,
             order: str = "sorted", only: list | None = None,
             preregistration: str | None = None,
             few_shot_texts: list | None = None,
             repo: str | None = None) -> dict:
    repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    collections = suite_module.load_suite(suite_root)
    manifest = suite_module.build_manifest(suite_root)
    tasks = collections["suite"] + collections["trio"]
    if only:
        wanted = set(only)
        tasks = [task for task in tasks if task.id in wanted]
        missing = wanted - {task.id for task in tasks}
        if missing:
            raise CourtError(f"unknown task ids: {sorted(missing)}")

    if order == "sorted":
        tasks.sort(key=lambda task: task.id)
    elif order.startswith("rotate:"):
        shift = int(order.split(":", 1)[1]) % max(1, len(tasks))
        tasks = tasks[shift:] + tasks[:shift]
    else:
        raise CourtError(f"unknown prompt order {order!r}")

    hygiene = None
    if few_shot_texts:
        hygiene = few_shot_disjoint(few_shot_texts, collections)
        if hygiene["offending_pairs"]:
            raise CourtError(
                "few-shot examples overlap suite items; refusing to generate "
                f"({hygiene['offenders']})")

    requests = []
    prompt_hashes = {}
    for task in tasks:
        prompt = task_prompt(task)
        if few_shot_texts:
            prompt = "\n\n".join(list(few_shot_texts) + [prompt])
        prompt_hashes[task.id] = sha256_bytes(prompt.encode("utf-8"))
        requests.append(adapters_request(task.id, prompt, max_new_tokens, stops, seed))

    adapter_metadata = adapter.metadata()
    generated = adapter.generate(requests)
    if len(generated) != len(requests):
        raise CourtError("adapter returned a different number of generations")

    prereg_digest = sha256_file(preregistration) if preregistration else None
    receipt = {
        "schema": SCHEMA_GENERATIONS,
        "stage": "generate",
        "stamp": stamp_now(),
        "revision": revision(repo),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "adapter": adapter_metadata,
        "suite": {
            "root": os.path.abspath(suite_root),
            "suite_sha256": manifest["suite_sha256"],
            "counts": manifest["counts"],
            "prompt_hashes": suite_prompt_hashes(collections),
        },
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "stops": list(stops),
            "seed": seed,
            "order": order,
            "greedy": True,
            "feedback": "none",
        },
        "preregistration": {
            "path": preregistration,
            "sha256": prereg_digest,
        },
        "few_shot": hygiene,
        "items": [
            {
                "id": request.item_id,
                "category": next(task.category for task in tasks
                                 if task.id == request.item_id),
                "prompt_sha256": prompt_hashes[request.item_id],
                "output": generation.as_dict(),
            }
            for request, generation in zip(requests, generated)
        ],
    }
    write_json_atomic(out, receipt)
    return receipt


def adapters_request(item_id: str, prompt: str, max_new_tokens: int, stops, seed):
    from .adapters import Request

    return Request(item_id=item_id, prompt=prompt, max_new_tokens=max_new_tokens,
                   stops=tuple(stops), seed=seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# stage 2: checking and scoring


def check(generations: dict, suite_root: str, out: str,
          preregistration: str | None = None, runner=None,
          keep: bool = False, workroot: str | None = None) -> dict:
    from . import checkers

    runner = runner or (lambda task, files: checkers.run_check(
        task, files, workroot or "/work/target/capability-work", keep=keep))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    collections = suite_module.load_suite(suite_root)
    tasks = {task.id: task for kind in ("suite", "trio") for task in collections[kind]}
    manifest = suite_module.build_manifest(suite_root)

    if generations.get("schema") != SCHEMA_GENERATIONS:
        raise CourtError("generations receipt has the wrong schema")
    if generations["suite"]["suite_sha256"] != manifest["suite_sha256"]:
        raise CourtError("the generations receipt was made against a different suite")

    prereg = None
    prereg_problems = []
    if preregistration:
        prereg = load_preregistration(preregistration)
        prereg_problems = verify_preregistration(prereg, manifest)
        recorded = (generations.get("preregistration") or {}).get("sha256")
        if recorded and recorded != sha256_file(preregistration):
            raise CourtError("the pre-registration changed after generation")
        if prereg_problems:
            raise CourtError(
                "the pre-registration does not match this suite or its thresholds: "
                + "; ".join(prereg_problems))

    outputs = {item["id"]: item for item in generations.get("items", [])}
    missing = sorted(set(tasks) - set(outputs))
    if missing:
        raise CourtError(f"generations receipt is missing items: {missing[:5]}")

    rows = []
    per_category: dict[str, list] = {}
    per_surface: dict[str, list] = {}
    counts: dict[str, int] = {}
    prompt_ok = True
    for task_id in sorted(tasks):
        task = tasks[task_id]
        recorded_prompt = outputs[task_id]["prompt_sha256"]
        expected_prompt = sha256_bytes(task_prompt(task).encode("utf-8"))
        if recorded_prompt != expected_prompt:
            prompt_ok = False
        text = outputs[task_id]["output"]["text"]
        extraction = extract.extract_files(
            text, task.entry, task.allowed_paths(), task.manifest_policy)
        if not extraction.ok:
            outcome = {"status": "invalid_solution", "problems": extraction.rejected}
        else:
            outcome = runner(task, extraction.files)
        status = outcome.get("status", "error")
        counts[status] = counts.get(status, 0) + 1
        value = 1.0 if status == "pass" else 0.0
        per_category.setdefault(task.category, []).append(value)
        surface = (task.metadata.get("shared_operation") or {}).get("surface")
        if surface:
            per_surface.setdefault(surface, []).append(value)
        rows.append({
            "id": task_id,
            "category": task.category,
            "surface": surface,
            "status": status,
            "value": value,
            "extraction": extraction.as_dict(),
            "tokens_out": outputs[task_id]["output"].get("tokens_out"),
            "latency_ms": outputs[task_id]["output"].get("latency_ms"),
            "evidence": {key: value for key, value in outcome.items()
                         if key in ("problems", "checker")} | {
                "stderr_sha256": (outcome.get("evidence") or {}).get("stderr_sha256"),
                "stderr_tail": ((outcome.get("evidence") or {}).get("stderr_tail") or "")[-500:],
                "address_space_bytes": ((outcome.get("evidence") or {}).get("limits") or {}
                                        ).get("address_space_bytes"),
                "wall_seconds": ((outcome.get("evidence") or {}).get("limits") or {}
                                 ).get("wall_seconds"),
            },
        })

    suite_categories = sorted({task.category for task in collections["suite"]})
    scored = {category: per_category.get(category, [])
              for category in suite_categories}
    empty_categories = [category for category, values in scored.items() if not values]
    if empty_categories:
        raise CourtError(f"the run did not cover categories {empty_categories}")

    composite = stats.composite(scored)
    composite_ci = stats.stratified_mean(scored)
    trio = {
        surface: stats.bootstrap_mean(values)
        for surface, values in sorted(per_surface.items())
    }
    trio_all = bool(trio) and all(entry["mean"] >= 1.0 for entry in trio.values())

    tool = generations.get("adapter", {}).get("tools")
    result = {
        "schema": SCHEMA_RESULT,
        "stage": "check",
        "stamp": stamp_now(),
        "revision": revision(repo),
        "created_utc": utc_now(),
        "command": " ".join(sys.argv),
        "suite": {
            "root": os.path.abspath(suite_root),
            "suite_sha256": manifest["suite_sha256"],
            "counts": manifest["counts"],
        },
        "generations": {
            "schema": generations.get("schema"),
            "stamp": generations.get("stamp"),
            "revision": generations.get("revision"),
            "adapter": generations.get("adapter"),
            "decoding": generations.get("decoding"),
            "few_shot": generations.get("few_shot"),
        },
        "preregistration": {
            "path": preregistration,
            "sha256": sha256_file(preregistration) if preregistration else None,
            "problems": prereg_problems,
        },
        "anti_gaming": {
            "prompt_hashes_match": prompt_ok,
            "checker_access": "separate-process-after-generation",
            "checker_visible_to_arm": False,
            "tools": tool,
            "tool_mode": "tools" if tool else "no-tools",
            "tool_parity": "enforced at comparison time",
            "feedback": (generations.get("decoding") or {}).get("feedback"),
            "offline_evidence": sandbox_network_isolation(),
            "few_shot_hygiene": generations.get("few_shot"),
        },
        "scores": {
            "primary_composite": composite,
            "primary_ci": composite_ci,
            "by_category": {category: stats.bootstrap_mean(values)
                            for category, values in scored.items()},
            "pooled": stats.bootstrap_mean([value for values in scored.values()
                                            for value in values]),
            "trio": trio,
            "trio_all_surfaces": trio_all,
            "counts": counts,
            "items": sum(counts.values()),
        },
        "items": rows,
    }
    if not prompt_ok:
        raise CourtError("a prompt hash does not match the current fixture; this run is "
                         "not against the registered suite")
    write_json_atomic(out, result)
    return result


# ---------------------------------------------------------------------------
# pre-registration


def registration_template(manifest: dict) -> dict:
    return {
        "schema": SCHEMA_PREREG,
        "registered_utc": utc_now(),
        "suite_sha256": manifest["suite_sha256"],
        "primary": {
            "endpoint": "category-balanced mean verified task success on the 24-task suite",
            "definition": "equal weight per category; the arithmetic mean inside each "
                          "category; three categories are compile-repair and four are "
                          "behavioural repair",
            "relative_threshold": PRIMARY_THRESHOLD,
            "bootstrap": {
                "kind": "paired percentile, items resampled within category, "
                        "categories weighted equally",
                "resamples": stats.RESAMPLES,
                "seed": stats.SEED,
                "level": stats.LEVEL,
            },
            "decision_rule": "improve the composite by at least 10% relative with a "
                             "paired 95% CI lower bound above zero, and regress no "
                             "category by more than 0.05 absolute",
            "secondary_max_category_regression": SECONDARY_MAX_CATEGORY_REGRESSION,
        },
        "trio": {
            "rule": "a transfer claim requires success on every surface "
                    "(rust-lifetime-chain, numeric-dependency, quantifier-chain) with "
                    "the per-surface breakdown reported",
            "requires_all_surfaces": True,
        },
        "capacity_scaling": {
            "criterion": "at approximately fixed active unique parameters "
                         f"(within {int(CAPACITY_ACTIVE_TOLERANCE * 100)}% of the "
                         "smallest rung), fit log(cold bytes per token) against "
                         "log(total accessible capacity); the slope's 95% CI upper "
                         "bound must be below 1",
            "active_tolerance": CAPACITY_ACTIVE_TOLERANCE,
        },
        "no_hidden_teacher": {
            "criterion": "teacher_calls_at_inference == 0 and a replay-court receipt "
                         "(P7.5) is named; an external adapter that cannot expose this "
                         "is recorded as unverified, never as satisfied",
        },
        "physical_pareto": {
            "criterion": "no control row may dominate the arm on success, cold bytes "
                         "per token, p95 latency and residency simultaneously",
        },
        "branches": {
            "research": "1B-class research branch, reported alone",
            "product": "3B-8B quantised local baseline, reported alone",
            "pooling": "never pooled",
        },
        "controls": list(CONTROL_ROWS),
        "planned_controls": {
            "3B-8B quantised local baseline": {
                "status": "planned",
                "reason": "no such checkpoint is present locally; a row is emitted "
                          "with unavailable fields and a reason, never a fabricated "
                          "number",
            },
        },
        "anti_gaming": {
            "prompt_order_invariance": True,
            "few_shot_must_be_disjoint_8_word_shingles": True,
            "checker_not_visible_at_inference": True,
            "tool_parity": True,
        },
    }


def load_preregistration(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != SCHEMA_PREREG:
        raise CourtError(f"{path}: schema must be {SCHEMA_PREREG!r}")
    return payload


def verify_preregistration(prereg: dict, manifest: dict) -> list:
    problems = []
    if prereg.get("suite_sha256") != manifest["suite_sha256"]:
        problems.append("the pre-registration binds a different suite hash")
    if prereg.get("primary", {}).get("relative_threshold") != PRIMARY_THRESHOLD:
        problems.append("the pre-registered primary threshold is not 0.10 relative")
    if (prereg.get("primary", {}).get("bootstrap", {}).get("seed") != stats.SEED
            or prereg.get("primary", {}).get("bootstrap", {}).get("resamples")
            != stats.RESAMPLES):
        problems.append("the pre-registered bootstrap seed or resample count changed")
    return problems


def register(path: str, suite_root: str, refuse_overwrite: bool = True) -> dict:
    if os.path.exists(path) and refuse_overwrite:
        raise CourtError(
            f"{path} already exists; pre-registration is append-only. Superseding it "
            f"requires a new file and a recorded reason")
    manifest = suite_module.build_manifest(suite_root)
    payload = registration_template(manifest)
    write_json_atomic(path, payload)
    return payload


# ---------------------------------------------------------------------------
# comparison, levels and complementarity


def _values_by_category(result: dict) -> dict:
    values: dict[str, dict] = {}
    for row in result["items"]:
        if row.get("surface"):
            continue
        values.setdefault(row["category"], {})[row["id"]] = row["value"]
    return {category: dict(sorted(items.items()))
            for category, items in sorted(values.items())}


def compare(arm: dict, baseline: dict, threshold: float = PRIMARY_THRESHOLD) -> dict:
    """Level 1 comparison: paired, category-stratified, with a decision rule."""
    if arm["suite"]["suite_sha256"] != baseline["suite"]["suite_sha256"]:
        raise CourtError("the two results were measured on different suites")
    arm_adapter = arm["generations"]["adapter"]
    base_adapter = baseline["generations"]["adapter"]
    if bool(arm_adapter.get("tools")) != bool(base_adapter.get("tools")):
        raise CourtError(
            "tool parity violated: a tool-using arm must be compared with tool-using "
            "baselines")
    arm_feedback = (arm["generations"].get("decoding") or {}).get("feedback")
    base_feedback = (baseline["generations"].get("decoding") or {}).get("feedback")
    if arm_feedback != base_feedback:
        raise CourtError(
            f"feedback parity violated: arm used {arm_feedback!r}, baseline used "
            f"{base_feedback!r}")

    arm_values = _values_by_category(arm)
    base_values = _values_by_category(baseline)
    categories = sorted(arm_values)
    if categories != sorted(base_values):
        raise CourtError("the two results cover different categories")
    paired_arm = {category: [arm_values[category][item] for item in arm_values[category]]
                  for category in categories}
    paired_base = {category: [base_values[category][item] for item in base_values[category]]
                   for category in categories}
    if any(sorted(arm_values[c]) != sorted(base_values[c]) for c in categories):
        raise CourtError("the two results do not contain the same item ids")

    diff = stats.paired_category_bootstrap(paired_arm, paired_base)
    relative = diff["relative_difference"]
    passed = (relative is not None and relative >= threshold
              and diff["ci_low"] > 0)
    regressions = {category: value for category, value in
                   diff["per_category_difference"].items()
                   if value < -SECONDARY_MAX_CATEGORY_REGRESSION}
    arm_trio = arm["scores"].get("trio", {})
    base_trio = baseline["scores"].get("trio", {})
    return {
        "schema": "selvedge/capability-comparison/v1",
        "arm": arm_adapter.get("name"),
        "baseline": base_adapter.get("name"),
        "suite_sha256": arm["suite"]["suite_sha256"],
        "level": 1,
        "level_statement": LEVELS[1],
        "primary": {
            "threshold_relative": threshold,
            "difference": diff,
            "threshold_met": passed,
            "category_regressions_beyond_margin": regressions,
        },
        "trio": {
            "arm": arm_trio,
            "baseline": base_trio,
            "arm_all_surfaces": arm["scores"].get("trio_all_surfaces"),
            "baseline_all_surfaces": baseline["scores"].get("trio_all_surfaces"),
        },
        "verdict": ("arm improves the primary composite by at least "
                    f"{int(threshold * 100)}% relative with a CI above zero"
                    if passed else
                    "no pre-registered primary improvement; no win is claimed"),
    }


def complementarity(arm_a: dict, arm_b: dict) -> dict:
    """P8.1 through the verifier: are two capabilities complementary by item?"""
    a_values = _values_by_category(arm_a)
    b_values = _values_by_category(arm_b)
    categories = sorted(a_values)
    if categories != sorted(b_values):
        raise CourtError("the two results cover different categories")
    per_category = {}
    union_by_category = {}
    for category in categories:
        ids = sorted(a_values[category])
        if ids != sorted(b_values[category]):
            raise CourtError("the two results do not contain the same item ids")
        union = [max(a_values[category][item], b_values[category][item]) for item in ids]
        per_category[category] = {
            "a": sum(a_values[category][item] for item in ids) / len(ids),
            "b": sum(b_values[category][item] for item in ids) / len(ids),
            "oracle_union": sum(union) / len(ids),
        }
        union_by_category[category] = union
    composite_a = sum(entry["a"] for entry in per_category.values()) / len(categories)
    composite_b = sum(entry["b"] for entry in per_category.values()) / len(categories)
    composite_union = sum(entry["oracle_union"] for entry in per_category.values()) / len(
        categories)
    return {
        "schema": "selvedge/capability-complementarity/v1",
        "arm_a": arm_a["generations"]["adapter"].get("name"),
        "arm_b": arm_b["generations"]["adapter"].get("name"),
        "per_category": per_category,
        "composite_a": composite_a,
        "composite_b": composite_b,
        "oracle_union": composite_union,
        "headroom_over_best_single": composite_union - max(composite_a, composite_b),
        "deployable": False,
        "note": "the union needs each item's outcome, so it bounds what a selector "
                "could reach; it is not a deployable arm",
    }


def capacity_scaling(rows: list, tolerance: float = CAPACITY_ACTIVE_TOLERANCE,
                     resamples: int = 2000, seed: int = stats.SEED) -> dict:
    """Level 2: cold bytes per token against accessible capacity at fixed active work."""
    usable = [row for row in rows
              if row.get("total_accessible_capacity") and row.get("cold_bytes_per_token")]
    for row in usable:
        if row.get("active_unique_params") is None:
            raise CourtError(
                "every capacity rung must report active_unique_params from the ledger")
    if len(usable) < 2:
        raise CourtError("capacity scaling needs at least two rungs with measured "
                         "capacity and cold bytes per token")
    smallest_active = min(row["active_unique_params"] for row in usable)
    fixed = [row for row in usable
             if abs(row["active_unique_params"] - smallest_active)
             <= tolerance * smallest_active]
    if len(fixed) < 2:
        raise CourtError(
            "fewer than two rungs share approximately the same active unique "
            "parameters; the fixed-active-compute criterion cannot be evaluated")
    xs = [math.log(row["total_accessible_capacity"]) for row in fixed]
    ys = [math.log(row["cold_bytes_per_token"]) for row in fixed]
    slope, intercept = _ols(xs, ys)
    import random
    rng = random.Random(seed)
    draws = []
    degenerate = 0
    for _ in range(resamples):
        indexes = [rng.randrange(len(xs)) for _ in range(len(xs))]
        try:
            sample_slope, _ = _ols([xs[index] for index in indexes],
                                   [ys[index] for index in indexes])
        except CourtError:
            degenerate += 1
            continue
        draws.append(sample_slope)
    if len(draws) < resamples // 2:
        raise CourtError(
            f"{degenerate} of {resamples} bootstrap draws had no slope; the rungs do "
            f"not identify a scaling law")
    draws.sort()
    low = stats.percentile(draws, 0.025)
    high = stats.percentile(draws, 0.975)
    return {
        "rungs": len(fixed),
        "total_rungs_measured": len(usable),
        "active_unique_params": fixed[0]["active_unique_params"],
        "slope": slope,
        "intercept": intercept,
        "ci_low": low,
        "ci_high": high,
        "sublinear": high < 1.0,
        "resamples": resamples,
        "degenerate_resamples": degenerate,
        "seed": seed,
        "method": "ordinary least squares on log-log, residual-free row bootstrap",
    }


def _ols(xs: list, ys: list) -> tuple:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        raise CourtError("capacity values are identical; no slope exists")
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


def pareto_dominates(a: dict, b: dict) -> bool:
    """True when `a` is no worse on every physical axis and better on one."""
    higher_better = ("task_success",)
    lower_better = ("cold_bytes_per_token", "p95_ms", "residency_bytes")
    better = False
    for key in higher_better:
        if a.get(key) is None or b.get(key) is None:
            raise CourtError(f"Pareto comparison needs {key} on both rows")
        if a[key] < b[key]:
            return False
        better = better or a[key] > b[key]
    for key in lower_better:
        if a.get(key) is None or b.get(key) is None:
            raise CourtError(f"Pareto comparison needs {key} on both rows")
        if a[key] > b[key]:
            return False
        better = better or a[key] < b[key]
    return better


def pareto_front(rows: list) -> list:
    front = []
    for index, row in enumerate(rows):
        if not any(pareto_dominates(other, row)
                   for other_index, other in enumerate(rows) if other_index != index):
            front.append(row)
    return front


def level3(result: dict, responsiveness: dict | None) -> dict:
    """Level 3: verified offline success plus measured responsiveness.

    Success comes from the check receipt; responsiveness must come from a named
    Court F receipt (or a summary of one). A missing measurement is a null with a
    reason, never a number, and the level is then reported as incomplete.
    """
    success = (result.get("scores") or {}).get("primary_composite")
    offline = (result.get("anti_gaming") or {}).get("offline_evidence")
    if responsiveness is None:
        return {
            "level": 3,
            "statement": LEVELS[3],
            "verified_success": success,
            "offline_evidence": offline,
            "p95_ms": {"value": None,
                       "reason": "no Court F responsiveness summary was supplied"},
            "complete": False,
        }
    p95 = responsiveness.get("p95_ms")
    source = responsiveness.get("source")
    if p95 is None or not source:
        raise CourtError("level 3 responsiveness needs both p95_ms and a source receipt")
    return {
        "level": 3,
        "statement": LEVELS[3],
        "verified_success": success,
        "offline_evidence": offline,
        "p95_ms": {"value": p95, "source": source},
        "complete": success is not None and offline == "no-default-route",
    }


def sandbox_network_isolation() -> str:
    """Measured at check time; recorded in the result receipt."""
    from . import sandbox

    return sandbox.network_isolation_evidence()


def no_hidden_teacher(adapter_metadata: dict, replay_receipt: str | None) -> dict:
    calls = (adapter_metadata.get("accounting") or {}).get("teacher_calls_at_inference")
    if calls is None:
        return {
            "satisfied": False,
            "status": "unverified",
            "reason": adapter_metadata.get("null_reasons", {}).get(
                "teacher_calls_at_inference",
                "the adapter does not expose teacher calls at inference"),
            "replay_receipt": replay_receipt,
        }
    if calls != 0:
        return {"satisfied": False, "status": "failed",
                "reason": f"teacher_calls_at_inference is {calls}, not 0",
                "replay_receipt": replay_receipt}
    if not replay_receipt:
        return {"satisfied": False, "status": "unverified",
                "reason": "no replay-court receipt (P7.5) is named",
                "replay_receipt": None}
    return {"satisfied": True, "status": "satisfied",
            "reason": "0 teacher calls declared and a replay receipt is named",
            "replay_receipt": replay_receipt}


# ---------------------------------------------------------------------------
# control matrix


def control_row(name: str, accounting: dict, status: str = "measured",
                null_reasons: dict | None = None, receipt: str | None = None) -> dict:
    if name not in CONTROL_ROWS:
        raise CourtError(f"unknown control row {name!r}")
    null_reasons = dict(null_reasons or {})
    reported = {}
    for field in CONTROL_FIELDS:
        value = accounting.get(field)
        if value is None:
            reason = null_reasons.get(field)
            if not reason:
                raise CourtError(
                    f"control row {name!r}: field {field!r} is missing and has no "
                    f"reason; explicit nulls need a reason")
            reported[field] = {"value": None, "reason": reason}
        else:
            reported[field] = {"value": value, "reason": None}
    row = {
        "row": name,
        "status": status,
        "fields": reported,
        "receipt": receipt,
    }
    validate_control_row(row)
    return row


def validate_control_row(row: dict) -> None:
    if row.get("row") not in CONTROL_ROWS:
        raise CourtError(f"unknown control row {row.get('row')!r}")
    if set(row.get("fields", {})) != set(CONTROL_FIELDS):
        raise CourtError(f"row {row['row']!r} does not report exactly the ten fields")
    for field, entry in row["fields"].items():
        if not isinstance(entry, dict) or "value" not in entry or "reason" not in entry:
            raise CourtError(f"row {row['row']!r}: field {field!r} is malformed")
        if entry["value"] is None and not entry["reason"]:
            raise CourtError(
                f"row {row['row']!r}: field {field!r} is null without a reason")
    if row.get("status") not in ("measured", "planned", "unavailable"):
        raise CourtError(f"row {row['row']!r}: unknown status {row.get('status')!r}")


def render_control_table(rows: list) -> str:
    for row in rows:
        validate_control_row(row)
    header = "| row | status | " + " | ".join(CONTROL_FIELDS) + " |"
    divider = "| --- | --- | " + " | ".join("---" for _ in CONTROL_FIELDS) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for field in CONTROL_FIELDS:
            entry = row["fields"][field]
            if entry["value"] is None:
                cells.append(f"null ({entry['reason']})")
            else:
                cells.append(str(entry["value"]))
        lines.append(f"| {row['row']} | {row['status']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prompt_order_invariance(first: dict, second: dict) -> dict:
    """The same items, two prompt orders, must produce the same outcomes."""
    first_rows = {row["id"]: row["status"] for row in first["items"]}
    second_rows = {row["id"]: row["status"] for row in second["items"]}
    if set(first_rows) != set(second_rows):
        raise CourtError("the two runs do not contain the same items")
    differences = {item: (first_rows[item], second_rows[item])
                   for item in first_rows if first_rows[item] != second_rows[item]}
    return {
        "invariant": not differences,
        "differences": differences,
        "n": len(first_rows),
    }
