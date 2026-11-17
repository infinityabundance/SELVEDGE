#!/usr/bin/env python3
"""S44: the complete stage-2 arm table, generated from the receipts.

Every arm that has an anchored receipt under `results/stage2/eval/<arm>/eval.json`
is rendered here together with

* its per-task anchored score and the harness's own standard error,
* the knowledge and math family means on the corrected total-cap basis
  (`docs/ws19-anchored-subset-amendment-preregistration-v2.json`: MMLU is all
  14,042 items, not a 2,000-item subset),
* its held-out next-token CE and teacher KL from
  `results/stage2/s41-heldout-comparison.json`, with the paired per-chunk delta
  against base,
* its active-per-token and accessible added parameter counts and the tokens it
  was trained on,
* **paired per-item** comparisons against `base` and against `adapter-2m`:
  the raw lm-evaluation-harness sample files are re-read, paired by
  `(leaf, doc_hash)`, and the difference in task accuracy is bootstrapped
  (percentile 95% CI) with the discordant-pair counts and a McNemar test. An arm
  whose raw samples are absent is marked `unpaired` rather than given a
  pseudo-replicate.

No number here is typed by hand: the two regions this script owns
(`report-s44-arm-table`, `report-s44-answers` in `results/REPORT.md`) are
declared under `external_regions` in `docs/perf-numbers.json`, and this script
is the only writer.

Usage
-----
    scripts/court python scripts/s44_arm_table.py --write            # regenerate
    scripts/court python scripts/s44_arm_table.py --check            # fail closed
    scripts/court python scripts/s44_arm_table.py --verify-items     # re-read raw samples
    scripts/court python scripts/s44_arm_table.py --self-test        # CPU unit checks

The MMLU raw files are about 200 MB of JSON each; a whole-file `json.load` needs
about 11.7 GiB of resident memory (measured), which is not a workload-sized
limit to request for reading four integers per item. The extractor below parses
one sample object at a time with `JSONDecoder.raw_decode` over a single buffered
string, so the peak is the file text plus one sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = "results/stage2/eval"
HELDOUT = "results/stage2/s41-heldout-comparison.json"
REPORT = "results/REPORT.md"
RECEIPT_DIR = "results/stage2"

SCHEMA = "selvedge/s44-arm-table/v1"
TABLE_SENTINEL = "report-s44-arm-table"
ANSWERS_SENTINEL = "report-s44-answers"

#: Arm display order and the run metadata each arm's receipt does not carry.
ARM_ORDER = [
    "base",
    "memory-ungrafted",
    "memory-grafted",
    "memory-2m",
    "adapter-2m",
    "adapter-20m",
    "full-20m-math",
]
ARM_LABEL = {
    "base": "`base` (frozen Qwen3.5-0.8B-Base)",
    "memory-ungrafted": "`memory-ungrafted` (init, no training)",
    "memory-grafted": "`memory-grafted` (S40 reconstruction init)",
    "memory-2m": "`memory-2m` (2M CE+KL)",
    "adapter-2m": "`adapter-2m` (2M CE+KL)",
    "adapter-20m": "`adapter-20m` (20M CE+KL)",
    "full-20m-math": "`full-20m-math` (20M, backbone, math stream)",
}

#: task -> (raw leaf test, sample field, filter). `primary` is the field the
#: eval receipt scores; `secondary` is reported but never used for a verdict.
TASKS = {
    "mmlu": {"family": "knowledge", "primary": ("acc", None), "leaf_prefix": "mmlu_"},
    "mmlu_pro": {"family": "knowledge", "primary": ("exact_match", "custom-extract"),
                 "leaf_prefix": "mmlu_pro_"},
    "gsm8k": {"family": "math", "primary": ("exact_match", "strict-match"),
              "secondary": ("exact_match", "flexible-extract")},
    "minerva_math500": {"family": "math", "primary": ("math_verify", None),
                        "secondary": ("exact_match", None)},
}
FAMILIES = {
    "knowledge": ["mmlu", "mmlu_pro"],
    "math": ["gsm8k", "minerva_math500"],
}
TASK_ORDER = ["mmlu", "mmlu_pro", "gsm8k", "minerva_math500"]

REFERENCES = ["base", "adapter-2m"]
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 44


# ---------------------------------------------------------------------------
# small helpers


def git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO,
                          check=True, capture_output=True, text=True).stdout.strip()


def revision() -> str:
    return git("rev-parse", "HEAD")



def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: str) -> str:
    return os.path.relpath(path, REPO)


# ---------------------------------------------------------------------------
# raw sample extraction (streaming; one sample object at a time)

_KEEP = ("doc_hash", "doc_id", "acc", "exact_match", "math_verify", "filter")


def iter_samples(path: str):
    """Yield `(leaf, sample)` for every sample in an lm-eval raw file.

    The file is read once into a string; samples are decoded one at a time so
    resident memory is the file text plus one decoded sample, not the object
    graph `json.load` would build. The top-level object is walked key by key:
    every key before `samples` (notably `config`, whose nested `configs` also
    contains a `samples: null` field) is skipped with `raw_decode`, and the
    `samples` object is walked leaf by leaf without ever being materialised.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    decoder = json.JSONDecoder()
    length = len(text)
    cursor = 0
    while text[cursor] in " \t\r\n":
        cursor += 1
    if text[cursor] != "{":
        raise ValueError(f"{path}: not a JSON object")
    cursor += 1
    while True:
        while cursor < length and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= length or text[cursor] == "}":
            return
        if text[cursor] != '"':
            raise ValueError(f"{path}: expected a key at {cursor}")
        key, cursor = decoder.raw_decode(text, cursor)
        while text[cursor] in " \t\r\n":
            cursor += 1
        if text[cursor] != ":":
            raise ValueError(f"{path}: expected ':' after key {key!r}")
        cursor += 1
        while text[cursor] in " \t\r\n":
            cursor += 1
        if key != "samples":
            _value, cursor = decoder.raw_decode(text, cursor)
            continue
        if text[cursor] != "{":
            raise ValueError(f"{path}: samples is not an object")
        cursor += 1
        while True:
            while cursor < length and text[cursor] in " \t\r\n,":
                cursor += 1
            if cursor >= length or text[cursor] == "}":
                cursor += 1
                break
            leaf, cursor = decoder.raw_decode(text, cursor)
            while text[cursor] in " \t\r\n":
                cursor += 1
            if text[cursor] != ":":
                raise ValueError(f"{path}: expected ':' after leaf {leaf!r}")
            cursor += 1
            while text[cursor] in " \t\r\n":
                cursor += 1
            if text[cursor] != "[":
                # `samples.config_source` is a scalar beside the leaf arrays.
                _value, cursor = decoder.raw_decode(text, cursor)
                continue
            cursor += 1
            while True:
                while cursor < length and text[cursor] in " \t\r\n,":
                    cursor += 1
                if text[cursor] == "]":
                    cursor += 1
                    break
                sample, cursor = decoder.raw_decode(text, cursor)
                yield leaf, {name: sample[name] for name in _KEEP if name in sample}


def item_vector(path: str, task: str, metric: str, flt: str | None):
    """Per-item correctness keyed by `(leaf, doc_id)`.

    Returns `(keys, values, doc_hashes)` where `keys` is the ordered list of
    `(leaf, doc_id)` identities, `values` is a float array, and `doc_hashes` is
    the parallel list of the harness's own content hashes. The identity lists
    are what pair two arms; the content hashes are the independent check that
    both arms scored the same items in the same order (identical reports reuse
    the same doc-hash sequence). `doc_hash` alone is not an identity: identical
    items inside a subject share it (27 of the 14,042 MMLU items do), so a
    doc-hash-keyed dictionary would silently drop them and mis-state the score.
    """
    rows: dict[tuple[str, int], float] = {}
    hashes: dict[tuple[str, int], str] = {}
    prefix = TASKS[task].get("leaf_prefix")
    for leaf, sample in iter_samples(path):
        if prefix is not None and not leaf.startswith(prefix):
            continue
        if prefix is None and leaf != task:
            continue
        if flt is not None and sample.get("filter") != flt:
            continue
        if metric not in sample:
            raise KeyError(f"{path}: leaf {leaf} has no {metric!r}")
        identity = (leaf, int(sample["doc_id"]))
        value = float(sample[metric])
        if identity in rows:
            raise ValueError(f"{path}: duplicate identity {identity}")
        rows[identity] = value
        hashes[identity] = sample.get("doc_hash")
    if not rows:
        raise ValueError(f"{path}: no samples matched task {task} metric {metric}")
    keys = list(rows)
    values = np.array([rows[key] for key in keys], dtype=np.float64)
    return keys, values, [hashes[key] for key in keys]


def raw_path(arm: str, task: str) -> str | None:
    import glob
    pattern = os.path.join(REPO, EVAL_DIR, arm, "raw", f"*-{task}.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# paired statistics


def bootstrap_delta_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference `mean(a) - mean(b)` with a percentile bootstrap CI.

    The resample indices are shared by both arms (paired), which is what the
    CI is for: the sampling error of the item set, not of two independent sets.
    """
    n = int(a.size)
    if b.size != n:
        raise ValueError("paired bootstrap needs equal item counts")
    difference = a - b
    observed = float(difference.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 500
    filled = 0
    while filled < BOOTSTRAP_RESAMPLES:
        take = min(chunk, BOOTSTRAP_RESAMPLES - filled)
        index = rng.integers(0, n, size=(take, n))
        draws[filled:filled + take] = difference[index].mean(axis=1)
        filled += take
    low, high = np.percentile(draws, [2.5, 97.5])
    arm_only = int(np.sum((a > 0.5) & (b < 0.5)))
    base_only = int(np.sum((a < 0.5) & (b > 0.5)))
    discordant = arm_only + base_only
    if discordant:
        z = (abs(arm_only - base_only) - 1.0) / math.sqrt(discordant)
        p_value = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    else:
        p_value = 1.0
    return {
        "n": n,
        "delta": observed,
        "ci95": [float(low), float(high)],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "item (paired by leaf, doc_id and doc_hash)",
        "discordant_reference_correct_only": base_only,
        "discordant_arm_correct_only": arm_only,
        "mcnemar_normal_p_two_sided": p_value,
        "paired": True,
    }


# ---------------------------------------------------------------------------
# receipts


def harvest_arms() -> tuple[dict, list[str]]:
    """Every arm with an anchored eval receipt, plus the arms that are missing.

    The arm identity is the receipt *directory*, not the payload's `arm` field:
    the `probe-*` directories carry `arm = base` or `arm = memory` and would
    otherwise overwrite the real arm's scores. Probe directories are excluded.
    """
    import glob
    arms: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(REPO, EVAL_DIR, "*", "eval.json"))):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("probe-"):
            continue
        payload = read_json(path)
        payload.setdefault("arm", name)
        arms[name] = payload
    missing = [arm for arm in ARM_ORDER if arm not in arms]
    return arms, missing


def heldout_by_arm() -> dict:
    path = os.path.join(REPO, HELDOUT)
    if not os.path.exists(path):
        return {}
    payload = read_json(path)
    return {"path": rel(path), "sha256": sha256_file(path),
            "stamp": payload.get("stamp"), "revision": payload.get("revision"),
            "arms": payload.get("arms", {})}


def run_tokens(arm: str, payload: dict) -> int | None:
    """Tokens the arm was actually trained on (realised, not registered).

    An untrained init arm and base are 0. A trained arm reports the checkpoint's
    `checkpoint_tokens_seen` when its eval receipt carries it; otherwise the run
    manifest's `steps * tokens_per_step` (976 x 2048 = 1,998,848 for the 2M arms,
    which is the registered 2,000,000 minus the last partial step).
    """
    report = payload.get("arm_report") or {}
    if report.get("checkpoint_tokens_seen") is not None:
        return int(report["checkpoint_tokens_seen"])
    if arm in ("base", "memory-ungrafted", "memory-grafted"):
        return 0
    run_dir = arm.replace("adapter-20m", "control-20m")
    manifest = os.path.join(REPO, "results/stage2/runs", run_dir, "manifest.json")
    if os.path.exists(manifest):
        values = read_json(manifest)
        if values.get("steps") and values.get("tokens_per_step"):
            return int(values["steps"]) * int(values["tokens_per_step"])
        if values.get("tokens") is not None:
            return int(values["tokens"])
    return None


def task_scores(payload: dict) -> dict:
    return {row["task"]: {"score": row["score"], "stderr": row.get("stderr"),
                          "primary_metric": row.get("primary_metric"),
                          "n": (row.get("n_samples") or {}).get("sample_len"),
                          "wall_clock_s": row.get("wall_clock_s")}
            for row in payload.get("tasks", [])}


def family_means(scores: dict) -> dict:
    """The `oracle/frontier/compare.py` arithmetic, restated here.

    Family mean is the unweighted mean of member task scores; the family
    standard error is `sqrt(sum(se_i^2)) / n`, labelled independent because the
    two tasks are different item sets.
    """
    out = {}
    for family, members in FAMILIES.items():
        present = [scores[name] for name in members if name in scores]
        if len(present) != len(members):
            continue
        mean = sum(row["score"] for row in present) / len(present)
        se = math.sqrt(sum((row["stderr"] or 0.0) ** 2 for row in present)) / len(present)
        out[family] = {"members": members, "mean": mean, "stderr": se}
    return out


def paired_all(arms: dict, jobs: list[tuple[str, str, str]]) -> dict:
    cache: dict[tuple[str, str], tuple[list, np.ndarray]] = {}
    out = {}
    for arm, reference, task in jobs:
        spec = TASKS[task]
        metric, flt = spec["primary"]
        key_arm = (arm, task)
        key_ref = (reference, task)
        try:
            if key_arm not in cache:
                path = raw_path(arm, task)
                if path is None:
                    raise FileNotFoundError(arm)
                cache[key_arm] = item_vector(path, task, metric, flt)
            if key_ref not in cache:
                path = raw_path(reference, task)
                if path is None:
                    raise FileNotFoundError(reference)
                cache[key_ref] = item_vector(path, task, metric, flt)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            out[f"{arm}|{reference}|{task}"] = {
                "paired": False,
                "reason": f"raw samples unavailable or mismatched: {exc}",
            }
            continue
        keys_arm, values_arm, hashes_arm = cache[key_arm]
        keys_ref, values_ref, hashes_ref = cache[key_ref]
        if keys_arm != keys_ref or hashes_arm != hashes_ref:
            out[f"{arm}|{reference}|{task}"] = {
                "paired": False,
                "reason": "the two arms' sample identities or doc hashes differ; "
                          "no pseudo-pairing",
            }
            continue
        stats = bootstrap_delta_paired(values_arm, values_ref)
        stats.update({"task": task, "arm": arm, "reference": reference,
                      "metric": metric, "filter": flt,
                      "arm_mean": float(values_arm.mean()),
                      "reference_mean": float(values_ref.mean())})
        out[f"{arm}|{reference}|{task}"] = stats
    return out


def _cgroup_peak() -> int | None:
    """The container's own measured memory peak, for sizing the next job."""
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return None


def bar_state() -> dict:
    run_dir = os.path.join(REPO, "results/stage2/runs/full-20m-math")
    log_path = os.path.join(run_dir, "log.jsonl")
    state = {"run_dir": rel(run_dir), "landed": False}
    eval_path = os.path.join(REPO, EVAL_DIR, "full-20m-math", "eval.json")
    state["eval_receipt"] = rel(eval_path) if os.path.exists(eval_path) else None
    manifest = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest):
        payload = read_json(manifest)
        state.update({"tokens": payload.get("tokens"), "steps": payload.get("steps"),
                      "manifest": rel(manifest)})
    return state


# ---------------------------------------------------------------------------
# build


def build(verify_items: bool = False) -> dict:
    arms, missing = harvest_arms()
    heldout = heldout_by_arm()
    order = [arm for arm in ARM_ORDER if arm in arms] + sorted(
        set(arms) - set(ARM_ORDER))
    table = {}
    for arm in order:
        payload = arms[arm]
        scores = task_scores(payload)
        report = payload.get("arm_report") or {}
        entry = {
            "kind": report.get("kind"),
            "eval_receipt": f"{EVAL_DIR}/{arm}/eval.json",
            "eval_receipt_sha256": sha256_file(os.path.join(REPO, EVAL_DIR, arm, "eval.json")),
            "eval_stamp": payload.get("stamp"),
            "eval_revision": payload.get("revision"),
            "tasks": scores,
            "families": family_means(scores),
            "tokens_seen": run_tokens(arm, payload),
            "active_per_token_added": report.get("active_per_token") or 0,
            "accessible_added_total": (report.get("accessible_added_total")
                                       or report.get("active_per_token") or 0),
            "checkpoint": report.get("checkpoint"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        }
        ho = (heldout.get("arms") or {}).get(arm)
        if ho:
            entry["heldout"] = {
                "ce": ho["ce"]["mean"], "ce_stderr": ho["ce"].get("stderr"),
                "kl": ho["kl"]["mean"], "kl_stderr": ho["kl"].get("stderr"),
                "delta_ce_vs_base": (ho.get("delta_ce_vs_reference") or {}).get("mean"),
                "delta_ce_stderr": (ho.get("delta_ce_vs_reference") or {}).get("stderr"),
                "delta_kl_vs_base": (ho.get("delta_kl_vs_reference") or {}).get("mean"),
                "delta_kl_stderr": (ho.get("delta_kl_vs_reference") or {}).get("stderr"),
                "chunks": ho.get("chunks"),
            }
        table[arm] = entry

    jobs = [(arm, reference, task)
            for arm in order
            for reference in REFERENCES
            if reference in arms and reference != arm
            for task in TASK_ORDER]
    paired = paired_all(arms, jobs)

    def rank(key: str) -> list[dict]:
        rows = [{"arm": arm, "value": table[arm]["families"][key]["mean"],
                 "stderr": table[arm]["families"][key]["stderr"]}
                for arm in order if key in table[arm]["families"]]
        rows.sort(key=lambda row: -row["value"])
        return rows

    ranking = {
        "math": rank("math"),
        "knowledge": rank("knowledge"),
        "heldout_ce": sorted(
            [{"arm": arm, "ce": table[arm]["heldout"]["ce"]}
             for arm in order if "heldout" in table[arm]],
            key=lambda row: row["ce"]),
    }

    receipt = {
        "schema": SCHEMA,
        "kind": "stage2-arm-table",
        "stamp": os.environ.get("S44_STAMP") or subprocess.run(
            ["date", "-u", "+%Y%m%dT%H%M%SZ"], capture_output=True, text=True,
            check=True).stdout.strip(),
        "created_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True,
            check=True).stdout.strip(),
        "revision": revision(),
        "basis": {
            "mmlu": {"items": 14042, "semantics": "all 57 subjects, every item",
                     "note": "the receipts passed --limit-map mmlu=2000, which the harness applies per "
                             "subject; every subject is below 2,000 items, so the effective basis is the "
                             "full set (docs/ws19-anchored-subset-amendment-preregistration-v2.json)"},
            "mmlu_pro": {"items": 504},
            "gsm8k": {"items": 1319, "primary": "exact_match,strict-match"},
            "minerva_math500": {"items": 500, "primary": "math_verify,none"},
            "family_mean": "unweighted mean of the member task scores; family SE sqrt(sum se^2)/n "
                           "(oracle/frontier/compare.py family_means)",
            "heldout": "2,000,000-token held-out slice, 3,906 non-overlapping 512-token chunks, "
                       "CE and KL(teacher_T || student_T) at T=2.0, paired per chunk",
        },
        "arms": table,
        "arm_order": order,
        "missing_arms": missing,
        "heldout_receipt": {key: value for key, value in heldout.items() if key != "arms"},
        "paired": paired,
        "paired_method": {
            "identity": "(leaf task, doc_id) per item, with the doc_hash sequence required to be "
                        "identical; the arms are paired only then, else the comparison is marked unpaired",
            "statistic": "difference in mean task accuracy with a paired percentile bootstrap CI",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "discordant": "counts of items where exactly one of the two arms is correct",
            "p_value": "two-sided McNemar normal approximation with continuity correction",
        },
        "ranking": ranking,
        "bar": bar_state(),
        "container_memory_peak_bytes": _cgroup_peak(),
    }
    if verify_items:
        receipt["item_verification"] = verify_receipt_items(receipt, arms)
    return receipt


def verify_receipt_items(receipt: dict, arms: dict) -> dict:
    """Re-read every raw file and compare the implied score to the receipt."""
    checks = {}
    for arm, entry in receipt["arms"].items():
        for task in TASK_ORDER:
            recorded = entry["tasks"].get(task)
            if recorded is None:
                continue
            path = raw_path(arm, task)
            if path is None:
                checks[f"{arm}|{task}"] = {"ok": False, "reason": "no raw file"}
                continue
            metric, flt = TASKS[task]["primary"]
            _keys, values, _hashes = item_vector(path, task, metric, flt)
            implied = float(values.mean())
            ok = abs(implied - recorded["score"]) < 1e-9
            checks[f"{arm}|{task}"] = {
                "ok": ok, "raw": rel(path), "raw_sha256": sha256_file(path),
                "items": int(values.size), "implied_score": implied,
                "receipt_score": recorded["score"],
            }
    return checks


# ---------------------------------------------------------------------------
# rendering


def f(value, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def render_table(receipt: dict) -> str:
    lines = [
        "| arm | tokens | added active/token | accessible added | MMLU | MMLU-Pro | GSM8K | MATH-500 | knowledge | math | held-out CE | held-out KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in receipt["arm_order"]:
        entry = receipt["arms"][arm]
        tasks = entry["tasks"]
        life = entry.get("heldout") or {}
        lines.append(
            "| {label} | {tokens} | {active} | {accessible} | {mmlu} | {pro} | {gsm} | {minerva} | "
            "{know} | {math} | {ce} | {kl} |".format(
                label=ARM_LABEL.get(arm, f"`{arm}`"),
                tokens=f"{entry['tokens_seen']:,}" if entry["tokens_seen"] is not None else "—",
                active=f"{entry['active_per_token_added']:,}",
                accessible=f"{entry['accessible_added_total']:,}",
                mmlu=_task_cell(tasks.get("mmlu")),
                pro=_task_cell(tasks.get("mmlu_pro")),
                gsm=_task_cell(tasks.get("gsm8k")),
                minerva=_task_cell(tasks.get("minerva_math500")),
                know=_family_cell(entry["families"].get("knowledge")),
                math=_family_cell(entry["families"].get("math")),
                ce=f(life.get("ce"), ".4f") if life else "—",
                kl=f(life.get("kl"), ".4f") if life else "—",
            )
        )
    bar = receipt["bar"]
    lines.append("")
    if bar.get("eval_receipt") is None:
        lines.append(
            "`full-20m-math` has no anchored receipt yet; the row is absent rather than "
            "estimated. Total-cap basis: MMLU is all "
            f"{receipt['basis']['mmlu']['items']:,} items, MMLU-Pro "
            f"{receipt['basis']['mmlu_pro']['items']:,}, GSM8K "
            f"{receipt['basis']['gsm8k']['items']:,}, MATH-500 "
            f"{receipt['basis']['minerva_math500']['items']:,} "
            "(amendment v2). Cells are `score ± the harness's standard error`; the family "
            "column is the unweighted member mean and its independent combined SE. "
            f"Held-out CE/KL are the 3,906-chunk means from `{HELDOUT}`."
        )
    return "\n".join(lines)


def _task_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    return f"{row['score'] * 100:.2f} ± {(row['stderr'] or 0.0) * 100:.2f}"


def _family_cell(row: dict | None) -> str:
    if row is None:
        return "—"
    return f"**{row['mean'] * 100:.2f}** ± {(row['stderr'] or 0.0) * 100:.2f}"


def render_answers(receipt: dict) -> str:
    arms = receipt["arms"]
    base_families = arms.get("base", {}).get("families", {})
    adapter = arms.get("adapter-2m", {})
    lines = []

    def delta_line(arm: str) -> str | None:
        entry = arms.get(arm)
        if not entry or not base_families:
            return None
        pieces = []
        for family in ("knowledge", "math"):
            row = entry["families"].get(family)
            base = base_families.get(family)
            if row is None or base is None:
                continue
            combined = math.sqrt((row["stderr"] or 0.0) ** 2 + (base["stderr"] or 0.0) ** 2)
            delta = row["mean"] - base["mean"]
            pieces.append(
                f"{family} {100 * delta:+.2f} ± {100 * combined:.2f} points vs base")
        if not pieces:
            return None
        return f"- `{arm}`: " + "; ".join(pieces) + "."

    lines.append("**Family deltas against the frozen base** (percentage points, combined independent SE):")
    lines.append("")
    for arm in receipt["arm_order"]:
        if arm in ("base", "full-20m-math"):
            continue
        line = delta_line(arm)
        if line:
            lines.append(line)
    lines.append("")

    paired = receipt["paired"]

    def pair_line(arm: str, reference: str, task: str) -> str | None:
        row = paired.get(f"{arm}|{reference}|{task}")
        if not row:
            return None
        if not row.get("paired"):
            return f"- `{arm}` vs `{reference}` on `{task}`: **unpaired** ({row['reason']})."
        low, high = row["ci95"]
        verdict = "CI excludes zero" if (low > 0 or high < 0) else "CI includes zero"
        return (f"- `{arm}` vs `{reference}` on `{task}`: {100 * row['delta']:+.2f} points, "
                f"95% CI [{100 * low:+.2f}, {100 * high:+.2f}], "
                f"items {row['discordant_arm_correct_only']} vs "
                f"{row['discordant_reference_correct_only']} discordant, "
                f"McNemar p {row['mcnemar_normal_p_two_sided']:.3g} — {verdict}.")

    for reference in REFERENCES:
        if reference not in arms:
            continue
        lines.append(f"**Paired per-item comparisons against `{reference}`** "
                     "(paired bootstrap by item, seed "
                     f"{receipt['paired_method']['seed']}, "
                     f"{receipt['paired_method']['resamples']} resamples):")
        lines.append("")
        for arm in receipt["arm_order"]:
            if arm == reference:
                continue
            for task in TASK_ORDER:
                line = pair_line(arm, reference, task)
                if line:
                    lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# regions


def _scan(text: str, where: str):
    import re
    begin = re.compile(r"^\s*<!--\s*BEGIN GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    end = re.compile(r"^\s*<!--\s*END GENERATED:\s*([A-Za-z0-9_.:-]+)\s*-->\s*$")
    lines = text.splitlines()
    regions = {}
    open_id = None
    start = None
    for index, line in enumerate(lines):
        match = begin.match(line)
        if match:
            if open_id is not None:
                raise ValueError(f"{where}:{index + 1}: nested BEGIN")
            open_id, start = match.group(1), index
            continue
        match = end.match(line)
        if match:
            if open_id != match.group(1):
                raise ValueError(f"{where}:{index + 1}: mismatched END")
            regions[open_id] = (start, index)
            open_id = None
    if open_id is not None:
        raise ValueError(f"{where}: BEGIN {open_id} never closed")
    return lines, regions


def _body(lines: list[str], span) -> str:
    return "\n".join(lines[span[0] + 1:span[1]])


def _replace(lines: list[str], span, body: str) -> list[str]:
    return lines[:span[0] + 1] + body.splitlines() + lines[span[1]:]


def write_regions(receipt: dict) -> None:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    for sentinel, body in ((TABLE_SENTINEL, render_table(receipt)),
                           (ANSWERS_SENTINEL, render_answers(receipt))):
        if sentinel not in regions:
            raise SystemExit(f"s44: {REPORT} has no BEGIN GENERATED: {sentinel}")
        lines = _replace(lines, regions[sentinel], body)
        lines, regions = _scan("\n".join(lines), REPORT)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def check_regions(receipt: dict) -> int:
    path = os.path.join(REPO, REPORT)
    with open(path, "r", encoding="utf-8") as handle:
        lines, regions = _scan(handle.read(), REPORT)
    problems = []
    for sentinel, renderer in ((TABLE_SENTINEL, render_table),
                               (ANSWERS_SENTINEL, render_answers)):
        if sentinel not in regions:
            problems.append(f"{REPORT}: missing region {sentinel}")
            continue
        if _body(lines, regions[sentinel]) != renderer(receipt):
            problems.append(f"{REPORT}: region {sentinel} is stale for the current receipts")
    for problem in problems:
        print(f"s44: {problem}", file=sys.stderr)
    return 1 if problems else 0


def latest_receipt() -> str:
    import glob
    matches = sorted(glob.glob(os.path.join(REPO, RECEIPT_DIR, "s44-arm-table-*.json")))
    if not matches:
        raise SystemExit("s44: no arm-table receipt exists; run --write first")
    return matches[-1]


# ---------------------------------------------------------------------------
# self-test: the arithmetic that does not need lm-eval files


def self_test() -> int:
    failures = []

    def check(name, condition, detail=""):
        print(f"s44 self-test: {name}: {'ok' if condition else 'FAIL'}{' — ' + detail if detail else ''}")
        if not condition:
            failures.append(name)

    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    stats = bootstrap_delta_paired(a, b)
    check("bootstrap is symmetric in expectation",
          abs((stats["delta"]) + bootstrap_delta_paired(b, a)["delta"]) < 1e-12)
    check("bootstrap reports the paired discordance",
          stats["discordant_arm_correct_only"] == 1
          and stats["discordant_reference_correct_only"] == 1)
    identical = bootstrap_delta_paired(a, a)
    check("identical arms give a zero-width zero delta",
          identical["delta"] == 0.0 and identical["ci95"] == [0.0, 0.0])

    # family arithmetic matches oracle/frontier/compare.py
    scores = {"mmlu": {"score": 0.48, "stderr": 0.0041},
              "mmlu_pro": {"score": 0.26, "stderr": 0.02},
              "gsm8k": {"score": 0.39, "stderr": 0.013},
              "minerva_math500": {"score": 0.25, "stderr": 0.019}}
    families = family_means(scores)
    expected = (0.48 + 0.26) / 2
    check("knowledge family mean", abs(families["knowledge"]["mean"] - expected) < 1e-12,
          f"{families['knowledge']['mean']:.6f}")
    check("knowledge family SE is sqrt(sum se^2)/n",
          abs(families["knowledge"]["stderr"]
              - math.sqrt(0.0041 ** 2 + 0.02 ** 2) / 2) < 1e-12)

    # the extractor must not need the whole object graph
    import tempfile
    payload = {"samples": {"mmlu_x": [
        {"doc_hash": "a", "doc_id": 0, "acc": 1.0},
        {"doc_hash": "b", "doc_id": 1, "acc": 0.0}],
        "other": [{"doc_hash": "z", "doc_id": 0, "acc": 1.0}]},
        "results": {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        temporary = handle.name
    try:
        keys, values, hashes = item_vector(temporary, "mmlu", "acc", None)
        check("extractor pairs by leaf and doc_id",
              keys == [("mmlu_x", 0), ("mmlu_x", 1)])
        check("extractor keeps the metric", list(values) == [1.0, 0.0])
        check("extractor keeps the content hashes", hashes == ["a", "b"])
    finally:
        os.unlink(temporary)

    print(f"s44 self-test: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="recompute from receipts and rewrite the receipt and the regions")
    mode.add_argument("--check", action="store_true",
                      help="fail closed when a region is stale for the receipts")
    mode.add_argument("--verify-items", action="store_true",
                      help="re-read every raw sample file and compare its implied score")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.verify_items:
        receipt = read_json(latest_receipt())
        arms, _missing = harvest_arms()
        checks = verify_receipt_items(receipt, arms)
        bad = {key: value for key, value in checks.items() if not value["ok"]}
        for key, value in sorted(checks.items()):
            print(f"s44 verify: {key}: {value['implied_score']:.6f} vs receipt "
                  f"{value['receipt_score']:.6f} — {'ok' if value['ok'] else 'MISMATCH'}")
        print(f"s44 verify: {len(checks) - len(bad)} of {len(checks)} item sets agree")
        return 0 if not bad else 1

    if args.check:
        receipt = read_json(latest_receipt())
        return check_regions(receipt)

    receipt = build(verify_items=True)
    os.makedirs(os.path.join(REPO, RECEIPT_DIR), exist_ok=True)
    out = os.path.join(REPO, RECEIPT_DIR, f"s44-arm-table-{receipt['stamp']}.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
        handle.write("\n")
    write_regions(receipt)
    print(f"s44: wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
