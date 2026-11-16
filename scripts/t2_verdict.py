#!/usr/bin/env python3
"""T2 verdict: receipts in, a guarded, deterministic verdict out.

    python3 scripts/t2_verdict.py --write       # results/t2-verdict.json + results/T2-VERDICT.md
    python3 scripts/t2_verdict.py --check       # re-render and compare, write nothing
    python3 scripts/t2_verdict.py               # print the verdict, write nothing

The rule itself is *not* re-implemented here. `oracle/eval/t2_decision.py` is the
registered reader; this script imports its `ArmMetrics`, `decide`, `gap_closed`
and the K15-K19 assembly, so the statistic that decides the pilot has exactly
one implementation. What this script adds is the gate in front of that reader:

* every required receipt must exist, for exactly the registered arm, seed and
  budget;
* every receipt's checkpoint digest must match its own sha256 sidecar;
* every receipt's config digest must match the config file *and* the training
  receipt, and the admitted active-parameter count must match the admission
  record (K10 fails closed);
* a routed arm (`X`, `H`, `H*`) whose evaluation does not carry the P12.1
  dispatch-mask field must be covered by a certified provenance amendment, and
  the amendment's digest is carried into the verdict;
* the corrected-dispatch sensitivity (`mask_order=pair`) is only accepted when
  the receipt itself records `pair`, the same checkpoint digest, and the
  pre-registered sensitivity label `S1`.

Any failure is a refusal (exit 3) with the reason named. A refusal never emits a
partial verdict.

The document is a render of `results/t2-verdict.json`: `--check` re-renders it
and compares byte for byte, and the decision-bearing scalars are also emitted as
`BEGIN GENERATED` regions registered in `docs/perf-numbers.json`, so
`scripts/court perf_numbers` resolves the same numbers by a second path.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "oracle", "eval"))
sys.path.insert(0, os.path.join(REPO, "oracle", "train"))

import perf_numbers  # noqa: E402
import metrics  # noqa: E402
import t2_decision  # noqa: E402

SCHEMA = "selvedge/t2-verdict/v1"
DOC_REL = "results/T2-VERDICT.md"
RECEIPT_REL = "results/t2-verdict.json"
MANIFEST_REL = "docs/perf-numbers.json"
ID_PREFIX = "t2verdict-"

ARMS = ("D", "M", "X", "H", "H*")
ROUTED = ("X", "H", "H*")
REGISTERED_SEED = 1001
REGISTERED_STEPS = 1024
REGISTERED_BUDGET_TOKENS = 67108864
AMENDMENT_REL = "results/t2/PROVENANCE-AMENDMENT-X-and-H.json"
PREREG_REL = "docs/capability-preregistration-v4.json"
SENSITIVITY_LABEL = "S1"


class Refusal(RuntimeError):
    """The verdict may not be emitted, and the reason is named."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(path: str, payload) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, REPO)
    except ValueError:  # pragma: no cover - different drives on Windows
        return path


def _tail(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]


def _path_tail_two(path: str) -> str:
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


# ---------------------------------------------------------------------------
# receipt gates


def verify_checkpoint(checkpoint: str, recorded: str | None, where: str) -> dict:
    """The checkpoint's bytes must match the digest the receipt claims."""
    if not checkpoint or not os.path.isfile(checkpoint):
        raise Refusal(f"{where}: checkpoint {checkpoint!r} does not exist")
    sidecar_path = checkpoint + ".sha256"
    if not os.path.isfile(sidecar_path):
        raise Refusal(f"{where}: no sha256 sidecar at {sidecar_path}")
    raw = open(sidecar_path, encoding="utf-8").read().strip()
    expected = None
    if raw.startswith("{"):
        expected = (json.loads(raw) or {}).get("sha256")
    elif raw:
        expected = raw.split()[0]
    if not expected:
        raise Refusal(f"{where}: sidecar {sidecar_path} carries no sha256")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise Refusal(f"{where}: checkpoint sha256 {actual} != sidecar {expected}")
    if recorded is not None and recorded != actual:
        raise Refusal(f"{where}: receipt records checkpoint sha256 {recorded}, "
                      f"the file hashes to {actual}")
    return {"path": _rel(checkpoint), "sha256": actual,
            "sidecar": _rel(sidecar_path), "bytes": os.path.getsize(checkpoint)}


def verify_config(receipt: dict, config_path: str, where: str) -> dict:
    if not config_path or not os.path.isfile(config_path):
        raise Refusal(f"{where}: config {config_path!r} does not exist")
    digest = sha256_file(config_path)
    declared = receipt.get("config_sha256")
    if declared is None:
        declared = (receipt.get("model") or {}).get("config_sha256")
    if declared is None:
        declared = (receipt.get("config") or {}).get("sha256")
    if declared != digest:
        raise Refusal(f"{where}: config digest does not match {config_path} "
                      f"({declared!r} != {digest})")
    return {"path": _rel(config_path), "sha256": digest}


def verify_admission(receipt: dict, admission_path: str, where: str) -> dict:
    if not admission_path or not os.path.isfile(admission_path):
        raise Refusal(f"{where}: admission record {admission_path!r} does not exist")
    digest = sha256_file(admission_path)
    admission = load_json(admission_path)
    if not admission.get("admitted"):
        raise Refusal(f"{where}: the admission record does not admit this arm")
    declared = receipt.get("admission_sha256")
    if declared is None:
        declared = (receipt.get("model") or {}).get("admission_sha256")
    if declared is None:
        declared = (receipt.get("admission") or {}).get("sha256")
    if declared != digest:
        raise Refusal(f"{where}: admission digest does not match {admission_path} "
                      f"({declared!r} != {digest})")
    recorded_active = None
    block = receipt.get("admission")
    if isinstance(block, dict):
        recorded_active = block.get("active_unique_params")
    if recorded_active is None:
        recorded_active = (receipt.get("model") or {}).get("active_unique_params")
    if recorded_active is None:
        recorded_active = receipt.get("admitted_active_unique")
    if recorded_active is not None and recorded_active != admission.get("active_unique_params"):
        raise Refusal(f"{where}: admitted active-parameter count {recorded_active} "
                      f"differs from the ledger's {admission.get('active_unique_params')}")
    return {"path": _rel(admission_path), "sha256": digest,
            "active_unique_params": admission.get("active_unique_params"),
            "total_accessible": admission.get("total_accessible")}


def verify_provenance(arm: str, evaluation: dict, training: dict,
                      amendment_path: str, amendment_cache: dict) -> dict:
    """A routed arm must prove which dispatch mask placement produced it."""
    recorded = (evaluation.get("instrument") or {}).get("dispatch_mask_order")
    if recorded is None:
        dispatch = evaluation.get("dispatch") or {}
        recorded = dispatch.get("mask_order")
    trained = training.get("dispatch_mask_order")
    info = {"arm": arm, "eval_recorded_mask_order": recorded,
            "training_recorded_mask_order": trained,
            "source": "evaluation receipt" if recorded else None,
            "provenance_amendment": None}
    if arm not in ROUTED:
        return info
    amendment = None
    if recorded is None or trained is None:
        if not amendment_path or not os.path.isfile(amendment_path):
            raise Refusal(
                f"{arm}: the receipt records no dispatch mask placement and no "
                f"provenance amendment was supplied; a routed checkpoint cannot be "
                f"scored without one")
        if amendment_path not in amendment_cache:
            amendment_cache[amendment_path] = load_json(amendment_path)
        amendment = amendment_cache[amendment_path]
        if amendment.get("decision") != "CERTIFIED":
            raise Refusal(f"{arm}: the provenance amendment does not certify the arm")
        if arm not in (amendment.get("subject_arms") or []):
            raise Refusal(f"{arm}: the provenance amendment does not cover this arm")
        binding = ((amendment.get("adapter_consumption") or {})
                   .get("binding_sha256") or {})
        expected = None
        for name, digest in binding.items():
            if _path_tail_two(name) == _path_tail_two(f"results/t2/{arm}/run.json"):
                expected = digest
                break
        if expected is None:
            raise Refusal(f"{arm}: the provenance amendment does not bind "
                          f"results/t2/{arm}/run.json")
        run_path = training.get("_path")
        if not run_path:
            raise Refusal(f"{arm}: no training receipt path to bind")
        if sha256_file(run_path) != expected:
            raise Refusal(f"{arm}: the training receipt digest does not match the "
                          f"provenance amendment's binding")
        inferred = amendment.get("inferred_mask_order")
        if recorded is not None and inferred != recorded:
            raise Refusal(f"{arm}: the amendment infers {inferred!r}, the evaluation "
                          f"records {recorded!r}")
        if trained is not None and inferred != trained:
            raise Refusal(f"{arm}: the amendment infers {inferred!r}, the training "
                          f"receipt records {trained!r}")
        recorded = inferred
        info.update({
            "eval_recorded_mask_order": None,
            "source": "provenance amendment",
            "provenance_amendment": {
                "path": _rel(amendment_path),
                "sha256": sha256_file(amendment_path),
                "decision": amendment.get("decision"),
                "inferred_mask_order": inferred,
            },
        })
    else:
        if recorded != trained:
            raise Refusal(f"{arm}: the evaluation records mask_order={recorded!r}, "
                          f"the training receipt records {trained!r}")
    if recorded not in ("legacy", "pair"):
        raise Refusal(f"{arm}: unknown dispatch mask placement {recorded!r}")
    info["resolved_mask_order"] = recorded
    return info


class Receipts:
    """The five arms plus D0, after every gate has passed."""

    def __init__(self, run_root: str, eval_root: str, d0_path: str,
                 amendment_path: str | None):
        self.run_root = run_root
        self.eval_root = eval_root
        self.d0_path = d0_path
        self.amendment_path = amendment_path
        self.arms: dict[str, t2_decision.ArmMetrics] = {}
        self.raw: dict[str, dict] = {}
        self.training: dict[str, dict] = {}
        self.guards: dict[str, dict] = {}
        self._amendment_cache: dict = {}
        self._load()

    def _arm_paths(self, arm: str) -> tuple[str, str, str, str]:
        cfg = os.path.join(REPO, "results", "train", "admission-configs", "t2",
                           f"{arm}.json")
        adm = os.path.join(REPO, "results", "train", "admission", "t2", f"{arm}.json")
        run = os.path.join(REPO, self.run_root, arm, "run.json")
        ev = os.path.join(REPO, self.eval_root, f"t2-{arm}.json")
        return cfg, adm, run, ev

    def _load(self) -> None:
        required = {}
        for arm in ARMS:
            cfg, adm, run, ev = self._arm_paths(arm)
            required[arm] = {"config": cfg, "admission": adm, "run": run, "eval": ev}
        required["D0"] = {"eval": os.path.join(REPO, self.d0_path)}
        missing = []
        for name, paths in required.items():
            for kind, path in paths.items():
                if not path or not os.path.isfile(path):
                    missing.append(f"{name} {kind}: {_rel(path) if path else 'unset'}")
        if missing:
            raise Refusal("required receipt(s) missing: " + "; ".join(sorted(missing)))

        self.guards["required_receipts"] = {
            "ok": True,
            "receipts": {name: {kind: _rel(path) for kind, path in sorted(paths.items())}
                         for name, paths in sorted(required.items())},
        }

        for arm in ARMS:
            cfg, adm, run, ev = self._arm_paths(arm)
            evaluation = load_json(ev)
            training = load_json(run)
            training["_path"] = run
            self.raw[arm] = evaluation
            self.training[arm] = training

            if evaluation.get("schema") not in (None, "selvedge/t2-eval/v1"):
                raise Refusal(f"{arm}: evaluation receipt has unknown schema "
                              f"{evaluation.get('schema')!r}")
            if (evaluation.get("arm") or training.get("arm")) != arm:
                raise Refusal(f"{arm}: a receipt is for arm "
                              f"{evaluation.get('arm') or training.get('arm')!r}")
            if training.get("seed") != REGISTERED_SEED:
                raise Refusal(f"{arm}: training seed {training.get('seed')!r} is not "
                              f"the registered {REGISTERED_SEED}")
            budget = training.get("budget") or {}
            if budget.get("steps") is not None and int(budget["steps"]) != REGISTERED_STEPS:
                raise Refusal(f"{arm}: training receipt reports {budget['steps']} steps, "
                              f"the registered pilot is {REGISTERED_STEPS}")
            measured = training.get("measured") or {}
            if measured.get("tokens") is not None and \
                    int(measured["tokens"]) != REGISTERED_BUDGET_TOKENS:
                raise Refusal(f"{arm}: measured tokens {measured['tokens']} are not the "
                              f"registered {REGISTERED_BUDGET_TOKENS}")

            results = evaluation.get("results") or {}
            absent = sorted({"humaneval", "gsm8k", "truthfulqa", "distributional"}
                            - set(results))
            if absent:
                raise Refusal(f"{arm}: evaluation receipt is missing {absent}")

            checkpoint = (evaluation.get("model") or {}).get("checkpoint")
            checkpoint_digest = (evaluation.get("model") or {}).get("checkpoint_sha256")
            if checkpoint is None:
                checkpoint = (training.get("checkpoint") or {}).get("path")
            if not checkpoint:
                raise Refusal(f"{arm}: no checkpoint path in either receipt")
            if not os.path.isabs(checkpoint):
                checkpoint = os.path.join(REPO, checkpoint)
            checkpoint_block = verify_checkpoint(
                checkpoint, checkpoint_digest, f"{arm} checkpoint")
            trained_digest = (training.get("checkpoint") or {}).get("sha256")
            if trained_digest and trained_digest != checkpoint_block["sha256"]:
                raise Refusal(f"{arm}: training receipt checkpoint sha256 "
                              f"{trained_digest} differs from the file and the "
                              f"evaluation receipt")
            config_block = verify_config(evaluation, cfg, f"{arm} evaluation")
            trained_config = (training.get("config") or {}).get("sha256")
            if trained_config != config_block["sha256"]:
                raise Refusal(f"{arm}: training receipt config digest "
                              f"{trained_config!r} differs from the config file")
            admission_block = verify_admission(evaluation, adm, f"{arm} evaluation")
            provenance = verify_provenance(arm, evaluation, training,
                                           self.amendment_path,
                                           self._amendment_cache)
            self.guards[arm] = {
                "checkpoint": checkpoint_block,
                "config": config_block,
                "admission": admission_block,
                "provenance": provenance,
                "eval_stamp": evaluation.get("stamp"),
                "eval_revision": evaluation.get("revision"),
                "training_revision": training.get("revision"),
                "seed": training.get("seed"),
            }
            self.arms[arm] = t2_decision.ArmMetrics(arm, evaluation)

        d0 = load_json(os.path.join(REPO, self.d0_path))
        absent = sorted({"humaneval", "gsm8k", "truthfulqa", "distributional"}
                        - set((d0.get("results") or {})))
        if absent:
            raise Refusal(f"D0: evaluation receipt is missing {absent}")
        self.raw["D0"] = d0
        self.guards["D0"] = {
            "eval_stamp": d0.get("stamp"),
            "eval_revision": d0.get("revision"),
            "checkpoint_sha256": (d0.get("model") or {}).get("checkpoint_sha256"),
        }
        self.arms["D0"] = t2_decision.ArmMetrics("D0", d0)

    # -- extra statistics -------------------------------------------------

    def per_task_paired(self, left: str, right: str, seed: int,
                        resamples: int) -> dict:
        a, b = self.arms[left], self.arms[right]
        out = {}
        for task in t2_decision.TASKS:
            if a.task_items[task] != b.task_items[task]:
                raise Refusal(f"{left}/{right}: {task} item ids differ, so the "
                              f"comparison is not paired")
            out[task] = metrics.paired_bootstrap(
                a.task_values[task], b.task_values[task], seed=seed,
                resamples=resamples)
        return out

    def sensitivity(self, arm: str, path: str, label: str) -> dict:
        if not path:
            return {"arm": arm, "present": False, "reason": "not run"}
        if not os.path.isfile(path):
            raise Refusal(f"{arm} sensitivity: receipt {_rel(path)} does not exist")
        receipt = load_json(path)
        if label != SENSITIVITY_LABEL:
            raise Refusal(f"{arm} sensitivity: unregistered label {label!r}")
        mask = (receipt.get("instrument") or {}).get("dispatch_mask_order")
        if mask != "pair":
            raise Refusal(f"{arm} sensitivity: the receipt records "
                          f"mask_order={mask!r}, the registered sensitivity is 'pair'")
        if (receipt.get("arm") or "") != arm:
            raise Refusal(f"{arm} sensitivity: receipt is for arm "
                          f"{receipt.get('arm')!r}")
        base_digest = self.guards[arm]["checkpoint"]["sha256"]
        recorded = (receipt.get("model") or {}).get("checkpoint_sha256")
        if recorded != base_digest:
            raise Refusal(f"{arm} sensitivity: checkpoint digest {recorded!r} is not "
                          f"the registered checkpoint {base_digest}")
        metrics_arm = t2_decision.ArmMetrics(arm, receipt)
        base = self.arms[arm]
        # A point difference only; the sensitivity is not re-thresholded and no
        # interval is attached to it.
        per_task = {}
        for task in t2_decision.TASKS:
            if metrics_arm.task_items[task] != base.task_items[task]:
                raise Refusal(f"{arm} sensitivity: {task} item ids differ")
            per_task[task] = {
                "pair": float(metrics_arm.task_values[task].mean()),
                "legacy": float(base.task_values[task].mean()),
                "difference": float(metrics_arm.task_values[task].mean()
                                    - base.task_values[task].mean()),
            }
        return {
            "arm": arm,
            "label": label,
            "present": True,
            "receipt": _rel(path),
            "receipt_sha256": sha256_file(path),
            "stamp": receipt.get("stamp"),
            "revision": receipt.get("revision"),
            "mask_order": mask,
            "checkpoint_sha256": recorded,
            "pc_pair": metrics_arm.pc,
            "pc_legacy": base.pc,
            "pc_difference_point": (None if metrics_arm.pc is None or base.pc is None
                                    else metrics_arm.pc - base.pc),
            "per_task": per_task,
            "not_rethresholded": True,
            "point_only": True,
            "note": "the registered verdict stands; this re-evaluation is a "
                    "sensitivity of the same checkpoint under the corrected mask and "
                    "it changes no threshold, arm or kill condition",
        }


# ---------------------------------------------------------------------------
# the registered rule, assembled from oracle/eval/t2_decision.py


def build_payload(receipts: Receipts, seed: int, resamples: int,
                  sensitivity: dict | None) -> dict:
    arms = receipts.arms
    order = ["D0", "D", "M", "X", "H", "H*"]
    arm_table = {}
    for name in order:
        arm = arms[name]
        arm_table[name] = {
            "pc": t2_decision.composite_bootstrap(arm.task_values, seed, resamples),
            "kl_mixed": {
                "mean": arm.kl_mixed,
                "mean_stored_in_receipt": arm.kl_mixed_stored,
                "definition": "equal-weight mean of the per-domain mean per-document "
                              "teacher KL (the harness's mixed interleaving)",
                "bootstrap": arm.mixed.get("kl"),
            },
            "per_task": arm.task_means,
            "per_domain": {
                domain: {
                    "kl": float(block["kl"].mean()),
                    "kl_n": int(block["kl"].size),
                    "ppl": float(block["ppl"].mean()),
                    "top1": float(block["top1"].mean()),
                    "documents": int(block["kl"].size),
                    "windows": block["windows"],
                }
                for domain, block in sorted(arm.domains.items())
            },
            "eval_stamp": arm.stamp,
            "eval_revision": arm.revision,
            "eval_seconds": arm.seconds,
        }

    comparisons = {
        "H*-H": t2_decision.decide(arms["H*"], arms["H"],
                                   t2_decision.THRESHOLDS["K15"], seed, resamples),
        "H*-D": t2_decision.decide(arms["H*"], arms["D"],
                                   t2_decision.THRESHOLDS["K16"], seed, resamples),
        "H*-D0": t2_decision.decide(arms["H*"], arms["D0"],
                                    t2_decision.THRESHOLDS["K16"], seed, resamples),
        "X-D": t2_decision.decide(arms["X"], arms["D"],
                                  t2_decision.THRESHOLDS["K18"], seed, resamples),
        "M-D0": t2_decision.decide(arms["M"], arms["D0"],
                                   t2_decision.THRESHOLDS["K17"], seed, resamples),
        "M-D": t2_decision.decide(arms["M"], arms["D"],
                                  t2_decision.THRESHOLDS["K17"], seed, resamples),
        # Reported for completeness; not a registered kill condition.
        "X-H": t2_decision.decide(arms["X"], arms["H"], 0.03, seed, resamples),
    }
    for label, comparison in comparisons.items():
        left, right = label.split("-")
        comparison["per_task"] = receipts.per_task_paired(left, right, seed, resamples)
    comparisons["X-H"]["registered_kill_condition"] = None
    comparisons["X-H"]["note"] = ("reported because the exploratory partial record "
                                  "compared X and H; it decides nothing")

    k17_gap = t2_decision.gap_closed(arms["M"], arms["D0"], arms["D"],
                                     comparisons["M-D0"]["decisive_statistic"])
    kill = {
        "K15": {
            "statement": "H* must improve over H on the decisive statistic by >= 3% "
                         "relative with the paired 95% CI excluding zero and no "
                         "regression beyond the margin on the other endpoint",
            "comparison": comparisons["H*-H"],
            "passes": comparisons["H*-H"]["passes"],
        },
        "K16": {
            "statement": "H* must improve over D by >= 5% relative with the paired "
                         "95% CI excluding zero and no regression beyond the margin",
            "comparison": comparisons["H*-D"],
            "passes": comparisons["H*-D"]["passes"],
        },
        "K17": {
            "condition": "M vs D0 >= 5% relative with CI excluding zero (clause 1) AND "
                         f"gap_closed >= {t2_decision.GAP_CLOSED_MIN} (clause 2)",
            "clause_1_passes": comparisons["M-D0"]["passes"],
            "clause_2": k17_gap,
            "passes": bool(comparisons["M-D0"]["passes"] and k17_gap["passes"]),
        },
        "K18": {
            "statement": "X must improve over D by >= 3% relative with the paired "
                         "95% CI excluding zero and no regression beyond the margin",
            "comparison": comparisons["X-D"],
            "passes": comparisons["X-D"]["passes"],
        },
    }
    kill["K19"] = {
        "statement": "scaling the bank, P3.5 phases C/D, or spending WS4/WS5 on the "
                     "conditional-capacity assumption requires both K15 and K16",
        "k15": kill["K15"]["passes"],
        "k16": kill["K16"]["passes"],
        "passes": bool(kill["K15"]["passes"] and kill["K16"]["passes"]),
    }
    for name, entry in kill.items():
        entry["verdict"] = "PASS" if entry["passes"] else "FAIL"

    h_star_vs_h = kill["K15"]["passes"]
    h_star_vs_d = kill["K16"]["passes"]
    if h_star_vs_h and h_star_vs_d:
        outcome = "hybrid_supported"
        interpretation = (
            "K15 and K16 pass: error-directed conditional capacity is supported at "
            "this scale; the bank-scaling ladder (P2.2) and P3.5 phases C/D become "
            "the next experiments.")
    elif h_star_vs_h and not h_star_vs_d:
        outcome = "error_direction_supported_but_not_over_dense"
        interpretation = (
            "K15 passes, K16 fails: error direction is a real routing effect but "
            "conditional capacity does not beat dense capacity of equal active size "
            "at this budget. The bank is not scaled.")
    elif not h_star_vs_h and h_star_vs_d:
        outcome = "inconsistent_k15_fails_k16_passes"
        interpretation = (
            "K16 passes while K15 fails: an inconsistent pattern that cannot be read "
            "as support for error direction. Reported as is; nothing is scaled, and "
            "the anomaly is recorded rather than explained away.")
    else:
        outcome = "error_direction_not_supported"
        interpretation = (
            "K15 fails: error-directed routing is not supported at this scale. WS3's "
            "central bet is recorded as a negative result, no bank scaling follows, "
            "and the mechanism is not carried into WS4/WS5 without a new "
            "pre-registration.")

    resources = {}
    for name in ARMS:
        training = receipts.training[name]
        measured = training.get("measured") or {}
        resources[name] = {
            "active_unique_params": receipts.guards[name]["admission"].get(
                "active_unique_params"),
            "total_accessible_params": (receipts.guards[name]["admission"].get(
                "total_accessible")
                or training.get("total_accessible")),
            "tokens": measured.get("tokens"),
            "tokens_per_second": measured.get("tokens_per_second"),
            "gpu_hours": measured.get("gpu_hours"),
            "vram_peak_gib": measured.get("vram_peak_gib"),
            "wall_seconds": measured.get("wall_seconds"),
            "seed": training.get("seed"),
            "revision": training.get("revision"),
            "run_receipt": _rel(os.path.join(REPO, receipts.run_root, name,
                                             "run.json")),
        }

    decisive = kill["K15"]["comparison"]
    one_seed = {
        "seed_registered": REGISTERED_SEED,
        "seeds_per_arm": 1,
        "seed_level_ci_claimed": False,
        "replication_present": False,
        "decisive_pair": "H*-H",
        "ci_half_width": decisive["one_seed_rule"]["ci_half_width"],
        "twice_half_width": decisive["one_seed_rule"]["twice_half_width"],
        "difference_abs": decisive["one_seed_rule"]["difference_abs"],
        "separable_from_noise": decisive["one_seed_rule"]["separable_from_noise"],
        "text": decisive["one_seed_rule"]["separable_text"],
        "rule": "T2.8: with one seed per arm no seed-level CI is claimed and the "
                "H*-H difference must exceed twice the item-level CI half-width to "
                "count; otherwise it is 'not separable from noise'",
    }

    payload = {
        "schema": SCHEMA,
        "stamp": stamp_now(),
        "utc": utc_now(),
        "revision": os.environ.get("SELVEDGE_REVISION", "unknown"),
        "contract": "docs/EXPERIMENT-CONTRACT.md T2.9/T2.10 with A8, A9 and the "
                    "dated sensitivity addition A10",
        "registered_seed": seed,
        "resamples": resamples,
        "guards": receipts.guards,
        "decision_order": {
            "primary": "PC = (HumanEval pass@1 + GSM8K exact match + TruthfulQA MC1) / 3",
            "co_primary": "P1 = pooled teacher KL on the fixed mixed interleaving",
            "fallback": "if a comparison's paired PC 95% CI contains zero, P1 is "
                        "decisive for that comparison",
            "margin_rule": "relative, or absolute +0.01 when the comparator's PC is "
                           "below 0.02",
            "no_regression": "KL <= 1.02x the comparator's and PC >= comparator - 0.01",
        },
        "arms": arm_table,
        "training": resources,
        "model_resources": t2_decision.model_resources(resources),
        "comparisons": comparisons,
        "kill_conditions": kill,
        "outcome": outcome,
        "interpretation": interpretation,
        "one_seed": one_seed,
        "sensitivity": sensitivity,
        "caveats": [
            "one seed per arm (T2.8): no seed-level CI is claimed",
            "the evaluation reports the compact training target; the full-vocabulary "
            "teacher KL is in the P2.4 instrument receipt (A9)",
            "the corrected-dispatch sensitivity (A10) re-evaluates the same X and H "
            "checkpoints; it changes no threshold and does not replace the "
            "registered verdict",
            "every number in results/T2-VERDICT.md is rendered from this receipt",
        ],
    }
    body = {key: value for key, value in payload.items()
            if key not in ("stamp", "utc")}
    payload["deterministic_body_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# the document


class Renderer:
    def __init__(self, payload: dict, receipt_rel: str, doc_rel: str):
        self.payload = payload
        self.receipt_rel = receipt_rel
        self.doc_rel = doc_rel
        self.stamp = payload["stamp"]
        self.revision = payload["revision"]
        self.blocks: list[dict] = []
        self.published: dict[str, str] = {}

    def value(self, pointer: str, fmt: str) -> str:
        return perf_numbers.render_value(
            perf_numbers.resolve_pointer(self.payload, pointer, pointer), fmt, pointer)

    def region(self, name: str, pointer: str, fmt: str, label: str) -> str:
        block_id = ID_PREFIX + name
        rendered = self.value(pointer, fmt)
        self.blocks.append({
            "id": block_id,
            "doc": self.doc_rel,
            "sentinel": block_id,
            "source": {"receipt": self.receipt_rel, "pointer": pointer},
            "format": fmt,
            "declared_receipt_stamp": self.stamp,
            "declared_revision": self.revision,
        })
        self.published[name] = rendered
        return "\n".join([label and f"{label}:" or "",
                          f"<!-- BEGIN GENERATED: {block_id} -->",
                          rendered,
                          f"<!-- END GENERATED: {block_id} -->"]).strip()


COMPARISON_TITLES = (
    ("H*-H", "K15", "`H*` vs `H` — error direction"),
    ("H*-D", "K16", "`H*` vs `D` — hybrid versus dense capacity"),
    ("X-D", "K18", "`X` vs `D` — routed specialists (T1 K13 replication)"),
    ("M-D0", "K17", "`M` vs `D0` — memory at equal active parameters"),
    ("M-D", "—", "`M` vs `D` — memory at unequal active parameters"),
    ("H*-D0", "—", "`H*` vs `D0` — hybrid versus the frozen core"),
    ("X-H", "—", "`X` vs `H` — reported, decides nothing"),
)


def render_document(r: Renderer, regions: dict) -> str:
    p = r.payload
    lines: list[str] = []
    lines.append("# T2 verdict — the five-arm error-directed capacity matrix")
    lines.append("")
    lines.append(f"Generated by `scripts/t2_verdict.py` from "
                 f"`{r.receipt_rel}` at {p['utc']} (receipt stamp "
                 f"`{p['stamp']}`). Every number in this document is rendered from "
                 f"that receipt; `--check` re-renders it and compares byte for byte, "
                 f"and the decision-bearing scalars are published as "
                 f"`BEGIN GENERATED` regions checked by the perf-numbers court.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(regions["outcome"])
    lines.append("")
    lines.append(p["interpretation"])
    lines.append("")
    lines.append("## Receipt gates")
    lines.append("")
    lines.append("Every arm's checkpoint was hashed against its own sidecar and its "
                 "config and admission digests against the files on disk. A mismatch "
                 "stops this reader before it produces a verdict, so the table below "
                 "is the set of receipts that passed.")
    lines.append("")
    lines.append("| arm | checkpoint sha256 | config sha256 | admission sha256 | "
                 "admitted active | dispatch mask | provenance source |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- |")
    for arm in ARMS:
        g = p["guards"][arm]
        prov = g["provenance"]
        lines.append(
            f"| {arm} | `{g['checkpoint']['sha256'][:16]}…` | "
            f"`{g['config']['sha256'][:16]}…` | "
            f"`{g['admission']['sha256'][:16]}…` | "
            f"{g['admission']['active_unique_params']:,} | "
            f"{prov.get('resolved_mask_order') or '—'} | {prov.get('source') or '—'} |")
    lines.append("")
    amendment = next((p["guards"][arm]["provenance"].get("provenance_amendment")
                      for arm in ROUTED
                      if p["guards"][arm]["provenance"].get("provenance_amendment")),
                     None)
    if amendment:
        lines.append(f"Provenance amendment: `{amendment['path']}`, sha256 "
                     f"`{amendment['sha256']}`, decision `{amendment['decision']}`, "
                     f"inferred mask order `{amendment['inferred_mask_order']}`.")
        lines.append("")
    lines.append("## Per-arm endpoints")
    lines.append("")
    lines.append("| arm | PC | HumanEval | GSM8K | TruthfulQA | teacher KL (mixed) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm in ("D0", "D", "M", "X", "H", "H*"):
        row = p["arms"][arm]
        lines.append(
            f"| {arm} | {regions['pc-' + arm.replace('*', 'star')]} | "
            f"{row['per_task']['humaneval']:.4f} | {row['per_task']['gsm8k']:.4f} | "
            f"{row['per_task']['truthfulqa']:.4f} | {row['kl_mixed']['mean']:.4f} |")
    lines.append("")
    lines.append("PC is the registered primary composite; the per-task columns are "
                 "the three task accuracies that compose it, and the last column is "
                 "the co-primary pooled teacher KL.")
    lines.append("")
    lines.append("## Registered comparisons")
    lines.append("")
    for label, condition, title in COMPARISON_TITLES:
        slug = label.replace("*", "star")
        lines.append(f"### {title} ({condition})")
        lines.append("")
        lines.append(regions[f"{slug}-decisive"])
        lines.append("")
        lines.append(regions[f"{slug}-pc"])
        lines.append("")
        lines.append(regions[f"{slug}-kl"])
        lines.append("")
        lines.append(regions[f"{slug}-verdict"])
        lines.append("")
        lines.append("| task | arm | comparator | difference | 95% paired CI |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for task in t2_decision.TASKS:
            entry = p["comparisons"][label]["per_task"][task]
            left, right = label.split("-")
            lines.append(
                f"| {task} | {p['arms'][left]['per_task'][task]:.4f} | "
                f"{p['arms'][right]['per_task'][task]:.4f} | "
                f"{entry['mean_difference']:+.4f} | "
                f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] |")
        lines.append("")
    lines.append("## Kill conditions")
    lines.append("")
    lines.append("| condition | statement | verdict |")
    lines.append("| --- | --- | --- |")
    for name in ("K15", "K16", "K17", "K18", "K19"):
        statement = p["kill_conditions"][name].get(
            "statement") or p["kill_conditions"][name].get("condition")
        lines.append(f"| {name} | {statement} | {regions['kill-' + name]} |")
    lines.append("")
    lines.append("K17 clause 2 (the fraction of the `D0` -> `D` gap that `M` closes) "
                 f"is {p['kill_conditions']['K17']['clause_2']['fraction']} "
                 f"(minimum {p['kill_conditions']['K17']['clause_2']['minimum']}).")
    lines.append("")
    lines.append("## Seed-level limitation")
    lines.append("")
    lines.append(regions["one-seed"])
    lines.append("")
    lines.append("## Corrected-dispatch sensitivity (registered addition A10)")
    lines.append("")
    if not p["sensitivity"]:
        lines.append("Not run in this receipt. The registered verdict above stands "
                     "without it.")
    else:
        lines.append("The same checkpoints re-evaluated with `mask_order=pair`. This "
                     "is a distinct registered configuration: it retrains nothing, "
                     "changes no threshold and is reported beside the registered "
                     "verdict, never instead of it.")
        lines.append("")
        lines.append("| arm | PC (legacy) | PC (pair) | difference | receipt |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for arm in ("X", "H"):
            entry = (p["sensitivity"].get(arm) or {})
            if not entry.get("present"):
                lines.append(f"| {arm} | — | — | — | not run |")
                continue
            lines.append(
                f"| {arm} | {entry['pc_legacy']:.4f} | {entry['pc_pair']:.4f} | "
                f"{entry['pc_difference_point']:+.4f} | `{entry['receipt']}` |")
        lines.append("")
    lines.append("## What this establishes")
    lines.append("")
    lines.append("- The registered decision rule was applied exactly, at the "
                 "registered seed and resample count, to receipts whose digests are "
                 "recorded above.")
    lines.append("- A failure is a negative result, not a missing one: every failing "
                 "kill condition above is a determination about this pilot at this "
                 "budget.")
    lines.append("")
    lines.append("## What this does not establish")
    lines.append("")
    lines.append("- No seed-level interval: one seed per arm.")
    lines.append("- No capability claim: this document is the loss- and task-based "
                 "instrument. The verified-task instrument is "
                 "`scripts/capability_compare.py` and its own receipts.")
    lines.append("- No claim about capacity beyond the accessible total: the "
                 "capacity ladder is out of scope (T2.1.1).")
    return "\n".join(lines) + "\n"


def define_regions(r: Renderer) -> dict[str, str]:
    out = {"outcome": r.region("outcome", "/outcome", "{value}",
                               "registered outcome")}
    for arm in ("D0", "D", "M", "X", "H", "H*"):
        slug = arm.replace("*", "star")
        out[f"pc-{slug}"] = r.region(f"pc-{slug}", f"/arms/{arm}/pc/mean",
                                     "{value:.4f}", "")
    for label, _condition, _title in COMPARISON_TITLES:
        slug = label.replace("*", "star")
        base = f"/comparisons/{label}"
        out[f"{slug}-decisive"] = r.region(
            f"{slug}-decisive", f"{base}/decisive_statistic", "{value}",
            "Decisive statistic")
        out[f"{slug}-pc"] = "\n".join([
            "Paired PC difference and interval:",
            r.region(f"{slug}-pc-delta", f"{base}/pc/difference/mean_difference",
                     "{value:+.4f}", "difference"),
            r.region(f"{slug}-pc-lo", f"{base}/pc/difference/ci_low",
                     "{value:+.4f}", "95% CI low"),
            r.region(f"{slug}-pc-hi", f"{base}/pc/difference/ci_high",
                     "{value:+.4f}", "95% CI high"),
        ])
        out[f"{slug}-kl"] = "\n".join([
            "Paired teacher-KL difference and interval (lower is better):",
            r.region(f"{slug}-kl-delta", f"{base}/kl/difference/mean_difference",
                     "{value:+.4f}", "difference"),
            r.region(f"{slug}-kl-lo", f"{base}/kl/difference/ci_low",
                     "{value:+.4f}", "95% CI low"),
            r.region(f"{slug}-kl-hi", f"{base}/kl/difference/ci_high",
                     "{value:+.4f}", "95% CI high"),
        ])
        out[f"{slug}-verdict"] = r.region(f"{slug}-verdict", f"{base}/verdict",
                                          "{value}", "Verdict")
    for name in ("K15", "K16", "K17", "K18", "K19"):
        out[f"kill-{name}"] = r.region(f"kill-{name}",
                                       f"/kill_conditions/{name}/verdict",
                                       "{value}", "")
    out["one-seed"] = "\n".join([
        r.region("one-seed-halves", "/one_seed/difference_abs", "{value:+.4f}",
                 "Observed absolute H*-H difference"),
        r.region("one-seed-twice", "/one_seed/twice_half_width", "{value:.4f}",
                 "Twice the item-level CI half-width"),
        r.region("one-seed-separable", "/one_seed/text", "{value}",
                 "separability"),
    ])
    return out


def merge_manifest(blocks: list[dict], manifest_rel: str) -> None:
    path = os.path.join(REPO, manifest_rel)
    manifest = perf_numbers.load_manifest(path)
    kept = [block for block in manifest["blocks"]
            if not block["id"].startswith(ID_PREFIX)]
    merged = kept + blocks
    merged.sort(key=lambda block: block["id"])
    manifest["blocks"] = merged
    write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=None,
                        help="the repository root every relative path resolves "
                             "against (default: the parent of scripts/)")
    parser.add_argument("--run-root", default="results/t2")
    parser.add_argument("--eval-root", default="results/eval")
    parser.add_argument("--d0", default="results/eval/t2-D0.json")
    parser.add_argument("--amendment", default=AMENDMENT_REL)
    parser.add_argument("--sensitivity", action="append", default=[],
                        help="ARM=path to a mask_order=pair re-evaluation receipt")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--out", default=RECEIPT_REL)
    parser.add_argument("--doc", default=DOC_REL)
    parser.add_argument("--manifest", default=MANIFEST_REL)
    parser.add_argument("--no-manifest", action="store_true",
                        help="do not register the generated regions (tests)")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    global REPO
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)
    if args.amendment and not os.path.isabs(args.amendment):
        args.amendment = os.path.join(REPO, args.amendment)
    sensitivity_paths: dict[str, str] = {}
    for entry in args.sensitivity:
        if "=" not in entry:
            raise SystemExit(f"--sensitivity expects ARM=path, got {entry!r}")
        arm, path = entry.split("=", 1)
        if arm not in ("X", "H"):
            raise SystemExit(f"--sensitivity covers X and H only, got {arm!r}")
        sensitivity_paths[arm] = path

    try:
        receipts = Receipts(args.run_root, args.eval_root, args.d0, args.amendment)
        sensitivity = {}
        for arm in ("X", "H"):
            if arm in sensitivity_paths:
                sensitivity[arm] = receipts.sensitivity(
                    arm, sensitivity_paths[arm], SENSITIVITY_LABEL)
            else:
                sensitivity[arm] = {"arm": arm, "present": False, "reason": "not run"}
        payload = build_payload(receipts, args.seed, args.resamples, sensitivity)
    except Refusal as error:
        print(f"t2_verdict: REFUSED — {error}", file=sys.stderr)
        return 3

    renderer = Renderer(payload, args.out, args.doc)
    regions = define_regions(renderer)
    text = render_document(renderer, regions)
    payload["document"] = {"path": args.doc, "blocks": len(renderer.blocks)}

    if args.check:
        doc_path = os.path.join(REPO, args.doc)
        receipt_path = os.path.join(REPO, args.out)
        if not os.path.isfile(doc_path) or not os.path.isfile(receipt_path):
            print("t2_verdict: cannot check; the receipt or the document is missing",
                  file=sys.stderr)
            return 1
        existing = open(doc_path, encoding="utf-8").read()
        if existing != text:
            print(f"t2_verdict: {args.doc} is stale against {args.out}; run --write",
                  file=sys.stderr)
            return 1
        print(f"t2_verdict: {args.doc} matches {args.out} "
              f"({len(renderer.blocks)} published region(s))")
        return 0

    if args.write:
        write_json(os.path.join(REPO, args.out), payload)
        write_text(os.path.join(REPO, args.doc), text)
        if not args.no_manifest:
            merge_manifest(renderer.blocks, args.manifest)
        print(f"t2_verdict: wrote {args.out} and {args.doc} "
              f"({len(renderer.blocks)} published region(s))")

    if not args.quiet:
        print(f"T2 verdict ({payload['stamp']}) — outcome: {payload['outcome']}")
        print()
        print(f"{'arm':4} {'PC':>7} {'KL mixed':>9} {'HE':>6} {'GSM8K':>7} {'MC1':>6}")
        for arm in ("D0", "D", "M", "X", "H", "H*"):
            row = payload["arms"][arm]
            print(f"{arm:4} {row['pc']['mean']:>7.4f} "
                  f"{row['kl_mixed']['mean']:>9.4f} "
                  f"{row['per_task']['humaneval']:>6.3f} "
                  f"{row['per_task']['gsm8k']:>7.3f} "
                  f"{row['per_task']['truthfulqa']:>6.3f}")
        print()
        for label in ("H*-H", "H*-D", "X-D", "M-D0", "M-D", "H*-D0", "X-H"):
            comparison = payload["comparisons"][label]
            pc = comparison["pc"]["difference"]
            print(f"{label:6} decisive={comparison['decisive_statistic']:2} "
                  f"PC {pc['mean_difference']:+.4f} "
                  f"[{pc['ci_low']:+.4f},{pc['ci_high']:+.4f}] "
                  f"-> {comparison['verdict']}")
        print()
        for name in ("K15", "K16", "K17", "K18", "K19"):
            print(f"{name}: {payload['kill_conditions'][name]['verdict']}")
        print()
        print(payload["interpretation"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
