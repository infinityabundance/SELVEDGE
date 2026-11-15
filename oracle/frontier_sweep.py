#!/usr/bin/env python3
"""M1 frontier sweep: exact group omission on real Qwen3 checkpoints.

Subcommands, each of which writes its own receipt:

  stats      calibration group statistics and selector rankings
  exactness  instrument gates G1 on real held-out windows, plus the files for
             the independent Rust cross-check
  cross-check compare the Python and Rust masked partial sums
  sweep      evaluate arms on the held-out split (uniform | layers | greedy | control)
  analyze    bootstrap CIs, the frontier JSON, the SVG plot and the table

The contract is `docs/EXPERIMENT-CONTRACT.md`, registered before any of this
ran. Nothing here trains, samples text or reads held-out text for selection.
All arithmetic is f32 with TF32 disabled; the teacher is the dense f32 model.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import masked_mlp

FRONTIER_SEED = 20261114
SCHEMA_STATS = "selvedge/frontier/stats-v1"
SCHEMA_EXACTNESS = "selvedge/frontier/exactness-v1"
SCHEMA_CROSS_CHECK = "selvedge/frontier/cross-check-v1"
SCHEMA_SWEEP = "selvedge/frontier/sweep-v1"
SCHEMA_FRONTIER = "selvedge/frontier/curve-v1"
GREEN = "\u2713"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def environment(device: torch.device) -> dict:
    driver = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        driver = None
    props = (
        torch.cuda.get_device_properties(device)
        if device.type == "cuda"
        else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if props else "cpu",
        "device_capability": f"{props.major}.{props.minor}" if props else None,
        "device_total_memory_bytes": props.total_memory if props else None,
        "driver": driver,
        "arithmetic": {
            "dtype": "float32",
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "allow_bf16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "seed": FRONTIER_SEED,
    }


def configure_arithmetic() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(FRONTIER_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FRONTIER_SEED)


def model_cache_relative_dir(lock: dict, model_id: str) -> dict:
    for entry in lock["models"]:
        if entry["id"] == model_id:
            return entry
    raise SystemExit(f"models lock has no entry for {model_id}")


def model_directory(entry: dict) -> str:
    root = "/hf-cache" if os.path.isdir("/hf-cache") else os.path.expanduser(
        "~/.cache/selvedge/hf"
    )
    return os.path.join(root, entry["cache_relative_dir"])


def load_model(model_dir: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=False,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    model.to(device)
    return model


def accounting_record(path: str) -> dict:
    record = load_json(path)
    budget = record["budget"]
    return {
        "receipt": path,
        "sha256": sha256_file(path),
        "stamp": record.get("stamp"),
        "revision": record.get("revision"),
        "ceiling": budget["ceiling"],
        "unique_learned_elements": budget["unique_learned_elements"],
        "always_active_backbone_elements": budget["always_active_backbone_elements"],
        "removable_ffn_elements": budget["removable_ffn_elements"],
        "max_compliant_ffn_elements": budget["max_compliant_ffn_elements"],
        "group_count": budget["group_count"],
        "group_width": budget["group_width"],
        "intermediate_per_layer": budget["intermediate_per_layer"],
        "hidden": record["model"]["hidden"],
    }


def load_split(path: str, expected_sha256: str) -> dict:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SystemExit(
            f"{path} sha256 {actual} does not match the split receipt {expected_sha256}"
        )
    return load_json(path)


def split_receipt_for(path: str) -> dict:
    """Find the tracked split receipt whose derived file and hash match `path`."""
    actual = sha256_file(path)
    matches = []
    for candidate in sorted(glob.glob("results/splits/split-*.json")):
        receipt = load_json(candidate)
        if os.path.abspath(receipt["derived"]["path"]) != os.path.abspath(path):
            continue
        if receipt["derived"]["sha256"] != actual:
            continue
        matches.append((receipt["stamp"], candidate, receipt))
    if not matches:
        raise SystemExit(
            f"no split receipt names {path} with its current sha256 {actual}; "
            "run oracle/dataset_prep.py first"
        )
    return max(matches, key=lambda item: item[0])[2]


def selection_hash(selection: list[list[int]]) -> str:
    return sha256_text(json.dumps(selection, separators=(",", ":")))


def make_masks(
    selection: list[list[int]], groups: int, intermediate: int, device: torch.device
) -> list[torch.Tensor]:
    return [
        masked_mlp.selected_mask(intermediate, groups, selected, device=device)
        for selected in selection
    ]


class WindowAccumulator:
    """Per-article aggregates: window means plus the pooled position tensors."""

    def __init__(self) -> None:
        self.kl_chunks: list[torch.Tensor] = []
        self.kl_window_means: list[float] = []
        self.top1_window_means: list[float] = []
        self.entropy_window_means: list[float] = []
        self.nll_teacher_sum = 0.0
        self.nll_masked_sum = 0.0
        self.nll_positions = 0
        self.positions = 0
        self.windows = 0

    def add(self, summary: dict, kl_tensor: torch.Tensor) -> None:
        self.windows += 1
        self.positions += summary["positions"]
        self.kl_chunks.append(kl_tensor)
        self.kl_window_means.append(summary["kl_mean"])
        self.top1_window_means.append(summary["top1_agreement"])
        self.entropy_window_means.append(summary["teacher_entropy_mean"])
        self.nll_teacher_sum += summary["nll_teacher_mean"] * summary["nll_positions"]
        self.nll_masked_sum += summary["nll_masked_mean"] * summary["nll_positions"]
        self.nll_positions += summary["nll_positions"]

    def finish(self, doc_id: str) -> dict:
        kl = torch.cat(self.kl_chunks) if self.kl_chunks else torch.zeros(0)
        count = len(self.kl_window_means) or 1
        return {
            "doc_id": doc_id,
            "windows": self.windows,
            "positions": self.positions,
            "kl_mean": float(sum(self.kl_window_means) / count),
            "kl_median": float(torch.median(kl)) if kl.numel() else 0.0,
            "kl_p95": float(torch.quantile(kl, 0.95)) if kl.numel() else 0.0,
            "kl_min": float(kl.min()) if kl.numel() else 0.0,
            "kl_max": float(kl.max()) if kl.numel() else 0.0,
            "top1_agreement": float(sum(self.top1_window_means) / count),
            "teacher_entropy_mean": float(sum(self.entropy_window_means) / count),
            "nll_teacher_sum": self.nll_teacher_sum,
            "nll_masked_sum": self.nll_masked_sum,
            "nll_positions": self.nll_positions,
        }


def aggregate_items(items: list[dict]) -> dict:
    if not items:
        return {}
    nll_positions = sum(item["nll_positions"] for item in items)
    teacher = sum(item["nll_teacher_sum"] for item in items)
    masked = sum(item["nll_masked_sum"] for item in items)
    ppl_ratio = (
        float(np.exp((masked - teacher) / nll_positions)) if nll_positions else 1.0
    )
    return {
        "items": len(items),
        "positions": sum(item["positions"] for item in items),
        "kl_mean": float(np.mean([item["kl_mean"] for item in items])),
        "kl_median": float(np.mean([item["kl_median"] for item in items])),
        "kl_p95_mean": float(np.mean([item["kl_p95"] for item in items])),
        "top1_agreement": float(np.mean([item["top1_agreement"] for item in items])),
        "teacher_entropy_mean": float(
            np.mean([item["teacher_entropy_mean"] for item in items])
        ),
        "ppl_ratio": ppl_ratio,
        "ppl_teacher": float(np.exp(teacher / nll_positions)) if nll_positions else 1.0,
        "nll_positions": nll_positions,
    }


def all_groups(accounting: dict) -> list[list[int]]:
    groups = accounting["group_count"]
    return [list(range(groups)) for _ in range(28)]


def arm_record(
    arm_id: str,
    kind: str,
    selection: list[list[int]],
    accounting: dict,
    device: torch.device,
    extra: dict | None = None,
) -> dict:
    active = masked_mlp.active_parameter_report(
        selection,
        hidden=accounting["hidden"],
        intermediate=accounting["intermediate_per_layer"],
        groups=accounting["group_count"],
        backbone_elements=accounting["always_active_backbone_elements"],
    )
    record = {
        "id": arm_id,
        "kind": kind,
        "selection": selection,
        "selection_sha256": selection_hash(selection),
        "groups_per_layer": [len(selected) for selected in selection],
        "active": active,
        "compliant": active["active_unique_elements"] <= accounting["ceiling"],
    }
    if extra:
        record.update(extra)
    return record


def build_uniform_arms(
    plans: dict, accounting: dict, device: torch.device
) -> list[dict]:
    layers = len(plans["magnitude"])
    groups = accounting["group_count"]
    arms = []
    for selector in ("random", "magnitude", "frequency"):
        for k in range(1, groups + 1):
            selection = [plans[selector][layer][:k] for layer in range(layers)]
            arms.append(
                arm_record(
                    f"uniform-{selector}-k{k}",
                    "uniform",
                    selection,
                    accounting,
                    device,
                    {"selector": selector, "k": k},
                )
            )
    arms.append(
        arm_record("masked-all", "reference", all_groups(accounting), accounting, device)
    )
    arms.append(
        arm_record(
            "masked-none-ffn",
            "floor",
            [[] for _ in range(layers)],
            accounting,
            device,
        )
    )
    arms.append(
        arm_record(
            "teacher-dense",
            "reference",
            all_groups(accounting),
            accounting,
            device,
            {"dense_reference": True},
        )
    )
    return arms


def build_layer_arms(accounting: dict, device: torch.device) -> list[dict]:
    layers = 28
    groups = accounting["group_count"]
    arms = [
        arm_record(
            "teacher-dense",
            "reference",
            [list(range(groups)) for _ in range(layers)],
            accounting,
            device,
            {"dense_reference": True},
        )
    ]
    for layer in range(layers):
        selection = [
            [] if index == layer else list(range(groups)) for index in range(layers)
        ]
        arms.append(
            arm_record(
                f"layer-omit-{layer}",
                "layer-omission",
                selection,
                accounting,
                device,
                {"omitted_layer": layer},
            )
        )
    per_layer = 3 * accounting["hidden"] * accounting["intermediate_per_layer"]
    affordable = accounting["max_compliant_ffn_elements"] // per_layer
    omit = layers - int(affordable)
    for name, omitted in (
        ("first", set(range(omit))),
        ("last", set(range(layers - omit, layers))),
    ):
        selection = [
            [] if index in omitted else list(range(groups)) for index in range(layers)
        ]
        arms.append(
            arm_record(
                f"layer-omit-{name}-{omit}",
                "layer-omission",
                selection,
                accounting,
                device,
                {"omitted_layers": sorted(omitted)},
            )
        )
    return arms


def attach_masks(arms: list[dict], accounting: dict, device: torch.device) -> None:
    for arm in arms:
        if "masks" not in arm:
            arm["masks"] = make_masks(
                arm["selection"],
                accounting["group_count"],
                accounting["intermediate_per_layer"],
                device,
            )


def evaluate_with_masks(
    model,
    controller,
    arms: list[dict],
    domains: dict,
    accounting: dict,
    device: torch.device,
) -> dict:
    """Evaluate arms whose GPU masks are in `arm['masks']`."""
    results = {arm["id"]: {domain: {} for domain in domains} for arm in arms}
    timings = {
        arm["id"]: {"seconds": 0.0, "forwards": 0, "nonfinite": 0} for arm in arms
    }
    nonfinite = []
    for domain, items in domains.items():
        for item in items:
            for window in item["windows"]:
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                controller.suspend()
                with torch.inference_mode():
                    teacher = model(input_ids=input_ids, use_cache=False).logits
                controller.resume()
                for arm in arms:
                    doc_id = item["doc_id"]
                    accumulator = results[arm["id"]][domain].setdefault(
                        doc_id, WindowAccumulator()
                    )
                    if arm.get("dense_reference"):
                        metrics = masked_mlp.logits_kl_metrics(
                            teacher, teacher, input_ids
                        )
                        seconds = 0.0
                    else:
                        use_cuda = device.type == "cuda"
                        start = torch.cuda.Event(enable_timing=True) if use_cuda else None
                        end = torch.cuda.Event(enable_timing=True) if use_cuda else None
                        if use_cuda:
                            start.record()
                        with torch.inference_mode():
                            controller.set_masks(arm["masks"])
                            masked = model(input_ids=input_ids, use_cache=False).logits
                            metrics = masked_mlp.logits_kl_metrics(
                                teacher, masked, input_ids
                            )
                        if use_cuda:
                            end.record()
                            torch.cuda.synchronize()
                            seconds = start.elapsed_time(end) / 1000.0
                        else:
                            seconds = 0.0
                        if not bool(torch.isfinite(masked).all()):
                            timings[arm["id"]]["nonfinite"] += 1
                            nonfinite.append(
                                {"arm": arm["id"], "domain": domain, "doc": doc_id}
                            )
                        del masked
                    summary = masked_mlp.summarize_positions(metrics)
                    accumulator.add(summary, metrics["kl"])
                    timings[arm["id"]]["seconds"] += seconds
                    timings[arm["id"]]["forwards"] += int(
                        not arm.get("dense_reference")
                    )
                del teacher
    for arm in arms:
        for domain in domains:
            results[arm["id"]][domain] = {
                doc_id: accumulator.finish(doc_id)
                for doc_id, accumulator in results[arm["id"]][domain].items()
            }
    return {"results": results, "timings": timings, "nonfinite": nonfinite}


def arms_to_receipt(arms, evaluation, domains) -> list[dict]:
    out = []
    for arm in arms:
        record = {key: value for key, value in arm.items() if key != "masks"}
        record["timing"] = evaluation["timings"][arm["id"]]
        record["domains"] = {}
        for domain in domains:
            items = list(evaluation["results"][arm["id"]][domain].values())
            record["domains"][domain] = {
                "items": items,
                "aggregate": aggregate_items(items),
            }
        out.append(record)
    return out


def plans_for(plans_path: str, accounting: dict) -> dict:
    plans = load_json(plans_path)
    if plans["groups"] != accounting["group_count"]:
        raise SystemExit("plan group count disagrees with the accounting receipt")
    if len(plans["magnitude"]) != 28:
        raise SystemExit("plan layer count is not 28")
    return plans


def command_stats(args) -> int:
    configure_arithmetic()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split_receipt = split_receipt_for(args.split)
    split = load_split(args.split, split_receipt["derived"]["sha256"])
    accounting = accounting_record(args.accounting)
    model_lock = load_json("models/MODELS.lock.json")
    entry = model_cache_relative_dir(model_lock, args.model_id)
    if entry["revision"] != args.model_revision:
        raise SystemExit(
            f"model revision {entry['revision']} != requested {args.model_revision}"
        )
    model = load_model(model_directory(entry), device)
    windows = [
        window
        for article in split["windows"]["calibration"]
        for window in article["windows"]
    ]
    statistics = masked_mlp.calibration_group_statistics(
        model, windows, accounting["group_count"], device
    )
    rankings = masked_mlp.rankings(statistics)
    rankings["random"] = masked_mlp.random_rankings(
        len(rankings["magnitude"]), accounting["group_count"], FRONTIER_SEED
    )
    stamp = stamp_now()
    plans = {
        "schema": "selvedge/frontier/plans-v1",
        "stamp": stamp,
        "model_id": args.model_id,
        "model_revision": entry["revision"],
        "split_sha256": split_receipt["derived"]["sha256"],
        "groups": accounting["group_count"],
        "layers": len(rankings["magnitude"]),
        "seed": FRONTIER_SEED,
        "statistics": statistics,
        **rankings,
    }
    write_json_atomic(args.out, plans)
    receipt = {
        "schema": SCHEMA_STATS,
        "stamp": stamp,
        "revision": git_head(),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "model": {
            "id": args.model_id,
            "revision": entry["revision"],
            "directory": model_directory(entry),
        },
        "environment": environment(device),
        "accounting": accounting,
        "data": {
            "split": args.split,
            "split_sha256": split_receipt["derived"]["sha256"],
            "split_receipt": args.split_receipt or None,
            "calibration_documents": len(split["windows"]["calibration"]),
            "calibration_windows": len(windows),
            "calibration_tokens": sum(len(window) for window in windows),
        },
        "plans": {
            "path": args.out,
            "sha256": sha256_file(args.out),
            "seed": FRONTIER_SEED,
            "rankings": {
                "random": rankings["random"],
                "magnitude": rankings["magnitude"],
                "frequency": rankings["frequency"],
            },
        },
        "statistics": statistics,
        "notes": [
            "selectors see calibration text only",
            "random permutations use torch.Generator(seed=20261114), one per layer, in layer order",
            "frequency is the fraction of tokens with group mass above twice the uniform share",
        ],
    }
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, f"stats-{args.model_id.replace('/', '-')}-{stamp}.json")
    write_json_atomic(receipt_path, receipt)
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"stats: {args.model_id}@{entry['revision']} device={device}")
        print(f"stats: {len(windows)} calibration windows, {sum(len(w) for w in windows)} tokens")
        print(f"stats: plans {args.out} sha256={receipt['plans']['sha256']}")
        print(f"stats: receipt {receipt_path}")
    return 0


def domains_from_split(split: dict) -> dict:
    return {
        "general": [
            {"doc_id": article["doc_id"], "windows": article["windows"]}
            for article in split["windows"]["heldout_general"]
        ],
        "code": [
            {"doc_id": item["doc_id"], "windows": item["windows"]}
            for item in split["windows"]["heldout_code"]
        ],
        "math": [
            {"doc_id": item["doc_id"], "windows": item["windows"]}
            for item in split["windows"]["heldout_math"]
        ],
    }


def base_receipt(args, kind: str, accounting: dict, entry: dict, device) -> dict:
    split_receipt = split_receipt_for(args.split)
    return {
        "schema": SCHEMA_SWEEP,
        "kind": kind,
        "stamp": args.stamp or stamp_now(),
        "revision": git_head(),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "model": {
            "id": args.model_id,
            "revision": entry["revision"],
            "directory": model_directory(entry),
            "hidden": accounting["hidden"],
            "intermediate": accounting["intermediate_per_layer"],
            "layers": 28,
            "groups": accounting["group_count"],
            "loaded_dtype": "float32",
        },
        "environment": environment(device),
        "accounting": accounting,
        "data": {
            "split": args.split,
            "split_sha256": split_receipt["derived"]["sha256"],
            "split_receipt": args.split_receipt,
            "heldout_counts": {
                key: {
                    "documents": value["documents"],
                    "windows": value["windows"],
                    "tokens": value["tokens"],
                }
                for key, value in split_receipt["counts"].items()
            },
        },
        "deviations": [],
        "nonfinite": [],
        "notes": [
            "teacher is the dense float32 model with TF32 disabled",
            "metrics are teacher-forced next-token distributions, not sampled text",
            "active parameter counts are computed from the realized mask and the accounting receipt",
            "FLOPs are derived from realized channel counts, not hardware counters",
        ],
    }


def command_sweep(args) -> int:
    configure_arithmetic()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split_receipt = split_receipt_for(args.split)
    split = load_split(args.split, split_receipt["derived"]["sha256"])
    accounting = accounting_record(args.accounting)
    model_lock = load_json("models/MODELS.lock.json")
    entry = model_cache_relative_dir(model_lock, args.model_id)
    if entry["revision"] != args.model_revision:
        raise SystemExit(
            f"model revision {entry['revision']} != requested {args.model_revision}"
        )
    plans = plans_for(args.plans, accounting)
    if plans["model_id"] != args.model_id:
        raise SystemExit("plan file belongs to a different model")
    model = load_model(model_directory(entry), device)
    controller = masked_mlp.MaskController(model)
    domains = domains_from_split(split)
    if args.limit_items:
        domains = {
            domain: items[: args.limit_items] for domain, items in domains.items()
        }
    if args.arms == "uniform":
        arms = build_uniform_arms(plans, accounting, device)
    elif args.arms == "layers":
        arms = build_layer_arms(accounting, device)
    elif args.arms == "control":
        arms = build_uniform_arms(plans, accounting, device)
    elif args.arms == "greedy":
        raise SystemExit("use the `greedy` subcommand for the greedy allocations")
    else:
        raise SystemExit(f"unknown arm set {args.arms}")
    attach_masks(arms, accounting, device)
    evaluation = evaluate_with_masks(
        model, controller, arms, domains, accounting, device
    )
    controller.restore()
    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, args.arms, accounting, entry, device)
    receipt["stamp"] = stamp
    receipt["plans"] = {"path": args.plans, "sha256": sha256_file(args.plans)}
    receipt["arms"] = arms_to_receipt(arms, evaluation, list(domains))
    receipt["nonfinite"] = evaluation["nonfinite"]
    receipt["data"]["limit_items"] = args.limit_items
    path = os.path.join(args.receipt_dir, f"sweep-{args.arms}-{args.model_id.replace('/', '-')}-{stamp}.json")
    write_json_atomic(path, receipt)
    if args.json:
        print(json.dumps({"receipt": path, "arms": len(arms), "stamp": stamp}))
    else:
        print(f"sweep: {args.arms} {args.model_id} arms={len(arms)}")
        for arm in receipt["arms"]:
            general = arm["domains"]["general"]["aggregate"]
            print(
                f"  {arm['id']:<32} active={arm['active']['active_unique_elements']:>11} "
                f"kl={general.get('kl_mean', float('nan')):.5f} "
                f"top1={general.get('top1_agreement', float('nan')):.4f} "
                f"ppl={general.get('ppl_ratio', float('nan')):.4f}"
            )
        print(f"sweep: receipt {path}")
    return 0


def probe_windows(split: dict, count: int = 6, length: int = 128) -> list[list[int]]:
    windows = []
    for article in split["windows"]["calibration"][:count]:
        window = article["windows"][0][:length]
        if len(window) != length:
            raise SystemExit("calibration probe window is shorter than 128 tokens")
        windows.append(window)
    if len(windows) != count:
        raise SystemExit("not enough calibration articles for the probe")
    return windows


def probe_kl(model, controller, probe_ids, teacher_probe, masks) -> float:
    controller.set_masks(masks)
    with torch.inference_mode():
        logits = model(input_ids=probe_ids, use_cache=False).logits
        total = 0.0
        for index in range(probe_ids.shape[0]):
            metrics = masked_mlp.logits_kl_metrics(
                teacher_probe[index : index + 1],
                logits[index : index + 1],
                probe_ids[index : index + 1],
            )
            total += float(metrics["kl"].mean())
    return total / probe_ids.shape[0]


def command_greedy(args) -> int:
    configure_arithmetic()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split_receipt = split_receipt_for(args.split)
    split = load_split(args.split, split_receipt["derived"]["sha256"])
    accounting = accounting_record(args.accounting)
    model_lock = load_json("models/MODELS.lock.json")
    entry = model_cache_relative_dir(model_lock, args.model_id)
    if entry["revision"] != args.model_revision:
        raise SystemExit(
            f"model revision {entry['revision']} != requested {args.model_revision}"
        )
    plans = plans_for(args.plans, accounting)
    if plans["model_id"] != args.model_id:
        raise SystemExit("plan file belongs to a different model")
    model = load_model(model_directory(entry), device)
    controller = masked_mlp.MaskController(model)
    groups = accounting["group_count"]
    layers = 28
    probe_ids = torch.tensor(
        probe_windows(split), dtype=torch.long, device=device
    )
    controller.suspend()
    with torch.inference_mode():
        teacher_probe = model(input_ids=probe_ids, use_cache=False).logits
    controller.resume()
    log = []
    start_utc = utc_now()

    def masks_for(selection):
        return make_masks(selection, groups, accounting["intermediate_per_layer"], device)

    # --- greedy addition from uniform five of the magnitude ranking ---------
    selection5 = [plans["magnitude"][layer][:5] for layer in range(layers)]
    base5 = probe_kl(model, controller, probe_ids, teacher_probe, masks_for(selection5))
    current5 = base5
    per_group = 3 * accounting["hidden"] * (
        accounting["intermediate_per_layer"] // groups
    )
    max_groups = accounting["max_compliant_ffn_elements"] // per_group
    current = sum(len(selected) for selected in selection5)
    while current < max_groups:
        candidates = []
        for layer in range(layers):
            for group in range(groups):
                if group in selection5[layer]:
                    continue
                trial = [list(selected) for selected in selection5]
                trial[layer] = sorted(trial[layer] + [group])
                kl = probe_kl(
                    model, controller, probe_ids, teacher_probe, masks_for(trial)
                )
                candidates.append(
                    {"step": current, "layer": layer, "group": group, "kl": kl}
                )
        best = min(candidates, key=lambda item: (item["kl"], item["layer"], item["group"]))
        selection5[best["layer"]] = sorted(
            selection5[best["layer"]] + [best["group"]]
        )
        current += 1
        current5 = best["kl"]
        log.append(
            {
                "allocation": "greedy-uniform5",
                "step": current,
                "candidates": len(candidates),
                "selected": best,
                "candidate_kl": candidates,
            }
        )
        print(
            f"greedy-uniform5: added layer {best['layer']} group {best['group']} "
            f"kl={best['kl']:.6f} groups={current}"
        )

    # --- greedy pair swaps from uniform four --------------------------------
    selection4 = [plans["magnitude"][layer][:4] for layer in range(layers)]
    base4 = probe_kl(model, controller, probe_ids, teacher_probe, masks_for(selection4))
    current4 = base4
    swaps = []
    for iteration in range(3):
        worst = [selection4[layer][-1] if selection4[layer] else None for layer in range(layers)]
        best_add = [
            next(
                (group for group in plans["magnitude"][layer] if group not in selection4[layer]),
                None,
            )
            for layer in range(layers)
        ]
        candidates = []
        for remove_layer in range(layers):
            if worst[remove_layer] is None:
                continue
            for add_layer in range(layers):
                if add_layer == remove_layer or best_add[add_layer] is None:
                    continue
                trial = [list(selected) for selected in selection4]
                trial[remove_layer].remove(worst[remove_layer])
                trial[add_layer] = sorted(trial[add_layer] + [best_add[add_layer]])
                kl = probe_kl(
                    model, controller, probe_ids, teacher_probe, masks_for(trial)
                )
                candidates.append(
                    {
                        "iteration": iteration,
                        "remove_layer": remove_layer,
                        "remove_group": worst[remove_layer],
                        "add_layer": add_layer,
                        "add_group": best_add[add_layer],
                        "kl": kl,
                    }
                )
        best = min(
            candidates,
            key=lambda item: (
                item["kl"],
                item["remove_layer"],
                item["add_layer"],
            ),
        )
        if best["kl"] < current4 - 1e-9:
            selection4[best["remove_layer"]].remove(best["remove_group"])
            selection4[best["add_layer"]] = sorted(
                selection4[best["add_layer"]] + [best["add_group"]]
            )
            current4 = best["kl"]
            accepted = True
        else:
            accepted = False
        swaps.append(
            {
                "iteration": iteration,
                "candidates": len(candidates),
                "accepted": accepted,
                "best": best,
                "candidate_kl": candidates,
            }
        )
        print(
            f"greedy-swap4: iteration {iteration} candidates={len(candidates)} "
            f"best={best['kl']:.6f} accepted={accepted}"
        )
        if not accepted:
            break

    # --- evaluate the allocations and their matched uniform baselines ------
    arms = [
        arm_record(
            "greedy-uniform5",
            "greedy",
            selection5,
            accounting,
            device,
            {"allocation": "greedy addition from uniform magnitude k=5"},
        ),
        arm_record(
            "greedy-swap4",
            "greedy",
            selection4,
            accounting,
            device,
            {"allocation": "greedy pair swaps from uniform magnitude k=4"},
        ),
        arm_record(
            "uniform-magnitude-k5",
            "uniform",
            [plans["magnitude"][layer][:5] for layer in range(layers)],
            accounting,
            device,
            {"selector": "magnitude", "k": 5},
        ),
        arm_record(
            "uniform-magnitude-k4",
            "uniform",
            [plans["magnitude"][layer][:4] for layer in range(layers)],
            accounting,
            device,
            {"selector": "magnitude", "k": 4},
        ),
    ]
    attach_masks(arms, accounting, device)
    domains = domains_from_split(split)
    if args.limit_items:
        domains = {
            domain: items[: args.limit_items] for domain, items in domains.items()
        }
    evaluation = evaluate_with_masks(
        model, controller, arms, domains, accounting, device
    )
    controller.restore()
    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, "greedy", accounting, entry, device)
    receipt["stamp"] = stamp
    receipt["plans"] = {"path": args.plans, "sha256": sha256_file(args.plans)}
    receipt["arms"] = arms_to_receipt(arms, evaluation, list(domains))
    receipt["nonfinite"] = evaluation["nonfinite"]
    receipt["data"]["limit_items"] = args.limit_items
    receipt["greedy"] = {
        "probe": {
            "articles": 6,
            "positions_per_article": 128,
            "total_positions": 768,
            "kind": (
                "exact teacher KL on the calibration probe; a calibration search, "
                "not a per-token runtime process"
            ),
        },
        "start_kl": {"greedy-uniform5": base5, "greedy-swap4": base4},
        "final_kl": {
            "greedy-uniform5": current5,
            "greedy-swap4": current4,
        },
        "max_whole_groups_at_ceiling": int(max_groups),
        "log": log,
        "swaps": swaps,
        "started_utc": start_utc,
        "finished_utc": utc_now(),
    }
    path = os.path.join(args.receipt_dir, f"sweep-greedy-{args.model_id.replace('/', '-')}-{stamp}.json")
    write_json_atomic(path, receipt)
    for arm in receipt["arms"]:
        general = arm["domains"]["general"]["aggregate"]
        print(
            f"  {arm['id']:<24} active={arm['active']['active_unique_elements']:>11} "
            f"kl={general['kl_mean']:.6f} top1={general['top1_agreement']:.4f} "
            f"ppl={general['ppl_ratio']:.4f}"
        )
    print(f"greedy: receipt {path}")
    return 0


def capture_layer_input(model, layer: int, window: list[int], device, rows: int):
    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = (
            inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:rows].contiguous()
        )

    handle = model.model.layers[layer].mlp.register_forward_hook(hook)
    try:
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    if "x" not in captured:
        raise SystemExit("failed to capture the layer input")
    return captured["x"]


def command_exactness(args) -> int:
    configure_arithmetic()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split_receipt = split_receipt_for(args.split)
    split = load_split(args.split, split_receipt["derived"]["sha256"])
    accounting = accounting_record(args.accounting)
    model_lock = load_json("models/MODELS.lock.json")
    entry = model_cache_relative_dir(model_lock, args.model_id)
    if entry["revision"] != args.model_revision:
        raise SystemExit(
            f"model revision {entry['revision']} != requested {args.model_revision}"
        )
    model = load_model(model_directory(entry), device)
    controller = masked_mlp.MaskController(model)
    domains = domains_from_split(split)
    windows = [item["windows"][0] for item in domains["general"][:2]]
    if args.limit_items:
        windows = windows[: args.limit_items]
    checks = []
    max_abs_all = 0.0
    kl_all = []
    bitwise_all = True
    for window in windows:
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        controller.suspend()
        with torch.inference_mode():
            teacher = model(input_ids=input_ids, use_cache=False).logits
        controller.resume()
        with torch.inference_mode():
            controller.set_dense()
            masked = model(input_ids=input_ids, use_cache=False).logits
        difference = (teacher - masked).abs()
        max_abs = float(difference.max())
        max_abs_all = max(max_abs_all, max_abs)
        bitwise = bool(torch.equal(teacher, masked))
        bitwise_all = bitwise_all and bitwise
        metrics = masked_mlp.logits_kl_metrics(teacher, masked, input_ids)
        mean_kl = float(metrics["kl"].mean())
        kl_all.append(mean_kl)
        checks.append(
            {
                "tokens": len(window),
                "max_abs_logit_difference": max_abs,
                "mean_kl": mean_kl,
                "bitwise_equal": bitwise,
                "finite": bool(torch.isfinite(masked).all()),
            }
        )
        del teacher, masked

    # Grouped reference versus the fast masked path on one real layer.
    layer = args.layer
    rows = args.rows
    selected = [int(value) for value in args.groups.split(",")]
    captured = capture_layer_input(model, layer, windows[0], device, rows)
    mlp = controller.originals[layer]
    mask = masked_mlp.selected_mask(
        accounting["intermediate_per_layer"],
        accounting["group_count"],
        selected,
        device=device,
    )
    with torch.inference_mode():
        activation = masked_mlp.dense_channels(mlp, captured)
        fast = mlp.down_proj(activation * mask)
        reference = masked_mlp.grouped_forward(
            mlp, captured, accounting["group_count"], selected
        )
    fast_diff = (fast - reference).abs()
    fast_check = {
        "layer": layer,
        "rows": rows,
        "selected_groups": selected,
        "max_abs_difference": float(fast_diff.max()),
        "mean_abs_difference": float(fast_diff.mean()),
        "tolerance": 1e-3,
        "passed": bool(float(fast_diff.max()) <= 1e-3),
    }

    # Input and Python reference output for the independent Rust cross-check.
    os.makedirs(args.artifact_dir, exist_ok=True)
    input_path = os.path.join(args.artifact_dir, f"rust-input-layer{layer}.f32")
    python_output = os.path.join(args.artifact_dir, f"python-output-layer{layer}.f32")
    captured_cpu = captured.detach().to("cpu", torch.float32).contiguous()
    reference_cpu = reference.detach().to("cpu", torch.float32).contiguous()
    captured_cpu.numpy().tofile(input_path)
    reference_cpu.numpy().tofile(python_output)
    request = {
        "schema": "selvedge/frontier/rust-cross-check-request/v1",
        "bundle": args.bundle,
        "layer": layer,
        "groups": selected,
        "rows": int(captured.shape[0]),
        "hidden": int(captured.shape[1]),
        "input": input_path,
        "python_output": python_output,
        "input_sha256": sha256_file(input_path),
        "python_output_sha256": sha256_file(python_output),
        "command_hint": (
            "scripts/court rust -- cargo run --release -p selvedge-cli -- court b-real "
            f"--bundle {args.bundle} --layer {layer} --groups "
            f"{','.join(str(value) for value in selected)} --input {input_path} "
            f"--rows {int(captured.shape[0])} --hidden {int(captured.shape[1])} "
            f"--out-raw {os.path.join(args.artifact_dir, f'rust-output-layer{layer}.f32')}"
        ),
    }
    request_path = os.path.join(args.artifact_dir, f"rust-request-layer{layer}.json")
    write_json_atomic(request_path, request)
    controller.restore()

    stamp = args.stamp or stamp_now()
    receipt = base_receipt(args, "exactness", accounting, entry, device)
    receipt["stamp"] = stamp
    receipt["checks"] = {
        "dense_vs_masked_all": {
            "windows": len(windows),
            "max_abs_logit_difference": max_abs_all,
            "mean_kl": float(np.mean(kl_all)),
            "bitwise_equal": bitwise_all,
            "tolerance_max_abs": 0.01,
            "tolerance_mean_kl": 1e-4,
            "passed": bool(max_abs_all <= 0.01 and float(np.mean(kl_all)) <= 1e-4),
        },
        "fast_path_vs_grouped_reference": fast_check,
    }
    receipt["rust_cross_check"] = {
        "request": request_path,
        "request_sha256": sha256_file(request_path),
        "input": input_path,
        "input_sha256": request["input_sha256"],
        "python_output": python_output,
        "python_output_sha256": request["python_output_sha256"],
        "status": "pending: run the Rust court b-real command and then `cross-check`",
        "command_hint": request["command_hint"],
    }
    path = os.path.join(args.receipt_dir, f"exactness-{args.model_id.replace('/', '-')}-{stamp}.json")
    write_json_atomic(path, receipt)
    print(
        f"exactness: dense-vs-masked-all max_abs={max_abs_all:.6g} "
        f"mean_kl={float(np.mean(kl_all)):.6g} bitwise={bitwise_all} "
        f"passed={receipt['checks']['dense_vs_masked_all']['passed']}"
    )
    print(
        f"exactness: fast-vs-grouped layer {layer} max_abs={fast_check['max_abs_difference']:.6g} "
        f"passed={fast_check['passed']}"
    )
    print(f"exactness: rust request {request_path}")
    print(f"exactness: receipt {path}")
    return 0 if receipt["checks"]["dense_vs_masked_all"]["passed"] else 1


def command_cross_check(args) -> int:
    rust = load_json(args.rust_report)
    if os.path.abspath(rust.get("output", "")) != os.path.abspath(args.rust_output):
        raise SystemExit("rust report names a different output file")
    if args.request:
        request = load_json(args.request)
        if rust.get("input_sha256") != request["input_sha256"]:
            raise SystemExit("rust input hash does not match the recorded request")
        if rust.get("layer") != request["layer"]:
            raise SystemExit("rust layer does not match the recorded request")
        if list(rust.get("selected_groups", [])) != list(request["groups"]):
            raise SystemExit("rust selected groups do not match the recorded request")
        if os.path.abspath(request["python_output"]) != os.path.abspath(args.python_output):
            raise SystemExit("request names a different python output")
        if request["python_output_sha256"] != sha256_file(args.python_output):
            raise SystemExit("python output hash does not match the recorded request")
    python_bytes = np.fromfile(args.python_output, dtype="<f4")
    rust_bytes = np.fromfile(args.rust_output, dtype="<f4")
    if python_bytes.shape != rust_bytes.shape:
        raise SystemExit(
            f"shape mismatch: python {python_bytes.shape}, rust {rust_bytes.shape}"
        )
    if python_bytes.size != args.rows * args.hidden:
        raise SystemExit("output size does not match rows*hidden")
    difference = np.abs(python_bytes.astype(np.float64) - rust_bytes.astype(np.float64))
    limit = 1e-3 * np.maximum(1.0, np.abs(rust_bytes.astype(np.float64)))
    violations = int((difference > limit).sum())
    report = {
        "schema": SCHEMA_CROSS_CHECK,
        "stamp": args.stamp or stamp_now(),
        "revision": git_head(),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "rust_report": args.rust_report,
        "rust_report_sha256": sha256_file(args.rust_report),
        "python_output": args.python_output,
        "python_output_sha256": sha256_file(args.python_output),
        "rust_output": args.rust_output,
        "rust_output_sha256": sha256_file(args.rust_output),
        "elements": int(python_bytes.size),
        "tolerance": "|a-b| <= 1e-3 * max(1, |b|), both f32",
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "violations": violations,
        "bitwise_equal": bool(np.array_equal(python_bytes, rust_bytes)),
        "passed": violations == 0,
        "rust": {
            "schema": rust.get("schema"),
            "layer": rust.get("layer"),
            "selected_groups": rust.get("selected_groups"),
            "rows": rust.get("rows"),
            "hidden": rust.get("hidden"),
            "input_sha256": rust.get("input_sha256"),
            "output_sha256": rust.get("output_sha256"),
        },
    }
    path = os.path.join(args.receipt_dir, f"cross-check-{report['stamp']}.json")
    write_json_atomic(path, report)
    print(
        f"cross-check: elements={report['elements']} max_abs={report['max_abs_difference']:.6g} "
        f"violations={violations} bitwise={report['bitwise_equal']} passed={report['passed']}"
    )
    print(f"cross-check: receipt {path}")
    return 0 if report["passed"] else 1


def bootstrap_indices(count: int, resamples: int, seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(0, count, size=(resamples, count))


def percentile_ci(values: np.ndarray) -> dict:
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "resamples": int(values.shape[0]),
    }


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict:
    return percentile_ci(np.mean(values[indices], axis=1))


def bootstrap_ppl(items: list[dict], indices: np.ndarray) -> dict:
    teacher = np.array([item["nll_teacher_sum"] for item in items])
    masked = np.array([item["nll_masked_sum"] for item in items])
    positions = np.array([item["nll_positions"] for item in items], dtype=np.float64)
    sampled_teacher = teacher[indices].sum(axis=1)
    sampled_masked = masked[indices].sum(axis=1)
    sampled_positions = positions[indices].sum(axis=1)
    ratios = np.exp((sampled_masked - sampled_teacher) / sampled_positions)
    result = percentile_ci(ratios)
    result["point"] = float(
        np.exp((masked.sum() - teacher.sum()) / positions.sum())
    ) if positions.sum() else 1.0
    return result


def arm_domain_metrics(items: list[dict], indices: np.ndarray) -> dict:
    kl = np.array([item["kl_mean"] for item in items], dtype=np.float64)
    top1 = np.array([item["top1_agreement"] for item in items], dtype=np.float64)
    p95 = np.array([item["kl_p95"] for item in items], dtype=np.float64)
    return {
        "items": len(items),
        "positions": int(sum(item["positions"] for item in items)),
        "kl_mean": float(kl.mean()),
        "kl_mean_ci": bootstrap_mean(kl, indices),
        "kl_p95_mean": float(p95.mean()),
        "kl_p95_ci": bootstrap_mean(p95, indices),
        "top1_agreement": float(top1.mean()),
        "top1_ci": bootstrap_mean(top1, indices),
        "ppl_ratio": bootstrap_ppl(items, indices),
        "teacher_entropy_mean": float(
            np.mean([item["teacher_entropy_mean"] for item in items])
        ),
    }


def command_analyze(args) -> int:
    sweep_paths = sorted(glob.glob(args.sweep_glob))
    if not sweep_paths:
        raise SystemExit(f"no sweep receipts match {args.sweep_glob}")
    receipts = [load_json(path) for path in sweep_paths]
    exactness = load_json(args.exactness) if args.exactness else None
    cross_check = load_json(args.cross_check) if args.cross_check else None
    arms = {}
    duplicates = []
    for receipt in receipts:
        for arm in receipt["arms"]:
            key = (receipt["model"]["id"], arm["id"])
            if key in arms:
                previous = arms[key]["record"]
                old = previous["domains"]["general"]["aggregate"]["kl_mean"]
                new = arm["domains"]["general"]["aggregate"]["kl_mean"]
                if abs(old - new) > 1e-9:
                    raise SystemExit(
                        f"duplicate arm {key} differs between receipts: {old} vs {new}"
                    )
                duplicates.append(
                    {"model": key[0], "arm": key[1], "kept": arms[key]["receipt"], "also_in": receipt["stamp"]}
                )
                continue
            arms[key] = {
                "model": receipt["model"],
                "receipt": receipt["stamp"],
                "record": arm,
            }
    thresholds = {
        "acceptable": {
            "kl_mean": 0.05,
            "ppl_ratio": 1.02,
            "top1_agreement": 0.95,
        },
        "usable": {"kl_mean": 0.25, "ppl_ratio": 1.10, "top1_agreement": 0.85},
    }
    out_arms = []
    for (model_id, arm_id), entry in sorted(arms.items()):
        record = entry["record"]
        domains = {}
        for domain, payload in record["domains"].items():
            items = payload["items"]
            if not items:
                continue
            indices = bootstrap_indices(len(items), args.resamples, FRONTIER_SEED)
            domains[domain] = arm_domain_metrics(items, indices)
        out_arms.append(
            {
                "model": model_id,
                "model_revision": entry["model"]["revision"],
                "id": arm_id,
                "kind": record["kind"],
                "selector": record.get("selector"),
                "k": record.get("k"),
                "compliant": record["compliant"],
                "active": record["active"],
                "groups_per_layer": record["groups_per_layer"],
                "selection_sha256": record["selection_sha256"],
                "timing": record["timing"],
                "domains": domains,
            }
        )

    def find(model_id: str, arm_id: str) -> dict | None:
        return next(
            (
                arm
                for arm in out_arms
                if arm["model"] == model_id and arm["id"] == arm_id
            ),
            None,
        )

    # --- verdict on the registered primary domain (wikitext held-out) -----
    teacher17 = "Qwen/Qwen3-1.7B"
    compliant_general = [
        arm
        for arm in out_arms
        if arm["model"] == teacher17
        and arm["compliant"]
        and arm["kind"] not in ("reference", "floor")
        and "general" in arm["domains"]
    ]
    best = min(compliant_general, key=lambda arm: arm["domains"]["general"]["kl_mean"]) if compliant_general else None
    best_verdict = None
    if best:
        general = best["domains"]["general"]
        acceptable = (
            general["kl_mean_ci"]["ci_high"] <= thresholds["acceptable"]["kl_mean"]
            and general["ppl_ratio"]["ci_high"] <= thresholds["acceptable"]["ppl_ratio"]
            and general["top1_ci"]["ci_low"] >= thresholds["acceptable"]["top1_agreement"]
        )
        usable = (
            general["kl_mean_ci"]["ci_high"] <= thresholds["usable"]["kl_mean"]
            and general["ppl_ratio"]["ci_high"] <= thresholds["usable"]["ppl_ratio"]
            and general["top1_ci"]["ci_low"] >= thresholds["usable"]["top1_agreement"]
        )
        best_verdict = {
            "arm": best["id"],
            "active_unique_elements": best["active"]["active_unique_elements"],
            "kl_mean": general["kl_mean"],
            "kl_ci_high": general["kl_mean_ci"]["ci_high"],
            "ppl_ratio": general["ppl_ratio"]["point"],
            "ppl_ci_high": general["ppl_ratio"]["ci_high"],
            "top1_agreement": general["top1_agreement"],
            "top1_ci_low": general["top1_ci"]["ci_low"],
            "acceptable": bool(acceptable),
            "usable": bool(usable),
            "h1_supported": bool(acceptable),
        }

    item_index = {}
    for receipt in receipts:
        for arm in receipt["arms"]:
            key = (receipt["model"]["id"], arm["id"])
            if key not in item_index:
                item_index[key] = arm["domains"]
    comparisons = []
    for greedy_id, uniform_id in (
        ("greedy-swap4", "uniform-magnitude-k4"),
        ("greedy-uniform5", "uniform-magnitude-k5"),
    ):
        if (
            (teacher17, greedy_id) not in item_index
            or (teacher17, uniform_id) not in item_index
        ):
            continue
        entry = {"greedy": greedy_id, "uniform": uniform_id, "domains": {}}
        for domain in ("general", "code", "math"):
            greedy_items = {
                item["doc_id"]: item
                for item in item_index[(teacher17, greedy_id)][domain]["items"]
            }
            uniform_items = {
                item["doc_id"]: item
                for item in item_index[(teacher17, uniform_id)][domain]["items"]
            }
            shared = sorted(set(greedy_items) & set(uniform_items))
            if len(shared) < 3:
                continue
            greedy_kl = np.array([greedy_items[key]["kl_mean"] for key in shared])
            uniform_kl = np.array([uniform_items[key]["kl_mean"] for key in shared])
            indices = bootstrap_indices(len(shared), args.resamples, FRONTIER_SEED)
            absolute = uniform_kl - greedy_kl
            relative = np.where(
                uniform_kl > 0, absolute / uniform_kl, 0.0
            )
            entry["domains"][domain] = {
                "items": len(shared),
                "uniform_kl": float(uniform_kl.mean()),
                "greedy_kl": float(greedy_kl.mean()),
                "absolute_reduction": bootstrap_mean(absolute, indices),
                "relative_reduction": bootstrap_mean(relative, indices),
            }
        comparisons.append(entry)

    transfer = []
    for entry in comparisons:
        general = entry["domains"].get("general")
        if not general:
            continue
        base = general["relative_reduction"]["mean"]
        for domain in ("code", "math"):
            value = entry["domains"].get(domain)
            if not value:
                continue
            reduction = value["relative_reduction"]["mean"]
            retention = (reduction / base) if base and base > 0 else None
            transfer.append(
                {
                    "greedy": entry["greedy"],
                    "uniform": entry["uniform"],
                    "domain": domain,
                    "relative_reduction": value["relative_reduction"],
                    "retention_vs_general": retention,
                    "passes_k4": bool(
                        reduction > 0
                        and value["relative_reduction"]["ci_low"] > 0
                        and retention is not None
                        and retention >= 0.25
                    ),
                }
            )

    stamp = args.stamp or stamp_now()
    frontier = {
        "schema": SCHEMA_FRONTIER,
        "stamp": stamp,
        "revision": git_head(),
        "preregistration_commit": args.preregistration_commit,
        "created_utc": utc_now(),
        "seed": FRONTIER_SEED,
        "resamples": args.resamples,
        "thresholds": thresholds,
        "sweep_receipts": [
            {"path": path, "sha256": sha256_file(path), "stamp": receipt["stamp"]}
            for path, receipt in zip(sweep_paths, receipts)
        ],
        "exactness": (
            {
                "path": args.exactness,
                "sha256": sha256_file(args.exactness),
                "checks": exactness["checks"],
                "rust_cross_check": exactness["rust_cross_check"],
            }
            if exactness
            else None
        ),
        "rust_cross_check": (
            {
                "path": args.cross_check,
                "sha256": sha256_file(args.cross_check),
                "passed": cross_check["passed"],
                "bitwise_equal": cross_check["bitwise_equal"],
                "max_abs_difference": cross_check["max_abs_difference"],
                "violations": cross_check["violations"],
                "elements": cross_check["elements"],
            }
            if cross_check
            else None
        ),
        "duplicate_arms_checked": duplicates,
        "arms": out_arms,
        "comparisons": comparisons,
        "transfer": transfer,
        "verdict": {
            "best_compliant_arm": best_verdict,
            "h1_supported": bool(best_verdict and best_verdict["h1_supported"]),
            "k3_non_uniform_value": None,
            "k4_transfer": transfer,
        },
    }
    if comparisons:
        swap = next(
            (entry for entry in comparisons if entry["greedy"] == "greedy-swap4"),
            None,
        )
        if swap and "general" in swap["domains"]:
            reduction = swap["domains"]["general"]["relative_reduction"]
            frontier["verdict"]["k3_non_uniform_value"] = {
                "relative_reduction": reduction,
                "passes_k3": bool(
                    reduction["mean"] >= 0.15 and reduction["ci_low"] > 0
                ),
            }

    # Stable keys for the generated-number manifest.
    def compact(arm_id: str, model: str = teacher17) -> dict:
        arm = find(model, arm_id)
        if not arm or "general" not in arm["domains"]:
            return {}
        general = arm["domains"]["general"]
        return {
            "active": arm["active"]["active_unique_elements"],
            "kl": general["kl_mean"],
            "kl_ci_low": general["kl_mean_ci"]["ci_low"],
            "kl_ci_high": general["kl_mean_ci"]["ci_high"],
            "ppl": general["ppl_ratio"]["point"],
            "top1": general["top1_agreement"],
            "entropy": general["teacher_entropy_mean"],
        }

    summary = {
        "ceiling": 1000000000,
        "arms": len(out_arms),
        "best_compliant_arm": (best_verdict or {}).get("arm"),
        "best_compliant": (best_verdict or {}),
        "magnitude_k5": compact("uniform-magnitude-k5"),
        "magnitude_k10": compact("uniform-magnitude-k10"),
        "magnitude_k12": compact("uniform-magnitude-k12"),
        "magnitude_k15": compact("uniform-magnitude-k15"),
        "magnitude_k16": compact("uniform-magnitude-k16"),
        "greedy_swap4": compact("greedy-swap4"),
        "greedy_uniform5": compact("greedy-uniform5"),
        "layer_omit_0": compact("layer-omit-0"),
        "layer_omit_13": compact("layer-omit-13"),
        "layer_omit_first_20": compact("layer-omit-first-20"),
        "control_k5": compact("uniform-magnitude-k5", "Qwen/Qwen3-0.6B"),
        "control_dense": compact("uniform-magnitude-k16", "Qwen/Qwen3-0.6B"),
        "greedy_add_code_retention": next(
            (
                entry["retention_vs_general"]
                for entry in transfer
                if entry["greedy"] == "greedy-uniform5" and entry["domain"] == "code"
            ),
            None,
        ),
        "greedy_add_math_retention": next(
            (
                entry["retention_vs_general"]
                for entry in transfer
                if entry["greedy"] == "greedy-uniform5" and entry["domain"] == "math"
            ),
            None,
        ),
        "exactness_max_abs": (
            exactness["checks"]["dense_vs_masked_all"]["max_abs_logit_difference"]
            if exactness
            else None
        ),
        "exactness_mean_kl": (
            exactness["checks"]["dense_vs_masked_all"]["mean_kl"]
            if exactness
            else None
        ),
        "cross_check_max_abs": (
            cross_check["max_abs_difference"] if cross_check else None
        ),
        "cross_check_elements": (
            cross_check["elements"] if cross_check else None
        ),
        "cross_check_violations": (
            cross_check["violations"] if cross_check else None
        ),
        "comparisons": comparisons,
        "transfer": transfer,
        "greedy_swap_vs_k4": next(
            (entry for entry in comparisons if entry["greedy"] == "greedy-swap4"),
            {},
        ),
        "greedy_add_vs_k5": next(
            (entry for entry in comparisons if entry["greedy"] == "greedy-uniform5"),
            {},
        ),
    }
    frontier["summary"] = summary
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "frontier.json")
    write_json_atomic(json_path, frontier)
    svg_path = os.path.join(args.out_dir, "frontier.svg")
    write_svg(frontier, svg_path)
    table_path = os.path.join(args.out_dir, "frontier.md")
    write_markdown_table(frontier, table_path)
    if args.frontier_md and os.path.exists(args.frontier_md):
        update_frontier_doc(args.frontier_md, table_path)
    print(f"analyze: {len(out_arms)} arms from {len(receipts)} receipts")
    if best_verdict:
        print(
            f"analyze: best compliant {best_verdict['arm']} "
            f"active={best_verdict['active_unique_elements']} "
            f"kl={best_verdict['kl_mean']:.6f} "
            f"ppl={best_verdict['ppl_ratio']:.4f} "
            f"top1={best_verdict['top1_agreement']:.4f} "
            f"h1={best_verdict['h1_supported']}"
        )
    print(f"analyze: frontier {json_path}")
    print(f"analyze: plot {svg_path}")
    print(f"analyze: table {table_path}")
    return 0


def series_for(frontier: dict, model: str, kind: str, selector: str | None) -> list[dict]:
    points = []
    for arm in frontier["arms"]:
        if arm["model"] != model or arm["kind"] != kind:
            continue
        if selector is not None and arm.get("selector") != selector:
            continue
        general = arm["domains"].get("general")
        if not general:
            continue
        points.append(
            {
                "id": arm["id"],
                "x": arm["active"]["active_unique_elements"],
                "y": general["kl_mean"],
                "top1": general["top1_agreement"],
                "ppl": general["ppl_ratio"]["point"],
                "compliant": arm["compliant"],
            }
        )
    points.sort(key=lambda point: point["x"])
    return points


def write_svg(frontier: dict, path: str) -> None:
    width, height = 900, 640
    left, right = 90, 30
    panels = [
        ("Qwen/Qwen3-1.7B", 40, 290),
        ("Qwen/Qwen3-0.6B", 350, 290),
    ]
    colors = {
        "random": "#888888",
        "magnitude": "#1f77b4",
        "frequency": "#d62728",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px} .axis{stroke:#333;stroke-width:1} '
        '.grid{stroke:#ddd;stroke-width:1} .lbl{font-size:12px;font-weight:bold}</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="20" class="lbl">Teacher KL vs realized active unique parameters '
        f'(dense f32 teacher, wikitext-2 held-out)</text>',
    ]
    max_x = 1.85e9
    max_y = 15.0
    for model, top, panel_height in panels:
        plot_left = left
        plot_width = width - left - right
        base = top + panel_height - 40
        lines.append(
            f'<text x="{plot_left}" y="{top - 6}" class="lbl">{model}</text>'
        )
        lines.append(
            f'<line x1="{plot_left}" y1="{base}" x2="{plot_left + plot_width}" y2="{base}" class="axis"/>'
        )
        lines.append(
            f'<line x1="{plot_left}" y1="{top + 10}" x2="{plot_left}" y2="{base}" class="axis"/>'
        )
        for tick in range(0, 7):
            y = top + 10 + (panel_height - 50) * tick / 6.0
            value = max_y * (6 - tick) / 6.0
            lines.append(
                f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" class="grid"/>'
            )
            lines.append(
                f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>'
            )
        for tick in range(0, 5):
            x = plot_left + plot_width * tick / 4.0
            value = max_x * tick / 4.0 / 1e9
            lines.append(
                f'<line x1="{x:.1f}" y1="{top + 10}" x2="{x:.1f}" y2="{base}" class="grid"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{base + 16}" text-anchor="middle">{value:.1f}B</text>'
            )
        # ceiling line
        ceiling_x = plot_left + plot_width * (1e9 / max_x)
        lines.append(
            f'<line x1="{ceiling_x:.1f}" y1="{top + 10}" x2="{ceiling_x:.1f}" y2="{base}" '
            'stroke="#ff7f0e" stroke-dasharray="4 4"/>'
        )
        lines.append(
            f'<text x="{ceiling_x + 4:.1f}" y="{top + 22}" fill="#ff7f0e">1e9</text>'
        )

        def project(x: float, y: float) -> tuple[float, float]:
            px = plot_left + plot_width * min(x / max_x, 1.0)
            py = base - (panel_height - 50) * min(y / max_y, 1.0)
            return px, py

        for selector in ("random", "magnitude", "frequency"):
            points = series_for(frontier, model, "uniform", selector)
            if not points:
                continue
            coords = " ".join(
                f"{project(point['x'], point['y'])[0]:.1f},{project(point['x'], point['y'])[1]:.1f}"
                for point in points
            )
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="{colors[selector]}" stroke-width="2"/>'
            )
            for point in points:
                px, py = project(point["x"], point["y"])
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{colors[selector]}"/>'
                )
        for arm in frontier["arms"]:
            if arm["model"] != model or arm["kind"] not in ("greedy",):
                continue
            general = arm["domains"].get("general")
            if not general:
                continue
            px, py = project(
                arm["active"]["active_unique_elements"], general["kl_mean"]
            )
            lines.append(
                f'<text x="{px + 5:.1f}" y="{py - 4:.1f}" fill="#2ca02c">+ {arm["id"]}</text>'
            )
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="none" stroke="#2ca02c" stroke-width="2"/>'
            )
        for arm in frontier["arms"]:
            if arm["model"] != model or arm["kind"] != "layer-omission":
                continue
            general = arm["domains"].get("general")
            if not general or ("first-" not in arm["id"] and "last-" not in arm["id"]):
                continue
            px, py = project(
                arm["active"]["active_unique_elements"], general["kl_mean"]
            )
            lines.append(
                f'<rect x="{px - 3:.1f}" y="{py - 3:.1f}" width="6" height="6" fill="#9467bd"/>'
            )
        legend_y = top + 18
        for index, selector in enumerate(("random", "magnitude", "frequency")):
            lines.append(
                f'<line x1="{plot_left + plot_width - 190}" y1="{legend_y + index * 14}" '
                f'x2="{plot_left + plot_width - 170}" y2="{legend_y + index * 14}" '
                f'stroke="{colors[selector]}" stroke-width="2"/>'
            )
            lines.append(
                f'<text x="{plot_left + plot_width - 165}" y="{legend_y + index * 14 + 4}">'
                f'{selector}</text>'
            )
    lines.append(
        f'<text x="{left}" y="{height - 12}">Generated by oracle/frontier_sweep.py analyze from '
        f'results/frontier/frontier.json. Dashed orange: 1e9 ceiling.</text>'
    )
    lines.append("</svg>")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def markdown_table_lines(frontier: dict) -> list[str]:
    lines = [
        "| model | arm | active unique | compliant | KL mean | KL 95% CI | PPL ratio | top-1 | top-1 95% CI |",
        "| --- | --- | ---: | :---: | ---: | --- | ---: | ---: | --- |",
    ]
    for arm in frontier["arms"]:
        general = arm["domains"].get("general")
        if not general:
            continue
        lines.append(
            "| {model} | {id} | {active} | {compliant} | {kl:.6f} | [{lo:.6f}, {hi:.6f}] | "
            "{ppl:.5f} | {top1:.4f} | [{tlo:.4f}, {thi:.4f}] |".format(
                model=arm["model"],
                id=arm["id"],
                active=arm["active"]["active_unique_elements"],
                compliant="yes" if arm["compliant"] else "no",
                kl=general["kl_mean"],
                lo=general["kl_mean_ci"]["ci_low"],
                hi=general["kl_mean_ci"]["ci_high"],
                ppl=general["ppl_ratio"]["point"],
                top1=general["top1_agreement"],
                tlo=general["top1_ci"]["ci_low"],
                thi=general["top1_ci"]["ci_high"],
            )
        )
    return lines


def update_frontier_doc(path: str, table_path: str) -> None:
    """Replace the table region in the human-readable report."""
    begin = "<!-- BEGIN FRONTIER TABLE -->"
    end = "<!-- END FRONTIER TABLE -->"
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if begin not in text or end not in text:
        raise SystemExit(f"{path} has no frontier table region")
    with open(table_path, "r", encoding="utf-8") as handle:
        table = handle.read().rstrip("\n")
    # Drop the generated file's own title and preamble (first blank line).
    table_lines = table.splitlines()
    start = next(
        (index for index, line in enumerate(table_lines) if line.startswith("| model |")),
        0,
    )
    body = "\n".join(table_lines[start:])
    before, _, remainder = text.partition(begin)
    _, _, after = remainder.partition(end)
    updated = f"{before}{begin}\n{body}\n{end}{after}"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)


def write_markdown_table(frontier: dict, path: str) -> None:
    lines = [
        "# M1 frontier table (generated)",
        "",
        "Generated by `oracle/frontier_sweep.py analyze` from "
        "`results/frontier/frontier.json`. Do not edit. The registered contract is "
        "`docs/EXPERIMENT-CONTRACT.md`.",
        "",
        f"Stamp: `{frontier['stamp']}`. Revision: `{frontier['revision']}`. "
        f"Bootstrap resamples: {frontier['resamples']}.",
        "",
    ] + markdown_table_lines(frontier)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
        p.add_argument("--model-revision", required=True)
        p.add_argument("--accounting", required=True)
        p.add_argument("--split", default="results/artifacts/split/m1-split.json")
        p.add_argument("--split-receipt", default=None)
        p.add_argument("--preregistration-commit", required=True)
        p.add_argument("--receipt-dir", default="results/frontier")
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--stamp", default=None)
        p.add_argument("--json", action="store_true")
        p.add_argument(
            "--limit-items",
            type=int,
            default=0,
            help="debug only: cap items per domain; recorded in the receipt",
        )

    stats = sub.add_parser("stats")
    common(stats)
    stats.add_argument("--out", required=True)

    sweep = sub.add_parser("sweep")
    common(sweep)
    sweep.add_argument("--plans", required=True)
    sweep.add_argument("--arms", required=True, choices=["uniform", "layers", "control"])

    greedy = sub.add_parser("greedy")
    common(greedy)
    greedy.add_argument("--plans", required=True)

    exactness = sub.add_parser("exactness")
    common(exactness)
    exactness.add_argument("--bundle", required=True)
    exactness.add_argument("--layer", type=int, default=14)
    exactness.add_argument("--rows", type=int, default=64)
    exactness.add_argument("--groups", default="0,3,5,9")
    exactness.add_argument("--artifact-dir", default="results/artifacts/frontier")

    cross = sub.add_parser("cross-check")
    cross.add_argument("--rust-report", required=True)
    cross.add_argument("--request", default=None)
    cross.add_argument("--rust-output", required=True)
    cross.add_argument("--python-output", required=True)
    cross.add_argument("--rows", type=int, required=True)
    cross.add_argument("--hidden", type=int, required=True)
    cross.add_argument("--preregistration-commit", required=True)
    cross.add_argument("--receipt-dir", default="results/frontier")
    cross.add_argument("--stamp", default=None)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--sweep-glob", default="results/frontier/sweep-*.json")
    analyze.add_argument("--exactness", default=None)
    analyze.add_argument("--cross-check", default=None)
    analyze.add_argument("--out-dir", default="results/frontier")
    analyze.add_argument("--frontier-md", default="results/M1-FRONTIER.md")
    analyze.add_argument("--resamples", type=int, default=10000)
    analyze.add_argument("--preregistration-commit", required=True)
    analyze.add_argument("--stamp", default=None)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "stats":
        return command_stats(args)
    if args.command == "sweep":
        return command_sweep(args)
    if args.command == "greedy":
        return command_greedy(args)
    if args.command == "exactness":
        return command_exactness(args)
    if args.command == "cross-check":
        return command_cross_check(args)
    if args.command == "analyze":
        return command_analyze(args)
    parser.error(f"unknown command {args.command}")
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
