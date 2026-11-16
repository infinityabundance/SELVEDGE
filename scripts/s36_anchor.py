#!/usr/bin/env python3
"""WS19 anchor scoreboard driver (S36).

Three subcommands. `lock` and `gap` are standard-library only; `run` imports the
pinned harness from the eval court image, because that is the only place the
harness exists.

  lock   Resolve the two scored checkpoints at the revisions a pre-registration
         pins, verify every cached file's byte size and sha256, and write
         models/FRONTIER.lock.json. With --net it also reads the Hub tree at the
         pinned SHA (so the *declared* sizes and LFS oids are the Hub's, not
         this machine's); offline it re-verifies what is already cached.

  run    Score one model on the pre-registered tasks with
         EleutherAI/lm-evaluation-harness, IN PROCESS, so that
         `torch.cuda.max_memory_allocated()` and the wall clock are evidence
         rather than an outside observation. One receipt per invocation; one
         entry per task.

  gap    Compute the 0.8B -> 2B gap per task and per family from two run
         receipts, and the arithmetic of the registered win condition.

Why a pre-registration file instead of flags on the command line: the config has
to be provably identical for both models, and it has to be provably *prior* to
the scores. `run` hashes docs/anchor-preregistration-v1.json and records the
hash in every receipt; it refuses to run a task that is not in the file.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

SCHEMA_PREREG = "selvedge/anchor-preregistration/v1"
SCHEMA_LOCK = "selvedge/frontier-lock/v1"
SCHEMA_RUN = "selvedge/anchor-run/v1"
SCHEMA_GAP = "selvedge/anchor-gap/v1"

DEFAULT_PREREG = "docs/anchor-preregistration-v1.json"
DEFAULT_LOCK = "models/FRONTIER.lock.json"
CACHE_ROOT = os.environ.get("HF_HOME", "/hf-cache")


class AnchorError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# small helpers


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return os.environ.get("SELVEDGE_REVISION", "unknown")


def sha256_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def http_json(url: str, limit: int = 64 * 1024 * 1024) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "selvedge-s36-anchor"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        raise AnchorError(f"{url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AnchorError(f"{url}: {exc.reason}") from exc
    if len(payload) > limit:
        raise AnchorError(f"{url}: response exceeded {limit} bytes")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AnchorError(f"{url}: not JSON ({exc})") from exc


def free_bytes(path: str) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def cgroup_peak_bytes() -> dict:
    """The container's own cgroup peak, so the next run can be sized from it."""
    out = {}
    for name in ("memory.peak", "memory.current", "memory.max", "memory.swap.max"):
        path = os.path.join("/sys/fs/cgroup", name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                out[name] = handle.read().strip()
        except OSError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# lock


def cmd_lock(args) -> int:
    prereg = load_json(args.prereg)
    prereg_digest = sha256_file(args.prereg)

    receipt = {
        "schema": SCHEMA_LOCK,
        "stamp": stamp(),
        "revision": git_revision(),
        "finished_utc": utcnow(),
        "network": bool(args.net),
        "cache_root": args.cache_root,
        "cache_root_note": "host ~/.cache/selvedge/hf mounted at /hf-cache",
        "preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "cache_free_bytes_before": free_bytes(args.cache_root),
        "models": [],
    }

    for entry in prereg["models"]:
        model_id = entry["id"]
        revision = entry["revision"]
        directory = os.path.join(args.cache_root, "models", model_id, revision)
        model = {
            "id": model_id,
            "revision": revision,
            "role": entry["role"],
            "license": entry["license"],
            "license_declared_by_hub": None,
            "cache_relative_dir": os.path.relpath(directory, args.cache_root),
            "files": [],
            "weight_bytes": 0,
            "weight_file_count": 0,
            "tree_listed": False,
            "missing": [],
        }

        declared: dict[str, dict] = {}
        if args.net:
            meta = http_json(f"https://huggingface.co/api/models/{model_id}")
            model["hub_main_sha"] = meta.get("sha")
            model["hub_card_license"] = (meta.get("cardData") or {}).get("license")
            tree = http_json(
                f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
            )
            declared = {
                item["path"]: {
                    "bytes": (item.get("lfs") or {}).get("size", item.get("size")),
                    "lfs_oid": (item.get("lfs") or {}).get("oid"),
                }
                for item in tree
                if item.get("type") == "file"
            }
            if not declared:
                raise AnchorError(f"{model_id}@{revision}: the Hub tree is empty")
            model["tree_listed"] = True
            model["tree_file_count"] = len(declared)

        if not os.path.isdir(directory):
            raise AnchorError(f"{directory} is absent; fetch the checkpoint first")
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            digest = sha256_file(path)
            record = {
                "path": name,
                "bytes": size,
                "sha256": digest,
                "declared_bytes": None,
                "lfs_oid": None,
                "sha256_matches_lfs_oid": None,
                "bytes_match_declared": None,
            }
            if model["tree_listed"]:
                hub = declared.get(name)
                if hub is None:
                    raise AnchorError(
                        f"{model_id}@{revision}: cached file {name!r} is not in the Hub tree"
                    )
                record["declared_bytes"] = hub["bytes"]
                record["lfs_oid"] = hub["lfs_oid"]
                record["bytes_match_declared"] = size == hub["bytes"]
                if hub["lfs_oid"]:
                    record["sha256_matches_lfs_oid"] = digest == hub["lfs_oid"]
                if not record["bytes_match_declared"]:
                    raise AnchorError(
                        f"{model_id}/{name}: cached {size} bytes, Hub declares {hub['bytes']}"
                    )
                if record["sha256_matches_lfs_oid"] is False:
                    raise AnchorError(
                        f"{model_id}/{name}: sha256 {digest} != Hub LFS oid {hub['lfs_oid']}"
                    )
            if name.endswith(".safetensors"):
                model["weight_bytes"] += size
                model["weight_file_count"] += 1
            model["files"].append(record)

        if model["tree_listed"]:
            missing = [
                name
                for name, hub in declared.items()
                if name.endswith(".safetensors") and not os.path.exists(os.path.join(directory, name))
            ]
            model["missing"] = missing
            if missing:
                raise AnchorError(f"{model_id}: weights not all cached, missing {missing}")
            expected = sum(
                hub["bytes"] for name, hub in declared.items() if name.endswith(".safetensors")
            )
            if model["weight_bytes"] != expected:
                raise AnchorError(
                    f"{model_id}: cached weight bytes {model['weight_bytes']} != Hub {expected}"
                )

        license_file = os.path.join(directory, "LICENSE")
        model["license_file_present"] = os.path.isfile(license_file)
        if model["license_file_present"]:
            with open(license_file, "r", encoding="utf-8", errors="replace") as handle:
                lines = [line.strip() for line in handle.readlines()[:8]]
            model["license_first_line"] = next((line for line in lines if line), "")
            model["license_head"] = [line for line in lines if line][:3]
            model["license_sha256"] = sha256_file(license_file)
        receipt["models"].append(model)
        print(
            f"  {model_id}@{revision[:12]} weights={model['weight_file_count']} "
            f"bytes={model['weight_bytes']} files={len(model['files'])} "
            f"license={model['license']}",
            flush=True,
        )

    receipt["cache_free_bytes_after"] = free_bytes(args.cache_root)
    write_json(args.lock, receipt)
    print(f"lock: wrote {args.lock} ({len(receipt['models'])} models)", flush=True)
    return 0


def _metric_candidates(primary: str) -> list[str]:
    """0.4.14 keys every metric as `name,filter`; `acc` is stored as `acc,none`."""
    return [primary, f"{primary},none"]


def resolve_metric(metrics: dict, primary: str) -> tuple[str, str]:
    for candidate in _metric_candidates(primary):
        if candidate in metrics:
            return candidate, stderr_key(candidate)
    matching = [key for key in metrics if key.split(",")[0] == primary and "stderr" not in key]
    if len(matching) == 1:
        return matching[0], stderr_key(matching[0])
    raise AnchorError(
        f"primary metric {primary!r} is ambiguous or absent; the harness reported {sorted(metrics)}"
    )


def stderr_key(metric: str) -> str:
    name, _, filt = metric.partition(",")
    if filt:
        return f"{name}_stderr,{filt}"
    return f"{name}_stderr"


def condense_config(config: dict | None) -> dict | None:
    """The parts of the harness's resolved task config that pin the instrument."""
    if not config:
        return None
    keep = (
        "task",
        "dataset_path",
        "dataset_name",
        "output_type",
        "num_fewshot",
        "doc_to_text",
        "doc_to_target",
        "doc_to_choice",
        "fewshot_delimiter",
        "target_delimiter",
        "generation_kwargs",
        "metric_list",
        "filter_list",
        "group",
        "task_list",
    )
    return {key: config[key] for key in keep if key in config}


# ---------------------------------------------------------------------------
# run


def task_evidence(name: str) -> dict:
    """The harness's own resolved config for a task, plus yaml hashes."""
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    index = getattr(manager, "task_index", {})
    entry = index.get(name)
    if entry is None:
        raise AnchorError(f"the harness has no task {name!r}")
    files = {}
    paths = [entry.yaml_path]
    sibling = os.path.join(os.path.dirname(str(entry.yaml_path)), "utils.py")
    if os.path.isfile(sibling):
        paths.append(sibling)
    for path in paths:
        path = str(path)
        files[path] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    return {
        "name": name,
        "kind": str(getattr(entry, "kind", "")),
        "yaml_path": str(entry.yaml_path),
        "yaml_and_utils": files,
        "resolved_config": entry.cfg,
    }


def harness_commit() -> tuple[str, str]:
    root = os.environ.get("LM_EVAL_HARNESS_ROOT", "/opt/lm-evaluation-harness")
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                root,
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # pragma: no cover - the court always has it
        raise AnchorError(f"cannot read the harness commit under {root}: {exc}") from exc
    try:
        import lm_eval

        version = getattr(lm_eval, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return commit, version


def cmd_run(args) -> int:
    import torch
    from lm_eval.evaluator import simple_evaluate

    prereg = load_json(args.prereg)
    if prereg.get("schema") != SCHEMA_PREREG:
        raise AnchorError(f"{args.prereg}: unexpected schema {prereg.get('schema')!r}")
    prereg_digest = sha256_file(args.prereg)

    # HumanEval executes generated code through the `evaluate` code_eval metric,
    # which refuses to run unless this variable is set. The court is the sandbox
    # that makes setting it defensible: no network, --cap-drop ALL,
    # no-new-privileges, a 2 GiB tmpfs, and a hard memory/pids/CPU limit.
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    # IFEval's post-processing sentence-tokenises with NLTK. The corpus lives in
    # the mounted cache rather than in the image, and its hashes are recorded in
    # results/frontier/nltk-data-punkt_tab.json. Without it the task raises
    # LookupError *after* generating, which is a silent way to lose a run.
    nltk_data = os.environ.setdefault("NLTK_DATA", os.path.join(args.cache_root, "nltk_data"))
    if not os.path.isdir(nltk_data):
        raise AnchorError(
            f"{nltk_data} is absent; the NLTK punkt_tab corpus IFEval needs was never fetched"
        )

    existing = None
    if args.append:
        if not os.path.isfile(args.out):
            raise AnchorError(f"--append needs an existing receipt at {args.out}")
        existing = load_json(args.out)
        if existing.get("model", {}).get("id") != args.model_id:
            raise AnchorError("--append into a receipt for a different model")
        if existing.get("preregistration", {}).get("sha256") != prereg_digest:
            raise AnchorError("--append into a receipt produced under a different pre-registration")
        if existing.get("smoke"):
            raise AnchorError("--append into a smoke receipt")
        if existing.get("complete"):
            raise AnchorError("--append into a receipt that is already complete")
        done = {entry["task"] for entry in existing["tasks"]}
        clashes = sorted(done.intersection(args.tasks))
        if clashes:
            raise AnchorError(f"--append would score {clashes} twice")

    declared = {task["name"]: task for task in prereg["tasks"]}
    unknown = [name for name in args.tasks if name not in declared]
    if unknown:
        raise AnchorError(f"tasks not in the pre-registration: {unknown}")

    models = {entry["id"]: entry for entry in prereg["models"]}
    model = models.get(args.model_id)
    if model is None:
        raise AnchorError(f"{args.model_id} is not in the pre-registration")
    model_dir = os.path.join(args.cache_root, "models", args.model_id, model["revision"])
    if not os.path.isdir(model_dir):
        raise AnchorError(f"{model_dir} is absent")

    lock = None
    if os.path.isfile(args.lock):
        lock = load_json(args.lock)
        locked = {entry["id"]: entry for entry in lock["models"]}.get(args.model_id)
        if locked is None:
            raise AnchorError(f"{args.model_id} is not in {args.lock}")
        if locked["revision"] != model["revision"]:
            raise AnchorError(
                f"{args.model_id}: lock revision {locked['revision']} != prereg {model['revision']}"
            )
        for record in locked["files"]:
            path = os.path.join(model_dir, record["path"])
            if not os.path.isfile(path):
                raise AnchorError(f"{path} is absent but the lock lists it")
            if os.path.getsize(path) != record["bytes"]:
                raise AnchorError(
                    f"{path}: {os.path.getsize(path)} bytes != locked {record['bytes']}"
                )
            if args.verify_weights and record["path"].endswith(".safetensors"):
                digest = sha256_file(path)
                if digest != record["sha256"]:
                    raise AnchorError(f"{path}: sha256 drifted from the lock")

    commit, version = harness_commit()
    if commit != prereg["harness"]["commit"]:
        raise AnchorError(
            f"harness commit {commit} != pre-registered {prereg['harness']['commit']}"
        )
    if args.prereg_sha256 and args.prereg_sha256 != prereg_digest:
        raise AnchorError("the pre-registration file changed under this run")

    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise AnchorError("CUDA is not available in this container; refusing a CPU fallback")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        device_name = torch.cuda.get_device_name(0)
        device_total = torch.cuda.get_device_properties(0).total_memory
    else:
        device_name = platform.processor() or "cpu"
        device_total = 0

    model_args = dict(prereg["model_args"])
    model_args["pretrained"] = model_dir
    model_args_str = ",".join(f"{key}={value}" for key, value in model_args.items())
    gen_kwargs = prereg.get("gen_kwargs")

    receipt = {
        "schema": SCHEMA_RUN,
        "stamp": stamp(),
        "started_utc": utcnow(),
        "revision": git_revision(),
        "model": {
            "id": args.model_id,
            "revision": model["revision"],
            "role": model["role"],
            "directory": model_dir,
            "license": model["license"],
        },
        "preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "harness": {
            "name": prereg["harness"]["name"],
            "commit": commit,
            "version": version,
            "image_ref": os.environ.get("SELVEDGE_COURT_IMAGE", "unknown"),
            "image_id": os.environ.get("SELVEDGE_COURT_IMAGE_ID", "unknown"),
        },
        "court": {
            "name": os.environ.get("SELVEDGE_COURT", "unknown"),
            "memory_limit": os.environ.get("SELVEDGE_COURT_MEM", "unknown"),
            "memory_limit_bytes": os.environ.get("SELVEDGE_COURT_MEM_BYTES", "unknown"),
            "cpus": os.environ.get("SELVEDGE_COURT_CPUS", "unknown"),
            "pids_limit": os.environ.get("SELVEDGE_COURT_PIDS", "unknown"),
            "tmpfs_bytes": os.environ.get("SELVEDGE_COURT_TMP_BYTES", "unknown"),
            "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset"),
        },
        "device": {
            "requested": device,
            "name": device_name,
            "total_memory_bytes": device_total,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "argv": sys.argv,
        "smoke": bool(args.smoke),
        "complete": False,
        "limit_override": args.limit,
        "prompting": prereg["prompting"],
        "model_args_resolved": model_args_str,
        "tasks": [],
    }

    fresh = receipt
    if existing is not None:
        # Keep the receipt's identity and its completed entries; record the new
        # invocation so a reader can see the receipt was built from more than one
        # court call and can find each call's argv.
        receipt = existing
        receipt.setdefault("invocations", []).append(
            {
                "stamp": fresh["stamp"],
                "started_utc": fresh["started_utc"],
                "argv": sys.argv,
                "tasks": list(args.tasks),
                "court": fresh["court"],
            }
        )
        receipt["harness"] = fresh["harness"]
        receipt["court"] = fresh["court"]
        receipt["device"] = fresh["device"]
        receipt["argv"] = sys.argv
    else:
        receipt["invocations"] = [
            {
                "stamp": fresh["stamp"],
                "started_utc": fresh["started_utc"],
                "argv": sys.argv,
                "tasks": list(args.tasks),
                "court": fresh["court"],
            }
        ]

    failed = 0
    evidence = {name: task_evidence(name) for name in args.tasks}
    for name in args.tasks:
        task = declared[name]
        limit = args.limit if args.limit is not None else task.get("limit")
        started = time.time()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        result = simple_evaluate(
            model=prereg["harness"]["model"],
            model_args=model_args_str,
            tasks=[name],
            num_fewshot=task["num_fewshot"],
            batch_size=task.get("batch_size", "auto"),
            max_batch_size=prereg.get("max_batch_size"),
            device=device,
            limit=limit,
            log_samples=True,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            gen_kwargs=gen_kwargs,
            confirm_run_unsafe_code=True,
            bootstrap_iters=prereg.get("bootstrap_iters", 100000),
            random_seed=prereg["seed"]["random_seed"],
            numpy_random_seed=prereg["seed"]["numpy"],
            torch_random_seed=prereg["seed"]["torch"],
            fewshot_random_seed=prereg["seed"]["fewshot"],
        )
        elapsed = time.time() - started
        if result is None:
            raise AnchorError(f"{name}: the harness returned no results")
        metrics = result["results"][name]
        metric_key, stderr_k = resolve_metric(metrics, task["primary_metric"])
        score = metrics[metric_key]
        subtasks = sorted(key for key in (result.get("samples") or {}) if key == name or key.startswith(name))
        logged = sum(len(result["samples"][key] or []) for key in subtasks)
        configs = result.get("configs") or {}
        harness_config = condense_config(configs.get(name))
        if harness_config is None:
            harness_config = {
                "group": True,
                "subtasks": {
                    key: condense_config(value)
                    for key, value in configs.items()
                    if key == name or key.startswith(name)
                },
            }
        entry = {
            "task": name,
            "score": score,
            "stderr": metrics.get(stderr_k),
            "primary_metric": metric_key,
            "primary_metric_registered": task["primary_metric"],
            "all_metrics": metrics,
            "n_samples": {
                "sample_len": metrics.get("sample_len"),
                "sample_count": metrics.get("sample_count", {}).get(metric_key)
                if isinstance(metrics.get("sample_count"), dict)
                else None,
                "logged": logged,
                "subtasks": {
                    key: (result.get("n-samples") or {}).get(key)
                    for key in sorted(result.get("n-samples") or {})
                },
            },
            "pre_registered": {
                "limit": limit,
                "limit_from_preregistration": task.get("limit"),
                "num_fewshot": task["num_fewshot"],
                "family": task["family"],
                "registered_reason": task.get("reason"),
            },
            "wall_clock_s": round(elapsed, 1),
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if device.startswith("cuda") else None
            ),
            "harness_config": harness_config,
            "task_definition": evidence[name],
            "cgroup": cgroup_peak_bytes(),
            "git_hash": result.get("git_hash"),
            "date": result.get("date"),
            "zero_score": score == 0.0,
        }
        receipt["tasks"].append(entry)

        # Written after every task, atomically, so an interrupted run leaves a
        # receipt that says exactly which tasks it completed instead of nothing.
        receipt["partial_through"] = name
        write_json(args.out, receipt)

        raw_path = os.path.join(args.raw_dir, f"{receipt['stamp']}-{args.model_id.replace('/', '__')}-{name}.json")
        os.makedirs(args.raw_dir, exist_ok=True)
        write_json(raw_path, result)
        entry["raw_result"] = raw_path

        print(
            f"  {name}: {metric_key}={score:.6f} stderr={entry['stderr']} "
            f"n={entry['n_samples']['sample_len']} wall={elapsed:.1f}s "
            f"peak_gib={(entry['peak_memory_allocated_bytes'] or 0) / 2**30:.3f}",
            flush=True,
        )
        if entry["zero_score"]:
            failed += 1
            print(
                f"  !! {name} scored exactly 0: this is a defect signal until the prompt, "
                "stop sequence and extraction are diagnosed from the logged samples",
                file=sys.stderr,
            )

    receipt["finished_utc"] = utcnow()
    receipt["cgroup"] = cgroup_peak_bytes()
    receipt.pop("partial_through", None)
    scored = {entry["task"] for entry in receipt["tasks"]}
    receipt["missing_tasks"] = [
        task["name"] for task in prereg["tasks"] if task["name"] not in scored
    ]
    receipt["complete"] = not receipt["missing_tasks"]
    receipt["zero_score_tasks"] = [entry["task"] for entry in receipt["tasks"] if entry["zero_score"]]
    write_json(args.out, receipt)
    print(f"run: wrote {args.out}", flush=True)
    if receipt["missing_tasks"]:
        print(
            f"run: INCOMPLETE — still missing {receipt['missing_tasks']}; the receipt says so "
            "and `gap` refuses it",
            file=sys.stderr,
        )
    if args.smoke:
        return 0
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# rebuild


def log_observations(log_path: str) -> dict[str, dict]:
    """Pull wall clock and device peak back out of the court log.

    The runner prints one line per task in a fixed shape; reading it back is how
    a receipt that was never written can still carry the run's own resource
    numbers instead of losing them. The log is committed next to the receipt.
    """
    import re

    pattern = re.compile(
        r"(?P<task>[a-z0-9_]+): (?P<metric>\S+)=(?P<score>[0-9.]+) "
        r"stderr=(?P<stderr>\S+) n=(?P<n>\S+) wall=(?P<wall>[0-9.]+)s "
        r"peak_gib=(?P<peak>[0-9.]+)"
    )
    out: dict[str, dict] = {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for match in pattern.finditer(line.replace("\r", "")):
                out[match.group("task")] = {
                    "metric": match.group("metric"),
                    "score": float(match.group("score")),
                    "wall_clock_s": float(match.group("wall")),
                    "peak_memory_allocated_bytes": int(float(match.group("peak")) * 2**30),
                }
    return out


def load_result(path: str) -> dict:
    """Raw harness results are committed gzipped; read either form."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return load_json(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cmd_pack(args) -> int:
    """Compress the raw harness results and index them by digest.

    One MMLU result file carries all 14042 samples and is ~200 MB of JSON, which
    is past GitHub's per-file limit and wasteful to store verbatim. gzip is
    lossless, so the evidence stays complete; the index pins both digests so a
    reader can prove the committed archive unpacks to the file the receipt names.
    """
    os.makedirs(args.raw_dir, exist_ok=True)
    entries = {}
    total_raw = 0
    total_gz = 0
    for path in sorted(glob.glob(os.path.join(args.raw_dir, "*.json"))):
        name = os.path.basename(path)
        if name == "INDEX.json":
            continue
        with open(path, "rb") as handle:
            payload = handle.read()
        archive = gzip.compress(payload, compresslevel=9, mtime=0)
        with open(path + ".gz", "wb") as handle:
            handle.write(archive)
        entries[name] = {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "gz_bytes": len(archive),
            "gz_sha256": sha256_bytes(archive),
        }
        total_raw += len(payload)
        total_gz += len(archive)
    index = {
        "schema": "selvedge/anchor-raw-index/v1",
        "note": (
            "the uncompressed *.json files are gitignored: one of them is a 200 MB MMLU "
            "result. The committed *.json.gz archives are lossless and are what the "
            "receipts' raw_result paths resolve to; `rebuild` reads either form."
        ),
        "packed_utc": utcnow(),
        "raw_dir": args.raw_dir,
        "files": entries,
        "total_bytes": total_raw,
        "total_gz_bytes": total_gz,
    }
    write_json(os.path.join(args.raw_dir, "INDEX.json"), index)
    print(
        f"pack: {len(entries)} result file(s), {total_raw} B -> {total_gz} B "
        f"({(100.0 * total_gz / total_raw) if total_raw else 0:.1f}% of raw)",
        flush=True,
    )
    return 0


def cmd_rebuild(args) -> int:
    prereg = load_json(args.prereg)
    if prereg.get("schema") != SCHEMA_PREREG:
        raise AnchorError(f"{args.prereg}: unexpected schema")
    prereg_digest = sha256_file(args.prereg)
    model = {entry["id"]: entry for entry in prereg["models"]}.get(args.model_id)
    if model is None:
        raise AnchorError(f"{args.model_id} is not in the pre-registration")
    commit, version = harness_commit()
    if commit != prereg["harness"]["commit"]:
        raise AnchorError(f"harness commit {commit} != pre-registered {prereg['harness']['commit']}")

    slug = args.model_id.replace("/", "__")
    invocations = []
    for spec in args.invocation:
        if "=" not in spec:
            raise AnchorError(f"--invocation expects STAMP=COURT_LOG, got {spec!r}")
        invocation_stamp, log_path = spec.split("=", 1)
        if not os.path.isfile(log_path):
            raise AnchorError(f"{log_path} is absent")
        invocations.append(
            {
                "stamp": invocation_stamp,
                "court_log": log_path,
                "court_log_sha256": sha256_file(log_path),
                "observed": log_observations(log_path),
            }
        )

    def raw_for(invocation_stamp: str, name: str) -> str | None:
        candidates = sorted(
            glob.glob(os.path.join(args.raw_dir, f"{invocation_stamp}-{slug}-{name}.json*"))
        )
        if not candidates:
            return None
        # The stamp prefix is the identity of the invocation; a smoke of the same
        # task under the same model must never be mistaken for the scored run.
        plain = [path for path in candidates if path.endswith(".json")]
        candidates = plain or candidates
        if len(candidates) > 1:
            raise AnchorError(f"{name}: {len(candidates)} raw results claim stamp {invocation_stamp}")
        return candidates[-1]

    entries = []
    missing = []
    used = {invocation["stamp"]: [] for invocation in invocations}
    for task in prereg["tasks"]:
        name = task["name"]
        # An appended task's raw file is written under the receipt's stamp, not the
        # append invocation's, so the file is looked for under every known stamp and
        # the court log line decides which invocation actually scored it.
        paths = []
        for candidate in [args.stamp] + [value["stamp"] for value in invocations]:
            found = raw_for(candidate, name)
            if found and found not in [path for _stamp, path in paths]:
                paths.append((candidate, found))
        if not paths:
            missing.append(name)
            continue
        observation_invocation = next(
            (value for value in invocations if name in value["observed"]), None
        )
        raw_stamp, raw_path = paths[0]
        invocation = observation_invocation or next(
            (value for value in invocations if value["stamp"] == raw_stamp), invocations[-1]
        )
        observed = observation_invocation["observed"][name] if observation_invocation else {}
        result = load_result(raw_path)
        metrics = result["results"][name]
        metric_key, stderr_k = resolve_metric(metrics, task["primary_metric"])
        if observed.get("metric") and observed["metric"] != metric_key:
            raise AnchorError(
                f"{name}: the log reports {observed['metric']} but the raw result's primary is "
                f"{metric_key}; refusing to mix two different reads of one run"
            )
        # The log prints six decimals, so agreement is checked at that precision.
        if observed and abs(observed["score"] - metrics[metric_key]) > 1e-6:
            raise AnchorError(
                f"{name}: the log reports {observed['score']} but the raw result says "
                f"{metrics[metric_key]}; the log and the result disagree"
            )
        configs = result.get("configs") or {}
        harness_config = condense_config(configs.get(name))
        if harness_config is None:
            harness_config = {
                "group": True,
                "subtasks": {
                    key: condense_config(value)
                    for key, value in configs.items()
                    if key == name or key.startswith(name)
                },
            }
        subtasks = sorted(
            key for key in (result.get("samples") or {}) if key == name or key.startswith(name)
        )
        with open(raw_path, "rb") as handle:
            raw_digest = sha256_bytes(handle.read())
        archive = raw_path + ".gz"
        entries.append(
            {
                "task": name,
                "score": metrics[metric_key],
                "stderr": metrics.get(stderr_k),
                "primary_metric": metric_key,
                "primary_metric_registered": task["primary_metric"],
                "all_metrics": metrics,
                "n_samples": {
                    "sample_len": metrics.get("sample_len"),
                    "sample_count": metrics.get("sample_count", {}).get(metric_key)
                    if isinstance(metrics.get("sample_count"), dict)
                    else None,
                    "logged": sum(len(result["samples"][key] or []) for key in subtasks),
                    "subtasks": {
                        key: (result.get("n-samples") or {}).get(key)
                        for key in sorted(result.get("n-samples") or {})
                    },
                },
                "pre_registered": {
                    "limit": task.get("limit"),
                    "num_fewshot": task["num_fewshot"],
                    "family": task["family"],
                    "registered_reason": task.get("reason"),
                },
                "wall_clock_s": observed.get("wall_clock_s"),
                "peak_memory_allocated_bytes": observed.get("peak_memory_allocated_bytes"),
                "peak_memory_reserved_bytes": None,
                "cgroup": None,
                "harness_config": harness_config,
                "task_definition": task_evidence(name),
                "git_hash": result.get("git_hash"),
                "date": result.get("date"),
                "zero_score": metrics[metric_key] == 0.0,
                "raw_result": raw_path,
                "raw_result_sha256": raw_digest,
                "raw_result_archive": archive if os.path.isfile(archive) else None,
                "provenance": {
                    "invocation_stamp": invocation["stamp"],
                    "raw_result": (
                        "the harness's own per-task result file, written by simple_evaluate; "
                        "the committed form is the .gz archive whose digest is pinned in "
                        "results/frontier/anchor-raw/INDEX.json"
                    ),
                    "wall_clock_and_peak": f"{invocation['court_log']} (the runner's per-task line)"
                    if observed
                    else None,
                    "note": (
                        "rebuilt rather than observed in one pass: the score is the "
                        "harness's own, the timings are the court log's"
                    ),
                },
            }
        )
        used[invocation["stamp"]].append(name)

    receipt = {
        "schema": SCHEMA_RUN,
        "stamp": args.stamp,
        "started_utc": utcnow(),
        "revision": git_revision(),
        "model": {
            "id": args.model_id,
            "revision": model["revision"],
            "role": model["role"],
            "directory": os.path.join(args.cache_root, "models", args.model_id, model["revision"]),
            "license": model["license"],
        },
        "preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "harness": {"name": prereg["harness"]["name"], "commit": commit, "version": version},
        "court": {
            "memory_limit": args.memory_limit,
            "note": "rebuilt from the court logs of the invocations that scored these tasks",
        },
        "device": {
            "note": "not re-observed here; each raw result file carries the run's pretty_env_info",
        },
        "smoke": False,
        "complete": not missing,
        "missing_tasks": missing,
        "assembly": {
            "method": (
                "rebuilt from the harness's own per-task result files (committed under "
                f"{args.raw_dir}) plus the per-task lines of the court log of the invocation "
                "that produced each one, because the single-pass receipt was lost when a later "
                "task failed after the earlier ones had been scored"
            ),
            "raw_dir": args.raw_dir,
            "raw_index": os.path.join(args.raw_dir, "INDEX.json"),
            "why": (
                "the first invocation scored seven tasks and then died on IFEval's missing NLTK "
                "corpus; every score below is the harness's own output, and every timing is the "
                "court log line of the invocation recorded beside it under `invocations`"
            ),
        },
        "invocations": [
            {
                "stamp": value["stamp"],
                "court_log": value["court_log"],
                "court_log_sha256": value["court_log_sha256"],
                "tasks": sorted(used[value["stamp"]]),
            }
            for value in invocations
        ],
        "argv": sys.argv,
        "prompting": prereg["prompting"],
        "model_args_resolved": ",".join(
            f"{key}={value}" for key, value in prereg["model_args"].items()
        ),
        "tasks": entries,
    }
    receipt["zero_score_tasks"] = [entry["task"] for entry in entries if entry["zero_score"]]
    write_json(args.out, receipt)
    for entry in entries:
        print(
            f"  {entry['task']:<18} {entry['primary_metric']}={entry['score']:.6f} "
            f"n={entry['n_samples']['sample_len']} wall={entry['wall_clock_s']}s "
            f"inv={entry['provenance']['invocation_stamp']}",
            flush=True,
        )
    print(f"rebuild: wrote {args.out} (missing={missing})", flush=True)
    return 0


# ---------------------------------------------------------------------------
# gap


def find_entry(receipt: dict, task: str) -> dict | None:
    for entry in receipt.get("tasks", []):
        if entry["task"] == task:
            return entry
    return None


def n_of(entry: dict):
    return (entry.get("n_samples") or {}).get("sample_len")


def cmd_gap(args) -> int:
    prereg = load_json(args.prereg)
    anchor = load_json(args.anchor)
    upper = load_json(args.upper)
    prereg_digest = sha256_file(args.prereg)

    for receipt, role in ((anchor, "anchor"), (upper, "upper")):
        if receipt.get("smoke"):
            raise AnchorError(f"{receipt['model']['id']}: smoke receipts are not scores")
        if not receipt.get("complete", True) and not args.allow_partial:
            raise AnchorError(
                f"{receipt['model']['id']}: the run is incomplete "
                f"({receipt.get('missing_tasks') or receipt.get('partial_through')}); a gap needs "
                "both full task sets, or --allow-partial for an explicitly partial table"
            )
        if receipt["preregistration"]["sha256"] != prereg_digest:
            raise AnchorError(
                f"{receipt['model']['id']}: run was produced under a different "
                "pre-registration; refusing to compare"
            )


    families = prereg["families"]
    # Family membership comes from the registered `families` block, not from the
    # per-task descriptive tag: the win condition is defined over those member
    # lists, and a task can carry a broader label than the family it belongs to.
    family_of = {
        member: family["name"] for family in families for member in family.get("members", [])
    }
    rows = []
    skipped = []
    for task in prereg["tasks"]:
        name = task["name"]
        low = find_entry(anchor, name)
        high = find_entry(upper, name)
        if low is None or high is None:
            skipped.append(
                {
                    "task": name,
                    "family": family_of.get(name),
                    "anchor_scored": low is not None,
                    "upper_scored": high is not None,
                    "reason": "a gap is only defined for a task both models scored under this config",
                }
            )
            continue
        gap_pp = (high["score"] - low["score"]) * 100.0
        half = gap_pp / 2.0
        target = (low["score"] + (high["score"] - low["score"]) / 2.0) * 100.0
        rows.append(
            {
                "task": name,
                "family": family_of.get(name),
                "task_label_family": task["family"],
                "registered_family_membership": family_of.get(name),
                "metric": low["primary_metric"],
                "anchor_score": low["score"],
                "anchor_stderr": low["stderr"],
                "anchor_n": n_of(low),
                "upper_score": high["score"],
                "upper_stderr": high["stderr"],
                "upper_n": n_of(high),
                "gap_pp": gap_pp,
                "half_gap_pp": half,
                "anchor_plus_half_gap_pp": target,
                "anchor_peak_gib": (low["peak_memory_allocated_bytes"] or 0) / 2**30,
                "upper_peak_gib": (high["peak_memory_allocated_bytes"] or 0) / 2**30,
                "anchor_wall_s": low["wall_clock_s"],
                "upper_wall_s": high["wall_clock_s"],
            }
        )

    family_rows = []
    for family in families:
        expected = family.get("members") or []
        members = [row for row in rows if row["task"] in expected]
        if not members:
            continue
        low = sum(member["anchor_score"] for member in members) / len(members)
        high = sum(member["upper_score"] for member in members) / len(members)
        family_rows.append(
            {
                "family": family["name"],
                "definition": family["definition"],
                "members": [member["task"] for member in members],
                "registered_members": expected,
                "complete": sorted(member["task"] for member in members) == sorted(expected),
                "missing_members": [name for name in expected if name not in {m["task"] for m in members}],
                "aggregation": "unweighted mean of member scores",
                "anchor_mean": low,
                "upper_mean": high,
                "gap_pp": (high - low) * 100.0,
                "half_gap_pp": (high - low) * 100.0 / 2.0,
                "anchor_plus_half_gap": low + (high - low) / 2.0,
            }
        )

    win = prereg["win_condition"]
    best = max(family_rows, key=lambda row: row["gap_pp"]) if family_rows else None
    receipt = {
        "schema": SCHEMA_GAP,
        "stamp": stamp(),
        "created_utc": utcnow(),
        "revision": git_revision(),
        "preregistration": {"path": args.prereg, "sha256": prereg_digest},
        "anchor": {
            "id": anchor["model"]["id"],
            "revision": anchor["model"]["revision"],
            "receipt": args.anchor,
            "stamp": anchor["stamp"],
        },
        "upper": {
            "id": upper["model"]["id"],
            "revision": upper["model"]["revision"],
            "receipt": args.upper,
            "stamp": upper["stamp"],
        },
        "harness_commit": anchor["harness"]["commit"],
        "identical_config": anchor["preregistration"]["sha256"] == upper["preregistration"]["sha256"],
        "partial": bool(skipped),
        "skipped_tasks": skipped,
        "tasks": rows,
        "families": family_rows,
        "win_condition": win,
        "win_condition_reading": (
            "the target model must reach anchor_score + half the anchor->upper gap on ONE "
            "family, measured with this receipt's task set and this harness config"
        ),
        "largest_family_gap": best,
    }
    write_json(args.out, receipt)
    for row in rows:
        print(
            f"  {row['task']:<18} anchor={row['anchor_score'] * 100:6.2f} "
            f"upper={row['upper_score'] * 100:6.2f} gap={row['gap_pp']:6.2f}pp "
            f"half={row['half_gap_pp']:6.2f}pp",
            flush=True,
        )
    for row in family_rows:
        print(
            f"  family {row['family']:<10} anchor={row['anchor_mean'] * 100:6.2f} "
            f"upper={row['upper_mean'] * 100:6.2f} gap={row['gap_pp']:6.2f}pp "
            f"half={row['half_gap_pp']:6.2f}pp target={row['anchor_plus_half_gap'] * 100:6.2f}",
            flush=True,
        )
    print(f"gap: wrote {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    lock = sub.add_parser("lock")
    lock.add_argument("--prereg", default=DEFAULT_PREREG)
    lock.add_argument("--lock", default=DEFAULT_LOCK)
    lock.add_argument("--cache-root", default=CACHE_ROOT)
    lock.add_argument("--net", action="store_true", help="read the Hub tree at the pinned SHA")
    lock.set_defaults(func=cmd_lock)

    run = sub.add_parser("run")
    run.add_argument("--prereg", default=DEFAULT_PREREG)
    run.add_argument("--lock", default=DEFAULT_LOCK)
    run.add_argument("--model-id", required=True)
    run.add_argument("--tasks", nargs="+", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--raw-dir", default="results/frontier/anchor-raw")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--cache-root", default=CACHE_ROOT)
    run.add_argument("--prereg-sha256", default=None)
    run.add_argument("--verify-weights", action="store_true")
    run.add_argument("--limit", type=int, default=None, help="smoke only: override the limit")
    run.add_argument("--smoke", action="store_true", help="label the receipt as a smoke measurement")
    run.add_argument("--append", action="store_true", help="add tasks to an incomplete receipt")
    run.set_defaults(func=cmd_run)

    rebuild = sub.add_parser("rebuild")
    rebuild.add_argument("--prereg", default=DEFAULT_PREREG)
    rebuild.add_argument("--model-id", required=True)
    rebuild.add_argument("--raw-dir", default="results/frontier/anchor-raw")
    rebuild.add_argument(
        "--invocation",
        action="append",
        required=True,
        help="STAMP=COURT_LOG, one per invocation whose raw results are being assembled",
    )
    rebuild.add_argument("--out", required=True)
    rebuild.add_argument("--stamp", required=True)
    rebuild.add_argument("--cache-root", default=CACHE_ROOT)
    rebuild.add_argument("--memory-limit", default="unknown")
    rebuild.set_defaults(func=cmd_rebuild)

    pack = sub.add_parser("pack")
    pack.add_argument("--raw-dir", default="results/frontier/anchor-raw")
    pack.set_defaults(func=cmd_pack)

    gap = sub.add_parser("gap")
    gap.add_argument("--prereg", default=DEFAULT_PREREG)
    gap.add_argument("--anchor", required=True)
    gap.add_argument("--upper", required=True)
    gap.add_argument("--out", required=True)
    gap.add_argument(
        "--allow-partial",
        action="store_true",
        help="compute gaps only for tasks both models scored, and say so in the receipt",
    )
    gap.set_defaults(func=cmd_gap)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AnchorError as exc:
        print(f"s36-anchor: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
