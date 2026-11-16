#!/usr/bin/env python3
"""Post-`H*` execution pipeline: verify, evaluate, score, and decide T2.

    scripts/post-hstar.sh [--resume] [--dry-run] [--stage NAME ...]

What it runs, in order, and nothing else:

  1. **preflight** — host memory and GPU-idle arithmetic. Refuses to start if a
     heavy GPU process or a `selvedge-` container is alive, or if
     `MemAvailable` cannot cover the largest stage's explicit limit plus a
     reserve. The arithmetic is written to a receipt.
  2. **verify** — the `H*` checkpoint's own sha256 sidecar must match its bytes,
     its summary must report the registered steps and tokens, its utility-label
     artifact must verify, and its training receipt must exist (harvested from
     the detached court state when the orchestrator died before composing it).
  3. **eval** — `H*` on the registered instrument (`oracle/eval/baseline.py
     --profile t2`), then the summary receipt.
  4. **instrument** — the P2.4 both-target check over the six students.
  5. **capability** — the verified-task court on `D`, `M`, `X`, `H`, `H*`:
     `generate` (GPU) then `check` (Rust court). `X` and `H` are scored under the
     certified provenance amendment; every arm records its own result receipt.
  6. **sensitivity** — the pre-registered corrected-dispatch re-evaluation
     (label `S1`) of the same `X` and `H` checkpoints with `mask_order=pair`.
     A distinct configuration: no retraining, no threshold change.
  7. **verdict** — `scripts/t2_verdict.py`: receipt-gated, deterministic, and
     refusing when any receipt is missing or mismatched.
  8. **compare** — `scripts/capability_compare.py`: the registered Level-1
     composite across arms, per category, conjunctive.

Every stage writes its state before and after it runs, so a death strands at
most one stage. A resume re-runs only stages that are not `done`. Every failure
stops the pipeline and records the reason; nothing downstream runs on a failed
stage.

Limits (directive 2). Every stage gets an explicit, workload-sized memory limit;
the limit is derived from that workload's measured VRAM peak (1.5x, bucketed),
with a floor of 24 GiB for the hybrid arms and 16 GiB for `D`/`M`/`X`. GPU
stages launch under `systemd-run --user --collect --unit=...` with
`MemoryMax`/`MemorySwapMax`/`TasksMax` set to the same values the court applies
inside the container. `--runner direct` (tests, and hosts without a user systemd)
runs the same argv through the court without the outer unit and says so in the
state; the container's own cgroup limits are unchanged either way.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SCHEMA = "selvedge/t2-post-hstar-state/v1"
STAGES = ("preflight", "verify", "eval", "instrument", "capability",
          "sensitivity", "verdict", "compare")
ARMS = ("D", "M", "X", "H", "H*")
ROUTED = ("X", "H", "H*")
HYBRID = ("H", "H*")
STEPS = 1024
BUDGET_TOKENS = STEPS * 65536
SEED = 1001
PREREG = "docs/capability-preregistration-v3.json"
AMENDMENT = "results/t2/PROVENANCE-AMENDMENT-X-and-H.json"
SENSITIVITY_LABEL = "S1"
GPU_BUSY_MIB = 3000
HEAVY_APP_MIB = 512
MEM_BUCKETS_GIB = (16, 24, 32, 48)
HEADROOM_FACTOR = 1.5
RESERVE_GIB = 2.0
CAPABILITY_CHECK_MEM_GIB = 12
CPU_STAGE_MEM_GIB = {"verdict": 8, "compare": 8}


class StageFailure(RuntimeError):
    def __init__(self, reason: str, exit_code: int = 1):
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code


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


def write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_name(arm: str) -> str:
    return "Hstar" if arm == "H*" else arm


# ---------------------------------------------------------------------------
# host probing and sizing


def read_meminfo() -> dict:
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                amount = rest.strip().split()
                if amount and amount[0].isdigit():
                    values[key] = int(amount[0]) / (1024 * 1024)
    except OSError:  # pragma: no cover - non-Linux
        pass
    return values


def probe_gpu() -> dict:
    state = {"available": False, "used_mib": None, "total_mib": None,
             "compute_apps": []}
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True).stdout.splitlines()[0]
        used_mib, total_mib = (int(float(part.strip())) for part in used.split(",")[:2])
        state.update({"available": True, "used_mib": used_mib,
                      "total_mib": total_mib})
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True).stdout.strip()
        for line in apps.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3 and parts[2].isdigit():
                state["compute_apps"].append(
                    {"pid": parts[0], "name": parts[1], "used_mib": int(parts[2])})
    except Exception as error:  # noqa: BLE001 - host dependent
        state["error"] = str(error)
    return state


def probe_containers() -> list:
    try:
        return subprocess.run(
            ["docker", "ps", "--filter", "name=selvedge-", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True).stdout.split()
    except Exception:  # noqa: BLE001
        return []


def git_revision(repo: str) -> str:
    value = os.environ.get("SELVEDGE_REVISION", "")
    if value:
        return value
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def bucket(need_gib: float) -> int:
    for size in MEM_BUCKETS_GIB:
        if size >= need_gib:
            return size
    return MEM_BUCKETS_GIB[-1]


def measured_peak(repo: str, arm: str) -> tuple[float | None, list]:
    """The arm's measured VRAM peak, from its own receipts (never asserted)."""
    peaks = []
    training = os.path.join(repo, "results/artifacts/train/t2", run_name(arm),
                            "summary.json")
    run = os.path.join(repo, "results/t2", arm, "run.json")
    evaluation = os.path.join(repo, "results/eval", f"t2-{arm}.json")
    for path, label in ((training, "training summary"),
                        (run, "training receipt"),
                        (evaluation, "evaluation receipt")):
        if not os.path.isfile(path):
            continue
        try:
            payload = load_json(path)
        except ValueError:
            continue
        value = payload.get("vram_peak_gib")
        if value is None:
            value = ((payload.get("measured") or {}).get("vram_peak_gib"))
        if isinstance(value, (int, float)) and value > 0:
            peaks.append({"source": os.path.relpath(path, repo), "kind": label,
                          "vram_peak_gib": float(value)})
    if not peaks:
        return None, []
    return max(entry["vram_peak_gib"] for entry in peaks), peaks


def size_stage(repo: str, stage: str, arm: str | None = None) -> dict:
    """Explicit, workload-sized host memory limit with its arithmetic."""
    if stage == "capability-check":
        return {"limit_gib": CAPABILITY_CHECK_MEM_GIB,
                "basis": "cargo + checker sandbox under the court's 45 s wall bound "
                         "(docs/CAPABILITY-COURT.md §10), no model weights",
                "peak_source": None, "peaks": []}
    if stage in CPU_STAGE_MEM_GIB:
        return {"limit_gib": CPU_STAGE_MEM_GIB[stage],
                "basis": "CPU-only reader over receipts; numpy arrays over items",
                "peak_source": None, "peaks": []}
    minimum = 24 if (arm in HYBRID or stage == "instrument") else 16
    peak, peaks = measured_peak(repo, arm) if arm else (None, [])
    if stage == "instrument":
        peaks = []
        best = 0.0
        for candidate in ARMS:
            value, sources = measured_peak(repo, candidate)
            peaks.extend(sources)
            best = max(best, value or 0.0)
        peak = best or None
        minimum = 24
    if peak is None:
        return {
            "limit_gib": minimum,
            "basis": f"no measured VRAM peak for {arm or stage}; the declared floor "
                     f"for this workload class is {minimum} GiB",
            "peak_source": None,
            "peaks": peaks,
            "arithmetic": {"headroom_factor": HEADROOM_FACTOR,
                           "measured_peak_gib": None,
                           "required_gib": None,
                           "floor_gib": minimum},
        }
    need = HEADROOM_FACTOR * peak
    limit = bucket(max(need, minimum))
    return {
        "limit_gib": limit,
        "basis": f"{HEADROOM_FACTOR}x the measured VRAM peak ({peak:.3f} GiB), "
                 f"bucketed to {limit} GiB, with a {minimum} GiB floor for this arm",
        "peak_source": peaks[-1]["source"] if peaks else None,
        "peaks": peaks,
        "arithmetic": {"headroom_factor": HEADROOM_FACTOR,
                       "measured_peak_gib": round(peak, 3),
                       "required_gib": round(need, 3),
                       "floor_gib": minimum,
                       "bucket_gib": limit},
    }


# ---------------------------------------------------------------------------
# runners


class Runner:
    """Every command runs in a court container with an explicit memory limit."""

    def __init__(self, repo: str, court: str, mode: str = "systemd",
                 log_path: str | None = None,
                 interpreter: list[str] | None = None):
        self.repo = repo
        self.court = court
        self.mode = mode
        self.log_path = log_path
        self.interpreter = list(interpreter or [])
        self.calls: list[dict] = []

    def _env(self, mem_gib: int, cpus: int, pids: int, tmp_gib: int) -> dict:
        env = dict(os.environ)
        env.update({
            "SELVEDGE_MEM": f"{mem_gib}g",
            "SELVEDGE_CPUS": str(cpus),
            "SELVEDGE_PIDS": str(pids),
            "SELVEDGE_TMP": f"{tmp_gib}g",
            "SELVEDGE_REVISION": git_revision(self.repo),
        })
        return env

    def run(self, argv: list[str], *, mem_gib: int, unit: str, cpus: int = 8,
            pids: int = 1024, tmp_gib: int = 8, gpu: bool = False,
            court_name: str = "oracle", dry_run: bool = False,
            note: str = "") -> dict:
        env = self._env(mem_gib, cpus, pids, tmp_gib)
        command = self.interpreter + [self.court]
        if gpu:
            command.append("--gpu")
        command += [court_name, "--"] + argv
        wrapped = command
        if self.mode == "systemd":
            wrapped = [
                "systemd-run", "--user", "--collect", "--pipe", f"--unit={unit}",
                f"--property=MemoryMax={mem_gib}G",
                f"--property=MemorySwapMax={mem_gib}G",
                f"--property=TasksMax={pids}",
                f"--property=CPUQuota={int(cpus * 100)}%",
                "--", *command,
            ]
        record = {"argv": command, "wrapped": wrapped, "mem_gib": mem_gib,
                  "cpus": cpus, "pids": pids, "tmp_gib": tmp_gib,
                  "unit": unit, "gpu": gpu, "court": court_name,
                  "mode": self.mode, "note": note}
        self.calls.append(record)
        if dry_run:
            record["dry_run"] = True
            print("DRY RUN: " + " ".join(wrapped))
            return record
        started = utc_now()
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(f"\n[{started}] {' '.join(wrapped)}\n")
                log.flush()
                completed = subprocess.run(wrapped, env=env, cwd=self.repo,
                                           stdout=log, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(wrapped, env=env, cwd=self.repo)
        record.update({"exit_code": completed.returncode, "started_utc": started,
                       "finished_utc": utc_now()})
        if completed.returncode != 0:
            raise StageFailure(
                f"command failed with exit {completed.returncode}: "
                f"{' '.join(command)}", completed.returncode)
        return record

    def cgroup(self, unit: str) -> dict:
        """What systemd actually applied, read after the unit stopped."""
        if self.mode != "systemd":
            return {"mode": self.mode,
                    "note": "no outer unit in this mode; the court's container "
                            "cgroup limits are recorded in its own receipt"}
        try:
            text = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=MemoryMax",
                 "--property=MemoryPeak", "--property=MemorySwapMax",
                 "--property=TasksMax", "--property=Result"],
                capture_output=True, text=True, check=True).stdout.strip()
            return {"values": dict(line.split("=", 1) for line in text.splitlines()
                                   if "=" in line)}
        except Exception as error:  # noqa: BLE001
            return {"error": str(error)}


# ---------------------------------------------------------------------------
# the pipeline


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.repo = os.path.abspath(args.repo_root)
        self.state_rel = args.state
        if args.dry_run:
            # A dry run must never leave a state that a real resume would skip.
            base, extension = os.path.splitext(args.state)
            self.state_rel = f"{base}.dry-run{extension}"
        self.state_path = os.path.join(self.repo, self.state_rel)
        self.log_path = os.path.join(self.repo, "results/t2/post-hstar.log")
        self.court = self._resolve_court(args.court)
        self.runner = Runner(self.repo, self.court, mode=args.runner,
                             log_path=self.log_path,
                             interpreter=args.court_interpreter)
        self.state = self._load_state()
        self.preflight_data: dict = {}

    def _resolve_court(self, court: str) -> str:
        if os.path.isabs(court) or "/" in court:
            return court if os.path.isabs(court) else os.path.join(self.repo, court)
        return os.path.join(self.repo, "scripts", court)

    # -- state ------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.args.resume and os.path.isfile(self.state_path):
            state = load_json(self.state_path)
            failure = state.get("failure")
            if failure:
                state.setdefault("failures", []).append(failure)
                state.pop("failure", None)
            state["status"] = "running"
            return state
        return {
            "schema": SCHEMA,
            "created_utc": utc_now(),
            "started_utc": utc_now(),
            "status": "running",
            "revision": git_revision(self.repo),
            "seed": SEED,
            "steps": STEPS,
            "budget_tokens": BUDGET_TOKENS,
            "arms": list(ARMS),
            "runner_mode": self.args.runner,
            "stages": {name: {"status": "pending"} for name in STAGES},
            "failures": [],
        }

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state)

    def stage(self, name: str) -> dict:
        return self.state["stages"].setdefault(name, {"status": "pending"})

    def begin(self, name: str, *, mem_gib: int, sizing: dict,
              outputs: list | None = None) -> None:
        entry = self.stage(name)
        entry.update({
            "status": "running",
            "started_utc": utc_now(),
            "revision": git_revision(self.repo),
            "seed": SEED,
            "host_memory_limit_gib": mem_gib,
            "sizing": sizing,
            "outputs": outputs or [],
        })
        entry.pop("failure", None)
        self.save()

    def finish(self, name: str, **fields) -> None:
        entry = self.stage(name)
        entry.update(fields)
        entry["status"] = "done"
        entry["finished_utc"] = utc_now()
        self.save()

    def fail(self, name: str, reason: str, exit_code: int = 1) -> int:
        entry = self.stage(name)
        entry.update({
            "status": "failed",
            "finished_utc": utc_now(),
            "failure": {"reason": reason, "exit_code": exit_code,
                        "utc": utc_now()},
        })
        self.state["status"] = "failed"
        self.state["failure"] = {"stage": name, "reason": reason,
                                 "exit_code": exit_code, "utc": utc_now()}
        self.save()
        self.say(f"FAILED in stage {name}: {reason}")
        self.say(f"state: {self.state_rel} — nothing after this stage was run")
        return exit_code

    def say(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, file=sys.stderr)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(line + "\n")

    def memory_guard(self, name: str, limit_gib: int) -> dict:
        """MemAvailable must cover this stage's explicit limit plus a reserve.

        Checked before the stage starts and recorded in its state entry, so the
        launch is never justified by a limit that the host cannot honour.
        """
        if self.args.dry_run:
            return {"checked": False, "reason": "dry run",
                    "limit_gib": limit_gib}
        if self.args.host_probe:
            probe = load_json(self.args.host_probe)
            available = ((probe.get("memory") or {}).get("available_gib"))
        else:
            available = round(read_meminfo().get("MemAvailable", 0.0), 3)
        arithmetic = {
            "checked": True,
            "available_gib": available,
            "limit_gib": limit_gib,
            "reserve_gib": RESERVE_GIB,
            "required_gib": round(limit_gib + RESERVE_GIB, 3),
            "fits": bool(available is not None
                         and available >= limit_gib + RESERVE_GIB),
            "utc": utc_now(),
        }
        self.stage(name)["memory_guard"] = arithmetic
        if not arithmetic["fits"]:
            raise StageFailure(
                f"{name}: MemAvailable {available} GiB cannot cover the stage limit "
                f"{limit_gib} GiB plus {RESERVE_GIB} GiB reserve", 3)
        return arithmetic

    def stage_max_limit(self, name: str) -> int:
        if name == "capability":
            return max(size_stage(self.repo, "capability-generate", arm)["limit_gib"]
                       for arm in ARMS)
        if name == "sensitivity":
            return max(size_stage(self.repo, "sensitivity", arm)["limit_gib"]
                       for arm in ("X", "H"))
        if name == "eval":
            return size_stage(self.repo, "eval", "H*")["limit_gib"]
        if name == "verify":
            return size_stage(self.repo, "verify", "H*")["limit_gib"]
        return size_stage(self.repo, name)["limit_gib"]

    # -- paths ------------------------------------------------------------

    def path(self, rel: str) -> str:
        return os.path.join(self.repo, rel)

    def checkpoint(self, arm: str) -> str:
        return self.path(f"results/artifacts/train/t2/{run_name(arm)}/"
                         f"step-{STEPS:06d}.pt")

    def eval_path(self, arm: str) -> str:
        return self.path(f"results/eval/t2-{arm}.json")

    def sensitivity_path(self, arm: str) -> str:
        return self.path(f"results/t2/{arm}/eval-pair-mask.json")

    def capability_dir(self, arm: str) -> str:
        return self.path(f"results/capability/runs/t2-{run_name(arm)}")

    def config_path(self, arm: str) -> str:
        return f"results/train/admission-configs/t2/{arm}.json"

    def admission_path(self, arm: str) -> str:
        return f"results/train/admission/t2/{arm}.json"

    def common_record(self, arms=ARMS) -> dict:
        record = {"revision": git_revision(self.repo), "seed": SEED}
        configs, admissions, peaks = {}, {}, {}
        for arm in arms:
            config = self.path(self.config_path(arm))
            admission = self.path(self.admission_path(arm))
            if os.path.isfile(config):
                configs[arm] = sha256_file(config)
            if os.path.isfile(admission):
                payload = load_json(admission)
                admissions[arm] = {
                    "active_unique_params": payload.get("active_unique_params"),
                    "admitted": payload.get("admitted"),
                    "sha256": sha256_file(admission),
                }
            peak, _sources = measured_peak(self.repo, arm)
            peaks[arm] = peak
        record["config_sha256"] = configs
        record["admission"] = admissions
        record["measured_vram_peak_gib"] = peaks
        return record

    # -- stage 1: preflight ----------------------------------------------

    def preflight(self) -> None:
        sizing = {"stages": {}}
        for arm in ARMS:
            sizing["stages"][f"capability-generate-{arm}"] = size_stage(
                self.repo, "capability-generate", arm)
        sizing["stages"]["eval"] = size_stage(self.repo, "eval", "H*")
        sizing["stages"]["instrument"] = size_stage(self.repo, "instrument")
        for arm in ("X", "H"):
            sizing["stages"][f"sensitivity-{arm}"] = size_stage(
                self.repo, "sensitivity", arm)
        sizing["stages"]["capability-check"] = size_stage(self.repo,
                                                         "capability-check")
        sizing["stages"]["verdict"] = size_stage(self.repo, "verdict")
        sizing["stages"]["compare"] = size_stage(self.repo, "compare")
        largest = max(entry["limit_gib"] for entry in sizing["stages"].values())
        self.begin("preflight", mem_gib=largest, sizing=sizing)
        if self.args.dry_run:
            self.say(f"DRY RUN preflight: largest stage limit {largest} GiB")
            self.finish("preflight", largest_limit_gib=largest,
                        dry_run=True)
            return

        if self.args.host_probe:
            probe = load_json(self.args.host_probe)
            mem = probe.get("memory") or {}
            gpu = probe.get("gpu") or {}
            containers = probe.get("containers") or []
        else:
            meminfo = read_meminfo()
            mem = {"total_gib": round(meminfo.get("MemTotal", 0.0), 3),
                   "available_gib": round(meminfo.get("MemAvailable", 0.0), 3),
                   "source": "/proc/meminfo"}
            gpu = probe_gpu()
            containers = probe_containers()

        available = mem.get("available_gib")
        if available is None:
            raise StageFailure("preflight: MemAvailable is unknown; refusing to "
                               "start without it")
        required = largest + RESERVE_GIB
        arithmetic = {
            "available_gib": available,
            "largest_stage_limit_gib": largest,
            "reserve_gib": RESERVE_GIB,
            "required_gib": round(required, 3),
            "fits": available >= required,
            "source": mem.get("source"),
        }

        problems = []
        if not arithmetic["fits"]:
            problems.append(
                f"MemAvailable {available:.3f} GiB cannot cover the largest stage "
                f"limit {largest} GiB plus a {RESERVE_GIB} GiB reserve "
                f"({required:.3f} GiB required)")
        if not gpu.get("available"):
            problems.append(f"the GPU cannot be probed: {gpu.get('error', 'nvidia-smi '
                                                                     'unavailable')}")
        else:
            if (gpu.get("used_mib") or 0) > GPU_BUSY_MIB:
                problems.append(
                    f"the GPU already holds {gpu['used_mib']} MiB "
                    f"(threshold {GPU_BUSY_MIB} MiB)")
            heavy = [app for app in gpu.get("compute_apps") or []
                     if app.get("used_mib", 0) >= HEAVY_APP_MIB]
            if heavy:
                problems.append(
                    "heavy compute processes are running: "
                    + ", ".join(f"{app['name']}({app['used_mib']}MiB)"
                                for app in heavy))
        if containers:
            problems.append("selvedge containers are running: "
                            + ", ".join(containers))

        receipt = {
            "schema": "selvedge/t2-post-hstar-preflight/v1",
            "stamp": stamp_now(),
            "utc": utc_now(),
            "revision": git_revision(self.repo),
            "memory": mem,
            "gpu": gpu,
            "containers": containers,
            "arithmetic": arithmetic,
            "sizing": sizing,
            "problems": problems,
            "ok": not problems,
        }
        out = f"results/t2/post-hstar-preflight-{receipt['stamp']}.json"
        write_json_atomic(self.path(out), receipt)
        if problems:
            raise StageFailure("preflight refused: " + "; ".join(problems), 3)
        self.preflight_data = receipt
        self.finish("preflight", largest_limit_gib=largest,
                    receipt=out, arithmetic=arithmetic,
                    gpu={"used_mib": gpu.get("used_mib"),
                         "total_mib": gpu.get("total_mib")},
                    memory=mem)

    # -- stage 2: verify -------------------------------------------------

    def _verify_sidecar(self, artifact: str, label: str) -> dict:
        if not os.path.isfile(artifact):
            raise StageFailure(f"verify: {label} {os.path.relpath(artifact, self.repo)} "
                               f"does not exist")
        sidecar = artifact + ".sha256"
        if not os.path.isfile(sidecar):
            raise StageFailure(f"verify: no sha256 sidecar for {label}")
        raw = open(sidecar, encoding="utf-8").read().strip()
        expected = None
        if raw.startswith("{"):
            expected = (json.loads(raw) or {}).get("sha256")
        elif raw:
            expected = raw.split()[0]
        if not expected:
            raise StageFailure(f"verify: sidecar for {label} carries no sha256")
        actual = sha256_file(artifact)
        if actual != expected:
            raise StageFailure(
                f"verify: {label} sha256 {actual} does not match its sidecar "
                f"{expected}")
        return {"path": os.path.relpath(artifact, self.repo), "sha256": actual,
                "bytes": os.path.getsize(artifact)}

    def _harvest_run_receipt(self) -> dict:
        run_receipt = self.path("results/t2/H*/run.json")
        if os.path.isfile(run_receipt):
            return {"harvested": False, "path": "results/t2/H*/run.json"}
        pilot_state = self.path("results/t2/pilot-state.json")
        pending = ""
        started = ""
        if os.path.isfile(pilot_state):
            state = load_json(pilot_state)
            block = (state.get("arms") or {}).get("H*") or {}
            pending = block.get("pending_state") or ""
            started = block.get("started_utc") or ""
        if not pending or not os.path.isfile(pending):
            raise StageFailure(
                "verify: results/t2/H*/run.json is missing and there is no pending "
                "court state to harvest it from; finish the training stage first "
                "with scripts/run-t2-pilot.sh --resume --arms 'H*'")
        court_receipt = ""
        name = os.path.basename(pending).removesuffix(".json")
        log = self.path(f"results/courts/logs/{name}.log")
        argv = [sys.executable, "scripts/t2_pilot.py", "arm-receipt",
                "--arm", "H*", "--run-name", "Hstar",
                "--training-state", pending, "--consume",
                "--launch-revision", git_revision(self.repo),
                "--container", name,
                "--court-receipt", court_receipt or "",
                "--court-log", log,
                "--started-utc", started or "",
                "--finished-utc", utc_now(),
                "--exit-code", "0"]
        completed = subprocess.run(argv, cwd=self.repo, capture_output=True,
                                   text=True)
        if completed.returncode != 0:
            raise StageFailure(
                "verify: could not harvest the H* training receipt: "
                + (completed.stderr.strip() or completed.stdout.strip()))
        return {"harvested": True, "path": "results/t2/H*/run.json",
                "stdout": completed.stdout.strip()[-500:]}

    def verify(self) -> None:
        sizing = size_stage(self.repo, "verify", "H*")
        self.begin("verify", mem_gib=sizing["limit_gib"], sizing=sizing)
        checkpoint = self.checkpoint("H*")
        if self.args.dry_run:
            self.say(f"DRY RUN [verify]: would hash {os.path.relpath(checkpoint, self.repo)} "
                     f"and its sidecar, check summary steps/tokens, hash "
                     f"utility-labels.pt, and harvest results/t2/H*/run.json if absent")
            self.finish("verify", dry_run=True)
            return
        block = {"checkpoint": self._verify_sidecar(checkpoint, "H* checkpoint")}
        summary_path = self.path("results/artifacts/train/t2/Hstar/summary.json")
        if not os.path.isfile(summary_path):
            raise StageFailure("verify: H* summary.json is missing; the run did not "
                               "finish")
        summary = load_json(summary_path)
        if int(summary.get("steps", -1)) != STEPS:
            raise StageFailure(f"verify: H* summary reports {summary.get('steps')} "
                               f"steps, the registered pilot is {STEPS}")
        if int(summary.get("tokens", -1)) != BUDGET_TOKENS:
            raise StageFailure(f"verify: H* summary reports {summary.get('tokens')} "
                               f"tokens, the registered budget is {BUDGET_TOKENS}")
        block["summary"] = {"steps": summary.get("steps"),
                            "tokens": summary.get("tokens"),
                            "tokens_per_second": summary.get("tokens_per_second"),
                            "vram_peak_gib": summary.get("vram_peak_gib"),
                            "finished_utc": summary.get("finished_utc")}
        labels = self.path("results/artifacts/train/t2/Hstar/utility-labels.pt")
        if not os.path.isfile(labels):
            raise StageFailure("verify: H*'s utility-label artifact is missing, so "
                               "Phase U did not complete and P12.2's diagnostics "
                               "cannot run later")
        block["utility_labels"] = self._verify_sidecar(labels, "utility labels")
        block["run_receipt"] = self._harvest_run_receipt()
        run = load_json(self.path("results/t2/H*/run.json"))
        block["run_receipt"].update({
            "arm": run.get("arm"), "seed": run.get("seed"),
            "revision": run.get("revision"),
            "dispatch_mask_order": run.get("dispatch_mask_order"),
            "checkpoint_sha256": (run.get("checkpoint") or {}).get("sha256"),
            "config_sha256": (run.get("config") or {}).get("sha256"),
            "admitted_active_unique": (run.get("admission") or {}).get(
                "active_unique_params"),
        })
        if block["run_receipt"]["checkpoint_sha256"] != block["checkpoint"]["sha256"]:
            raise StageFailure("verify: the training receipt's checkpoint digest "
                               "does not match the checkpoint file")
        config = self.path(self.config_path("H*"))
        if sha256_file(config) != block["run_receipt"]["config_sha256"]:
            raise StageFailure("verify: the H* config file changed after training")
        self.finish("verify", checkpoint=block["checkpoint"],
                    config_sha256=sha256_file(config),
                    admission=self.common_record(("H*",))["admission"],
                    measured_vram_peak_gib=block["summary"]["vram_peak_gib"],
                    utility_labels=block["utility_labels"],
                    run_receipt=block["run_receipt"],
                    vram_peak_gib=block["summary"]["vram_peak_gib"])

    # -- stage 3: evaluate H* --------------------------------------------

    def _eval_receipt_ok(self, path: str, arm: str, digest: str | None,
                         mask: str | None = None) -> dict | None:
        if not os.path.isfile(path):
            return None
        try:
            receipt = load_json(path)
        except ValueError:
            return None
        results = receipt.get("results") or {}
        if not {"humaneval", "gsm8k", "truthfulqa", "distributional"} <= set(results):
            return None
        if (receipt.get("arm") or "") != arm:
            return None
        recorded = (receipt.get("model") or {}).get("checkpoint_sha256")
        if digest and recorded != digest:
            return None
        if mask is not None:
            if (receipt.get("instrument") or {}).get("dispatch_mask_order") != mask:
                return None
        return receipt

    def compose_eval_summary(self, arm: str, receipt_rel: str) -> dict:
        """Write results/t2/<arm>/eval.json from the raw evaluation receipt.

        This is the same summary `scripts/t2_pilot.py eval-receipt` writes, kept
        here so the pipeline is self-contained (and so a test can point it at a
        fixture tree without touching the real one). It reads the receipt and
        writes a summary; it measures nothing.
        """
        receipt = load_json(self.path(receipt_rel))
        results = receipt.get("results") or {}
        task = {
            "humaneval_pass_at_1": (results.get("humaneval") or {}).get("pass_at_1"),
            "gsm8k_exact_match": (results.get("gsm8k") or {}).get("exact_match"),
            "truthfulqa_mc1": (results.get("truthfulqa") or {}).get("mc1"),
        }
        scores = [value for value in task.values() if value is not None]
        primary = sum(scores) / len(scores) if scores else None
        distributional = results.get("distributional") or {}
        payload = {
            "schema": "selvedge/t2-eval-summary/v1",
            "arm": arm,
            "stamp": stamp_now(),
            "utc": utc_now(),
            "eval_receipt": receipt_rel,
            "eval_receipt_stamp": receipt.get("stamp"),
            "eval_revision": receipt.get("revision"),
            "checkpoint_sha256": (receipt.get("model") or {}).get("checkpoint_sha256"),
            "config_sha256": (receipt.get("model") or {}).get("config_sha256"),
            "compliance_label": receipt.get("compliance_label"),
            "primary_composite_pc": primary,
            "primary_kl_mixed_p1": (distributional.get("mixed") or {}).get("kl"),
            "per_task": task,
            "per_domain": {
                domain: {
                    "kl": values.get("kl"),
                    "ppl": values.get("ppl"),
                    "top1": values.get("top1"),
                    "window_count": values.get("window_count"),
                    "documents": len(values.get("documents") or []),
                }
                for domain, values in (distributional.get("domains") or {}).items()
                if isinstance(values, dict)
            },
            "pooled": {
                "kl": distributional.get("kl"),
                "ppl": distributional.get("ppl"),
                "top1": distributional.get("top1"),
            },
            "mixed": distributional.get("mixed"),
            "instrument": receipt.get("instrument"),
            "seconds": receipt.get("seconds"),
            "vram_peak_gib": receipt.get("vram_peak_gib"),
            "command": receipt.get("command"),
            "determinism": receipt.get("determinism"),
        }
        out = f"results/t2/{arm}/eval.json"
        write_json_atomic(self.path(out), payload)
        return {"path": out, "primary_composite_pc": primary}

    def eval_hstar(self) -> None:
        sizing = size_stage(self.repo, "eval", "H*")
        self.begin("eval", mem_gib=sizing["limit_gib"], sizing=sizing)
        checkpoint = self.checkpoint("H*")
        if self.args.dry_run:
            self.runner.run(
                [sys.executable, "oracle/eval/baseline.py", "--profile", "t2",
                 "--arm", "H*", "--seed", str(SEED),
                 "--checkpoint", os.path.relpath(checkpoint, self.repo),
                 "--config", self.config_path("H*"),
                 "--admission", self.admission_path("H*"),
                 "--out", "results/eval/t2-H*.json"],
                mem_gib=sizing["limit_gib"],
                unit=f"selvedge-post-hstar-eval-{stamp_now()}",
                gpu=True, dry_run=True, note="registered instrument, H*")
            self.finish("eval", dry_run=True)
            return
        digest = sha256_file(checkpoint)
        out_rel = "results/eval/t2-H*.json"
        if self._eval_receipt_ok(self.path(out_rel), "H*", digest):
            self.say("[eval] H* receipt is already current for this checkpoint")
            receipt = load_json(self.path(out_rel))
        else:
            self.runner.run(
                [sys.executable, "oracle/eval/baseline.py", "--profile", "t2",
                 "--arm", "H*", "--seed", str(SEED),
                 "--checkpoint", os.path.relpath(checkpoint, self.repo),
                 "--config", self.config_path("H*"),
                 "--admission", self.admission_path("H*"),
                 "--out", out_rel],
                mem_gib=sizing["limit_gib"], unit=f"selvedge-post-hstar-eval-{stamp_now()}",
                gpu=True, dry_run=self.args.dry_run,
                note="registered instrument, H*")
            if self.args.dry_run:
                self.finish("eval", dry_run=True)
                return
            receipt = self._eval_receipt_ok(self.path(out_rel), "H*", digest)
            if receipt is None:
                raise StageFailure("eval: the H* evaluation receipt is missing, "
                                   "incomplete, or for a different checkpoint")
        summary = self.compose_eval_summary("H*", out_rel)
        self.finish("eval", receipt=out_rel,
                    summary=summary["path"],
                    checkpoint_sha256=digest,
                    config_sha256=sha256_file(self.path(self.config_path("H*"))),
                    admission=self.common_record(("H*",))["admission"],
                    vram_peak_gib=receipt.get("vram_peak_gib"),
                    eval_stamp=receipt.get("stamp"),
                    eval_revision=receipt.get("revision"))

    # -- stage 4: P2.4 instrument check ----------------------------------

    def instrument(self) -> None:
        sizing = size_stage(self.repo, "instrument")
        self.begin("instrument", mem_gib=sizing["limit_gib"], sizing=sizing)
        students = ["--student", "D0"]
        for arm in ARMS:
            students += ["--student",
                         f"{arm}:{os.path.relpath(self.checkpoint(arm), self.repo)}:"
                         f"{self.config_path(arm)}:{self.admission_path(arm)}:t2"]
        self.runner.run(
            [sys.executable, "oracle/eval/instrument_check.py", *students],
            mem_gib=sizing["limit_gib"],
            unit=f"selvedge-post-hstar-instrument-{stamp_now()}",
            gpu=True, dry_run=self.args.dry_run, note="P2.4 both-target check")
        if self.args.dry_run:
            self.finish("instrument", dry_run=True)
            return
        pattern = re.compile(r"results/train/instrument-\d{8}T\d{6}Z\.json$")
        candidates = sorted(name for name in os.listdir(self.path("results/train"))
                            if pattern.match(f"results/train/{name}"))
        if not candidates:
            raise StageFailure("instrument: no instrument receipt was written")
        latest = f"results/train/{candidates[-1]}"
        self.finish("instrument", receipt=latest,
                    receipt_sha256=sha256_file(self.path(latest)),
                    measured_vram_peak_gib=self.common_record(("X",))[
                        "measured_vram_peak_gib"]["X"])

    # -- stage 5: capability court ---------------------------------------

    def _capability_result_ok(self, arm: str) -> dict | None:
        path = os.path.join(self.capability_dir(arm), "result.json")
        if not os.path.isfile(path):
            return None
        try:
            result = load_json(path)
        except ValueError:
            return None
        prereg = self.path(PREREG)
        if ((result.get("preregistration") or {}).get("sha256")
                != sha256_file(prereg)):
            return None
        if not (result.get("scores") or {}).get("category_minimum_met"):
            return None
        if (result.get("scores") or {}).get("items", 0) < 72:
            return None
        provenance = ((result.get("generations") or {}).get("adapter") or {}).get(
            "provenance") or {}
        if arm in ROUTED:
            if provenance.get("dispatch_mask_order") not in ("legacy", "pair"):
                return None
            if arm in ("X", "H"):
                amendment = provenance.get("provenance_amendment") or {}
                if not amendment.get("used"):
                    return None
                if amendment.get("sha256") != sha256_file(self.path(AMENDMENT)):
                    return None
        return result

    def _mask_order(self, arm: str) -> str:
        try:
            run = load_json(self.path(f"results/t2/{arm}/run.json"))
        except (OSError, ValueError):
            return "legacy"
        recorded = run.get("dispatch_mask_order")
        if recorded in ("legacy", "pair"):
            return recorded
        return "legacy"

    def capability(self) -> None:
        outputs = [f"results/capability/runs/t2-{run_name(arm)}/result.json"
                   for arm in ARMS]
        self.begin("capability", mem_gib=CAPABILITY_CHECK_MEM_GIB,
                   sizing={"per_arm": {arm: size_stage(
                       self.repo, "capability-generate", arm) for arm in ARMS},
                       "check_gib": CAPABILITY_CHECK_MEM_GIB},
                   outputs=outputs)
        for arm in ARMS:
            result = self._capability_result_ok(arm)
            if result is not None:
                self.say(f"[capability] {arm} result is already current; not re-running")
                continue
            directory = self.capability_dir(arm)
            os.makedirs(directory, exist_ok=True)
            generations_rel = (f"results/capability/runs/t2-{run_name(arm)}/"
                               f"generations.json")
            result_rel = (f"results/capability/runs/t2-{run_name(arm)}/result.json")
            sizing = size_stage(self.repo, "capability-generate", arm)
            generate = [sys.executable, "scripts/capability_court.py", "generate",
                        "--t2-arm", arm,
                        "--t2-checkpoint",
                        os.path.relpath(self.checkpoint(arm), self.repo),
                        "--t2-config", self.config_path(arm),
                        "--t2-admission", self.admission_path(arm),
                        "--t2-training-receipt", f"results/t2/{arm}/run.json",
                        "--t2-mask-order", self._mask_order(arm),
                        "--preregistration", PREREG,
                        "--out", generations_rel]
            if arm in ("X", "H"):
                generate += ["--provenance-amendment", AMENDMENT]
            self.runner.run(
                generate, mem_gib=sizing["limit_gib"],
                unit=f"selvedge-post-hstar-cap-{run_name(arm).lower()}-{stamp_now()}",
                gpu=True, dry_run=self.args.dry_run,
                note=f"capability generate, {arm}")
            if self.args.dry_run:
                continue
            check = [sys.executable, "scripts/capability_court.py", "check",
                     "--generations", generations_rel,
                     "--preregistration", PREREG,
                     "--out", result_rel]
            self.runner.run(
                check, mem_gib=CAPABILITY_CHECK_MEM_GIB,
                unit=f"selvedge-post-hstar-check-{run_name(arm).lower()}-{stamp_now()}",
                cpus=8, tmp_gib=4, court_name="rust", dry_run=False,
                note=f"capability check, {arm}")
            result = self._capability_result_ok(arm)
            if result is None:
                raise StageFailure(
                    f"capability: {arm}'s result receipt is missing, does not bind "
                    f"the pre-registration, misses the category minimum, or lacks "
                    f"the provenance its routed dispatch requires")
            self.stage("capability").setdefault("arms", {})[arm] = {
                "generations": generations_rel,
                "result": result_rel,
                "result_sha256": sha256_file(os.path.join(directory, "result.json")),
                "composite": (result.get("scores") or {}).get("primary_composite"),
                "vram_peak_gib": measured_peak(self.repo, arm)[0],
                "host_memory_limit_gib": sizing["limit_gib"],
            }
            self.save()
        self.finish("capability", arms=self.stage("capability").get("arms", {}),
                    common=self.common_record())

    # -- stage 6: corrected-dispatch sensitivity --------------------------

    def sensitivity(self) -> None:
        self.begin("sensitivity", mem_gib=24,
                   sizing={arm: size_stage(self.repo, "sensitivity", arm)
                           for arm in ("X", "H")},
                   outputs=[f"results/t2/{arm}/eval-pair-mask.json"
                            for arm in ("X", "H")])
        records = {}
        for arm in ("X", "H"):
            sizing = size_stage(self.repo, "sensitivity", arm)
            out_rel = f"results/t2/{arm}/eval-pair-mask.json"
            if self.args.dry_run:
                self.runner.run(
                    [sys.executable, "oracle/eval/baseline.py", "--profile", "t2",
                     "--arm", arm, "--seed", str(SEED),
                     "--checkpoint", os.path.relpath(self.checkpoint(arm), self.repo),
                     "--config", self.config_path(arm),
                     "--admission", self.admission_path(arm),
                     "--mask-order", "pair",
                     "--out", out_rel],
                    mem_gib=sizing["limit_gib"],
                    unit=f"selvedge-post-hstar-sens-{arm.lower()}-{stamp_now()}",
                    gpu=True, dry_run=True,
                    note=f"corrected-dispatch sensitivity {SENSITIVITY_LABEL}, {arm}")
                continue
            digest = sha256_file(self.checkpoint(arm))
            current = self._eval_receipt_ok(self.path(out_rel), arm, digest, "pair")
            if current is not None:
                self.say(f"[sensitivity] {arm} pair-mask receipt is current")
                receipt = current
            else:
                self.runner.run(
                    [sys.executable, "oracle/eval/baseline.py", "--profile", "t2",
                     "--arm", arm, "--seed", str(SEED),
                     "--checkpoint", os.path.relpath(self.checkpoint(arm), self.repo),
                     "--config", self.config_path(arm),
                     "--admission", self.admission_path(arm),
                     "--mask-order", "pair",
                     "--out", out_rel],
                    mem_gib=sizing["limit_gib"],
                    unit=f"selvedge-post-hstar-sens-{arm.lower()}-{stamp_now()}",
                    gpu=True, dry_run=self.args.dry_run,
                    note=f"corrected-dispatch sensitivity {SENSITIVITY_LABEL}, {arm}")
                if self.args.dry_run:
                    continue
                receipt = self._eval_receipt_ok(self.path(out_rel), arm, digest, "pair")
                if receipt is None:
                    raise StageFailure(
                        f"sensitivity: {arm}'s pair-mask receipt is missing, "
                        f"incomplete, for a different checkpoint, or does not record "
                        f"mask_order=pair")
            records[arm] = {
                "label": SENSITIVITY_LABEL,
                "receipt": out_rel,
                "receipt_sha256": sha256_file(self.path(out_rel)),
                "mask_order": "pair",
                "checkpoint_sha256": digest,
                "vram_peak_gib": receipt.get("vram_peak_gib"),
                "host_memory_limit_gib": sizing["limit_gib"],
            }
        self.finish("sensitivity", label=SENSITIVITY_LABEL, receipts=records,
                    common=self.common_record(("X", "H")))

    # -- stage 7: the verdict ---------------------------------------------

    def verdict(self) -> None:
        sizing = size_stage(self.repo, "verdict")
        self.begin("verdict", mem_gib=sizing["limit_gib"], sizing=sizing,
                   outputs=["results/t2-verdict.json", "results/T2-VERDICT.md"])
        argv = [sys.executable, "scripts/t2_verdict.py",
                "--repo-root", self.repo,
                "--d0", "results/eval/t2-D0.json",
                "--sensitivity", "X=results/t2/X/eval-pair-mask.json",
                "--sensitivity", "H=results/t2/H/eval-pair-mask.json",
                "--out", "results/t2-verdict.json",
                "--doc", "results/T2-VERDICT.md",
                "--write"]
        if self.args.dry_run:
            self.runner.run(argv, mem_gib=sizing["limit_gib"],
                            unit=f"selvedge-post-hstar-verdict-{stamp_now()}",
                            cpus=4, tmp_gib=2, court_name="python", dry_run=True,
                            note="registered decision rule")
            self.finish("verdict", dry_run=True)
            return
        self.runner.run(argv, mem_gib=sizing["limit_gib"],
                        unit=f"selvedge-post-hstar-verdict-{stamp_now()}",
                        cpus=4, tmp_gib=2, court_name="python",
                        dry_run=self.args.dry_run,
                        note="registered decision rule")
        if self.args.dry_run:
            self.finish("verdict", dry_run=True)
            return
        receipt = load_json(self.path("results/t2-verdict.json"))
        self.finish("verdict", receipt="results/t2-verdict.json",
                    receipt_sha256=sha256_file(self.path("results/t2-verdict.json")),
                    document="results/T2-VERDICT.md",
                    outcome=receipt.get("outcome"),
                    kill_conditions={name: entry.get("verdict")
                                     for name, entry in
                                     receipt["kill_conditions"].items()},
                    deterministic_body_sha256=receipt.get(
                        "deterministic_body_sha256"))

    # -- stage 8: the capability comparison --------------------------------

    def compare(self) -> None:
        sizing = size_stage(self.repo, "compare")
        self.begin("compare", mem_gib=sizing["limit_gib"], sizing=sizing,
                   outputs=["results/capability/runs/t2-comparison.json"])
        argv = [sys.executable, "scripts/capability_compare.py",
                "--repo-root", self.repo]
        for arm in ARMS:
            argv += ["--result",
                     f"{arm}=results/capability/runs/t2-{run_name(arm)}/result.json"]
        argv += ["--preregistration", PREREG,
                 "--out", "results/capability/runs/t2-comparison.json"]
        if self.args.dry_run:
            self.runner.run(argv, mem_gib=sizing["limit_gib"],
                            unit=f"selvedge-post-hstar-compare-{stamp_now()}",
                            cpus=4, tmp_gib=2, court_name="python", dry_run=True,
                            note="registered Level-1 comparison")
            self.finish("compare", dry_run=True)
            return
        self.runner.run(argv, mem_gib=sizing["limit_gib"],
                        unit=f"selvedge-post-hstar-compare-{stamp_now()}",
                        cpus=4, tmp_gib=2, court_name="python",
                        dry_run=self.args.dry_run,
                        note="registered Level-1 comparison")
        if self.args.dry_run:
            self.finish("compare", dry_run=True)
            return
        payload = load_json(self.path("results/capability/runs/t2-comparison.json"))
        self.finish("compare",
                    receipt="results/capability/runs/t2-comparison.json",
                    receipt_sha256=sha256_file(
                        self.path("results/capability/runs/t2-comparison.json")),
                    outcome=payload.get("outcome"),
                    gate=payload.get("gate"))

    # -- driver ------------------------------------------------------------

    def run(self) -> int:
        selected = [name for name in STAGES
                    if not self.args.stage or name in self.args.stage]
        if self.args.stage:
            unknown = set(self.args.stage) - set(STAGES)
            if unknown:
                raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
        self.state["status"] = "running"
        self.save()
        self.say(f"post-H* pipeline starting: stages={','.join(selected)} "
                 f"runner={self.args.runner} resume={self.args.resume}")
        for name in selected:
            entry = self.stage(name)
            if entry.get("status") == "done" and not self.args.force:
                self.say(f"[{name}] already done; skipping")
                continue
            if self.preflight_data.get("ok") is False:
                return self.fail(name, "preflight refused; nothing may start", 3)
            try:
                if name != "preflight":
                    self.memory_guard(name, self.stage_max_limit(name))
                    self.save()
                self.say(f"[{name}] start")
                getattr(self, self._handler(name))()
            except StageFailure as error:
                return self.fail(name, error.reason, error.exit_code)
            except Exception as error:  # noqa: BLE001 - fail closed, named
                return self.fail(name, f"unexpected {type(error).__name__}: {error}", 70)
            self.say(f"[{name}] ok")
        self.state["status"] = "done"
        self.state["finished_utc"] = utc_now()
        self.save()
        self.say("post-H* pipeline complete; state: " + self.state_rel)
        return 0

    def _handler(self, name: str) -> str:
        return {"eval": "eval_hstar"}.get(name, name)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=REPO)
    parser.add_argument("--state", default="results/t2/post-hstar-state.json")
    parser.add_argument("--court", default="scripts/court",
                        help="the court launcher (replaced by a fixture in tests)")
    parser.add_argument("--court-interpreter", action="append", default=[],
                        help="executable prefix for --court (tests: the fixture is "
                             "a Python file on a noexec tmpfs)")
    parser.add_argument("--runner", choices=["systemd", "direct"], default="systemd")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the recorded state")
    parser.add_argument("--stage", action="append", default=[],
                        help="run only this stage (repeatable; stages run in order)")
    parser.add_argument("--force", action="store_true",
                        help="re-run a stage recorded as done")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host-probe", default=None,
                        help="JSON with memory/gpu/containers for a dry run or a "
                             "CPU fixture; production never uses it")
    args = parser.parse_args(argv)
    return Pipeline(args).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
