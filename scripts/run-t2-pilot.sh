#!/usr/bin/env bash
# T2 pilot orchestrator — the five-arm error-directed capacity matrix.
#
#   scripts/run-t2-pilot.sh [--resume] [--arms "D M X H H*"] [--force-targets]
#                           [--no-instrument] [--from-eval] [--dry-run]
#
# What it does, in order, and nothing else:
#
#   1. guards: free-disk arithmetic for the exact artifacts this run writes and
#      an idle-GPU check. Blocked means stopped, with the arithmetic recorded.
#   2. teacher targets: build the shared cache at the registered budget if it is
#      incomplete, then verify it. A verification failure stops everything.
#   3. arms, strictly sequential (one GPU): launch each arm as a *detached*
#      court container, wait for it, record its exit code, and evaluate it with
#      the registered harness immediately on completion.
#   4. the P2.4 instrument check over all five final checkpoints.
#   5. the pre-registered decision analysis (paired CIs, K15-K19).
#
# Every step runs through scripts/court (directive 1). The training containers
# are launched with `docker run -d` through the court's `--detach` mode, so a
# lost session does not kill a run: `scripts/court wait <state>` finishes it.
# State is written to results/t2/pilot-state.json after every transition, and
# every failure is recorded there before this script exits non-zero.
#
# Environment (hard limits, fail-closed inside the court):
#   SELVEDGE_MEM=28g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g
#
# Resuming: `scripts/run-t2-pilot.sh --resume` continues from the recorded
# state. It has four recovery paths, tried in this order:
#
#   1. a live or exited detached container recorded in the arm's pending state:
#      `scripts/court wait` finishes it and writes its receipt and log copy;
#   2. no awaitable container, but a **verified final checkpoint** (`step-N.pt`
#      matching its sidecar sha256 and, when present, a summary that agrees on
#      steps and tokens): the training stage is marked `trained` and the run
#      continues at evaluation without retraining;
#   3. a completed evaluation receipt for exactly that checkpoint (its recorded
#      `checkpoint_sha256` equals the sidecar): the eval stage is not re-run;
#   4. otherwise, training resumes from the latest checkpoint.
#
# `--from-eval` is the operator's "do not train anything" switch: it implies
# `--resume` and refuses to start an arm unless its final checkpoint verifies.
#
# Every stage transition is written to results/t2/pilot-state.json as it
# happens, so a host-process death can strand at most one stage, never a
# finished checkpoint. An arm whose container exited non-zero without a
# verified checkpoint is *not* resumed automatically; the failure stays
# recorded until a human clears it. A previous `failure` block is archived into
# `failures` on resume (never deleted) so the live state describes the run that
# is actually in progress.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

budget_tokens=67108864
steps=1024
seed=1001
arms="D M X H H*"
resume=0
force_targets=0
instrument=1
from_eval=0
dry_run=0

state_rel="results/t2/pilot-state.json"
state_abs="$repo_root/$state_rel"
log_rel="results/t2/orchestrator.log"
log_abs="$repo_root/$log_rel"
run_root="results/artifacts/train/t2"
eval_root="results/eval"

usage() {
  sed -n '2,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (($#)); do
  case "$1" in
    --resume) resume=1; shift ;;
    --arms) arms="$2"; shift 2 ;;
    --force-targets) force_targets=1; shift ;;
    --no-instrument) instrument=0; shift ;;
    --from-eval) from_eval=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) printf 'run-t2-pilot: unknown argument %q\n' "$1" >&2; exit 64 ;;
  esac
done

mkdir -p results/t2

if ((from_eval)); then
  resume=1
fi

IFS=' ' read -r -a arm_list <<<"$arms"

say() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log() {
  say "$*" | tee -a "$log_abs" >&2
}

t2p() {
  python3 scripts/t2_pilot.py "$@"
}

join_args() {
  local out="" arg
  for arg in "$@"; do
    out+="$arg "
  done
  printf '%s' "${out% }"
}

state_set() {
  t2p state-set --state "$state_abs" --field "$1" --value "$2"
}

fail_now() {
  local phase="$1" code="$2" detail="$3"
  state_set status '"failed"'
  state_set failure "{\"phase\": \"$phase\", \"exit_code\": $code, \"detail\": \"$detail\", \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
  log "FAILED in phase $phase (exit $code): $detail"
  log "state: $state_rel — nothing after this point was run"
  exit "$code"
}

# `court_launch <argv...>`: detached launch, prints the pending-state path.
court_launch() {
  scripts/court --gpu --detach oracle -- "$@" 2>>"$log_abs"
}

# `court_await <pending-state>`: block, capture the log and the receipt.
declare COURT_RECEIPT=""
court_await() {
  local pending="$1" pathfile rc
  pathfile=$(mktemp)
  set +e
  SELVEDGE_RECEIPT_PATHFILE="$pathfile" scripts/court wait "$pending" 2>&1 |
    tee -a "$log_abs" >&2
  rc=${PIPESTATUS[0]}
  set -e
  COURT_RECEIPT=""
  if [[ -s "$pathfile" ]]; then
    COURT_RECEIPT="$(cat "$pathfile")"
  fi
  rm -f "$pathfile"
  return "$rc"
}

# `court_run <phase> <argv...>`: detached launch + wait, aborting on failure.
court_run() {
  local phase="$1"
  shift
  if ((dry_run)); then
    log "DRY RUN [$phase]: $(join_args "$@")"
    return 0
  fi
  local pending
  pending=$(court_launch "$@") || fail_now "$phase" 69 "detached launch refused"
  [[ -n "$pending" ]] || fail_now "$phase" 69 "no pending-state path from the court"
  log "[$phase] launched; state=$(basename "$pending")"
  court_await "$pending" || fail_now "$phase" "$?" "court wait returned non-zero"
  log "[$phase] ok; receipt=${COURT_RECEIPT:-none}"
}

arm_run_name() {
  case "$1" in
    'H*') printf 'Hstar' ;;
    *) printf '%s' "$1" ;;
  esac
}

arm_config() { printf 'results/train/admission-configs/t2/%s.json' "$1"; }
arm_admission() { printf 'results/train/admission/t2/%s.json' "$1"; }

# The registered per-arm argv (results/T2-PLAN.md §3, contract T2.6/T2.8).
arm_argv() {
  local arm="$1" out="$2" extra=()
  case "$arm" in
    D | M) ;;
    X) extra+=(--load-balance) ;;
    H) extra+=(--load-balance --warmup-steps 64) ;;
    'H*') extra+=(--warmup-steps 64 --calibration-windows 32 --router-steps 200
                  --labels-out "$out/utility-labels.pt") ;;
    *) fail_now "arm-argv" 64 "unknown arm $arm" ;;
  esac
  ARM_ARGV=(python3 oracle/train/loop.py
    --arm "$arm" --profile t2
    --config "$(arm_config "$arm")"
    --admission "$(arm_admission "$arm")"
    --out "$out" --steps "$steps" --seed "$seed"
    --micro-batch 1 --accum 64 --micro-positions 512
    --checkpoint-every 100 --zero-point --deterministic
    "${extra[@]}")
}

latest_checkpoint() {
  local out="$1"
  python3 - "$out" <<'PY'
import json, os, re, sys
directory = sys.argv[1]
latest = os.path.join(directory, "latest.json")
if os.path.exists(latest):
    try:
        print(json.load(open(latest, encoding="utf-8")).get("checkpoint", ""))
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
names = sorted(name for name in os.listdir(directory)
               if re.fullmatch(r"step-\d+\.pt", name)) if os.path.isdir(directory) else []
print(os.path.join(directory, names[-1]) if names else "")
PY
}

# `verified_final_checkpoint <out> <steps>`: succeed only when the arm's final
# checkpoint exists, its bytes match the sidecar sha256, and (when a summary is
# present) the summary reports exactly the registered steps and tokens. This is
# the test that lets a resume skip a training run that already finished but
# whose orchestrator died. Prints one JSON object on success.
verified_final_checkpoint() {
  local out="$1" step_count="$2"
  python3 - "$out" "$step_count" <<'PY'
import hashlib, json, os, sys

out, steps = sys.argv[1], int(sys.argv[2])
tokens_expected = steps * 65536
final = os.path.join(out, f"step-{steps:06d}.pt")


def fail(reason: str) -> None:
    print(reason, file=sys.stderr)
    raise SystemExit(1)


if not os.path.isfile(final):
    fail(f"no final checkpoint at {final}")
sidecar = final + ".sha256"
if not os.path.isfile(sidecar):
    fail(f"no sha256 sidecar at {sidecar}")
raw = open(sidecar, encoding="utf-8").read().strip()
expected = None
if raw.startswith("{"):
    expected = (json.loads(raw) or {}).get("sha256")
elif raw:
    expected = raw.split()[0]
if not expected:
    fail(f"sidecar {sidecar} carries no sha256")
digest = hashlib.sha256()
with open(final, "rb") as handle:
    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    fail(f"sha256 mismatch for {final}: {actual} != {expected}")
summary = None
summary_path = os.path.join(out, "summary.json")
if os.path.isfile(summary_path):
    summary = json.load(open(summary_path, encoding="utf-8"))
    if int(summary.get("steps", -1)) != steps:
        fail(f"summary reports {summary.get('steps')} steps, expected {steps}")
    if int(summary.get("tokens", -1)) != tokens_expected:
        fail(f"summary reports {summary.get('tokens')} tokens, expected {tokens_expected}")
log_path = os.path.join(out, "train.jsonl")
last_step = None
if os.path.isfile(log_path):
    for line in open(log_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            last_step = int(json.loads(line).get("step", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
print(json.dumps({
    "ok": True,
    "checkpoint": final,
    "sha256": actual,
    "bytes": os.path.getsize(final),
    "steps": steps,
    "tokens": tokens_expected,
    "summary_present": summary is not None,
    "summary_finished_utc": None if summary is None else summary.get("finished_utc"),
    "train_log_last_step": last_step,
}, sort_keys=True))
PY
}

# `court_log_from_receipt <receipt>`: the log path a court receipt records.
court_log_from_receipt() {
  python3 - "${1:-}" <<'PY'
import json, os, sys
path = sys.argv[1] if len(sys.argv) > 1 else ""
if path and os.path.exists(path):
    try:
        print(json.load(open(path, encoding="utf-8")).get("log") or "")
    except Exception:
        print("")
else:
    print("")
PY
}

# `eval_receipt_is_current <receipt> <checkpoint> <arm>`: succeed when a
# completed evaluation receipt already exists for exactly this checkpoint, so a
# death between the receipt write and the state transition cannot cost a second
# full evaluation. Prints one JSON object on success.
eval_receipt_is_current() {
  python3 - "${1:-}" "${2:-}" "${3:-}" <<'PY'
import json, os, sys

receipt_path, checkpoint, arm = sys.argv[1:4]
sidecar_path = checkpoint + ".sha256"
if not (receipt_path and os.path.isfile(receipt_path) and os.path.isfile(sidecar_path)):
    raise SystemExit(1)
try:
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    sidecar = json.load(open(sidecar_path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
model = receipt.get("model") or {}
results = receipt.get("results") or {}
expected = sidecar.get("sha256")
complete = {"distributional", "humaneval", "gsm8k"} <= set(results)
ok = bool(expected) and model.get("checkpoint_sha256") == expected \
    and (receipt.get("arm") or "") == arm and complete
if not ok:
    raise SystemExit(1)
print(json.dumps({"ok": True, "checkpoint_sha256": expected,
                  "receipt": receipt_path, "stamp": receipt.get("stamp"),
                  "seconds": receipt.get("seconds")}, sort_keys=True))
PY
}

# `compose_arm_receipt <arm> <run_name> <training_state> <exit_code>`: writes
# results/t2/<arm>/run.json once, then records the state transition. Safe to
# call after a harvest or after a completed training stage. Returns non-zero —
# and leaves the arm untrained — when the receipt could not be written.
compose_arm_receipt() {
  local arm="$1" run_name="$2" training_state="$3" exit_code="$4" started container_name
  started=$(t2p state-get --state "$state_abs" --field "arms.$arm.started_utc" || true)
  container_name=""
  if [[ -n "$training_state" ]]; then
    container_name=$(basename "$training_state" .json)
  fi
  t2p arm-receipt --arm "$arm" --run-name "$run_name" \
    --training-state "$training_state" --consume \
    --launch-revision "$(git rev-parse HEAD)" \
    --container "$container_name" \
    --court-receipt "${COURT_RECEIPT:-}" \
    --started-utc "$started" \
    --finished-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --exit-code "$exit_code" >>"$log_abs" 2>&1 || {
    log "[$arm] arm receipt could not be composed; the arm stays untrained"
    return 1
  }
  state_set "arms.$arm.status" '"trained"'
  state_set "arms.$arm.receipt" "\"results/t2/$arm/run.json\""
  state_set "arms.$arm.pending_state" '""'
  state_set "arms.$arm.container_exit_code" "$exit_code"
}

# ---------------------------------------------------------------------------
# 0. guards
# ---------------------------------------------------------------------------
log "T2 pilot starting: arms='$arms' budget=$budget_tokens steps=$steps seed=$seed resume=$resume"
if ((dry_run)); then
  log "DRY RUN: guards and phases are printed, nothing is launched"
fi

guard_rc=0
t2p guard --budget-tokens "$budget_tokens" | tee -a "$log_abs" >&2 || guard_rc=$?
if ((guard_rc == 3)); then
  log "guard refused the launch: see results/t2/launch-guard.json"
  log "the arithmetic and the disk figures are in that file; nothing was launched"
  exit 3
elif ((guard_rc != 0)); then
  log "guard failed with exit $guard_rc"
  exit "$guard_rc"
fi

# ---------------------------------------------------------------------------
# 1. state
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
if ((dry_run)); then
  log "DRY RUN [targets-build]: python3 oracle/train/teacher_cache.py build --tokens $budget_tokens"
  log "DRY RUN [targets-verify]: python3 oracle/train/teacher_cache.py verify"
  log "DRY RUN [eval-D0]: python3 oracle/eval/baseline.py --profile t1 --arm D --seed $seed --out $eval_root/t2-D0.json"
  for plan_arm in "${arm_list[@]}"; do
    plan_out="$run_root/$(arm_run_name "$plan_arm")"
    arm_argv "$plan_arm" "$plan_out"
    log "DRY RUN [$plan_arm train]: $(join_args "${ARM_ARGV[@]}")"
    log "DRY RUN [$plan_arm eval]: python3 oracle/eval/baseline.py --profile t2 --arm $plan_arm --seed $seed --checkpoint $plan_out/step-$(printf '%06d' "$steps").pt --config $(arm_config "$plan_arm") --admission $(arm_admission "$plan_arm") --out $eval_root/t2-$plan_arm.json"
  done
  log "DRY RUN [instrument]: python3 oracle/eval/instrument_check.py --student D0 --student <arm>:<checkpoint>:<config>:<admission>:t2 (x5)"
  log "DRY RUN [decision]: python3 oracle/eval/t2_decision.py --d0 ... --arm <arm>=<receipt> (x5) --out-dir results/t2 --seed 20261115 --resamples 10000"
  log "DRY RUN: no state is written and nothing is launched"
  exit 0
fi

if ((resume)) && [[ -f "$state_abs" ]]; then
  log "resuming from $state_rel"
  t2p state-init --state "$state_abs" --budget-tokens "$budget_tokens" \
    --steps "$steps" --seed "$seed" --budget-stamp 20261115T130456Z >/dev/null
  t2p state-archive-failure --state "$state_abs" | tee -a "$log_abs" >&2
else
  t2p state-init --state "$state_abs" --budget-tokens "$budget_tokens" \
    --steps "$steps" --seed "$seed" --budget-stamp 20261115T130456Z >/dev/null
  state_set status '"running"'
  state_set started_utc "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
fi
state_set status '"running"'

# ---------------------------------------------------------------------------
# 2. teacher targets
# ---------------------------------------------------------------------------
targets_ok=0
if ((force_targets)); then
  # An explicit rebuild: the operator wants a fresh cache at this budget.
  log "[targets] rebuild forced by --force-targets"
  targets_ok=0
elif t2p targets --budget-tokens "$budget_tokens" >/dev/null 2>&1; then
  targets_ok=1
fi

if ((targets_ok == 0)); then
  log "[targets] cache incomplete for $budget_tokens tokens; building"
  state_set phases.targets.status '"building"'
  state_set phases.targets.started_utc "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
  SELVEDGE_MEM="${SELVEDGE_MEM:-28g}" SELVEDGE_TMP="${SELVEDGE_TMP:-8g}" \
    court_run targets-build python3 oracle/train/teacher_cache.py build \
    --tokens "$budget_tokens"
fi

log "[targets] verifying the cache against a live teacher"
state_set phases.targets.status '"verifying"'
SELVEDGE_MEM="${SELVEDGE_MEM:-28g}" SELVEDGE_TMP="${SELVEDGE_TMP:-8g}" \
  court_run targets-verify python3 oracle/train/teacher_cache.py verify
targets_tokens=$(python3 - <<'PY'
import json
print(json.load(open("results/artifacts/train/teacher-targets.json", encoding="utf-8"))["tokens"])
PY
)
state_set phases.targets.status '"verified"'
state_set phases.targets.tokens "$targets_tokens"
state_set phases.targets.verified true
state_set phases.targets.verified_utc "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
state_set phases.targets.receipt "\"${COURT_RECEIPT:-}\""
log "[targets] verified, $targets_tokens tokens"

# ---------------------------------------------------------------------------
# 3. evaluation of the frozen reference D0 (never trained)
# ---------------------------------------------------------------------------
d0_done=$(t2p state-get --state "$state_abs" --field phases.eval_D0.status || true)
if [[ "$d0_done" != "done" ]]; then
  log "[eval-D0] the frozen equal-active dense reference"
  state_set phases.eval_D0.status '"running"'
  SELVEDGE_MEM="${SELVEDGE_MEM:-28g}" SELVEDGE_TMP="${SELVEDGE_TMP:-8g}" \
    court_run eval-D0 python3 oracle/eval/baseline.py --profile t1 --arm D --seed "$seed" \
    --out "$eval_root/t2-D0.json"
  t2p eval-receipt --arm D0 --eval "$eval_root/t2-D0.json" >>"$log_abs" 2>&1 || true
  state_set phases.eval_D0.status '"done"'
  state_set phases.eval_D0.receipt "\"$eval_root/t2-D0.json\""
  state_set phases.eval_D0.court_receipt "\"${COURT_RECEIPT:-}\""
  log "[eval-D0] done"
fi

# ---------------------------------------------------------------------------
# 4. arms, strictly sequential
# ---------------------------------------------------------------------------
for arm in "${arm_list[@]}"; do
  run_name=$(arm_run_name "$arm")
  out="$run_root/$run_name"
  eval_out="$eval_root/t2-$arm.json"
  status=$(t2p state-get --state "$state_abs" --field "arms.$arm.status" || true)
  status="${status:-pending}"

  if [[ "$status" == "evaluated" || "$status" == "skipped" ]]; then
    log "[$arm] already $status; skipping"
    continue
  fi

  mkdir -p "$out"

  # ---- training --------------------------------------------------------
  if [[ "$status" != "trained" && "$status" != "evaluating" ]]; then
    arm_argv "$arm" "$out"
    if ((dry_run)); then
      log "DRY RUN [$arm train]: $(join_args "${ARM_ARGV[@]}")"
      log "DRY RUN [$arm eval]: python3 oracle/eval/baseline.py --profile t2 --arm $arm --seed $seed --checkpoint $out/step-$(printf '%06d' "$steps").pt --config $(arm_config "$arm") --admission $(arm_admission "$arm") --out $eval_out"
      continue
    fi

    container_exit=0
    wait_rc=0
    pending=$(t2p state-get --state "$state_abs" --field "arms.$arm.pending_state" || true)
    COURT_RECEIPT=$(t2p state-get --state "$state_abs" --field "arms.$arm.court_receipt" || true)

    if ((from_eval)); then
      # The operator said: do not train. A verified final checkpoint is the only
      # thing that may reach evaluation; anything else stops here with a reason.
      if ! verified_final_checkpoint "$out" "$steps" >/dev/null 2>&1; then
        fail_now "train-$arm" 79 "--from-eval: no verified final checkpoint at $out"
      fi
      log "[$arm] --from-eval: verified final checkpoint, skipping training"
      verified_final_checkpoint "$out" "$steps" | tee -a "$log_abs" >&2
      state_set "arms.$arm.recovered" "{\"kind\": \"verified-final-checkpoint\", \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"reason\": \"--from-eval\"}"
      state_set "arms.$arm.status" '"trained"'
      COURT_RECEIPT="$(t2p state-get --state "$state_abs" --field "arms.$arm.court_receipt" || true)"
      container_exit=unobserved
    elif ((resume)); then
      if [[ -n "$pending" ]] && [[ -f "$pending" ]] &&
         docker inspect "$(basename "$pending" .json)" >/dev/null 2>&1; then
        # Path 1: the detached container is still attributable. `docker wait`
        # returns its exit code immediately when it has already stopped.
        log "[$arm] resuming the wait on $(basename "$pending")"
        state_set "arms.$arm.status" '"running"'
        set +e
        court_await "$pending"
        wait_rc=$?
        set -e
        if ((wait_rc != 0)); then
          if verified_final_checkpoint "$out" "$steps" >/dev/null 2>&1; then
            log "[$arm] court wait exited $wait_rc but the final checkpoint verifies; harvesting it"
            state_set "arms.$arm.container_failure" "{\"exit_code\": $wait_rc, \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"detail\": \"court wait non-zero; verified final checkpoint harvested\"}"
          else
            fail_now "train-$arm" "$wait_rc" "resumed court wait failed and no verified final checkpoint exists"
          fi
        fi
        container_exit=$wait_rc
      elif verified_final_checkpoint "$out" "$steps" >/dev/null 2>&1; then
        # Path 2: the orchestrator died after the container finished (or the
        # container was removed). The verified checkpoint is the artifact; do
        # not retrain it. There is no court attribution, and the receipt says so.
        log "[$arm] recovered a verified final checkpoint; skipping training"
        verified_final_checkpoint "$out" "$steps" | tee -a "$log_abs" >&2
        state_set "arms.$arm.recovered" "{\"kind\": \"verified-final-checkpoint\", \"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
        state_set "arms.$arm.status" '"trained"'
        COURT_RECEIPT=""
        container_exit=unobserved
      else
        # Path 3: no finished run. Resume the interrupted one from its latest
        # verified-safe checkpoint, or fail closed if only an unverified final
        # checkpoint is on disk.
        resume_from=$(latest_checkpoint "$out")
        final_checkpoint="$out/step-$(printf '%06d' "$steps").pt"
        if [[ "$resume_from" == "$final_checkpoint" ]]; then
          fail_now "train-$arm" 79 "$final_checkpoint exists but failed verification; refusing to resume from it"
        fi
        if [[ -n "$resume_from" ]]; then
          log "[$arm] resuming training from $resume_from"
          ARM_ARGV+=(--resume "$resume_from")
          state_set "arms.$arm.resumed_from" "\"$resume_from\""
        fi
        state_set "arms.$arm.status" '"running"'
        state_set "arms.$arm.started_utc" "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
        pending=$(court_launch "${ARM_ARGV[@]}") || fail_now "train-$arm" 69 "detached launch refused"
        state_set "arms.$arm.pending_state" "\"$pending\""
        log "[$arm] training launched: $(basename "$pending")"
        set +e
        court_await "$pending"
        wait_rc=$?
        set -e
        ((wait_rc == 0)) || fail_now "train-$arm" "$wait_rc" "training container failed"
        container_exit=0
      fi
    else
      state_set "arms.$arm.status" '"running"'
      state_set "arms.$arm.started_utc" "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
      pending=$(court_launch "${ARM_ARGV[@]}") || fail_now "train-$arm" 69 "detached launch refused"
      state_set "arms.$arm.pending_state" "\"$pending\""
      log "[$arm] training launched: $(basename "$pending")"
      set +e
      court_await "$pending"
      wait_rc=$?
      set -e
      ((wait_rc == 0)) || fail_now "train-$arm" "$wait_rc" "training container failed"
      container_exit=0
    fi

    # Persist the court attribution before composing the run receipt, so a
    # death between the wait and the receipt cannot lose it.
    if [[ -n "${COURT_RECEIPT:-}" ]]; then
      state_set "arms.$arm.court_receipt" "\"$COURT_RECEIPT\""
      state_set "arms.$arm.court_log" "\"$(court_log_from_receipt "$COURT_RECEIPT")\""
    fi
    state_set "arms.$arm.finished_utc" "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""

    if [[ ! -f "results/t2/$arm/run.json" ]]; then
      compose_arm_receipt "$arm" "$run_name" "${pending:-}" "${container_exit:-0}" ||
        fail_now "train-receipt-$arm" 70 "could not compose the arm receipt"
    else
      state_set "arms.$arm.status" '"trained"'
      log "[$arm] run receipt already present; reusing results/t2/$arm/run.json"
    fi
    log "[$arm] trained; receipt results/t2/$arm/run.json"
  else
    # Training was already recorded. Rebuild a missing receipt from the
    # artifacts if a previous process died between the state transition and the
    # write; never retrain.
    if [[ ! -f "results/t2/$arm/run.json" ]]; then
      pending=$(t2p state-get --state "$state_abs" --field "arms.$arm.pending_state" || true)
      COURT_RECEIPT=$(t2p state-get --state "$state_abs" --field "arms.$arm.court_receipt" || true)
      compose_arm_receipt "$arm" "$run_name" "${pending:-}" 0 ||
        fail_now "train-receipt-$arm" 70 "could not reconstruct the arm receipt"
    fi
  fi

  # ---- evaluation ------------------------------------------------------
  final_step=$(python3 -c "print(f'{int($steps):06d}')")
  checkpoint="$out/step-$final_step.pt"
  # Nothing unverified reaches evaluation, and nothing verified is retrained.
  verified_final_checkpoint "$out" "$steps" >/dev/null 2>&1 ||
    fail_now "eval-$arm" 79 "final checkpoint $checkpoint does not verify; refusing to evaluate"
  if current=$(eval_receipt_is_current "$eval_out" "$checkpoint" "$arm"); then
    log "[$arm] eval receipt already current for this checkpoint; not re-running"
    log "  $current"
  else
    state_set "arms.$arm.status" '"evaluating"'
    log "[$arm] evaluating $checkpoint"
    SELVEDGE_MEM="${SELVEDGE_MEM:-28g}" SELVEDGE_TMP="${SELVEDGE_TMP:-8g}" \
      court_run "eval-$arm" python3 oracle/eval/baseline.py --profile t2 --arm "$arm" --seed "$seed" \
      --checkpoint "$checkpoint" --config "$(arm_config "$arm")" \
      --admission "$(arm_admission "$arm")" --out "$eval_out" ||
      fail_now "eval-$arm" "$?" "evaluation failed"
  fi
  t2p eval-receipt --arm "$arm" --eval "$eval_out" >>"$log_abs" 2>&1 ||
    fail_now "eval-receipt-$arm" 70 "could not summarise the eval receipt"
  state_set "arms.$arm.status" '"evaluated"'
  state_set "arms.$arm.eval" "\"$eval_out\""
  state_set "arms.$arm.eval_summary" "\"results/t2/$arm/eval.json\""
  state_set "arms.$arm.eval_court_receipt" "\"${COURT_RECEIPT:-}\""
  log "[$arm] evaluated; summary results/t2/$arm/eval.json"
done

# ---------------------------------------------------------------------------
# 5. P2.4 instrument check over the final checkpoints
# ---------------------------------------------------------------------------
missing=0
for arm in D M X H 'H*'; do
  status=$(t2p state-get --state "$state_abs" --field "arms.$arm.status" || true)
  [[ "$status" == "evaluated" ]] || missing=1
done
if ((missing)); then
  log "[instrument] skipped: not every arm is evaluated (partial run)"
fi

if ((instrument && missing == 0)); then
  check_status=$(t2p state-get --state "$state_abs" --field phases.instrument.status || true)
  if [[ "${check_status:-pending}" != "done" ]]; then
    log "[instrument] P2.4 both-target check over the five final checkpoints"
    state_set phases.instrument.status '"running"'
    students=(--student D0)
    for arm in D M X H 'H*'; do
      run_name=$(arm_run_name "$arm")
      checkpoint="$run_root/$run_name/step-$(printf '%06d' "$steps").pt"
      students+=(--student "$arm:$checkpoint:$(arm_config "$arm"):$(arm_admission "$arm"):t2")
    done
    court_run instrument python3 oracle/eval/instrument_check.py "${students[@]}" ||
      fail_now instrument "$?" "the P2.4 instrument check failed"
    state_set phases.instrument.status '"done"'
    state_set phases.instrument.court_receipt "\"${COURT_RECEIPT:-}\""
  fi
fi

# ---------------------------------------------------------------------------
# 6. the pre-registered decision
# ---------------------------------------------------------------------------
if ((missing)); then
  log "[decision] skipped: a partial run cannot be decided"
  state_set status '"partial"'
  log "pilot paused with missing arms; state: $state_rel"
  exit 0
fi
log "[decision] applying the registered decision rule"
state_set phases.decision.status '"running"'
court_run decision python3 oracle/eval/t2_decision.py \
  --d0 "$eval_root/t2-D0.json" \
  --arm "D=$eval_root/t2-D.json" \
  --arm "M=$eval_root/t2-M.json" \
  --arm "X=$eval_root/t2-X.json" \
  --arm "H=$eval_root/t2-H.json" \
  --arm "H*=$eval_root/t2-H*.json" \
  --out-dir results/t2 --seed 20261115 --resamples 10000 ||
  fail_now decision "$?" "the decision analysis failed"
state_set phases.decision.status '"done"'
state_set phases.decision.court_receipt "\"${COURT_RECEIPT:-}\""

state_set status '"done"'
state_set finished_utc "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
log "pilot complete; state: $state_rel; report: results/T2-PILOT.md"
log "next: scripts/court commit_identity, scripts/court perf_numbers, then commit"
