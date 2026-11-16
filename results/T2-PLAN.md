# T2 plan — the five-arm error-directed capacity matrix

Status: **registered, instrument-checked, smoke-tested, ready to launch.** The
registration is `docs/EXPERIMENT-CONTRACT.md` T2 (commit `46a21c5`, before any
measured comparison, amended by A8/A9). No full-budget run has started. Every
number in the tables below is generated from
`results/train/budget-20261115T130456Z.json` by `scripts/perf_numbers.py`;
nothing here is typed by hand.

## 1. Arms and budgets

One frozen Qwen3-0.6B core (`596,049,920` active), one data order, one token
budget, one optimiser, one seed (`1001`). Active and accessible counts are from
the Rust admission records in `results/train/admission/t2/`; T2's `D` is T1's
`C`, and T1's `D` is `D0` here (contract T2.1).

| arm | path | active unique | total accessible | steps |
| --- | --- | ---: | ---: | ---: |
| `D` | frozen core + dense width-2048 adapter | 772,210,688 | 772,210,688 | 1024 |
| `M` | frozen core + hashed memory at 4 layers | 598,153,216 | 866,586,624 | 1024 |
| `X` | frozen core + 8 specialists/layer, `k=4` | 772,440,064 | 948,600,832 | 1024 |
| `H` | `X` + `M`, distillation router + load balance | 774,543,360 | 1,219,137,536 | 1024 |
| `H*` | `H` with a utility-trained router | 774,543,360 | 1,219,137,536 | 1024 |

Budget per arm:

<!-- BEGIN GENERATED: t2-plan-budget-tokens -->
67108864
<!-- END GENERATED: t2-plan-budget-tokens -->

tokens = 1024 steps x 65,536 tokens, derived by T2.7 from the measured slowest
arm. Replication: `H*`, `H`, `D` at seeds `2001-2003` and a quarter budget
(16,777,216 tokens) each (A8).

## 2. Measured throughput and memory

Measured on the RTX 4080 SUPER in the GPU court, training steps only,
deterministic kernels, 512-position micro-forwards. `hours` is the time the
declared budget needs at that rate; `resume delta` is
`max |loss_full - loss_resumed|` over 25 continuation steps.

Each number is a generated region sourced from
`results/train/budget-20261115T130456Z.json`; the receipt is the single source.

**D** (dense control)

- tokens/s:

  <!-- BEGIN GENERATED: t2-plan-D-tokens-per-second -->
7943
  <!-- END GENERATED: t2-plan-D-tokens-per-second -->

- VRAM peak (GiB):

  <!-- BEGIN GENERATED: t2-plan-D-vram -->
5.45
  <!-- END GENERATED: t2-plan-D-vram -->

- hours at the declared budget:

  <!-- BEGIN GENERATED: t2-plan-D-hours -->
2.35
  <!-- END GENERATED: t2-plan-D-hours -->

- resume delta:

  <!-- BEGIN GENERATED: t2-plan-D-resume-delta -->
0.0e+00
  <!-- END GENERATED: t2-plan-D-resume-delta -->

**M** (memory only)

- tokens/s:

  <!-- BEGIN GENERATED: t2-plan-M-tokens-per-second -->
9320
  <!-- END GENERATED: t2-plan-M-tokens-per-second -->

- VRAM peak (GiB):

  <!-- BEGIN GENERATED: t2-plan-M-vram -->
4.21
  <!-- END GENERATED: t2-plan-M-vram -->

- hours at the declared budget:

  <!-- BEGIN GENERATED: t2-plan-M-hours -->
2.00
  <!-- END GENERATED: t2-plan-M-hours -->

- resume delta:

  <!-- BEGIN GENERATED: t2-plan-M-resume-delta -->
0.0e+00
  <!-- END GENERATED: t2-plan-M-resume-delta -->

**X** (expert only)

- tokens/s:

  <!-- BEGIN GENERATED: t2-plan-X-tokens-per-second -->
5035
  <!-- END GENERATED: t2-plan-X-tokens-per-second -->

- VRAM peak (GiB):

  <!-- BEGIN GENERATED: t2-plan-X-vram -->
8.82
  <!-- END GENERATED: t2-plan-X-vram -->

- hours at the declared budget:

  <!-- BEGIN GENERATED: t2-plan-X-hours -->
3.70
  <!-- END GENERATED: t2-plan-X-hours -->

- resume delta:

  <!-- BEGIN GENERATED: t2-plan-X-resume-delta -->
0.0e+00
  <!-- END GENERATED: t2-plan-X-resume-delta -->

**H** (naive hybrid)

- tokens/s:

  <!-- BEGIN GENERATED: t2-plan-H-tokens-per-second -->
4767
  <!-- END GENERATED: t2-plan-H-tokens-per-second -->

- VRAM peak (GiB):

  <!-- BEGIN GENERATED: t2-plan-H-vram -->
10.86
  <!-- END GENERATED: t2-plan-H-vram -->

- hours at the declared budget:

  <!-- BEGIN GENERATED: t2-plan-H-hours -->
3.91
  <!-- END GENERATED: t2-plan-H-hours -->

- resume delta:

  <!-- BEGIN GENERATED: t2-plan-H-resume-delta -->
0.0e+00
  <!-- END GENERATED: t2-plan-H-resume-delta -->

**H*** (error-directed hybrid)

- tokens/s:

  <!-- BEGIN GENERATED: t2-plan-Hstar-tokens-per-second -->
5012
  <!-- END GENERATED: t2-plan-Hstar-tokens-per-second -->

- VRAM peak (GiB):

  <!-- BEGIN GENERATED: t2-plan-Hstar-vram -->
10.85
  <!-- END GENERATED: t2-plan-Hstar-vram -->

- hours at the declared budget:

  <!-- BEGIN GENERATED: t2-plan-Hstar-hours -->
3.72
  <!-- END GENERATED: t2-plan-Hstar-hours -->

- resume delta:

  <!-- BEGIN GENERATED: t2-plan-Hstar-resume-delta -->
0.0e+00
  <!-- END GENERATED: t2-plan-Hstar-resume-delta -->

Total for one seed of all five arms:

<!-- BEGIN GENERATED: t2-plan-total-gpu-hours -->
15.68
<!-- END GENERATED: t2-plan-total-gpu-hours -->

GPU-hours. The full smoke evidence, including
the identical-checkpoint determinism result and the resource blocks, is in
`results/train/smoke-20261115T103325Z.json`,
`results/train/smoke-20261115T114908Z.json` and
`results/train/smoke-20261115T125618Z.json`.

Bytes per token (modelled at bf16, never presented as measured I/O) are
generated per arm in `results/T2-PILOT.md` §2 from the decision receipt.
They follow from the declared slices: `8 x 256 x 2` value bytes,
`4 x 524,288 x 2` projection bytes, `4 x 1,024 x 2` gate bytes, and the
`k = 4` expert-plus-router read at bf16. The memory table is resident at
`536,870,912` bytes (bf16) and is **not** charged per token.

**Correction, 2026-11-15 (S8b), after this plan was committed and the pilot
launched.** The table that stood here listed `352,550,912` in its "expert weight
bytes (+router)" column for `X`, `H` and `H*`. That quantity is the specialist
bank's *element count* at the full eight-specialist width: not a byte count, and
not the read the deployed `k = 4` router performs. The column is removed rather
than restated, because a byte figure is a measurement of a declared dtype and
belongs in a generated region — which `results/T2-PILOT.md` §2 provides, with
the whole-bank figure reported separately from the per-token read. The cells
escaped the perf-numbers court precisely because they were hand-written. No arm,
threshold, budget, seed or kill condition is changed by this correction; see the
S8b entry in `docs/WORK-LOG.md`.

## 3. Exact launch commands

All commands run in the GPU court. The environment is identical for every arm:

```sh
export SELVEDGE_MEM=28g SELVEDGE_TMP=8g
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
scripts/train-admission-t2.sh     # configs + Rust admission + mirror check
```

Then one command per arm (replace the arm name; `D` shown):

```sh
scripts/court --gpu oracle -- python3 oracle/train/loop.py \
  --arm D --profile t2 \
  --config results/train/admission-configs/t2/D.json \
  --admission results/train/admission/t2/D.json \
  --out results/artifacts/train/t2/D --steps 1024 \
  --micro-batch 1 --accum 64 --micro-positions 512 \
  --checkpoint-every 100 --zero-point --deterministic
```

`M`: identical with `M` in the three paths. `X`: identical, plus
`--load-balance`. `H`: identical to `X`, plus `--warmup-steps 64`. `H*`:

```sh
scripts/court --gpu oracle -- python3 oracle/train/loop.py \
  --arm 'H*' --profile t2 \
  --config 'results/train/admission-configs/t2/H*.json' \
  --admission 'results/train/admission/t2/H*.json' \
  --out results/artifacts/train/t2/Hstar --steps 1024 \
  --micro-batch 1 --accum 64 --micro-positions 512 \
  --checkpoint-every 100 --zero-point --deterministic \
  --warmup-steps 64 --calibration-windows 32 --router-steps 200 \
  --labels-out results/artifacts/train/t2/Hstar/utility-labels.pt
```

`H*` labels at step 64 (its warm-up), then freezes the router and continues.
The label phase is 28 layers x 9 passes x 32,768 tokens; its manifest and router
trajectories are written to `router-phase.json` in the output directory.

Full evaluation, after each arm's final checkpoint:

```sh
scripts/court --gpu oracle -- python3 oracle/eval/baseline.py \
  --profile t2 --arm D \
  --checkpoint results/artifacts/train/t2/D/step-001024.pt \
  --config results/train/admission-configs/t2/D.json \
  --admission results/train/admission/t2/D.json \
  --out results/eval/t2-D.json
```

The evaluation path itself is exercised by
`results/eval/t2-M-smoke.json` and `results/eval/t2-Hstar-smoke.json` (five items
per task, two held-out windows per domain), both of which carry the arm's
active/accessible counts and the bytes/token resource block.

and the P2.4 safeguard (both targets on the held-out sample) for all five:

```sh
scripts/court --gpu oracle -- python3 oracle/eval/instrument_check.py \
  --student D0 \
  --student 'D:results/artifacts/train/t2/D/step-001024.pt:results/train/admission-configs/t2/D.json:results/train/admission/t2/D.json:t2' \
  --student 'M:results/artifacts/train/t2/M/step-001024.pt:results/train/admission-configs/t2/M.json:results/train/admission/t2/M.json:t2' \
  --student 'X:results/artifacts/train/t2/X/step-001024.pt:results/train/admission-configs/t2/X.json:results/train/admission/t2/X.json:t2' \
  --student 'H:results/artifacts/train/t2/H/step-001024.pt:results/train/admission-configs/t2/H.json:results/train/admission/t2/H.json:t2' \
  --student 'H*:results/artifacts/train/t2/Hstar/step-001024.pt:results/train/admission-configs/t2/H*.json:results/train/admission/t2/H*.json:t2'
```

Replication seeds (same commands at `--steps 256 --seed 2001|2002|2003` for
`H*`, `H`, `D`, with `--warmup-steps 16` for `H*` and `--calibration-windows 32
--router-steps 200`).

## 4. What each outcome would mean

The decision order and thresholds are contract T2.9/T2.10; this section only
restates the mapping from result to consequence.

| result | consequence |
| --- | --- |
| `H*` > `H` on the primary statistic by >= 3% (CI lower bound > 0), no regression on the other endpoint, **and** `H*` > `D` by >= 5% | K15 and K16 pass: error-directed conditional capacity is supported at this scale. Scaling the bank (P2.2) and P3.5 phases C/D become the next experiments. |
| `H*` > `H` but not `D` | K15 passes, K16 fails: error direction is a real routing effect but conditional capacity does not beat dense capacity of equal active size at this budget. No bank scaling; the memory and expert paths continue separately only if K17/K18 pass. |
| `H*` does not beat `H` | K15 fails: error-directed routing is not supported at this scale. WS3's central bet is recorded as a negative result; the error-directed mechanism is not carried into WS4/WS5 without a new pre-registration. |
| `X` does not beat `D` (K18) | routed specialists lose to a dense adapter of the same active size, replicating T1's K13; the hybrid result is interpreted as "memory plus a losing expert path". |
| `M` beats `D0` but not the `D0 -> D` gap (K17) | memory is a cheap capacity store, not a substitute for active capacity; no memory ladder. |
| `M` does not beat `D0` | the memory path is retired at this scale. |
| every arm inside noise of `D0` | the budget or adapter capacity is too small for any effect; reported as such, not as a claim about the architecture. |

## 5. Risks that could invalidate the pilot

- **One seed.** The pilot is one seed per arm unless the replication runs. Any
  margin inside twice the item-level CI half-width is reported as "not
  separable from noise" (T2.8).
- **The utility labels are one-shot.** `u_e` is measured once, in the full-bank
  reference set, and the router is frozen afterwards. An expert that duplicates
  another is under-penalised, and periodic relabelling is deferred (T2.5/T2.6).
- **The compact training target is 0.397 nats/token optimistic** (A9). The bias
  is nearly constant across students, but a ranking reversal inside a CI cannot
  be excluded; evaluation therefore reports both targets.
- **Cold start.** The first 64 steps train the naive router, so `H*` measures
  "utility routing after a naive warm-up", not utility routing from step 0.
- **Compliance is a union count.** Every arm is admitted by the Rust ledger,
  but the `D`/`X`/`H`/`H*` active counts differ by up to 2.3M elements (0.3%);
  the comparison is matched on tokens and training compute, not exactly on
  active parameters, and the difference favours the hybrids' cost.
- **The 16 GiB device.** `H` and `H*` peak near the measured VRAM in §2 with the
  bf16 memory table; a driver or cuBLAS change that raises the footprint would
  fail closed (OOM) rather than silently change the run.
- **Token budget.** The declared budget is what 5 hours per arm buys on this
  GPU. If that is too small for the effect, the result is "budget too small",
  which T2.11 pre-registers as a distinct outcome from "the architecture
  fails".

## 6. Successor courts ready to launch (P8.2, P11.1, T3, P9.3)

These are registered in `docs/EXPERIMENT-CONTRACT.md` as separate sections; they
do not alter T2. They run **only after the pilot releases the GPU**. Each carries
its own receipt; none replaces a T2 number. Every command runs in the GPU court
with the same hard limits as the pilot.

Environment for all of them:

```sh
export SELVEDGE_MEM=28g SELVEDGE_TMP=8g
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 6.1 P8.2 — bounded-gate composition of `D` and `M`

Prerequisite: the T2 `D` and `M` final checkpoints. Admission first (the Rust
ledger decides the union):

```sh
scripts/p8-composition-admission.sh
```

Then gate-only training on the T2 calibration slice (256 steps, 262,144 tokens):

```sh
scripts/court --gpu oracle -- python3 oracle/train/composition.py --run-composition \
  --config-d results/train/admission-configs/t2/D.json \
  --admission-d results/train/admission/t2/D.json \
  --config-m results/train/admission-configs/t2/M.json \
  --admission-m results/train/admission/t2/M.json \
  --checkpoint-d results/artifacts/train/t2/D/step-001024.pt \
  --checkpoint-m results/artifacts/train/t2/M/step-001024.pt \
  --out results/artifacts/train/p8/composition \
  --steps 256 --kappa 0.05 --gate-lr 3e-3 --seed 1001 \
  --calibration-windows 32 --positions 1024 \
  --hf-root /hf-cache --artifact-dir results/artifacts/train --deterministic
```

Score the four references and the per-item oracle on all 26 fresh items:

```sh
scripts/court --gpu oracle -- python3 oracle/train/composition.py --score-composition \
  --config-d results/train/admission-configs/t2/D.json \
  --admission-d results/train/admission/t2/D.json \
  --config-m results/train/admission-configs/t2/M.json \
  --admission-m results/train/admission/t2/M.json \
  --checkpoint-d results/artifacts/train/t2/D/step-001024.pt \
  --checkpoint-m results/artifacts/train/t2/M/step-001024.pt \
  --gate-checkpoint results/artifacts/train/p8/composition/gate-000256.pt \
  --hf-root /hf-cache \
  --out "results/p8/composition-eval-$(date -u +%Y%m%dT%H%M%SZ).json"
```

The decisive statistic is `gate-minus-sum` in the receipt (contract P8.2.4).

### 6.2 P11.1 — causal bottleneck repair

CPU fixture check (no GPU, writes a fixture receipt):

```sh
scripts/court oracle -- python3 oracle/eval/causal_repair.py \
  --mode self-test --out "results/p11/fixture-$(date -u +%Y%m%dT%H%M%SZ).json"
```

The 0.6B run needs the failed-item artifact `--items` (a JSON list of
`{id, input_ids, label}`; contract P11.1.1). **This artifact does not exist at
this revision**; producing it from the `P12.9` suite is the remaining piece of
engineering. With it (`--layers all` means the pre-registered 8-layer candidate
set and the run uses 60 oracle steps per item/layer, per the contract):

```sh
scripts/court --gpu oracle -- python3 oracle/eval/causal_repair.py --mode checkpoint \
  --items results/p11/failed-items.json \
  --model-size 0.6b --hf-root /hf-cache \
  --layers all --oracle-variant surrogate --seed 1001 \
  --out "results/p11/causal-repair-$(date -u +%Y%m%dT%H%M%SZ).json"
```

### 6.3 T3 — latent-geometry sweep

Arithmetic and specs (CPU; no model):

```sh
scripts/court oracle -- python3 oracle/train/latent_geometry.py --sweep \
  --out results/t3/sweep.json --emit-specs results/t3/specs
```

Per point `L` in `128 256 384 512`: emit the admission config, admit it with the
Rust ledger, train on the T2 data order and budget, then measure.

```sh
scripts/court oracle -- python3 oracle/train/latent_geometry.py \
  --emit-config results/train/admission-configs/t3/L128.json \
  --spec results/t3/specs/L128.json

scripts/court rust -- sh -c '
cargo build --release -q -p selvedge-cli
mkdir -p results/train/admission/t3
./target/release/selvedge admit \
  --config results/train/admission-configs/t3/L128.json \
  --out results/train/admission/t3/L128.json
'

scripts/court --gpu oracle -- python3 oracle/train/latent_geometry.py --train \
  --spec results/t3/specs/L128.json \
  --config results/train/admission-configs/t3/L128.json \
  --admission results/train/admission/t3/L128.json \
  --out results/artifacts/train/t3/L128 --steps 1024 \
  --micro-batch 1 --accum 64 --micro-positions 512 \
  --checkpoint-every 100 --zero-point --deterministic

scripts/court --gpu oracle -- python3 oracle/train/latent_geometry.py --report \
  --spec results/t3/specs/L128.json \
  --checkpoint results/artifacts/train/t3/L128/step-001024.pt \
  --measure-utilities --calibration-positions 1024 \
  --utilities-out results/t3/utilities-L128.pt \
  --out results/t3/report-L128.json

scripts/court --gpu oracle -- python3 oracle/eval/baseline.py --profile t3 \
  --arm T3-L128 --latent-spec results/t3/specs/L128.json \
  --checkpoint results/artifacts/train/t3/L128/step-001024.pt \
  --config results/train/admission-configs/t3/L128.json \
  --admission results/train/admission/t3/L128.json \
  --out results/eval/t3-L128.json
```

The other three points are identical with `L256`, `L384`, `L512`. The reference
point is the existing T2 `X` evaluation (`results/eval/t2-X.json`), re-scored in
the same session by the same instrument if the receipt is older than the code.

### 6.4 P9.3 — combination utility on the hybrid arm

Prerequisite: the T2 `H*` checkpoint (preferred) or `H`. 64 calibration windows,
four conditions, controller comparison:

```sh
scripts/court --gpu oracle -- python3 oracle/eval/combination_utility.py \
  --mode checkpoint --arm 'H*' \
  --checkpoint 'results/artifacts/train/t2/Hstar/step-001024.pt' \
  --config 'results/train/admission-configs/t2/H*.json' \
  --admission 'results/train/admission/t2/H*.json' \
  --calibration-windows 64 --artifact-dir results/artifacts/train \
  --out "results/p9/combination-$(date -u +%Y%m%dT%H%M%SZ).json"
```

If `H*` is unavailable, the same command with `--arm H` and the `H` paths. The
receipt names the arm and the checkpoint hash.

### 6.5 Ordering and constraints

- P8.2 needs `D` and `M` only; it can run immediately after the pilot.
- P9.3 needs `H*` (or `H`); it is the last of these four in the queue.
- T3 trains four arms at the full T2 budget (`1024` steps each). On the measured
  throughput envelope that is roughly four times one T2 arm; run the four points
  serially, never concurrently, and record each point's VRAM before the next.
- No command above may run while the pilot's container is alive.

## 7. Post-`H*` execution and verdict pipeline

The GPU window between "`H*` finished" and "T2 has a verdict" is spent only on
measurements. Everything that can be prepared before the window is prepared and
CPU-tested now: the evaluation commands, the capability court commands with the
provenance amendment, the registered sensitivity (`S1`), the verdict reader and
the cross-arm comparison. The orchestrator is `scripts/post-hstar.sh`, the stage
logic is `scripts/post_hstar.py`, and the state machine is resumable.

### 7.1 The one command

```sh
scripts/post-hstar.sh --resume
```

It runs the stages below in order, stops on the first hard failure with the reason
recorded, and skips every stage already recorded `done` in
`results/t2/post-hstar-state.json`. `--dry-run` prints the exact argv of every
stage without launching anything. `--stage NAME` runs one stage (repeatable).
`--fresh` discards the state and starts over.

Prerequisites it checks and refuses on:

- a heavy GPU process (`>= 512 MiB`) or more than `3000 MiB` of GPU memory in use;
- a live container whose name starts with `selvedge-`;
- `MemAvailable` below the largest stage limit plus a `2 GiB` reserve;
- `docker` or the daemon unavailable (there is no host-execution fallback);
- `systemd-run` unavailable, unless `--runner direct` is passed and recorded.

The preflight arithmetic and the GPU/container probe are written to
`results/t2/post-hstar-preflight-<stamp>.json` whether it proceeds or refuses.

### 7.2 Stages, guards and stop conditions

| # | stage | what it does | guard that stops it |
| ---: | --- | --- | --- |
| 1 | `preflight` | host memory, GPU-idle and container arithmetic; per-stage limits | GPU busy, container alive, memory short, no docker/systemd |
| 2 | `verify` | `H*` checkpoint sidecar, summary steps/tokens, utility-label sidecar, training receipt | any digest mismatch; missing summary; missing labels; no training receipt and no pending court state to harvest one |
| 3 | `eval` | `H*` on the registered instrument; then `results/t2/H*/eval.json` | receipt absent, incomplete, for another checkpoint, or another arm |
| 4 | `instrument` | P2.4 both-target check over `D0` and the five arms | no instrument receipt written |
| 5 | `capability` | `generate` (GPU) then `check` (Rust court) for `D`, `M`, `X`, `H`, `H*` | result missing the registration digest, the category minimum, or the routed-arm provenance |
| 6 | `sensitivity` | `S1`: re-evaluate `X` and `H` with `mask_order=pair` | receipt not `pair`, wrong checkpoint, wrong arm |
| 7 | `verdict` | `scripts/t2_verdict.py --write` | any missing/mismatched receipt, missing provenance, or a `S1` receipt that is not `pair` |
| 8 | `compare` | `scripts/capability_compare.py` across arms | unregistered result, digest/suite/threshold mismatch, missing routed-arm provenance |

Every stage writes `status`, `started_utc`, `finished_utc`, the git revision, the
seed, the config digest(s), the admitted active-parameter count, the measured
VRAM peak where one exists, the host memory limit, the limit's arithmetic, its
court calls and its outputs. A failed stage writes a `failure` block and the
pipeline exits with that stage's code; nothing downstream runs.

### 7.3 Exact per-stage commands

The orchestrator builds these argv lists; they are listed so a stage can be
re-run by hand without guessing.

Verify and evaluate `H*`:

```sh
SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 oracle/eval/baseline.py --profile t2 \
  --arm 'H*' --seed 1001 \
  --checkpoint 'results/artifacts/train/t2/Hstar/step-001024.pt' \
  --config 'results/train/admission-configs/t2/H*.json' \
  --admission 'results/train/admission/t2/H*.json' \
  --out 'results/eval/t2-H*.json'

python3 scripts/t2_pilot.py eval-receipt --arm 'H*' --eval 'results/eval/t2-H*.json'
```

The P2.4 check over the six students:

```sh
SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 oracle/eval/instrument_check.py \
  --student D0 \
  --student 'D:results/artifacts/train/t2/D/step-001024.pt:results/train/admission-configs/t2/D.json:results/train/admission/t2/D.json:t2' \
  --student 'M:results/artifacts/train/t2/M/step-001024.pt:results/train/admission-configs/t2/M.json:results/train/admission/t2/M.json:t2' \
  --student 'X:results/artifacts/train/t2/X/step-001024.pt:results/train/admission-configs/t2/X.json:results/train/admission/t2/X.json:t2' \
  --student 'H:results/artifacts/train/t2/H/step-001024.pt:results/train/admission-configs/t2/H.json:results/train/admission/t2/H.json:t2' \
  --student 'H*:results/artifacts/train/t2/Hstar/step-001024.pt:results/train/admission-configs/t2/H*.json:results/train/admission/t2/H*.json:t2'
```

Capability court, one arm per pair of commands. `X` and `H` take the certified
provenance amendment; `D` and `M` carry no dispatch and do not. `H*`'s mask order
is read from its own training receipt, so the orchestrator passes what the
receipt records rather than a default.

```sh
SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 scripts/capability_court.py generate \
  --t2-arm X \
  --t2-checkpoint 'results/artifacts/train/t2/X/step-001024.pt' \
  --t2-config 'results/train/admission-configs/t2/X.json' \
  --t2-admission 'results/train/admission/t2/X.json' \
  --t2-training-receipt 'results/t2/X/run.json' \
  --t2-mask-order legacy \
  --provenance-amendment results/t2/PROVENANCE-AMENDMENT-X-and-H.json \
  --preregistration docs/capability-preregistration-v3.json \
  --out 'results/capability/runs/t2-X/generations.json'

SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \
  scripts/court rust -- python3 scripts/capability_court.py check \
  --generations 'results/capability/runs/t2-X/generations.json' \
  --preregistration docs/capability-preregistration-v3.json \
  --out 'results/capability/runs/t2-X/result.json'
```

Substitute `ARM`/`RUN` for the other four. `H*` is `Hstar` on disk and `H*` in
the config name. `D` and `M` keep `--t2-mask-order legacy` and simply omit
`--provenance-amendment`: the adapter's mask check applies only to routed arms,
and those two have no dispatch.

The registered sensitivity `S1` (contract T2.12 A10), evaluation only:

```sh
SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 oracle/eval/baseline.py --profile t2 \
  --arm X --seed 1001 \
  --checkpoint 'results/artifacts/train/t2/X/step-001024.pt' \
  --config 'results/train/admission-configs/t2/X.json' \
  --admission 'results/train/admission/t2/X.json' \
  --mask-order pair \
  --out 'results/t2/X/eval-pair-mask.json'
```

and the same with `ARM=H`, `SELVEDGE_MEM=24g`. The registered receipts
`results/eval/t2-X.json` and `results/eval/t2-H.json` are never edited.

The verdict and the comparison:

```sh
SELVEDGE_MEM=8g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \
  scripts/court python -- python3 scripts/t2_verdict.py --write \
  --d0 results/eval/t2-D0.json \
  --sensitivity X=results/t2/X/eval-pair-mask.json \
  --sensitivity H=results/t2/H/eval-pair-mask.json \
  --out results/t2-verdict.json --doc results/T2-VERDICT.md

SELVEDGE_MEM=8g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \
  scripts/court python -- python3 scripts/capability_compare.py \
  --result D=results/capability/runs/t2-D/result.json \
  --result M=results/capability/runs/t2-M/result.json \
  --result X=results/capability/runs/t2-X/result.json \
  --result H=results/capability/runs/t2-H/result.json \
  --result 'H*=results/capability/runs/t2-Hstar/result.json' \
  --preregistration docs/capability-preregistration-v3.json \
  --out results/capability/runs/t2-comparison.json
```

### 7.4 Memory limits and their arithmetic

Every stage's limit is derived from that workload's own measured VRAM peak:
`limit = bucket(1.5 x peak)`, buckets `{16, 24, 32, 48}` GiB, with a floor of
`16 GiB` for `D`/`M`/`X` and `24 GiB` for the hybrid arms and the six-student
instrument check. No stage inherits the `8 GiB` documentation default.

| workload | derived limit |
| --- | ---: |
| capability generate, `D` | 16g |
| capability generate, `M` | 16g |
| capability generate, `X` | 24g |
| capability generate, `H` | 24g |
| capability generate and eval, `H*` | 24g |
| P2.4 instrument check (six students) | 24g |
| capability check (Rust court) | 12g |
| verdict, comparison (CPU readers) | 8g |

The measured VRAM peak each limit was derived from is not restated here: it is
read from the arm's own receipt at launch and written, with the multiplication
and the bucket, into that stage's `sizing.arithmetic` in
`results/t2/post-hstar-state.json` and into every call record. `X` is the one
non-hybrid arm whose 1.5x headroom does not fit the `16 GiB` floor, so the table
says `24g` for it rather than rounding down; that is why the rule is a formula
and not a fixed table. `MemAvailable` is re-checked immediately before every
stage (`memory_guard`), not only at preflight, and the check is recorded in that
stage's state entry.

GPU stages launch under `systemd-run --user --collect --unit=selvedge-post-hstar-<stage>-<stamp>`
with `MemoryMax` and `MemorySwapMax` equal to the same limit the court applies
inside the container, and `TasksMax`/`CPUQuota` set from the same argv. `--runner
direct` exists for a host without a user systemd and for the CPU tests; it is
recorded in the state and in every call record.

### 7.5 Failure and resume semantics

- State: `results/t2/post-hstar-state.json`; log: `results/t2/post-hstar.log`.
- A stage is `pending` → `running` → `done` or `failed`.
- A resume archives the previous `failure` into `failures` (never deletes it) and
  re-runs only stages that are not `done`.
- A failure stops the pipeline with the failing stage's exit code, the command and
  the reason. A verdict is never emitted from a partial receipt set: the reader
  refuses with a named reason and writes nothing.
- Nothing may run while the training container is alive. Poll it read-only with
  `scripts/poll-t2-pilot.sh`; never signal it.

### 7.6 The CPU tests that gate this pipeline

```sh
SELVEDGE_MEM=8g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \
  scripts/court python -m pytest \
  oracle/tests/test_t2_verdict.py \
  oracle/tests/test_capability_compare.py \
  oracle/tests/test_post_hstar.py
```

Named refusals and behaviours they pin: missing receipt, config-hash mismatch,
admission-hash mismatch, checkpoint-sidecar mismatch, missing provenance
amendment, amendment that does not bind the receipt, `S1` receipt that is not
`pair`, unregistered capability result, mismatched pre-registration digest,
result on another suite, result scored against another threshold, result below
the category minimum, an aggregate win that fails a category regression, an arm
with lower compute but worse quality, a resumable mid-stage failure, GPU-busy /
low-memory / live-container preflight refusals, a capability `generate` failure,
and a dry run that launches nothing.
