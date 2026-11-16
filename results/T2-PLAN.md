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
  --out results/p8/composition-eval-20261116T000000Z.json
```

The decisive statistic is `gate-minus-sum` in the receipt (contract P8.2.4).

### 6.2 P11.1 — causal bottleneck repair

CPU fixture check (no GPU, writes a fixture receipt):

```sh
scripts/court oracle -- python3 oracle/eval/causal_repair.py \
  --mode self-test --out results/p11/fixture-20261116T000000Z.json
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
  --out results/p11/causal-repair-20261116T000000Z.json
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
  --out results/p9/combination-20261116T000000Z.json
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
