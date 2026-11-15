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

Budget: **67,108,864 tokens per arm** = 1024 steps x 65,536 tokens, derived by
T2.7 from the measured slowest arm. Replication: `H*`, `H`, `D` at seeds
`2001-2003` and 16,777,216 tokens each (A8).

## 2. Measured throughput and memory

Measured on the RTX 4080 SUPER (16,376 MiB) in the GPU court, training steps
only, deterministic kernels, 512-position micro-forwards. `hours` is the time
the declared 67,108,864-token budget needs at that rate; `resume delta` is
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

Total for one seed of all five arms: **15.68 GPU-hours** (the receipt's
`/common/total_gpu_hours_at_common_budget`). The full smoke evidence, including
the identical-checkpoint determinism result and the resource blocks, is in
`results/train/smoke-20261115T103325Z.json`,
`results/train/smoke-20261115T114908Z.json` and
`results/train/smoke-20261115T125618Z.json`.

Bytes per token (modelled at bf16, never presented as measured I/O):

| arm | memory value bytes | memory projection bytes | memory gate bytes | expert weight bytes (+router) |
| --- | ---: | ---: | ---: | ---: |
| `M` | 4,096 | 4,194,304 | 8,192 | 0 |
| `X` | 0 | 0 | 0 | 352,550,912 |
| `H`, `H*` | 4,096 | 4,194,304 | 8,192 | 352,550,912 |

These follow from the declared slices: `8 x 256 x 2` value bytes,
`4 x 524,288 x 2` projection bytes, `4 x 1,024 x 2` gate bytes, and
`(28 x 4 x 3 x 512 x 1024 + 229,376) x 2` expert-plus-router bytes. The memory
table is resident at 536,870,912 bytes (bf16) and is **not** charged per token.

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
- **The 16 GiB device.** `H` and `H*` peak at about 10.9 GiB with the bf16
  memory table; a driver or cuBLAS change that raises the footprint would fail
  closed (OOM) rather than silently change the run.
- **Token budget.** 67,108,864 tokens is what 5 hours per arm buys on this GPU.
  If that is too small for the effect, the result is "budget too small", which
  T2.11 pre-registers as a distinct outcome from "the architecture fails".
