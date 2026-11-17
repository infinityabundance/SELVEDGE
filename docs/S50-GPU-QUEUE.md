# S50/S51 GPU queue — ordered, not run

The corrected-optimiser 20M bar (`results/stage2/runs/s50-bar-20m-lr1e-5`,
lr 1e-5) is training. **No command on this page may start before its
`receipt.json` exists** and the run has exited. Every command below is
serialised: one GPU job at a time, each with an explicit, workload-sized
`SELVEDGE_MEM` derived from a measured peak for that workload. The host must have
`limit + 4 GiB` available before a launch; the launchers check this themselves.

Check the gate:

```bash
test -f results/stage2/runs/s50-bar-20m-lr1e-5/receipt.json
pgrep -af 'oracle/frontier/train.py'   # must print nothing
```

## (a) Score the 1e-5 bar's 5/10/20M milestones

The milestones are `ckpt-002441` (4,999,168 tokens), `ckpt-004882`
(9,998,336), `ckpt-007323` (14,997,504) and `ckpt-009765` (19,998,720). The
6 GiB eval limit is the registered one and carries headroom over the cgroup peak
measured for this harness and model. The S53 decision rule also needs the same
fixed held-out batches measured for **every saved checkpoint**, not only the
milestones; that is the fixed-batch receipt the curve script consumes.

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/s50_bar_milestones.sh

# the S53 rule's input: every saved checkpoint on the same fixed batches
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s50_fixed_batch_kl.py \
    --stream-manifest results/stage2/stage2-math-stream-20261117T071120Z.json \
    --start-token 2000000 \
    --run-dir results/stage2/runs/s50-bar-20m-lr1e-5 \
    --out results/stage2/s50-bar-20m-lr1e-5-fixed-batch-<stamp>.json

SELVEDGE_MEM=2g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court python -- python3 scripts/s50_bar_curve.py \
    --fixed-batch-receipt results/stage2/s50-bar-20m-lr1e-5-fixed-batch-<stamp>.json
```

`scripts/s50_bar_curve.py` refuses to decide without the fixed-batch receipt. It
writes the curve receipt, the fixed-batch CE/KL trend over every saved
checkpoint, the registered "still paying off" thresholds, the harness endpoint
confirmation and the resulting 100M-extension decision. Nothing is typed by
hand. The old harness-points OLS slope is still reported, labelled superseded.

## (b) 200-step fixed-batch KL gate at lr 3e-5 and 5e-5

Run both rates with the corrected optimiser, then measure the same fixed batches
for the init and every checkpoint. The 12 GiB limit is the one used by the
existing 200-step revalidation receipts; the 3e-5/5e-5 runs are the same
workload (student + teacher + 200 steps).

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
    --arm full \
    --stream-manifest results/stage2/stage2-math-stream-20261117T071120Z.json \
    --tokens 409600 --seq-len 512 --micro-batch 1 --grad-accum 4 \
    --lr 3e-5 --temperature 2.0 --kl-weight 0.5 \
    --optimizer adam8bit --optim-block 256 --optim-numerics stochastic \
    --schedule constant --warmup 20 --grad-clip 1.0 --seed 0 \
    --ckpt-every 50 --keep-checkpoints 8 --log-every 50 \
    --run-dir results/stage2/runs/s50-revalidate-200-lr3e-5 \
    --vram-limit-gib 12.5 --abort-update-factor 10.0

SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s50_fixed_batch_kl.py \
    --stream-manifest results/stage2/stage2-math-stream-20261117T071120Z.json \
    --start-token 2000000 \
    --run-dir results/stage2/runs/s50-revalidate-200-lr3e-5 \
    --out results/stage2/s50-fixed-batch-kl-math-lr3e-5-<stamp>.json
```

Repeat the pair with `--lr 5e-5` and `lr5e-5` in the run directory and output
name. The gate is the **fixed-batch** teacher KL falling from `init` to
`ckpt-000200`; the per-step log KL is recorded but is not the gate, because every
step sees a different batch.

## (c) If, and only if, 3e-5 passes the KL gate: launch the 3e-5 20M bar

The 1e-5 rate was selected by the KL gate, not by a search for the largest stable
rate. 3e-5 is 3x that rate. The S49 3e-5 run is not evidence against it: that
run was produced by the defective second moment and is invalidated as a whole.
The 3x test therefore asks whether the lower rate bought stability at the cost of
convergence, with the same corrected optimiser and the same tripwires
(update factor 10.0, CE factor 1.5).

```bash
SELVEDGE_MEM=20g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/s50_bar_20m.sh --lr 3e-5 \
    --run-dir results/stage2/runs/s50-bar-20m-lr3e-5 \
    --kl-receipt results/stage2/s50-fixed-batch-kl-math-lr3e-5-<stamp>.json \
    --detach
```

The launcher refuses a non-default rate without a KL-gate receipt and records
its SHA-256 in `launch.json`. When the run exits:

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/s50_bar_milestones.sh \
    --run-dir results/stage2/runs/s50-bar-20m-lr3e-5 \
    --eval-root results/stage2/eval/s50-bar-20m-lr3e-5

SELVEDGE_MEM=2g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court python -- python3 scripts/s50_bar_curve.py \
    --run-dir results/stage2/runs/s50-bar-20m-lr3e-5 \
    --eval-root results/stage2/eval/s50-bar-20m-lr3e-5 \
    --out results/stage2/s50-bar-curve-lr3e-5-<stamp>.json
```

If 3e-5 fails the KL gate, **do not launch a 5e-5 bar**; the 5e-5 KL receipt is
the record and the 1e-5 bar remains the eligible one. A failing 3e-5 20M bar is
also a result: it says the 1e-5 rate is the correct operating point for this
optimiser, not that a higher rate is forbidden.

## (d) Real-gradient first-moment validation

Capture (GPU; one file, two probes, model not updated between probes):

```bash
SELVEDGE_MEM=20g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s50_capture_real_gradients.py \
    --out-dir results/artifacts/stage2/s50-real-gradients \
    --steps 2 --max-elements-per-tensor 500000
```

The 20g limit is the bar's: the same student and teacher are resident, and the
previous bar's cgroup peak was 8.76 GB (floor max(1.5x, +1 GiB) = 13.1 GB). The
fixture is a `.pt` file under `results/artifacts/`, which is gitignored; the
tracked evidence is `capture.json` with the fixture SHA-256.

CPU harness (no GPU; runs on the captured tensors):

```bash
SELVEDGE_MEM=8g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court oracle -- python3 scripts/s50_real_gradient_validation.py \
    --fixture results/artifacts/stage2/s50-real-gradients/fixture.pt \
    --out results/stage2/s50-real-gradient-validation-<stamp>.json
```

Decision rule (in the receipt): the int8 first moment stays if the worst
relative update L2 is `< 0.40`, the max `|update|` ratio against AdamW is
`< 3.0`, and the final parameter error is `< 0.02`. If int8 fails and the
`first_moment_storage="bfloat16"` arm passes, the switch is mandatory for the
next launch (`4.000 B/element`, projected 12.34 GiB against the 12.5 GiB limit). If
both fail, the first moment is not the cause and a storage switch must not be
used to excuse the gradient or step-size regime. Do not change the running bar
mid-run.

## (e) P19.9 counterfactual pre-read router, 2M screen

The router interface is `oracle/frontier/utility_routing.py` (`select` returns a
boolean mask, `execute` fetches only selected rows, the two-stage gate cannot
resurrect a rejected token). Its runner is now implemented:

- `oracle/frontier/routing_runner.py` — paired rollouts, the counterfactual
  labels, pre-read training, mask-enforced selection audit and read accounting;
- `scripts/s52_p19_9_router.py` — the executable stages;
- `oracle/tests/test_s52_p19_9_runner.py` — the CPU contract;
- `docs/s52-p19.9-router-screen-registration-v1.json` — the registered arms,
  win bar and next route;
- `docs/s52-p19.9-calibration-prompts-v1.json` — the registered 64-prompt
  GSM8K-train calibration set (48 fit / 8 calib / 8 holdout), disjoint from the
  harness tasks.

The stage gates, all enforced by the driver rather than remembered:

1. supervision is `counterfactual-outcome` (primary) or
   `counterfactual-token-loss` (secondary) through `labels_from_records`, with
   the prompt-level fit/calib/holdout split enforced and the same prompt text
   forbidden in two splits;
2. the routed read calls `execute` with a read path that records the unique
   table rows it gathers; the receipt reports that ledger beside
   `physical_read_accounting` (unique rows, tier bytes, cache hits), and the
   read-magnitude surrogate `measured_cost(read)` is never used;
3. the GPU stages refuse to start before the bar receipt and the real-gradient
   validation receipt exist, and a `switch-first-moment-to-bfloat16` decision
   forces `--first-moment bfloat16`;
4. the container limit must come from this workload's own smoke peak. No smoke
   has run yet, so the number below is deliberately absent: run `rollouts` once
   with a generous limit, read `container_memory.memory_peak` from its receipt,
   and size the real run from that measured peak (factor 1.5, never below
   +1 GiB) before any screen number is quoted.

```bash
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=512m \
  scripts/s52_p19_9_screen.sh frame

# smoke: size the limit from the receipt's container_memory.memory_peak later
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/s52_p19_9_screen.sh rollouts

SELVEDGE_MEM=4g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/s52_p19_9_screen.sh train

SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/s52_p19_9_screen.sh eval

SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=512m \
  scripts/s52_p19_9_screen.sh screen
```

The screen is registered (arms `utility-outcome`, `utility-ce`,
`no-memory-gate`, `matched-dense`, with `gated-dense` as the attribution
control) and **not run**. The win bar and the next route are in the
registration; no 20M router arm may be registered without a win here.

## (f) Model-level optimiser comparison, custom int8 state vs fp32 AdamW

The CPU harness is `oracle/frontier/optim_compare.py`; its contract is
`oracle/tests/test_s53_optimizer_comparison.py` and it passes on fake states.
The GPU driver is `scripts/s53_optimizer_comparison.py`. It captures one exact
initial trainable state, restores it for each arm, uses the same stream offsets
and the same fixed evaluation batches for both arms, and reports fixed-batch
teacher KL plus held-out CE at init and every 25 steps for 200 steps. The arms
run sequentially so only one student is resident.

No smoke of this workload has run, so **no container limit is written here**.
Run a short smoke first with a generous explicit `SELVEDGE_MEM`, read
`container_memory.memory_peak` from its receipt, and size the 200-step run at
max(1.5x peak, peak + 1 GiB) before any number is quoted. The queued command is:

```bash
SELVEDGE_MEM=<from the smoke receipt> SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 \
SELVEDGE_TMP=2g scripts/court --gpu oracle -- \
  python3 scripts/s53_optimizer_comparison.py \
    --stream-manifest results/stage2/stage2-math-stream-20261117T071120Z.json \
    --start-token 2000000 --steps 200 --eval-every 25 --batches 8 \
    --seq-len 512 --grad-accum 4 --lr 1e-5 --temperature 2.0 --kl-weight 0.5 \
    --grad-clip 1.0 --seed 0 \
    --out results/stage2/s53-optimiser-comparison-<stamp>.json
```

This is queued, not run. It does not replace the real-gradient first-moment
validation in (d); it asks the model-level question that a per-tensor parity
test cannot answer.

## Numbers in this page and their receipts

These are operational limits and byte projections, not new measurements:

- **8.76 GB** is `container_memory.memory_peak` = 8,762,351,616 bytes in
  `results/stage2/runs/full-20m-math/receipt.json`. That run is invalidated as a
  score by S50, but the model and teacher it held are the same, so its resource
  peak is the sizing precedent it was already used for at the S49 launch.
- **11.885 GiB** is `peak_allocated_gib` in
  `results/stage2/s50-revalidation-20261117T210326Z.json` (stamp
  `20261117T210326Z`, revision `547953f`), the corrected 200-step run.
- **4.000 B/element** and the **12.34 GiB** bf16-first-moment projection are the
  byte arithmetic in `oracle/frontier/optim8bit.py` over the measured
  498,113,344 trainable elements: int8 is 3.015625 B/element, bf16 is 4.000
  B/element, and the difference is +0.457 GiB over the measured 11.885 GiB peak.
- The decision thresholds **0.40**, **3.0** and **0.02** are stated in
  `scripts/s50_real_gradient_validation.py` with their basis; they are not
  measurements.
