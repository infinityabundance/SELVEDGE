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
(9,998,336) and `ckpt-009765` (19,998,720). The 6 GiB eval limit is the
registered one and carries headroom over the cgroup peak measured for this
harness and model.

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/s50_bar_milestones.sh

SELVEDGE_MEM=2g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court python -- python3 scripts/s50_bar_curve.py
```

`scripts/s50_bar_curve.py` writes the curve receipt, the OLS math-family slope
and the registered 100M-extension decision. Nothing is typed by hand.

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

## (e) P19.9 counterfactual pre-read router, 2M screen — BLOCKED

The router module and its CPU contract are committed
(`oracle/frontier/utility_routing.py`: `select` returns a boolean mask, `execute`
fetches only selected rows, the two-stage gate cannot resurrect a rejected
token). The GPU training runner is **not implemented yet**, so no executable
command is written here. The queue entry is a gate:

1. the runner must consume `counterfactual-outcome` or
   `counterfactual-token-loss` supervision from `labels_from_records`, with the
   prompt-level fit/holdout split enforced;
2. it must call `execute` with a read path that records unique rows, tier bytes
   and cache hits, and it must report `physical_read_accounting` in its receipt;
3. its memory must be sized from its own smoke peak, because a bf16 first moment
   leaves only 0.16 GiB of the 12.5 GiB VRAM limit and the router adds parameters;
4. only then may a 2M screen run, and it must run after (d) so the first-moment
   storage is settled.

Writing a fake invocation for a script that does not exist would be fabricated
evidence. The command will be added when the runner and its CPU tests pass.

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
