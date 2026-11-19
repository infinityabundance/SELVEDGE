# S60 GPU queue — ordered, one job at a time

Nothing in this file has run. Every entry waits for the running stage-2b
rollout job (`scripts/s59_student_rollouts.py`) to exit, for the GPU to be idle,
and for host `MemAvailable` to exceed the new limit plus 4 GiB. Each command has
an explicit `SELVEDGE_MEM`; none is launched without one.

Companion documents: `docs/S60-STAGE2-REVISED.md` (the registered recipe),
`docs/S60-LITERATURE.md`, `docs/S60-VLLM-AND-DISK.md`.

**S63 correction (2026-11-19).** The rollout job that is running was launched
with the old stopping criterion and K=4. Its saved-token and throughput numbers
are **textual replay only, not compute saved**: a batched `generate` advances
every row until the slowest row stops, so cutting one row's text does not reduce
the batch's forward cost. Its accepted file also has no per-problem ledger, so
its rows cannot supply the corrected p = successes/attempts or the K=8 p(1-p)
weights. The corrected K=8 design and the queue order (interpolation first,
corrected preference stage second, CURE later) are registered in
`docs/ws19-stage2-correction-s63-preregistration-v1.json`.

## 0. Gate: wait for the rollout pass

```sh
pgrep -af 's59_student_rollouts.py|oracle/frontier/train.py|s54_post_trained_eval.py'
# must print nothing before any entry below
awk '/MemAvailable/ {printf "MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo
```

## 1. Long-row smoke for the S60 SFT geometry (first, because it is unknown)

S60 changed the launcher's default row width from 1250 to 2500 to preserve the
11 MATH examples longer than 2048 tokens and all 191 longer than 1250
(`results/stage2/s60-long-example-share.json`, stamp `20261119T032353Z`,
revision `9c179e3`; 13.3% of teacher supervised tokens). A 2500-
token row with activation checkpointing and the gathered LM head has **not** been
measured on this card. The smoke exists to size the real run, not to prove the
loss.

```sh
SELVEDGE_MEM=18g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
    --arm full \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --student-lock models/FRONTIER-POST-TRAINED.lock.json \
    --student-id Qwen/Qwen3.5-0.8B \
    --student-revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --sft-examples results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl \
    --sft-examples-per-row 1 --sft-overflow drop \
    --objective sft --loss-unit token \
    --length-aware-batching --sft-logits-gather --gradient-checkpointing \
    --kl-weight 0 --tokens 100000 --seq-len 2500 \
    --micro-batch 1 --grad-accum 1 \
    --lr 1e-5 --optimizer adam8bit --optim-block 256 --optim-numerics stochastic \
    --schedule cosine --min-lr-ratio 0.1 --warmup 40 --grad-clip 1.0 --seed 0 \
    --ckpt-every 40 --keep-checkpoints 1 --log-every 20 \
    --run-dir results/stage2/runs/s60-long-row-smoke \
    --vram-limit-gib 13.0 --abort-update-factor 0 --abort-ce-factor 0
```

The run is a smoke, not a score. Read the run receipt's
`peak_allocated_gib`, `peak_reserved_gib` and `container_memory.memory_peak`;
the real stage-2a/2b run's `SELVEDGE_MEM` is `max(1.5 x peak, peak + 1 GiB)`,
recorded before launch. If the smoke OOMs, the row width is reduced to 2048
(11 examples / 21,688 supervised tokens dropped, measured) rather than the
memory limit being guessed upward.

## 2. Corpus re-verification with the examples hash (CPU, but required)

```sh
SELVEDGE_MEM=4g SELVEDGE_CPUS=4 SELVEDGE_PIDS=1024 SELVEDGE_TMP=1g \
  scripts/court oracle -- python3 scripts/s57_verify_stage2a.py \
    --receipt results/artifacts/stage2/s58-combined/stage2a-accepted-combined-receipt.json \
    --accepted results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
    --examples results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl \
    --examples-receipt results/artifacts/stage2/s58-combined/stage2a-combined-examples-receipt.json \
    --out results/stage2/s57-stage2a-verification.json
```

The result must be `corpus_usable=true` with
`reverification.all_correct=true` and
`examples.matches=true`; the launcher refuses otherwise.

## 3. Weight-interpolation diagnostic (GPU, calibration only)

Interpolates the post-trained 0.8B and the S58 best checkpoint
(`results/stage2/runs/s58-stage2a/ckpt-004000.pt`) over a fixed grid, writes one
checkpoint per coefficient and scores **calibration splits drawn from the train
split** — never MATH-500 test or GSM8K test. The coefficient is chosen on those
calibration numbers, not on a test score.

```sh
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu eval -- python3 scripts/s60_interpolate.py \
    --checkpoint results/stage2/runs/s58-stage2a/ckpt-004000.pt \
    --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --grid 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
    --exclude-ids results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
    --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
    --calibration-limit 300 --evaluate \
    --out-dir results/stage2/s60-interpolation \
    --out results/stage2/s60-interpolation/receipt.json
```

`SELVEDGE_MEM=12g` is the S58 eval floor (the 0.8B eval receipt's measured
cgroup peak plus headroom); the evaluator loads one 0.8B model, not two. The
receipt is `provisional=true` until the MMLU calibration receipts are supplied:

```sh
# one MMLU calibration pass per coefficient, on the total-cap harness
for t in 0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000; do
  SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 scripts/s43_score_total_cap.py \
    --arm "interp-$t" \
    --full-checkpoint "results/stage2/s60-interpolation/interp-$t-arm.pt" \
    --tasks mmlu --total-cap-map mmlu=2000 \
    --registration docs/ws19-anchored-subset-amendment-preregistration-v2.json \
    --out "results/stage2/s60-interpolation/mmlu-$t.json"
done
# then re-run s60_interpolate.py with --mmlu-receipts for each file and
# without --evaluate to apply the retention bound to the selection
```

The MMLU pass is a retention input, not an accuracy objective. A coefficient
whose calibration MMLU drop exceeds 1.0 point is ineligible; ties choose the
smaller departure from the post-trained model.

## 4. Revised stage-2 LR gate, then the screen

After the smoke sizes the memory and the corpus verification passes:

```sh
# the fixed-batch KL gate at 3e-5, 1e-4, 3e-4; starting rate 1e-4
# (the registration and the S50 gate receipt define the threshold)
# ... one script invocation per rate, one GPU job at a time ...
```

The gate is the S50 fixed-batch KL instrument. Accuracy is not the selection
input. If every rate rises, the run does not launch and the screen is reported.

## 5. External frontier check (S59-A)

`docs/ws19-frontier-check-s59a-preregistration-v1.json`. Order: ReasonLite-0.6B
(OpenRAIL-D, ungated), Qwen3-0.6B (Apache-2.0, cached), then
MobileLLM-R1.5-950M **only if the operator grants access**. Each model is one
`scripts/s54_post_trained_eval.py` pass under the identical protocol; no gated
weights are fetched without a recorded grant.

## 6. vLLM image (deferred, see `docs/S60-VLLM-AND-DISK.md`)

Not queued as a GPU job: the image cannot be built on the 26 GiB `/run` tmpfs.
After the rollout job exits, `docker builder prune -f` reclaims 6.8 GiB; a
durable build needs a second dockerd with a data root on `/mnt/2tb_crucial`.
