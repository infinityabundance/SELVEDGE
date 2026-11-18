# S54 GPU queue — ordered, one job at a time

The corrected-optimiser 20M bar (`results/stage2/runs/s50-bar-20m-lr1e-5`,
lr 1e-5) **finished** at step 9,765 with
`receipt.json` (stamp `20261118T001025Z`, revision `2e79ae3`) and verified by
`results/stage2/s48-bar-verify-20261118T020529Z.json`. Its milestone evals are
complete in `results/stage2/eval/s50-bar-20m-lr1e-5/`; the fixed-batch receipt
is `results/stage2/s50-bar-20m-lr1e-5-fixed-batch-20261118T021900Z.json` and the
curve is `results/stage2/s50-bar-curve-20261118T075110Z.json`. The scores are in
`results/REPORT.md:report-s50-bar-curve`.

Stage 1 is measured for the greedy non-thinking protocol on both post-trained
checkpoints; the table and gaps are in `results/REPORT.md:report-s54-stage1`.
The thinking mode truncated at the registered cap and is recorded as truncation
evidence, not a score.

**Remaining order (S54):**

1. optimizer validations (real-gradient first moment, then the model-level
   comparison);
2. stage 2 — on-policy distillation with failed-rollout corrective supervision;
3. deferred: the 3e-5/5e-5 fixed-batch KL gate, the conditional 3e-5 bar, and the
   P19.9 2M screen.

Every command is serialised. Each has an explicit, workload-sized `SELVEDGE_MEM`;
the host must have `limit + 4 GiB` of `MemAvailable` before a launch. An OOM kill
is a recorded failure and the limit arithmetic is fixed before a rerun, never
rerun with the same undersized limit.

Check the gate:

```bash
test -f results/stage2/runs/s50-bar-20m-lr1e-5/receipt.json
pgrep -af 'oracle/frontier/train.py|s43_score_total_cap.py'   # must print nothing
```

## 1. Score the bar's 5/10/20M milestones and decide the 100M extension

Milestones: `ckpt-002441` (4,999,168 tokens), `ckpt-004882` (9,998,336),
`ckpt-007323` (14,997,504) and `ckpt-009765` (19,998,720). The 6 GiB eval limit
is the registered one (1.83x the 3.269 GiB cgroup peak measured for this harness
and model). All three milestones are complete; the command is kept for replay.

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  scripts/s50_bar_milestones.sh

# the S53 rule's input: every saved checkpoint on the same fixed batches.
# The 16 GiB measurement run recorded container memory.peak = 14.03 GiB, so
# future fixed-batch runs use 21 GiB = max(1.5x peak, peak + 1 GiB).
SELVEDGE_MEM=16g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s50_fixed_batch_kl.py \
    --stream-manifest results/stage2/stage2-math-stream-20261117T071120Z.json \
    --start-token 2000000 \
    --run-dir results/stage2/runs/s50-bar-20m-lr1e-5 \
    --out results/stage2/s50-bar-20m-lr1e-5-fixed-batch-<stamp>.json

SELVEDGE_MEM=2g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court python -- python3 scripts/s50_bar_curve.py \
    --fixed-batch-receipt results/stage2/s50-bar-20m-lr1e-5-fixed-batch-<stamp>.json
```

`s50_bar_curve.py` refuses to decide without the fixed-batch receipt. It writes
the curve receipt, the CE/KL trend over every saved checkpoint, the registered
"still paying off" thresholds, the harness endpoint confirmation, the 100M
decision, and the `report-s50-bar-curve` region in `results/REPORT.md`. The old
three-point OLS slope is retained and labelled superseded.

## 2. Stage 1 — measure the post-trained frontier

Both post-trained checkpoints are already in the persistent cache and verified
against `models/FRONTIER-POST-TRAINED.lock.json`: 0.8B
`2fc06364715b967f1860aea9cf38778875588b17` (weight
`04b1c301…`), 2B `15852e8c16360a2fea060d615a32b45270f8a8fc` (weight
`aa33250c…`), both Apache-2.0. The fetch receipts are
`results/frontier-models/20261116-fetch-qwen35-08b.json` and
`results/frontier/s53-fetch-2b-chat.json`. No re-fetch is needed.

**Engine.** vLLM 0.34.0 is the registered future engine but is **not installed**
in the offline eval image. The pinned transformers stack (5.23.0 / torch 2.15.0)
loads the checkpoints; every stage-1 number is a transformers number and the
receipt says so. A vLLM number is not claimed.

Smoke each model first and size the real limit from its own cgroup peak:

```bash
SELVEDGE_MEM=16g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu eval -- python3 scripts/s54_post_trained_eval.py \
    --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --model-id Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --tasks gsm8k,math500 --modes non-thinking --decodings greedy \
    --limit 3 --out results/stage1/s54-smoke-0.8b.json
```

Then greedy, non-thinking, both tasks (the next number). The 0.8B smoke
recorded `container_memory.memory_peak` = 1.72 GiB, so the full run used 6 GiB
(3.5x the smoke peak).

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu eval -- python3 scripts/s54_post_trained_eval.py \
    --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --model-id Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --tasks gsm8k,math500 --modes non-thinking --decodings greedy \
    --batch-size 32 --vram-limit-gib 13.0 \
    --out results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json
```

Repeat for the 2B. Its smoke recorded `container_memory.memory_peak` = 6.12 GiB,
so the full 2B run used 10 GiB (1.63x the smoke peak). Thinking mode was run on a
50-problem subset and truncated every generation at the registered cap; it is
recorded as truncation evidence, not a score. The maj@16 subsets use the same
protocol and are paired at the problem level.

Render the table after the runs:

```bash
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=512m \
  scripts/court python -- python3 scripts/s54_stage1_report.py --write
```

## 3. Optimiser validations

### 3a. Real-gradient first-moment validation

Capture (GPU; one file, two probes, model not updated between probes):

```bash
SELVEDGE_MEM=20g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s50_capture_real_gradients.py \
    --out-dir results/artifacts/stage2/s50-real-gradients \
    --steps 2 --max-elements-per-tensor 500000
```

The 20g limit is the bar's: the same student and teacher are resident, and the
bar's cgroup peak was 12.281 GiB (floor max(1.5x, +1 GiB) = 18.4 GiB).

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
bfloat16 arm passes, the switch is mandatory for the next launch. Do not change
a running job mid-run.

### 3b. Model-level optimiser comparison, custom int8 state vs fp32 AdamW

No smoke of this workload has run. Run a short smoke with a generous explicit
`SELVEDGE_MEM`, read `container_memory.memory_peak` from its receipt, and size
the 200-step run at max(1.5x peak, peak + 1 GiB) before any number is quoted.

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

This asks the model-level question a per-tensor parity test cannot answer.

## 4. Stage 2 — on-policy distillation

Registration: `docs/ws19-post-trained-route-registration-v1.json`. The S54
amendment `docs/ws19-post-trained-route-stage2-amendment-s54-v1.json` supersedes
the supervision and win clauses **before any stage-2 run**:

- correct rollouts keep per-token reverse KL;
- failed rollouts are included when the post-trained 2B produces a continuation
  that `frontier.rft.verify` confirms; supervision starts at the first *textual
  divergence* (renamed by the S56 correction: a token mismatch is not a
  reasoning error), not at the whole failed continuation;
- failed rollouts without a verified teacher continuation contribute zero and
  are counted;
- the win now also requires MMLU and MMLU-Pro to stay inside the registered
  general-capability bound.

Stage 2 may start only after items 1 and 3 settle the bar and the first-moment
storage. A 20-step smoke sizes the container from its own cgroup peak; the
registered recipe is lr 1e-5, 20M tokens, seed 0, corrected Adam8bit.

```bash
SELVEDGE_MEM=<1.5x the stage-2 smoke peak, never below +1 GiB> \
SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
    --arm full --sft-examples <the refresh-1 examples> \
    --tokens 20000000 --seq-len 512 --micro-batch 1 --grad-accum 4 \
    --lr 1e-5 --temperature 2.0 --kl-weight 0.5 \
    --optimizer adam8bit --optim-block 256 --optim-numerics stochastic \
    --schedule cosine --min-lr-ratio 0.1 --warmup 200 --grad-clip 1.0 --seed 0 \
    --run-dir results/stage2/runs/s54-stage2-post-trained \
    --vram-limit-gib 12.5 --abort-update-factor 10.0
```

The exact refresh, corrective-supervision and accounting flags are the
implementation step; they must be receipted before launch. The queue entry is
ordered, not a substitute for the registered implementation.

## 5. Deferred

### 5a. 200-step fixed-batch KL gate at lr 3e-5 and 5e-5

Deferred. The corrected 1e-5 bar is the eligible operating point; a rate search
is not the next number. If it is run, the gate is the fixed-batch teacher KL
falling from `init` to `ckpt-000200`, not the per-step log KL.

### 5b. Conditional 3e-5 20M bar

Deferred and conditional on 5a passing. A failing 3e-5 gate is a result: it says
1e-5 is the correct operating point, not that a higher rate is forbidden.

### 5c. P19.9 2M router screen

Deferred. S54 repaired the pre-run defects: the accuracy control (`gated-dense`)
and the byte control (`memory`, dense always-read) are separate; the gather
ledger reports requested/unique/repeated addresses per call and per generation
with an explicit cache model; physical bytes are request-based and the unique
count is labelled `modelled`; the holdout is 24 distinct problems and the
statistics are a prompt-level paired bootstrap. See
`docs/S52-REPAIR-S54.md` and
`docs/s52-p19.9-router-screen-registration-v2.json`. The screen is a
mechanism validation, not a capability claim, and is not run.

## Numbers in this page and their receipts

Operational limits and byte projections, not new measurements:

- **12.281 GiB** is `container_memory.memory_peak` = 13,187,850,240 bytes in
  `results/stage2/runs/s50-bar-20m-lr1e-5/receipt.json` (stamp
  `20261118T001025Z`, revision `2e79ae3`).
- **6.476 GiB** is `peak_allocated_gib` in
  `results/stage2/s50-fixed-batch-kl-math-lr1e-5-20261117T205600Z.json` for the
  fixed-batch workload.
- **3.269 GiB** is the cgroup peak of the anchored eval harness in the S48 bar
  score; the 6 GiB milestone limit is 1.83x it.
- **4.000 B/element** and the bf16-first-moment projection are the byte
  arithmetic in `oracle/frontier/optim8bit.py` over the measured
  498,113,344 trainable elements.
- The extension thresholds **0.0**, **0.10** and **0.30** are the registered
  S53/S54 decision constants, not measurements.
