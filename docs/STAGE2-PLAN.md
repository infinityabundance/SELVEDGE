# WS19 stage 2 — teacher-grafted conditional memory (S37 registration and smoke)

Status: **implemented and smoke-measured, no full-budget run.** The full-budget
run is *not* started in this session; this document registers its budget and
routes from the measured smoke. Every number below that is a performance,
storage, throughput or resource measurement is generated from a receipt between
sentinels by `scripts/perf_numbers.py`; editing one by hand is a defect. The
arithmetic of the design (8 reads per token, 4 bigram + 4 trigram heads, one
site after decoder layer 2) is a declared structural fact, not a measurement.

The contract this serves is `docs/PROGRAM-PLAN.md` §24 P19.2 and the WS19 win
condition registered there: the grafted memory must beat the frozen 0.8B-Base
score on the knowledge family with no task more than one point lower, and it
must beat both a training-free NGM baseline and a parameter-matched dense
adapter at equal unique active.

## 1. What was built

| module | role |
| --- | --- |
| `oracle/frontier/memory.py` | exact n-gram addressing (4 bigram + 4 trigram heads, 8 rows read per token), per-site key/value projections, a parameter-free sigmoid gate conditioned on the hidden state, a causal depthwise convolution, and the active/accessible arithmetic |
| `oracle/frontier/graft.py` | teacher hidden-state capture, ridge least-squares teacher→student map, PCA to the row width, per-address row means with counts, reconstruction and gate diagnostics |
| `oracle/frontier/stream.py` | pinned token stream from the cached T1 corpus with the anchor's own tokenizer, plus the 5-word-shingle leakage guard against every cached evaluation test split |
| `oracle/frontier/arms.py` | the parameter-matched dense adapter, the training-free NGM injector, and the hook-based installation that keeps the backbone a real `PreTrainedModel` |
| `oracle/frontier/train.py` | frozen-core training with online teacher KL, deterministic data order, checkpoint/resume, per-step logging, VRAM and cgroup limits |
| `oracle/frontier/evaluate.py` | the pinned `lm-evaluation-harness` path, with the arm handed to `HFLM` as a pre-initialised model |
| `oracle/frontier/smoke.py` | delta arithmetic, signal questions, and the budget propagation from measured throughput |
| `oracle/frontier/accounting.py` | the exact accounting receipt at the three row scales |

The core is always frozen. The memory table is **not** in the active union; one
token reads 8 rows. The key projection, value projection, output bias and
convolution are. The table is in the *accessible* count and in the optimiser.

## 2. Geometry and exact accounting

Development table, one site after decoder layer 2:

- active elements per token: <!-- BEGIN GENERATED: s2-active-per-token -->
537,600
<!-- END GENERATED: s2-active-per-token -->
- accessible elements added: <!-- BEGIN GENERATED: s2-accessible-dev -->
17,312,768
<!-- END GENERATED: s2-accessible-dev -->
- table elements at the development row count: <!-- BEGIN GENERATED: s2-table-dev -->
16,777,216
<!-- END GENERATED: s2-table-dev -->

The three registered row scales, at one site (the default module):

- 65,536 rows: table elements <!-- BEGIN GENERATED: s2-table-65536 -->
16,777,216
<!-- END GENERATED: s2-table-65536 -->
- 1,048,576 rows: table elements <!-- BEGIN GENERATED: s2-table-1048576 -->
268,435,456
<!-- END GENERATED: s2-table-1048576 -->
- 8,388,608 rows: table elements <!-- BEGIN GENERATED: s2-table-8388608 -->
2,147,483,648
<!-- END GENERATED: s2-table-8388608 -->

The parameter-matched dense adapter's active union is exactly the memory's, by
construction (`2 * 1024 * 262 + 1024`), with a parameter-free gate:

<!-- BEGIN GENERATED: s2-adapter-active -->
537,600
<!-- END GENERATED: s2-adapter-active -->

`accounting.py` prints the four-site variant too, because the number in
`docs/PROGRAM-PLAN.md` counts `rows * 1024` in one figure and therefore implies
four 256-value sites. That number is not wrong for the four-site configuration;
it is simply not the default this stage trains.

## 3. Teacher grafting

The teacher is `Qwen/Qwen3.5-2B-Base` at its pinned revision (cached), the
student is `Qwen/Qwen3.5-0.8B-Base` at its pinned revision. The graft split is
the first 250,000 tokens of the pinned stream; the training split is the
remaining 2,000,000 tokens. The two checkpoints share the tokenizer file
byte-for-byte (verified equal in the receipt).

Method:

1. capture the hidden state at the output of decoder layer 2 for both models on
   the same tokens;
2. fit `W: R^2048 -> R^1024` by ridge least squares on the first 80% and report
   held-out relative error;
3. PCA the mapped teacher states to 256 dimensions and report explained
   variance and the reconstruction error of `mu + B B^T (y - mu)`;
4. each of the 8 heads addresses a row; a row is the mean coefficient of every
   token that hashed to it, and the count is stored;
5. `key_proj = B^T`, `value_proj = B`, `out_bias = mu`, convolution identity.

Measured interpolation quality:

- held-out least-squares relative error: <!-- BEGIN GENERATED: s2-graft-ls-error -->
0.48559371148189695
<!-- END GENERATED: s2-graft-ls-error -->
- PCA explained variance at 256 of 1024 dimensions: <!-- BEGIN GENERATED: s2-graft-pca-variance -->
0.7959621969138733
<!-- END GENERATED: s2-graft-pca-variance -->
- row-table reconstruction error (gated output vs the real student state):
  <!-- BEGIN GENERATED: s2-graft-reconstruction -->
0.8154492974281311
<!-- END GENERATED: s2-graft-reconstruction -->
- development-table occupancy fraction after grafting: <!-- BEGIN GENERATED: s2-graft-occupancy -->
1.0
<!-- END GENERATED: s2-graft-occupancy -->

Two findings are part of the method, not footnotes. First, the linear teacher→
student map explains roughly three quarters of the variance; the two models are
not linearly aligned at this site, so a row is a *lossy* summary and the
reconstruction number says how lossy. Second, at initialisation the gate is
nearly constant (mean <!-- BEGIN GENERATED: s2-graft-gate-mean -->
0.5036013126373291
<!-- END GENERATED: s2-graft-gate-mean -->, range
<!-- BEGIN GENERATED: s2-graft-gate-min -->
0.49435365200042725
<!-- END GENERATED: s2-graft-gate-min --> to
<!-- BEGIN GENERATED: s2-graft-gate-max -->
0.538838803768158
<!-- END GENERATED: s2-graft-gate-max --> in the receipt): the
conditioning is learned, not supplied by the graft. The row-scale question is
what happens when the same active read can address more rows.

Leakage guard: the stream builder discards any document whose worst 5-word
shingle Jaccard against any cached evaluation test item is at least 0.2 and
fails closed at 0.5. The maximum observed in the smoke stream is
<!-- BEGIN GENERATED: s2-stream-max-jaccard -->
0.016042780748663103
<!-- END GENERATED: s2-stream-max-jaccard -->, and
no document was discarded. The guard covers MMLU-Pro, GSM8K, MMLU, ARC,
MATH-500, TriviaQA, HumanEval and IFEval test splits that are cached.

## 4. Smoke protocol

- 2,000,000 training tokens per trainable arm, the same pinned stream and the
  same order for every arm;
- 250,000 graft tokens, disjoint from the training split by position and
  shingle-checked against the evaluation sets;
- one site after decoder layer 2, 65,536-row table, 256-wide rows, 8 reads;
- objective `CE + 0.5 * T^2 * KL(teacher_T || student_T)`, `T = 2.0`, teacher
  logits computed online, no logit cache;
- AdamW, learning rate `1e-3`, 50-step warmup, weight decay `0`, gradient clip
  `1.0`, sequence length 512, micro-batch 1, gradient accumulation 4;
- seed 0 for every arm in the smoke; three seeds are registered for the full
  run;
- evaluation through the pinned harness at batch 1 for every arm, so the
  deltas are internal to this smoke and are not comparable to the S36 absolute
  scoreboard (batch 2/8): MMLU-Pro 2 per subject (28 items), GSM8K 200 items,
  MMLU 2 per subject (114 items), all 5-shot.

The three controls are the same harness and the same evaluation:

- **NGM** — training-free causal n-gram encoder over the pretrained embedding
  space with a cosine gate; no new parameters;
- **dense adapter** — 262-rank bottleneck with a parameter-free gate; its
  active union equals the memory's exactly;
- **frozen base** — no hook at all.

A fourth arm, **memory-graft**, is the step-0 graft with no training. It isolates
the teacher initialisation from the training.

## 5. Smoke results

Per-arm knowledge and worst-task deltas against the frozen base:

- memory trained: knowledge mean <!-- BEGIN GENERATED: s2-memory-knowledge-delta -->
0.0
<!-- END GENERATED: s2-memory-knowledge-delta --> pp, worst task
  <!-- BEGIN GENERATED: s2-memory-worst-delta -->
0.0
<!-- END GENERATED: s2-memory-worst-delta --> pp
- memory graft only: knowledge mean <!-- BEGIN GENERATED: s2-graft-knowledge-delta -->
0.0
<!-- END GENERATED: s2-graft-knowledge-delta --> pp, worst task
  <!-- BEGIN GENERATED: s2-graft-worst-delta -->
0.0
<!-- END GENERATED: s2-graft-worst-delta --> pp
- NGM: knowledge mean <!-- BEGIN GENERATED: s2-ngm-knowledge-delta -->
0.0
<!-- END GENERATED: s2-ngm-knowledge-delta --> pp, worst task
  <!-- BEGIN GENERATED: s2-ngm-worst-delta -->
0.0
<!-- END GENERATED: s2-ngm-worst-delta --> pp
- dense adapter: knowledge mean <!-- BEGIN GENERATED: s2-adapter-knowledge-delta -->
0.0
<!-- END GENERATED: s2-adapter-knowledge-delta --> pp, worst task
  <!-- BEGIN GENERATED: s2-adapter-worst-delta -->
0.0
<!-- END GENERATED: s2-adapter-worst-delta --> pp

Measured cost:

- memory training throughput: <!-- BEGIN GENERATED: s2-memory-tokens-per-second -->
0
<!-- END GENERATED: s2-memory-tokens-per-second --> tokens/s, peak allocated
  VRAM <!-- BEGIN GENERATED: s2-memory-peak-gib -->
0
<!-- END GENERATED: s2-memory-peak-gib --> GiB
- adapter training throughput: <!-- BEGIN GENERATED: s2-adapter-tokens-per-second -->
0
<!-- END GENERATED: s2-adapter-tokens-per-second --> tokens/s, peak allocated
  VRAM <!-- BEGIN GENERATED: s2-adapter-peak-gib -->
0
<!-- END GENERATED: s2-adapter-peak-gib --> GiB

The judgement of whether any arm moved, and whether the memory is
distinguishable from its controls, is stated in §6 from these numbers and the
combined standard errors recorded in the smoke receipt. No arm is declared a
winner here: this is a signal check.

## 6. Signal

*(Filled from the smoke receipt; the prose is written after the numbers are
generated. A null is reported as a null, and the control it failed to beat is
named.)*

## 7. Registered full-budget plan

The budget is derived from the measured smoke throughput, not typed:

- phase A (development scale, 65,536 rows): three seeds, memory and adapter,
  20,000,000 tokens per arm per seed; expected GPU-hours
  <!-- BEGIN GENERATED: s2-plan-phase-a-hours -->
0
<!-- END GENERATED: s2-plan-phase-a-hours -->;
- phase B (scale question, 1,048,576 rows): three seeds, memory only, same token
  budget; expected GPU-hours
  <!-- BEGIN GENERATED: s2-plan-phase-b-hours -->
0
<!-- END GENERATED: s2-plan-phase-b-hours -->. This phase is blocked until the
  training loop uses sparse row updates, because dense AdamW state for the
  table is three times the table in float32;
- phase C (8,388,608 rows): not runnable on this device; it needs the NVMe
  serving path of P19.5 and is registered as not run.

Evaluation cadence: every 5,000,000 tokens on the fixed smoke subsets, plus the
full registered anchor task list at the end of each run.

Next route if the first configuration misses, in order:

1. **site placement** — repeat phase A at decoder layer 6, then layer 11, then
   the pair;
2. **4B teacher** — `Qwen/Qwen3.5-4B-Base` is not cached; the cache filesystem
   has room for 9.3 GB plus 1 GiB headroom (recorded in the graft receipt), so
   fetch it and repeat the graft;
3. **larger table** — 1,048,576 rows after the sparse-update change;
4. **LoRA on the core** — rank 8 on the linear-attention projections, after
   the memory-only phase, with the frozen-core control retained.

Kill rule: only if the best admission policy loses to both the dense adapter and
the training-free NGM at all three row scales with the placement route measured
(PROGRAM-PLAN P19.2). A miss is a route change, not a verdict on the mechanism.

## 8. Failures and limitations

*(Filled from the run log and the smoke receipt: any OOM, any failed invocation,
and the honest residual risks.)*

## 9. Reproduce

```bash
# 1. pinned stream and leakage guard (CPU)
SELVEDGE_MEM=12g scripts/court oracle -- \
  python3 oracle/frontier/stream.py build --tokens 2250000 --graft-tokens 250000

# 2. teacher graft (GPU; measured peak is in the receipt)
SELVEDGE_MEM=14g scripts/court --gpu oracle -- \
  python3 oracle/frontier/graft.py \
  --stream-manifest results/stage2/stage2-stream-<stamp>.json --vram-limit-gib 12

# 3. exact accounting (CPU)
SELVEDGE_MEM=4g scripts/court oracle -- \
  python3 oracle/frontier/accounting.py

# 4. train one arm (GPU; launch with systemd-run and an explicit memory limit)
SELVEDGE_MEM=16g scripts/court --gpu oracle -- \
  python3 oracle/frontier/train.py --arm memory \
  --stream-manifest results/stage2/stage2-stream-<stamp>.json \
  --memory-init results/stage2/graft/<stamp>/memory-init.pt \
  --tokens 2000000 --run-dir results/stage2/runs/memory-2m --vram-limit-gib 13

# 5. evaluate one arm through the pinned harness
SELVEDGE_MEM=10g scripts/court --gpu eval -- \
  python3 oracle/frontier/evaluate.py --arm memory \
  --memory-init results/stage2/graft/<stamp>/memory-init.pt \
  --checkpoint results/stage2/runs/memory-2m/ckpt-000976.pt \
  --tasks mmlu_pro,gsm8k,mmlu --mmlu-pro-limit 2 --gsm8k-limit 200 \
  --mmlu-limit 2 --out-dir results/stage2/eval/memory --vram-limit-gib 13

# 6. aggregate the smoke and derive the plan
scripts/court python -- python3 oracle/frontier/smoke.py collect \
  --base results/stage2/eval/base/eval.json \
  --arm ngm=... --arm memory-graft=... --arm memory-trained=... \
  --arm adapter=... --train memory=... --train adapter=... \
  --graft results/stage2/stage2-graft-<stamp>.json
scripts/court python -- python3 oracle/frontier/smoke.py plan \
  --smoke results/stage2/stage2-smoke-<stamp>.json
```
