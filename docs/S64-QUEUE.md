# S64 GPU queue — ordered, one job at a time

Nothing in this file has run except the steps marked *done*. Every GPU entry
waits for the previous job to exit, for the GPU to be idle, and for host
`MemAvailable` to exceed the new limit plus 4 GiB. Each command carries an
explicit `SELVEDGE_MEM`; none is launched without one.

Registrations: `docs/ws19-stage2-correction-s63-preregistration-v1.json`
(corrected stage 2 and the queue order) and
`docs/S60-GPU-QUEUE.md` (the interpolation clause and its MMLU retention
input). `docs/S64-MILESTONE-PREP.md` records the scoring discipline.

## Gate

```sh
pgrep -af 's59_student_rollouts.py|oracle/frontier/train.py|s54_post_trained_eval.py'
# must print nothing before any entry below
awk '/MemAvailable/ {printf "MemAvailable %.1f GiB\n", $2/1048576}' /proc/meminfo
```

The K=4 pilot (`stage2b-gsm8k`, old stopping, no ledger) is a pilot only. Its
saved-token and throughput numbers are textual replay, not compute saved, and
it cannot supply the corrected p or the K=8 weights. Its receipt is
`results/artifacts/stage2/s59-rft/pilot-receipt.json`.

## 1. Weight-interpolation diagnostic (GPU, calibration only)

The registered command is in `docs/S60-GPU-QUEUE.md` §3. It interpolates the
post-trained 0.8B with `results/stage2/runs/s58-stage2a/ckpt-004000.pt` over
the fixed grid and scores **GSM8K-cal and MATH-cal drawn from the train split**.
The test splits are never read. The MMLU calibration pass follows, one
coefficient per job, on the total-cap harness; the selection is provisional
until those receipts exist, then re-applied without `--evaluate`.

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

# one MMLU calibration pass per coefficient, serially, then re-apply the bound
for t in 0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000; do
  SELVEDGE_MEM=12g scripts/court --gpu eval -- python3 scripts/s43_score_total_cap.py \
    --arm "interp-$t" \
    --full-checkpoint "results/stage2/s60-interpolation/interp-$t-arm.pt" \
    --tasks mmlu --total-cap-map mmlu=2000 \
    --registration docs/ws19-anchored-subset-amendment-preregistration-v2.json \
    --out "results/stage2/s60-interpolation/mmlu-$t.json"
done
```

MMLU is a retention input, not an accuracy objective. A coefficient whose
calibration MMLU drop exceeds the registered 1.0-point bound is ineligible;
ties choose the smaller departure from the post-trained model.

## 2. Corrected K=8 rollouts (GPU, four shards serially)

The launch record is
`results/artifacts/stage2/s64-rollouts/launch-record.json`. It is produced by
`scripts/s64_shard_window.py` from the pilot receipt's realised tokens/sample
and aggregate token rate; it recomputes the registered 4.5 h per-shard bound
and refuses a shard that does not fit. All four shards fit in the record that
precedes this queue.

The commands are in the record's `commands` array. The shape is:

```sh
SELVEDGE_MEM=4g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s59_student_rollouts.py \
    --datasets gsm8k-train --select-gsm8k 1200 --k-gsm8k 8 \
    --shards 2 --shard-index <0|1> --batch-prompts 8 \
    --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
    --rejected-out results/artifacts/stage2/s64-rollouts/<label>-rejected.jsonl \
    --out-dir results/artifacts/stage2/s64-rollouts --label <label> \
    --vram-limit-gib 12.0
```

`math-train` shards use `--select-math 600 --k-math 8`. Each shard writes a
per-problem ledger, a failed-continuation sidecar and a receipt. The receipt
carries `compute_saved: false` and the measured textual-replay token count
beside the loop-cut and truncation rates.

Memory basis: the running K=4 job's cgroup `memory.peak` and the pilot
receipt's `cgroup_memory.memory_peak` are both far below the 4g limit, which is
1.5x the larger measured peak plus 1 GiB. The corrected shard keeps the same
64 concurrent sequences (`--batch-prompts 8` at K=8, the pilot's
`--batch-prompts 16` at K=4).

## 3. Teacher and frozen-reference passes (GPU, then CPU/GPU)

```sh
SELVEDGE_MEM=14g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_data.py teacher \
    --ledger results/artifacts/stage2/s64-rollouts/gsm8k-shard0-ledger.jsonl \
    --ledger results/artifacts/stage2/s64-rollouts/gsm8k-shard1-ledger.jsonl \
    --ledger results/artifacts/stage2/s64-rollouts/math-shard0-ledger.jsonl \
    --ledger results/artifacts/stage2/s64-rollouts/math-shard1-ledger.jsonl \
    --rejected results/artifacts/stage2/s64-rollouts/gsm8k-shard0-rejected.jsonl \
    --rejected results/artifacts/stage2/s64-rollouts/gsm8k-shard1-rejected.jsonl \
    --rejected results/artifacts/stage2/s64-rollouts/math-shard0-rejected.jsonl \
    --rejected results/artifacts/stage2/s64-rollouts/math-shard1-rejected.jsonl \
    --teacher-corpus results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
    --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
    --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B/15852e8c... \
    --generate-missing --vram-limit-gib 13.0 \
    --exclude-ids results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
    --out-dir results/artifacts/stage2/s64-preference --label pairs
```

The teacher side is a **verified-correct teacher continuation from the
student's own prefix**; the corpus row is used only when its `prompt_ids` are
token-identical, and a generated continuation is conditioned on those exact
ids. The rejected side is the student's failed continuation. No teacher
weights are co-resident with the reference.

```sh
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_data.py reference \
    --pairs results/artifacts/stage2/s64-preference/pairs-pairs.jsonl \
    --reference-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
    --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
    --out-dir results/artifacts/stage2/s64-preference --label pairs-with-reference

# and again for the p=0 recovery pairs
... --pairs results/artifacts/stage2/s64-preference/pairs-recovery.jsonl \
    --label recovery-with-reference
```

The cache is the per-token `log pi_0(a|h)` on the fixed actions. The registered
token-length normalisation is the per-token mean; `sum` is a diagnostic only.

## 4. The three training arms (GPU, serial)

One arm per invocation, one GPU job at a time. `SELVEDGE_MEM` is **sized from
the arm's own measured peak**: run a short bounded smoke first, read the
receipt's `peak_allocated_gib` / `peak_reserved_gib` and cgroup
`memory_peak`, then set the real run's limit to
`max(1.5 x peak, peak + 1 GiB)` and record the basis. The 12g below is the
bounded smoke limit, not a measurement.

```sh
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_launch.py \
    --arm preference \
    --pairs results/artifacts/stage2/s64-preference/pairs-with-reference.jsonl \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364... \
    --budget-tokens 5000000 --cadence 500000,1000000,2000000,5000000 \
    --out-dir results/stage2/runs/s64-preference --vram-limit-gib 13.0

# matched SFT control: the same pairs, chosen continuations only, same budget
... --arm matched-sft --pairs <the same pairs> --budget-tokens 5000000 ...

# teacher-guided recovery: the p=0 pairs with the matched mean positive weight
... --arm recovery \
    --recovery-pairs results/artifacts/stage2/s64-preference/recovery-with-reference.jsonl ...
```

The receipt records the geometry (`rank=64`, `alpha=32`, `scaling=0.5`, every
text-tower linear except the output head), the merged checkpoints, the merge
report, and the active-union arithmetic. After the merge the adapter is not a
deployed identity, so the active union is 752,393,024 elements.

## 5. Milestone scoring (GPU eval serially, CPU pairing)

The launcher prints the milestone plan command with the actual step numbers it
wrote. `docs/S64-MILESTONE-PREP.md` is the discipline and the command shape.
The plan's stage-1, retention and pairing commands run serially; no test split
is a selection input.

## 6. What is deferred

* CURE intervention screen: later, and only after its own registration defines
  the protocol, cost and kill condition.
* vLLM: see `docs/S60-VLLM-AND-DISK.md`; the image cannot be built on the 26 GiB
  `/run` tmpfs.
