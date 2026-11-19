# Shutdown state — 2026-11-18T09:30Z

> **Superseded 2026-11-19T02:35Z by S58.** The corpus was finished, the stream
> built, the stage-2a arm trained and scored on both registered instruments, and
> the retention bound was measured. The arm **does not win**: GSM8K improves
> (+0.0288, 95% interval [+0.0008, +0.0576]) while MATH-500 loses 11.0 points
> (interval [−0.1540, −0.0660]) and MMLU loses 1.94 points (interval
> [−0.0276, −0.0113], outside the 1.0-point bound). The full report — corpus
> counts, stream accounting, training curve, the two aborted launches and their
> diagnoses, all milestone scores and the price of each registered next route —
> is `docs/S58-STAGE2A.md`. Nothing below is a claim about the current state; it
> is the 2026-11-18 snapshot kept for the record.

Written at operator shutdown. Every running job was stopped and its state recorded.
Nothing below is a capability claim; it is the resume map.

## Repository

- Branch `staging`, pushed. The last commit before this file is `8df7850`.
- Working tree contains only the partial stage-2a data copy added by this file's commit.
- No container is running for this program. The S55 data-generation container
  (`1336551010bd`) was stopped with `docker stop -t 30` at 09:26Z.

## What is measured and valid

| item | value | receipt |
|---|---|---|
| T2 five-arm verdict | `error_direction_not_supported` | `results/T2-VERDICT.md` |
| Post-trained Qwen3.5-0.8B greedy, non-thinking | GSM8K 54.21 ± 1.37, MATH-500 38.20 ± 2.17 | `results/stage1/` |
| Post-trained Qwen3.5-2B greedy, non-thinking | GSM8K 78.39 ± 1.13, MATH-500 65.80 ± 2.12 | `results/stage1/` |
| Greedy → maj@16 gap, 0.8B, 50-problem subsets | GSM8K +18.0 pp [+6, +32]; MATH-500 +18.0 pp [+6, +30] | `results/stage1/s54-gaps-0.8b.json` |
| Corrected-optimiser bar (Base student), 20M | GSM8K 41.70, MATH-500 18.60, MMLU 46.96, MMLU-Pro 23.02 | `results/stage2/eval/s50-bar-20m-lr1e-5/m20/eval.json` |
| Optimizer defect + invalidations | int8 second-moment zeroing; old runs invalidated | `results/stage2/s50-invalidation-*.json` |
| Termination calibration | 12.5% of truncated answers are loops; 0/5,791 finished | `results/stage1/s56-termination-calibration.json` |
| Frontier ceiling certificate | post-trained 0.8B text tower 752,393,024 | `results/accounting/` |

## What was running at shutdown, and what survived

Stage-2a teacher-solution generation (post-trained 2B generating solutions to
train-split prompts, verified against train labels):

- GSM8K-train: 3,500 prompts sampled, **2,606 verified solutions kept**.
- MATH-train: 672 of 2,400 prompts sampled, **391 verified solutions kept**.
- Total 2,997 accepted examples, written incrementally and preserved as
  `results/artifacts/stage2/s55-stage2a/stage2a-accepted-partial-20261118T0930Z.jsonl`
  and, tracked in git, `results/stage2/data/stage2a-accepted-partial-20261118T0930Z.jsonl.gz`.
- MATH generation is **incomplete** (672/2,400). Resume by re-running the
  generator with the same arguments; do not treat the partial set as the full
  registered corpus.

Nothing else was executing. No training run was in flight.

## Known blockers

- **vLLM image cannot be built on this host as configured.** Docker's data root
  is a 26 GiB tmpfs with ~2.8 GiB free; the pinned build needs several GB.
  No released vLLM supports torch 2.15; the pinned candidate is vLLM 0.34.0 with
  torch 2.14.0. Evidence: `results/stage2/s56-vllm-build.json`, `docs/S56-VLLM.md`.
  Either free Docker space (prune local images, move the data root to disk) or
  accept the transformers fallback (~260–900 tok/s) and label it.
- **Rust Qwen3.5 kernels** (DeltaNet scan, conv state, partial-RoPE gated
  attention) gate only the executed-set audit and NVMe serving (stages 5–6).

## Exact next steps

1. **Finish stage-2a data.** Re-run
   `scripts/s55_stage2_data.py` with the same arguments to complete MATH-train
   (2,400 prompts), then verify acceptance rates and teacher-solution truncation.
2. **Build the SFT stream** with completion-only loss, complete examples, and
   ~10–15% general replay; report processed vs supervised tokens separately.
3. **Train the stage-2 student** (post-trained 0.8B, tied embedding frozen,
   fixed optimiser, lr 1e-5 or the validated rate) and score at **2M, 5M and
   final** tokens.
4. **Score against the frontier** with the identical protocol:
   beat **54.21 GSM8K** and **38.20 MATH-500** with paired problem-level 95%
   intervals above zero, while MMLU and MMLU-Pro stay within 1.0 point.
5. Only then consider stage 3 (looping layers 8–15 with zero-init re-entry gate
   and per-depth LoRA, distilling maj@16 consensus as privileged context).

## Resume commands

```sh
# courts
scripts/court commit_identity
scripts/court perf_numbers
scripts/court status_consistency
scripts/court prereg_provenance

# data generation (resume MATH)
SELVEDGE_MEM=14g scripts/court --gpu oracle -- python3 scripts/s55_stage2_data.py \
  --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
  --datasets math-train --select-math 2400 --k-math 1 --batch-prompts 32 \
  --max-new-tokens-math 1024 --vram-limit-gib 13.0 \
  --out-dir results/artifacts/stage2/s55-stage2a --label stage2a-math
```

Read `docs/S50-GPU-QUEUE.md` for the ordered GPU plan and
`results/REPORT.md` for the generated status blocks. All eligibility labels are
derived from invalidation receipts; do not cite an invalidated number as the
current bar.
