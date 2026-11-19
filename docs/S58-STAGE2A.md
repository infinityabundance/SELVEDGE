# S58 — stage-2a verified-teacher-solution SFT: run report

Status: **complete, and the arm does not win.** The corpus, stream, training run
and every score below are receipted. This page states what was done, what was
measured, and what the registered next routes cost — it makes no capability
claim that a receipt does not carry.

Every measured number on this page names the receipt that produced it, with that
receipt's stamp and the repository revision recorded inside it.

| artifact | receipt | stamp | revision |
|---|---|---|---|
| MATH-train at the 1024 cap | `results/stage2/data/s58-math-1024-receipt-20261118T162618Z.json` | `20261118T162618Z` | `eaeae26` |
| cap pilot, 200 prompts at 2048 | `results/stage2/data/s58-math-cap2048-pilot-receipt-20261118T163550Z.json` | `20261118T163550Z` | `eaeae26` |
| MATH-train at the 2048 cap | `results/stage2/data/s58-math-2048-receipt-20261118T175815Z.json` | `20261118T175815Z` | `9ea15c9` |
| GSM8K 1/12 shard at 1024 | `results/stage2/data/s58-gsm8k-shard0-receipt-20261118T180615Z.json` | `20261118T180615Z` | `9ea15c9` |
| merged corpus | `results/stage2/data/s58-corpus-combined-receipt-20261118T180643Z.json` | `20261118T180643Z` | `9ea15c9` |
| SFT stream | `results/stage2/data/s58-sft-stream-receipt-20261118T193245Z.json` | `20261118T193245Z` | see note |
| training | `results/stage2/runs/s58-stage2a/receipt.json` | `20261118T210038Z` | `e5381f5` |
| held-out solution CE | `results/stage2/s58-stage2a-heldout-ce.json` | `20261118T210346Z` | `e5381f5` |
| final accuracy | `results/stage1/s58-stage2a-final-nonthinking-greedy.json` | `20261118T220558Z` | `b1f9c1c` |
| paired intervals | `results/stage1/s58-paired-final-vs-post.json` | `20261118T221829Z` | `b1f9c1c` |
| post-trained MMLU control | `results/stage2/eval/s58-post-0.8b-baseline-mmlu/eval.json` | `20261118T224845Z` | `bd09763` |

Note on the stream receipt: `scripts/s55_stage2a_examples.py` did not record a
`revision` field when that receipt was written. It now does. The receipt's
`examples.sha256` (`9426e36d…`) is the byte-identical stream the trainer
consumed, as recorded in `results/stage2/runs/s58-stage2a/launch.json`.

## 1. Finishing the corpus without clobbering it

`scripts/s55_stage2_data.py` opens its accepted JSONL with mode `"w"`. A re-run
of the same `--out-dir/--label` truncates whatever is there, which is how the
2,997-row partial corpus was one command away from destruction. Two changes:

* the generator now **fails closed** on an existing accepted file unless
  `--overwrite` is passed, and the receipt records the write mode;
* `scripts/s58_merge_corpus.py` merges accepted files: exact-token dedup, one
  solution per prompt, deterministic order, per-source and per-run accounting,
  contamination screen, optional tracked `.gz`.

No existing artifact was rewritten. The partial corpus
(`results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl`, tracked as
`results/stage2/data/stage2a-accepted-partial-20261118T0930Z.jsonl.gz`) is
unchanged and is source 1 of the merge.

### The 1024 cap was censoring MATH

MATH-train at the registered 1024 cap, all 2,400 prompts, k = 1
(`s58-math-1024-receipt-20261118T162618Z.json`, stamp `20261118T162618Z`,
revision `eaeae26`): 2,400 sampled, 991 verified-kept, 1,267 truncated without
EOS (52.8%), 142 wrong, 0 duplicates, maximum contamination Jaccard 0.155.

Half the attempts reaching the cap is the cap doing the selection, so the cap
question was measured rather than assumed. The same 200 prompts (one shard of
the same stratified selection) were re-generated at 2048
(`s58-math-cap2048-pilot-receipt-20261118T163550Z.json`, stamp
`20261118T163550Z`, revision `eaeae26`): acceptance 49.0% → 58.5%, truncation
52.8% → 33.5%, 28 prompts gained and 9 lost. That justifies the registered cap
increase; the full 2,400-prompt 2048 run accepted 1,342 with 35.0% truncation
(`s58-math-2048-receipt-20261118T175815Z.json`, stamp `20261118T175815Z`,
revision `9ea15c9`).

A cap is a censoring limit, not a distributional parameter, so the merged corpus
is the **union** of both MATH runs plus the GSM8K rows.

### GSM8K: the accepted rows survived, the generation receipt did not

The GSM8K half of the partial run finished before the shutdown (2,606 accepted
over the 3,500 registered prompts) but its receipt died with the container. The
accepted count and the selection are recoverable — every accepted id lies inside
`rft.stratified_select(items, 3500, seed 5500)`, verified — but the truncation
rate is not. A 1/12 shard of the same selection was re-run at 1024
(`s58-gsm8k-shard0-receipt-20261118T180615Z.json`, stamp `20261118T180615Z`,
revision `9ea15c9`): 292 sampled, 221 kept (75.7%), 41 truncated (14.0%), 30
wrong. This is a measured estimate on a stratified 1/12 sample, labelled as an
estimate; it is not presented as the full run's number.

### Merged corpus

`s58-corpus-combined-receipt-20261118T180643Z.json`, stamp `20261118T180643Z`,
revision `9ea15c9`:

| source | rows | distinct prompts | coverage of the registered prompts | solution tokens |
|---|---|---|---|---|
| GSM8K | 2,639 | 2,639 | 0.754 of 3,500 | 1,101,239 |
| MATH | 1,445 | 1,445 | 0.602 of 2,400 | 1,129,465 |
| total | 4,084 | 4,084 | — | 2,230,704 |

Merge removed 5 exact duplicate solutions and 1,462 second-solutions-per-prompt
(the 1024 MATH run and the 2048 MATH run cover overlapping prompts; the first
source wins). Contamination screen against every cached eval set: maximum
5-word-shingle Jaccard 0.155, discard threshold 0.20, fail threshold 0.50, no
fail-closed.

## 2. The SFT stream

`scripts/s55_stage2a_examples.py`, receipt `s58-sft-stream-receipt-20261118T193245Z.json`
(stamp `20261118T193245Z`):

* completion-only: `prompt_ids` are `-100`, every solution token and the
  trailing EOS are supervised (`frontier.sft_data.completion_example`);
* one complete example per row (`--sft-examples-per-row 1`): no example is split
  and no example attends to another;
* teacher rows 4,002 with 2,186,321 supervised solution tokens; general replay
  565 `fineweb-edu` documents with 327,043 supervised tokens = **0.130** replay,
  inside the registered 10–15%;
* held-out slice: 82 rows drawn as a seeded stratified 2% (`dataset` ×
  solution-length quartile), removed from training, used only for early stop;
* at `seq_len` 1250, 191 examples exceed the row, are dropped **whole** and
  counted, never split.

At the trainer's packing (`results/stage2/runs/s58-stage2a/manifest.json`): 4,376
rows, 2,225,863 supervised positions, 2,929,820 padding positions.

**Processed tokens versus supervised solution tokens** are never conflated:
10,940,000 processed tokens (two passes) against 2,225,863 supervised positions
per pass, i.e. 4,451,726 supervised positions over the run. The processed number
includes padding and prompt positions by construction.

### Row order

The first stream concatenated the teacher rows and then the contiguous replay
block. That is an implementation artifact, and it broke the run twice (section
3). The stream now writes a **seeded deterministic permutation**
(`--order seeded-shuffle`, seed 5500) and records the mode, seed and row count in
the receipt. The trainer still never shuffles; the order is pinned in the file
and hashed into the run manifest.

## 3. Training

`scripts/s57_stage2a_train.sh` → `oracle/frontier/train.py`, receipt
`results/stage2/runs/s58-stage2a/receipt.json` (stamp `20261118T210038Z`,
revision `e5381f5`):

* student: post-trained Qwen3.5-0.8B `2fc06364…`, full text tower, tied
  embedding frozen (498,113,344 trained of 752,393,024);
* objective: `--objective sft`, completion-only CE, `kl_weight 0`;
* optimiser: Adam8bit block 256, int8 first moment (retained by
  `s55-real-gradient-validation.json`), bf16 second moment, stochastic weight
  rounding; cosine to 0.1, warmup 200, clip 1.0, lr 1e-5, seed 0;
* 4,376 steps, 2,225,863 supervised positions per pass, **10,940,000 processed
  tokens**, 2,081 tok/s, 5,257 s;
* peak 12.728 GiB allocated / 13.086 GiB reserved VRAM; container cgroup peak
  15,032,385,536 bytes with zero OOM kills — the container **reached** its 14g
  limit, so the next run of this shape must be given more;
* the update tripwire never fired; its maximum was 8.93 of the 10.0 bound, which
  is close enough to record;
* at peak LR, 89.2% of the 498,113,344 trainable elements were unchanged
  bitwise, against 94.7% that round-to-nearest would have left unchanged: the
  stochastic rounding moved the elements the naive update would freeze.

### Two aborted launches, both diagnosed and kept

| launch | aborted at | ratio | cause | evidence |
|---|---|---|---|---|
| 1 | step 218 | 1.517 | baseline window inside the 200-step warmup: LR ≤ 5e-6 against full 1e-5 | `results/stage2/runs/s58-stage2a-abort-20261118T181418Z/abort.json` |
| 2 | step 1952 | 1.534 | contiguous replay block: teacher-solution CE 0.31 against replay CE 2.8 | `results/stage2/runs/s58-stage2a-abort2-20261118T191215Z/abort.json` |

Neither was a divergence: no non-finite step, no loss spike away from the data
boundary, max|update|/lr inside the 10.0 bound. The first fix moved the baseline
window to 400 steps (warmup plus 200 at full LR); the second interleaved the
replay. Factor, lr, warmup, schedule, optimiser and budget are unchanged.

## 4. Held-out solution CE is negative

`results/stage2/s58-stage2a-heldout-ce.json` (stamp `20261118T210346Z`,
revision `e5381f5`), 79 rows / 40,195 supervised positions, completion-only:

| checkpoint | tokens seen | held-out CE |
|---|---|---|
| base (step 0) | 0 | 0.2495 |
| 000800 | 2,000,000 | 0.3478 |
| 001200 | 3,000,000 | 0.3448 |
| 001600 | 4,000,000 | 0.3327 |
| 002000 | 5,000,000 | 0.3169 |
| 003200 | 8,000,000 | 0.3133 |
| 004000 | 10,000,000 | **0.3087** |
| 004376 | 10,940,000 | 0.3104 |

Every checkpoint is worse than the base model on the distribution it was trained
on, and both datasets degrade (GSM8K 0.2494 → 0.2969 at step 2000; MATH 0.2495 →
0.3418). The registered early-stop rule selects the least-bad checkpoint
(`ckpt-004000`), not an improvement. **This is the run's clearest negative
result**, and it did not stop the accuracy from improving on GSM8K — CE on
teacher wording and answer accuracy are different quantities.

## 5. Scores

Same instrument as the stage-1 baseline (`scripts/s54_post_trained_eval.py`):
full GSM8K (1,319) and MATH-500 (500), greedy, non-thinking, zero-shot with
`\boxed{}`, batch 32, cap 4,096, three extraction bases side by side.
Receipt `results/stage1/s58-stage2a-final-nonthinking-greedy.json` (stamp
`20261118T220558Z`, revision `b1f9c1c`); the control is
`results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json`.

| task | basis | post-trained 0.8B | stage-2a | delta |
|---|---|---|---|---|
| GSM8K | flexible | 0.5421 | 0.5709 | +0.0288 |
| GSM8K | strict | 0.5406 | 0.5701 | +0.0295 |
| GSM8K | exact | 0.5489 | 0.5792 | +0.0303 |
| MATH-500 | flexible | 0.3820 | 0.2720 | −0.1100 |
| MATH-500 | strict | 0.3820 | 0.2720 | −0.1100 |
| MATH-500 | exact | 0.4060 | 0.2900 | −0.1160 |

Prompt-level paired bootstrap (4,000 resamples, seed 53) from
`results/stage1/s58-paired-final-vs-post.json` (stamp `20261118T221829Z`,
revision `b1f9c1c`):

| task | delta | 95% interval | interval above zero |
|---|---|---|---|
| GSM8K | +0.0288 | [+0.0008, +0.0576] | yes, barely |
| MATH-500 | −0.1100 | [−0.1540, −0.0660] | no |

Termination at the same checkpoints (from the eval receipt's own summary):

| task | truncated | loop-classified | mean new tokens |
|---|---|---|---|
| GSM8K | 0.0227 (base 0.0697) | 0.0053 | 573 (base 722) |
| MATH-500 | 0.2100 (base 0.3200) | 0.0840 | 1,695 (base 1,897) |

MMLU and MMLU-Pro on the registered total-cap harness (MMLU 14,042 items,
MMLU-Pro 504), control `s58-post-0.8b-baseline-mmlu/eval.json` (stamp
`20261118T224845Z`, revision `bd09763`):

| task | post-trained 0.8B | stage-2a | delta | paired 95% interval | bound |
|---|---|---|---|---|---|
| MMLU | 0.5048 | 0.4854 | −0.0194 | [−0.0276, −0.0113] | **fails** the 1.0-point bound on both clauses |
| MMLU-Pro | 0.2242 | 0.2222 | −0.0020 | [−0.0437, +0.0397] | point estimate inside the bound; the interval clause is not met on 504 items |

Receipts: `results/stage2/eval/s58-stage2a-final-mmlu/eval.json` and
`results/stage2/s58-paired-retention-vs-post.json`. The retention clause is
assessed on its own terms: the bound has two parts (point drop ≤ 1.0 and paired
interval lower bound ≥ −1.0), and MMLU fails both. MMLU-Pro's point drop is
0.2 points but a 504-item paired interval cannot resolve a 1-point bound, so its
lower bound is −4.37; that is a statement about sample size, not a claim that
MMLU-Pro moved.

### Milestone cadence (the early dip is real)

The S56 cadence scores every registered milestone on the same instrument, and
that is what found the shape of this run. Receipts:
`results/stage1/s58-stage2a-m2-nonthinking-greedy.json` (stamp
`20261119T010711Z`, checkpoint step 800, 2,000,000 tokens seen) and the final
receipt above.

| milestone | processed tokens | GSM8K flexible | MATH-500 flexible | GSM8K truncated | MATH truncated / loop |
|---|---|---|---|---|---|
| base | 0 | 0.5421 | 0.3820 | 0.0697 | 0.3200 / — |
| 2M | 2,000,000 | 0.3101 | 0.1080 | 0.0675 | 0.3080 / 0.1640 |
| final (ckpt-004000) | 10,000,000 | 0.5709 | 0.2720 | 0.0227 | 0.2100 / 0.0840 |

The 2M checkpoint is catastrophic on both tasks: GSM8K −23.2 points (paired
interval [−0.2623, −0.2024]) and MATH-500 −27.4 points ([−0.3180, −0.2300]),
against a held-out CE that is also the worst measured (0.3478). The endpoint
recovered GSM8K above the base model and recovered MATH only part of the way.
A run scored only at its endpoint would have hidden this entirely, and the
trajectory is the strongest evidence in this campaign that lr 1e-5 is too high
for this objective: the model is thrown far off its optimum early and the
cosine decay lets it walk part of the way back.

## 6. Verdict

**Not a win.** The registered win requires both GSM8K and MATH-500 to beat the
post-trained 0.8B with prompt-level paired 95% intervals above zero, and the
general-capability bound to hold. GSM8K clears the interval but MATH-500 loses
11.0 points with an interval entirely below zero, so the conjunction fails on
the second clause; MMLU independently fails the retention bound, so the arm
fails both halves of the win condition.

The failure mode is measured rather than inferred: the candidate boxes more MATH
answers (78.8% vs 68.2%), truncates less (21.0% vs 32.0%) and is wrong more
often, with shorter generations. It learned to answer briefly and confidently,
which is what a corpus dominated by short GSM8K solutions teaches, and MATH
rewards the long derivation the base model was already producing.

## 7. Registered next routes, and what each would cost

1. **More or better teacher tokens.** Partly applied: the 2048 cap raised MATH
   prompt coverage from 0.413 to 0.602 with direct evidence (section 1), and the
   arm still loses MATH. More of the same data is not indicated by this result.
2. **Same-prefix teacher KL.** Not run, and not runnable as registered on this
   device: the student peaks at 12.728 GiB at `seq_len` 1250 and the post-trained
   2B teacher adds ~3.8 GiB of weights plus its activations, which is over the
   16,376 MiB card. Shrinking `seq_len` to fit would drop most MATH solutions,
   which is the corpus the arm needs. This needs either a longer-context device
   or a registered shorter-row variant.
3. **A validated higher lr.** Contraindicated by measurement: the training CE
   rose monotonically with the LR ramp in both launches and the held-out CE rose
   at every checkpoint, so a higher rate has no support here.
4. **A second epoch.** Already executed: the run makes two passes and the
   held-out CE minimum sits at step 4000, inside the second pass.

What the evidence points at, for the next registration rather than this one: the
loss should not be plain completion CE on a corpus whose supervised-headroom is
already ~0.25 nats/token, and a mixed GSM/MATH corpus teaches one answer length
for two different tasks. A length- or subject-conditioned curriculum, or a
retention term against the post-trained model, are both untested here and are
not claimed.

## 8. Reproduction

```sh
# corpus (GPU)
SELVEDGE_MEM=14g scripts/court --gpu oracle -- python3 scripts/s55_stage2_data.py \
  --teacher-dir /hf-cache/models/Qwen/Qwen3.5-2B/15852e8c… \
  --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364… \
  --datasets math-train --select-math 2400 --k-math 1 --batch-prompts 32 \
  --max-new-tokens-math 2048 --vram-limit-gib 13.0 \
  --out-dir results/artifacts/stage2/s58-stage2a-math-2048 --label stage2a-math-2048

# merge (CPU)
SELVEDGE_MEM=6g scripts/court oracle -- python3 scripts/s58_merge_corpus.py \
  --source results/artifacts/stage2/s55-stage2a/stage2a-accepted.jsonl \
  --source results/artifacts/stage2/s58-stage2a-math-full/stage2a-math-full-accepted.jsonl \
  --source results/artifacts/stage2/s58-stage2a-math-2048/stage2a-math-2048-accepted.jsonl \
  --source results/artifacts/stage2/s58-gsm8k-shard0/gsm8k-1024-shard0-accepted.jsonl \
  --prompt-universe math=2400 --prompt-universe gsm8k=3500 \
  --label stage2a-accepted-combined --out-dir results/artifacts/stage2/s58-combined

# stream (CPU)
SELVEDGE_MEM=8g scripts/court oracle -- python3 scripts/s55_stage2a_examples.py \
  --accepted results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl \
  --tokenizer-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364… \
  --label stage2a-combined --out-dir results/artifacts/stage2/s58-combined \
  --replay-fraction 0.13 --replay-max-tokens 768 --seq-len 1250 \
  --holdout-fraction 0.02 --order seeded-shuffle

# train (GPU)
SELVEDGE_MEM=14g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/s57_stage2a_train.sh \
  --examples results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl \
  --examples-receipt results/artifacts/stage2/s58-combined/stage2a-combined-examples-receipt.json \
  --run-dir results/stage2/runs/s58-stage2a

# held-out CE, then score
SELVEDGE_MEM=12g scripts/court --gpu oracle -- python3 scripts/s55_heldout_ce.py \
  --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364… \
  --holdout results/artifacts/stage2/s58-combined/stage2a-combined-holdout.jsonl \
  --run-dir results/stage2/runs/s58-stage2a --seq-len 1250 \
  --out results/stage2/s58-stage2a-heldout-ce.json

SELVEDGE_MEM=8g scripts/court --gpu eval -- python3 scripts/s54_post_trained_eval.py \
  --model-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364… \
  --model-id Qwen/Qwen3.5-0.8B --revision 2fc06364… \
  --full-checkpoint results/stage2/runs/s58-stage2a/ckpt-004000.pt \
  --tasks gsm8k,math500 --modes non-thinking --decodings greedy --batch-size 32 \
  --vram-limit-gib 13.0 --out results/stage1/s58-stage2a-final-nonthinking-greedy.json
```
