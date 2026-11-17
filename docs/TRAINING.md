# Training path — corpus, teacher targets, objective and the T1 infrastructure court

Status: **implemented infrastructure, no training run.** This document describes
the T1 pieces that exist under `oracle/train/` and `oracle/eval/`, the objective
they are pinned to, and the determination of the five test failures recorded at
revision `56c648fe2c8b4c261112513f0e6d55dc8aea6049`
(`results/courts/logs/oracle-20261115T070709Z-262951.log`). No T1 measurement is
claimed anywhere in this file.

The registered contract is the T1 section of `docs/EXPERIMENT-CONTRACT.md`
(T1.2 architectures, T1.4 teacher targets, T1.6 optimiser, T1.9 accounting).
This document does not change any arm, threshold or kill condition.

## 1. What exists

| module | role | CPU-tested |
| --- | --- | --- |
| `oracle/train/corpus.py` | source specs, row-licence allowlist, document materialisation, contamination checks | yes |
| `oracle/train/teacher_cache.py` | packed top-16 logits / int32 indices / fp32 log-sum-exp, 100 bytes per position, chunk manifest and live verify | yes |
| `oracle/train/modeling.py` | routed-group FFN, specialist bank, dense adapter, zero-point reports | yes |
| `oracle/train/loop.py` | optimiser loop, LRU/global-batch scheduling, the distillation loss, checkpoint hashing | loss only |
| `oracle/train/arms.py` | declares each arm's parameter slices and FLOPs for the Rust admission ledger | yes |
| `oracle/train/budget.py` | Python mirror of the union arithmetic, cross-check only | yes |
| `oracle/train/smoke.py` | step-0 identity / one-step smoke path | no (GPU) |
| `oracle/eval/` | baseline, harness, metrics and task definitions | partly |

The Rust ledger remains the source of truth for the ceiling (`docs/ARCHITECTURE.md`,
T1.9); `oracle/train/budget.py` and `arms.python_mirror` exist only to catch a
declaration that the ledger would refuse.

## 2. The registered objective

`loop.distillation_loss` implements exactly T1.4:

```
L = - sum_j p_t(j) log p_s(j) - p_tail log p_s(tail)
p_t(j)      = exp(v_j - lse)                       # teacher top-16
p_tail      = 1 - sum_j p_t(j)                     # teacher tail MASS
p_s(j)      = softmax(logits)_j                    # student, same support
p_s(tail)   = 1 - sum_j p_s(j)                     # student tail mass
```

Three properties are deliberate:

1. **The support term is exact.** For the stored top-16 indices the student's
   log-probability is computed from the full logits and the full vocabulary
   log-sum-exp, not from a renormalised top-16 softmax. The student's mass on
   the other 151,920 tokens is therefore inside `p_s(tail)`.
2. **The tail is one bucket.** The cache stores the teacher's tail *mass*, never
   its tail *distribution*. The objective is therefore not the full-vocabulary
   cross-entropy, and it is not possible to make it so from this cache. It is
   identical for every arm, which is what the comparison needs.
3. **No hard-label term is added** (T1.4, last sentence). Every arm sees the
   same soft target.

`chunked_logsumexp` computes the vocabulary log-sum-exp in f32 chunks and
checkpoints each chunk, so the f32 temporary is bounded; the combination of
per-chunk log-sum-exps is exact in the log domain.

### Why the tail denominator is clamped at 1e-7

`p_s(tail)` is `1 - sum` of 16 f32 probabilities. Its absolute resolution is the
f32 rounding of that sum, about `1e-7`. When the student puts essentially all
mass on the support, the unclamped `log(0)` is `-inf` and the gradient is not
finite. The clamp replaces a sub-resolution tail with the smallest mass the
arithmetic can distinguish. This is a numerical floor, not a modelling choice,
and it is documented at the call site.

## 3. Determination of the five recorded failures

The five failures were recorded at revision
`56c648fe2c8b4c261112513f0e6d55dc8aea6049` in
`results/courts/logs/oracle-20261115T070709Z-262951.log`. By the time that
revision was committed, the test file already carried corrected expectations and
the suite passed at HEAD: the recorded log was stale, which is not by itself a
resolution. Each case is therefore re-derived below as either an implementation
defect or a wrong test expectation, and the replacement tests are **stricter**
than the ones they replace. The determination of record is this section plus the
tests at the revision that adds it.

| test | recorded failure | determination | resolution |
| --- | --- | --- | --- |
| `test_python_mirror_unions_overlapping_and_adjacent_ranges` | expected 460, got 310 | **wrong expectation.** The fixture's selector is 100 elements, the three `ffn` ranges union to 200 and the `norm` range is 10: the union is 310. 460 is not the sum (310), the execution-only union (210) or the all-identity sum (410). | expectation corrected to the arithmetic; a second test now proves union < sum for repeated reads of one identity |
| `test_document_text_template` | returned `"Problem: None\nSolution: 2"` | **implementation defect.** `document_text` formatted the template before checking for `None`, so a missing field became the literal string `None` and a malformed training document would have entered the corpus. | the `None`/`KeyError` check now runs before formatting |
| `test_pack_unpack_round_trip_is_exact` | `array_equal` false with numerically identical arrays | **wrong test.** The fixture draws random bf16 bit patterns, which include NaNs; `NaN != NaN`, so value equality can never hold for that fixture. | the test now asserts bit equality of the widened values (the strictest form) and the 100-byte record length |
| `test_adapter_zero_report_is_strict` | `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | **wrong test.** The perturbation ran outside `torch.no_grad()`. | the perturbation runs under `no_grad` and touches one element; the report's module count and zero tolerance are now asserted too |
| `test_distillation_loss_matches_bruteforce` | got 3.4980, expected 4.0765 | **wrong test expectation.** The expectation was the full-vocabulary cross-entropy. The registered T1.4 target is the top-16 support plus one tail bucket, so the two quantities are different functions; with 32 random teacher logits and a 16-token support most of the teacher mass is in the tail, which is why the gap is large. | the test is replaced by (a) an independent float64 transcription of T1.4, (b) support-order invariance, and (c) the decisive check: at full support (`k = vocab`) the objective equals the full cross-entropy to `1e-9` and its gradient is finite |

The quoted losses are a historical quotation of the recorded log at revision
`56c648fe2c8b4c261112513f0e6d55dc8aea6049`; they are not current measurements.

The distillation-loss case is the one that mattered. The failing number was not
a bug in the loss: it was the support-plus-tail objective disagreeing with an
expectation that contradicted the contract. The new full-support test is what
distinguishes "the implementation is a defensible objective" from "the test was
edited until it passed": at `k = vocab` the tail is empty and the implementation
must reproduce the ordinary cross-entropy exactly, which it does.

## 4. What the infrastructure does not establish

- **No arm has been trained.** The step-0 identity and one-step smoke paths
  exist; a full T1 run has not happened, and no quality number is claimed.
- **The teacher cache is not built at the registered budget.** The builder,
  manifest, chunk checksums and the live-verify comparison exist; the 12.5 GiB
  cache for the registered token budget has not been materialised.
- **The tail bucket is lossy for the teacher, not for the student.** The student
  distribution is exact; the teacher target is restricted. A different objective
  (full-vocabulary teacher, hard labels, top-1 only) would be a different arm
  and is not registered.
- **bf16 teacher values.** The top-16 logits are stored in bf16 per T1.4, so the
  teacher probabilities carry bf16 resolution even though the sum is computed in
  f32.
- **The corpus licence filter is row-level.** `codeparrot-clean` has no
  dataset-level licence; the allowlist is a row filter and is recorded as a
  residual risk in `docs/LIMITATIONS.md`.

## 5. Reproduce

```bash
# CPU infrastructure court (the five cases above live here)
scripts/court oracle -- python3 -m pytest oracle/tests/test_t1_infrastructure.py -q

# Full oracle suite
scripts/court oracle -- python3 -m pytest oracle/tests -q

# Corpus materialisation, teacher cache and verification (GPU, explicit network)
scripts/court --net --gpu oracle -- python3 oracle/train/corpus.py build
scripts/court --gpu oracle -- python3 oracle/train/teacher_cache.py build
scripts/court --gpu oracle -- python3 oracle/train/teacher_cache.py verify
```

The court writes a receipt under `results/courts/` for every run; a claim in
this document that is not backed by such a receipt is labelled as design intent.

## 6. T2 modules (error-directed capacity matrix)

The T2 contract is `docs/EXPERIMENT-CONTRACT.md` T2 (with amendments A8 and A9).
This section describes what exists, not what it proves. No full-budget T2 run
has started; the evidence here is a five-arm smoke.

| module | role | status |
| --- | --- | --- |
| `oracle/train/memory.py` | deterministic hashed memory: two n-gram heads, 2^18-row 256-dim tables at four insertion layers, two reads per token per layer, a 512->1024 projection and a sigmoid gate | implemented, CPU-tested |
| `oracle/train/modeling.py` | `build_arm(..., profile="t2")` for `D`, `M`, `X`, `H`, `H*`; deterministic slot-wise combine; extra-parameter ledger; resource report | implemented, CPU-tested |
| `oracle/train/utility.py` | leave-one-expert-out utility labels, checkpointed with a manifest; the listwise-KL router head | implemented, exercised in the H* smoke |
| `oracle/train/loop.py` | T2 phases: warm-up, utility labelling, frozen-router joint phase; deterministic kernels; micro-position sub-batching | implemented, exercised |
| `oracle/eval/instrument_check.py` | P2.4: the compact target against a live full-vocabulary teacher | implemented, receipted |
| `oracle/eval/baseline.py` | evaluates any arm through the T1.7 harness and adds the bytes/token resource block | implemented, exercised |
| `crates/selvedge-cli/tests/t2_admission.rs` | admits every T2 config at its design count and refuses each when pushed over the ceiling | implemented, passing |

### 6.1 Memory accounting, exactly

Per insertion layer the active union is the two retrieved 256-element rows, the
`512x1024` projection and the `1024x1` gate:

```
active/layer = 2*256 + 512*1024 + 1024 = 525,824
four layers  = 2,103,296
```

The table itself is **not** charged; a read touches two rows. Total accessible
memory state is `4 * 2^18 * 256 = 268,435,456` values plus the projection and
gates, i.e. `270,536,704` added elements. `HashedMemory.active_parameter_report()`
emits this, and `results/train/admission-configs/t2/{M,H}.json` declares exactly
the same slices, which the Rust ledger admits and a test cross-checks.

Bytes are kept separate and labelled: resident table = `536,870,912` bytes at
bf16 (the stored and trained dtype; `1,073,741,824` is the fp32-equivalent size)
and modelled read bytes per token = `4,206,592` at bf16. The read figure is a
modelled lower bound on distinct identities read, never measured I/O.

### 6.2 The utility pipeline

`utility.label_utilities` measures, per layer and specialist,
`u_e(t) = L(without e) - L(with e)` over a fixed calibration slice, with the
full bank as the reference set: one pass for the reference and one per
specialist, 9 passes per layer over 28 layers. The pre-expert hidden state is
captured by a forward pre-hook under `torch.no_grad()` and stored with the
labels; `utility.train_router` then fits the layer's router to
`softmax(standardised utilities)` with a listwise KL, never running the core
again. Leakage is argued three ways in the contract (T2.5) and tested: the
features are bitwise identical under two different expert masks, the labels
never enter a forward, and the calibration range is inside the
contamination-filtered training stream.

### 6.3 Throughput and budget

The measured per-arm throughput, VRAM peak, hours at the declared budget and
resume deltas are generated in `results/T2-PLAN.md` §2 from
`results/train/budget-20261115T130456Z.json`. The declared budget is derived
from the slowest arm by T2.7; the arithmetic is in contract A8 and the number is
generated in `results/T2-PLAN.md`.
No number is repeated here, because a repeated measurement would have to be
regenerated anyway.

### 6.4 What the T2 infrastructure does not establish

- **No T2 arm has been trained at budget.** The smoke is 50 steps per arm.
- **The utility labels are one-shot** and measured in the full-bank reference
  set; an expert that duplicates another is under-penalised.
- **The compact target is optimistic by a measured amount** (A9), and the
  safeguard is to report both targets rather than to retrain on a full target.
- **Determinism is a kernel property.** It holds here with
  `torch.use_deterministic_algorithms(True)`; without the flag the dense control
  itself is not reproducible.

## 7. Temporal state in a training loop (and the rule that was wrong)

No looped arm exists yet, but the state contract binds every later stage, so it
is written down here before the code that will depend on it. The rule, corrected
at S38 (`docs/PROGRAM-PLAN.md` P19.4, `docs/RUNTIME.md` §13.3):

1. **State is carried across tokens** inside one sequence. `S_{k,t} -> S_{k,t+1}`.
   A pass is not a one-token function and the training loop must not reset the
   state between positions of a sequence.
2. **State is reset between sequences**, at the batch-sample boundary: a new
   document is an independent sequence, so every pass starts from the zero state
   for it. With packed sequences this is a reset at each document boundary within
   the pack, not only at the pack boundary.
3. **Each pass owns its state.** A K-pass arm allocates K independent
   `(conv, DeltaNet, KV)` triples; the gradient path may connect passes through
their *activations*, never through their state buffers.

The reference implementation exposes exactly this (`PassState` in
`oracle/reference/qwen35.py`, with the `-Base` receipt at
`results/reference/20261117T003300Z-qwen35-base-state-contract.json`). The stage-2
trainer (`oracle/frontier/train.py`) trains a single pass and therefore needs no
per-pass bookkeeping yet; when it grows a loop it must allocate the state from
the same object rather than inventing a second convention.

The checkpoint consequence is small but real: a resumable arm must persist the
state or a definable resume point, or declare that it resumes at a sequence
boundary. The stage-2 trainer persists optimiser and RNG state and starts each
step from a fresh sequence, which is the conservative choice and is why its
resume is exact.
