# SELVEDGE results — honest current state

Status: **M2 complete**. The pre-registered M1 frontier and the M2 per-token
selector/oracle experiment have been measured. M1 rejected static whole-group
omission. M2 rejects the inference-only, post-hoc per-token selection branch:
even a non-deployable per-token oracle at the compliant budget is far outside
the registered usable tier, and no deployable selector beats activation
magnitude at matched realized active parameters. The reports are
`results/M1-FRONTIER.md` and `results/M2-SELECTORS.md`; the registered contract
is `docs/EXPERIMENT-CONTRACT.md` (M1 and M2 sections, amendment A3).

This file states what is established, with the receipt that establishes it, and
what is not. It is not a benchmark report: no latency, throughput or quality
number has been measured, and none is claimed.

## Established

1. **A pinned teacher oracle exists and runs offline after one documented fetch
   step.** The oracle court image pins Python, torch (CUDA 13.2 build),
   transformers, tokenizers, safetensors and numpy. A real CUDA matmul and SiLU
   ran inside the container under device passthrough; the probe and receipt record
   the device and the `--gpus all` fallback reason.
2. **Two teachers are acquired, licensed and hash-pinned.** Qwen3-0.6B and
   Qwen3-1.7B at fixed commit SHAs, Apache-2.0, with every file's SHA-256 and
   size in `models/MODELS.lock.json`. The license text is reproduced verbatim in
   `docs/MODELS.md`.
3. **Exact accounting on real safetensors.** `selvedge inspect-model` reads the
   real headers, rejects unclassified tensors, verifies the embedding/output tie
   byte-for-byte, and reports component counts, the always-active backbone,
   removable FFN, ceiling arithmetic and FLOP estimates.
4. **The 1e9 ceiling is not binding for Qwen3-0.6B and is binding for
   Qwen3-1.7B.** Qwen3-4B's backbone alone exceeds the ceiling, so it cannot be
   a compliant teacher; it is deliberately not downloaded.
5. **A real checkpoint converts to group shards and reconstructs exactly on real
   activations.** The default grouping is sixteen contiguous groups per layer,
   stored as f32; the shard count and payload bytes are in each receipt. The
   grouped-from-bundle path and the dense-from-checkpoint path agree bitwise on
   every compared component.
6. **The f32-equivalent teacher computation is reproduced within a documented
   bound**, and the native bf16 output is reproduced when the teacher's rounding
   sequence is emulated explicitly. Both checks pass with zero violations.
7. **The masked-MLP instrument is exact on the measured paths.** With every
   group selected it is bitwise equal to the unmodified dense f32 teacher on
   real held-out windows. An independent Rust grouped sum over the published
   bundle agrees with the Python grouped reference on a real layer and a real
   four-group subset within the registered f32 bound.
8. **The quality-versus-active-parameter frontier is measured, with receipts.**
   Static uniform, magnitude, frequency, whole-layer and greedy allocations
   were evaluated on a pinned, hash-split held-out set with paired bootstrap
   CIs. Every arm at or below the ceiling is far outside the registered
   acceptable and usable tiers. The curve is a cliff near the fully dense end.
9. **The static result is negative and reproducible.** Two independent complete
   runs produced bitwise-identical per-arm aggregates. Failed attempts,
   amendments and the dirty-worktree superseded receipts are retained.
10. **The per-token oracle potential is measured and negative.** A
    non-deployable oracle that chooses an exact local subset per token and per
    layer, at the same realized budget, was evaluated at 16, 64 and 4 groups.
    The 64-group oracle is the best and is still far outside the registered
    usable tier. A bounded exact-KL refinement does not rescue it.
11. **Per-token selection helps but cannot escape the cliff.** At matched
    realized active parameters, per-token activation magnitude beats the best
    static M1 arm, and every other deployable per-token arm is worse than
    per-token magnitude. The learned unary, learned interaction-aware,
    gradient-attribution and corrected-attribution arms are all worse.
12. **The Rust Atlas boundary is implemented and parity-tested.**
    `selvedge-atlas` owns the quality-observation record, the interaction
    graph, the leakage-bounded selector input, seven selectors, the two-view
    head contract and independent matched-budget accounting. `selvedge atlas
    fit|select|compare|cost` run; the Python/Rust parity, leakage, graph and
    calibration tests pass on exported real fixtures.

## Numbers (generated from receipts)

- Qwen3-0.6B unique learned parameters:
  <!-- BEGIN GENERATED: report-06b-unique -->
596049920
  <!-- END GENERATED: report-06b-unique -->
- Qwen3-1.7B always-active backbone:
  <!-- BEGIN GENERATED: report-17b-backbone -->
663610368
  <!-- END GENERATED: report-17b-backbone -->
- Qwen3-1.7B affordable FFN elements per token:
  <!-- BEGIN GENERATED: report-17b-max-ffn -->
336389632
  <!-- END GENERATED: report-17b-max-ffn -->
- Qwen3-4B backbone (config projection, weights not downloaded):
  <!-- BEGIN GENERATED: report-4b-backbone -->
1332870656
  <!-- END GENERATED: report-4b-backbone -->
- Qwen3-0.6B grouped f32 vs teacher f32 max abs:
  <!-- BEGIN GENERATED: report-06b-f32-max-abs -->
0.005859375
  <!-- END GENERATED: report-06b-f32-max-abs -->
- Qwen3-1.7B grouped bf16-emulated vs teacher bf16 max abs:
  <!-- BEGIN GENERATED: report-17b-bf16-max-abs -->
4.0
  <!-- END GENERATED: report-17b-bf16-max-abs -->
- Qwen3-0.6B f32 bundle payload bytes:
  <!-- BEGIN GENERATED: report-06b-payload -->
1056964608
  <!-- END GENERATED: report-06b-payload -->
- Qwen3-1.7B f32 bundle payload bytes:
  <!-- BEGIN GENERATED: report-17b-payload -->
4227858432
  <!-- END GENERATED: report-17b-payload -->
- Best compliant arm (greedy-uniform5) active unique parameters:
  <!-- BEGIN GENERATED: report-best-active -->
998630400
  <!-- END GENERATED: report-best-active -->
- Best compliant arm mean teacher KL, nats/token:
  <!-- BEGIN GENERATED: report-best-kl -->
7.706657
  <!-- END GENERATED: report-best-kl -->
- Best compliant arm PPL ratio:
  <!-- BEGIN GENERATED: report-best-ppl -->
1196.792
  <!-- END GENERATED: report-best-ppl -->
- Uniform magnitude k15 (noncompliant) mean teacher KL:
  <!-- BEGIN GENERATED: report-k15-kl -->
0.279397
  <!-- END GENERATED: report-k15-kl -->
- Registered acceptable tier KL bound (nats/token):
  <!-- BEGIN GENERATED: report-acceptable-kl -->
0.05
  <!-- END GENERATED: report-acceptable-kl -->

## Not established

- **No capability claim.** The frontier measures next-token distributions on
  pinned text. KL on Wikipedia is not a code, math, reasoning or factuality
  benchmark. See `docs/LIMITATIONS.md`.
- **No trained or upcycled model.** Post-hoc per-token selection is measured
  and negative. A model trained or upcycled to tolerate the sparse FFN, a
  distilled smaller teacher, quantization and a learned memory layer are
  untested. The negative result bounds post-hoc inference-only selection, not
  sparsity in general.
- **No measured speed, latency, storage bandwidth or FLOPs.** FLOP values are
  derived from realized channel counts, not hardware counters. Arm wall clock
  is a cost observation, not a benchmark.
- **No scheduler or runtime.** `selvedge-atlas` exists; `selvedge-store`,
  `selvedge-sim` and `selvedge-runtime` do not. The Atlas cost head is
  explicitly unavailable until M3, and no transfer time is assumed.
- **The native-bf16 match is a model of PyTorch's arithmetic on these inputs.**
  A different kernel or backend may round differently.

## Artifacts

The activation files, bundles, dataset cache and derived split live under the
gitignored `results/artifacts/` directory; the tracked receipts carry their
hashes, sizes and measurements. Rebuilding is documented in
`docs/BUDGET-MATH.md`, `docs/DATASETS.md` and `docs/WORK-LOG.md`.

## Next

Move to the trained/upcycled branch with a fresh pre-registration: a dense
compliant Qwen3-0.6B base plus external capacity, an UpIT/SIMoE-style
dense-to-sparse upcycle of Qwen3-1.7B trained to tolerate the compliant group
budget, and a distilled smaller teacher, matched for active parameters, FLOPs
and training tokens. The M2 oracle bound says post-hoc selection cannot close
the gap; the next question is whether training can change the model so the same
budget is usable. Do not resume inference-only selector tuning without a new
oracle bound.
