# S53 post-trained route — registered, not scored

The machine-readable registration is
`docs/ws19-post-trained-route-registration-v1.json`; the identity lock is
`models/FRONTIER-POST-TRAINED.lock.json`. Both were committed before any
post-trained checkpoint was scored. Nothing in this page is a measurement.

## What changed

The win target is the **post-trained** `Qwen/Qwen3.5-0.8B`
(`2fc0636…`, text tower 752,393,024 elements), not `Qwen3.5-0.8B-Base`.
Base remains a control. The post-trained 2B (`15852e8…`, text tower
1,881,825,088) is the stage-2 teacher and the stage-1 upper reference.

Both post-trained checkpoints are audited:

- 0.8B: `results/accounting/20261116T183609Z-qwen35-08b-text-only.json`
  (per-tensor elements) and
  `results/frontier/20261117T002500Z-revision-compare-08b-chat-vs-base.json`
  (header/tensor table/shard hash vs Base);
- 2B: `results/frontier/20261117T233931Z-revision-compare-2b-chat-vs-base.json`
  (fresh S53 audit), `results/frontier/s53-fetch-2b-chat.json` (fetched
  `aa33250c…`) and `results/frontier-models/20261116-probe-p2.json`
  (per-tensor elements).

Licences are Apache-2.0 in every case; the chat and base licence files hash
differently and each hash is recorded separately. The chat tokenizer adds
`<tool_response>`, `</tool_response>`, `<think>`, `</think>` at ids
248066–248069; the Base tokenizer does not. Each checkpoint is prompted with its
own tokenizer and chat template.

## Stages

1. **Measure.** GSM8K (1,319) and MATH-500 (500), chat protocol, zero-shot CoT
   with `\boxed{}`, greedy and maj@16, non-thinking and thinking, for the
   post-trained 0.8B and 2B plus the Base controls. vLLM is pinned to 0.34.0
   (x86_64 wheel `79ca4b3c…`); the recipe flags are `--language-model-only`,
   `--reasoning-parser qwen3`, `--enable-prefix-caching` and the
   `enable_thinking` chat-template flag. Tokens/s is measured per model × mode ×
   decoding. The report table is fixed in the registration.
2. **Distil 2B into post-trained 0.8B.** GSM8K-train + MATH-train prompts,
   student rollouts refreshed every 1,000 steps, per-token reverse KL at T=2.0
   with a verified-answer (0/1) reward gate. Full backbone, tied embedding
   frozen, 498,113,344 trainable and 752,393,024 unique active; corrected
   Adam8bit block 256 stochastic rounding, lr 1e-5, cosine, warmup 200, 20M
   tokens. **Win:** greedy beats post-trained 0.8B on both tasks with paired 95%
   item-level intervals above zero. The S54 stage-2 amendment supersedes the
   supervision and win clauses: **failed** rollouts are included when the 2B
   teacher produces a verified continuation, and supervision starts at the
   first *textual divergence* between the student and the teacher (renamed from
   the S54 wording "first error" by the S56 correction: a token mismatch is not
   a reasoning error); the win also
   requires MMLU and MMLU-Pro to stay within an explicit 1.0-point
   general-capability bound. See
   `docs/ws19-post-trained-route-stage2-amendment-s54-v1.json`.
3. **Loop layers 8–15 R times.** Zero-init re-entry gate and zero-init per-pass
   depth-LoRA, so R=1 is bit-identical to stage 2. The teacher is stage-2 at R=1
   with privileged context = a solution that is both the maj@16 consensus and
   verified against the train label; the student is the looped model on its own
   rollouts; per-token reverse KL. **Win:** greedy R=4 beats stage-2 greedy on a
   paired interval, closes at least two thirds of the stage-2 greedy→maj@16 gap
   on both tasks, and uses at most 0.30 of the stage-2 maj@16 FLOPs per answer.
   The S54 stage-3 amendment makes the accuracy clause the evidence and keeps the
   FLOPs bound as a guard: four greedy passes against sixteen sampled answers is
   near-automatic arithmetic, not a result. See
   `docs/ws19-post-trained-route-stage3-amendment-s54-v1.json`.
4. **Confirm.** Three seeds of the winning recipe, the Rust executed-set audit,
   greedy + maj@16 with FLOPs per answer against post-trained 0.8B under the same
   protocol. No seed may reverse the stage-3 direction.

## Risks registered with their mitigations

Thinking-loop/truncation, teacher–student co-residency, per-pass state leakage,
train/test contamination, formatting-vs-reasoning, and FLOPs accounting are all
named in the registration with a mitigation and a measured report field. In
particular, a `\boxed{}` formatting gain is not a reasoning win, and the R=4
claim requires an executed-set FLOPs ratio, not a parameter-count comparison.

## Not established

No stage has run. The vLLM 0.34.0 recipe page names the 397B-A17B model, so a
stage-1 smoke must confirm the 0.8B/2B chat checkpoints on this host before any
score. **S54 note:** vLLM 0.34.0 is not installed in the offline eval image, so
the stage-1 measurement uses the pinned transformers stack and the receipt
records `engine = transformers`, `vllm_available = false`. A vLLM number is not
claimed.

## S54 pre-run amendments

- Stage 2: verified teacher continuations and first-textual-divergence-suffix
  corrective supervision on failed rollouts, plus a general-capability
  preservation bound in the win
  (`docs/ws19-post-trained-route-stage2-amendment-s54-v1.json`). The S56
  correction renames the quantity and defers the step-local correction until a
  process verifier exists
  (`docs/ws19-post-trained-route-stage2a-amendment-s56-v1.json`).
- Stage 3: the accuracy clause is the evidence; the FLOPs ratio is a guard
  (`docs/ws19-post-trained-route-stage3-amendment-s54-v1.json`).
- The P19.9 screen was repaired before any run: the accuracy and byte controls
  are separated, the ledger is request-based with a cache model, the holdout is
  expanded and the statistics are prompt-level (`docs/S52-REPAIR-S54.md`).
