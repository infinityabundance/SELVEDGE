# Budget math on real Qwen3 checkpoints

Status: **measured on the real checkpoints** with `selvedge inspect-model`, and
published only from receipts. The question this document answers is narrow and
checkable: under the strict `unique active learned parameters per output token
<= 1e9` rule, is the ceiling actually binding for the candidate teachers?

Receipts:

- `results/accounting/20261114T201533Z-qwen3-0.6b.json`
- `results/accounting/20261114T201533Z-qwen3-1.7b.json`
- `results/accounting/20261114T205700Z-qwen3-4b-config-projection.json`
- disk: `results/disk/disk-20261114T205532Z.json`
- model acquisition: `results/models/fetch-20261114T201604Z.json`

Every number below is emitted by `scripts/perf_numbers.py` from one of those
files. Nothing here is copied from a model card.

## Counting conventions

These conventions are part of the result; a different convention gives a
different answer.

1. **Unique, not total, not reads.** The ledger counts each learned element at
   most once per token, even when it is read more than once.
2. **The embedding/output tie is verified, not trusted.** `config.json` declares
   `tie_word_embeddings`, but the accounting also compares
   `model.embed_tokens.weight` and `lm_head.weight` elementwise after exact bf16
   widening. The tie is applied only when the bytes are identical. Both Qwen3
   checkpoints pass that check, so the matrix is counted once.
3. **Embedding and output projection are whole matrices.** A token performs one
   row lookup and one full-vocabulary projection; across a token stream the
   whole table is in use, so counting one row would understate the learned
   substrate. Both are counted in full, and a tied pair once.
4. **Always-active backbone** = embedding + all attention projections + q/k
   RMSNorm weights + layer/final RMSNorm weights + the output projection as the
   model actually executes it (shared with the embedding when tied).
5. **Removable FFN** = every gate/up/down element in every block. This is the
   only part a group selector can omit.
6. **Metadata is not learned.** Quantization scales, if any, would be reported
   as auxiliary bytes. These checkpoints are dense bf16, so there are none.

`selvedge inspect-model` reads safetensors headers for element counts, rejects
any tensor it does not classify, and checks every shape against `config.json`.
A missing, extra, or mis-shaped tensor is a hard error.

The ceiling itself:

<!-- BEGIN GENERATED: bm-ceiling -->
1000000000
<!-- END GENERATED: bm-ceiling -->

## Qwen3-0.6B

- checkpoint tensor elements (tie counted twice):
  <!-- BEGIN GENERATED: bm-06b-checkpoint-elements -->
751632384
  <!-- END GENERATED: bm-06b-checkpoint-elements -->
- unique learned elements (tie-aware):
  <!-- BEGIN GENERATED: bm-06b-unique -->
596049920
  <!-- END GENERATED: bm-06b-unique -->
- embedding:
  <!-- BEGIN GENERATED: bm-06b-embedding -->
155582464
  <!-- END GENERATED: bm-06b-embedding -->
- attention (projections plus q/k norms):
  <!-- BEGIN GENERATED: bm-06b-attention -->
176167936
  <!-- END GENERATED: bm-06b-attention -->
- layer and final norms:
  <!-- BEGIN GENERATED: bm-06b-norms -->
58368
  <!-- END GENERATED: bm-06b-norms -->
- output projection (tied to the embedding, counted once):
  <!-- BEGIN GENERATED: bm-06b-lm-head -->
155582464
  <!-- END GENERATED: bm-06b-lm-head -->
- removable FFN:
  <!-- BEGIN GENERATED: bm-06b-ffn -->
264241152
  <!-- END GENERATED: bm-06b-ffn -->
- always-active backbone:
  <!-- BEGIN GENERATED: bm-06b-backbone -->
331808768
  <!-- END GENERATED: bm-06b-backbone -->
- headroom under the ceiling:
  <!-- BEGIN GENERATED: bm-06b-headroom -->
403950080
  <!-- END GENERATED: bm-06b-headroom -->
- FFN elements affordable with every layer running:
  <!-- BEGIN GENERATED: bm-06b-max-ffn -->
668191232
  <!-- END GENERATED: bm-06b-max-ffn -->
- affordable FFN share:
  <!-- BEGIN GENERATED: bm-06b-max-ffn-fraction -->
1.0
  <!-- END GENERATED: bm-06b-max-ffn-fraction -->

Verdict:

<!-- BEGIN GENERATED: bm-06b-verdict -->
not_binding
<!-- END GENERATED: bm-06b-verdict -->

The full dense model is under the ceiling. The ceiling does not force sparsity
on this teacher; any sparsity here is a cost/quality choice. It is therefore a
control, not a compliance experiment.

## Qwen3-1.7B

- checkpoint tensor elements (tie counted twice):
  <!-- BEGIN GENERATED: bm-17b-checkpoint-elements -->
2031739904
  <!-- END GENERATED: bm-17b-checkpoint-elements -->
- unique learned elements (tie-aware):
  <!-- BEGIN GENERATED: bm-17b-unique -->
1720574976
  <!-- END GENERATED: bm-17b-unique -->
- embedding:
  <!-- BEGIN GENERATED: bm-17b-embedding -->
311164928
  <!-- END GENERATED: bm-17b-embedding -->
- attention (projections plus q/k norms):
  <!-- BEGIN GENERATED: bm-17b-attention -->
352328704
  <!-- END GENERATED: bm-17b-attention -->
- layer and final norms:
  <!-- BEGIN GENERATED: bm-17b-norms -->
116736
  <!-- END GENERATED: bm-17b-norms -->
- output projection (tied to the embedding, counted once):
  <!-- BEGIN GENERATED: bm-17b-lm-head -->
311164928
  <!-- END GENERATED: bm-17b-lm-head -->
- removable FFN:
  <!-- BEGIN GENERATED: bm-17b-ffn -->
1056964608
  <!-- END GENERATED: bm-17b-ffn -->
- always-active backbone:
  <!-- BEGIN GENERATED: bm-17b-backbone -->
663610368
  <!-- END GENERATED: bm-17b-backbone -->
- headroom under the ceiling:
  <!-- BEGIN GENERATED: bm-17b-headroom -->
-720574976
  <!-- END GENERATED: bm-17b-headroom -->
- FFN elements affordable with every layer running:
  <!-- BEGIN GENERATED: bm-17b-max-ffn -->
336389632
  <!-- END GENERATED: bm-17b-max-ffn -->
- affordable FFN share:
  <!-- BEGIN GENERATED: bm-17b-max-ffn-fraction -->
0.31826007176959326
  <!-- END GENERATED: bm-17b-max-ffn-fraction -->
- affordable whole channels per layer:
  <!-- BEGIN GENERATED: bm-17b-channels-per-layer -->
1955
  <!-- END GENERATED: bm-17b-channels-per-layer -->
- intermediate channels per layer:
  <!-- BEGIN GENERATED: bm-17b-intermediate-per-layer -->
6144
  <!-- END GENERATED: bm-17b-intermediate-per-layer -->
- affordable default groups per layer:
  <!-- BEGIN GENERATED: bm-17b-whole-groups -->
5
  <!-- END GENERATED: bm-17b-whole-groups -->
- full-FFN layers affordable if others are skipped:
  <!-- BEGIN GENERATED: bm-17b-full-ffn-layers -->
8.91128200954861
  <!-- END GENERATED: bm-17b-full-ffn-layers -->

Verdict:

<!-- BEGIN GENERATED: bm-17b-verdict -->
binding_backbone_fits
<!-- END GENERATED: bm-17b-verdict -->

The dense model is over the ceiling while the always-on backbone fits. The
ceiling is binding: a compliant token can execute only the affordable FFN share.
This is the teacher that makes the 1e9 constraint an actual design constraint
rather than a slogan.

## Arithmetic estimates per token

FLOPs below are formula estimates, not measurements. The formulas are in
`docs/ARCHITECTURE.md` and are repeated in the receipt itself. The receiving
runtime, not this document, is responsible for measured throughput.

Qwen3-0.6B:

- always-on backbone FLOPs:
  <!-- BEGIN GENERATED: bm-06b-flops-backbone -->
664064000
  <!-- END GENERATED: bm-06b-flops-backbone -->
- full FFN FLOPs, all layers:
  <!-- BEGIN GENERATED: bm-06b-flops-ffn -->
528998400
  <!-- END GENERATED: bm-06b-flops-ffn -->
- dense total at context 1:
  <!-- BEGIN GENERATED: bm-06b-flops-dense -->
1193294016
  <!-- END GENERATED: bm-06b-flops-dense -->
- one default group's FFN matmul:
  <!-- BEGIN GENERATED: bm-06b-group-flops -->
1179648
  <!-- END GENERATED: bm-06b-group-flops -->
- parameters in one default group:
  <!-- BEGIN GENERATED: bm-06b-group-elements -->
589824
  <!-- END GENERATED: bm-06b-group-elements -->

Qwen3-1.7B:

- always-on backbone FLOPs:
  <!-- BEGIN GENERATED: bm-17b-flops-backbone -->
1327783936
  <!-- END GENERATED: bm-17b-flops-backbone -->
- full FFN FLOPs, all layers:
  <!-- BEGIN GENERATED: bm-17b-flops-ffn -->
2114961408
  <!-- END GENERATED: bm-17b-flops-ffn -->
- dense total at context 1:
  <!-- BEGIN GENERATED: bm-17b-flops-dense -->
3442976960
  <!-- END GENERATED: bm-17b-flops-dense -->
- one default group's FFN matmul:
  <!-- BEGIN GENERATED: bm-17b-group-flops -->
4718592
  <!-- END GENERATED: bm-17b-group-flops -->
- parameters in one default group:
  <!-- BEGIN GENERATED: bm-17b-group-elements -->
2359296
  <!-- END GENERATED: bm-17b-group-elements -->

## Is Qwen3-4B needed? No.

Its public `config.json` (pinned revision in `models/model-spec.json`) was
downloaded and accounted without downloading weights
(`scripts/courts/config_projection.py`). Its always-on backbone alone is:

- backbone:
  <!-- BEGIN GENERATED: bm-4b-backbone -->
1332870656
  <!-- END GENERATED: bm-4b-backbone -->
- unique dense total:
  <!-- BEGIN GENERATED: bm-4b-unique -->
4022468096
  <!-- END GENERATED: bm-4b-unique -->
- removable FFN:
  <!-- BEGIN GENERATED: bm-4b-ffn -->
2689597440
  <!-- END GENERATED: bm-4b-ffn -->
- verdict:

<!-- BEGIN GENERATED: bm-4b-verdict -->
impossible_backbone_over_ceiling
<!-- END GENERATED: bm-4b-verdict -->

The backbone is above 1e9, so no group selection can make Qwen3-4B compliant.
Downloading its weights would not answer a compliance question. It is recorded
as a reference model that is deliberately not downloaded. The binding teacher
in the Qwen3 family is Qwen3-1.7B: its backbone fits and its FFN does not.

## Disk budget

Measured inside the courts, not assumed:

- repository filesystem free bytes:
  <!-- BEGIN GENERATED: bm-disk-repo-free -->
77037703168
  <!-- END GENERATED: bm-disk-repo-free -->
- model-cache filesystem free bytes:
  <!-- BEGIN GENERATED: bm-disk-cache-free -->
22788182016
  <!-- END GENERATED: bm-disk-cache-free -->

The f32 FFN bundles are stored under the gitignored `results/artifacts/`
directory. Their sizes are in `results/REPORT.md` generated from the real-parity
receipt.

## Reproduce

```bash
scripts/build-courts.sh oracle
scripts/fetch-models.sh
scripts/court rust -- cargo run --release -p selvedge-cli -- inspect-model \
  /hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca --json
scripts/court rust -- cargo run --release -p selvedge-cli -- inspect-model \
  /hf-cache/models/Qwen/Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --json
scripts/court python -- python3 scripts/courts/config_projection.py \
  --config models/reference-configs/qwen3-4b-config.json \
  --model-id Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --out results/accounting/<stamp>-qwen3-4b-config-projection.json \
  --stamp <stamp> --revision-git <git-sha>
```

## Limitations

- Element counts are exact for the files on disk. They do not prove that the
  architecture executes every tensor; the teacher forward is the evidence for
  that, in `results/REPORT.md`.
- FLOPs are arithmetic estimates from documented formulas. Attention-core FLOPs
  depend on context length and are reported separately in the receipt.
- The embedding whole-matrix convention is a choice. If a deployment counted
  only the touched row, the 0.6B and 1.7B headroom would be larger; the mandate
  explicitly requires counting the output projection, and the embedding is the
  same matrix when tied.
- Qwen3-4B is a config projection, not a checkpoint measurement. It is labelled
  as such in its receipt.
