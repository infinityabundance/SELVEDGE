# WS19 frontier target — verified identity, verified architecture, measured disk

Status: **verification and registration only (S32, 2026-11-16).** No training run
was started, no registered T2 or verdict receipt was modified, and no claim in
this document rests on a model card's marketing sentence. Everything numeric here
is either printed from a receipt between generated sentinels or is an exact
integer quoted from a receipt; `scripts/perf_numbers.py --check` is the court that
enforces that.

The program this document serves is registered in `docs/PROGRAM-PLAN.md` §24
(WS19) and §25 (WS20). Read §24 first if you want the plan; read this if you want
to know whether the plan's premises are true.

## 1. Method and receipts

Tools:

- `scripts/s32_frontier_models.py` — `probe`, `fetch`, `disk`, `ceiling`. Standard
  library only. Run inside the Docker court with an explicit `SELVEDGE_MEM`.
- `scripts/prior_art_fetch.py` — extended with the WS19 name list (rounds 25–28).

Receipts (all committed; each carries `stamp` and the `revision` that produced it):

| receipt | what it records |
| --- | --- |
| `results/frontier-models/20261116-probe-p2.json` | Hub identity, revision, every file's bytes and LFS oid, hashed small files, license file text, and **per-tensor element counts read from each safetensors header** |
| `results/frontier-models/20261116-probe-p1.json` | the first probe pass, kept: it shows the classifier evolving and records the missing-id search |
| `results/frontier-models/20261116-fetch-qwen35-08b.json` | the one downloaded checkpoint, with a SHA-256 verified against the Hub's LFS oid |
| `results/frontier-models/20261116-disk.json` | filesystem facts, cache formula, every cache size at every requested budget |
| `results/frontier-models/20261116-ceiling.json` | always-active vs removable split and layer-type counts per model |
| `results/prior-art/prior-art-fetch-20261116-round25-ws19.jsonl` | first literature pass (Semantic Scholar rate-limited most queries) |
| `...-round26-ws19-arxiv.jsonl`, `...-round27-ws19-names.jsonl`, `...-round28-ws19-venues.jsonl` | the documented arXiv fallbacks, name-shape probes, and venue probes |

Why a header read is exact accounting: a safetensors file starts with an 8-byte
little-endian header length and a JSON header that gives every tensor's `dtype`,
`shape` and `data_offsets`. Element count is the product of the shape; bytes are
elements times dtype width. The probe refuses the shard unless
`data_offsets[1] - data_offsets[0]` equals that product, and it refuses the model
unless the summed bytes plus header plus the 8-byte prefix equal the file size the
Hub declared. The shard totals are then checked against the Hub's own
`safetensors.total`. This is how a parameter count is obtained without downloading
weights. It is *not* the strict tie-aware union audit that `selvedge inspect-model`
performs — §5 is about exactly that gap.

## 2. Existence, revision, license

Every id below was resolved through `https://huggingface.co/api/models/<id>` and
then pinned to the commit SHA the Hub returned; the tree listing was read at that
SHA, not at `main`. Access time is 2026-11-16 UTC and is recorded per model in the
probe receipt.

**`Qwen/Qwen3.5-0.8B`** — exists; chat/instruct; the proposed anchor.

- revision:
  <!-- BEGIN GENERATED: frontier-qwen35-08b-revision -->
2fc06364715b967f1860aea9cf38778875588b17
  <!-- END GENERATED: frontier-qwen35-08b-revision -->
- license: Apache-2.0; the repository's `LICENSE` file was fetched and hashed
  (Apache License 2.0 text, hash in the receipt).
- text-tower elements:
  <!-- BEGIN GENERATED: frontier-qwen35-08b-text-elements -->
752393024 elements
  <!-- END GENERATED: frontier-qwen35-08b-text-elements -->
- vision-tower elements:
  <!-- BEGIN GENERATED: frontier-qwen35-08b-vision-elements -->
100592896 elements
  <!-- END GENERATED: frontier-qwen35-08b-vision-elements -->
- MTP-head elements:
  <!-- BEGIN GENERATED: frontier-qwen35-08b-mtp-elements -->
20452864 elements
  <!-- END GENERATED: frontier-qwen35-08b-mtp-elements -->
- checkpoint elements (all towers):
  <!-- BEGIN GENERATED: frontier-qwen35-08b-total-elements -->
873438784 elements
  <!-- END GENERATED: frontier-qwen35-08b-total-elements -->
- weight bytes (the single safetensors shard):
  <!-- BEGIN GENERATED: frontier-qwen35-08b-weight-bytes -->
1746942600 bytes
  <!-- END GENERATED: frontier-qwen35-08b-weight-bytes -->
- downloaded shard SHA-256 (equals the Hub LFS oid):
  <!-- BEGIN GENERATED: frontier-qwen35-08b-weight-sha256 -->
04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
  <!-- END GENERATED: frontier-qwen35-08b-weight-sha256 -->

**`Qwen/Qwen3.5-2B`** — exists; the proposed upper end of the gap. Its text tower
is already above the ceiling, so it can only ever be a *reference*, never a
target (§6).

- revision:
  <!-- BEGIN GENERATED: frontier-qwen35-2b-revision -->
15852e8c16360a2fea060d615a32b45270f8a8fc
  <!-- END GENERATED: frontier-qwen35-2b-revision -->
- text-tower elements:
  <!-- BEGIN GENERATED: frontier-qwen35-2b-text-elements -->
1881825088 elements
  <!-- END GENERATED: frontier-qwen35-2b-text-elements -->
- checkpoint elements:
  <!-- BEGIN GENERATED: frontier-qwen35-2b-total-elements -->
2274069824 elements
  <!-- END GENERATED: frontier-qwen35-2b-total-elements -->
- weight bytes:
  <!-- BEGIN GENERATED: frontier-qwen35-2b-weight-bytes -->
4548221488 bytes
  <!-- END GENERATED: frontier-qwen35-2b-weight-bytes -->

**`Qwen/Qwen3.5-4B`** — exists; size reference only.

- revision:
  <!-- BEGIN GENERATED: frontier-qwen35-4b-revision -->
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  <!-- END GENERATED: frontier-qwen35-4b-revision -->
- text-tower elements:
  <!-- BEGIN GENERATED: frontier-qwen35-4b-text-elements -->
4205751296 elements
  <!-- END GENERATED: frontier-qwen35-4b-text-elements -->
- checkpoint elements:
  <!-- BEGIN GENERATED: frontier-qwen35-4b-total-elements -->
4659865088 elements
  <!-- END GENERATED: frontier-qwen35-4b-total-elements -->
- weight bytes:
  <!-- BEGIN GENERATED: frontier-qwen35-4b-weight-bytes -->
9319828096 bytes
  <!-- END GENERATED: frontier-qwen35-4b-weight-bytes -->

**`Qwen/Qwen3.5-9B`** — exists; size reference only. Not downloaded: its weight
bytes exceed the cache budget in §7.

- revision:
  <!-- BEGIN GENERATED: frontier-qwen35-9b-revision -->
c202236235762e1c871ad0ccb60c8ee5ba337b9a
  <!-- END GENERATED: frontier-qwen35-9b-revision -->
- text-tower elements:
  <!-- BEGIN GENERATED: frontier-qwen35-9b-text-elements -->
8953803264 elements
  <!-- END GENERATED: frontier-qwen35-9b-text-elements -->
- checkpoint elements:
  <!-- BEGIN GENERATED: frontier-qwen35-9b-total-elements -->
9653104368 elements
  <!-- END GENERATED: frontier-qwen35-9b-total-elements -->
- weight bytes:
  <!-- BEGIN GENERATED: frontier-qwen35-9b-weight-bytes -->
19306310880 bytes
  <!-- END GENERATED: frontier-qwen35-9b-weight-bytes -->

**`Qwen/Qwen3.5-{0.8B,2B,4B,9B}-Base`** — all four exist, Apache-2.0, and carry
the same element counts as their chat counterpart (verified tensor by tensor in
the probe receipt; the base checkpoints differ in values, not in shape). Revisions
are pinned in the receipt; do not reuse the chat SHA for a base checkpoint.

- 0.8B-Base revision:
  <!-- BEGIN GENERATED: frontier-qwen35-08b-base-revision -->
dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
  <!-- END GENERATED: frontier-qwen35-08b-base-revision -->
- 2B-Base revision:
  <!-- BEGIN GENERATED: frontier-qwen35-2b-base-revision -->
b1485b2fa6dfa1287294f269f5fb618e03d52d7c
  <!-- END GENERATED: frontier-qwen35-2b-base-revision -->
- 4B-Base revision:
  <!-- BEGIN GENERATED: frontier-qwen35-4b-base-revision -->
1001bb4d826a52d1f399e183466143f4da7b741b
  <!-- END GENERATED: frontier-qwen35-4b-base-revision -->
- 9B-Base revision:
  <!-- BEGIN GENERATED: frontier-qwen35-9b-base-revision -->
68c46c4b3498877f3ef123c856ecfde50c39f404
  <!-- END GENERATED: frontier-qwen35-9b-base-revision -->

**`LiquidAI/LFM2.5-1.2B`** — **does not exist.** The API returned HTTP 401 with
`{"error":"Invalid username or password."}`, which is what the Hub returns for a
repository that is not there; the search recorded in the receipt
(`/api/models?search=LFM2.5-1.2B`) returns the closest verified alternatives:
`LiquidAI/LFM2.5-1.2B-Instruct`, `-Base`, `-Thinking`, `-JP`, plus quantised and
ONNX variants. No substitution is made silently: the plan's "1.2B stretch model"
means the **-Instruct** checkpoint, whose numbers are below.

**`LiquidAI/LFM2.5-1.2B-Instruct`** — exists; the proposed stretch model.

- revision:
  <!-- BEGIN GENERATED: frontier-lfm-instruct-revision -->
b9d816ab510f9a28dca37cb182e9da8b0226e13c
  <!-- END GENERATED: frontier-lfm-instruct-revision -->
- license: **LFM Open License v1.0** (`license: other` on the Hub; the `LICENSE`
  file was fetched, its first line is `LFM Open License v1.0`, and its hash is in
  the receipt). This is **not** Apache-2.0 and not an OSI licence; the plan must
  treat it as a legal-review item before any weight redistribution.
- text elements (there is no vision tower and no MTP head):
  <!-- BEGIN GENERATED: frontier-lfm-instruct-text-elements -->
1170340608 elements
  <!-- END GENERATED: frontier-lfm-instruct-text-elements -->
- checkpoint elements:
  <!-- BEGIN GENERATED: frontier-lfm-instruct-total-elements -->
1170340608 elements
  <!-- END GENERATED: frontier-lfm-instruct-total-elements -->
- weight bytes:
  <!-- BEGIN GENERATED: frontier-lfm-instruct-weight-bytes -->
2340697936 bytes
  <!-- END GENERATED: frontier-lfm-instruct-weight-bytes -->

`LiquidAI/LFM2.5-1.2B-Base` exists at the revision in the receipt with identical
element counts; licence identical.

**`google/gemma-3-1b-it`** — exists but is **gated (`manual`)**: the weight shard
returns HTTP 401 to an anonymous range request, so no header accounting and no
LICENSE file hash could be produced. What is verified is the revision and the
Hub's own parameter total; what is *not* verified is the tensor-level count, and
the licence is the Gemma Terms of Use (`license: gemma`), not an open licence. The
plan therefore treats it as a **reference only** and does not build on it.

- revision:
  <!-- BEGIN GENERATED: frontier-gemma-revision -->
dcc83ea841ab6100d6b47a070329e1ba4cf78752
  <!-- END GENERATED: frontier-gemma-revision -->
- Hub-reported parameter total (not header-verified; see the text):
  <!-- BEGIN GENERATED: frontier-gemma-hub-total-elements -->
999885952 elements
  <!-- END GENERATED: frontier-gemma-hub-total-elements -->

### What the identity check changed about the plan

1. The anchor is **not a 752M checkpoint**; it is an 873M multimodal checkpoint
   whose *text tower* is 752,393,024 elements (§4).
2. `LFM2.5-1.2B` does not exist; the stretch model is `-Instruct`, under a
   non-Apache licence.
3. `gemma-3-1b-it` is gated and cannot be audited without accepting Google's
   terms; it stays a reference.
4. The 2B reference's text tower is **above** the ceiling, so the "0.8B→2B gap"
   is a gap in capability, not in compliant parameters — the plan must not imply
   otherwise.

## 3. Architecture facts for Qwen3.5-0.8B — vendor, config, header

Three independent sources were read: the vendor `README.md` at the pinned
revision, the pinned `config.json`, and the tensor names and shapes in the
safetensors header. `VERIFIED` below means all three agree; where they do not,
this document says so.

| claim | vendor README | `config.json` | header | verdict |
| --- | --- | --- | --- | --- |
| text-layer count | "Number of Layers" given | `num_hidden_layers` present | layer prefixes agree | **VERIFIED** |
| layout: Gated DeltaNet + periodic attention | "6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))" | `layer_types`: three linear then one full, repeated | `linear_attn.*` and `self_attn.*` on disjoint layer sets | **VERIFIED** |
| which layers are attention | implied by the layout | `layer_types` marks full attention every `full_attention_interval` layers | the same six layers carry `q_proj`/`k_proj`/`v_proj`; the other eighteen carry `linear_attn.*` | **VERIFIED** (indices 3, 7, 11, 15, 19, 23) |
| hidden width | "Hidden Dimension" given | `hidden_size` present | norm shapes agree | **VERIFIED** |
| vocabulary | "Token Embedding: 248320 (Padded)" | `vocab_size` present | `embed_tokens` rows agree | **VERIFIED** |
| tied embeddings | "LM Output: 248320 (Tied to token embedding)" | `tie_word_embeddings: true` | **no `lm_head.weight` tensor exists** | **VERIFIED** |
| MTP head | "MTP: trained with multi-steps" | `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings: false` | `mtp.fc`, `mtp.layers.0.*`, `mtp.norm`, `mtp.pre_fc_norm_*` present | **VERIFIED** |
| DeltaNet heads / dims | 16 V heads, 16 QK heads, dim 128 | `linear_num_key_heads`, `linear_key_head_dim` | `in_proj_qkv` is `[6144, 1024]` = 16×3×128 | **VERIFIED** |
| attention heads / dims | 8 Q heads, 2 KV heads, dim 256, RoPE dim 64 | `num_attention_heads`, `num_key_value_heads`, `head_dim`, `partial_rotary_factor` | `q_proj` is `[4096, 1024]` (the output gate is fused), `k_proj` is `[512, 1024]` | **VERIFIED** |
| FFN | intermediate 3584 | `intermediate_size` | `gate_proj`/`up_proj` are `[3584, 1024]` | **VERIFIED** |
| "sparse Mixture-of-Experts" | claimed in the README highlights | **no expert tensors; dense `mlp.*` on every layer; `mlp_only_layers: []`** | no router, no `experts.*` | **CONTRADICTED for this size** — the sentence is inherited from the larger Qwen3.5 mixture models and is false for 0.8B |
| vision tower | "Causal Language Model with Vision Encoder" | `vision_config` present | `model.visual.*` present | **VERIFIED** (and it is why the checkpoint total differs from the text tower) |
| "Number of Parameters" | "0.8B" | — | text tower is smaller than the checkpoint | the vendor page states **no** 752M figure; the 0.8B label is the checkpoint, not the text path |

The exact config values, quoted from the pinned `config.json` as recorded in the
probe receipt:

- text layers:
  <!-- BEGIN GENERATED: frontier-arch-layers -->
24
  <!-- END GENERATED: frontier-arch-layers -->
- full-attention interval (one attention layer every N text layers):
  <!-- BEGIN GENERATED: frontier-arch-attn-interval -->
4
  <!-- END GENERATED: frontier-arch-attn-interval -->
- hidden width:
  <!-- BEGIN GENERATED: frontier-arch-hidden -->
1024
  <!-- END GENERATED: frontier-arch-hidden -->
- vocabulary:
  <!-- BEGIN GENERATED: frontier-arch-vocab -->
248320
  <!-- END GENERATED: frontier-arch-vocab -->
- MTP layers:
  <!-- BEGIN GENERATED: frontier-arch-mtp-layers -->
1
  <!-- END GENERATED: frontier-arch-mtp-layers -->
- DeltaNet key/value heads:
  <!-- BEGIN GENERATED: frontier-arch-lin-heads -->
16
  <!-- END GENERATED: frontier-arch-lin-heads -->
- DeltaNet key head dimension:
  <!-- BEGIN GENERATED: frontier-arch-lin-kdim -->
128
  <!-- END GENERATED: frontier-arch-lin-kdim -->
- attention heads:
  <!-- BEGIN GENERATED: frontier-arch-qheads -->
8
  <!-- END GENERATED: frontier-arch-qheads -->
- attention KV heads:
  <!-- BEGIN GENERATED: frontier-arch-kvheads -->
2
  <!-- END GENERATED: frontier-arch-kvheads -->
- attention head dimension:
  <!-- BEGIN GENERATED: frontier-arch-head-dim -->
256
  <!-- END GENERATED: frontier-arch-head-dim -->
- FFN intermediate width:
  <!-- BEGIN GENERATED: frontier-arch-intermediate -->
3584
  <!-- END GENERATED: frontier-arch-intermediate -->

Consequence for the runtime, stated as a fact rather than a risk in the abstract:
the executed text path is **not** a standard transformer. Eighteen of its
twenty-four layers are Gated DeltaNet layers (short convolution with kernel 4, a
gated delta-rule state update, `A_log`/`dt_bias`), six are gated softmax attention
with RoPE applied to a quarter of each head, and the attention path has mRoPE
sections. The repository's runtime implements softmax attention, RoPE, RMSNorm and
SwiGLU only (`crates/selvedge-runtime/src/kernels.rs`); there is no delta-rule
scan, no convolution state, no mRoPE and no output gate. That is registered as a
blocking prerequisite (P19.0) and not hand-waved.

## 4. Is the claimed ~752M correct?

<!-- BEGIN GENERATED: frontier-verdict-text-elements -->
752393024 elements in the text tower
<!-- END GENERATED: frontier-verdict-text-elements -->

<!-- BEGIN GENERATED: frontier-verdict-total-elements -->
873438784 elements in the whole checkpoint
<!-- END GENERATED: frontier-verdict-total-elements -->

**Verdict.** The 752M figure is **correct as the text-tower count** and **wrong as
the model's parameter count**. The numbers the audit will use are printed above:
the text tower is the first generated number in this section and the whole
checkpoint is the second. The checkpoint is multimodal: the vision encoder is a
real part of it and the MTP head is a real trained part of it. The plan may quote
the text-tower count only with the words "text tower", and must state the
checkpoint total alongside it. The vendor page prints neither figure: it says
"0.8B". Any document that presents the text-tower count as the vendor's stated
size is inventing a source.

## 5. `selvedge inspect-model` refuses this checkpoint

The mandate asks for the strict tie-aware unique-active count from the repository's
own auditor. It was run, in the rust court, on the downloaded checkpoint:

```
$ scripts/court rust -- cargo run --release -p selvedge-cli -- \
    inspect-model /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 --json
{"schema":"selvedge/inspect-model-error/v1","ok":false,
 "error":"FFN tensor map is inconsistent: .../config.json is not a readable Qwen3 config: missing field `hidden_size` at line 96 column 1"}
```

It failed closed, which is the correct behaviour: the auditor supports
`model_type == "qwen3"` and reads top-level `hidden_size`; Qwen3.5 nests its
language config under `text_config` and its tensors under `model.language_model.*`,
and it also carries a vision tower and an MTP head the auditor has no component for.
Court receipts: `results/courts/rust-20261116T181516Z-5237e7.json` and its log.

What this document can therefore claim: **exact element counts by component, with
the tie verified by the absence of `lm_head.weight` plus `tie_word_embeddings:
true`** — but not the auditor's own union traversal, and not a per-token executed
set. The plan must not describe 752,393,024 as "the audited unique-active count"
until P19.0 lands. The difference matters: the auditor is what proves an executed
set equals a declared set, and an element sum is not that proof.

## 6. Ceiling arithmetic for the registered 1e9 rule

The rule is unchanged: a token's audit must show fewer than one billion
unique-active elements. For a dense model that is the elements the token's
execution touches, ties counted once.

Qwen3.5-0.8B, text path:

- text-tower elements:
  <!-- BEGIN GENERATED: ceiling-08b-text -->
752393024 elements
  <!-- END GENERATED: ceiling-08b-text -->
- of which FFN (removable in a sharded design):
  <!-- BEGIN GENERATED: ceiling-08b-ffn -->
264241152 elements
  <!-- END GENERATED: ceiling-08b-ffn -->
- always-active backbone:
  <!-- BEGIN GENERATED: ceiling-08b-always-active -->
488151872 elements
  <!-- END GENERATED: ceiling-08b-always-active -->
- headroom under the ceiling, text only:
  <!-- BEGIN GENERATED: ceiling-08b-headroom -->
247606976 elements
  <!-- END GENERATED: ceiling-08b-headroom -->
- MTP head (executed only in multi-token drafting):
  <!-- BEGIN GENERATED: ceiling-08b-mtp -->
20452864 elements
  <!-- END GENERATED: ceiling-08b-mtp -->
- per-token elements if MTP is executed:
  <!-- BEGIN GENERATED: ceiling-08b-text-plus-mtp -->
772845888 elements
  <!-- END GENERATED: ceiling-08b-text-plus-mtp -->
- vision tower (never executed by a text-only runtime):
  <!-- BEGIN GENERATED: ceiling-08b-vision -->
100592896 elements
  <!-- END GENERATED: ceiling-08b-vision -->

Qwen3.5-2B (reference, not a target): text tower

  <!-- BEGIN GENERATED: ceiling-2b-text -->
1881825088 elements
  <!-- END GENERATED: ceiling-2b-text -->
,
always-active
  <!-- BEGIN GENERATED: ceiling-2b-always-active -->
975855424 elements
  <!-- END GENERATED: ceiling-2b-always-active -->
.
The text tower alone is over the ceiling, which is why it can only be the
capability reference.

LFM2.5-1.2B-Instruct (stretch): text

  <!-- BEGIN GENERATED: ceiling-lfm-text -->
1170340608 elements
  <!-- END GENERATED: ceiling-lfm-text -->
,
always-active
  <!-- BEGIN GENERATED: ceiling-lfm-always-active -->
365034240 elements
  <!-- END GENERATED: ceiling-lfm-always-active -->
,
FFN
  <!-- BEGIN GENERATED: ceiling-lfm-ffn -->
805306368 elements
  <!-- END GENERATED: ceiling-lfm-ffn -->
.
Its backbone fits the ceiling with room to spare; its dense total does not.

Reading: the anchor leaves a quarter of a billion elements of headroom for grafted
memory before the ceiling binds, and the LFM stretch model leaves substantially
more. Neither number is a promise that memory will help; it is only the arithmetic
the audit will hold the program to. The arithmetic must be re-run **per token**
with the ledger once an executor exists, because MTP and vision change the
executed set, not just the total.

## 7. Disk arithmetic

The cache formula for a teacher top-k logit cache is
`n_tokens × (k × (4 + 2) + 4)` bytes: one u32 token id and one bf16 logit per
entry, plus a per-token header. It is the plan's formula, applied without rounding.

Two filesystems matter and they are different: the repository filesystem, where
caches and derived bundles land, and the Hugging Face cache filesystem, where
checkpoints land.

- repository free bytes at the time of measurement:
  
    <!-- BEGIN GENERATED: disk-repo-free-bytes -->
96132562944 bytes
    <!-- END GENERATED: disk-repo-free-bytes -->
- repository headroom reserved by the plan's own rule:
  
    <!-- BEGIN GENERATED: disk-repo-headroom-bytes -->
21474836480 bytes
    <!-- END GENERATED: disk-repo-headroom-bytes -->
- cache filesystem free bytes:
  
    <!-- BEGIN GENERATED: disk-cache-free-bytes -->
22066839552 bytes
    <!-- END GENERATED: disk-cache-free-bytes -->
- cache filesystem budget after headroom (this is the download budget):
  
    <!-- BEGIN GENERATED: disk-cache-budget-bytes -->
19919355904 bytes
    <!-- END GENERATED: disk-cache-budget-bytes -->

### Teacher top-k logit caches

- 0.5B tokens at k = 64:
  
    <!-- BEGIN GENERATED: disk-cache-500m-k64 -->
194000000000 bytes
    <!-- END GENERATED: disk-cache-500m-k64 -->
- 0.5B tokens at k = 32:
  
    <!-- BEGIN GENERATED: disk-cache-500m-k32 -->
98000000000 bytes
    <!-- END GENERATED: disk-cache-500m-k32 -->
- 0.25B tokens at k = 64:
  
    <!-- BEGIN GENERATED: disk-cache-250m-k64 -->
97000000000 bytes
    <!-- END GENERATED: disk-cache-250m-k64 -->
- 0.25B tokens at k = 32:
  
    <!-- BEGIN GENERATED: disk-cache-250m-k32 -->
49000000000 bytes
    <!-- END GENERATED: disk-cache-250m-k32 -->
- 0.1B tokens at k = 64:
  
    <!-- BEGIN GENERATED: disk-cache-100m-k64 -->
38800000000 bytes
    <!-- END GENERATED: disk-cache-100m-k64 -->
- 0.1B tokens at k = 32:
  
    <!-- BEGIN GENERATED: disk-cache-100m-k32 -->
19600000000 bytes
    <!-- END GENERATED: disk-cache-100m-k32 -->

The plan as proposed budgeted "190 GB" for teacher top-k logits. That number is
reproduced here as a quotation of the proposal, not as a measurement. Against the
free bytes printed above, only the two smaller configurations fit with the
reserved headroom; the proposed budget does not, and the difference is not a
rounding error — it is a factor of roughly two and a half. The largest
configuration that does fit, with headroom, is:

- largest fitting cache bytes:
  <!-- BEGIN GENERATED: disk-largest-fitting -->
49000000000 bytes
  <!-- END GENERATED: disk-largest-fitting -->
- largest fitting k = 64 cache bytes:
  <!-- BEGIN GENERATED: disk-largest-fitting-k64 -->
38800000000 bytes
  <!-- END GENERATED: disk-largest-fitting-k64 -->

### Checkpoints and derived bundles

Checkpoint bytes are printed per model in §2. What fits on the cache filesystem
with headroom:

| what | fits | why |
| --- | --- | --- |
| Qwen3.5-0.8B (the anchor) | **yes** | downloaded, hash-verified |
| Qwen3.5-2B (the gap reference) | **yes** | needed for the scoreboard's upper end |
| LFM2.5-1.2B-Instruct (stretch) | **yes** | fetched only when the stretch stage opens |
| Qwen3.5-4B, Qwen3.5-9B | **no** | reference-only; their headers were read, their weights are not needed |
| gemma-3-1b-it | **no** | gated; cannot be fetched anonymously at all |

A derived bundle at bf16 costs the checkpoint's elements times two, i.e. the same
magnitude as the checkpoint itself; a `docs/STORAGE.md` bundle adds its index and
digest overhead. This is why the plan budgets one derived bundle for the anchor,
not one per stage.

### Recommendation

Adopt **0.25B tokens at k = 32** as the teacher-cache budget for the first claim
attempt:

- it is the largest configuration that fits with the reserved headroom;
- k = 32 at 0.25B tokens costs less than k = 64 at 0.1B tokens *and* buys two and
  a half times the token coverage, and the on-policy distillation stage spends
  teacher scores per student token, so coverage is the scarcer resource;
- k = 64 remains available at 0.1B tokens as the fidelity arm if the pilot shows
  the k = 32 truncation is binding.

Honest caveat: neither configuration is a pretraining corpus. 0.25B tokens is a
distillation corpus for a 752M text tower, not a Chinchilla-comparable budget, and
the plan must not describe the resulting model as "trained" as if it were
pretrained. Re-derive this table before every stage; the filesystem free bytes are
a measurement, not a constant.

## 8. What this document does not establish

- It does not establish quality for any of these checkpoints. No capability number
  was produced; no harness was run.
- It does not establish that the LFM licence permits the proposed use. That is
  legal review, and the licence is not Apache-2.0.
- It does not reproduce the vendor's benchmark table. Those numbers are the
  vendor's claims and are not in any receipt here.
- It does not prove the absence of prior art for the per-pass memory idea;
  `docs/PRIOR-ART-AND-NOVELTY.md` §16–§17 records what was searched and what the
  closest matches are.
- It does not close the `inspect-model` gap. Until P19.0 lands, "unique active" in
  this document means "elements in the executed component", never "the auditor's
  audited union".
