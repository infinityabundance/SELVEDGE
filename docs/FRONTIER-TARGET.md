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

**`Qwen/Qwen3.5-0.8B`** — exists; chat/instruct; **the chat revision, not the
anchor** (S38). The WS19 anchor is `Qwen/Qwen3.5-0.8B-Base`, which is the
checkpoint the scoreboard scores, the pre-registration names and the lock file
pins. This revision is retained here because S32's probe, S33's reference parity
receipt and the first two ceiling certificates were produced on it; §10 measures
what does and does not transfer between the two.

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

**The `-Base` 0.8B checkpoint is the WS19 anchor.** `Qwen/Qwen3.5-0.8B` is a chat
revision of the same graph and is **not** the anchor: it has different weight
bytes and four additional special tokens (§10). Every anchor command in this
repository names the `-Base` revision.

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

Two housekeeping facts, recorded so nobody looks for them later: the anchor has
**no `generation_config.json`** (its sampling guidance lives in the model card, so
a harness must pin sampling explicitly rather than inherit it), and none of these
checkpoints is in `models/model-spec.json` yet. That file pins the **teacher**
candidates the oracle lock verifies; adding a frontier checkpoint there before
the kernel half of P19.0 can execute it would create a lock entry nothing in the
serving path can run.

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
prerequisite for the **serving** stages (P19.0, kernel half) and not hand-waved;
stages 1–4 execute the validated PyTorch reference instead, so the quality
question does not wait on a Rust kernel.

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

## 5. `selvedge inspect-model` accepts this checkpoint (S33; re-issued for the
`-Base` anchor at S38)

The certificate below was produced on `Qwen/Qwen3.5-0.8B` (chat) at S33, before
the scoreboard existed. S38 re-ran the same auditor against the **`-Base`**
checkpoint that the scoreboard actually scores, and the `-Base` receipt is now
the anchor certificate; the chat receipts are kept because they are what S33
tested and because §10 uses them as the comparison.

S32 recorded that the auditor refused the anchor. It no longer does. S33 added
accounting **schema v2**, which is a second auditor rather than a patch to v1:
v1 refuses a config it does not understand and stays exactly as it was, and the
CLI dispatches on `model_type` read from `config.json`. There is no fallback, so
an unknown type still fails closed.

The refusal as S32 recorded it (kept, because it is why v2 exists):

```
$ scripts/court rust -- cargo run --release -p selvedge-cli -- \
    inspect-model /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 --json
{"schema":"selvedge/inspect-model-error/v1","ok":false,
 "error":"FFN tensor map is inconsistent: .../config.json is not a readable Qwen3 config: missing field `hidden_size` at line 96 column 1"}
```

What v2 does with the same bytes, in the rust court, offline, with the model
cache mounted read-only:

```
$ scripts/court rust -- cargo run --release -p selvedge-cli -- inspect-model \
    /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --json --active-path text-only --certify-ceiling 1000000000 \
    --expect-totals text=752393024,vision=100592896,mtp=20452864,checkpoint=873438784
schema=selvedge/model-accounting/v2  ok=true  verdict=certified
```

Receipts: `results/accounting/20261116T183609Z-qwen35-08b-text-only.json`,
`...-text-and-mtp.json` and `...-refused-untied.json`; court receipts
`results/courts/rust-20261116T183609Z-9be033.json` and
`results/courts/rust-20261116T183625Z-e0e559.json`.

The four counts S32 derived from the header are reproduced **exactly**, and each
is now also a checked expectation that fails the run if it drifts:

text tower, unique and tie-aware

<!-- BEGIN GENERATED: qwen35-text-unique -->
752393024
<!-- END GENERATED: qwen35-text-unique -->

vision tower

<!-- BEGIN GENERATED: qwen35-vision-elements -->
100592896
<!-- END GENERATED: qwen35-vision-elements -->

MTP head

<!-- BEGIN GENERATED: qwen35-mtp-elements -->
20452864
<!-- END GENERATED: qwen35-mtp-elements -->

whole checkpoint, elements

<!-- BEGIN GENERATED: qwen35-checkpoint-elements -->
873438784
<!-- END GENERATED: qwen35-checkpoint-elements -->

whole checkpoint, dtype-aware bytes

<!-- BEGIN GENERATED: qwen35-checkpoint-bytes -->
1746882752
<!-- END GENERATED: qwen35-checkpoint-bytes -->

the Hub's `model.safetensors.index.json` `metadata.total_size`

<!-- BEGIN GENERATED: qwen35-hub-total-bytes -->
1746882752
<!-- END GENERATED: qwen35-hub-total-bytes -->

Those last two agreeing is the external check S32 could not make. The byte total
is dtype-aware — 452 bf16 tensors plus 36 f32 tensors (every `A_log` and every
`linear_attn.norm.weight`, 2,592 elements at four bytes) — and only that
arithmetic reproduces the Hub's number to the byte. An element count alone cannot
distinguish 1,746,877,568 from 1,746,882,752, which is why the receipt carries
both.

**The tie is verified, not assumed.** There is no `lm_head.weight` tensor in this
checkpoint. The receipt records `treated_as_shared: true` for a reason that is
narrower than "the bytes match": it records *tie by absence plus config*, and
`compared_elements: 0`. A v1-style byte comparison is not available here because
there is no second matrix to compare. Declaring the pair untied is refused on the
real checkpoint — `--assume-untied` exits 1 with

```
the caller declared the weights untied, but no lm_head.weight exists; the only
output projection is the embedding
```

**What is declared and checked.** `--active-path` makes the execution path an
input to the audit instead of a sentence in a plan:

- every tensor must belong to exactly one tower (`text`, `vision`, `mtp`), and an
  unclassified tensor is a hard error;
- the declared towers must exist in the bytes — declaring `text-and-mtp` on a
  checkpoint without MTP weights is refused, with the arithmetic still printed;
- the declared text-path key set is checked for **namespace disjointness** against
  the excluded towers (computed, not assumed), so a text tensor living under
  `model.visual.` or `mtp.` would make the declaration a lie and is refused;
- cross-tower reads are recorded: `mtp_use_dedicated_embeddings` is false, so the
  MTP head reads the language model's embedding, and the receipt says so instead
  of double-counting it;
- `--certify-ceiling N` prints the component arithmetic and a verdict, and sets
  the exit code.

**What this does not establish.** It is still not a per-token *executed-set*
proof. The auditor proves that a declared set of weights is internally consistent
and that its union is under the ceiling; it does not watch a kernel read a
pointer. That is what Court E is for, and it needs the kernels registered in
`docs/RUNTIME.md`.

### 5.1 The `-Base` certificate (S38)

The audited checkpoint is now the benchmarked checkpoint. Command, in the rust
court, offline, with the model cache mounted read-only:

```
$ scripts/court rust -- cargo run --release -p selvedge-cli -- inspect-model \
    /hf-cache/models/Qwen/Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --json --active-path text-only --certify-ceiling 1000000000 \
    --expect-totals text=752393024,vision=100592896,mtp=20452864,checkpoint=873438784 \
    --report-out results/accounting/20261117T001252Z-qwen35-08b-base-text-only.json
schema=selvedge/model-accounting/v2  ok=true  verdict=certified
```

Receipts: `results/accounting/20261117T001252Z-qwen35-08b-base-text-only.json`
(certified, `752,393,024` unique active, headroom `247,606,976`),
`...-base-text-and-mtp.json` (certified at `772,845,888`) and
`...-base-refused-untied.json` (`ok=false`, refused for declaring an untied head
that does not exist). Court receipts: `results/courts/rust-20261117T001830Z-37d602.json`,
`...-20261117T001831Z-594fc0.json` and the text-and-mtp invocation's own.

The four checked expectations match on the `-Base` bytes exactly, and the
checkpoint's streaming digest is
`19d0b4b980cfe94f9856f6a767e14b670bbf0844a440607345c58fd2b1d40076` against the
chat revision's `112580451bc3179cfa4af8d446c442ed377747f2083e9702bdaef601492e3a01`.
The digests differ because the weights differ; the accounting does not, because
the graph does not (§10).

## 6. Ceiling arithmetic for the registered 1e9 rule

The rule is unchanged: a token's audit must show fewer than one billion
unique-active elements. For a dense model that is the elements the token's
execution touches, ties counted once.

Qwen3.5-0.8B-Base (the anchor), text path — the counts are identical for the
chat revision of the same graph (§10), and this section's certificate is the
`-Base` one:

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

### 6.1 The planned per-token union for P19, computed (S33)

The ceiling rule is not about a checkpoint's total: it is about what one token
reads. `scripts/s33_budget_union.py` computes that union from the accounting
receipt above, with the design constants declared in the receipt rather than
assumed in prose:

| term | declared value |
| --- | --- |
| LoRA rank | 16 |
| LoRA layers | 8 |
| LoRA sites per layer | the FFN triple, gate + up + down |
| passes | 4 |
| memory projections and gates | 2 + 2, each `H x H`, per memory |
| memory rows read per token | 256 |
| FFN | dense-active (no sharding credit taken) |

Receipt: `results/accounting/20261116T184127Z-qwen35-budget-union.json`.

The union is a count of **distinct** elements, so reuse across the four passes
changes the union and the per-token *work* differently. Both readings are
printed; the honest budget is the larger one.

Shared adapters across the passes:

<!-- BEGIN GENERATED: qwen35-union-text-plan -->
759405376
<!-- END GENERATED: qwen35-union-text-plan -->

elements, headroom

<!-- BEGIN GENERATED: qwen35-union-text-plan-headroom -->
240594624
<!-- END GENERATED: qwen35-union-text-plan-headroom -->

A distinct adapter set per pass:

<!-- BEGIN GENERATED: qwen35-union-distinct-per-pass -->
777296704
<!-- END GENERATED: qwen35-union-distinct-per-pass -->

elements, headroom

<!-- BEGIN GENERATED: qwen35-union-distinct-headroom -->
222703296
<!-- END GENERATED: qwen35-union-distinct-headroom -->

If the vision tower were also executed by the path, the union becomes

<!-- BEGIN GENERATED: qwen35-union-plus-vision -->
859998272
<!-- END GENERATED: qwen35-union-plus-vision -->

and with vision and MTP both active

<!-- BEGIN GENERATED: qwen35-union-plus-vision-plus-mtp -->
880451136
<!-- END GENERATED: qwen35-union-plus-vision-plus-mtp -->

Worst case printed by the receipt — a distinct adapter set per pass, vision and
MTP both executed — leaves

<!-- BEGIN GENERATED: qwen35-union-worst-headroom -->
101657536
<!-- END GENERATED: qwen35-union-worst-headroom -->

elements of headroom.

Three facts follow, and they are the point of the calculation:

1. **The planned design fits the ceiling with room to spare**, on every reading.
   The anchor is not ceiling-bound; the ceiling does not decide the design. What
   decides it is whether the mechanism helps.
2. **Nothing in the union exceeds the ceiling even with the vision tower and the
   MTP head both executed**, so the ceiling is not a reason to hide either of
   them from the ledger. The declared path is declared for honesty, not to buy
   headroom.
3. **The FFN is counted dense.** That is 264,241,152 elements of the union that a
   sharded execution could remove, so every headroom printed here is a floor.

What this is not: a measurement. It is arithmetic over declared constants and
receipted component counts. The design constants are choices; the component
counts are not. When P19's memory design changes, the receipt changes with it,
and the numbers above are regenerated rather than edited.

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

### The 2B fetch decision, measured (S36)

The rule for this session was explicit: fetch the gap reference only if it fits
with ten gibibytes of headroom left on the cache filesystem. It was fetched, and
the accounting is a receipt rather than a sentence. Cache free bytes before the
fetch:

<!-- BEGIN GENERATED: frontier-2b-fetch-cache-free-before -->
20333662208
<!-- END GENERATED: frontier-2b-fetch-cache-free-before -->

bytes the fetch asked for (the tree's declared total for the files it downloads):

<!-- BEGIN GENERATED: frontier-2b-fetch-requested-bytes -->
4567836299
<!-- END GENERATED: frontier-2b-fetch-requested-bytes -->

cache free bytes after it:

<!-- BEGIN GENERATED: frontier-2b-fetch-cache-free-after -->
15785103360
<!-- END GENERATED: frontier-2b-fetch-cache-free-after -->

Receipt: `results/frontier-models/20261116-s36-fetch-qwen35-2b-base.json`, which
also records that the branch head at fetch time **was** the pinned revision (so
nothing moved under the pin) and that the shard's computed sha256 equals the
Hub's LFS oid. The headroom that remained is the difference between the last two
numbers; it is above ten gibibytes, which is the condition the decision was
taken under. The fetch ran in the eval court, one model at a time, with its own
memory limit — no two downloads or scorings ever overlapped.

### Recommendation

**Superseded 2026-11-16 (S35): the route is online teacher forwards, not a cache.**

The recommendation below was made against the cache route and is retained as
history. The program changed route for a measured reason: the cache was the only
binding disk constraint, and an online teacher removes it (2B bf16 is 3.5 GiB of
weights against a 49 GB cache; 4B at 8-bit is 3.9 GiB). The saving, the remaining
throughput requirement and the residency are generated in
`docs/PROGRAM-PLAN.md` §24 ("The teacher route") from
`results/frontier-models/20261116T193500Z-frontier-arithmetic.json`. What survives
from this section: the **formula** below, the reserved-headroom rule, and the
requirement that any cache that is reinstated be re-derived here rather than
budgeted from prose.

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
  was produced; no harness was run. **Superseded by §9 (S36)**, which ran the
  pinned harness and produced the anchored scoreboard; the sentence is kept
  because it is true of the S32 verification this section reports, and §9 is the
  receipt-backed correction.
- It does not establish that the LFM licence permits the proposed use. That is
  legal review, and the licence is not Apache-2.0.
- It does not reproduce the vendor's benchmark table. Those numbers are the
  vendor's claims and are not in any receipt here.
- It does not prove the absence of prior art for the per-pass memory idea;
  `docs/PRIOR-ART-AND-NOVELTY.md` §16–§17 records what was searched and what the
  closest matches are.
- It does not establish an **executed** set. The auditor proves that a declared
  path is internally consistent and that its union is under the ceiling; it does
  not observe a kernel reading a pointer. "Audited ≤ 1e9 unique active" still
  requires Court E, and Court E requires the kernels registered in
  `docs/RUNTIME.md` §21.
- It does not establish that the **reference forward agrees with the vendor
  runtime bitwise**. It agrees to argmax equality and 8/8 greedy tokens on four
  prompts (`results/reference/20261116T184032Z-qwen35-reference.json`), which is
  far stronger than a tolerance on one tensor and still short of bitwise.
- It does not establish any FLOP or latency number for the hybrid path. No FLOP
  formula is defined for a DeltaNet scan in this repository, so
  `inspect-model` emits none for this schema rather than inventing one.

## 9. The anchored scoreboard (S36)

This section is the fixed instrument every later WS19 stage is measured against.
It answers a question §8 explicitly said this document did not answer: *what does
this checkpoint actually score, measured by an independent harness, before anyone
trains anything?*

**No training happened here.** No registered receipt was modified, no target
model exists, and every number below was produced by
`EleutherAI/lm-evaluation-harness` — not by SELVEDGE code — over the two Base
checkpoints pinned in §2. The harness runs in `docker/Dockerfile.eval`, an image
built on the oracle court's vendor stack, pinned by image id in
`docker/IMAGES.lock` and by commit in the pre-registration file.

### Why the instrument is pinned before the scores

`docs/anchor-preregistration-v1.json` is the single source of the config: the two
revisions, the task list, the few-shot counts, the sample limits, the batch sizes,
the prompt and extraction behaviour, the seed set, and the paragraph that defines
the win condition. Every run receipt carries the **sha256 of that file**, the
harness commit re-read from the image at run time, and the harness's own resolved
task config for each task (`doc_to_text`, delimiter, `generation_kwargs`,
`metric_list`, `filter_list`) together with sha256 hashes of the task yaml and its
`utils.py`. The run refuses to proceed when the harness commit differs from the
pre-registration, when a task is not in the file, or when a cached checkpoint file
differs in size from `models/FRONTIER.lock.json`; `gap` refuses to compare two
receipts produced under different pre-registration digests. That is what makes
"the identical config applies to both models" a checked property rather than a
sentence.

Two choices in that file are load-bearing and worth stating plainly:

- **Completion-style prompts, no chat template.** Both checkpoints are Base
  models. `apply_chat_template` is false, `fewshot_as_multiturn` is false, and
  every task uses the harness's own `doc_to_text` and its own few-shot pool. No
  model-specific prompt tuning exists anywhere in the pipeline.
- **Fixed batch sizes, not `auto`.** The harness autotuner allocates until the
  device refuses, which would turn a resource measurement into a record of the
  autotuner's failed attempt, and the device is shared with a desktop session.
  The sizes are derived from the smoke receipts named in the pre-registration.

### What the scored path is

`transformers` resolves `model_type: qwen3_5` to `Qwen3_5ForCausalLM` (the vendor's
"VLM compatibility" mapping), and the loaded parameter count is checked against
the audited text tower. So the harness scores the **text tower only**: the vision
tower and the MTP head are never instantiated. The two fused kernels the model
would like (`causal_conv1d`, `flash-linear-attention`) are absent from the eval
court, so transformers runs its own reference PyTorch implementations of the short
convolution and the gated delta rule. That is the exact path rather than an
approximation, and it is also the slow path — which is why the sample limits exist
and why the wall clock is recorded per task.

### The scoreboard

Scores are percentages; `SE` is the harness's own bootstrap standard error on the
primary metric; `n` is the number of scored items, not the number requested.

<!-- BEGIN GENERATED: frontier-anchor-scoreboard -->
| task | metric | n | few-shot | batch | anchor | SE | 2B | SE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mmlu_pro` | `exact_match,custom-extract` | 504 | 5 | 8 | 27.58 | 1.98 | 38.29 | 2.14 |
| `mmlu` | `acc,none` | 14042 | 5 | 2 | 48.22 | 0.41 | 53.95 | 0.41 |
| `gsm8k` | `exact_match,strict-match` | 1319 | 5 | 8 | 38.82 | 1.34 | 59.67 | 1.35 |
| `minerva_math500` | `math_verify,none` | 500 | 4 | 8 | 23.40 | 1.90 | 35.20 | 2.14 |
| `humaneval` | `pass@1,create_test` | 164 | 0 | 8 | 20.73 | 3.18 | 31.71 | 3.64 |
| `ifeval` | `prompt_level_strict_acc,none` | 541 | 0 | 8 | 26.06 | 1.89 | 34.01 | 2.04 |
| `arc_challenge` | `acc_norm,none` | 1172 | 25 | 4 | 43.26 | 1.45 | 54.18 | 1.46 |
| `triviaqa` | `exact_match,remove_whitespace` | 1000 | 5 | 8 | 19.60 | 1.26 | 30.50 | 1.46 |
<!-- END GENERATED: frontier-anchor-scoreboard -->

Reading notes that are claims about the instrument rather than the models:

- `mmlu_pro` and `mmlu` are group tasks. `limit` counts **per subject** for a group,
  so the registered MMLU-Pro sample is a stratified slice across all fourteen
  subjects, and `mmlu` runs every item of all fifty-seven.
- `minerva_math500` is scored with symbolic `math_verify`, not string match;
  `mmlu_pro` uses the harness's `custom-extract` filter; `gsm8k`'s primary metric
  is the strict `#### <n>` extraction (the flexible extraction is recorded in the
  receipt alongside it and is never used to choose a headline number).
- `humaneval` executes generated code inside the court (`unsafe_code: true` in the
  harness yaml, code evaluation enabled only in that sandbox).

### The gap to the reference, and the win condition

The registered win condition is "close at least half the gap on one family". The
arithmetic it needs is per task and per family:

<!-- BEGIN GENERATED: frontier-anchor-gap -->
| task | family | anchor | 2B | gap (pp) | half gap (pp) | target = anchor + half |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mmlu_pro` | knowledge | 27.58 | 38.29 | 10.71 | 5.36 | 32.94 |
| `mmlu` | knowledge | 48.22 | 53.95 | 5.73 | 2.87 | 51.09 |
| `gsm8k` | math | 38.82 | 59.67 | 20.85 | 10.42 | 49.24 |
| `minerva_math500` | math | 23.40 | 35.20 | 11.80 | 5.90 | 29.30 |
| `humaneval` | — | 20.73 | 31.71 | 10.98 | 5.49 | 26.22 |
| `ifeval` | — | 26.06 | 34.01 | 7.95 | 3.97 | 30.04 |
| `arc_challenge` | — | 43.26 | 54.18 | 10.92 | 5.46 | 48.72 |
| `triviaqa` | — | 19.60 | 30.50 | 10.90 | 5.45 | 25.05 |

| family | members | anchor mean | 2B mean | gap (pp) | half gap (pp) | target |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| knowledge | `mmlu_pro`, `mmlu` | 37.90 | 46.12 | 8.22 | 4.11 | 42.01 |
| math | `gsm8k`, `minerva_math500` | 31.11 | 47.43 | 16.32 | 8.16 | 39.27 |
<!-- END GENERATED: frontier-anchor-gap -->

Family means are the **unweighted** mean of the member task scores, fixed in the
pre-registration before any score existed, because weighting by item count would
let MMLU's full test set decide a family mean that GSM8K is equally supposed to
describe.

Only the registered member lists define a family. ARC-Challenge, TriviaQA,
HumanEval and IFEval are anchored tasks and their gaps are printed above, but they
are deliberately **not** members of either family: the win condition was
registered over two tasks per family, and quietly widening a family after seeing
the scores is exactly the kind of move this document exists to prevent. Their
per-task targets are printed so a later stage can use them if it wants, labelled
as per-task rather than as a family.

The two families pull in different directions. The knowledge family is a short
gap and the math family a long one, so "half the gap on one family" is a much
cheaper target on knowledge than on math — which is worth knowing *before* a
training run rather than after, and is the main reason this scoreboard exists.

### The launch record

Both models were scored one at a time, each in its own court invocation, each
with an explicit memory limit sized from a measured peak rather than from a
default. The numbers below are read from the two run receipts, not typed:

<!-- BEGIN GENERATED: frontier-anchor-launch -->
| run | model | container limit | container peak (B) | device peak (GiB) | device total (GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| anchor | `Qwen/Qwen3.5-0.8B-Base` | 8g | n/a (assembled) | 10.05 | n/a |
| gap reference | `Qwen/Qwen3.5-2B-Base` | 12g | 7195586560 | 12.15 | 15.61 |
<!-- END GENERATED: frontier-anchor-launch -->

The anchor's limit is its smoke's measured container peak times 1.5 plus a 1 GiB
floor margin, which lands under the court default it was given. The gap
reference's limit is the anchor's measured peak scaled by the ratio of the two
checkpoints' weight bytes, times 1.5 plus the same margin; its measured peak came
in under that limit, so the arithmetic held. The device peak is the largest
`torch.cuda.max_memory_allocated` over that run's tasks, taken from the receipt —
MMLU's loglikelihood batch sets it, which is why that batch size is the small one.
Host `MemAvailable` was checked before each launch, and the two runs never
touched the GPU at the same time.

The 2B reference completed in a single invocation. The anchor's receipt is an
assembly of two invocations with an `assembly` block and per-entry `invocations`
provenance, for the reason recorded there.

One arithmetic caveat that belongs next to the table rather than in a footnote:
the win condition is a statement about *scores on this instrument*, and a family
mean over two tasks inherits the sampling error of both. The gap is large enough
on both families that no plausible standard-error combination reverses its sign,
but "half the gap" is a scoreboard target, not a significance test.

### Determinism, and what would make a number a defect

Three failure modes are checked rather than assumed:

1. **A silent zero.** A task scoring exactly `0.000` is flagged in the receipt and
   fails the run's exit code. It was hit once (TriviaQA, n=8) and diagnosed from
   the logged samples: the prompt was well formed and the extraction returned a
   string, the model simply answered the wrong entity eight times. A hundred-item
   probe then scored non-zero, so the zero was a small-sample artefact and not a
   harness defect. That diagnosis is what makes the reported number recordable.
2. **A drifted config.** `gap` refuses receipts whose pre-registration digest
   differs, and the scoreboard generator refuses to mix two runs of one model.
3. **A stale table.** The four generated regions are owned by
   `scripts/s36_scoreboard.py`, which fails closed when a body does not match the
   receipts on disk, and are declared as external regions in
   `docs/perf-numbers.json` so the perf-numbers court refuses a generated region
   with no owner.

### What this section does not establish

- It is not a quality claim about the DeltaNet hybrid as an architecture. It is
  what these two checkpoints score, and the 2B reference is over the ceiling, so
  the gap is a capability gap and not a compliant-parameter gap.
- It is not tournament-grade. Subsampled tasks (`mmlu_pro`, `triviaqa`) carry the
  sampling error printed above them; the full-set tasks do not.
- It does not cover the gated or non-open checkpoints. `google/gemma-3-1b-it`
  answers HTTP 401 to anonymous requests and `LiquidAI/LFM2.5-1.2B-Instruct`
  carries the non-OSI LFM Open License v1.0, so neither is scored here; the
  pre-registration records both exclusions and their reasons.
- It does not make the vendor's published numbers comparable. Those are the
  vendor's instrument, and no vendor number appears in this document.

## 10. The audited model is the benchmarked model (S38)

The ceiling certificate was produced for `Qwen/Qwen3.5-0.8B` (chat) while the
scoreboard in §9 scored `Qwen/Qwen3.5-0.8B-Base`. An audit of one revision is not
an audit of another unless that is *measured*, so it was measured. The tool is
`scripts/s38_revision_compare.py`; the receipt is
`results/frontier/20261117T002500Z-revision-compare-08b-chat-vs-base.json`; the
court receipt is `results/courts/oracle-20261117T001929Z-3c53a9.json`.

Method, in the same shape as §1: the safetensors header of each shard is read
(the 8-byte little-endian length, then the JSON table of name, dtype, shape and
`data_offsets`); the raw header bytes are hashed; the tensor table is sorted and
hashed; each whole shard is hashed by streaming; and the small files that decide
compatibility (`config.json`, the tokenizer files, `vocab.json`, `merges.txt`,
`LICENSE`) are hashed individually. The shard's tensor byte total is checked
against the Hub index's own `total_size` before any comparison is drawn.

### 10.1 The measured answer

Two shards, read header-first. The header size in bytes:

<!-- BEGIN GENERATED: revcmp-header-bytes -->
59,840 bytes
<!-- END GENERATED: revcmp-header-bytes -->

and the number of tensors it describes:

<!-- BEGIN GENERATED: revcmp-tensor-count -->
488
<!-- END GENERATED: revcmp-tensor-count -->

The chat shard's whole-file sha256 is `04b1c301…fe4696`; the base shard's is

<!-- BEGIN GENERATED: revcmp-base-shard-sha256 -->
c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c
<!-- END GENERATED: revcmp-base-shard-sha256 -->

and the two shards are the same size, the chat one at

<!-- BEGIN GENERATED: revcmp-chat-weight-bytes -->
1,746,942,600 bytes
<!-- END GENERATED: revcmp-chat-weight-bytes -->

and the base one at

<!-- BEGIN GENERATED: revcmp-base-weight-bytes -->
1,746,942,600 bytes
<!-- END GENERATED: revcmp-base-weight-bytes -->

so a size check cannot tell them apart and only a hash can. `config.json` is
byte-identical; the base copy hashes to

<!-- BEGIN GENERATED: revcmp-config-sha256 -->
b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
<!-- END GENERATED: revcmp-config-sha256 -->

with the chat copy's `b90b86f3…de9b4204` equal to it. The BPE vocabulary has

<!-- BEGIN GENERATED: revcmp-chat-vocab-entries -->
248,044
<!-- END GENERATED: revcmp-chat-vocab-entries -->

entries in both files, and the merge list is byte-identical. The added-token
tables are not: the chat revision declares

<!-- BEGIN GENERATED: revcmp-chat-added-tokens -->
26
<!-- END GENERATED: revcmp-chat-added-tokens -->

added tokens against the base revision's

<!-- BEGIN GENERATED: revcmp-base-added-tokens -->
22
<!-- END GENERATED: revcmp-base-added-tokens -->

**Are the text towers identical?** The **graph** is identical and the **weights**
are not. Every tensor has the same name, dtype, shape and byte offset, the header
is byte-identical, `config.json` is byte-identical, and the element counts of §5
and §6 therefore transfer between the revisions exactly. The shard hashes differ,
so the two checkpoints hold different values: the chat revision is a
post-trained (chat) version of the same architecture.

**The tokenizer is the one place where "same graph" does not imply "same
vocabulary object".** The BPE vocabulary and the merge list are byte-identical
(the `vocab.json` and `merges.txt` files match, and the digests of the
`tokenizer.json` `model.vocab` and `model.merges` entries match). What differs is
the added-token table: the chat revision carries four extra control tokens —
`<tool_response>`, `</tool_response>`, `<think>`, `</think>` — at ids
248,066–248,069, which the Base revision does not define. Two consequences, both
recorded rather than fixed by hand:

1. **The distortion this defect could have caused did not happen to the score.**
   Both scored checkpoints are Base (§9), and the stage-2 teacher/student pair is
   also Base-on-Base: the pinned stream receipt records the anchor's tokenizer
   sha256 and states it is the same file in 0.8B-Base and 2B-Base. So the
   distillation pairing is byte-identical at the tokenizer.
2. **A chat teacher paired with a Base student would be a different comparison.**
   The chat tokenizer can emit four tokens the Base student never saw in
   pretraining. Their ids fall in the embedding rows that the BPE vocabulary does
   not name (the vocabulary-versus-embedding-row gap recorded in §4), so the rows
   exist in both graphs but only the chat checkpoint has trained them. Nothing in
   WS19 pairs a chat checkpoint with a Base one, and this is now the reason it
   must not.

### 10.2 What moves, and what the fix changed

Nothing numeric moved: the ceiling, the element counts, the headroom and the
accounting are properties of the graph, and the graph is identical. What moved is
**identity**, and it moved in four places, all now naming `-Base`:

- the anchor certificate: §5.1 is the `-Base` certificate, and the chat
  certificates are retained as the S33 record;
- `docs/MODELS.md`: the frontier section says which revision is the anchor and
  which is only the same graph;
- `docs/PROGRAM-PLAN.md` §24: the Target paragraph names the `-Base` revision;
- `models/FRONTIER.lock.json`: already pinned the `-Base` revisions (S36), which
  is why the lock did not need re-writing — the defect was in the docs and the
  certificate, not in the lock.

### 10.3 What this section does not establish

- It does not re-derive the reference parity receipt
  (`results/reference/20261116T184032Z-qwen35-reference.json`), which was measured
  on the chat revision. The reference *implementation* is graph-level and is
  re-exercised on the `-Base` bytes by the state test at S38, but the
  vendor-parity comparison itself has not been re-run on `-Base`. That is the
  honest gap, and it is a one-command job (`scripts/s33_reference_parity.py`)
  when a GPU court is next open.
- It does not establish that the chat and base revisions score the same. They
  share a graph and differ in weights, so a capability difference is expected
  and is not measured here.
