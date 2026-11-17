# Teacher models: identity, license and hashes

Status: **acquired and verified**. Both checkpoints are Apache-2.0. Weights live
only in the persistent Hugging Face cache on the host
(`~/.cache/selvedge/hf`, mounted read-write at `/hf-cache` inside the oracle
court); no weights are committed to this repository. The tracked lock file
`models/MODELS.lock.json` records every file's byte size and SHA-256.

Receipts:

- acquisition: `results/models/fetch-20261114T201604Z.json`
- tokenizer identity: `results/models/tokenizer-20261114T205700Z-qwen3-0.6b.json`
  and `...-qwen3-1.7b.json`
- exact parameter accounting: `results/accounting/20261114T201533Z-qwen3-0.6b.json`
  and `...-qwen3-1.7b.json`

## Models

**`Qwen/Qwen3-0.6B`** — control: the dense model is under the 1e9 ceiling.

- revision:
  <!-- BEGIN GENERATED: models-06b-revision -->
c1899de289a04d12100db370d81485cdf75e47ca
  <!-- END GENERATED: models-06b-revision -->
- license: Apache-2.0
- weight bytes:
  <!-- BEGIN GENERATED: models-06b-weight-bytes -->
1503300328
  <!-- END GENERATED: models-06b-weight-bytes -->
- weight files:
  <!-- BEGIN GENERATED: models-06b-weight-file-count -->
1
  <!-- END GENERATED: models-06b-weight-file-count -->

**`Qwen/Qwen3-1.7B`** — binding teacher: the backbone fits, the dense model does
not.

- revision:
  <!-- BEGIN GENERATED: models-17b-revision -->
70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  <!-- END GENERATED: models-17b-revision -->
- license: Apache-2.0
- weight bytes:
  <!-- BEGIN GENERATED: models-17b-weight-bytes -->
4063515592
  <!-- END GENERATED: models-17b-weight-bytes -->
- weight files:
  <!-- BEGIN GENERATED: models-17b-weight-file-count -->
2
  <!-- END GENERATED: models-17b-weight-file-count -->

**`Qwen/Qwen3-4B`** — reference only; **not downloaded**, because its
always-on backbone alone exceeds 1e9 (`docs/BUDGET-MATH.md`). Pinned revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Apache-2.0.

Weight hashes:

- Qwen3-0.6B `model.safetensors` SHA-256:
  <!-- BEGIN GENERATED: models-06b-weight-sha256 -->
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
  <!-- END GENERATED: models-06b-weight-sha256 -->
- Qwen3-1.7B first shard SHA-256:
  <!-- BEGIN GENERATED: models-17b-weight-sha256-a -->
169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5
  <!-- END GENERATED: models-17b-weight-sha256-a -->
- Qwen3-1.7B second shard SHA-256:
  <!-- BEGIN GENERATED: models-17b-weight-sha256-b -->
912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9
  <!-- END GENERATED: models-17b-weight-sha256-b -->
- LICENSE file SHA-256 (both models ship the identical file):
  <!-- BEGIN GENERATED: models-license-sha256 -->
832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e
  <!-- END GENERATED: models-license-sha256 -->
- `config.json` SHA-256 (Qwen3-0.6B):
  <!-- BEGIN GENERATED: models-06b-config-sha256 -->
660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
  <!-- END GENERATED: models-06b-config-sha256 -->

The weight and `tokenizer.json` hashes were checked against the Hugging Face LFS
object id at download time (`verified_source = "hf_lfs_oid"`). Small files that
are not stored in LFS were hashed locally and recorded as `computed_sha256`; the
lock says which is which. A mismatch fails the fetch closed and never replaces an
existing verified file.

## Tokenizer identity

- class:
  <!-- BEGIN GENERATED: models-tokenizer-class -->
Qwen2Tokenizer
  <!-- END GENERATED: models-tokenizer-class -->
- tokenizer length, including added tokens:
  <!-- BEGIN GENERATED: models-tokenizer-length -->
151669
  <!-- END GENERATED: models-tokenizer-length -->
- tokenizer vocabulary:
  <!-- BEGIN GENERATED: models-tokenizer-vocab -->
151643
  <!-- END GENERATED: models-tokenizer-vocab -->
- model embedding rows, from the checkpoint:
  <!-- BEGIN GENERATED: models-embedding-rows -->
151936
  <!-- END GENERATED: models-embedding-rows -->
- `tokenizer.json` SHA-256:
  <!-- BEGIN GENERATED: models-tokenizer-json-sha256 -->
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
  <!-- END GENERATED: models-tokenizer-json-sha256 -->
- end-of-sequence token id: 151645

The tokenizer vocabulary is **smaller than the embedding matrix**. The unused
rows are still learned parameters and are counted in the accounting. They are
not reachable tokens and must not be counted as tokenizer vocabulary. This is a
documented discrepancy, not a silent assumption.

The oracle loads tokenizers with `trust_remote_code=False` and
`local_files_only=True`. No model repository code is executed.

## Intended use

- Teacher forward passes and FFN activation capture for Court A and the
  selector work.
- Exact FFN tensor conversion into SELVEDGE group bundles.
- No redistribution: weights stay in the host cache and are not committed.
- No evaluation claim is made from these models yet. Any later evaluation
  dataset must be added with its own revision, license and contamination note.

### Contamination caveat

No evaluation dataset is named in this phase. The only prompt set is
`fixtures/prompts/qwen3-teacher-prompts.json`, which was written for this
repository and is not drawn from a benchmark. Any dataset introduced later must
be checked for (a) literal overlap with the prompts used for calibration and
(b) membership in the teacher's pretraining corpus; a benchmark score computed
on text the teacher memorised is not evidence for a sparse-execution claim.

## License

The following is the complete `LICENSE` file shipped with both checkpoints,
reproduced verbatim. Its SHA-256 is above and both files are byte-identical.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2024 Alibaba Cloud

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Acquisition procedure

```bash
# one networked step; idempotent and hash-checked
scripts/fetch-models.sh

# offline verification against models/MODELS.lock.json
scripts/fetch-models.sh --verify-only
```

`scripts/fetch-models.sh` runs `oracle/fetch_models.py` inside the oracle court
with `--net`. It resolves the pinned commit SHA, compares every file with the
Hugging Face LFS object id when one exists, streams to a `.partial` file,
fsyncs, and renames. A present file whose hash matches is never re-downloaded or
overwritten. A present file whose hash does not match is a failure. The resolved
revision is printed and must equal the pinned revision.

## Known risks

- The tokenizer is the slow `Qwen2Tokenizer`, not a fast tokenizer, because that
  is what the pinned `tokenizer_config.json` names. Tokenization is therefore a
  Python-loop cost in the oracle, not a performance claim.
- The tokenizer vocabulary and the embedding row count disagree, as recorded
  above. The unused rows are counted as learned parameters.
- Normalization and chat-template choices affect activations. This phase uses
  raw tokenization with no chat template because the prompts are plain text; any
  later comparison must pin the template too.
- The license text is reproduced for audit convenience. The authoritative file
  is the one in the cache, whose SHA-256 is recorded.

## Frontier checkpoints are locked separately (`models/FRONTIER.lock.json`)

The two checkpoints the WS19 scoreboard scores are **not** in
`models/MODELS.lock.json`. That file pins the teacher candidates the oracle lock
verifies, and `oracle/fetch_models.py` regenerates it wholesale from
`models/model-spec.json`; adding a frontier entry there would either be dropped on
the next teacher fetch or create a lock entry nothing in the serving path can run
(the reasoning is in `docs/FRONTIER-TARGET.md` §2).

Instead `models/FRONTIER.lock.json` records, per checkpoint: the pinned revision,
the Hub's own tree listing at that revision (declared byte size and LFS oid per
file), the cached file's computed sha256, whether the sha256 equals the Hub's LFS
oid, the licence id from the model card, the `LICENSE` file's hash and opening
lines, and the cache filesystem's free bytes before and after.

It is produced by `scripts/s36_anchor.py lock` — in the eval court, with network
only for the Hub tree — and it fails closed on a cached file that is absent, a
size that disagrees with the Hub, a sha256 that disagrees with the LFS oid, or a
cached file the Hub tree does not list. The scoreboard run re-checks every locked
file's byte size before it loads a checkpoint, so a silently truncated or replaced
weight file cannot be scored.

Both pinned checkpoints are Apache-2.0: the model card's `license` field says
`apache-2.0` and the cached `LICENSE` file is the Apache License 2.0 text with the
same hash for both (`50cbab8a892c5f29...`). The two reference models that cannot be
scored are excluded in `docs/anchor-preregistration-v1.json` with reasons rather
than omitted: `google/gemma-3-1b-it` is gated and `LiquidAI/LFM2.5-1.2B-Instruct`
is under the non-OSI LFM Open License v1.0.

### Which revision is the anchor (`-Base`, corrected at S38)

The WS19 anchor is **`Qwen/Qwen3.5-0.8B-Base`** at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. `Qwen/Qwen3.5-0.8B` is the **chat**
revision of the same graph: it is not the anchor, it is not scored, and no anchor
command may name it.

The difference was measured rather than assumed
(`results/frontier/20261117T002500Z-revision-compare-08b-chat-vs-base.json`,
written up in `docs/FRONTIER-TARGET.md` §10):

- **the graph is identical.** The safetensors header is byte-identical, every
  tensor has the same name, dtype, shape and byte offset, and `config.json` is
  byte-identical. Any parameter count, ceiling or union arithmetic therefore
  transfers between the two revisions;
- **the weights are not.** The two shards hash differently, so a capability score
  does not transfer;
- **the tokenizer files are not identical.** The BPE vocabulary and the merge
  list are byte-identical, but the chat revision adds four control tokens
  (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`) that Base does
  not define. Both WS19 checkpoints are Base, and the stage-2 stream receipt
  records that 0.8B-Base and 2B-Base ship the same `tokenizer.json` by hash, so
  the teacher/student pairing is unaffected — but a chat checkpoint may not be
  paired with a Base one as a teacher/student pair, because the chat tokenizer
  can emit four tokens the Base model never trained.

`docs/FRONTIER-TARGET.md` §5.1 is the ceiling certificate for the `-Base` bytes;
the earlier chat certificates are kept as the S33 record.
