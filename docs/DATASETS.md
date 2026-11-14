# Evaluation datasets: identity, license, hashes and split

Status: **pinned and fetched** for the M1 frontier experiment. Dataset files are
downloaded into the gitignored `results/artifacts/datasets/` cache by an
explicit, networked step; no corpus text is committed. The tracked lock file
`data/datasets.lock.json` records every file's byte size and SHA-256, and the
tracked receipt `results/datasets/fetch-20261114T212155Z.json` records the
verification.

This document answers three questions: which data was used, under which
license, and how the calibration and held-out sets were kept apart. It does not
report results. Results are in `results/M1-FRONTIER.md`.

## Datasets

**wikitext-2-raw-v1, validation split** — calibration and held-out
general/factual text.

- repository: `Salesforce/wikitext` (dataset)
- revision:
  <!-- BEGIN GENERATED: datasets-wikitext-revision -->
b08601e04326c79dfdd32d625aee71d232d685c3
  <!-- END GENERATED: datasets-wikitext-revision -->
- file: `wikitext-2-raw-v1/validation-00000-of-00001.parquet`
- bytes:
  <!-- BEGIN GENERATED: datasets-wikitext-bytes -->
657209
  <!-- END GENERATED: datasets-wikitext-bytes -->
- SHA-256:
  <!-- BEGIN GENERATED: datasets-wikitext-sha256 -->
204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c
  <!-- END GENERATED: datasets-wikitext-sha256 -->
- license: CC BY-SA 3.0 and GFDL (the dataset card tags
  `license:cc-by-sa-3.0` and `license:gfdl`).
- use: article-level documents. Calibration articles are used only to compute
  selector statistics and the greedy allocation. Held-out articles are used
  only for measurement.

**HumanEval, test split** — held-out code domain slice.

- repository: `openai/openai_humaneval` (dataset)
- revision:
  <!-- BEGIN GENERATED: datasets-humaneval-revision -->
7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544
  <!-- END GENERATED: datasets-humaneval-revision -->
- file: `openai_humaneval/test-00000-of-00001.parquet`
- bytes:
  <!-- BEGIN GENERATED: datasets-humaneval-bytes -->
83920
  <!-- END GENERATED: datasets-humaneval-bytes -->
- SHA-256:
  <!-- BEGIN GENERATED: datasets-humaneval-sha256 -->
2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6
  <!-- END GENERATED: datasets-humaneval-sha256 -->
- license: MIT.
- use: the `prompt` field only (function signature and docstring). The
  canonical solution and the tests are never fed to the model.

**GSM8K, test split** — held-out math domain slice.

- repository: `openai/gsm8k` (dataset)
- revision:
  <!-- BEGIN GENERATED: datasets-gsm8k-revision -->
740312add88f781978c0658806c59bc2815b9866
  <!-- END GENERATED: datasets-gsm8k-revision -->
- file: `main/test-00000-of-00001.parquet`
- bytes:
  <!-- BEGIN GENERATED: datasets-gsm8k-bytes -->
419088
  <!-- END GENERATED: datasets-gsm8k-bytes -->
- SHA-256:
  <!-- BEGIN GENERATED: datasets-gsm8k-sha256 -->
ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59
  <!-- END GENERATED: datasets-gsm8k-sha256 -->
- license: MIT.
- use: the `question` field only. The answer is never fed to the model.

All three files are Hugging Face LFS objects. The fetcher compares the
downloaded SHA-256 with the LFS object id the API reports, so the hash is
verified against the repository's own object identity, not only recomputed
locally.

## What is committed and what is fetched

Committed:

- `data/dataset-spec.json`: the pins (repository, revision, file, license,
  role).
- `data/datasets.lock.json`: the resolved bytes and SHA-256 per file.
- `results/datasets/fetch-*.json`: the fetch/verify receipt.
- `results/splits/split-*.json`: the split receipt, with every document's
  SHA-256, token count, window count and assignment, and the near-duplicate
  statistics.

Not committed:

- the corpus and the derived token-id file. They live under
  `results/artifacts/datasets/` and `results/artifacts/split/`, both gitignored.
  The split receipt records the derived file's SHA-256 so it can be checked
  against a re-derivation.

## Split

Wikitext documents are articles: a level-1 header line (` = Title = `) plus
every following line until the next such header. There are 60 articles in the
validation split. They are ranked by the hex SHA-256 of their normalized text;
the 16 lowest-hash articles are calibration, and of the next 44 by hash the
first 40 that pass the near-duplicate filter are held-out. The split depends
only on the document hash, never on its content or its model behaviour.

Tokenization uses the pinned Qwen3-1.7B tokenizer
(`Qwen2Tokenizer`, `trust_remote_code=False`, `local_files_only=True`).
Calibration articles contribute up to four non-overlapping 512-token windows;
held-out articles up to two. No window crosses an article boundary. Code
prompts are the 40 lowest numeric `task_id`s, one sequence each. Math questions
are the 40 lowest SHA-256 hashes of normalized question text, one sequence each.

Counts and near-duplicate statistics are in the split receipt. The registered
design and thresholds are in `docs/EXPERIMENT-CONTRACT.md`.

## Leakage and contamination

- Calibration and held-out article sets are disjoint.
- Before selection, every calibration x held-out-candidate pair is scored with
  word 5-shingle Jaccard. A candidate at or above 0.2 is discarded. The split
  receipt records the pair count, the maximum Jaccard before and after the
  filter, and the number of pairs in `[0.2, 0.5)` and `>= 0.5`.
- Selectors and the greedy allocation never see held-out text, code, math,
  teacher logits, gold answers or token labels.
- **Residual risk: pretraining contamination.** Wikitext-2, HumanEval and
  GSM8K are public and are plausibly part of Qwen3's pretraining corpus. We
  have no local membership test for the checkpoint. Absolute perplexity is
  therefore optimistic and must not be compared with published numbers. The
  KL and PPL-ratio comparisons are relative to the same teacher on the same
  tokens and do not depend on whether the teacher has seen the text.
- None of this is a capability benchmark. KL on Wikipedia text is not a test of
  code generation, arithmetic or factuality.

## Reproduce

```bash
scripts/fetch-datasets.sh              # explicit network step; idempotent, hash-verified
scripts/fetch-datasets.sh --verify-only  # offline verification against the lock
scripts/court oracle -- python3 oracle/dataset_prep.py \
  --preregistration-commit <registration-sha>
```

The derived split is written under `results/artifacts/split/`, which is
gitignored; the receipt under `results/splits/` is tracked.

## Limitations

- The dataset cards' license tags are recorded, not interpreted as legal
  advice. Wikimedia text is CC BY-SA 3.0 / GFDL; HumanEval and GSM8K are MIT.
- The near-duplicate check is exact shingle Jaccard, not MinHash. With 60
  articles the exact pairwise computation is cheap and needs no approximation.
- A 5-word shingle threshold of 0.2 is a policy choice, registered before the
  split was built. A differently thresholded check could discard different
  candidates; the receipt makes the choice checkable.
