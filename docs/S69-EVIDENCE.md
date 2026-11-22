# S69 evidence — how to check the corpus from a plain checkout

`docs/S69-RESULT.md` cites receipts that were produced from rollouts and
preference pairs written under `results/artifacts/`. That directory is
gitignored (it holds large per-run artifacts), so before this change an outside
reviewer could read the receipts but could not inspect the corpus they describe.

The corpus is now in the tree, losslessly compressed, with one index per group.
Model weights, adapters and training checkpoints are **not** committed: they are
`*.pt` files, remain outside the index, and are named only by the SHA-256 the
arm receipts already record.

## Where it lives

| group | source (gitignored) | tracked archive | tracked index |
|---|---|---|---|
| rollout corpus | `results/artifacts/stage2/s69-rollouts/` | `results/stage2/s69-rollouts/raw/*.jsonl.gz`, `*.json.gz` | `results/stage2/s69-rollouts/INDEX.json` |
| preference data | `results/artifacts/stage2/s69-preference/` | `results/stage2/s69-preference/raw/*.jsonl.gz`, `*.json.gz` | `results/stage2/s69-preference/INDEX.json` |

Each index entry records, for one artifact:

* `source` — the original path under `results/artifacts/`;
* `stored` — the tracked `.gz` archive;
* `sha256` — the SHA-256 of the **uncompressed** bytes, which is what the S69
  receipts record;
* `bytes` — the uncompressed byte count;
* `rows` — the line count for JSONL, `null` for a receipt;
* `gz_bytes`, `gz_sha256` — the compressed size and digest;
* `citations` — every receipt field that records the same digest, as
  `receipt#pointer`, with the recorded value and whether it matches.

The receipts themselves (`soup-receipt.json`, `pairs-with-reference-receipt.json`,
`recovery-with-reference-receipt.json`) are included because they are the hashes'
origin; the per-arm receipts under `results/stage2/s69-preference/*/` were already
tracked.

## One command to verify

From a clean checkout, with no GPU, no network and the standard library only:

```bash
python3 scripts/s69_evidence.py verify
```

It reads the tracked `.gz` archives and indexes, decompresses every archive,
recomputes its SHA-256, byte count and row count, and re-reads every cited
receipt field to confirm the digest still matches. It prints a single `OK` line
with the artifact, row and citation counts, and exits non-zero on the first
mismatched digest, row count, byte count or stale citation. It never writes and
never needs `results/artifacts/`.

The court form used in this repository (same script, hard memory/CPU/pids/tmp
limits, no network):

```bash
env SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=256 SELVEDGE_TMP=512m \
  scripts/docker-court.sh -- python3 scripts/s69_evidence.py verify
```

## Regenerating the archives and index

With the original artifacts present:

```bash
python3 scripts/s69_evidence.py pack
```

`pack` gzips at level 9 with a zero timestamp and no stored filename (so the
output is deterministic), writes the archives, derives every digest and count
from the bytes, resolves each citation, and writes both indexes. If any digest
disagrees with the receipt that records it, it marks that citation `matches:
false`, writes the defect into the index, and exits non-zero. A mismatch is
never smoothed over.
