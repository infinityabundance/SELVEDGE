# The S31 selection-item population (generated) — do not edit by hand

This tree is **derived** from `fixtures/capability/tasks` by
`oracle/capability/population.py`. Every item names its source task, that
source's tree hash and the S28 split it belongs to in `task.json`
(`derived_from`), and `POPULATION.json` binds the whole tree to the source suite
hash and records every item's own tree digest.

Four families, all of them selection items:

- `--fn--<file>--<function>` — one function is removed from a solved crate; the
  arm chooses among 4-6 candidate definitions.
- `--file--<file>` — one file is reverted to its initial state; the arm chooses
  among 4-6 candidate whole files.
- `--manifest` — the manifest is the question (dependency-compat), the code is
  solved.
- `--diag` — diagnostic-identification, unchanged from S28, reported on its own
  metric and never pooled with executable success.

Every candidate is downloaded into a crate and compiled by
`verify-formats`; exactly one must pass and every other must fail. The
verification receipt is the evidence, not this file.

The sealed holdback (`fixtures/capability/sealed`) is never read here. The
calibration/scored split from S28 is retained as an attribute of each item so a
reader can still separate the two; it is **not** a holdout for this population.

Verification, with the real checkers (CPU):

```bash
SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \
scripts/court rust -- python3 scripts/capability_court.py verify-formats \
  --root fixtures/capability-population --prune \
  --out results/capability/derivation-verify-population.json
```
