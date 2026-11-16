# Calibration split (S28) — generated, do not edit by hand

This directory is **derived** from `fixtures/capability/tasks` by
`oracle/capability/derive.py`. Every item names its source task and that source's
tree hash in `task.json` (`derived_from`), and `DERIVATION.json` binds the whole
tree to the source suite hash. A hand edit is a failed `derive-formats --check`.

- `tasks/` — the calibration items, in four formats (see
  `docs/CAPABILITY-COURT.md` §13);
- `trio/`, `probes/` — empty; this collection has no trio or probes;
- `DERIVATION.json` — the derivation receipt: the split rule, the source ids, the
  per-format counts, the items that could not be derived and why, and the
  manifest hash;
- `MANIFEST.json` — the capability-suite manifest, with `format`, `tier`,
  `metric` and `extraction_policy` per item;
- `VERSION.json` — **absent until the calibration pass runs.** It is the
  calibrated suite version: the items that survived the registered acceptance
  window. Its absence is the honest statement that this split is a hypothesis.

Verification, with the real checkers (CPU):

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \
scripts/court rust -- python3 scripts/capability_court.py verify-formats \
  --root fixtures/capability-calibrated --prune \
  --out results/capability/derivation-verify-calibration.json
```

`--prune` removes a patch candidate or diagnostic option that the checker shows
*also* passes, and drops an item left with too few choices; the removals are in
the verification receipt. Re-running `derive-formats --force` recreates the
pre-prune candidates, so the order is derive → verify.

This split is **not** the scored suite, provides **no** capability claim, and is
disjoint from the scored split and from the sealed holdback by construction.
