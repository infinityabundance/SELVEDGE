# Fresh capability suite (P12.9, P11.21)

Every item here is original to the SELVEDGE repository and licensed
`MIT OR Apache-2.0`, the same terms as the tree (`LICENSE`). No prompt, test or
solution is copied from a public benchmark; the contamination receipt and its
method are described in `docs/CAPABILITY-COURT.md` §4.

## Layout

```
fixtures/capability/
    MANIFEST.json      generated: per-task hash, capability, checker, limits
    tasks/             the capability suite
    trio/              the anti-superficial-specialisation trio
    probes/            checker-sandbox probes (not capability items)
```

Each task directory contains:

| path | meaning |
| --- | --- |
| `task.json` | metadata, limits, declared failing evidence, declared wrong mutation |
| `initial/` | the failing state handed to the arm |
| `hidden/` | tests the harness owns and restores over anything the arm writes |
| `reference/` | one reference solution that must pass the checker |

The manifest's `tasks[]` entries carry the fields P12.9 asks for: `tree_sha256`,
`license`, `capability`, `difficulty` (an author label, not a measurement),
`checker.kind`, `checker.timeout_seconds`, `checker.address_space_bytes`, plus
the failing-state expectation and the reference/hidden file lists. The eight
suite categories are `lifetime-borrow`, `failing-test`, `api-migration`,
`dependency-compat`, `concurrency`, `trait-impl`, `repo-scale` and
`semantic-repair` (compile-clean but behaviourally wrong); `trio` and `probe`
are separate collections and never enter the primary composite.

## Regenerating

The tree is materialised from `oracle/capability/authoring/`, which is the
reviewable source of every item:

```bash
SELVEDGE_MEM=4g SELVEDGE_CPUS=2 scripts/court rust -- \
    python3 scripts/capability_authoring.py --materialize --manifest
SELVEDGE_MEM=4g SELVEDGE_CPUS=2 scripts/court rust -- \
    python3 scripts/capability_authoring.py --check
```

Do not edit the materialised files by hand: `--check` fails closed on drift.
`MANIFEST.json` is the hash record a run is bound to; the pre-registration in
`docs/capability-preregistration-v2.json` pins the expanded suite hash (v1 pins
the historical 24-task suite and is retained unchanged).
