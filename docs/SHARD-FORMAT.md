# SELVEDGE shard container format v1

Status: **implemented** in `selvedge-format`. This document is normative for the
validator and for any converter that writes bundles. It describes what the code
does today; a future version must increment the container version and this
document together.

The format has one job: make a shard's identity, contents and physical layout
checkable before the bytes are used, and make publication atomic so a reader
never observes a half-written bundle.

## Byte layout

```text
offset 0                    8        10                       18
+---------------------------+--------+------------------------+-------------------+
| magic "SLVDGFMT" (8 bytes) | u16 LE | u64 LE manifest length | manifest JSON     |
+---------------------------+--------+------------------------+-------------------+
                                                               | (exact length)    |
                                                               +-------------------+
                                                               | zero padding      |
                                                               | to payload_offset |
                                                               +-------------------+---+
                                                                                   | payload |
                                                                                   | payload_len bytes
                                                                                   +---------+
```

- `magic` is exactly the eight ASCII bytes `SLVDGFMT`.
- `version` is a little-endian `u16`; this build writes `1` and rejects every
  other value.
- `manifest_len` is a little-endian `u64`.
- `manifest` is UTF-8 JSON of exactly `manifest_len` bytes.
- `payload_offset = align_up(18 + manifest_len, alignment)`.
- The file length must be exactly `payload_offset + payload_len`; there is no
  trailing data.
- Shard `offset` values are relative to the start of the payload, not the file.

All integers in the header and in canonical identity encodings are
little-endian. JSON numbers are ordinary JSON values; the manifest's own
semantic values are validated independently.

## Manifest v1

The manifest is a JSON object with `deny_unknown_fields` semantics: an unknown
field is a parse error, not a silently ignored extension.

| field | type | meaning |
| --- | --- | --- |
| `schema` | string | exactly `selvedge/shard-manifest/v1` |
| `schema_version` | u16 | exactly `1` |
| `bundle_id` | string | stable id used in file names; `A-Za-z0-9._-`, must not start with `.` |
| `model` | object | `ModelDigest`: checkpoint digest, optional upstream revision, optional tokenizer digest |
| `transforms` | string[] | sorted, unique transform/repacking ids; every shard's `transform_id` must appear here |
| `alignment` | u64 | power of two, at least 512; every shard offset and `payload_len` are multiples |
| `payload_len` | u64 | non-zero, a multiple of `alignment` |
| `created_by` | string | producing tool and version |
| `shards` | object[] | shard records, validated below |

`ModelDigest` records three things:

- `digest`: SHA-256 over the canonical checkpoint fileset encoding described
  below;
- `upstream_revision`: optional string, recorded verbatim (empty is rejected);
- `tokenizer_digest`: optional SHA-256.

The fileset digest is defined as `SHA-256(encoding)` where the encoding, in the
domain `selvedge/model-digest/v1`, is:

```text
u64  file count
per file, sorted by path:
  str  normalized relative path
  32B  file sha256
  u64  file length
```

Sorting is by byte-wise path order. Duplicate paths are rejected. The encoding
contains no timestamps, host names or map iteration order, so the digest is
reproducible on any platform.

### Shard record

| field | meaning |
| --- | --- |
| `shard_id` | content-derived identity, recomputed by the validator |
| `role` | `TensorRole` stable name (for example `attn.q`, `mlp.gate`, `other:<name>`) |
| `layer` | optional `u32` layer index |
| `channels` | `{kind:"range",start,end}` (half-open) or `{kind:"list",indices:[...]}` (strictly increasing) |
| `channel_axis` | logical axis the channel set applies to |
| `shape` | logical shape of the stored slice; `shape[channel_axis] == channel count` |
| `storage_order` | `row_major`, `column_major`, or `transposed` with an explicit axis permutation |
| `dtype` | stable dtype name, including packed variants and an `unsupported:<name>` rejection path |
| `quant` | `null` for unquantized types; otherwise block size, scales extent and optional zero-point extent |
| `offset` | payload-relative extent offset; multiple of the bundle alignment |
| `length` | extent length in bytes, including auxiliary data |
| `sha256` | SHA-256 of the whole extent: weights followed by auxiliary bytes |
| `alignment` | must equal the bundle alignment |
| `compatibility_id` | label shared by shards that may execute together |
| `transform_id` | listed in `transforms` |
| `co_access` | sorted, unique shard ids expected to be read together |
| `depends_on` | sorted, unique shard ids that must be loaded first |
| `calibration` | optional `{method, dataset_digest?, notes?}` provenance |

Links are validated: self-links and links to shards that are not in the
manifest are rejected. `layer` is caller-defined and is not restricted by role;
the role/layer convention is a converter decision, not a format inference.

### Extent layout and quantization

An unquantized shard has `length == bytes_for(element_count(shape), dtype)`.

A quantized shard stores, in this exact order:

```text
[ weights: bytes_for(elements, dtype) ]
[ scales:  blocks * bytes_per_scale     ]   offset == weight bytes
[ zero points: blocks * bytes_per_zero  ]   optional, immediately after scales
```

`blocks = elements / block_size`. `element_count` must be a whole number of
blocks, and a packed dtype's block size must equal `quant.block_size`.
`scales` and `zero_points` must be unpacked numeric dtypes (for example `f32`,
`f16`, `i8`, `u8`). The sum of the three regions must equal `length` exactly;
there is no unaccounted space inside a shard extent. Scale and zero-point bytes
are non-learned metadata and are never counted as learned parameters by the
ledger.

The fixture's `i4_packed` and `ternary_packed` naming follows the core dtype
rules: 4 bits and 2 bits per logical element respectively, block-aligned.

## Shard identity

`ShardId` is a SHA-256 digest over this canonical encoding, in the domain
`selvedge/shard-id/v1`:

```text
u16  format_version
32B  model.digest
u8   has_upstream_revision, then str revision if present
u8   has_tokenizer_digest, then 32B tokenizer digest if present
str  role stable name (Other becomes "other:<name>")
u8   has_layer, then u32 layer if present
str  "range" or "list"
     range: u64 start, u64 end
     list:  u64 count, then u64 index per entry
u32  channel_axis
str  dtype stable name (for example "i4_packed:32")
str  quant scheme stable name ("none", "symmetric:<block>", "affine:<block>:<0|1>")
str  transform_id
```

`str` is a `u64` byte length followed by raw UTF-8 bytes; `32B` is the raw
digest. The domain tag itself is the ASCII prefix `SELVEDGE-CANONICAL-V1\0`
followed by a length-prefixed domain string. There is no floating point, no map
iteration and no platform-dependent integer width in this encoding.

A validator recomputes every `shard_id` from the record's semantic fields. A
stored id that does not match is `ShardIdMismatch`; duplicate ids are rejected.
Physical fields (`offset`, `length`, `sha256`, `alignment`, links, calibration)
are deliberately not part of the identity: they describe where the bytes are,
not what the shard is. Repacking changes the bytes and is represented by
`transform_id`, which *is* part of the identity.

## Validation pipeline

`validate_bundle(path, options)` runs in this order and returns the first typed
error:

1. file length is at least the header length;
2. file length does not exceed `options.max_bundle_bytes`;
3. magic and container version;
4. manifest length is below both the configured cap and the hard cap, and fits
   inside the file, before any allocation;
5. manifest bytes are UTF-8 and parse with no unknown fields;
6. manifest semantic validation: schema, bundle id, alignment, payload length,
   transforms, shard uniqueness, per-shard shape/channel/dtype/quantization
   consistency, alignment and in-payload bounds, content-derived ids, links;
7. extent overlap detection over the sorted extents;
8. `payload_offset + payload_len == file length` exactly;
9. expected model digest and compatibility id, when supplied;
10. per-shard SHA-256 according to the verification mode;
11. cancellation check.

Verification modes:

- `All`: every shard is hashed; the report lists each verified shard.
- `Selected(ids)`: only those ids are hashed; unknown ids are an error and the
  report lists the skipped shards.
- `StructuralOnly`: no payload hashes are computed. The report says
  `verify_mode: "structural_only"` and lists every shard as skipped. This mode
  is explicitly **not** integrity verification.

A hash mismatch, a malformed extent, a duplicate id, an unsupported dtype or a
version mismatch is always an error, never a warning.

## Sidecar

Publication optionally writes `<bundle_id>.manifest.json` beside the bundle.
The sidecar is advisory and contains: schema
`selvedge/bundle-sidecar/v1`, bundle id, bundle file name, whole-file SHA-256,
manifest SHA-256, model digest, payload length, alignment and shard count.
`validate_sidecar(bundle, sidecar)` recomputes all of these and rejects a stale
or transplanted sidecar. The in-file manifest is authoritative; the sidecar
exists so a hot-reload controller can detect a bundle it did not expect.

## Atomic publication and crash behavior

`publish_bundle` and `publish_bundle_from_reader`:

1. validate the manifest and every size limit;
2. refuse to replace an existing bundle or sidecar unless `force` is set;
3. write to a uniquely named temporary file in the destination directory;
4. write header, manifest, zero padding and exactly `payload_len` payload bytes,
   hashing everything as it is written;
5. fsync the temporary file;
6. when `verify_before_publish` is enabled (default), run the full validator on
   the temporary file — the exact bytes that are about to become visible;
7. publish without overwrite by hard-linking the temporary file into place, or
   by an atomic rename when `force` is set;
8. publish the sidecar the same way;
9. fsync the directory;
10. remove the temporary file on every failure path and leave an existing bundle
    untouched.

Consequences:

- A reader sees either the old complete bundle, no bundle at all, or the new
  complete bundle. It cannot see a partially written file, because the final
  name never refers to an incomplete inode.
- A crash before step 7 leaves only a hidden temporary file, which the writer
  removes on error. A crash between the bundle and sidecar renames leaves the
  new bundle with an old sidecar; the sidecar validator rejects that state
  loudly, and the bundle itself remains valid.
- `force` uses rename, which replaces atomically on the target filesystem. The
  writer has already validated the complete temporary file, so a visible forced
  replacement is a complete bundle.

## Versioning rules

- The magic is fixed. A change to it is a different container, not v2.
- The `u16` container version increments for any change to header layout, file
  layout or identity encoding that old readers cannot interpret. Old readers
  reject an unknown version; they never guess.
- The manifest has both a schema string and a `schema_version`. Adding a
  required field, changing a field's meaning, or changing validation semantics
  requires a new schema version. Unknown fields are already rejected, so a
  reader cannot silently ignore a new required field.
- The shard-id domain string is versioned independently
  (`selvedge/shard-id/v1`) and the model-digest domain likewise. A change to
  either encoding requires a new domain string; existing ids must not be reused
  under a changed algorithm.
- Compatibility ids are opaque labels owned by the converter. They are compared
  for equality, never parsed by the container layer.

## Alignment

The default alignment is 4096 bytes. It is a configuration value and a
benchmark target, not a claim that 4096 is optimal for every device. The format
requirement is that shard extent offsets and the payload length are multiples of
the declared alignment. The tracked fixture uses 512 bytes so the committed
artifact stays small; that is a fixture choice, not a recommendation.

Padding between extents and after the final shard is not covered by any shard
hash. It is zero-filled by the writer, and a modification confined to padding
is not detected by bundle validation. Integrity claims therefore apply to the
header, manifest semantics and shard extents, not to padding bytes.

## Threat model for untrusted artifacts

Validated:

- truncation, trailing bytes, wrong magic, version skew, wrong endianness;
- oversized manifests and manifests that do not fit the file, checked before
  allocation;
- malformed JSON, unknown fields, duplicate ids, non-content-derived ids;
- out-of-bounds or overlapping extents, misalignment, inconsistent
  shape/dtype/quantization byte counts;
- payload corruption through per-shard hashes;
- declared model digest and compatibility id mismatches, when an expectation is
  supplied;
- stale sidecars;
- path tricks in `bundle_id`: it is restricted to a safe character set, does
  not start with `.`, and file names are derived by the writer rather than
  supplied as paths.

Not provided:

- **Authenticity or signatures.** SHA-256 detects accidental corruption and
  naive tampering, but an attacker who can rewrite both the manifest and the
  payload can produce a self-consistent bundle with new hashes. Pin the model
  digest, revision and compatibility id from a trusted external source before
  loading untrusted artifacts.
- **Manifest authenticity.** The manifest is not self-hashed inside the bundle.
  The sidecar binds the manifest digest, but the sidecar is a separate file.
- **Padding integrity.** See above.
- **Semantic model correctness.** The format proves that the bytes are the bytes
  the manifest describes. It does not prove that the tensors implement the
  claimed architecture; that is Court A/E work.

## Bounds and resource behavior

- Manifest parsing is capped by configuration and by a hard ceiling, and the
  manifest buffer is allocated with `try_reserve_exact` only after the cap
  check.
- Shard reads in `BundleFile` require a caller-supplied byte cap and use
  `try_reserve_exact`.
- Hashing uses a fixed streaming buffer.
- Validation reads only the extents it is asked to hash; `StructuralOnly`
  performs no payload I/O.
- `MemoryBundle` encodes and validates through the same code paths as the
  on-disk writer and validator.

## Property and hostile-corpus testing

`crates/selvedge-format/tests/court_d.rs` contains named tests for every
failure mode above plus property tests:

- arbitrary byte strings never panic the validator;
- random single-byte mutations of a valid bundle never panic;
- a deterministic hostile corpus (payload flips in every shard, structural
  truncations, every header byte flipped, manifest corruption, trailing
  garbage) is always rejected.

A `cargo-fuzz` target is not checked in. If one is added, it should call
`selvedge_format::validate_bytes` on arbitrary input with the default limits,
and stay offline and OOM-bounded like the courts. The proptest harness already
covers the parser boundary deterministically and in the offline court.
