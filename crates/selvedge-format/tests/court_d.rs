//! Court D — storage integrity, atomic publication and crash behavior.
//!
//! Every failure mode named in the mandate has a named test here. The bundles
//! are built locally so the tests can construct deliberately invalid manifests
//! that the public writer would refuse to produce.

#![allow(clippy::unwrap_used)]

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use proptest::prelude::*;
use selvedge_core::{
    ChannelSet, DType, FORMAT_VERSION, ModelDigest, QuantScheme, Sha256Digest, ShardId,
    ShardIdentity, StorageOrder, TensorRole, align_up, bytes_for, element_count, testutil::TempDir,
};
use selvedge_format::FormatError;
use selvedge_format::catalog::{BundleCatalog, ReloadVerdict};
use selvedge_format::manifest::{
    AuxExtent, HEADER_LEN, MAGIC, MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION, Manifest,
    QuantizationSpec, ShardRecord,
};
use selvedge_format::memory::MemoryBundle;
use selvedge_format::reader::BundleFile;
use selvedge_format::sidecar::{read_sidecar, validate_sidecar};
use selvedge_format::validation::{
    BundleHeader, ValidationOptions, VerifyMode, validate_bundle, validate_bytes,
};
use selvedge_format::writer::{
    WriteOptions, encode_bundle_bytes, publish_bundle, publish_bundle_from_reader,
};

const ALIGNMENT: u64 = 512;

fn model() -> ModelDigest {
    ModelDigest::from_parts(
        Sha256Digest::of(b"court-d-model"),
        Some("court-d-rev".to_owned()),
        Some(Sha256Digest::of(b"court-d-tokenizer")),
    )
    .unwrap()
}

struct Built {
    manifest: Manifest,
    payload: Vec<u8>,
}

fn f32_bytes(seed: u8, elements: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity((elements * 4) as usize);
    for index in 0..elements {
        let value = ((index as f32) * 0.031 + f32::from(seed)) * 0.01;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn build_fixture() -> Built {
    let model = model();
    let mut payload = Vec::new();
    struct Pending {
        role: TensorRole,
        channels: ChannelSet,
        axis: u32,
        shape: Vec<u64>,
        dtype: DType,
        quant: Option<QuantizationSpec>,
        transform: &'static str,
        extent: Vec<u8>,
        offset: u64,
        id: ShardId,
    }
    let mut pending: Vec<Pending> = Vec::new();

    {
        let mut add = |role: TensorRole,
                       channels: ChannelSet,
                       axis: u32,
                       shape: Vec<u64>,
                       dtype: DType,
                       quant: Option<QuantizationSpec>,
                       transform: &'static str,
                       extent: Vec<u8>| {
            let identity = ShardIdentity {
                format_version: FORMAT_VERSION,
                model: model.clone(),
                role: role.clone(),
                layer: Some(0),
                channels: channels.clone(),
                channel_axis: axis,
                dtype: dtype.clone(),
                quant: quant.as_ref().map(QuantizationSpec::scheme),
                transform_id: transform.to_owned(),
            };
            let id = ShardId::derive(&identity).unwrap();
            pending.push(Pending {
                role,
                channels,
                axis,
                shape,
                dtype,
                quant,
                transform,
                extent,
                offset: 0,
                id,
            });
        };

        add(
            TensorRole::EmbedTokens,
            ChannelSet::range(0, 16).unwrap(),
            0,
            vec![16, 8],
            DType::F32,
            None,
            "identity",
            f32_bytes(1, 128),
        );
        add(
            TensorRole::AttnQ,
            ChannelSet::range(0, 8).unwrap(),
            0,
            vec![8, 8],
            DType::F32,
            None,
            "identity",
            f32_bytes(2, 64),
        );
        add(
            TensorRole::MlpGate,
            ChannelSet::range(0, 8).unwrap(),
            0,
            vec![8, 8],
            DType::F32,
            None,
            "identity",
            f32_bytes(3, 64),
        );
        // One I8 block-quantized shard: 64 weights + 2 f32 scales.
        let mut quantized = Vec::with_capacity(72);
        for index in 0..64_u64 {
            quantized.push(((index as i8).wrapping_mul(3)).to_le_bytes()[0]);
        }
        for index in 0..2 {
            quantized.extend_from_slice(&(0.01_f32 * (index as f32 + 1.0)).to_le_bytes());
        }
        add(
            TensorRole::MlpGate,
            ChannelSet::range(8, 16).unwrap(),
            0,
            vec![8, 8],
            DType::I8,
            Some(QuantizationSpec {
                block_size: 32,
                scales: Some(AuxExtent {
                    offset: 64,
                    length: 8,
                    dtype: DType::F32,
                }),
                zero_points: None,
            }),
            "i8-symmetric-block32",
            quantized,
        );
    }

    for shard in &mut pending {
        let offset = align_up(payload.len() as u64, ALIGNMENT).unwrap();
        payload.resize(usize::try_from(offset).unwrap(), 0);
        shard.offset = offset;
        payload.extend_from_slice(&shard.extent);
    }
    let payload_len = align_up(payload.len() as u64, ALIGNMENT).unwrap();
    payload.resize(usize::try_from(payload_len).unwrap(), 0);

    let records: Vec<ShardRecord> = pending
        .iter()
        .map(|shard| ShardRecord {
            shard_id: shard.id,
            role: shard.role.clone(),
            layer: Some(0),
            channels: shard.channels.clone(),
            channel_axis: shard.axis,
            shape: shard.shape.clone(),
            storage_order: StorageOrder::RowMajor,
            dtype: shard.dtype.clone(),
            quant: shard.quant.clone(),
            offset: shard.offset,
            length: shard.extent.len() as u64,
            sha256: Sha256Digest::of(&shard.extent),
            alignment: ALIGNMENT,
            compatibility_id: "court-d-v1".to_owned(),
            transform_id: shard.transform.to_owned(),
            co_access: Vec::new(),
            depends_on: Vec::new(),
            calibration: None,
        })
        .collect();

    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        bundle_id: "court-d".to_owned(),
        model,
        transforms: vec!["i8-symmetric-block32".to_owned(), "identity".to_owned()],
        alignment: ALIGNMENT,
        payload_len,
        created_by: "selvedge-format-tests".to_owned(),
        shards: records,
    };
    manifest.validate().unwrap();
    Built { manifest, payload }
}

fn encoded(built: &Built) -> Vec<u8> {
    encode_bundle_bytes(&built.manifest, &built.payload).unwrap()
}

fn options() -> ValidationOptions {
    ValidationOptions::default()
}

fn temp_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().starts_with(".selvedge-tmp-"))
                .unwrap_or(false)
        })
        .collect()
}

fn raw_with_manifest(manifest: &Manifest, payload: &[u8], version: u16) -> Vec<u8> {
    let manifest_bytes = serde_json::to_vec(manifest).unwrap();
    let payload_offset =
        align_up(HEADER_LEN + manifest_bytes.len() as u64, manifest.alignment).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&manifest_bytes);
    out.resize(usize::try_from(payload_offset).unwrap(), 0);
    out.extend_from_slice(payload);
    out
}

#[test]
fn valid_bundle_round_trips_and_reports_totals() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let report = validate_bytes(&bytes, &options()).unwrap();
    assert_eq!(report.bundle_id, "court-d");
    assert_eq!(report.shard_count, built.manifest.shards.len() as u64);
    assert_eq!(report.payload_len, built.manifest.payload_len);
    assert_eq!(
        report.total_shard_bytes,
        built.manifest.total_shard_bytes().unwrap()
    );
    assert_eq!(report.verified_shards.len(), built.manifest.shards.len());
    assert!(report.skipped_shards.is_empty());
    assert_eq!(report.verify_mode, "all");
    assert_eq!(report.compatibility_ids, vec!["court-d-v1".to_owned()]);
}

#[test]
fn bit_flip_in_payload_fails_hash_verification() {
    let built = build_fixture();
    let mut bytes = encoded(&built);
    let report = validate_bytes(&bytes, &options()).unwrap();
    let first = &built.manifest.shards[0];
    let index = usize::try_from(report.payload_offset + first.offset + 3).unwrap();
    bytes[index] ^= 0x01;
    match validate_bytes(&bytes, &options()) {
        Err(FormatError::HashMismatch { shard_id, .. }) => {
            assert_eq!(shard_id, first.shard_id);
        }
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

#[test]
fn corrupted_manifest_bytes_are_rejected() {
    let built = build_fixture();
    let mut bytes = encoded(&built);
    bytes[HEADER_LEN as usize] = 0xFF;
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::ManifestNotUtf8)
    ));
}

#[test]
fn truncated_header_is_rejected() {
    assert!(matches!(
        validate_bytes(&[0_u8; 10], &options()),
        Err(FormatError::HeaderTruncated { .. })
    ));
}

#[test]
fn truncated_manifest_is_rejected() {
    let built = build_fixture();
    let bytes = encoded(&built);
    assert!(matches!(
        validate_bytes(&bytes[..20], &options()),
        Err(FormatError::ManifestOutOfBounds { .. })
    ));
}

#[test]
fn truncated_payload_is_rejected() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let truncated = &bytes[..bytes.len() - 16];
    assert!(matches!(
        validate_bytes(truncated, &options()),
        Err(FormatError::FileLengthMismatch { .. })
    ));
}

#[test]
fn offset_outside_payload_is_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[0].offset = manifest.payload_len;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::ExtentOutOfBounds { .. })
    ));
}

#[test]
fn length_corruption_is_rejected_with_byte_length_mismatch() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[0].length += ALIGNMENT;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::ByteLengthMismatch { .. })
    ));
}

#[test]
fn overlapping_extents_are_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[1].offset = manifest.shards[0].offset;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::OverlappingExtents { .. })
    ));
}

#[test]
fn duplicate_shard_ids_are_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[1].shard_id = manifest.shards[0].shard_id;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::DuplicateShardId(_))
    ));
}

#[test]
fn stored_id_that_is_not_content_derived_is_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[0].role = TensorRole::Other("audit.reclassified".to_owned());
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::ShardIdMismatch { .. })
    ));
}

#[test]
fn wrong_endianness_version_is_rejected() {
    let built = build_fixture();
    let mut bytes = encoded(&built);
    bytes[8..10].copy_from_slice(&1_u16.to_be_bytes());
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::UnsupportedVersion { found: 256 })
    ));
}

#[test]
fn version_skew_is_rejected() {
    let built = build_fixture();
    let bytes = raw_with_manifest(&built.manifest, &built.payload, 2);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::UnsupportedVersion { found: 2 })
    ));
}

#[test]
fn unsupported_dtype_is_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[0].dtype = DType::Unsupported("q4_k".to_owned());
    manifest.shards[0].quant = None;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::UnsupportedDType { .. })
    ));
}

#[test]
fn oversized_manifest_is_rejected_before_allocation() {
    let mut header = Vec::new();
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(&1_u16.to_le_bytes());
    header.extend_from_slice(&(9_u64 * 1024 * 1024).to_le_bytes());
    assert!(matches!(
        validate_bytes(&header, &options()),
        Err(FormatError::ManifestTooLarge { .. })
    ));

    let built = build_fixture();
    let bytes = encoded(&built);
    let capped = ValidationOptions {
        max_manifest_bytes: 64,
        ..ValidationOptions::default()
    };
    assert!(matches!(
        validate_bytes(&bytes, &capped),
        Err(FormatError::ManifestTooLarge { .. })
    ));
}

#[test]
fn file_length_mismatch_is_rejected() {
    let built = build_fixture();
    let mut bytes = encoded(&built);
    bytes.push(0);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::FileLengthMismatch { .. })
    ));
}

#[test]
fn alignment_violation_is_rejected() {
    let built = build_fixture();
    let mut manifest = built.manifest.clone();
    manifest.shards[0].offset = 1;
    let bytes = raw_with_manifest(&manifest, &built.payload, 1);
    assert!(matches!(
        validate_bytes(&bytes, &options()),
        Err(FormatError::AlignmentViolation { .. })
    ));
}

#[test]
fn model_digest_and_compatibility_mismatch_are_rejected() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let digest_options = ValidationOptions {
        expected_model_digest: Some(Sha256Digest::of(b"another model")),
        ..ValidationOptions::default()
    };
    assert!(matches!(
        validate_bytes(&bytes, &digest_options),
        Err(FormatError::ModelDigestMismatch { .. })
    ));
    let compatibility_options = ValidationOptions {
        expected_compatibility_id: Some("not-this-one".to_owned()),
        ..ValidationOptions::default()
    };
    assert!(matches!(
        validate_bytes(&bytes, &compatibility_options),
        Err(FormatError::CompatibilityMismatch { .. })
    ));
}

#[test]
fn structural_only_skips_every_hash() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let report = validate_bytes(&bytes, &ValidationOptions::structural()).unwrap();
    assert!(report.verified_shards.is_empty());
    assert_eq!(report.skipped_shards.len(), built.manifest.shards.len());
    assert_eq!(report.verify_mode, "structural_only");
}

#[test]
fn selected_verification_verifies_only_selected_shards() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let selected_id = built.manifest.shards[0].shard_id;
    let selected = ValidationOptions {
        verify: VerifyMode::Selected(vec![selected_id]),
        ..ValidationOptions::default()
    };
    let report = validate_bytes(&bytes, &selected).unwrap();
    assert_eq!(report.verified_shards, vec![selected_id]);
    assert_eq!(report.skipped_shards.len(), built.manifest.shards.len() - 1);
    assert_eq!(report.verify_mode, "selected");
}

#[test]
fn unknown_selected_shard_is_rejected() {
    let built = build_fixture();
    let bytes = encoded(&built);
    let unknown = ShardId::from_digest(Sha256Digest::of(b"absent"));
    let options = ValidationOptions {
        verify: VerifyMode::Selected(vec![unknown]),
        ..ValidationOptions::default()
    };
    assert!(matches!(
        validate_bytes(&bytes, &options),
        Err(FormatError::UnknownSelectedShard(_))
    ));
}

#[test]
fn atomic_publication_is_immutable_without_force() {
    let dir = TempDir::new("court-d-publish").unwrap();
    let built = build_fixture();
    let published = publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions::default(),
    )
    .unwrap();
    let original = std::fs::read(&published.path).unwrap();
    let sidecar = published.sidecar.clone().unwrap();
    validate_sidecar(&published.path, &sidecar).unwrap();
    assert!(matches!(
        publish_bundle(
            dir.path(),
            &built.manifest,
            &built.payload,
            &WriteOptions::default()
        ),
        Err(FormatError::ImmutableBundleExists(_))
    ));
    assert_eq!(std::fs::read(&published.path).unwrap(), original);
    publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions {
            force: true,
            ..WriteOptions::default()
        },
    )
    .unwrap();
    validate_sidecar(&published.path, &sidecar).unwrap();
}

struct FailingReader {
    data: Vec<u8>,
    position: usize,
    fail_after: usize,
}

impl Read for FailingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.fail_after {
            return Err(io::Error::other("injected payload failure"));
        }
        let remaining = (self.fail_after - self.position)
            .min(self.data.len() - self.position)
            .min(buffer.len());
        if remaining == 0 {
            return Err(io::Error::other("injected payload failure"));
        }
        buffer[..remaining].copy_from_slice(&self.data[self.position..self.position + remaining]);
        self.position += remaining;
        Ok(remaining)
    }
}

#[test]
fn interrupted_publication_leaves_no_partial_bundle() {
    let dir = TempDir::new("court-d-interrupt").unwrap();
    let built = build_fixture();
    let mut reader = FailingReader {
        data: built.payload.clone(),
        position: 0,
        fail_after: 10,
    };
    let error = publish_bundle_from_reader(
        dir.path(),
        &built.manifest,
        &mut reader,
        &WriteOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(error, FormatError::Io(_)));
    assert!(!dir.path().join("court-d.slv").exists());
    assert!(!dir.path().join("court-d.manifest.json").exists());
    assert!(temp_files(dir.path()).is_empty());
}

#[test]
fn failed_forced_publication_preserves_the_existing_bundle() {
    let dir = TempDir::new("court-d-preserve").unwrap();
    let built = build_fixture();
    let published = publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions::default(),
    )
    .unwrap();
    let original = std::fs::read(&published.path).unwrap();
    let mut reader = FailingReader {
        data: built.payload.clone(),
        position: 0,
        fail_after: 10,
    };
    assert!(
        publish_bundle_from_reader(
            dir.path(),
            &built.manifest,
            &mut reader,
            &WriteOptions {
                force: true,
                ..WriteOptions::default()
            },
        )
        .is_err()
    );
    assert_eq!(std::fs::read(&published.path).unwrap(), original);
    assert!(temp_files(dir.path()).is_empty());
}

#[test]
fn short_and_long_payload_streams_are_rejected() {
    let dir = TempDir::new("court-d-streams").unwrap();
    let built = build_fixture();
    let short = built.payload[..built.payload.len() - 1].to_vec();
    let mut short_reader = io::Cursor::new(short);
    assert!(matches!(
        publish_bundle_from_reader(
            dir.path(),
            &built.manifest,
            &mut short_reader,
            &WriteOptions::default()
        ),
        Err(FormatError::PayloadShort { .. })
    ));
    let mut long = built.payload.clone();
    long.push(0);
    let mut long_reader = io::Cursor::new(long);
    assert!(matches!(
        publish_bundle_from_reader(
            dir.path(),
            &built.manifest,
            &mut long_reader,
            &WriteOptions::default()
        ),
        Err(FormatError::PayloadLong { .. })
    ));
    assert!(!dir.path().join("court-d.slv").exists());
}

#[test]
fn stale_sidecar_is_rejected() {
    let dir = TempDir::new("court-d-sidecar").unwrap();
    let built = build_fixture();
    let published = publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions::default(),
    )
    .unwrap();
    let sidecar = published.sidecar.clone().unwrap();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    value["shard_count"] = serde_json::json!(0);
    std::fs::write(&sidecar, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(
        read_sidecar(&sidecar).is_err() || validate_sidecar(&published.path, &sidecar).is_err()
    );
    match validate_sidecar(&published.path, &sidecar) {
        Err(FormatError::SidecarMismatch { .. }) => {}
        other => panic!("expected SidecarMismatch, got {other:?}"),
    }
}

#[test]
fn concurrent_readers_never_observe_a_torn_bundle() {
    let dir = TempDir::new("court-d-concurrent").unwrap();
    let built = build_fixture();
    let published = publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions::default(),
    )
    .unwrap();
    let path = published.path.clone();
    let mut readers = Vec::new();
    for _ in 0..2 {
        let path = path.clone();
        readers.push(std::thread::spawn(move || {
            let options = ValidationOptions::default();
            for _ in 0..10 {
                validate_bundle(&path, &options)
                    .map_err(|error| format!("reader observed {error}"))
                    .unwrap();
            }
        }));
    }
    for _ in 0..5 {
        publish_bundle(
            dir.path(),
            &built.manifest,
            &built.payload,
            &WriteOptions {
                force: true,
                ..WriteOptions::default()
            },
        )
        .unwrap();
    }
    for reader in readers {
        reader.join().unwrap();
    }
    validate_bundle(&path, &ValidationOptions::default()).unwrap();
}

#[test]
fn cancellable_read_aborts_and_the_reader_recovers() {
    let dir = TempDir::new("court-d-cancel").unwrap();
    let built = build_fixture();
    let published = publish_bundle(
        dir.path(),
        &built.manifest,
        &built.payload,
        &WriteOptions::default(),
    )
    .unwrap();
    let first = built.manifest.shards[0].clone();
    let mut bundle = BundleFile::open(&published.path).unwrap();
    let cancel = AtomicBool::new(true);
    assert!(matches!(
        bundle.read_shard_cancellable(&first.shard_id, 1 << 20, Some(&cancel)),
        Err(FormatError::Cancelled)
    ));
    cancel.store(false, Ordering::Relaxed);
    let bytes = bundle.read_shard(&first.shard_id, 1 << 20).unwrap();
    assert_eq!(bytes.len() as u64, first.length);
    assert_eq!(Sha256Digest::of(&bytes), first.sha256);
}

#[test]
fn hot_reload_rejects_a_conflicting_bundle_id() {
    let built = build_fixture();
    let mut catalog = BundleCatalog::new();
    assert_eq!(
        catalog.offer(&built.manifest).unwrap(),
        ReloadVerdict::AcceptedNew
    );
    assert_eq!(
        catalog.offer(&built.manifest).unwrap(),
        ReloadVerdict::AcceptedIdentical
    );
    let mut conflicting = built.manifest.clone();
    conflicting.bundle_id = built.manifest.bundle_id.clone();
    conflicting.created_by = "other-writer".to_owned();
    conflicting.validate().unwrap();
    match catalog.offer(&conflicting).unwrap() {
        ReloadVerdict::RejectedConflict { existing, incoming } => {
            assert_eq!(existing.bundle_id, incoming.bundle_id);
            assert_ne!(existing.manifest_sha256, incoming.manifest_sha256);
        }
        other => panic!("expected RejectedConflict, got {other:?}"),
    }
    assert_eq!(catalog.len(), 1);
    assert_eq!(
        catalog.get("court-d").unwrap(),
        &selvedge_format::CatalogEntry::from_manifest(&built.manifest).unwrap()
    );
}

#[test]
fn memory_bundle_shares_the_validation_path() {
    let built = build_fixture();
    let memory =
        MemoryBundle::from_parts(built.manifest.clone(), &built.payload, &options()).unwrap();
    let first = built.manifest.shards[0].clone();
    assert_eq!(
        memory.read_shard(&first.shard_id).unwrap(),
        &built.payload[usize::try_from(first.offset).unwrap()
            ..usize::try_from(first.offset + first.length).unwrap()]
    );
    assert_eq!(
        memory.payload().unwrap().len() as u64,
        built.manifest.payload_len
    );
    let report = memory.validate(&options()).unwrap();
    assert_eq!(report.shard_count, built.manifest.shards.len() as u64);

    let mut corrupt = memory.bytes().to_vec();
    let payload_offset = usize::try_from(memory.payload_offset()).unwrap();
    corrupt[payload_offset + 5] ^= 0x01;
    assert!(MemoryBundle::from_validated_bytes(corrupt, &options()).is_err());
}

#[test]
fn header_encoding_round_trips() {
    let header = BundleHeader {
        version: 1,
        manifest_len: 1234,
    };
    let bytes = header.encode();
    let mut cursor = io::Cursor::new(bytes);
    assert_eq!(BundleHeader::read_from(&mut cursor).unwrap(), header);
}

#[test]
fn hostile_corpus_is_rejected_without_panicking() {
    let built = build_fixture();
    let valid = encoded(&built);
    let report = validate_bytes(&valid, &options()).unwrap();
    let mut mutations: Vec<Vec<u8>> = Vec::new();
    // Payload bytes of every shard: must be hash-detected.
    for shard in &built.manifest.shards {
        let mut bytes = valid.clone();
        let index = usize::try_from(report.payload_offset + shard.offset).unwrap();
        bytes[index] ^= 0x01;
        mutations.push(bytes);
    }
    // Truncations at every structural boundary.
    for cut in [0, 5, 17, 18, 19, 100] {
        mutations.push(valid[..cut.min(valid.len())].to_vec());
    }
    // Header and manifest byte corruption.
    for index in 0..18 {
        let mut bytes = valid.clone();
        bytes[index] ^= 0xFF;
        mutations.push(bytes);
    }
    let mut bytes = valid.clone();
    bytes[HEADER_LEN as usize + 2] ^= 0xFF;
    mutations.push(bytes);
    // Trailing garbage.
    let mut bytes = valid.clone();
    bytes.extend_from_slice(&[0xAA; 7]);
    mutations.push(bytes);

    for mutation in mutations {
        if let Ok(report) = validate_bytes(&mutation, &options()) {
            panic!(
                "hostile mutation accepted as valid (bundle {})",
                report.bundle_id
            );
        }
    }
}

fn assert_fixture_types() {
    // Keep the helper's byte arithmetic honest with the shared core functions.
    for shard in build_fixture().manifest.shards {
        let elements = element_count(&shard.shape).unwrap();
        let weight = bytes_for(elements, &shard.dtype).unwrap();
        assert!(weight <= shard.length);
        if let DType::I8 = shard.dtype {
            assert!(shard.quant.is_some());
            assert_eq!(
                shard.quant.as_ref().unwrap().scheme(),
                QuantScheme::Symmetric { block_size: 32 }
            );
        }
    }
}

#[test]
fn fixture_helper_types_are_consistent() {
    assert_fixture_types();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn arbitrary_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..8192)) {
        let _ = validate_bytes(&data, &ValidationOptions::default());
    }

    #[test]
    fn mutations_of_a_valid_bundle_never_panic(index in any::<prop::sample::Index>(), replacement in any::<u8>()) {
        let built = build_fixture();
        let mut bytes = encoded(&built);
        if bytes.is_empty() {
            return Ok(());
        }
        let position = index.index(bytes.len());
        bytes[position] = replacement;
        let _ = validate_bytes(&bytes, &ValidationOptions::default());
    }
}

/// Ensure the writing helper is exercised for trait visibility.
#[allow(dead_code)]
fn _write_probe(mut writer: impl Write) -> io::Result<()> {
    writer.write_all(b"probe")
}
