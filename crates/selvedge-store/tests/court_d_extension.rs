//! Court D extensions for the storage engine.
//!
//! Every scenario runs on a real, atomically published bundle produced by
//! `selvedge-fixtures`; nothing here mocks the reader. The scenarios cover
//! cancellation under load, backpressure, checksum-failure recovery,
//! cache-budget refusal, concurrent prefetch and read, prefetch of a shard
//! outside the execution scope, a small-cache stress pass, hot-reload
//! rejection, the metrics schema, and the typed exact-mode invariant.

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::collections::BTreeSet;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use selvedge_core::testutil::TempDir;
use selvedge_fixtures::publish_tiny_bundle;
use selvedge_format::{BundleCatalog, BundleFile, Manifest, ReloadVerdict};
use selvedge_store::*;

struct Fixture {
    _dir: TempDir,
    path: PathBuf,
    manifest: Manifest,
    payload_offset: u64,
    ids: Vec<selvedge_core::ShardId>,
    scope: ExecutionScope,
}

fn fixture() -> Fixture {
    let dir = TempDir::new("store-court-d").unwrap();
    let published = publish_tiny_bundle(dir.path(), false).unwrap();
    let path = published.path.clone();
    let bundle = BundleFile::open(&path).unwrap();
    let manifest = bundle.manifest().clone();
    let payload_offset = bundle.payload_offset();
    let ids: Vec<_> = manifest.shards.iter().map(|shard| shard.shard_id).collect();
    let scope = ExecutionScope::exact(ids.iter().copied()).unwrap();
    Fixture {
        _dir: dir,
        path,
        manifest,
        payload_offset,
        ids,
        scope,
    }
}

fn base_config() -> StoreConfig {
    StoreConfig {
        cache_bytes: 0,
        policy: PolicyKind::None,
        max_shard_bytes: 1 << 20,
        verification: VerificationMode::Always,
        queue: QueueLimits {
            max_pending: 64,
            max_in_flight: 16,
            max_in_flight_bytes: 64 << 20,
        },
        max_run_bytes: 4 << 20,
        coalesce_gap_bytes: 0,
        metrics: None,
    }
}

fn config(policy: PolicyKind, cache_bytes: u64) -> StoreConfig {
    let mut config = base_config();
    config.policy = policy;
    config.cache_bytes = cache_bytes;
    config
}

fn coalescing_config(policy: PolicyKind, cache_bytes: u64, run_bytes: u64) -> StoreConfig {
    let mut config = config(policy, cache_bytes);
    config.max_shard_bytes = 4096;
    config.max_run_bytes = run_bytes.max(config.max_shard_bytes);
    config
}

fn largest_shard(manifest: &Manifest) -> u64 {
    manifest
        .shards
        .iter()
        .map(|shard| shard.length)
        .max()
        .unwrap_or(0)
}

fn payload_bytes(manifest: &Manifest) -> u64 {
    manifest.shards.iter().map(|shard| shard.length).sum()
}

fn expected_bytes(fixture: &Fixture, shard: &selvedge_core::ShardId) -> Vec<u8> {
    let record = fixture.manifest.shard(shard).unwrap();
    let file = std::fs::File::open(&fixture.path).unwrap();
    let mut buffer = vec![0_u8; record.length as usize];
    file.read_exact_at(&mut buffer, fixture.payload_offset + record.offset)
        .unwrap();
    buffer
}

fn read_shard(
    store: &ShardStore,
    scope: &ExecutionScope,
    id: &selvedge_core::ShardId,
    buffer: &mut [u8],
) -> ReadReport {
    let len = store.manifest().shard(id).unwrap().length as usize;
    store.read_required(scope, id, &mut buffer[..len]).unwrap()
}

#[test]
fn exact_scope_refuses_duplicates_empty_and_out_of_scope_reads() {
    let fixture = fixture();
    assert!(ExecutionScope::exact(Vec::new()).is_err());
    let duplicate = vec![fixture.ids[0], fixture.ids[0]];
    assert!(ExecutionScope::exact(duplicate).is_err());

    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let narrow = ExecutionScope::exact([fixture.ids[0]]).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let error = store
        .read_required(&narrow, &fixture.ids[1], &mut buffer)
        .unwrap_err();
    assert!(matches!(error, StoreError::NotInScope { .. }));
    let outcome = store
        .prefetch_advisory(&narrow, &[PrefetchHint::new(fixture.ids[1])])
        .unwrap();
    assert_eq!(
        outcome.rejected,
        vec![(fixture.ids[1], PrefetchRejection::NotInScope)]
    );
    assert!(outcome.admitted.is_empty());
}

#[test]
fn buffer_size_mismatch_is_refused_before_any_io() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let record = fixture.manifest.shard(&fixture.ids[0]).unwrap();
    let mut short = vec![0_u8; record.length as usize - 1];
    let error = store
        .read_required(&fixture.scope, &fixture.ids[0], &mut short)
        .unwrap_err();
    assert!(matches!(error, StoreError::BufferSize { .. }));
    assert_eq!(store.stats().reads, 0);
}

#[test]
fn cancel_under_load_releases_every_reservation() {
    let queue = TransferQueue::new(QueueLimits {
        max_pending: 8,
        max_in_flight: 4,
        max_in_flight_bytes: 8,
    })
    .unwrap();
    let shard = selvedge_core::ShardId::from_digest(selvedge_core::Sha256Digest::of(b"q"));
    let mut dropped_tickets = 0;
    for _ in 0..4 {
        let ticket = queue
            .submit(shard, 2, Tier::Nvme, Tier::HostRam, false, None)
            .unwrap();
        drop(ticket);
        dropped_tickets += 1;
    }
    let mut in_flight = Vec::new();
    for _ in 0..2 {
        let ticket = queue
            .submit(shard, 2, Tier::Nvme, Tier::HostRam, false, None)
            .unwrap();
        in_flight.push(ticket.start());
    }
    assert_eq!(queue.snapshot().in_flight_bytes, 4);
    drop(in_flight);
    let snapshot = queue.snapshot();
    assert_eq!(snapshot.in_flight_bytes, 0);
    assert_eq!(snapshot.stats.cancelled, dropped_tickets + 2);
    assert!(snapshot.pending.is_empty());
    assert!(snapshot.in_flight.is_empty());
}

#[test]
fn backpressure_is_caller_visible_and_prefetch_only_by_bytes() {
    let queue = TransferQueue::new(QueueLimits {
        max_pending: 2,
        max_in_flight: 2,
        max_in_flight_bytes: 4,
    })
    .unwrap();
    let shard = selvedge_core::ShardId::from_digest(selvedge_core::Sha256Digest::of(b"b"));
    let first = queue
        .submit(shard, 2, Tier::Nvme, Tier::HostRam, false, None)
        .unwrap();
    let second = queue
        .submit(shard, 2, Tier::Nvme, Tier::HostRam, false, None)
        .unwrap();
    let error = queue
        .submit(shard, 2, Tier::Nvme, Tier::HostRam, false, None)
        .unwrap_err();
    assert!(matches!(error, StoreError::Backpressure { .. }));
    drop(first);
    drop(second);
    let required = queue
        .submit(shard, 8, Tier::Nvme, Tier::HostRam, false, None)
        .unwrap();
    let in_flight = required.start();
    let speculative = queue
        .submit(shard, 8, Tier::Nvme, Tier::HostRam, true, None)
        .unwrap_err();
    assert!(matches!(speculative, StoreError::Backpressure { .. }));
    drop(in_flight);
    assert_eq!(queue.snapshot().in_flight_bytes, 0);
}

#[test]
fn gpu_staging_is_declared_but_refused() {
    let queue = TransferQueue::new(QueueLimits {
        max_pending: 2,
        max_in_flight: 2,
        max_in_flight_bytes: 64,
    })
    .unwrap();
    let shard = selvedge_core::ShardId::from_digest(selvedge_core::Sha256Digest::of(b"g"));
    let error = queue
        .submit(shard, 8, Tier::HostRam, Tier::GpuStaging, false, None)
        .unwrap_err();
    assert!(matches!(error, StoreError::Config { .. }));
    assert!(!Tier::GpuStaging.implemented());
}

#[test]
fn checksum_failure_fails_closed_and_later_reads_recover() {
    let fixture = fixture();
    let victim = fixture.ids[0];
    let record = fixture.manifest.shard(&victim).unwrap().clone();
    let clean = expected_bytes(&fixture, &victim);
    {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fixture.path)
            .unwrap();
        let mut byte = [0_u8; 1];
        file.read_exact_at(&mut byte, fixture.payload_offset + record.offset)
            .unwrap();
        byte[0] ^= 0xff;
        file.write_all_at(&byte, fixture.payload_offset + record.offset)
            .unwrap();
    }
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let error = store
        .read_required(
            &fixture.scope,
            &victim,
            &mut buffer[..record.length as usize],
        )
        .unwrap_err();
    match error {
        StoreError::Checksum { shard, .. } => assert_eq!(shard, victim),
        other => panic!("expected a checksum error, got {other:?}"),
    }
    assert_eq!(store.stats().checksum_failures, 1);
    let other = fixture.ids[1];
    read_shard(&store, &fixture.scope, &other, &mut buffer);
    {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fixture.path)
            .unwrap();
        file.write_all_at(&clean[..1], fixture.payload_offset + record.offset)
            .unwrap();
    }
    let recovered = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    read_shard(&recovered, &fixture.scope, &victim, &mut buffer);
    let len = record.length as usize;
    assert_eq!(&buffer[..len], &clean[..]);
    assert_eq!(recovered.stats().checksum_failures, 0);
}

#[test]
fn pinned_cache_refuses_a_second_shard_instead_of_overflowing() {
    let fixture = fixture();
    let first = fixture.ids[0];
    let second = fixture.ids[1];
    let budget = fixture.manifest.shard(&first).unwrap().length;
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Pinned, budget)).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    read_shard(&store, &fixture.scope, &first, &mut buffer);
    let error = store
        .read_required(
            &fixture.scope,
            &second,
            &mut buffer[..fixture.manifest.shard(&second).unwrap().length as usize],
        )
        .unwrap_err();
    assert!(matches!(error, StoreError::CacheBudgetExceeded { .. }));
    assert_eq!(store.stats().cache_refusals, 1);
    let hit = read_shard(&store, &fixture.scope, &first, &mut buffer);
    assert!(hit.hit);
    assert_eq!(hit.source, ReadSource::HostRam);
    assert!(store.pin(&first));
    assert_eq!(store.cache_stats().pinned_entries, 1);
}

#[test]
fn lru_reuses_a_repeated_shard_and_evicts() {
    let fixture = fixture();
    let capacity = largest_shard(&fixture.manifest) * 2;
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Lru, capacity)).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let mut hits = 0_u64;
    for _pass in 0..3 {
        for id in fixture.scope.ids() {
            // Read each shard twice in a row: the second read must hit.
            let first = read_shard(&store, &fixture.scope, id, &mut buffer);
            assert!(!first.hit);
            let second = read_shard(&store, &fixture.scope, id, &mut buffer);
            assert!(second.hit);
            assert_eq!(second.bytes_used, first.bytes_used);
            hits += 1;
        }
    }
    let stats = store.stats();
    assert_eq!(stats.hits, hits);
    assert_eq!(stats.reads, stats.hits + stats.misses);
    assert_eq!(stats.bytes_used, stats.bytes_completed);
    assert!(store.cache_stats().evictions > 0);
    assert!(store.cache_stats().used_bytes <= store.cache_stats().capacity_bytes);
}

#[test]
fn concurrent_readers_and_one_prefetcher_never_observe_torn_data() {
    let fixture = fixture();
    let shard_bytes = largest_shard(&fixture.manifest);
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Lru, shard_bytes * 3)).unwrap();
    let expected: Vec<(selvedge_core::ShardId, Vec<u8>)> = fixture
        .ids
        .iter()
        .map(|id| (*id, expected_bytes(&fixture, id)))
        .collect();
    let stop = Arc::new(AtomicBool::new(false));
    let scope = fixture.scope.clone();
    let ids = fixture.ids.clone();
    let store_ref = &store;
    let expected_ref = &expected;
    let scope_ref = &scope;
    let stop_ref = &stop;
    let mut checks = 0_u64;
    std::thread::scope(|threads| {
        let mut handles = Vec::new();
        for _reader in 0..4 {
            let ids = &ids;
            handles.push(threads.spawn(move || {
                let mut buffer = vec![0_u8; shard_bytes as usize];
                let mut done = 0_u64;
                for _round in 0..12 {
                    for id in ids {
                        let record = store_ref.manifest().shard(id).unwrap();
                        let slice = &mut buffer[..record.length as usize];
                        let report = store_ref
                            .read_required(scope_ref, id, slice)
                            .expect("concurrent read must succeed");
                        assert!(report.verified);
                        let want = expected_ref
                            .iter()
                            .find(|(candidate, _)| candidate == id)
                            .map(|(_, bytes)| bytes)
                            .unwrap();
                        assert_eq!(&slice[..], &want[..], "torn read of {id:?}");
                        done += 1;
                    }
                }
                done
            }));
        }
        let prefetcher = threads.spawn(move || {
            let mut rounds = 0_u64;
            while !stop_ref.load(Ordering::Relaxed) {
                let hints: Vec<PrefetchHint> = scope_ref
                    .ids()
                    .iter()
                    .copied()
                    .map(PrefetchHint::new)
                    .collect();
                store_ref.prefetch_advisory(scope_ref, &hints).unwrap();
                rounds += 1;
            }
            rounds
        });
        for handle in handles {
            checks += handle.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        assert!(prefetcher.join().unwrap() > 0);
    });
    assert!(checks > 0);
    let stats = store.stats();
    assert_eq!(stats.checksum_failures, 0);
    assert!(stats.prefetch_requests > 0);
}

#[test]
fn prefetch_of_an_unknown_shard_id_is_rejected() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Lru, 1 << 20)).unwrap();
    let bogus = selvedge_core::ShardId::from_digest(selvedge_core::Sha256Digest::of(b"no-such"));
    let outcome = store
        .prefetch_advisory(&fixture.scope, &[PrefetchHint::new(bogus)])
        .unwrap();
    assert_eq!(
        outcome.rejected,
        vec![(bogus, PrefetchRejection::NotInScope)]
    );
    assert_eq!(outcome.runs, 0);

    let widened = ExecutionScope::exact(fixture.ids.iter().copied().chain([bogus])).unwrap();
    let outcome = store
        .prefetch_advisory(&widened, &[PrefetchHint::new(bogus)])
        .unwrap();
    assert_eq!(
        outcome.rejected,
        vec![(bogus, PrefetchRejection::UnknownShard)]
    );
}

#[test]
fn prefetch_coalesces_adjacent_extents_without_changing_execution() {
    let fixture = fixture();
    let max_shard = largest_shard(&fixture.manifest);
    let store = ShardStore::open(
        &fixture.path,
        coalescing_config(
            PolicyKind::Lru,
            payload_bytes(&fixture.manifest),
            max_shard * 4,
        ),
    )
    .unwrap();
    let hints: Vec<PrefetchHint> = fixture
        .scope
        .ids()
        .iter()
        .copied()
        .map(PrefetchHint::new)
        .collect();
    let outcome = store.prefetch_advisory(&fixture.scope, &hints).unwrap();
    assert_eq!(outcome.admitted.len(), fixture.ids.len());
    assert_eq!(outcome.rejected.len(), 0);
    assert!(outcome.runs < fixture.ids.len() as u64);
    assert!(outcome.reads_avoided_by_coalescing > 0);
    let mut buffer = vec![0_u8; max_shard as usize];
    for id in fixture.scope.ids() {
        let report = read_shard(&store, &fixture.scope, id, &mut buffer);
        assert!(report.hit, "prefetched shard {id:?} should be resident");
        assert_eq!(report.source, ReadSource::HostRam);
    }
    assert_eq!(store.stats().prefetch_bytes_used, store.stats().bytes_used);
}

#[test]
fn physical_order_and_plan_order_deliver_the_same_set() {
    let fixture = fixture();
    let shard_bytes = largest_shard(&fixture.manifest);
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let mut buffer = vec![0_u8; shard_bytes as usize];
    let plan = store
        .read_scope(&fixture.scope, &mut buffer, ReadOrder::Plan)
        .unwrap();
    let physical = store
        .read_scope(&fixture.scope, &mut buffer, ReadOrder::Physical)
        .unwrap();
    assert_eq!(plan.shards, physical.shards);
    assert_eq!(plan.bytes_used, physical.bytes_used);
    assert_eq!(plan.errors, 0);
    assert_eq!(physical.errors, 0);
    assert_eq!(plan.bytes_used, payload_bytes(&fixture.manifest));
}

#[test]
fn coalesced_pass_verifies_every_shard_and_counts_device_bytes() {
    let fixture = fixture();
    let max_shard = largest_shard(&fixture.manifest);
    let store = ShardStore::open(
        &fixture.path,
        coalescing_config(PolicyKind::None, 0, max_shard * 4),
    )
    .unwrap();
    let report = store.read_scope_coalesced(&fixture.scope).unwrap();
    assert_eq!(report.shards, fixture.ids.len() as u64);
    assert_eq!(report.errors, 0);
    let expected = payload_bytes(&fixture.manifest);
    assert_eq!(report.bytes_used, expected);
    assert_eq!(store.stats().device_bytes, expected);
    assert_eq!(store.stats().checksum_failures, 0);
}

#[test]
fn hot_reload_rejects_a_conflicting_bundle_and_the_store_is_unchanged() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Lru, 1 << 20)).unwrap();
    let mut catalog = BundleCatalog::new();
    assert!(matches!(
        catalog.offer(&fixture.manifest).unwrap(),
        ReloadVerdict::AcceptedNew
    ));
    assert!(matches!(
        catalog.offer(&fixture.manifest).unwrap(),
        ReloadVerdict::AcceptedIdentical
    ));
    let mut mutated = fixture.manifest.clone();
    mutated.created_by = "someone-else".to_owned();
    let verdict = catalog.offer(&mutated).unwrap();
    assert!(matches!(verdict, ReloadVerdict::RejectedConflict { .. }));
    assert!(!verdict.is_accepted());
    let shard = fixture.ids[0];
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    read_shard(&store, &fixture.scope, &shard, &mut buffer);
    let len = fixture.manifest.shard(&shard).unwrap().length as usize;
    assert_eq!(&buffer[..len], &expected_bytes(&fixture, &shard)[..]);
}

#[test]
fn metrics_jsonl_has_one_record_per_read_with_both_clocks() {
    let fixture = fixture();
    let metrics_dir = TempDir::new("store-metrics").unwrap();
    let metrics_path = metrics_dir.path().join("store.jsonl");
    let mut store_config = config(PolicyKind::Lru, 1 << 20);
    store_config.metrics = Some(metrics_path.clone());
    let store = ShardStore::open(&fixture.path, store_config).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let mut expected_records = 0_u64;
    for _ in 0..2 {
        for id in fixture.scope.ids() {
            read_shard(&store, &fixture.scope, id, &mut buffer);
            expected_records += 1;
        }
    }
    store.flush_metrics().unwrap();
    assert_eq!(store.metrics_records(), expected_records);
    let text = std::fs::read_to_string(&metrics_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len() as u64, expected_records);
    let mut previous = 0_u64;
    for line in lines {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(value["schema"], METRICS_SCHEMA);
        assert_eq!(value["event"], "read");
        assert!(value["wall_unix_ns"].as_u64().unwrap() > 0);
        let mono = value["mono_ns"].as_u64().unwrap();
        assert!(mono >= previous);
        previous = mono;
        for field in [
            "bytes_requested",
            "bytes_completed",
            "bytes_used",
            "bytes_wasted",
            "device_bytes",
            "queue_wait_ns",
            "transfer_ns",
            "verify_ns",
            "cache_insert_ns",
            "eviction_ns",
            "stall_ns",
            "read_amplification",
            "hit",
            "verified",
            "verification",
        ] {
            assert!(!value[field].is_null(), "missing field {field}");
        }
    }
}

#[test]
fn publication_proof_skips_hashing_only_when_it_names_the_open_bundle() {
    let fixture = fixture();
    let bundle = BundleFile::open(&fixture.path).unwrap();
    let mut registry =
        ProofRegistry::new(fixture.manifest.bundle_id.clone(), bundle.manifest_sha256());
    let shard = fixture.ids[0];
    let record = fixture.manifest.shard(&shard).unwrap();
    registry
        .insert(VerificationProof {
            shard,
            sha256: record.sha256,
            bundle_id: fixture.manifest.bundle_id.clone(),
            manifest_sha256: bundle.manifest_sha256(),
            verified_unix_ns: 1,
            producer: "court-test".to_owned(),
        })
        .unwrap();
    let mut store_config = config(PolicyKind::None, 0);
    store_config.verification = VerificationMode::PublicationProof(Arc::new(registry));
    let store = ShardStore::open(&fixture.path, store_config).unwrap();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let report = read_shard(&store, &fixture.scope, &shard, &mut buffer);
    assert_eq!(report.verification, VerificationMethod::PublicationProof);
    let other = fixture.ids[1];
    let report = read_shard(&store, &fixture.scope, &other, &mut buffer);
    assert_eq!(report.verification, VerificationMethod::Sha256Read);
}

#[test]
fn a_proof_for_another_bundle_is_refused_at_open() {
    let fixture = fixture();
    let other = selvedge_core::Sha256Digest::of(b"different-manifest");
    let registry = ProofRegistry::new(fixture.manifest.bundle_id.clone(), other);
    let mut store_config = config(PolicyKind::None, 0);
    store_config.verification = VerificationMode::PublicationProof(Arc::new(registry));
    let error = ShardStore::open(&fixture.path, store_config).unwrap_err();
    assert!(matches!(error, StoreError::ProofRejected { .. }));
}

#[test]
fn invalid_configuration_is_refused_before_io() {
    let fixture = fixture();
    let error = ShardStore::open(&fixture.path, config(PolicyKind::Lru, 0)).unwrap_err();
    assert!(matches!(error, StoreError::Config { .. }));
    let error = ShardStore::open(&fixture.path, config(PolicyKind::None, 1)).unwrap_err();
    assert!(matches!(error, StoreError::Config { .. }));
    let mut too_small = config(PolicyKind::None, 0);
    too_small.max_shard_bytes = 1;
    let error = ShardStore::open(&fixture.path, too_small).unwrap_err();
    assert!(matches!(error, StoreError::Config { .. }));
}

#[test]
fn layout_report_describes_the_real_extent_placement() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let report = store.layout_report();
    assert_eq!(report.shard_count, fixture.ids.len() as u64);
    assert_eq!(
        report.sum_extent_bytes + report.padding_bytes,
        report.payload_len
    );
    assert!(
        report.padding_bytes > 0,
        "the fixture pads 144-byte extents"
    );
    assert!(report.aligned_extents > 0);
    assert!(report.coalesced_runs_unbounded < report.shard_count);
    let unique: BTreeSet<_> = store.physical_shard_ids().into_iter().collect();
    assert_eq!(unique.len(), fixture.ids.len());
}

#[test]
fn small_cache_stress_reuses_bytes_and_never_corrupts() {
    let fixture = fixture();
    let capacity = largest_shard(&fixture.manifest) * 3;
    let store = ShardStore::open(&fixture.path, config(PolicyKind::Lru, capacity)).unwrap();
    let expected: Vec<(selvedge_core::ShardId, Vec<u8>)> = fixture
        .ids
        .iter()
        .map(|id| (*id, expected_bytes(&fixture, id)))
        .collect();
    let mut buffer = vec![0_u8; largest_shard(&fixture.manifest) as usize];
    let mut reads = 0_u64;
    let mut hits = 0_u64;
    for _round in 0..6 {
        for id in &fixture.ids {
            let record = fixture.manifest.shard(id).unwrap();
            let slice = &mut buffer[..record.length as usize];
            let report = store.read_required(&fixture.scope, id, slice).unwrap();
            let want = expected
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, bytes)| bytes)
                .unwrap();
            assert_eq!(&slice[..], &want[..]);
            reads += 1;
            if report.hit {
                hits += 1;
            }
            let repeat = store.read_required(&fixture.scope, id, slice).unwrap();
            assert!(repeat.hit);
            assert_eq!(&slice[..], &want[..]);
            reads += 1;
            hits += 1;
        }
    }
    let stats = store.stats();
    assert_eq!(stats.reads, reads);
    assert_eq!(stats.hits, hits);
    assert_eq!(stats.hits + stats.misses, reads);
    assert!(store.cache_stats().used_bytes <= store.cache_stats().capacity_bytes);
    assert_eq!(stats.checksum_failures, 0);
}

#[test]
fn prefetched_bytes_are_only_counted_useful_when_delivered() {
    let fixture = fixture();
    let max_shard = largest_shard(&fixture.manifest);
    let store = ShardStore::open(
        &fixture.path,
        coalescing_config(
            PolicyKind::Lru,
            payload_bytes(&fixture.manifest),
            max_shard * 4,
        ),
    )
    .unwrap();
    let hints: Vec<PrefetchHint> = fixture.ids.iter().copied().map(PrefetchHint::new).collect();
    let outcome = store.prefetch_advisory(&fixture.scope, &hints).unwrap();
    assert_eq!(store.stats().prefetch_bytes_used, 0);
    assert_eq!(outcome.bytes_admitted, store.stats().prefetch_bytes_read);
    let mut buffer = vec![0_u8; max_shard as usize];
    let id = fixture.ids[0];
    let report = read_shard(&store, &fixture.scope, &id, &mut buffer);
    assert!(report.hit);
    assert_eq!(
        store.stats().prefetch_bytes_used,
        fixture.manifest.shard(&id).unwrap().length
    );
    let stats = store.stats();
    assert_eq!(
        stats.prefetch_bytes_wasted,
        stats.prefetch_bytes_read - stats.prefetch_bytes_used,
        "waste must be the prefetched bytes that were never delivered"
    );
    assert!(stats.prefetch_bytes_wasted > 0);
}

#[test]
fn a_prefetch_that_is_never_delivered_is_all_waste() {
    let fixture = fixture();
    let max_shard = largest_shard(&fixture.manifest);
    let store = ShardStore::open(
        &fixture.path,
        coalescing_config(
            PolicyKind::Lru,
            payload_bytes(&fixture.manifest),
            max_shard * 4,
        ),
    )
    .unwrap();
    let hints: Vec<PrefetchHint> = fixture.ids.iter().copied().map(PrefetchHint::new).collect();
    let outcome = store.prefetch_advisory(&fixture.scope, &hints).unwrap();
    assert!(outcome.bytes_admitted > 0);
    let stats = store.stats();
    assert_eq!(stats.prefetch_bytes_used, 0);
    assert_eq!(stats.prefetch_bytes_wasted, stats.prefetch_bytes_read);
}

#[test]
fn no_cache_never_stores_and_reports_unit_amplification() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    let shard = fixture.ids[0];
    let record = fixture.manifest.shard(&shard).unwrap();
    let mut buffer = vec![0_u8; record.length as usize];
    for _ in 0..2 {
        let report = store
            .read_required(&fixture.scope, &shard, &mut buffer)
            .unwrap();
        assert!(!report.hit);
        assert_eq!(report.read_amplification, 1.0);
        assert_eq!(report.device_bytes, record.length);
    }
    let stats = store.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 2);
    assert_eq!(store.cache_stats().entries, 0);
}

#[test]
fn path_and_manifest_digest_are_recorded() {
    let fixture = fixture();
    let store = ShardStore::open(&fixture.path, config(PolicyKind::None, 0)).unwrap();
    assert_eq!(store.path(), Path::new(&fixture.path));
    assert_eq!(store.bundle_id(), fixture.manifest.bundle_id);
    let bundle = BundleFile::open(&fixture.path).unwrap();
    assert_eq!(store.manifest_sha256(), bundle.manifest_sha256());
    assert_eq!(store.shard_ids().len(), fixture.ids.len());
}
