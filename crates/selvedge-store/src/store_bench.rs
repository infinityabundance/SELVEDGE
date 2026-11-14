//! Exact-mode store measurement on a real bundle.
//!
//! One run answers: how many bytes does a full exact-mode pass move, how long
//! does it take, what are the per-shard latency tail percentiles, what does the
//! host page cache contribute, and what does a store cache contribute when the
//! page cache is dropped between passes.
//!
//! The executed set is the whole bundle in every arm (all shards, unchanged
//! semantics). Only the read order, the coalescing and the cache policy vary.
//! The two passes are:
//!
//! 1. **pass 1** — store cache empty, page cache dropped: a cold physical read;
//! 2. **pass 2** — page cache dropped again, store cache retained: the only
//!    measurement that isolates the store's own cache from the kernel's.
//!
//! A separate warm arm leaves the page cache populated for pass 2, which
//! measures the operating system's cache and is explicitly not attributed to the
//! store.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use selvedge_core::{Sha256Digest, ShardId};
use selvedge_format::BundleFile;
use serde::Serialize;

use crate::cache::CacheStats;
use crate::error::{Result, StoreError};
use crate::io_scan::{format_stamp, utc_now};
use crate::layout::{Layout, LayoutReport};
use crate::pagecache::{AdviceReport, PageCache};
use crate::store::{
    ExecutionScope, PolicyKind, PrefetchHint, ReadOrder, ScopeReadReport, ShardStore, StoreConfig,
    VerificationMode,
};
use crate::sysfs::{self, DeviceCounters, Environment, MountInfo};
use crate::tier::QueueLimits;

/// Report schema string.
pub const STORE_BENCH_SCHEMA: &str = "selvedge/store-bench/v1";

/// Bench configuration.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Bundle to read.
    pub bundle: PathBuf,
    /// Directory whose filesystem free space is recorded.
    pub workdir: PathBuf,
    /// Directory for per-arm JSONL metrics; one file per arm, when set.
    pub metrics_dir: Option<PathBuf>,
    /// Arm names to run; empty means the default list.
    pub arms: Vec<String>,
    /// Coalescing cap as a multiple of the largest extent.
    pub run_multiple: u64,
    /// Layer index used by the prefetch-waste arm.
    pub prefetch_layer: u32,
    /// Stamp to record; defaults to now.
    pub stamp: Option<String>,
    /// Revision to record; defaults to the court environment.
    pub revision: Option<String>,
    /// Exact argv.
    pub argv: Vec<String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            bundle: PathBuf::new(),
            workdir: PathBuf::from("."),
            metrics_dir: None,
            arms: Vec::new(),
            run_multiple: 4,
            prefetch_layer: 0,
            stamp: None,
            revision: None,
            argv: Vec::new(),
        }
    }
}

/// Facts about the measured bundle.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BundleInfo {
    /// Path.
    pub path: PathBuf,
    /// File length.
    pub file_len_bytes: u64,
    /// Payload length.
    pub payload_len_bytes: u64,
    /// Shard count.
    pub shard_count: u64,
    /// Smallest extent.
    pub min_extent_bytes: u64,
    /// Largest extent.
    pub max_extent_bytes: u64,
    /// Bundle id.
    pub bundle_id: String,
    /// Manifest digest.
    pub manifest_sha256: Sha256Digest,
    /// Verification mode used.
    pub verification: String,
}

/// Measured device movement for one pass.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePass {
    /// Block device.
    pub name: String,
    /// Bytes read by the device during the pass.
    pub read_bytes: u64,
    /// Read requests completed by the device.
    pub reads_completed: u64,
    /// Device bytes divided by bytes delivered.
    pub amplification: f64,
    /// Whether the device shows reads consistent with a cold pass (at least half
    /// of the delivered bytes).
    pub cold_consistent: bool,
}

/// One pass over the whole execution scope.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PassStats {
    /// Shards delivered.
    pub shards: u64,
    /// Bytes delivered to the caller in total.
    pub bytes_used: u64,
    /// Store cache hits during the pass.
    pub hits: u64,
    /// Bytes delivered from the store cache during the pass.
    pub cache_hit_bytes: u64,
    /// Wall time.
    pub elapsed_ns: u64,
    /// Delivered-byte throughput in MiB/s.
    pub throughput_mib_s: f64,
    /// Per-shard stall p50.
    pub stall_p50_ns: u64,
    /// p95.
    pub stall_p95_ns: u64,
    /// p99.
    pub stall_p99_ns: u64,
    /// Slowest shard.
    pub stall_max_ns: u64,
    /// Sum of per-shard stalls.
    pub stall_total_ns: u64,
    /// Errors observed.
    pub errors: u64,
    /// Whether the page cache was dropped immediately before the pass.
    pub page_cache_dropped: bool,
    /// Advice record for the drop, when one was issued.
    pub drop: Option<AdviceReport>,
    /// Device movement, when counters are available.
    pub device: Option<DevicePass>,
}

/// One measured arm.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArmResult {
    /// Arm name.
    pub arm: String,
    /// Cache policy name.
    pub policy: String,
    /// Configured cache budget.
    pub cache_configured_bytes: u64,
    /// Cache budget as a fraction of the payload.
    pub cache_fraction: f64,
    /// Share of the payload the budget can hold, capped at one.
    pub cache_payload_share: f64,
    /// Read order.
    pub order: String,
    /// Whether reads were coalesced.
    pub coalesced: bool,
    /// Whether pass 2 dropped the page cache (false = OS-warm pass).
    pub pass2_page_cache_dropped: bool,
    /// First pass.
    pub pass1: PassStats,
    /// Second pass, when run.
    pub pass2: Option<PassStats>,
    /// Cache counters at the end of the arm.
    pub cache: CacheStats,
    /// Aggregate store counters at the end of the arm.
    pub store: crate::metrics::StoreStats,
    /// Notes specific to the arm.
    pub notes: Vec<String>,
    /// Set when the arm was not run.
    pub skipped: Option<String>,
}

/// The whole measurement.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchReport {
    /// Always [`STORE_BENCH_SCHEMA`].
    pub schema: &'static str,
    /// Receipt stamp.
    pub stamp: String,
    /// Git revision.
    pub revision: String,
    /// Wall-clock start.
    pub started_utc: String,
    /// Wall-clock finish.
    pub finished_utc: String,
    /// Exact argv.
    pub argv: Vec<String>,
    /// Environment.
    pub environment: Environment,
    /// Mount and free space.
    pub mount: MountInfo,
    /// Bundle facts.
    pub bundle: BundleInfo,
    /// Layout summary.
    pub layout: LayoutReport,
    /// Device idle-window delta, the counter noise floor.
    pub device_idle_delta: Option<DeviceCounters>,
    /// Idle window duration.
    pub device_idle_window_ns: u64,
    /// Arms, in execution order.
    pub arms: Vec<ArmResult>,
    /// Notes a reader must know.
    pub notes: Vec<String>,
}

/// Run the configured arms against the bundle.
///
/// # Errors
///
/// Rejects a missing bundle, unreadable environment facts, an invalid arm name,
/// and any read failure. An arm that cannot fit the memory limit is recorded as
/// skipped, never silently dropped.
pub fn run(config: &BenchConfig) -> Result<BenchReport> {
    if config.bundle.as_os_str().is_empty() {
        return Err(StoreError::config("no bundle supplied"));
    }
    if config.run_multiple == 0 {
        return Err(StoreError::config("run_multiple is zero"));
    }
    let started_utc = utc_now();
    let stamp = config.stamp.clone().unwrap_or_else(now_stamp);
    let environment = sysfs::environment()?;
    let mount = sysfs::mount_for(&config.workdir)?;
    let (bundle_info, layout, ids) = describe_bundle(&config.bundle)?;

    let device_name = mount.device_name.clone();
    let idle_before = device_name
        .as_ref()
        .and_then(|name| sysfs::device_counters(name).ok());
    let idle_start = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let idle_window_ns = u64::try_from(idle_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let device_idle_delta = match (idle_before, device_name.as_ref()) {
        (Some(before), Some(name)) => sysfs::device_counters(name)
            .ok()
            .map(|after| after.delta(&before)),
        _ => None,
    };

    let mut notes = vec![
        "every arm executes the whole bundle: selected set == executed set, no shard is skipped"
            .to_owned(),
        "pass 2 drops the page cache again, so its bytes come from the store cache or the device, \
         not from the kernel"
            .to_owned(),
        "the `none-warm` arm deliberately leaves the page cache populated; its pass 2 measures the \
         operating system, not the store"
            .to_owned(),
        "device counters are device-wide; the idle-window delta is the measured noise floor"
            .to_owned(),
    ];
    if mount.free_bytes < 1024 * 1024 * 1024 {
        notes.push(format!(
            "free space on {} is {} bytes; the bench writes only small receipts and JSONL metrics, \
             never bundle-sized test files",
            mount.target, mount.free_bytes
        ));
    }

    let scope = ExecutionScope::exact(ids)?;
    let memory_limit = environment
        .court_memory_bytes
        .unwrap_or(environment.memory.total_bytes);
    let selected = if config.arms.is_empty() {
        default_arms()
    } else {
        config.arms.clone()
    };
    let mut arms = Vec::new();
    for arm in &selected {
        arms.push(run_arm(
            config,
            arm,
            &scope,
            bundle_info.payload_len_bytes,
            layout.max_extent_bytes,
            memory_limit,
            &device_name,
        )?);
    }
    Ok(BenchReport {
        schema: STORE_BENCH_SCHEMA,
        stamp,
        revision: config
            .revision
            .clone()
            .or_else(|| environment.revision.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        started_utc,
        finished_utc: utc_now(),
        argv: config.argv.clone(),
        environment,
        mount,
        bundle: bundle_info,
        layout,
        device_idle_delta,
        device_idle_window_ns: idle_window_ns,
        arms,
        notes,
    })
}

fn describe_bundle(path: &Path) -> Result<(BundleInfo, LayoutReport, Vec<ShardId>)> {
    let bundle = BundleFile::open(path)?;
    let manifest = bundle.manifest();
    let layout = Layout::from_manifest(manifest)?;
    let min_extent = manifest
        .shards
        .iter()
        .map(|shard| shard.length)
        .min()
        .unwrap_or(0);
    let max_extent = manifest
        .shards
        .iter()
        .map(|shard| shard.length)
        .max()
        .unwrap_or(0);
    let info = BundleInfo {
        path: path.to_path_buf(),
        file_len_bytes: std::fs::metadata(path)?.len(),
        payload_len_bytes: manifest.payload_len,
        shard_count: manifest.shards.len() as u64,
        min_extent_bytes: min_extent,
        max_extent_bytes: max_extent,
        bundle_id: manifest.bundle_id.clone(),
        manifest_sha256: bundle.manifest_sha256(),
        verification: "always".to_owned(),
    };
    let ids = manifest.shards.iter().map(|shard| shard.shard_id).collect();
    Ok((info, layout.report(), ids))
}

/// The default arm list, in a fixed order so receipt pointers are stable.
#[must_use]
pub fn default_arms() -> Vec<String> {
    vec![
        "none-cold".to_owned(),
        "none-warm".to_owned(),
        "plan-order".to_owned(),
        "coalesced".to_owned(),
        "lru-1of8".to_owned(),
        "lru-1of4".to_owned(),
        "lru-1of2".to_owned(),
        "lru-2x".to_owned(),
        "lfu-1of4".to_owned(),
        "least-stale-1of4".to_owned(),
        "pinned-all".to_owned(),
        "prefetch-layer-waste".to_owned(),
    ]
}

struct ArmSpec {
    policy: PolicyKind,
    cache_bytes: u64,
    order: ReadOrder,
    coalesced: bool,
    warm_pass2: bool,
    prefetch_waste: bool,
}

fn arm_spec(arm: &str, payload: u64) -> Result<ArmSpec> {
    let spec = match arm {
        "none-cold" | "plan-order" => ArmSpec {
            policy: PolicyKind::None,
            cache_bytes: 0,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "none-warm" => ArmSpec {
            policy: PolicyKind::None,
            cache_bytes: 0,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: true,
            prefetch_waste: false,
        },
        "coalesced" => ArmSpec {
            policy: PolicyKind::None,
            cache_bytes: 0,
            order: ReadOrder::Physical,
            coalesced: true,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "lru-1of8" => ArmSpec {
            policy: PolicyKind::Lru,
            cache_bytes: payload / 8,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "lru-1of4" => ArmSpec {
            policy: PolicyKind::Lru,
            cache_bytes: payload / 4,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "lru-1of2" => ArmSpec {
            policy: PolicyKind::Lru,
            cache_bytes: payload / 2,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "lru-2x" => ArmSpec {
            policy: PolicyKind::Lru,
            cache_bytes: payload.saturating_mul(2),
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "lfu-1of4" => ArmSpec {
            policy: PolicyKind::Lfu,
            cache_bytes: payload / 4,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "least-stale-1of4" => ArmSpec {
            policy: PolicyKind::LeastStale,
            cache_bytes: payload / 4,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "pinned-all" => ArmSpec {
            policy: PolicyKind::Pinned,
            cache_bytes: payload,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: false,
        },
        "prefetch-layer-waste" => ArmSpec {
            policy: PolicyKind::Lru,
            cache_bytes: payload,
            order: ReadOrder::Plan,
            coalesced: false,
            warm_pass2: false,
            prefetch_waste: true,
        },
        other => {
            return Err(StoreError::config(format!(
                "unknown arm {other:?}; known arms are {}",
                default_arms().join(", ")
            )));
        }
    };
    Ok(spec)
}

fn run_arm(
    config: &BenchConfig,
    arm: &str,
    scope: &ExecutionScope,
    payload_len: u64,
    max_extent: u64,
    memory_limit: u64,
    device_name: &Option<String>,
) -> Result<ArmResult> {
    let spec = arm_spec(arm, payload_len)?;
    // A cache never holds more than the distinct bytes, so the reservation is
    // bounded by the payload; a 1 GiB reserve covers buffers and the runtime.
    let effective_cache = spec.cache_bytes.min(payload_len);
    let reserve = 1024 * 1024 * 1024_u64;
    let empty_pass = PassStats {
        shards: 0,
        bytes_used: 0,
        hits: 0,
        cache_hit_bytes: 0,
        elapsed_ns: 0,
        throughput_mib_s: 0.0,
        stall_p50_ns: 0,
        stall_p95_ns: 0,
        stall_p99_ns: 0,
        stall_max_ns: 0,
        stall_total_ns: 0,
        errors: 0,
        page_cache_dropped: true,
        drop: None,
        device: None,
    };
    let mut result = ArmResult {
        arm: arm.to_owned(),
        policy: spec.policy.name().to_owned(),
        cache_configured_bytes: spec.cache_bytes,
        cache_fraction: if payload_len == 0 {
            0.0
        } else {
            spec.cache_bytes as f64 / payload_len as f64
        },
        cache_payload_share: if payload_len == 0 {
            0.0
        } else {
            (spec.cache_bytes as f64 / payload_len as f64).min(1.0)
        },
        order: match spec.order {
            ReadOrder::Plan => "plan".to_owned(),
            ReadOrder::Physical => "physical".to_owned(),
        },
        coalesced: spec.coalesced,
        pass2_page_cache_dropped: !spec.warm_pass2,
        pass1: empty_pass.clone(),
        pass2: None,
        cache: CacheStats {
            policy: spec.policy.name().to_owned(),
            ..CacheStats::empty()
        },
        store: crate::metrics::StoreStats::default(),
        notes: Vec::new(),
        skipped: None,
    };
    if effective_cache.saturating_add(reserve) > memory_limit {
        result.skipped = Some(format!(
            "effective cache {effective_cache} + reserve {reserve} exceeds the memory limit \
             {memory_limit}"
        ));
        return Ok(result);
    }
    let metrics_path = config.metrics_dir.as_ref().map(|dir| {
        let safe = arm.replace('/', "_");
        dir.join(format!("{safe}.jsonl"))
    });
    let store_config = StoreConfig {
        cache_bytes: spec.cache_bytes,
        policy: spec.policy,
        max_shard_bytes: max_extent,
        verification: VerificationMode::Always,
        queue: QueueLimits {
            max_pending: 4096,
            max_in_flight: 64,
            max_in_flight_bytes: max_extent.saturating_mul(64),
        },
        max_run_bytes: max_extent
            .saturating_mul(config.run_multiple)
            .max(max_extent),
        coalesce_gap_bytes: 0,
        metrics: metrics_path.clone(),
    };
    let store = ShardStore::open(&config.bundle, store_config)?;
    let mut buffer = vec![0_u8; max_extent as usize];

    if spec.prefetch_waste {
        result.pass1 = prefetch_waste_pass(config, &store, &mut buffer, device_name)?;
        result.notes.push(
            "the prefetch arm delivers a quarter of the planned layer; the remaining prefetched \
             bytes are genuine waste for an abandoned plan, and the executed set is unchanged"
                .to_owned(),
        );
    } else {
        let drop1 = PageCache::drop_file(&config.bundle)?;
        result.pass1 = measure_pass(
            &store,
            scope,
            &mut buffer,
            spec.coalesced,
            spec.order,
            Some(drop1),
            device_name,
        )?;
        let drop2 = if spec.warm_pass2 {
            None
        } else {
            Some(PageCache::drop_file(&config.bundle)?)
        };
        result.pass2 = Some(measure_pass(
            &store,
            scope,
            &mut buffer,
            spec.coalesced,
            spec.order,
            drop2,
            device_name,
        )?);
    }
    result.store = store.stats();
    result.cache = store.cache_stats();
    store.flush_metrics()?;
    if let Some(path) = &metrics_path {
        result.notes.push(format!(
            "request metrics written to {path:?} ({} records)",
            store.metrics_records()
        ));
    }
    Ok(result)
}

fn measure_pass(
    store: &ShardStore,
    scope: &ExecutionScope,
    buffer: &mut [u8],
    coalesced: bool,
    order: ReadOrder,
    drop: Option<AdviceReport>,
    device_name: &Option<String>,
) -> Result<PassStats> {
    let stats_before = store.stats();
    let device_before = device_snapshot(device_name);
    let report: ScopeReadReport = if coalesced {
        store.read_scope_coalesced(scope)?
    } else {
        store.read_scope(scope, buffer, order)?
    };
    let device_after = device_snapshot(device_name);
    let stats_after = store.stats();
    let hits = stats_after.hits.saturating_sub(stats_before.hits);
    let mut pass = PassStats {
        shards: report.shards,
        bytes_used: report.bytes_used,
        hits,
        cache_hit_bytes: 0,
        elapsed_ns: report.elapsed_ns,
        throughput_mib_s: mib_s(report.bytes_used, report.elapsed_ns),
        stall_p50_ns: report.stall_p50_ns,
        stall_p95_ns: report.stall_p95_ns,
        stall_p99_ns: report.stall_p99_ns,
        stall_max_ns: report.stall_max_ns,
        stall_total_ns: report.stall_ns,
        errors: report.errors,
        page_cache_dropped: drop.is_some(),
        drop,
        device: None,
    };
    pass.device = device_pass(device_before, device_after, device_name, pass.bytes_used);
    if let Some(device) = &pass.device {
        pass.cache_hit_bytes = pass.bytes_used.saturating_sub(device.read_bytes);
    }
    Ok(pass)
}

fn prefetch_waste_pass(
    config: &BenchConfig,
    store: &ShardStore,
    buffer: &mut [u8],
    device_name: &Option<String>,
) -> Result<PassStats> {
    let layer = config.prefetch_layer;
    let layer_ids: Vec<ShardId> = store
        .manifest()
        .shards
        .iter()
        .filter(|shard| shard.layer == Some(layer))
        .map(|shard| shard.shard_id)
        .collect();
    if layer_ids.is_empty() {
        return Err(StoreError::precondition(format!(
            "no shards carry layer {layer}"
        )));
    }
    let layer_scope = ExecutionScope::exact(layer_ids.iter().copied())?;
    let drop = PageCache::drop_file(&config.bundle)?;
    let device_before = device_snapshot(device_name);
    let start = Instant::now();
    let hints: Vec<PrefetchHint> = layer_ids.iter().copied().map(PrefetchHint::new).collect();
    let outcome = store.prefetch_advisory(&layer_scope, &hints)?;
    let deliver = (layer_ids.len() / 4).max(1);
    let stats_before = store.stats();
    let mut stalls = Vec::new();
    for id in layer_ids.iter().take(deliver) {
        let len = store
            .manifest()
            .shard(id)
            .ok_or(StoreError::UnknownShard(*id))?
            .length as usize;
        let started = Instant::now();
        store.read_required(&layer_scope, id, &mut buffer[..len])?;
        stalls.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let stats = store.stats();
    let device_after = device_snapshot(device_name);
    let (_, p50, p95, p99, max) = crate::metrics::percentiles(&mut stalls);
    let mut pass = PassStats {
        shards: deliver as u64,
        bytes_used: stats.bytes_used,
        hits: stats.hits.saturating_sub(stats_before.hits),
        cache_hit_bytes: stats.bytes_used,
        elapsed_ns,
        throughput_mib_s: mib_s(stats.bytes_used, elapsed_ns),
        stall_p50_ns: p50,
        stall_p95_ns: p95,
        stall_p99_ns: p99,
        stall_max_ns: max,
        stall_total_ns: stats.stall_ns,
        errors: 0,
        page_cache_dropped: true,
        drop: Some(drop),
        device: None,
    };
    pass.device = device_pass(device_before, device_after, device_name, pass.bytes_used);
    let _ = outcome;
    Ok(pass)
}

fn device_snapshot(name: &Option<String>) -> Option<DeviceCounters> {
    let name = name.as_ref()?;
    sysfs::device_counters(name).ok()
}

fn device_pass(
    before: Option<DeviceCounters>,
    after: Option<DeviceCounters>,
    name: &Option<String>,
    delivered: u64,
) -> Option<DevicePass> {
    let (Some(name), Some(before), Some(after)) = (name, before, after) else {
        return None;
    };
    let delta = after.delta(&before);
    let read_bytes = delta.read_bytes();
    Some(DevicePass {
        name: name.clone(),
        read_bytes,
        reads_completed: delta.reads_completed,
        amplification: if delivered == 0 {
            0.0
        } else {
            read_bytes as f64 / delivered as f64
        },
        cold_consistent: read_bytes.saturating_mul(2) >= delivered,
    })
}

fn mib_s(bytes: u64, elapsed_ns: u64) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0)) / (elapsed_ns as f64 / 1_000_000_000.0)
}

/// Current UTC stamp in the shared receipt format.
#[must_use]
pub fn now_stamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format_stamp(seconds)
}
