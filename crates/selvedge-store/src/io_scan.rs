//! Physical I/O characterisation on real hardware.
//!
//! Everything here is measured, never modelled. The command takes an existing
//! file (in the recorded runs, the real 1.7B f32 bundle), issues positional
//! reads, and records per-operation latencies, throughput, the block device's own
//! sector counters, and whether the page cache was dropped. There is no synthetic
//! bandwidth number anywhere.
//!
//! Controlled conditions:
//!
//! - **cold** measurements issue `POSIX_FADV_DONTNEED` over the file first and
//!   then read; the device counter delta shows whether the reads reached the
//!   device;
//! - **warm** measurements read the same range twice and report the second pass;
//! - the file is read-only, so the command cannot fill the filesystem;
//! - every buffer is bounded by the largest block size or one coalesced run.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use selvedge_core::{Sha256Builder, Sha256Digest};
use serde::Serialize;

use crate::error::{Result, StoreError};
use crate::metrics::percentiles;
use crate::pagecache::{AdviceReport, PageCache};
use crate::sysfs::{self, DeviceCounters, Environment, MountInfo};

/// Report schema string.
pub const IO_SCAN_SCHEMA: &str = "selvedge/io-scan/v1";

/// Measurement configuration. Defaults are the recorded run's parameters.
#[derive(Clone, Debug)]
pub struct IoScanConfig {
    /// File to read. Must exist and be at least `min_bytes`.
    pub file: PathBuf,
    /// Directory whose filesystem is reported for free space.
    pub workdir: PathBuf,
    /// Block sizes for sequential and random sweeps.
    pub block_sizes: Vec<u64>,
    /// Bytes read per sequential size.
    pub sequential_bytes: u64,
    /// Samples per random-read size.
    pub random_samples: Vec<usize>,
    /// Queue depths for the parallelism sweep.
    pub queue_depths: Vec<usize>,
    /// Block size for the queue-depth sweep.
    pub qd_block_bytes: u64,
    /// Total bytes per queue-depth point.
    pub qd_total_bytes: u64,
    /// Offsets (mod 65536) tested for alignment effects.
    pub alignment_offsets: Vec<u64>,
    /// Bytes read per alignment point (the real f32 shard length by default).
    pub alignment_read_bytes: u64,
    /// Over-read size compared against `alignment_read_bytes` useful bytes.
    pub overread_bytes: u64,
    /// Bytes hashed for the checksum cost measurement.
    pub checksum_bytes: u64,
    /// Drop the page cache before cold measurements.
    pub drop_cache: bool,
    /// Minimum file size accepted.
    pub min_bytes: u64,
    /// Deterministic sampling seed.
    pub seed: u64,
    /// Wall-clock timestamp to record, for reproducibility of the receipt name.
    pub stamp: Option<String>,
    /// Git revision to record; the court environment supplies it.
    pub revision: Option<String>,
    /// Exact argv to record.
    pub argv: Vec<String>,
}

impl Default for IoScanConfig {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            workdir: PathBuf::from("."),
            block_sizes: vec![4096, 65_536, 1_048_576, 8_388_608],
            sequential_bytes: 256 * 1024 * 1024,
            random_samples: vec![2048, 512, 128, 32],
            queue_depths: vec![1, 4, 16],
            qd_block_bytes: 65_536,
            qd_total_bytes: 256 * 1024 * 1024,
            alignment_offsets: vec![512, 4096, 65_536],
            alignment_read_bytes: 3 * 1024 * 1024,
            overread_bytes: 8 * 1024 * 1024,
            checksum_bytes: 1024 * 1024 * 1024,
            drop_cache: true,
            min_bytes: 16 * 1024 * 1024,
            seed: 0x5eed_5eed_u64,
            stamp: None,
            revision: None,
            argv: Vec::new(),
        }
    }
}

/// Validated configuration summary, recorded in the report.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScanPlan {
    /// Block sizes.
    pub block_sizes: Vec<u64>,
    /// Sequential bytes per size.
    pub sequential_bytes: u64,
    /// Random samples per size.
    pub random_samples: Vec<usize>,
    /// Queue depths.
    pub queue_depths: Vec<usize>,
    /// Queue-depth block size.
    pub qd_block_bytes: u64,
    /// Queue-depth total bytes.
    pub qd_total_bytes: u64,
    /// Alignment offsets.
    pub alignment_offsets: Vec<u64>,
    /// Alignment read size.
    pub alignment_read_bytes: u64,
    /// Over-read size.
    pub overread_bytes: u64,
    /// Checksum bytes.
    pub checksum_bytes: u64,
    /// Whether the page cache is dropped for cold runs.
    pub drop_cache: bool,
    /// Sampling seed.
    pub seed: u64,
}

/// Facts about the measured file.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileInfo {
    /// Path.
    pub path: PathBuf,
    /// Length in bytes.
    pub len_bytes: u64,
}

/// Page-cache control evidence.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheControl {
    /// Whether cold measurements dropped the cache.
    pub dropped: bool,
    /// Advice records issued, one per cold point.
    pub drops: Vec<AdviceReport>,
    /// Warm probe latency in nanoseconds.
    pub warm_probe_ns: u64,
    /// Cold probe latency in nanoseconds after a drop.
    pub cold_probe_ns: u64,
    /// Ratio cold/warm; above one means the drop had an effect.
    pub cold_warm_ratio: f64,
    /// Whether the ratio is above 2, the documented evidence threshold.
    pub drop_effective: bool,
}

/// One sequential point.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialPoint {
    /// Block size.
    pub block_bytes: u64,
    /// Total bytes read.
    pub bytes: u64,
    /// Operations issued.
    pub operations: u64,
    /// Wall time.
    pub elapsed_ns: u64,
    /// Throughput in MiB/s (1 MiB = 1,048,576 bytes).
    pub throughput_mib_s: f64,
    /// Median operation latency.
    pub p50_ns: u64,
    /// p95.
    pub p95_ns: u64,
    /// p99.
    pub p99_ns: u64,
    /// Slowest operation.
    pub max_ns: u64,
    /// Device bytes during the point, when counters are available.
    pub device_bytes: Option<u64>,
    /// Device bytes divided by requested bytes.
    pub device_amplification: Option<f64>,
    /// Whether the page cache was dropped first.
    pub cold: bool,
}

/// One random-read point.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomPoint {
    /// Read size.
    pub block_bytes: u64,
    /// Samples.
    pub samples: u64,
    /// Bytes requested.
    pub bytes_requested: u64,
    /// Device bytes during the point.
    pub device_bytes: Option<u64>,
    /// Device amplification.
    pub device_amplification: Option<f64>,
    /// Median.
    pub p50_ns: u64,
    /// p95.
    pub p95_ns: u64,
    /// p99.
    pub p99_ns: u64,
    /// Fastest.
    pub min_ns: u64,
    /// Slowest.
    pub max_ns: u64,
    /// Mean.
    pub mean_ns: f64,
    /// Whether the page cache was dropped first.
    pub cold: bool,
}

/// One queue-depth point.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueDepthPoint {
    /// Concurrent readers.
    pub queue_depth: u64,
    /// Block size per read.
    pub block_bytes: u64,
    /// Total bytes read.
    pub bytes: u64,
    /// Wall time.
    pub elapsed_ns: u64,
    /// Aggregate throughput.
    pub throughput_mib_s: f64,
    /// Per-read median.
    pub p50_ns: u64,
    /// p95.
    pub p95_ns: u64,
    /// p99.
    pub p99_ns: u64,
    /// Device bytes.
    pub device_bytes: Option<u64>,
    /// Device amplification.
    pub device_amplification: Option<f64>,
    /// Whether the page cache was dropped first.
    pub cold: bool,
}

/// One alignment/over-read point.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AlignmentPoint {
    /// Description of the point.
    pub label: String,
    /// File offset used.
    pub offset: u64,
    /// Offset modulo 65536.
    pub alignment: u64,
    /// Bytes the operation requested.
    pub bytes_requested: u64,
    /// Bytes the computation actually needed.
    pub bytes_useful: u64,
    /// Device bytes observed.
    pub device_bytes: Option<u64>,
    /// Requested divided by useful.
    pub requested_per_useful: f64,
    /// Device bytes divided by useful, the measured amplification.
    pub device_per_useful: Option<f64>,
    /// Wall time.
    pub elapsed_ns: u64,
    /// Whether the page cache was dropped first.
    pub cold: bool,
}

/// Checksum cost.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChecksumPoint {
    /// Algorithm.
    pub algorithm: &'static str,
    /// Bytes hashed.
    pub bytes: u64,
    /// Wall time of the warm hash pass.
    pub warm_elapsed_ns: u64,
    /// Warm throughput in MiB/s.
    pub warm_throughput_mib_s: f64,
    /// Nanoseconds per GiB of warm verification.
    pub warm_ns_per_gib: f64,
    /// Wall time of the cold read-and-hash pass.
    pub cold_elapsed_ns: u64,
    /// Cold throughput in MiB/s.
    pub cold_throughput_mib_s: f64,
    /// Device bytes during the cold pass.
    pub cold_device_bytes: Option<u64>,
    /// Digest of the hashed range, for reproduction.
    pub digest: Sha256Digest,
    /// Per-shard hash cost at the measured shard size.
    pub shard_bytes: u64,
    /// Measured nanoseconds to hash one shard of that size.
    pub shard_hash_ns: u64,
}

/// Device block.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceBlock {
    /// Block device name.
    pub name: String,
    /// Counters before the scan.
    pub before: DeviceCounters,
    /// Counters after the scan.
    pub after: DeviceCounters,
    /// Delta over the scan.
    pub delta: DeviceCounters,
    /// Idle-window delta measured before any scan work, the noise floor.
    pub idle_delta: DeviceCounters,
    /// Idle window duration.
    pub idle_window_ns: u64,
}

/// The whole measurement.
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IoScanReport {
    /// Always [`IO_SCAN_SCHEMA`].
    pub schema: &'static str,
    /// Receipt stamp, UTC `YYYYMMDDTHHMMSSZ`.
    pub stamp: String,
    /// Git revision from the court environment.
    pub revision: String,
    /// Wall-clock start.
    pub started_utc: String,
    /// Wall-clock finish.
    pub finished_utc: String,
    /// Exact argv.
    pub argv: Vec<String>,
    /// Environment facts.
    pub environment: Environment,
    /// Mount and free-space facts.
    pub mount: MountInfo,
    /// Measured file.
    pub file: FileInfo,
    /// Plan that was executed.
    pub plan: ScanPlan,
    /// Device counters, when the mount is a block device.
    pub device: Option<DeviceBlock>,
    /// Page-cache control evidence.
    pub cache_control: CacheControl,
    /// Sequential points.
    pub sequential: Vec<SequentialPoint>,
    /// Sequential points with readahead disabled (`POSIX_FADV_RANDOM`), which
    /// isolates the per-operation device latency from the kernel's readahead.
    pub sequential_noreadahead: Vec<SequentialPoint>,
    /// Random points.
    pub random: Vec<RandomPoint>,
    /// Queue-depth points.
    pub queue_depth: Vec<QueueDepthPoint>,
    /// Alignment points.
    pub alignment: Vec<AlignmentPoint>,
    /// Checksum cost.
    pub checksum: ChecksumPoint,
    /// Notes a reader must know to interpret the numbers.
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
struct DeviceSampler {
    name: Option<String>,
}

impl DeviceSampler {
    fn snapshot(&self) -> Option<DeviceCounters> {
        let name = self.name.as_ref()?;
        sysfs::device_counters(name).ok()
    }

    fn window(&self) -> DeviceWindow {
        DeviceWindow {
            before: self.snapshot(),
        }
    }
}

struct DeviceWindow {
    before: Option<DeviceCounters>,
}

impl DeviceWindow {
    fn close(&self, sampler: &DeviceSampler, requested: u64) -> (Option<u64>, Option<f64>) {
        let after = sampler.snapshot();
        match (self.before, after) {
            (Some(before), Some(after)) => {
                let bytes = after.delta(&before).read_bytes();
                let amplification = if requested == 0 {
                    0.0
                } else {
                    bytes as f64 / requested as f64
                };
                (Some(bytes), Some(amplification))
            }
            _ => (None, None),
        }
    }
}

/// Run the scan.
///
/// # Errors
///
/// Rejects a missing or too-small file, an invalid configuration, read failures
/// and any failure to collect the mandatory environment facts. Optional device
/// counters degrade to `None` with a note rather than aborting.
pub fn run(config: &IoScanConfig) -> Result<IoScanReport> {
    validate(config)?;
    let started_utc = utc_now();
    let stamp = config.stamp.clone().unwrap_or_else(stamp_now);
    let environment = sysfs::environment()?;
    let mount = sysfs::mount_for(&config.workdir)?;
    let file_meta = std::fs::metadata(&config.file)
        .map_err(|err| StoreError::precondition(format!("cannot stat {:?}: {err}", config.file)))?;
    let len = file_meta.len();
    if len < config.min_bytes {
        return Err(StoreError::precondition(format!(
            "file is {len} bytes, below the {} byte minimum",
            config.min_bytes
        )));
    }
    let file = File::open(&config.file)?;
    let mut notes = Vec::new();
    let sampler = DeviceSampler {
        name: mount.device_name.clone(),
    };
    if sampler.name.is_none() {
        notes.push(format!(
            "mount {:?} is not a block device; device counters omitted",
            mount.target
        ));
    }
    let idle_before = sampler.snapshot();
    let idle_start = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let idle_window_ns = u64::try_from(idle_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let idle_delta = match (idle_before, sampler.snapshot()) {
        (Some(before), Some(after)) => Some(after.delta(&before)),
        _ => None,
    };

    let max_block = config.block_sizes.iter().copied().max().unwrap_or(4096);
    let buffer_bytes = max_block
        .max(config.overread_bytes)
        .max(config.qd_block_bytes)
        .max(1);
    let mut buffer = vec![0_u8; usize::try_from(buffer_bytes)?];

    // Page-cache control evidence: a warm 4 KiB probe and a cold one.
    let probe_offset = aligned_offset(len, 4096, 4096);
    let mut probe = [0_u8; 4096];
    file.read_exact_at(&mut probe, probe_offset)?;
    let warm_start = Instant::now();
    file.read_exact_at(&mut probe, probe_offset)?;
    let warm_probe_ns = u64::try_from(warm_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let mut drops = Vec::new();
    if config.drop_cache {
        drops.push(PageCache::drop_file(&config.file)?);
    }
    let cold_start = Instant::now();
    file.read_exact_at(&mut probe, probe_offset)?;
    let cold_probe_ns = u64::try_from(cold_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let cold_warm_ratio = if warm_probe_ns == 0 {
        0.0
    } else {
        cold_probe_ns as f64 / warm_probe_ns as f64
    };

    let device_before = sampler.snapshot();
    let mut sequential = Vec::new();
    for block in config.block_sizes.iter().copied() {
        sequential.push(sequential_point(
            &file,
            &config.file,
            len,
            block,
            config.sequential_bytes,
            config.drop_cache,
            &mut drops,
            &sampler,
            false,
        )?);
    }
    let mut sequential_noreadahead = Vec::new();
    for block in config.block_sizes.iter().copied() {
        sequential_noreadahead.push(sequential_point(
            &file,
            &config.file,
            len,
            block,
            config.sequential_bytes,
            config.drop_cache,
            &mut drops,
            &sampler,
            true,
        )?);
    }
    let mut random = Vec::new();
    for (index, block) in config.block_sizes.iter().copied().enumerate() {
        let samples = config
            .random_samples
            .get(index)
            .copied()
            .unwrap_or(config.random_samples.last().copied().unwrap_or(64));
        random.push(random_point(
            &file,
            &config.file,
            len,
            block,
            samples,
            config.drop_cache,
            &mut drops,
            config.seed ^ block,
            &mut buffer,
            &sampler,
        )?);
    }
    let mut queue_depth = Vec::new();
    for depth in &config.queue_depths {
        queue_depth.push(qd_point(
            &file,
            &config.file,
            len,
            *depth,
            config.qd_block_bytes,
            config.qd_total_bytes,
            config.drop_cache,
            &mut drops,
            &sampler,
        )?);
    }
    let mut alignment = Vec::new();
    for offset_mod in &config.alignment_offsets {
        alignment.push(alignment_point(
            &file,
            &config.file,
            len,
            *offset_mod,
            config.alignment_read_bytes,
            config.drop_cache,
            &mut drops,
            &sampler,
        )?);
    }
    alignment.push(overread_point(
        &file,
        &config.file,
        len,
        config.overread_bytes,
        config.alignment_read_bytes,
        config.drop_cache,
        &mut drops,
        &mut buffer,
        &sampler,
    )?);
    let checksum = checksum_point(
        &file,
        &config.file,
        len,
        config.checksum_bytes,
        config.alignment_read_bytes,
        config.drop_cache,
        &mut drops,
        &sampler,
    )?;

    let device_after = sampler.snapshot();
    let device = match (sampler.name.as_ref(), device_before, device_after) {
        (Some(name), Some(before), Some(after)) => Some(DeviceBlock {
            name: name.clone(),
            before,
            after,
            delta: after.delta(&before),
            idle_delta: idle_delta.unwrap_or_default(),
            idle_window_ns,
        }),
        _ => None,
    };
    notes.push(
        "device counters are read from /sys/block/<dev>/stat and are device-wide, not \
         per-process: the idle-window delta is the measured noise floor, and background host \
         I/O is not excluded"
            .to_owned(),
    );
    notes.push(
        "the file is smaller than host RAM, so coldness is enforced by POSIX_FADV_DONTNEED \
         and evidenced by the cold/warm probe ratio and by device bytes approximately equal to \
         requested bytes"
            .to_owned(),
    );
    notes.push(
        "sequential per-operation latency includes kernel readahead; it is not a pure device \
         latency"
            .to_owned(),
    );
    if mount.free_bytes < config.min_bytes {
        notes.push(format!(
            "free space on {} is {} bytes, below the {} byte minimum; no write was attempted",
            mount.target, mount.free_bytes, config.min_bytes
        ));
    }
    let plan = ScanPlan {
        block_sizes: config.block_sizes.clone(),
        sequential_bytes: config.sequential_bytes,
        random_samples: config.random_samples.clone(),
        queue_depths: config.queue_depths.clone(),
        qd_block_bytes: config.qd_block_bytes,
        qd_total_bytes: config.qd_total_bytes,
        alignment_offsets: config.alignment_offsets.clone(),
        alignment_read_bytes: config.alignment_read_bytes,
        overread_bytes: config.overread_bytes,
        checksum_bytes: config.checksum_bytes,
        drop_cache: config.drop_cache,
        seed: config.seed,
    };
    Ok(IoScanReport {
        schema: IO_SCAN_SCHEMA,
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
        file: FileInfo {
            path: config.file.clone(),
            len_bytes: len,
        },
        plan,
        device,
        cache_control: CacheControl {
            dropped: config.drop_cache,
            drops,
            warm_probe_ns,
            cold_probe_ns,
            cold_warm_ratio,
            drop_effective: cold_warm_ratio > 2.0,
        },
        sequential,
        sequential_noreadahead,
        random,
        queue_depth,
        alignment,
        checksum,
        notes,
    })
}

fn validate(config: &IoScanConfig) -> Result<()> {
    if config.file.as_os_str().is_empty() {
        return Err(StoreError::config("no measurement file supplied"));
    }
    if config.block_sizes.is_empty() {
        return Err(StoreError::config("no block sizes supplied"));
    }
    if config.block_sizes.contains(&0) {
        return Err(StoreError::config("block size is zero"));
    }
    if config.random_samples.len() != config.block_sizes.len() {
        return Err(StoreError::config(
            "random_samples must have one entry per block size",
        ));
    }
    if config.queue_depths.contains(&0) {
        return Err(StoreError::config("queue depth is zero"));
    }
    if config.qd_block_bytes == 0 || config.alignment_read_bytes == 0 || config.checksum_bytes == 0
    {
        return Err(StoreError::config(
            "qd_block_bytes, alignment_read_bytes and checksum_bytes must be non-zero",
        ));
    }
    Ok(())
}

fn drop_if_cold(path: &Path, cold: bool, drops: &mut Vec<AdviceReport>) -> Result<()> {
    if cold {
        drops.push(PageCache::drop_file(path)?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sequential_point(
    file: &File,
    path: &Path,
    len: u64,
    block: u64,
    total_bytes: u64,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    sampler: &DeviceSampler,
    no_readahead: bool,
) -> Result<SequentialPoint> {
    let bytes = total_bytes.min(len.saturating_sub(block));
    if bytes < block {
        return Err(StoreError::precondition(
            "file is too small for a sequential point",
        ));
    }
    let operations = bytes / block;
    drop_if_cold(path, cold, drops)?;
    let window = sampler.window();
    if no_readahead {
        PageCache::random(file, 0)?;
    } else {
        PageCache::sequential(file, 0)?;
    }
    let mut buffer = vec![0_u8; usize::try_from(block)?];
    let mut samples = Vec::with_capacity(usize::try_from(operations)?);
    let start = Instant::now();
    for index in 0..operations {
        let offset = index * block;
        let op_start = Instant::now();
        file.read_exact_at(&mut buffer, offset)?;
        samples.push(u64::try_from(op_start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        std::hint::black_box(&buffer);
    }
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (device_bytes, device_amplification) = window.close(sampler, operations * block);
    let (_, p50, p95, p99, max) = percentiles(&mut samples);
    Ok(SequentialPoint {
        block_bytes: block,
        bytes: operations * block,
        operations,
        elapsed_ns,
        throughput_mib_s: mib_per_second(operations * block, elapsed_ns),
        p50_ns: p50,
        p95_ns: p95,
        p99_ns: p99,
        max_ns: max,
        device_bytes,
        device_amplification,
        cold,
    })
}

#[allow(clippy::too_many_arguments)]
fn random_point(
    file: &File,
    path: &Path,
    len: u64,
    block: u64,
    samples: usize,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    seed: u64,
    buffer: &mut [u8],
    sampler: &DeviceSampler,
) -> Result<RandomPoint> {
    let usable = len.saturating_sub(block);
    if usable < block {
        return Err(StoreError::precondition(
            "file is too small for random reads",
        ));
    }
    drop_if_cold(path, cold, drops)?;
    PageCache::random(file, 0)?;
    let window = sampler.window();
    let mut state = SplitMix64::new(seed);
    let mut latencies = Vec::with_capacity(samples);
    let slice = &mut buffer[..usize::try_from(block)?];
    for _ in 0..samples {
        let offset = state.next_below(usable / block) * block;
        let op_start = Instant::now();
        file.read_exact_at(slice, offset)?;
        latencies.push(u64::try_from(op_start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        std::hint::black_box(&slice[0]);
    }
    let bytes_requested = block.saturating_mul(samples as u64);
    let (device_bytes, device_amplification) = window.close(sampler, bytes_requested);
    let mut sorted = latencies.clone();
    let (count, p50, p95, p99, max) = percentiles(&mut sorted);
    let total: u64 = latencies.iter().sum();
    Ok(RandomPoint {
        block_bytes: block,
        samples: count as u64,
        bytes_requested,
        device_bytes,
        device_amplification,
        p50_ns: p50,
        p95_ns: p95,
        p99_ns: p99,
        min_ns: *latencies.iter().min().unwrap_or(&0),
        max_ns: max,
        mean_ns: if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        },
        cold,
    })
}

#[allow(clippy::too_many_arguments)]
fn qd_point(
    file: &File,
    path: &Path,
    len: u64,
    depth: usize,
    block: u64,
    total_bytes: u64,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    sampler: &DeviceSampler,
) -> Result<QueueDepthPoint> {
    let bytes = total_bytes.min(len.saturating_sub(block));
    let per_reader = bytes / depth as u64;
    let operations = per_reader / block;
    if operations == 0 {
        return Err(StoreError::precondition(
            "file is too small for a queue-depth point",
        ));
    }
    drop_if_cold(path, cold, drops)?;
    PageCache::random(file, 0)?;
    let window = sampler.window();
    let latencies: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(depth);
        for reader in 0..depth {
            let file = file.try_clone()?;
            let latencies = &latencies;
            let base = reader as u64 * per_reader;
            handles.push(scope.spawn(move || -> Result<()> {
                let mut buffer = vec![0_u8; usize::try_from(block)?];
                let mut local = Vec::with_capacity(usize::try_from(operations)?);
                for index in 0..operations {
                    let offset = base + index * block;
                    let op_start = Instant::now();
                    file.read_exact_at(&mut buffer, offset)?;
                    local.push(u64::try_from(op_start.elapsed().as_nanos()).unwrap_or(u64::MAX));
                    std::hint::black_box(&buffer[0]);
                }
                let mut guard = match latencies.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                guard.extend(local);
                Ok(())
            }));
        }
        for handle in handles {
            let joined = handle
                .join()
                .map_err(|_| StoreError::precondition("a queue-depth reader thread panicked"))?;
            joined?;
        }
        Ok(())
    })?;
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let bytes_read = per_reader * depth as u64;
    let (device_bytes, device_amplification) = window.close(sampler, bytes_read);
    let mut collected = match latencies.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    let (_, p50, p95, p99, _max) = percentiles(&mut collected);
    Ok(QueueDepthPoint {
        queue_depth: depth as u64,
        block_bytes: block,
        bytes: bytes_read,
        elapsed_ns,
        throughput_mib_s: mib_per_second(bytes_read, elapsed_ns),
        p50_ns: p50,
        p95_ns: p95,
        p99_ns: p99,
        device_bytes,
        device_amplification,
        cold,
    })
}

#[allow(clippy::too_many_arguments)]
fn alignment_point(
    file: &File,
    path: &Path,
    len: u64,
    offset_mod: u64,
    read_bytes: u64,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    sampler: &DeviceSampler,
) -> Result<AlignmentPoint> {
    let base = aligned_offset(len, read_bytes, 65_536);
    let offset = base.saturating_add(offset_mod);
    if offset.saturating_add(read_bytes) > len {
        return Err(StoreError::precondition(
            "file is too small for the alignment point",
        ));
    }
    drop_if_cold(path, cold, drops)?;
    PageCache::random(file, 0)?;
    let window = sampler.window();
    let mut buffer = vec![0_u8; usize::try_from(read_bytes)?];
    let start = Instant::now();
    file.read_exact_at(&mut buffer, offset)?;
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (device_bytes, device_amplification) = window.close(sampler, read_bytes);
    Ok(AlignmentPoint {
        label: format!("shard-extent offset aligned at {offset_mod}"),
        offset,
        alignment: offset % 65_536,
        bytes_requested: read_bytes,
        bytes_useful: read_bytes,
        device_bytes,
        requested_per_useful: 1.0,
        device_per_useful: device_amplification,
        elapsed_ns,
        cold,
    })
}

#[allow(clippy::too_many_arguments)]
fn overread_point(
    file: &File,
    path: &Path,
    len: u64,
    overread_bytes: u64,
    useful_bytes: u64,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    buffer: &mut [u8],
    sampler: &DeviceSampler,
) -> Result<AlignmentPoint> {
    let offset = aligned_offset(len, overread_bytes, 65_536);
    if offset.saturating_add(overread_bytes) > len {
        return Err(StoreError::precondition(
            "file is too small for the over-read point",
        ));
    }
    drop_if_cold(path, cold, drops)?;
    PageCache::random(file, 0)?;
    let window = sampler.window();
    let slice = &mut buffer[..usize::try_from(overread_bytes)?];
    let start = Instant::now();
    file.read_exact_at(slice, offset)?;
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (device_bytes, device_per_useful) = window.close(sampler, useful_bytes);
    Ok(AlignmentPoint {
        label: format!("over-read {overread_bytes} bytes to use {useful_bytes}"),
        offset,
        alignment: offset % 65_536,
        bytes_requested: overread_bytes,
        bytes_useful: useful_bytes,
        device_bytes,
        requested_per_useful: overread_bytes as f64 / useful_bytes as f64,
        device_per_useful,
        elapsed_ns,
        cold,
    })
}

#[allow(clippy::too_many_arguments)]
fn checksum_point(
    file: &File,
    path: &Path,
    len: u64,
    checksum_bytes: u64,
    shard_bytes: u64,
    cold: bool,
    drops: &mut Vec<AdviceReport>,
    sampler: &DeviceSampler,
) -> Result<ChecksumPoint> {
    let bytes = checksum_bytes.min(len);
    if bytes == 0 {
        return Err(StoreError::precondition("file is empty"));
    }
    let mut buffer = vec![0_u8; 1024 * 1024];
    // Warm pass: read once to populate, then hash while re-reading.
    PageCache::sequential(file, 0)?;
    let mut offset = 0_u64;
    while offset < bytes {
        let want = usize::try_from((bytes - offset).min(buffer.len() as u64))?;
        file.read_exact_at(&mut buffer[..want], offset)?;
        offset += want as u64;
    }
    let mut hasher = Sha256Builder::new();
    offset = 0;
    let warm_start = Instant::now();
    while offset < bytes {
        let want = usize::try_from((bytes - offset).min(buffer.len() as u64))?;
        file.read_exact_at(&mut buffer[..want], offset)?;
        hasher.update(&buffer[..want]);
        offset += want as u64;
    }
    let warm_elapsed_ns = u64::try_from(warm_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let digest = hasher.finish();
    // Cold pass: drop, then read and hash.
    drop_if_cold(path, cold, drops)?;
    let window = sampler.window();
    let mut hasher = Sha256Builder::new();
    offset = 0;
    let cold_start = Instant::now();
    while offset < bytes {
        let want = usize::try_from((bytes - offset).min(buffer.len() as u64))?;
        file.read_exact_at(&mut buffer[..want], offset)?;
        hasher.update(&buffer[..want]);
        offset += want as u64;
    }
    let cold_elapsed_ns = u64::try_from(cold_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (cold_device_bytes, _) = window.close(sampler, bytes);
    let cold_digest = hasher.finish();
    if cold_digest != digest {
        return Err(StoreError::precondition(
            "cold and warm checksum passes disagree; the file changed under the scan",
        ));
    }
    // Single-shard hash cost.
    let mut shard_buffer = vec![0_u8; usize::try_from(shard_bytes)?];
    let shard_offset = aligned_offset(len, shard_bytes, 65_536);
    file.read_exact_at(&mut shard_buffer, shard_offset)?;
    let shard_start = Instant::now();
    let mut hasher = Sha256Builder::new();
    hasher.update(&shard_buffer);
    let _ = hasher.finish();
    let shard_hash_ns = u64::try_from(shard_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    Ok(ChecksumPoint {
        algorithm: "sha256",
        bytes,
        warm_elapsed_ns,
        warm_throughput_mib_s: mib_per_second(bytes, warm_elapsed_ns),
        warm_ns_per_gib: if gib == 0.0 {
            0.0
        } else {
            warm_elapsed_ns as f64 / gib
        },
        cold_elapsed_ns,
        cold_throughput_mib_s: mib_per_second(bytes, cold_elapsed_ns),
        cold_device_bytes,
        digest,
        shard_bytes,
        shard_hash_ns,
    })
}

fn mib_per_second(bytes: u64, elapsed_ns: u64) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0)) / (elapsed_ns as f64 / 1_000_000_000.0)
}

fn aligned_offset(len: u64, size: u64, alignment: u64) -> u64 {
    let span = len.saturating_sub(size).min(alignment);
    span - (span % alignment.max(1))
}

/// Deterministic SplitMix64 sampler.
#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        self.next() % bound
    }
}

/// Current wall-clock timestamp as `YYYY-MM-DDTHH:MM:SSZ`, UTC.
#[must_use]
pub fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format_timestamp(seconds, false)
}

fn stamp_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format_timestamp(seconds, true)
}

/// Format a Unix timestamp as the compact receipt stamp
/// (`YYYYMMDDTHHMMSSZ`), UTC.
#[must_use]
pub fn format_stamp(seconds: u64) -> String {
    format_timestamp(seconds, true)
}

/// Format a Unix timestamp as `YYYY-MM-DDTHH:MM:SSZ` or `YYYYMMDDTHHMMSSZ`.
fn format_timestamp(seconds: u64, compact: bool) -> String {
    let days = seconds / 86_400;
    let remainder = seconds % 86_400;
    let (hour, minute, second) = (remainder / 3600, (remainder % 3600) / 60, remainder % 60);
    let (year, month, day) = civil_from_days(days);
    if compact {
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
    }
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as u64, m, d)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_formatting_is_utc_and_compact_variant_is_stable() {
        assert_eq!(format_timestamp(0, true), "19700101T000000Z");
        assert_eq!(format_timestamp(0, false), "1970-01-01T00:00:00Z");
        assert_eq!(format_timestamp(1_700_000_000, true), "20231114T221320Z");
    }

    #[test]
    fn sampler_is_deterministic_and_bounded() {
        let mut left = SplitMix64::new(7);
        let mut right = SplitMix64::new(7);
        for _ in 0..64 {
            let value = left.next();
            assert_eq!(value, right.next());
        }
        let mut bounded = SplitMix64::new(11);
        for _ in 0..64 {
            assert!(bounded.next_below(1000) < 1000);
        }
    }

    #[test]
    fn aligned_offsets_stay_inside_the_file() {
        assert_eq!(aligned_offset(1000, 100, 4096), 0);
        assert_eq!(aligned_offset(20_000, 4096, 4096), 4096);
    }

    #[test]
    fn small_scan_over_a_synthetic_file_produces_a_report() {
        let temp = selvedge_core::testutil::TempDir::new("io-scan").unwrap();
        let path = temp.path().join("data.bin");
        let mut data = vec![0_u8; 2 * 1024 * 1024];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        std::fs::write(&path, &data).unwrap();
        let config = IoScanConfig {
            file: path,
            workdir: temp.path().to_path_buf(),
            block_sizes: vec![4096, 65_536],
            sequential_bytes: 1024 * 1024,
            random_samples: vec![32, 16],
            queue_depths: vec![1, 4],
            qd_block_bytes: 4096,
            qd_total_bytes: 256 * 1024,
            alignment_offsets: vec![512, 4096],
            alignment_read_bytes: 256 * 1024,
            overread_bytes: 512 * 1024,
            checksum_bytes: 512 * 1024,
            drop_cache: false,
            min_bytes: 1024 * 1024,
            seed: 1234,
            stamp: Some("20260101T000000Z".to_owned()),
            revision: Some("test".to_owned()),
            argv: vec!["selvedge".to_owned()],
        };
        let report = run(&config).unwrap();
        assert_eq!(report.sequential.len(), 2);
        assert_eq!(report.sequential_noreadahead.len(), 2);
        assert_eq!(report.random.len(), 2);
        assert_eq!(report.queue_depth.len(), 2);
        assert!(report.checksum.warm_elapsed_ns > 0);
        assert_eq!(report.alignment.len(), 3);
        assert!(!report.cache_control.dropped);
        assert_eq!(report.stamp, "20260101T000000Z");
        assert_eq!(report.revision, "test");
    }
}
