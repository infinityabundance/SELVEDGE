//! Calibration of the transfer model against measured receipts.
//!
//! The simulator does not assume a rate. It is calibrated by reading the real
//! per-request JSONL that `selvedge store bench --metrics-dir` writes and the
//! `io-scan` receipts, and it records:
//!
//! - the empirical distribution of every measured component (transfer,
//!   SHA-256 verification, queue wait, cache insertion, eviction, host-RAM hit
//!   copy) as a 257-point quantile grid;
//! - parametric fits (exponential and lognormal) with quantile residuals, so a
//!   reader can see whether an exponential assumption would have been wrong;
//! - a size law for reads of a size that was not measured, fitted log-log to
//!   the `io-scan` random-read p50 curve and labelled as the only
//!   extrapolation in the model.
//!
//! The distribution of record is the **empirical** one. Parametric fits are
//! reported for interpolation and for sensitivity analysis; the simulator never
//! silently substitutes a curve the data did not support.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use selvedge_core::{Sha256Builder, Sha256Digest, ShardId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::prng::{SplitMix64, empirical_quantile};
use crate::trace::TraceLimits;

/// Schema of a calibration receipt.
pub const CALIBRATION_SCHEMA: &str = "selvedge/sim-calibration/v1";

/// Number of quantile points retained from a sample.
const GRID_POINTS: usize = 257;

/// Cap on metrics lines read from one file.
const MAX_METRICS_LINES: u64 = 5_000_000;

/// One input file, pinned by digest so the calibration cannot drift.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptRef {
    /// Repository-relative or absolute path.
    pub path: String,
    /// SHA-256 of the file's bytes.
    pub sha256: Sha256Digest,
    /// Bytes read.
    pub bytes: u64,
    /// Role the file played, e.g. `transfer+verify` or `size_law`.
    pub role: String,
}

impl ReceiptRef {
    /// Hash an existing file and build a reference.
    ///
    /// # Errors
    ///
    /// Propagates I/O failures.
    pub fn of(path: &Path, role: &str) -> Result<Self> {
        let mut builder = Sha256Builder::new();
        let mut bytes = 0_u64;
        let mut file = std::fs::File::open(path)?;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = std::io::Read::read(&mut file, &mut buffer)?;
            if read == 0 {
                break;
            }
            builder.update(&buffer[..read]);
            bytes += read as u64;
        }
        Ok(Self {
            path: path.display().to_string(),
            sha256: builder.finish(),
            bytes,
            role: role.to_owned(),
        })
    }
}

/// One `selvedge/store-metrics/v1` line, restricted to the fields the
/// calibration and the replay check need.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsRecord {
    /// Always `selvedge/store-metrics/v1`.
    pub schema: String,
    /// `read`, `prefetch` or `prefetch_batch`.
    pub event: String,
    /// Wall clock, Unix nanoseconds.
    pub wall_unix_ns: u64,
    /// Monotonic nanoseconds since the store epoch.
    pub mono_ns: u64,
    /// Shard.
    pub shard: ShardId,
    /// `nvme` or `host_ram`.
    pub source: String,
    /// Bytes the shard needs.
    pub bytes_requested: u64,
    /// Bytes that arrived.
    pub bytes_completed: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Bytes moved but not used.
    pub bytes_wasted: u64,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Time waiting for the cache lock or queue admission.
    pub queue_wait_ns: u64,
    /// Time moving bytes.
    pub transfer_ns: u64,
    /// Time hashing and comparing.
    pub verify_ns: u64,
    /// Time copying into the cache.
    pub cache_insert_ns: u64,
    /// Time spent selecting and evicting victims.
    pub eviction_ns: u64,
    /// Exposed consumer stall.
    pub stall_ns: u64,
    /// Measured `device_bytes / bytes_used`.
    pub read_amplification: f64,
    /// Whether the bytes came from the cache.
    pub hit: bool,
    /// Whether the bytes were verified before use.
    pub verified: bool,
    /// Verification method.
    pub verification: String,
    /// Entries evicted to admit this shard.
    pub evicted_entries: u64,
}

/// One measured read sequence, used for replay validation.
#[derive(Clone, Debug, Default)]
pub struct MetricsSequence {
    /// Reads in measured order.
    pub reads: Vec<MetricsRecord>,
    /// Prefetch events seen (not part of the required sequence).
    pub prefetch_events: u64,
}

/// Read a metrics JSONL file with bounded input.
///
/// # Errors
///
/// Rejects a wrong schema, an over-limit line or record count, malformed JSON
/// and I/O failures.
pub fn read_metrics(path: &Path, limits: TraceLimits) -> Result<MetricsSequence> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut sequence = MetricsSequence::default();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if line.len() as u64 > limits.max_line_bytes {
            return Err(SimError::limit(
                "metrics_line_bytes",
                line.len() as u64,
                limits.max_line_bytes,
            ));
        }
        if sequence.reads.len() as u64 + sequence.prefetch_events >= MAX_METRICS_LINES {
            return Err(SimError::limit(
                "metrics_records",
                sequence.reads.len() as u64 + 1,
                MAX_METRICS_LINES,
            ));
        }
        let record: MetricsRecord = serde_json::from_str(&line)?;
        if record.schema != "selvedge/store-metrics/v1" {
            return Err(SimError::invalid(
                "metrics_schema",
                "line is not selvedge/store-metrics/v1",
            ));
        }
        if record.event == "read" {
            sequence.reads.push(record);
        } else {
            sequence.prefetch_events += 1;
        }
    }
    Ok(sequence)
}

/// Descriptive statistics and a retained quantile grid for one sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleStats {
    /// Sample size.
    pub n: u64,
    /// Mean.
    pub mean: f64,
    /// Population standard deviation.
    pub sd: f64,
    /// Minimum.
    pub min: u64,
    /// p50.
    pub p50: u64,
    /// p90.
    pub p90: u64,
    /// p95.
    pub p95: u64,
    /// p99.
    pub p99: u64,
    /// Maximum.
    pub max: u64,
    /// Quantile grid, `GRID_POINTS` entries ascending; entry `k` is the
    /// `k / (len - 1)` quantile.
    pub grid: Vec<u64>,
}

impl SampleStats {
    /// Build statistics from a sample of nanoseconds.
    ///
    /// # Errors
    ///
    /// Rejects an empty sample.
    pub fn of(sample: &[u64]) -> Result<Self> {
        if sample.is_empty() {
            return Err(SimError::invalid("sample", "is empty"));
        }
        let mut sorted = sample.to_vec();
        sorted.sort_unstable();
        let n = sorted.len() as f64;
        let mean = sorted.iter().map(|value| *value as f64).sum::<f64>() / n;
        let variance = if sorted.len() > 1 {
            sorted
                .iter()
                .map(|value| {
                    let delta = *value as f64 - mean;
                    delta * delta
                })
                .sum::<f64>()
                / n
        } else {
            0.0
        };
        let grid = (0..GRID_POINTS)
            .map(|index| {
                let q = index as f64 / (GRID_POINTS - 1) as f64;
                empirical_quantile(&sorted, q).map_or(0, |value| value.round() as u64)
            })
            .collect();
        Ok(Self {
            n: sorted.len() as u64,
            mean,
            sd: variance.sqrt(),
            min: sorted[0],
            p50: crate::prng::percentile_u64(&sorted, 50.0),
            p90: crate::prng::percentile_u64(&sorted, 90.0),
            p95: crate::prng::percentile_u64(&sorted, 95.0),
            p99: crate::prng::percentile_u64(&sorted, 99.0),
            max: *sorted.last().unwrap_or(&0),
            grid,
        })
    }

    /// Interpolated empirical quantile from the retained grid.
    ///
    /// # Errors
    ///
    /// Rejects an empty grid.
    pub fn quantile(&self, q: f64) -> Result<f64> {
        empirical_quantile(&self.grid, q)
    }

    /// Deterministic sample from the retained grid.
    ///
    /// # Errors
    ///
    /// Rejects an empty grid.
    pub fn sample(&self, rng: &mut SplitMix64) -> Result<f64> {
        let u = rng.next_f64();
        self.quantile(u)
    }
}

/// One parametric fit and its quantile residuals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitReport {
    /// Quantity fitted, e.g. `transfer_ns`.
    pub quantity: String,
    /// `exponential` or `lognormal`.
    pub family: String,
    /// Fitted parameters, family-specific.
    pub params: BTreeMap<String, f64>,
    /// Empirical quantile, fitted quantile and relative error.
    pub residuals: Vec<QuantileResidual>,
    /// Largest absolute relative error.
    pub max_abs_rel_error: f64,
    /// Mean absolute relative error.
    pub mean_abs_rel_error: f64,
}

/// One quantile residual.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantileResidual {
    /// Quantile.
    pub q: f64,
    /// Empirical value.
    pub empirical_ns: f64,
    /// Fitted value.
    pub fitted_ns: f64,
    /// `(fitted - empirical) / empirical`.
    pub rel_error: f64,
}

/// Quantiles used for the residual table.
pub const RESIDUAL_QUANTILES: [f64; 7] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];

/// Inverse standard-normal CDF (Acklam's rational approximation).
///
/// Accurate to about 1.15e-9 over the whole range, which is far below the
/// measurement resolution used here.
#[must_use]
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOWER: f64 = 0.024_25;
    const UPPER: f64 = 1.0 - LOWER;
    let p = p.clamp(1e-16, 1.0 - 1e-16);
    if p < LOWER || p > UPPER {
        let q = if p < LOWER { p } else { 1.0 - p };
        let r = (-2.0 * q.ln()).sqrt();
        // Acklam's tail polynomial is negated in the published formula; the
        // lower tail then takes the negation of that value. Omitting the
        // leading minus silently swaps the two tails.
        let numerator = ((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5];
        let denominator = (((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0;
        let mut value = -numerator / denominator;
        if p < LOWER {
            value = -value;
        }
        return value;
    }
    let q = p - 0.5;
    let r = q * q;
    let mut value = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    // One Halley refinement step.
    let e = 0.5 * erfc(-value / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (value * value / 2.0).exp();
    value -= u / (1.0 + value * u / 2.0);
    value
}

/// Complementary error function (Numerical Recipes `erfcc`, |error| < 1.2e-7).
#[must_use]
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let answer = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { answer } else { 2.0 - answer }
}

/// Fit exponential and lognormal families and report quantile residuals.
///
/// # Errors
///
/// Rejects an empty or all-zero sample.
pub fn fit_reports(quantity: &str, stats: &SampleStats) -> Result<Vec<FitReport>> {
    if stats.n == 0 {
        return Err(SimError::invalid("sample", "is empty"));
    }
    if stats.p50 == 0 {
        return Err(SimError::invalid(
            "sample",
            "median is zero, so relative residuals are undefined",
        ));
    }
    let mean = stats.mean.max(f64::MIN_POSITIVE);
    // Exponential: rate = 1/mean.
    let rate = 1.0 / mean;
    // Lognormal from the quantile grid: the grid is uniform in probability, so
    // the mean and variance of `ln x` over it are the distribution's moments.
    let logs: Vec<f64> = stats
        .grid
        .iter()
        .map(|value| (*value).max(1) as f64)
        .map(f64::ln)
        .collect();
    let mu = logs.iter().sum::<f64>() / logs.len() as f64;
    let variance = logs
        .iter()
        .map(|value| (value - mu) * (value - mu))
        .sum::<f64>()
        / logs.len() as f64;
    let sigma = variance.sqrt().max(1e-6);
    let mut reports = Vec::new();
    for family in ["exponential", "lognormal"] {
        let mut residuals = Vec::new();
        let mut max_abs = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        for q in RESIDUAL_QUANTILES {
            let empirical = stats.quantile(q)?;
            let fitted = if family == "exponential" {
                -(1.0 - q).ln() / rate
            } else {
                (mu + sigma * normal_quantile(q)).exp()
            };
            let rel = if empirical > 0.0 {
                (fitted - empirical) / empirical
            } else {
                0.0
            };
            max_abs = max_abs.max(rel.abs());
            sum_abs += rel.abs();
            residuals.push(QuantileResidual {
                q,
                empirical_ns: empirical,
                fitted_ns: fitted,
                rel_error: rel,
            });
        }
        let mut params = BTreeMap::new();
        if family == "exponential" {
            params.insert("rate_per_ns".to_owned(), rate);
            params.insert("mean_ns".to_owned(), mean);
        } else {
            params.insert("mu".to_owned(), mu);
            params.insert("sigma".to_owned(), sigma);
        }
        reports.push(FitReport {
            quantity: quantity.to_owned(),
            family: family.to_owned(),
            params,
            residuals,
            max_abs_rel_error: max_abs,
            mean_abs_rel_error: sum_abs / RESIDUAL_QUANTILES.len() as f64,
        });
    }
    Ok(reports)
}

/// Piecewise log-log size law fitted to measured read latencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SizeLaw {
    /// Measured points `(bytes, p50_ns)`, ascending by bytes.
    pub points: Vec<SizePoint>,
    /// Log-log slope of each segment, same length as `points - 1`.
    pub segment_slopes: Vec<f64>,
    /// Provenance string, e.g. the io-scan receipt.
    pub source: String,
}

/// One measured point of the size law.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SizePoint {
    /// Request size.
    pub bytes: u64,
    /// Measured p50 latency.
    pub p50_ns: u64,
}

impl SizeLaw {
    /// Fit a size law from measured points.
    ///
    /// # Errors
    ///
    /// Rejects fewer than two points, unsorted or non-positive sizes, and a
    /// non-positive latency.
    pub fn fit(mut points: Vec<SizePoint>, source: &str) -> Result<Self> {
        if points.len() < 2 {
            return Err(SimError::invalid("size_law", "needs at least two points"));
        }
        points.sort_by_key(|point| point.bytes);
        for point in &points {
            if point.bytes == 0 || point.p50_ns == 0 {
                return Err(SimError::invalid("size_law", "point is zero"));
            }
        }
        let mut slopes = Vec::with_capacity(points.len() - 1);
        for pair in points.windows(2) {
            let [low, high] = pair else {
                continue;
            };
            let slope = (high.p50_ns as f64 / low.p50_ns as f64).ln()
                / (high.bytes as f64 / low.bytes as f64).ln();
            slopes.push(slope);
        }
        Ok(Self {
            points,
            segment_slopes: slopes,
            source: source.to_owned(),
        })
    }

    /// Predicted p50 latency for a request of `bytes`, in nanoseconds.
    ///
    /// Interpolates log-log between measured points. Outside the measured range
    /// it extrapolates with the nearest segment's slope; that extrapolation is
    /// the one approximation in the model and is labelled as such in
    /// `docs/SIMULATOR.md`.
    ///
    /// # Errors
    ///
    /// Rejects a zero-byte request.
    pub fn latency_ns(&self, bytes: u64) -> Result<f64> {
        if bytes == 0 {
            return Err(SimError::invalid("size_law", "request is zero bytes"));
        }
        let first = self.points[0];
        let last = *self.points.last().unwrap_or(&first);
        if bytes <= first.bytes {
            return Ok(first.p50_ns as f64 * (bytes as f64 / first.bytes as f64));
        }
        if bytes >= last.bytes {
            let slope = self.segment_slopes.last().copied().unwrap_or(1.0);
            return Ok(last.p50_ns as f64 * (bytes as f64 / last.bytes as f64).powf(slope));
        }
        for (index, pair) in self.points.windows(2).enumerate() {
            let [low, high] = pair else {
                continue;
            };
            if bytes >= low.bytes && bytes <= high.bytes {
                let slope = self.segment_slopes[index];
                let ratio = bytes as f64 / low.bytes as f64;
                return Ok(low.p50_ns as f64 * ratio.powf(slope));
            }
        }
        Ok(last.p50_ns as f64)
    }
}

/// The calibrated cost model the simulator runs on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// Always [`CALIBRATION_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub stamp: String,
    /// Revision the inputs were produced at.
    pub revision: String,
    /// Input files, hashed.
    pub inputs: Vec<ReceiptRef>,
    /// Reference shard size the empirical models were measured at.
    pub shard_bytes: u64,
    /// Cold per-shard transfer time.
    pub transfer: SampleStats,
    /// SHA-256 verification time for one cold shard.
    pub verify: SampleStats,
    /// Queue-wait time observed on the miss path.
    pub queue_wait: SampleStats,
    /// Cache insertion time.
    pub insert: SampleStats,
    /// Eviction time.
    pub eviction: SampleStats,
    /// Host-RAM hit copy time.
    pub hit_copy: SampleStats,
    /// Size law for requests of another size.
    pub size_law: SizeLaw,
    /// Parametric fits with residuals.
    pub fits: Vec<FitReport>,
    /// Distribution of record for each component.
    pub empirical_is_model_of_record: bool,
    /// Caveats a reader must carry.
    pub notes: Vec<String>,
}

impl Calibration {
    /// Transfer time for `bytes`, sampled deterministically.
    ///
    /// # Errors
    ///
    /// Propagates size-law and sampling failures.
    pub fn sample_transfer_ns(&self, rng: &mut SplitMix64, bytes: u64) -> Result<f64> {
        self.scale_and_sample(&self.transfer, rng, bytes)
    }

    /// Verification time for `bytes`.
    ///
    /// Hashing is linear in bytes, so this scales linearly rather than through
    /// the device size law.
    ///
    /// # Errors
    ///
    /// Propagates sampling failures.
    pub fn sample_verify_ns(&self, rng: &mut SplitMix64, bytes: u64) -> Result<f64> {
        self.linear_and_sample(&self.verify, rng, bytes)
    }

    /// Host-RAM copy time for `bytes`.
    ///
    /// A copy is linear in bytes.
    ///
    /// # Errors
    ///
    /// Propagates sampling failures.
    pub fn sample_hit_ns(&self, rng: &mut SplitMix64, bytes: u64) -> Result<f64> {
        self.linear_and_sample(&self.hit_copy, rng, bytes)
    }

    /// Cache insertion time for one shard.
    ///
    /// # Errors
    ///
    /// Propagates sampling failures.
    pub fn sample_insert_ns(&self, rng: &mut SplitMix64) -> Result<f64> {
        self.insert.sample(rng)
    }

    /// Eviction time for `count` evictions.
    ///
    /// # Errors
    ///
    /// Propagates sampling failures.
    pub fn sample_eviction_ns(&self, rng: &mut SplitMix64, count: u64) -> Result<f64> {
        Ok(self.eviction.sample(rng)? * count as f64)
    }

    /// Queue wait observed on the miss path.
    ///
    /// # Errors
    ///
    /// Propagates sampling failures.
    pub fn sample_queue_wait_ns(&self, rng: &mut SplitMix64) -> Result<f64> {
        self.queue_wait.sample(rng)
    }

    fn scale_and_sample(
        &self,
        stats: &SampleStats,
        rng: &mut SplitMix64,
        bytes: u64,
    ) -> Result<f64> {
        let sample = stats.sample(rng)?;
        let reference = self.size_law.latency_ns(self.shard_bytes)?;
        if reference <= 0.0 {
            return Ok(sample);
        }
        let scaled = self.size_law.latency_ns(bytes)? / reference;
        Ok(sample * scaled)
    }

    fn linear_and_sample(
        &self,
        stats: &SampleStats,
        rng: &mut SplitMix64,
        bytes: u64,
    ) -> Result<f64> {
        let sample = stats.sample(rng)?;
        if self.shard_bytes == 0 {
            return Ok(sample);
        }
        Ok(sample * bytes as f64 / self.shard_bytes as f64)
    }

    /// Load a calibration receipt.
    ///
    /// # Errors
    ///
    /// Rejects a wrong schema and propagates I/O and JSON failures.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let calibration: Self = serde_json::from_slice(&bytes)?;
        if calibration.schema != CALIBRATION_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                format!("calibration schema is not {CALIBRATION_SCHEMA}"),
            ));
        }
        Ok(calibration)
    }

    /// Write the receipt.
    ///
    /// # Errors
    ///
    /// Propagates I/O and serialization failures.
    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// One metrics input for calibration.
#[derive(Clone, Debug)]
pub struct CalibrationInput {
    /// Role label recorded in the receipt.
    pub role: String,
    /// Path to a `store-metrics` JSONL file.
    pub path: PathBuf,
}

/// Build a calibration from measured metrics and an `io-scan` receipt.
///
/// # Errors
///
/// Rejects empty inputs, a missing size-law section, non-finite statistics and
/// any I/O failure.
pub fn calibrate(
    inputs: &[CalibrationInput],
    io_scan_path: &Path,
    revision: &str,
    stamp: &str,
) -> Result<Calibration> {
    if inputs.is_empty() {
        return Err(SimError::invalid("inputs", "no metrics files supplied"));
    }
    let mut refs = Vec::new();
    let mut transfer = Vec::new();
    let mut verify = Vec::new();
    let mut queue_wait = Vec::new();
    let mut insert = Vec::new();
    let mut eviction = Vec::new();
    let mut hit_copy = Vec::new();
    let mut sizes = Vec::new();
    for input in inputs {
        refs.push(ReceiptRef::of(&input.path, &input.role)?);
        let sequence = read_metrics(&input.path, TraceLimits::default())?;
        for read in &sequence.reads {
            if read.hit {
                if read.stall_ns > 0 {
                    hit_copy.push(read.stall_ns);
                }
                continue;
            }
            if read.transfer_ns > 0 {
                transfer.push(read.transfer_ns);
                sizes.push(read.bytes_requested);
            }
            if read.verify_ns > 0 {
                verify.push(read.verify_ns);
            }
            queue_wait.push(read.queue_wait_ns);
            if read.cache_insert_ns > 0 {
                insert.push(read.cache_insert_ns);
            }
            if read.eviction_ns > 0 {
                eviction.push(read.eviction_ns);
            }
        }
    }
    sizes.sort_unstable();
    let shard_bytes = if sizes.is_empty() {
        return Err(SimError::invalid(
            "shard_bytes",
            "no cold reads found in the calibration inputs",
        ));
    } else {
        sizes[sizes.len() / 2]
    };
    if transfer.is_empty() {
        return Err(SimError::invalid("transfer", "no transfer samples"));
    }
    if verify.is_empty() {
        return Err(SimError::invalid("verify", "no verification samples"));
    }
    // Non-zero defaults for components that a run may not exercise.
    if queue_wait.is_empty() {
        queue_wait.push(0);
    }
    if insert.is_empty() {
        insert.push(0);
    }
    if eviction.is_empty() {
        eviction.push(0);
    }
    if hit_copy.is_empty() {
        hit_copy.push(0);
    }
    let transfer_stats = SampleStats::of(&transfer)?;
    let verify_stats = SampleStats::of(&verify)?;
    let mut fits = fit_reports("transfer_ns", &transfer_stats)?;
    fits.extend(fit_reports("verify_ns", &verify_stats)?);

    refs.push(ReceiptRef::of(io_scan_path, "size_law")?);
    let io_bytes = std::fs::read(io_scan_path)?;
    let io: serde_json::Value = serde_json::from_slice(&io_bytes)?;
    let random = io
        .get("random")
        .and_then(|value| value.as_array())
        .ok_or_else(|| SimError::invalid("io_scan", "missing random-read section"))?;
    let mut points = Vec::new();
    for entry in random {
        let bytes = entry
            .get("block_bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| SimError::invalid("io_scan", "random point has no block_bytes"))?;
        let p50 = entry
            .get("p50_ns")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| SimError::invalid("io_scan", "random point has no p50_ns"))?;
        points.push(SizePoint { bytes, p50_ns: p50 });
    }
    if points.len() < 2 {
        return Err(SimError::invalid(
            "io_scan",
            "random-read section needs at least two sizes",
        ));
    }
    let size_law = SizeLaw::fit(points, &io_scan_path.display().to_string())?;
    Ok(Calibration {
        schema: CALIBRATION_SCHEMA.to_owned(),
        stamp: stamp.to_owned(),
        revision: revision.to_owned(),
        inputs: refs,
        shard_bytes,
        transfer: transfer_stats,
        verify: verify_stats,
        queue_wait: SampleStats::of(&queue_wait)?,
        insert: SampleStats::of(&insert)?,
        eviction: SampleStats::of(&eviction)?,
        hit_copy: SampleStats::of(&hit_copy)?,
        size_law,
        fits,
        empirical_is_model_of_record: true,
        notes: vec![
            "the empirical quantile grid is the model of record; the exponential \
             and lognormal fits are reported with residuals so a reader can see \
             which one the data support"
                .to_owned(),
            "latency for a request size that was not measured is extrapolated \
             with the log-log io-scan size law; that is the only interpolation \
             in the model"
                .to_owned(),
            "the measured engine issues one synchronous read at a time, so the \
             calibration describes a serialized path; simulated queue depth is \
             a what-if, not a measured configuration"
                .to_owned(),
        ],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn metrics_line(shard: u8, transfer: u64, verify: u64, hit: bool) -> String {
        format!(
            "{{\"schema\":\"selvedge/store-metrics/v1\",\"event\":\"read\",\
             \"wall_unix_ns\":1,\"mono_ns\":1,\
             \"shard\":\"sha256:{}\",\"source\":\"{}\",\
             \"bytes_requested\":768,\"bytes_completed\":768,\"bytes_used\":768,\
             \"bytes_wasted\":0,\"device_bytes\":{},\"queue_wait_ns\":10,\
             \"transfer_ns\":{},\"verify_ns\":{},\"cache_insert_ns\":20,\
             \"eviction_ns\":2,\"stall_ns\":{},\"read_amplification\":1.0,\
             \"hit\":{},\"verified\":true,\"verification\":\"sha256_read\",\
             \"evicted_entries\":0}}",
            hex64(shard),
            if hit { "host_ram" } else { "nvme" },
            if hit { 0 } else { 768 },
            transfer,
            verify,
            transfer + verify + 30,
            hit
        )
    }

    fn hex64(tag: u8) -> String {
        let mut text = String::new();
        for index in 0..32 {
            text.push_str(&format!("{:02x}", tag.wrapping_add(index as u8)));
        }
        text
    }

    #[test]
    fn empirical_quantiles_match_sample() {
        let sample: Vec<u64> = (1..=100).collect();
        let stats = SampleStats::of(&sample).unwrap();
        assert_eq!(stats.n, 100);
        assert_eq!(stats.min, 1);
        assert_eq!(stats.max, 100);
        assert!((stats.quantile(0.5).unwrap() - 50.5).abs() < 1.0);
        assert_eq!(stats.grid.len(), GRID_POINTS);
        assert!(stats.grid.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn normal_quantile_matches_known_values() {
        for (p, expected) in [
            (0.5, 0.0),
            (0.841_344_7, 1.0),
            (0.975, 1.959_964),
            (0.99, 2.326_348),
            (0.01, -2.326_348),
            (0.1, -1.281_552),
        ] {
            let got = normal_quantile(p);
            assert!(
                (got - expected).abs() < 1e-4,
                "normal_quantile({p}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn normal_quantile_has_unit_moments() {
        // The endpoints of a clamped grid sit at ±8.2 and would dominate the
        // second moment, so the interior grid is used.
        let zs: Vec<f64> = (1..=255)
            .map(|k| normal_quantile(k as f64 / 256.0))
            .collect();
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let variance = zs.iter().map(|z| (z - mean) * (z - mean)).sum::<f64>() / zs.len() as f64;
        assert!((mean).abs() < 1e-3, "mean {mean}");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance {variance}; q99 {} q999 {}",
            normal_quantile(0.99),
            normal_quantile(0.999)
        );
    }

    #[test]
    fn fit_reports_residuals_and_can_reject_exponential() {
        // A lognormal-shaped sample: exponential should fit worse. The sample is
        // large enough that its own tails are not the binding approximation; a
        // small sample would truncate the tail and the parametric fit would
        // inherit that truncation, which is exactly why the empirical grid is
        // the model of record.
        let mut sample = Vec::new();
        for index in 1..=20_000_u64 {
            let q = index as f64 / 20_001.0;
            let value = (1.0e5 * (3.0 + 0.8 * normal_quantile(q)).exp()).round();
            sample.push(value as u64);
        }
        let stats = SampleStats::of(&sample).unwrap();
        let reports = fit_reports("transfer_ns", &stats).unwrap();
        assert!(
            (normal_quantile(0.1) + 1.281_551_565_544_600_4).abs() < 1e-6,
            "normal_quantile(0.1) = {}",
            normal_quantile(0.1)
        );
        assert_eq!(reports.len(), 2);
        let exponential = reports
            .iter()
            .find(|report| report.family == "exponential")
            .unwrap();
        let lognormal = reports
            .iter()
            .find(|report| report.family == "lognormal")
            .unwrap();
        assert!(
            lognormal.max_abs_rel_error < exponential.max_abs_rel_error,
            "lognormal {} {:?} vs exponential {} {:?}",
            lognormal.max_abs_rel_error,
            lognormal.params,
            exponential.max_abs_rel_error,
            exponential.params
        );
        assert_eq!(exponential.residuals.len(), RESIDUAL_QUANTILES.len());
    }

    #[test]
    fn size_law_interpolates_and_extrapolates() {
        let law = SizeLaw::fit(
            vec![
                SizePoint {
                    bytes: 4096,
                    p50_ns: 100_000,
                },
                SizePoint {
                    bytes: 1_048_576,
                    p50_ns: 1_000_000,
                },
                SizePoint {
                    bytes: 8_388_608,
                    p50_ns: 8_000_000,
                },
            ],
            "test",
        )
        .unwrap();
        let mid = law.latency_ns(524_288).unwrap();
        assert!(mid > 100_000.0 && mid < 1_000_000.0);
        let huge = law.latency_ns(16_777_216).unwrap();
        assert!(huge > 8_000_000.0);
        assert!(law.latency_ns(0).is_err());
    }

    #[test]
    fn calibration_reads_metrics_and_writes_receipt() {
        let dir = selvedge_core::testutil::TempDir::new("sim-cal").unwrap();
        let metrics = dir.join("m.jsonl");
        let mut text = String::new();
        for index in 0..40_u8 {
            text.push_str(&metrics_line(
                index,
                900_000 + u64::from(index) * 1000,
                300_000,
                false,
            ));
            text.push('\n');
        }
        text.push_str(&metrics_line(1, 0, 0, true));
        text.push('\n');
        std::fs::write(&metrics, text).unwrap();
        let io = dir.join("io.json");
        std::fs::write(
            &io,
            r#"{"random":[{"block_bytes":4096,"p50_ns":131356},
                          {"block_bytes":1048576,"p50_ns":1201676}]}"#,
        )
        .unwrap();
        let calibration = calibrate(
            &[CalibrationInput {
                role: "transfer+verify".to_owned(),
                path: metrics.clone(),
            }],
            &io,
            "rev",
            "20261116T000000Z",
        )
        .unwrap();
        assert_eq!(calibration.shard_bytes, 768);
        assert_eq!(calibration.transfer.n, 40);
        assert_eq!(calibration.hit_copy.n, 1);
        assert!(calibration.empirical_is_model_of_record);
        assert_eq!(calibration.inputs.len(), 2);
        let out = dir.join("cal.json");
        calibration.write(&out).unwrap();
        let loaded = Calibration::load(&out).unwrap();
        assert_eq!(loaded.schema, calibration.schema);
        assert_eq!(loaded.inputs, calibration.inputs);
        assert_eq!(loaded.shard_bytes, calibration.shard_bytes);
        assert_eq!(loaded.transfer, calibration.transfer);
        assert_eq!(loaded.verify, calibration.verify);
        assert_eq!(loaded.queue_wait, calibration.queue_wait);
        assert_eq!(loaded.insert, calibration.insert);
        assert_eq!(loaded.eviction, calibration.eviction);
        assert_eq!(loaded.hit_copy, calibration.hit_copy);
        assert_eq!(loaded.size_law, calibration.size_law);
        assert_eq!(loaded.fits.len(), calibration.fits.len());
        assert_eq!(loaded.notes, calibration.notes);
        let mut rng = SplitMix64::new(1);
        let sample = loaded.sample_transfer_ns(&mut rng, 768).unwrap();
        assert!(sample > 0.0);
    }

    #[test]
    fn calibration_fails_closed_without_cold_reads() {
        let dir = selvedge_core::testutil::TempDir::new("sim-cal").unwrap();
        let metrics = dir.join("m.jsonl");
        std::fs::write(&metrics, metrics_line(1, 0, 0, true)).unwrap();
        let io = dir.join("io.json");
        std::fs::write(
            &io,
            r#"{"random":[{"block_bytes":4096,"p50_ns":1},
                          {"block_bytes":8192,"p50_ns":2}]}"#,
        )
        .unwrap();
        let error = calibrate(
            &[CalibrationInput {
                role: "x".to_owned(),
                path: metrics,
            }],
            &io,
            "rev",
            "stamp",
        )
        .unwrap_err();
        assert!(error.to_string().contains("no cold reads"));
    }
}
