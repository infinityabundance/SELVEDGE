//! Versioned execution traces: the only input the simulator accepts.
//!
//! A trace is a JSONL file of [`TraceRecord`]s plus a sidecar
//! [`TraceManifest`] that carries the shard table, the record count, the
//! SHA-256 of the exact record bytes and the provenance that says whether the
//! trace is measured, reconstructed or synthetic.
//!
//! # The invariant that makes the simulator honest
//!
//! A trace describes **what was selected** in one step: the shard ids, the
//! selector that produced them and its mode. It never describes what the model
//! *should* have selected. There is no field for an expected set, a target, a
//! gold answer or a future token, and every struct uses
//! `deny_unknown_fields`, so a trace that tries to carry one fails to
//! deserialize. The simulator can therefore change latency, residency and
//! bytes, and can never change semantics.
//!
//! # Exact versus approximate
//!
//! `mode: exact` is a claim about the *selection*: the recorded set is the
//! model's native computation set for that step. The trace format cannot prove
//! it, so the manifest is required to name the basis. The only basis this build
//! accepts is [`ExactBasisKind::WholeBundlePlanRead`]: every record selects the
//! complete validated shard table, which is what the exact-mode storage pass
//! executes. A selector that predicts a subset cannot claim this basis, so the
//! M1/M2 selector traces are exported as `approximate`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use selvedge_core::{SelectionMode, Sha256Builder, Sha256Digest, ShardId, native_set_digest};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};

/// Schema of one record line.
pub const TRACE_SCHEMA: &str = "selvedge/execution-trace/v1";
/// Schema of the trace sidecar manifest.
pub const TRACE_MANIFEST_SCHEMA: &str = "selvedge/execution-trace-manifest/v1";

/// Default cap on records read from one trace.
pub const DEFAULT_MAX_RECORDS: u64 = 1_000_000;
/// Default cap on shards in one trace manifest.
pub const DEFAULT_MAX_SHARDS: u64 = 1_000_000;
/// Default cap on the byte length of one JSONL line.
pub const DEFAULT_MAX_LINE_BYTES: u64 = 4 * 1024 * 1024;

/// Where a trace came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceSource {
    /// Produced by a run of the real engine or a real measurement receipt.
    Measured,
    /// Reconstructed offline from a tracked receipt or fixture.
    Reconstructed,
    /// Produced by the in-crate generator for a controlled study.
    Synthetic,
}

impl TraceSource {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Reconstructed => "reconstructed",
            Self::Synthetic => "synthetic",
        }
    }
}

/// Provenance recorded next to every trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceProvenance {
    /// Human-readable origin, e.g. the command or fixture that produced it.
    pub origin: String,
    /// Tracked receipt the trace was derived from, when one exists.
    pub receipt: Option<String>,
    /// SHA-256 of that receipt, so the reference cannot drift.
    pub receipt_sha256: Option<Sha256Digest>,
    /// Git revision the trace was produced at.
    pub revision: String,
    /// UTC stamp, `YYYYMMDDTHHMMSSZ`.
    pub stamp: String,
    /// Free-form notes that qualify the trace.
    pub notes: Vec<String>,
}

impl TraceProvenance {
    /// Validate the provenance record.
    ///
    /// # Errors
    ///
    /// Rejects an empty origin, revision or stamp, and a receipt reference
    /// without a digest.
    pub fn validate(&self) -> Result<()> {
        if self.origin.is_empty() {
            return Err(SimError::invalid("origin", "is empty"));
        }
        if self.revision.is_empty() {
            return Err(SimError::invalid("revision", "is empty"));
        }
        if self.stamp.is_empty() {
            return Err(SimError::invalid("stamp", "is empty"));
        }
        if self.receipt.is_some() != self.receipt_sha256.is_some() {
            return Err(SimError::invalid(
                "receipt",
                "a receipt reference must name its sha256 and vice versa",
            ));
        }
        Ok(())
    }
}

/// The bundle a trace's shard ids belong to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceBundle {
    /// Bundle identifier.
    pub bundle_id: String,
    /// Manifest digest of the validated bundle.
    pub manifest_sha256: Sha256Digest,
}

/// The basis on which a trace claims `mode: exact`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExactBasisKind {
    /// Every record selects the complete validated shard table, which is what
    /// the exact-mode storage pass executes.
    WholeBundlePlanRead,
}

/// Declared evidence for a trace's exact-mode claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactBasis {
    /// Accepted basis.
    pub kind: ExactBasisKind,
    /// Digest of the full shard set, computed with
    /// [`selvedge_core::native_set_digest`].
    pub native_set_digest: Sha256Digest,
    /// Number of shards in that set.
    pub shard_count: u64,
}

/// One shard in the trace's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceShard {
    /// Content-derived shard id.
    pub id: ShardId,
    /// Extent length in bytes.
    pub bytes: u64,
    /// Payload-relative offset in the source bundle, used for physical order.
    pub offset: u64,
}

/// Real timestamps, when the trace was exported from a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceTimestamps {
    /// Monotonic nanoseconds when the selection decision was taken.
    pub decision_mono_ns: u64,
    /// Monotonic nanoseconds when the step's data was ready, when recorded.
    pub complete_mono_ns: Option<u64>,
}

/// Measured observations for one step, when the trace is measured.
///
/// These are *observations of the real run*, not simulator inputs. The
/// simulator may use them for validation; it never treats them as a prediction
/// and it never lets them change the selected set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedStep {
    /// Consumer stall measured for this step.
    pub stall_ns: u64,
    /// Bytes actually read from the backing tier.
    pub bytes_transferred: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Shards served from host RAM.
    pub hits: u64,
    /// Shards fetched from the backing tier.
    pub misses: u64,
}

/// One step of an execution trace: one token's selection for one layer, or one
/// whole-bundle pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceRecord {
    /// Record schema tag; must equal [`TRACE_SCHEMA`].
    pub schema: String,
    /// Request identifier (a prompt or a batch).
    pub request_id: String,
    /// Phase identifier (a task phase inside the request).
    pub phase_id: String,
    /// Token index inside the request, zero-based.
    pub token_index: u64,
    /// Decoder layer this step belongs to.
    pub layer: u32,
    /// Selected shard ids, sorted ascending and unique.
    pub selected: Vec<ShardId>,
    /// Selector that produced the selection.
    pub selector_id: String,
    /// `exact` or `approximate`; must equal the manifest's mode.
    pub mode: SelectionMode,
    /// Optional task label used for per-task cold-byte accounting.
    pub task: Option<String>,
    /// Real timestamps, when the trace is measured.
    pub timestamps: Option<TraceTimestamps>,
    /// Real observations, when the trace is measured.
    pub observed: Option<ObservedStep>,
}

impl TraceRecord {
    /// Validate the record against a shard table and a mode.
    ///
    /// # Errors
    ///
    /// Rejects a wrong schema tag, empty identifiers, an empty selection, an
    /// unsorted or duplicated selection, a shard outside the table and a mode
    /// that disagrees with the manifest.
    pub fn validate(
        &self,
        table: &BTreeMap<ShardId, TraceShard>,
        mode: SelectionMode,
    ) -> Result<()> {
        if self.schema != TRACE_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                format!("record schema {} is not {TRACE_SCHEMA}", self.schema),
            ));
        }
        for (name, value) in [
            ("request_id", &self.request_id),
            ("phase_id", &self.phase_id),
            ("selector_id", &self.selector_id),
        ] {
            if value.is_empty() {
                return Err(SimError::invalid("record", format!("{name} is empty")));
            }
            if value.len() > 256 {
                return Err(SimError::invalid(
                    "record",
                    format!("{name} is longer than 256 bytes"),
                ));
            }
        }
        if self.mode != mode {
            return Err(SimError::invalid(
                "mode",
                format!("record mode {} disagrees with trace mode {mode}", self.mode),
            ));
        }
        if self.selected.is_empty() {
            return Err(SimError::invalid("selected", "is empty"));
        }
        for pair in self.selected.windows(2) {
            if pair[0] >= pair[1] {
                return Err(SimError::invalid(
                    "selected",
                    "must be strictly ascending and unique",
                ));
            }
        }
        for shard in &self.selected {
            if !table.contains_key(shard) {
                return Err(SimError::invalid(
                    "selected",
                    format!("shard {} is not in the trace shard table", shard.short()),
                ));
            }
        }
        Ok(())
    }
}

/// The trace sidecar: everything needed to interpret and check the record file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceManifest {
    /// Sidecar schema tag; must equal [`TRACE_MANIFEST_SCHEMA`].
    pub schema: String,
    /// Stable trace identifier.
    pub trace_id: String,
    /// Measured, reconstructed or synthetic.
    pub source: TraceSource,
    /// Provenance for the file.
    pub provenance: TraceProvenance,
    /// Bundle identity its shard ids belong to.
    pub bundle: TraceBundle,
    /// Selection mode shared by every record.
    pub mode: SelectionMode,
    /// Present exactly when `mode` is `exact`.
    pub exact_basis: Option<ExactBasis>,
    /// Shard table, sorted by id.
    pub shards: Vec<TraceShard>,
    /// Number of record lines.
    pub records: u64,
    /// SHA-256 over `record_line_bytes || b"\n"` for every record.
    pub records_sha256: Sha256Digest,
    /// Generator parameters when `source` is `synthetic`.
    pub generator: Option<serde_json::Value>,
    /// Creation stamp, `YYYYMMDDTHHMMSSZ`.
    pub created_utc: String,
}

/// Input caps applied while reading a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceLimits {
    /// Maximum records.
    pub max_records: u64,
    /// Maximum shards in the table.
    pub max_shards: u64,
    /// Maximum bytes in one JSONL line.
    pub max_line_bytes: u64,
}

impl Default for TraceLimits {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_MAX_RECORDS,
            max_shards: DEFAULT_MAX_SHARDS,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
        }
    }
}

/// A loaded and verified trace.
#[derive(Clone, Debug)]
pub struct Trace {
    /// Sidecar manifest.
    pub manifest: TraceManifest,
    /// Records in file order.
    pub records: Vec<TraceRecord>,
}

impl Trace {
    /// Sidecar path for a record path: `<stem>.manifest.json`.
    #[must_use]
    pub fn manifest_path_for(records_path: &Path) -> PathBuf {
        let parent = records_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = records_path
            .file_stem()
            .map_or_else(|| "trace".to_owned(), |stem| stem.to_string_lossy().into());
        parent.join(format!("{stem}.manifest.json"))
    }

    /// Build a trace from records, sealing the manifest with their digest.
    ///
    /// # Errors
    ///
    /// Rejects a wrong sidecar schema, an empty record list, a mode/basis
    /// mismatch, duplicate shards, malformed shards and any record that fails
    /// validation. In exact mode it also checks the declared basis against the
    /// full shard set.
    pub fn new(mut manifest: TraceManifest, records: Vec<TraceRecord>) -> Result<Self> {
        manifest.records_sha256 = Self::records_sha256(&records)?;
        manifest.records = records.len() as u64;
        Self::from_parts(manifest, records)
    }

    /// Load a trace and its sidecar, verify the record digest and validate
    /// every record.
    ///
    /// # Errors
    ///
    /// Propagates I/O and JSON failures and rejects a digest mismatch, a
    /// record-count mismatch, an over-limit input and any invalid record.
    pub fn load(records_path: &Path, manifest_path: &Path, limits: TraceLimits) -> Result<Self> {
        let manifest_bytes = std::fs::read(manifest_path)?;
        let manifest: TraceManifest = serde_json::from_slice(&manifest_bytes)?;
        let mut builder = Sha256Builder::new();
        let mut records = Vec::new();
        let file = File::open(records_path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let length = line.len() as u64;
            if length > limits.max_line_bytes {
                return Err(SimError::limit(
                    "trace_line_bytes",
                    length,
                    limits.max_line_bytes,
                ));
            }
            if records.len() as u64 >= limits.max_records {
                return Err(SimError::limit(
                    "trace_records",
                    records.len() as u64 + 1,
                    limits.max_records,
                ));
            }
            builder.update(line.as_bytes());
            builder.update(b"\n");
            let record: TraceRecord = serde_json::from_str(&line)?;
            records.push(record);
        }
        let observed = builder.finish();
        if observed != manifest.records_sha256 {
            return Err(SimError::invalid(
                "records_sha256",
                format!(
                    "records hash {observed} does not match manifest {}",
                    manifest.records_sha256
                ),
            ));
        }
        if manifest.records != records.len() as u64 {
            return Err(SimError::invalid(
                "records",
                format!(
                    "manifest declares {} records, file has {}",
                    manifest.records,
                    records.len()
                ),
            ));
        }
        Self::from_parts(manifest, records)
    }

    /// Validate an already-parsed manifest and its records.
    ///
    /// # Errors
    ///
    /// Same rejections as [`Trace::load`] minus the file I/O.
    pub fn from_parts(manifest: TraceManifest, records: Vec<TraceRecord>) -> Result<Self> {
        if manifest.schema != TRACE_MANIFEST_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                format!(
                    "manifest schema {} is not {TRACE_MANIFEST_SCHEMA}",
                    manifest.schema
                ),
            ));
        }
        if manifest.trace_id.is_empty() {
            return Err(SimError::invalid("trace_id", "is empty"));
        }
        if manifest.created_utc.is_empty() {
            return Err(SimError::invalid("created_utc", "is empty"));
        }
        manifest.provenance.validate()?;
        if manifest.shards.is_empty() {
            return Err(SimError::invalid("shards", "table is empty"));
        }
        if manifest.shards.len() as u64 > DEFAULT_MAX_SHARDS {
            return Err(SimError::limit(
                "trace_shards",
                manifest.shards.len() as u64,
                DEFAULT_MAX_SHARDS,
            ));
        }
        if records.is_empty() {
            return Err(SimError::invalid("records", "trace has no records"));
        }
        let table = Self::table_from(&manifest)?;
        Self::check_offsets(&manifest)?;
        let expected = Self::records_sha256(&records)?;
        if expected != manifest.records_sha256 {
            return Err(SimError::invalid(
                "records_sha256",
                "records do not match the manifest digest",
            ));
        }
        if manifest.records != records.len() as u64 {
            return Err(SimError::invalid(
                "records",
                "record count does not match the manifest",
            ));
        }
        // Exact-mode basis.
        match (manifest.mode, &manifest.exact_basis) {
            (SelectionMode::Exact, None) => {
                return Err(SimError::invalid(
                    "exact_basis",
                    "an exact-mode trace must declare its basis",
                ));
            }
            (SelectionMode::Approximate, Some(_)) => {
                return Err(SimError::invalid(
                    "exact_basis",
                    "an approximate-mode trace cannot declare an exact basis",
                ));
            }
            (SelectionMode::Approximate, None) => {}
            (SelectionMode::Exact, Some(basis)) => {
                let ids: Vec<ShardId> = manifest.shards.iter().map(|shard| shard.id).collect();
                let digest = native_set_digest(&manifest.bundle.bundle_id, &ids)?;
                if digest != basis.native_set_digest {
                    return Err(SimError::invalid(
                        "exact_basis",
                        "declared native-set digest does not match the shard table",
                    ));
                }
                if basis.shard_count != ids.len() as u64 {
                    return Err(SimError::invalid(
                        "exact_basis",
                        "declared shard count does not match the shard table",
                    ));
                }
                match basis.kind {
                    ExactBasisKind::WholeBundlePlanRead => {
                        for (index, record) in records.iter().enumerate() {
                            if record.selected != ids {
                                return Err(SimError::invalid(
                                    "exact_basis",
                                    format!(
                                        "record {index} does not select the whole bundle, so \
                                         the whole-bundle basis does not hold"
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }
        for (index, record) in records.iter().enumerate() {
            record
                .validate(&table, manifest.mode)
                .map_err(|error| match error {
                    SimError::Invalid { field, reason } => {
                        SimError::invalid(field, format!("record {index}: {reason}"))
                    }
                    other => other,
                })?;
        }
        Ok(Self { manifest, records })
    }

    fn table_from(manifest: &TraceManifest) -> Result<BTreeMap<ShardId, TraceShard>> {
        let mut table = BTreeMap::new();
        for shard in &manifest.shards {
            if shard.bytes == 0 {
                return Err(SimError::invalid("shards", "a shard has zero bytes"));
            }
            if table.insert(shard.id, *shard).is_some() {
                return Err(SimError::invalid(
                    "shards",
                    format!("duplicate shard {}", shard.id.short()),
                ));
            }
        }
        Ok(table)
    }

    fn check_offsets(manifest: &TraceManifest) -> Result<()> {
        let mut ordered: Vec<&TraceShard> = manifest.shards.iter().collect();
        ordered.sort_by_key(|shard| (shard.offset, shard.id));
        for window in ordered.windows(2) {
            let [left, right] = window else {
                continue;
            };
            if left.offset.saturating_add(left.bytes) > right.offset {
                return Err(SimError::invalid(
                    "shards",
                    format!(
                        "extents {} and {} overlap",
                        left.id.short(),
                        right.id.short()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Shard table keyed by id.
    ///
    /// # Errors
    ///
    /// Rejects a duplicate or zero-byte shard.
    pub fn table(&self) -> Result<BTreeMap<ShardId, TraceShard>> {
        Self::table_from(&self.manifest)
    }

    /// Sorted shard table.
    #[must_use]
    pub fn shards(&self) -> &[TraceShard] {
        &self.manifest.shards
    }

    /// SHA-256 over the serialized records (`line || "\n"`).
    ///
    /// # Errors
    ///
    /// Propagates serialization failure.
    pub fn records_sha256(records: &[TraceRecord]) -> Result<Sha256Digest> {
        let mut builder = Sha256Builder::new();
        for record in records {
            let line = serde_json::to_string(record)?;
            builder.update(line.as_bytes());
            builder.update(b"\n");
        }
        Ok(builder.finish())
    }

    /// Write the record file and its sidecar.
    ///
    /// # Errors
    ///
    /// Rejects a wrong schema and propagates I/O and serialization failures.
    pub fn write(&self, records_path: &Path, manifest_path: &Path) -> Result<()> {
        if self.manifest.schema != TRACE_MANIFEST_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                "manifest schema is not current",
            ));
        }
        if let Some(parent) = records_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = manifest_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(records_path)?;
        let mut writer = BufWriter::new(file);
        for record in &self.records {
            let line = serde_json::to_string(record)?;
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        let manifest_bytes = serde_json::to_vec_pretty(&self.manifest)?;
        std::fs::write(manifest_path, manifest_bytes)?;
        Ok(())
    }

    /// Total bytes of one record's selection.
    #[must_use]
    pub fn record_bytes(&self, record: &TraceRecord, table: &BTreeMap<ShardId, TraceShard>) -> u64 {
        record
            .selected
            .iter()
            .map(|id| table.get(id).map_or(0, |shard| shard.bytes))
            .fold(0_u64, u64::saturating_add)
    }

    /// Requests in first-seen order with their record indices.
    #[must_use]
    pub fn requests(&self) -> Vec<(String, Vec<usize>)> {
        let mut order: Vec<(String, Vec<usize>)> = Vec::new();
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for (position, record) in self.records.iter().enumerate() {
            match index.get(&record.request_id) {
                Some(slot) => order[*slot].1.push(position),
                None => {
                    index.insert(record.request_id.clone(), order.len());
                    order.push((record.request_id.clone(), vec![position]));
                }
            }
        }
        order
    }

    /// Phases in first-seen order with their record indices.
    #[must_use]
    pub fn phases(&self) -> Vec<(String, Vec<usize>)> {
        let mut order: Vec<(String, Vec<usize>)> = Vec::new();
        let mut index: BTreeMap<(String, String), usize> = BTreeMap::new();
        for (position, record) in self.records.iter().enumerate() {
            let key = (record.request_id.clone(), record.phase_id.clone());
            match index.get(&key) {
                Some(slot) => order[*slot].1.push(position),
                None => {
                    index.insert(key, order.len());
                    order.push((record.phase_id.clone(), vec![position]));
                }
            }
        }
        order
    }

    /// Distinct tasks in first-seen order.
    #[must_use]
    pub fn tasks(&self) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut order = Vec::new();
        for record in &self.records {
            if let Some(task) = &record.task
                && seen.insert(task.clone())
            {
                order.push(task.clone());
            }
        }
        order
    }

    /// Reject a trace whose mode is approximate but whose declared selector is
    /// the native dense path. This is a guard against mislabelled exports, not
    /// a semantic proof.
    ///
    /// # Errors
    ///
    /// Rejects a selector id containing `native-dense` in approximate mode.
    pub fn check_label(&self) -> Result<()> {
        if self.manifest.mode == SelectionMode::Approximate
            && self
                .records
                .iter()
                .any(|record| record.selector_id.contains("native-dense"))
        {
            return Err(SimError::invalid(
                "selector_id",
                "an approximate trace cannot name the native dense selector",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use selvedge_core::testutil::TempDir;

    fn shard(tag: u8, bytes: u64, offset: u64) -> TraceShard {
        TraceShard {
            id: ShardId::from_digest(Sha256Digest::of(&[tag])),
            bytes,
            offset,
        }
    }

    fn manifest(mode: SelectionMode, shards: Vec<TraceShard>) -> TraceManifest {
        let basis = if mode == SelectionMode::Exact {
            let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
            Some(ExactBasis {
                kind: ExactBasisKind::WholeBundlePlanRead,
                native_set_digest: native_set_digest("bundle", &ids).unwrap(),
                shard_count: ids.len() as u64,
            })
        } else {
            None
        };
        TraceManifest {
            schema: TRACE_MANIFEST_SCHEMA.to_owned(),
            trace_id: "t".to_owned(),
            source: TraceSource::Synthetic,
            provenance: TraceProvenance {
                origin: "unit-test".to_owned(),
                receipt: None,
                receipt_sha256: None,
                revision: "deadbeef".to_owned(),
                stamp: "20261116T000000Z".to_owned(),
                notes: Vec::new(),
            },
            bundle: TraceBundle {
                bundle_id: "bundle".to_owned(),
                manifest_sha256: Sha256Digest::of(b"m"),
            },
            mode,
            exact_basis: basis,
            shards,
            records: 0,
            records_sha256: Sha256Digest::of(b""),
            generator: None,
            created_utc: "20261116T000000Z".to_owned(),
        }
    }

    fn record(selected: Vec<ShardId>, mode: SelectionMode) -> TraceRecord {
        TraceRecord {
            schema: TRACE_SCHEMA.to_owned(),
            request_id: "r0".to_owned(),
            phase_id: "p0".to_owned(),
            token_index: 0,
            layer: 0,
            selected,
            selector_id: "unit".to_owned(),
            mode,
            task: Some("code".to_owned()),
            timestamps: None,
            observed: None,
        }
    }

    #[test]
    fn trace_round_trip_and_hash() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 4096)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let trace = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(vec![ids[0]], SelectionMode::Approximate)],
        )
        .unwrap();
        let dir = TempDir::new("sim-trace").unwrap();
        let records = dir.join("t.jsonl");
        let sidecar = dir.join("t.manifest.json");
        trace.write(&records, &sidecar).unwrap();
        let loaded = Trace::load(&records, &sidecar, TraceLimits::default()).unwrap();
        assert_eq!(loaded.manifest.records, 1);
        assert_eq!(loaded.records, trace.records);
        assert_eq!(
            loaded.manifest.records_sha256,
            trace.manifest.records_sha256
        );
    }

    #[test]
    fn trace_rejects_out_of_table_shard() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 4096)];
        let unknown = ShardId::from_digest(Sha256Digest::of(b"nope"));
        let error = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(vec![unknown], SelectionMode::Approximate)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("not in the trace shard table"));
    }

    #[test]
    fn trace_rejects_noncanonical_selection() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 4096)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let mut ordered = ids.clone();
        ordered.reverse();
        if ordered == ids {
            ordered = ids.clone();
        }
        let error = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(ordered, SelectionMode::Approximate)],
        );
        // Either sorted (the ids happened to be ascending) or rejected.
        if let Ok(trace) = error {
            assert!(
                trace.records[0]
                    .selected
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
        }
    }

    #[test]
    fn trace_rejects_duplicate_selection() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 4096)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let error = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(vec![ids[0], ids[0]], SelectionMode::Approximate)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("strictly ascending"));
    }

    #[test]
    fn trace_rejects_semantic_fields() {
        // A trace may not carry an expected set; unknown fields fail closed.
        let shards = vec![shard(1, 100, 0)];
        let trace = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(
                vec![ShardId::from_digest(Sha256Digest::of(&[1]))],
                SelectionMode::Approximate,
            )],
        )
        .unwrap();
        let dir = TempDir::new("sim-trace").unwrap();
        let records = dir.join("t.jsonl");
        let sidecar = dir.join("t.manifest.json");
        trace.write(&records, &sidecar).unwrap();
        let mut line = std::fs::read_to_string(&records).unwrap();
        line = line.replacen('{', "{\"expected\":[],", 1);
        std::fs::write(&records, line).unwrap();
        let error = Trace::load(&records, &sidecar, TraceLimits::default()).unwrap_err();
        assert!(error.to_string().contains("json"));
    }

    #[test]
    fn trace_manifest_hash_mismatch_fails_closed() {
        let shards = vec![shard(1, 100, 0)];
        let trace = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(
                vec![ShardId::from_digest(Sha256Digest::of(&[1]))],
                SelectionMode::Approximate,
            )],
        )
        .unwrap();
        let dir = TempDir::new("sim-trace").unwrap();
        let records = dir.join("t.jsonl");
        let sidecar = dir.join("t.manifest.json");
        trace.write(&records, &sidecar).unwrap();
        let mut text = std::fs::read_to_string(&records).unwrap();
        text = text.replace("\"layer\":0", "\"layer\":1");
        std::fs::write(&records, text).unwrap();
        let error = Trace::load(&records, &sidecar, TraceLimits::default()).unwrap_err();
        assert!(error.to_string().contains("does not match manifest"));
    }

    #[test]
    fn trace_exact_mode_requires_whole_bundle_set() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 4096)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let error = Trace::new(
            manifest(SelectionMode::Exact, shards),
            vec![record(vec![ids[0]], SelectionMode::Exact)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("whole bundle"));
    }

    #[test]
    fn trace_exact_mode_without_basis_is_rejected() {
        let shards = vec![shard(1, 100, 0)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let mut m = manifest(SelectionMode::Exact, shards);
        m.exact_basis = None;
        let error = Trace::new(m, vec![record(ids, SelectionMode::Exact)]).unwrap_err();
        assert!(error.to_string().contains("must declare its basis"));
    }

    #[test]
    fn trace_line_cap_is_enforced() {
        let shards = vec![shard(1, 100, 0)];
        let trace = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(
                vec![ShardId::from_digest(Sha256Digest::of(&[1]))],
                SelectionMode::Approximate,
            )],
        )
        .unwrap();
        let dir = TempDir::new("sim-trace").unwrap();
        let records = dir.join("t.jsonl");
        let sidecar = dir.join("t.manifest.json");
        trace.write(&records, &sidecar).unwrap();
        let limits = TraceLimits {
            max_line_bytes: 8,
            ..TraceLimits::default()
        };
        let error = Trace::load(&records, &sidecar, limits).unwrap_err();
        assert!(matches!(error, SimError::Limit { .. }));
    }

    #[test]
    fn overlapping_shards_are_rejected() {
        let shards = vec![shard(1, 100, 0), shard(2, 200, 50)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let error = Trace::new(
            manifest(SelectionMode::Approximate, shards),
            vec![record(vec![ids[0]], SelectionMode::Approximate)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("overlap"));
    }

    #[test]
    fn approximate_trace_cannot_name_native_dense() {
        let shards = vec![shard(1, 100, 0)];
        let ids: Vec<ShardId> = shards.iter().map(|shard| shard.id).collect();
        let mut rec = record(vec![ids[0]], SelectionMode::Approximate);
        rec.selector_id = "native-dense".to_owned();
        let trace = Trace::new(manifest(SelectionMode::Approximate, shards), vec![rec]).unwrap();
        assert!(trace.check_label().is_err());
    }
}
