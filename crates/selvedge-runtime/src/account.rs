//! Per-token accounting records and the run summary.
//!
//! Every executed token produces one [`AccountingRow`], written as JSON Lines so a
//! long run is streamable and a reader can audit any single token without
//! re-reading the whole file. The row embeds
//! [`selvedge_core::TokenAccounting`] verbatim and adds only the fields that
//! describe *which* computation produced it.
//!
//! # What a row contains
//!
//! - `accounting.active_unique_params`: union of learned elements the token used
//!   (tied embedding counted once, MLP fragments counted per shard);
//! - `accounting.parsed_parameter_reads`: element reads including repeats (the
//!   tied output projection is a full read of the embedding every token);
//! - `accounting.learned_bytes_by_tier`: bytes by tier, from the ledger;
//! - `accounting.flops` and `accounting.controller_flops`: model FLOPs from the
//!   documented `2*M*N*K` formula plus the separately reported selector cost;
//! - `budget`: the verdict the gate returned *before* the token executed.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use selvedge_core::{BudgetVerdict, TokenAccounting};
use serde::Serialize;

use crate::error::{Result, RuntimeError};

/// Schema string for one accounting row.
pub const ACCOUNTING_SCHEMA: &str = "selvedge/runtime-token-accounting/v1";
/// Schema string for the run summary.
pub const RUN_SCHEMA: &str = "selvedge/runtime-run/v1";

/// Which part of a run produced a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Prompt processing.
    Prefill,
    /// Generated token.
    Decode,
}

impl Phase {
    /// Stable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// Decoded-residency counters for one token.
///
/// Separate from `read_calls`/`cache_hits`, which are the **storage** cache's
/// counters. `decodes` counts conversion passes this token needed; `hits` counts
/// shards served straight from execution residency with neither a store read nor
/// a decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct DecodeRow {
    /// Conversion passes performed for this token.
    pub decodes: u64,
    /// Shards served from the decoded residency.
    pub hits: u64,
    /// Shards looked up in the residency.
    pub lookups: u64,
    /// Residency byte budget.
    pub capacity_bytes: u64,
    /// Bytes resident at the end of the token.
    pub resident_bytes: u64,
    /// Largest residency observed over the run so far.
    pub peak_bytes: u64,
    /// Entries evicted over the run so far.
    pub evictions: u64,
}

impl DecodeRow {
    /// Fraction of this token's lookups served from residency.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }
}

/// Asynchronous-prefetch accounting for one token.
///
/// `exposed_stall_ns` is time compute spent blocked on a shard it needed;
/// `hidden_ns` is fetch time that elapsed behind compute. `fetch_busy_ns +
/// residual_wait_ns == hidden_ns + exposed_stall_ns` holds for every token (see
/// verify: [`crate::prefetch::StallAccounting::conserves`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PrefetchRow {
    /// Layers that ran an asynchronous fetcher.
    pub layers: u64,
    /// Shards fetched.
    pub shards_fetched: u64,
    /// Bytes delivered by the fetcher.
    pub fetched_bytes: u64,
    /// Bytes of those that came from the backing tier.
    pub device_bytes: u64,
    /// Fetched bytes compute consumed.
    pub used_bytes: u64,
    /// Fetched bytes compute never consumed.
    pub wasted_bytes: u64,
    /// Largest staging buffer used.
    pub staging_high_water_bytes: u64,
    /// Nanoseconds the fetch timeline was busy.
    pub fetch_busy_ns: u64,
    /// Nanoseconds compute spent waiting for a named shard.
    pub exposed_stall_ns: u64,
    /// Fetch time that elapsed behind compute.
    pub hidden_ns: u64,
    /// Waited time during which the fetcher was not busy.
    pub residual_wait_ns: u64,
    /// Waits that blocked.
    pub waits: u64,
    /// Dependencies already satisfied when asked for.
    pub ready_without_wait: u64,
    /// Median per-wait stall in nanoseconds.
    pub stall_p50_ns: u64,
    /// 95th percentile per-wait stall in nanoseconds.
    pub stall_p95_ns: u64,
    /// Slowest single wait.
    pub stall_max_ns: u64,
}

/// Batch metadata for a token executed inside a batched prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BatchRow {
    /// Positions in this batch.
    pub size: u64,
    /// This token's index inside the batch.
    pub index: u64,
    /// Absolute position of the batch's first token.
    pub first_position: u64,
    /// Whether this row carries the batch's physical counters (`read_calls`,
    /// `cache_hits`, `bytes_device`) and its wall time. Exactly one row of a
    /// batch does, so totals over rows stay equal to the physical totals and no
    /// elapsed time is counted twice.
    pub physical_counters: bool,
}

/// One per-token accounting row.
#[derive(Clone, Debug, Serialize)]
pub struct AccountingRow {
    /// Schema string.
    pub schema: &'static str,
    /// Prompt or generated token.
    pub phase: Phase,
    /// Absolute position in the sequence.
    pub position: u64,
    /// Token id that was embedded.
    pub token_id: u32,
    /// Wall time for this token's forward pass.
    pub elapsed_ns: u64,
    /// Execution mode label.
    pub mode: &'static str,
    /// Bundle executed.
    pub bundle_id: String,
    /// Layers executed.
    pub layers_executed: u64,
    /// Group shards executed (three per group).
    pub group_shards_executed: u64,
    /// Store read calls issued for this token.
    pub read_calls: u64,
    /// Reads served from the cache.
    pub cache_hits: u64,
    /// Bytes the backing tier delivered for this token.
    pub bytes_device: u64,
    /// Groups the selector asked for and the budget gate removed.
    pub groups_trimmed: u64,
    /// Decoded-residency counters for this token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeRow>,
    /// Asynchronous-prefetch counters for this token, when one ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefetch: Option<PrefetchRow>,
    /// Batch metadata, when this token ran inside a batched prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchRow>,
    /// The gate's verdict, produced before execution.
    pub budget: BudgetVerdict,
    /// The mandate's accounting record.
    pub accounting: TokenAccounting,
}

/// JSONL writer for [`AccountingRow`].
#[derive(Debug)]
pub struct AccountingWriter {
    sink: Option<BufWriter<File>>,
    rows: u64,
}

impl AccountingWriter {
    /// A writer that discards rows.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            sink: None,
            rows: 0,
        }
    }

    /// Create (truncate) a JSONL file.
    ///
    /// # Errors
    ///
    /// Rejects an unwritable path.
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        Ok(Self {
            sink: Some(BufWriter::new(file)),
            rows: 0,
        })
    }

    /// Whether rows are being kept.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.sink.is_some()
    }

    /// Rows written.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Append one row.
    ///
    /// # Errors
    ///
    /// Propagates serialization and I/O failures.
    pub fn write(&mut self, row: &AccountingRow) -> Result<()> {
        let Some(sink) = self.sink.as_mut() else {
            return Ok(());
        };
        let line = serde_json::to_string(row)?;
        sink.write_all(line.as_bytes())?;
        sink.write_all(b"\n")?;
        self.rows = self.rows.saturating_add(1);
        Ok(())
    }

    /// Flush buffered rows.
    ///
    /// # Errors
    ///
    /// Propagates I/O failures.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(sink) = self.sink.as_mut() {
            sink.flush()?;
        }
        Ok(())
    }
}

/// Latency percentiles in nanoseconds.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Latency {
    /// Samples.
    pub count: u64,
    /// Median.
    pub p50_ns: u64,
    /// 95th percentile.
    pub p95_ns: u64,
    /// 99th percentile.
    pub p99_ns: u64,
    /// Slowest.
    pub max_ns: u64,
    /// Mean, rounded down.
    pub mean_ns: u64,
}

impl Latency {
    /// Compute from unsorted samples.
    #[must_use]
    pub fn from_samples(samples: &mut [u64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let count = samples.len() as u64;
        let total: u64 = samples
            .iter()
            .fold(0_u64, |acc, value| acc.saturating_add(*value));
        samples.sort_unstable();
        // Nearest-rank: the smallest value at or above the `fraction` quantile.
        let at = |fraction: f64| -> u64 {
            let rank = ((samples.len() as f64) * fraction).ceil() as usize;
            samples[rank.saturating_sub(1).min(samples.len() - 1)]
        };
        Self {
            count,
            p50_ns: at(0.50),
            p95_ns: at(0.95),
            p99_ns: at(0.99),
            max_ns: samples[samples.len() - 1],
            mean_ns: total / count,
        }
    }
}

/// Streaming configuration as recorded in a receipt.
#[derive(Clone, Debug, Serialize)]
pub struct StreamSummary {
    /// Cache policy.
    pub policy: String,
    /// Cache capacity in bytes.
    pub cache_bytes: u64,
    /// Whether the synchronous prefetch was on.
    pub prefetch: bool,
    /// Whether the asynchronous prefetch was on.
    pub async_prefetch: bool,
    /// Bounded depth of the asynchronous fetcher.
    pub prefetch_depth: u64,
    /// Fetch order.
    pub read_order: String,
    /// Injected per-read delay.
    pub injected_read_delay_ns: u64,
    /// Verification mode.
    pub verification: String,
    /// Decoded-residency byte budget (0 means disabled).
    pub decode_cache_bytes: u64,
    /// How the prompt was executed (`scalar` or `batched`).
    pub prefill: String,
    /// Largest batch the batched prefill forms.
    pub prefill_batch: u64,
}

/// Totals over the whole run.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AccountingTotals {
    /// Tokens accounted.
    pub tokens: u64,
    /// Sum of per-token active unique sets (not a union across tokens).
    pub active_unique_sum: u64,
    /// Largest per-token active unique set.
    pub active_unique_max: u64,
    /// Sum of element reads.
    pub reads_sum: u64,
    /// Sum of model FLOPs.
    pub flops_sum: u64,
    /// Sum of controller FLOPs.
    pub controller_flops_sum: u64,
    /// Bytes read per tier over the run, summed from the store's own counters.
    pub bytes_device: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Read calls.
    pub read_calls: u64,
    /// Decode passes performed over the run.
    pub decodes: u64,
    /// Shards served from the decoded residency over the run.
    pub decode_hits: u64,
    /// Exposed async-prefetch stall over the run.
    pub exposed_stall_ns: u64,
    /// Fetch time hidden behind compute over the run.
    pub hidden_ns: u64,
    /// Prefetch bytes fetched but not consumed over the run.
    pub wasted_prefetch_bytes: u64,
}

impl AccountingTotals {
    /// Fold one token in.
    pub fn observe(&mut self, row: &AccountingRow) {
        self.tokens = self.tokens.saturating_add(1);
        let active = row.accounting.active_unique_params;
        self.active_unique_sum = self.active_unique_sum.saturating_add(active);
        self.active_unique_max = self.active_unique_max.max(active);
        self.reads_sum = self
            .reads_sum
            .saturating_add(row.accounting.parsed_parameter_reads);
        self.flops_sum = self.flops_sum.saturating_add(row.accounting.flops);
        self.controller_flops_sum = self
            .controller_flops_sum
            .saturating_add(row.accounting.controller_flops);
        self.bytes_device = self.bytes_device.saturating_add(row.bytes_device);
        self.cache_hits = self.cache_hits.saturating_add(row.cache_hits);
        self.read_calls = self.read_calls.saturating_add(row.read_calls);
        if let Some(decode) = &row.decode {
            self.decodes = self.decodes.saturating_add(decode.decodes);
            self.decode_hits = self.decode_hits.saturating_add(decode.hits);
        }
        if let Some(prefetch) = &row.prefetch {
            self.exposed_stall_ns = self
                .exposed_stall_ns
                .saturating_add(prefetch.exposed_stall_ns);
            self.hidden_ns = self.hidden_ns.saturating_add(prefetch.hidden_ns);
            self.wasted_prefetch_bytes = self
                .wasted_prefetch_bytes
                .saturating_add(prefetch.wasted_bytes);
        }
    }
}

/// The `--json` document a run prints.
#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    /// Schema string.
    pub schema: &'static str,
    /// Whether every gate, witness and comparison passed.
    pub ok: bool,
    /// Repository revision, when the caller supplied one.
    pub revision: Option<String>,
    /// Receipt stamp, when the caller supplied one.
    ///
    /// `scripts/runtime-courts.sh` exports `SELVEDGE_STAMP` so that the run's own
    /// JSON can be the receipt `docs/perf-numbers.json` reads.
    pub stamp: Option<String>,
    /// Execution mode.
    pub mode: String,
    /// Selector name.
    pub selector: String,
    /// Bundle id.
    pub bundle_id: String,
    /// Manifest digest.
    pub manifest_sha256: String,
    /// Model directory.
    pub model_dir: String,
    /// Prompt kind (`text`, `ids`, `chat`).
    pub prompt_kind: String,
    /// Prompt token ids.
    pub prompt_tokens: Vec<u32>,
    /// Generated token ids.
    pub generated_tokens: Vec<u32>,
    /// Generated text, decoded by the tokenizer when one is configured.
    pub generated_text: String,
    /// SHA-256 of each token's f32 logits, in order.
    pub logits_digests: Vec<String>,
    /// SHA-256 over the concatenated logits digests.
    pub sequence_digest: String,
    /// Layers.
    pub layers: usize,
    /// Groups per layer.
    pub groups_per_layer: usize,
    /// Shards in the bundle.
    pub shards: usize,
    /// Payload bytes of the bundle.
    pub payload_bytes: u64,
    /// Backbone elements held in host RAM.
    pub backbone_elements: u64,
    /// Total accessible learned elements (backbone plus every bundle shard).
    pub total_accessible_elements: u64,
    /// Prefill latency.
    pub prefill: Latency,
    /// Prompt tokens per second over the prefill phase.
    pub prefill_tokens_per_second: f64,
    /// Decode latency.
    pub decode: Latency,
    /// Generated tokens per second over the decode phase.
    pub decode_tokens_per_second: f64,
    /// Totals over every executed token.
    pub totals: AccountingTotals,
    /// Streaming configuration.
    pub stream: StreamSummary,
    /// Budget limit.
    pub budget_limit: u64,
    /// Compliance mode label.
    pub compliance: String,
    /// Whether the within-budget fallback removed groups.
    pub fallback_applied: bool,
    /// Semantic exactness classification.
    pub exactness: serde_json::Value,
    /// The two verdicts, for approximate runs.
    pub verdict: Option<serde_json::Value>,
    /// Reference-logits comparison, when asked for.
    pub comparison: Option<crate::run::LogitsComparison>,
    /// Accounting file written, when asked for.
    pub accounting_path: Option<String>,
    /// Environment facts that make the numbers interpretable.
    pub environment: serde_json::Value,
}

/// Refuse a malformed total rather than print a wrong one.
///
/// # Errors
///
/// Rejects a zero token count.
pub fn require_tokens(total: &AccountingTotals) -> Result<()> {
    if total.tokens == 0 {
        return Err(RuntimeError::model("no token was accounted"));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn latency_percentiles_are_index_based() {
        let mut samples: Vec<u64> = (1..=100).collect();
        let latency = Latency::from_samples(&mut samples);
        assert_eq!(latency.count, 100);
        assert_eq!(latency.p50_ns, 50);
        assert_eq!(latency.p95_ns, 95);
        assert_eq!(latency.max_ns, 100);
    }

    #[test]
    fn disabled_writer_counts_nothing() {
        let mut writer = AccountingWriter::disabled();
        assert!(!writer.is_enabled());
        assert_eq!(writer.rows(), 0);
        writer.flush().unwrap();
    }
}
