//! The discrete-event execution and deadline simulator.
//!
//! # Model
//!
//! Time is nanoseconds. The consumer walks the trace's records in order. For
//! each record it needs the selected shards at `need_time`, which is the time
//! the previous record's compute finished. A shard is either resident (a hit),
//! already in flight as a prefetch (the consumer waits for its completion), or
//! missing (a required transfer is issued now).
//!
//! Missing shards are grouped into coalesced runs over their physical order. A
//! run occupies one device server from `start` to
//! `start + transfer(run_bytes) + verify(run_bytes)`, where both components are
//! sampled from the calibrated empirical distribution. `queue_depth` servers run
//! concurrently; a byte budget bounds in-flight work. The consumer's exposed
//! stall for a record is the largest completion time among the record's runs
//! and awaited prefetches minus `need_time`, so a record whose data is resident
//! contributes zero.
//!
//! Phase warm-up is charged separately: the bytes fetched to establish a
//! working set are neither "needed by the record" nor "wasted", so they are
//! reported as `bytes_phase_cold` and appear in their own conservation term.
//!
//! # Determinism
//!
//! Every ordering is explicit:
//!
//! - records are processed in file order;
//! - shards inside a record are processed in physical placement order;
//! - server choice is the earliest free time, ties broken by the lowest server
//!   index;
//! - cache victims are chosen by the policy's own key, ties broken by shard id;
//! - prefetch candidates are sorted by shard id before admission;
//! - a preemption cancels the in-flight prefetch with the latest finish, ties
//!   broken by the lowest transfer id.
//!
//! Randomness comes only from one `SplitMix64` seeded by [`SimConfig::seed`].
//!
//! # What a prefetch can never do
//!
//! A predictor returns candidates. The engine fetches them early, or discards
//! them, and either way the record's required set comes from the trace. A wrong
//! prediction is a measured stall or wasted byte, never a different selection.

use std::collections::{BTreeMap, BTreeSet};

use selvedge_core::ShardId;
use selvedge_store::PolicyKind;

use crate::calibration::Calibration;
use crate::envelope::{Envelope, EnvelopeSet, Invalidation};
use crate::error::{Result, SimError};
use crate::placement::Placement;
use crate::policy::{PredictContext, Predictor, SimCache};
use crate::prng::SplitMix64;
use crate::report::{
    ConfigSummary, LatencySummary, PhaseSummary, RecordOutcome, SimSummary, TaskSummary,
};
use crate::trace::{Trace, TraceRecord, TraceShard};

/// What to do when the in-flight byte budget would be exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackpressureMode {
    /// Block the consumer until room exists and count the wait.
    Wait,
    /// Fail the run closed; no report is produced.
    Fail,
}

/// Where the phase working set comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkingSetSource {
    /// Shards used most often in completed records (strictly past-only).
    History,
    /// The active envelope's candidate list, built from a disjoint prefix.
    Envelope,
}

impl WorkingSetSource {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::History => "history",
            Self::Envelope => "envelope",
        }
    }
}

/// Two-timescale scheduling: a phase working set plus token-level selection
/// inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoTimescaleConfig {
    /// Where the working set comes from.
    pub source: WorkingSetSource,
    /// Fraction of the cache budget the working set may use.
    pub capacity_fraction: f64,
    /// Minimum past uses before a shard can join a history working set.
    pub min_usage: u64,
}

impl TwoTimescaleConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Rejects an out-of-range capacity fraction.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.capacity_fraction) || self.capacity_fraction <= 0.0 {
            return Err(SimError::invalid("capacity_fraction", "must be in (0, 1]"));
        }
        Ok(())
    }
}

/// Simulator configuration.
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Cache policy.
    pub policy: PolicyKind,
    /// Cache capacity in bytes.
    pub cache_bytes: u64,
    /// Concurrent device servers.
    pub queue_depth: u32,
    /// In-flight byte budget.
    pub max_in_flight_bytes: u64,
    /// Gap bridged when coalescing adjacent extents.
    pub coalesce_gap_bytes: u64,
    /// Largest coalesced run.
    pub max_run_bytes: u64,
    /// Charge SHA-256 verification on every transferred byte.
    pub verify: bool,
    /// Compute time per record, available for prefetch overlap.
    pub compute_ns_per_record: u64,
    /// Per-record lifetime budget; a stall above it is a deadline miss.
    pub deadline_ns: Option<u64>,
    /// Random seed.
    pub seed: u64,
    /// Backpressure behaviour.
    pub backpressure: BackpressureMode,
    /// Physical layout, when placement is being studied.
    pub placement: Option<Placement>,
    /// Two-timescale working set.
    pub two_timescale: Option<TwoTimescaleConfig>,
    /// Cancel prefetches whose shards are outside the new phase's envelope.
    pub cancel_prefetch_on_phase_change: bool,
    /// Preempt a speculative transfer when a required one needs its server.
    pub preempt_prefetch_for_required: bool,
    /// First record index to evaluate; earlier records are skipped untouched.
    pub start_record: u64,
    /// Cap on prefetch candidates per record.
    pub max_prefetch_shards: usize,
    /// Fault injection: a shard whose transfer must fail verification.
    pub inject_checksum_failure: Option<ShardId>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            policy: PolicyKind::None,
            cache_bytes: 0,
            queue_depth: 1,
            max_in_flight_bytes: 512 * 1024 * 1024,
            coalesce_gap_bytes: 0,
            max_run_bytes: 32 * 1024 * 1024,
            verify: true,
            compute_ns_per_record: 0,
            deadline_ns: None,
            seed: 0x5E1D_E6E5_11ED_0001,
            backpressure: BackpressureMode::Wait,
            placement: None,
            two_timescale: None,
            cancel_prefetch_on_phase_change: true,
            preempt_prefetch_for_required: true,
            start_record: 0,
            max_prefetch_shards: 4096,
            inject_checksum_failure: None,
        }
    }
}

impl SimConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Rejects an inconsistent cache configuration, a zero queue depth, a
    /// non-positive deadline, an invalid working-set fraction and a run cap
    /// that is too small to hold one trigger's bytes.
    pub fn validate(&self) -> Result<()> {
        if self.queue_depth == 0 {
            return Err(SimError::invalid("queue_depth", "must be non-zero"));
        }
        if self.policy == PolicyKind::None {
            if self.cache_bytes != 0 {
                return Err(SimError::invalid(
                    "cache_bytes",
                    "must be zero when the policy is none",
                ));
            }
        } else if self.cache_bytes == 0 {
            return Err(SimError::invalid(
                "cache_bytes",
                "a policy-backed cache needs a non-zero capacity",
            ));
        }
        if self.max_in_flight_bytes == 0 {
            return Err(SimError::invalid("max_in_flight_bytes", "must be non-zero"));
        }
        if self.max_run_bytes == 0 {
            return Err(SimError::invalid("max_run_bytes", "must be non-zero"));
        }
        if self.deadline_ns == Some(0) {
            return Err(SimError::invalid("deadline_ns", "must be non-zero"));
        }
        if let Some(two) = &self.two_timescale {
            two.validate()?;
        }
        Ok(())
    }

    /// A receipt-shaped summary.
    #[must_use]
    pub fn summary(&self, predictor: &dyn Predictor, start_record: u64) -> ConfigSummary {
        ConfigSummary {
            policy: self.policy.name().to_owned(),
            cache_bytes: self.cache_bytes,
            queue_depth: self.queue_depth,
            max_in_flight_bytes: self.max_in_flight_bytes,
            coalesce_gap_bytes: self.coalesce_gap_bytes,
            max_run_bytes: self.max_run_bytes,
            verify: self.verify,
            compute_ns_per_record: self.compute_ns_per_record,
            deadline_ns: self.deadline_ns,
            seed: self.seed,
            backpressure: match self.backpressure {
                BackpressureMode::Wait => "wait".to_owned(),
                BackpressureMode::Fail => "fail".to_owned(),
            },
            predictor: predictor.name().to_owned(),
            predictor_uses_future: predictor.uses_future(),
            two_timescale: self.two_timescale.is_some(),
            working_set_source: self
                .two_timescale
                .map_or("none".to_owned(), |two| two.source.name().to_owned()),
            start_record,
        }
    }
}

/// A completed simulation.
#[derive(Clone, Debug)]
pub struct Simulation {
    /// Aggregate counters.
    pub summary: SimSummary,
    /// Per-phase rows.
    pub phases: Vec<PhaseSummary>,
    /// Per-task rows.
    pub tasks: Vec<TaskSummary>,
    /// Per-record outcomes.
    pub outcomes: Vec<RecordOutcome>,
    /// Envelope invalidations.
    pub invalidations: u64,
}

#[derive(Clone, Debug)]
struct Transfer {
    id: u64,
    shard: Option<ShardId>,
    bytes: u64,
    finish_ns: u64,
    server: usize,
    prefetch: bool,
    consumed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunPlan {
    members: Vec<ShardId>,
    total_bytes: u64,
    gap_bytes: u64,
}

struct Engine<'a> {
    trace: &'a Trace,
    calibration: &'a Calibration,
    config: SimConfig,
    table: BTreeMap<ShardId, TraceShard>,
    position: BTreeMap<ShardId, Vec<u64>>,
    cache: SimCache,
    rng: SplitMix64,
    servers: Vec<u64>,
    transfers: Vec<Transfer>,
    next_id: u64,
    now: u64,
    history: BTreeMap<ShardId, u64>,
    current_phase: Option<String>,
    active_envelope: Option<Envelope>,
    envelope_records: u64,
    envelope_needed_bytes: u64,
    envelope_miss_bytes: u64,
    pinned_working_set: Vec<ShardId>,
    pending_cold: Vec<ShardId>,
    prefetch_resident: BTreeSet<ShardId>,
    last_prefetch_candidates: u64,
    invalidations: u64,
    summary: SimSummary,
    stall_samples: Vec<u64>,
    outcomes: Vec<RecordOutcome>,
    phase_rows: BTreeMap<String, PhaseRow>,
    task_rows: BTreeMap<String, TaskRow>,
    start_record: usize,
}

#[derive(Clone, Debug, Default)]
struct PhaseRow {
    records: u64,
    bytes_needed: u64,
    bytes_device_read: u64,
    bytes_phase_cold: u64,
    samples: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
struct TaskRow {
    records: u64,
    bytes_needed: u64,
    bytes_device_read: u64,
    bytes_phase_cold: u64,
    samples: Vec<u64>,
}

/// Run a simulation.
///
/// # Errors
///
/// Rejects an invalid configuration, a start record beyond the trace, a shard
/// missing from the table, an impossible in-flight budget, a backpressure
/// refusal in [`BackpressureMode::Fail`] and an injected checksum failure.
pub fn simulate(
    trace: &Trace,
    calibration: &Calibration,
    config: SimConfig,
    predictor: &mut dyn Predictor,
    envelopes: Option<&EnvelopeSet>,
) -> Result<Simulation> {
    config.validate()?;
    if trace.records.is_empty() {
        return Err(SimError::invalid("trace", "has no records"));
    }
    let start_record = config.start_record as usize;
    if start_record >= trace.records.len() {
        return Err(SimError::invalid(
            "start_record",
            "is not a record index in this trace",
        ));
    }
    let table = trace.table()?;
    let position = physical_map(trace, config.placement.as_ref(), &table)?;
    let cache = SimCache::new(config.policy, config.cache_bytes)?;
    let mut engine = Engine {
        trace,
        calibration,
        rng: SplitMix64::new(config.seed),
        servers: vec![0; config.queue_depth.max(1) as usize],
        cache,
        table,
        position,
        transfers: Vec::new(),
        next_id: 1,
        now: 0,
        history: BTreeMap::new(),
        current_phase: None,
        active_envelope: None,
        envelope_records: 0,
        envelope_needed_bytes: 0,
        envelope_miss_bytes: 0,
        pinned_working_set: Vec::new(),
        pending_cold: Vec::new(),
        prefetch_resident: BTreeSet::new(),
        last_prefetch_candidates: 0,
        invalidations: 0,
        summary: SimSummary {
            deadline_ns: config.deadline_ns,
            ..SimSummary::default()
        },
        stall_samples: Vec::new(),
        outcomes: Vec::new(),
        phase_rows: BTreeMap::new(),
        task_rows: BTreeMap::new(),
        start_record,
        config,
    };
    engine.run(predictor, envelopes)?;
    engine.finish()
}

fn physical_map(
    trace: &Trace,
    placement: Option<&Placement>,
    table: &BTreeMap<ShardId, TraceShard>,
) -> Result<BTreeMap<ShardId, Vec<u64>>> {
    let mut position: BTreeMap<ShardId, Vec<u64>> = BTreeMap::new();
    match placement {
        Some(placement) => {
            for extent in &placement.extents {
                position
                    .entry(extent.shard)
                    .or_default()
                    .push(extent.offset);
            }
            for shard in table.keys() {
                if !position.contains_key(shard) {
                    return Err(SimError::invalid(
                        "placement",
                        format!("placement omits shard {}", shard.short()),
                    ));
                }
            }
        }
        None => {
            for shard in trace.shards() {
                position.entry(shard.id).or_default().push(shard.offset);
            }
        }
    }
    Ok(position)
}

impl Engine<'_> {
    fn run(
        &mut self,
        predictor: &mut dyn Predictor,
        envelopes: Option<&EnvelopeSet>,
    ) -> Result<()> {
        let start = self.start_record;
        let records = self.trace.records.len();
        for index in start..records {
            self.settle(self.now)?;
            let record = self.trace.records[index].clone();
            self.begin_phase(&record, envelopes);
            self.process_record(index, &record, predictor)?;
        }
        Ok(())
    }

    fn begin_phase(&mut self, record: &TraceRecord, envelopes: Option<&EnvelopeSet>) {
        if self.current_phase.as_deref() == Some(record.phase_id.as_str()) {
            return;
        }
        self.current_phase = Some(record.phase_id.clone());
        self.envelope_records = 0;
        self.envelope_needed_bytes = 0;
        self.envelope_miss_bytes = 0;
        let previous = self.active_envelope.take();
        self.active_envelope =
            envelopes.and_then(|set| set.envelope_for(&record.phase_id).cloned());
        if previous.is_some() && self.active_envelope.is_none() {
            self.invalidations = self.invalidations.saturating_add(1);
        }
        if self.config.cancel_prefetch_on_phase_change {
            let allowed: Option<BTreeSet<ShardId>> = self
                .active_envelope
                .as_ref()
                .map(|envelope| envelope.candidate.iter().copied().collect());
            let mut cancelled = 0_u64;
            let mut kept = Vec::with_capacity(self.transfers.len());
            for transfer in std::mem::take(&mut self.transfers) {
                let keep = !transfer.prefetch
                    || transfer.consumed
                    || allowed
                        .as_ref()
                        .is_none_or(|set| transfer.shard.is_some_and(|shard| set.contains(&shard)));
                if keep {
                    kept.push(transfer);
                } else {
                    cancelled = cancelled.saturating_add(transfer.bytes);
                    self.servers[transfer.server] = self.now;
                }
            }
            self.transfers = kept;
            self.summary.bytes_prefetch_cancelled = self
                .summary
                .bytes_prefetch_cancelled
                .saturating_add(cancelled);
        }
        // Choose the phase working set; shards not resident become warm-up.
        self.pending_cold.clear();
        if let Some(two) = self.config.two_timescale {
            let candidates = self.working_set_candidates(two);
            let budget = (self.config.cache_bytes as f64 * two.capacity_fraction) as u64;
            let mut used = 0_u64;
            for shard in candidates {
                let bytes = self.table.get(&shard).map_or(0, |shard| shard.bytes);
                if used.saturating_add(bytes) > budget {
                    break;
                }
                used = used.saturating_add(bytes);
                if !self.cache.contains(&shard) {
                    self.pending_cold.push(shard);
                }
            }
        }
    }

    fn working_set_candidates(&self, two: TwoTimescaleConfig) -> Vec<ShardId> {
        match two.source {
            WorkingSetSource::Envelope => self
                .active_envelope
                .as_ref()
                .map(|envelope| envelope.candidate.clone())
                .unwrap_or_default(),
            WorkingSetSource::History => {
                let mut ranked: Vec<(u64, ShardId)> = self
                    .history
                    .iter()
                    .filter(|(_, count)| **count >= two.min_usage.max(1))
                    .map(|(shard, count)| (*count, *shard))
                    .collect();
                ranked.sort_unstable_by(|left, right| {
                    right.0.cmp(&left.0).then(left.1.cmp(&right.1))
                });
                ranked.into_iter().map(|(_, shard)| shard).collect()
            }
        }
    }

    fn process_record(
        &mut self,
        index: usize,
        record: &TraceRecord,
        predictor: &mut dyn Predictor,
    ) -> Result<()> {
        let need_time = self.now;
        let mut required: Vec<ShardId> = record.selected.clone();
        let mut cold: BTreeSet<ShardId> = BTreeSet::new();
        for shard in std::mem::take(&mut self.pending_cold) {
            if record.selected.binary_search(&shard).is_err() {
                cold.insert(shard);
                required.push(shard);
            }
        }
        required.sort_unstable();
        required.dedup();
        let bytes_needed: u64 = record
            .selected
            .iter()
            .map(|shard| self.table.get(shard).map_or(0, |shard| shard.bytes))
            .fold(0_u64, u64::saturating_add);

        let mut hit_bytes = 0_u64;
        let mut hit_cost = 0.0_f64;
        let mut to_fetch: Vec<ShardId> = Vec::new();
        let mut awaited: Vec<(ShardId, u64)> = Vec::new();
        let mut prefetch_used_bytes = 0_u64;
        for shard in &required {
            let bytes = self.table.get(shard).map_or(0, |shard| shard.bytes);
            let selected = record.selected.binary_search(shard).is_ok();
            if self.cache.get(shard) {
                if selected {
                    hit_bytes = hit_bytes.saturating_add(bytes);
                    // A resident shard that arrived speculatively is a prefetch
                    // use, not a reader's fetch.
                    if self.prefetch_resident.remove(shard) {
                        self.summary.bytes_prefetch_used =
                            self.summary.bytes_prefetch_used.saturating_add(bytes);
                        prefetch_used_bytes = prefetch_used_bytes.saturating_add(bytes);
                    }
                }
                // A cache hit is a copy, not a free read: the measured engine
                // charges it in the same stall.
                hit_cost += self.calibration.sample_hit_ns(&mut self.rng, bytes)?;
                continue;
            }
            let index = self.transfers.iter().position(|transfer| {
                transfer.prefetch && !transfer.consumed && transfer.shard == Some(*shard)
            });
            if let Some(index) = index {
                self.transfers[index].consumed = true;
                let finish_ns = self.transfers[index].finish_ns;
                awaited.push((*shard, finish_ns));
                if selected {
                    // The bytes were fetched speculatively but are served
                    // from host RAM at need time, so they are a hit for the
                    // record and a use for the prefetch.
                    hit_bytes = hit_bytes.saturating_add(bytes);
                    self.summary.bytes_prefetch_used =
                        self.summary.bytes_prefetch_used.saturating_add(bytes);
                    prefetch_used_bytes = prefetch_used_bytes.saturating_add(bytes);
                    hit_cost += self.calibration.sample_hit_ns(&mut self.rng, bytes)?;
                }
                continue;
            }
            to_fetch.push(*shard);
        }

        let selected: BTreeSet<ShardId> = record.selected.iter().copied().collect();
        let runs = self.plan_runs(&to_fetch);
        let mut finish = need_time;
        let mut required_miss = 0_u64;
        let mut phase_cold = 0_u64;
        let mut overread = 0_u64;
        let mut device = 0_u64;
        for run in &runs {
            let (server, start) = self.reserve_slot(need_time, run.total_bytes)?;
            let transfer = self
                .calibration
                .sample_transfer_ns(&mut self.rng, run.total_bytes)?;
            let verify = if self.config.verify {
                self.calibration
                    .sample_verify_ns(&mut self.rng, run.total_bytes)?
            } else {
                0.0
            };
            if let Some(failed) = self.config.inject_checksum_failure
                && run.members.contains(&failed)
            {
                return Err(SimError::invalid(
                    "checksum",
                    format!("injected verification failure for shard {}", failed.short()),
                ));
            }
            // Admit the run's members and charge the engine's insertion and
            // eviction cost, which the measured engine includes in its stall.
            let mut inserted = 0_u64;
            let mut evictions = 0_u64;
            for member in &run.members {
                let bytes = self.table.get(member).map_or(0, |shard| shard.bytes);
                let outcome = self.cache.insert(*member, bytes)?;
                self.prefetch_resident.remove(member);
                if outcome.admitted {
                    inserted = inserted.saturating_add(1);
                }
                evictions = evictions.saturating_add(outcome.evictions);
            }
            let insert_cost = if inserted > 0 {
                self.calibration.sample_insert_ns(&mut self.rng)?
            } else {
                0.0
            };
            let eviction_cost = if evictions > 0 {
                self.calibration
                    .sample_eviction_ns(&mut self.rng, evictions)?
            } else {
                0.0
            };
            let done = start
                .saturating_add(transfer.round().max(0.0) as u64)
                .saturating_add(verify.round().max(0.0) as u64)
                .saturating_add(insert_cost.round().max(0.0) as u64)
                .saturating_add(eviction_cost.round().max(0.0) as u64);
            self.commit_required(server, run.total_bytes, done, run);
            finish = finish.max(done);
            for member in &run.members {
                let bytes = self.table.get(member).map_or(0, |shard| shard.bytes);
                if selected.contains(member) {
                    required_miss = required_miss.saturating_add(bytes);
                } else {
                    phase_cold = phase_cold.saturating_add(bytes);
                }
            }
            overread = overread.saturating_add(run.gap_bytes);
            device = device.saturating_add(run.total_bytes);
        }
        let awaited_finish = awaited.iter().map(|(_, at)| *at).max().unwrap_or(need_time);
        finish = finish.max(awaited_finish);
        let stall = finish
            .saturating_sub(need_time)
            .saturating_add(hit_cost.round().max(0.0) as u64);
        self.now = need_time
            .saturating_add(stall)
            .saturating_add(self.config.compute_ns_per_record);
        self.settle(self.now)?;
        self.pin_working_set();

        let counters = self.cache.counters();
        self.summary.hits = counters.hits;
        self.summary.misses = counters.misses;
        self.summary.evictions = counters.evictions;
        self.summary.evicted_bytes = counters.evicted_bytes;
        self.summary.evicted_unused_bytes = counters.evicted_unused_bytes;
        self.summary.records = self.summary.records.saturating_add(1);
        self.summary.bytes_needed = self.summary.bytes_needed.saturating_add(bytes_needed);
        self.summary.bytes_hit = self.summary.bytes_hit.saturating_add(hit_bytes);
        self.summary.bytes_required_miss = self
            .summary
            .bytes_required_miss
            .saturating_add(required_miss);
        self.summary.bytes_phase_cold = self.summary.bytes_phase_cold.saturating_add(phase_cold);
        self.summary.bytes_overread = self.summary.bytes_overread.saturating_add(overread);
        self.summary.bytes_device_read = self.summary.bytes_device_read.saturating_add(device);
        let deadline_missed = self
            .config
            .deadline_ns
            .is_some_and(|deadline| stall > deadline);
        if deadline_missed {
            self.summary.deadline_misses = self.summary.deadline_misses.saturating_add(1);
        }
        self.summary.evaluated_records = self.summary.evaluated_records.saturating_add(1);
        self.stall_samples.push(stall);
        self.envelope_accounting(bytes_needed, bytes_needed.saturating_sub(hit_bytes));
        self.issue_prefetch(index, record, predictor, finish)?;
        for shard in &record.selected {
            *self.history.entry(*shard).or_insert(0) += 1;
        }
        self.outcomes.push(RecordOutcome {
            index: index as u64,
            start_ns: need_time,
            end_ns: self.now,
            request_id: record.request_id.clone(),
            phase_id: record.phase_id.clone(),
            token_index: record.token_index,
            layer: record.layer,
            bytes_needed,
            bytes_hit: hit_bytes,
            bytes_fetched: required_miss,
            stall_ns: stall,
            deadline_missed,
            prefetch_candidates: self.last_prefetch_candidates,
            prefetch_used_bytes,
        });
        self.record_rows(record, stall, bytes_needed, device, phase_cold);
        Ok(())
    }

    fn envelope_accounting(&mut self, needed: u64, missed: u64) {
        let Some(envelope) = self.active_envelope.clone() else {
            return;
        };
        self.envelope_records = self.envelope_records.saturating_add(1);
        self.envelope_needed_bytes = self.envelope_needed_bytes.saturating_add(needed);
        self.envelope_miss_bytes = self.envelope_miss_bytes.saturating_add(missed);
        let mut invalidate = false;
        for rule in &envelope.invalidate_on {
            match rule {
                Invalidation::PhaseChange => {}
                Invalidation::AfterRecords { records } => {
                    if self.envelope_records > *records {
                        invalidate = true;
                    }
                }
                Invalidation::MissRateAbove { rate, min_records } => {
                    if self.envelope_records >= *min_records && self.envelope_needed_bytes > 0 {
                        let observed =
                            self.envelope_miss_bytes as f64 / self.envelope_needed_bytes as f64;
                        if observed > *rate {
                            invalidate = true;
                        }
                    }
                }
            }
        }
        if invalidate {
            self.active_envelope = None;
            self.invalidations = self.invalidations.saturating_add(1);
        }
    }

    fn issue_prefetch(
        &mut self,
        index: usize,
        record: &TraceRecord,
        predictor: &mut dyn Predictor,
        issue_time: u64,
    ) -> Result<()> {
        let context = PredictContext {
            record_index: index,
            current: record,
            previous: index.checked_sub(1).map(|slot| &self.trace.records[slot]),
            next: self.trace.records.get(index + 1),
            envelope: self.active_envelope.as_ref(),
            capacity_bytes: self.config.cache_bytes,
            table: &self.table,
        };
        let mut candidates = predictor.predict(&context);
        candidates.sort_unstable();
        candidates.dedup();
        candidates.truncate(self.config.max_prefetch_shards.max(1));
        let proposed = candidates.len() as u64;
        self.last_prefetch_candidates = proposed;
        let mut accepted = 0_u64;
        let mut rejected = 0_u64;
        for shard in candidates {
            if self.cache.contains(&shard)
                || self.transfers.iter().any(|transfer| {
                    transfer.prefetch && !transfer.consumed && transfer.shard == Some(shard)
                })
            {
                continue;
            }
            let bytes = self.table.get(&shard).map_or(0, |shard| shard.bytes);
            if bytes == 0 {
                continue;
            }
            if self.cache.used().saturating_add(bytes) > self.cache.capacity() {
                rejected = rejected.saturating_add(1);
                continue;
            }
            let (server, start) = self.reserve_slot(issue_time, bytes)?;
            let transfer = self.calibration.sample_transfer_ns(&mut self.rng, bytes)?;
            let verify = if self.config.verify {
                self.calibration.sample_verify_ns(&mut self.rng, bytes)?
            } else {
                0.0
            };
            let insert = self.calibration.sample_insert_ns(&mut self.rng)?;
            let done = start
                .saturating_add(transfer.round().max(0.0) as u64)
                .saturating_add(verify.round().max(0.0) as u64)
                .saturating_add(insert.round().max(0.0) as u64);
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.servers[server] = done;
            self.transfers.push(Transfer {
                id,
                shard: Some(shard),
                bytes,
                finish_ns: done,
                server,
                prefetch: true,
                consumed: false,
            });
            accepted = accepted.saturating_add(1);
        }
        self.summary.prefetch_candidates =
            self.summary.prefetch_candidates.saturating_add(proposed);
        self.summary.prefetch_accepted = self.summary.prefetch_accepted.saturating_add(accepted);
        self.summary.prefetch_rejected_capacity = self
            .summary
            .prefetch_rejected_capacity
            .saturating_add(rejected);
        Ok(())
    }

    fn pin_working_set(&mut self) {
        let Some(two) = self.config.two_timescale else {
            return;
        };
        let budget = (self.config.cache_bytes as f64 * two.capacity_fraction) as u64;
        let mut pinned = Vec::new();
        let mut used = 0_u64;
        for shard in self.working_set_candidates(two) {
            let bytes = self.table.get(&shard).map_or(0, |shard| shard.bytes);
            if used.saturating_add(bytes) > budget {
                break;
            }
            if self.cache.contains(&shard) && self.cache.pin(&shard) {
                used = used.saturating_add(bytes);
                pinned.push(shard);
            }
        }
        for shard in std::mem::take(&mut self.pinned_working_set) {
            if !pinned.contains(&shard) {
                self.cache.unpin(&shard);
            }
        }
        self.pinned_working_set = pinned;
    }

    fn settle(&mut self, now: u64) -> Result<()> {
        let mut completed: Vec<Transfer> = Vec::new();
        let mut kept: Vec<Transfer> = Vec::new();
        for transfer in std::mem::take(&mut self.transfers) {
            if transfer.finish_ns <= now {
                completed.push(transfer);
            } else {
                kept.push(transfer);
            }
        }
        self.transfers = kept;
        completed.sort_unstable_by_key(|transfer| (transfer.finish_ns, transfer.id));
        for transfer in completed {
            if transfer.prefetch {
                self.summary.bytes_prefetch =
                    self.summary.bytes_prefetch.saturating_add(transfer.bytes);
                self.summary.bytes_device_read = self
                    .summary
                    .bytes_device_read
                    .saturating_add(transfer.bytes);
                if let Some(shard) = transfer.shard {
                    self.cache.insert(shard, transfer.bytes)?;
                    // A transfer that was awaited by a record was already
                    // counted as used; registering it again would double-count.
                    if !transfer.consumed {
                        self.prefetch_resident.insert(shard);
                    }
                }
            }
        }
        Ok(())
    }

    fn plan_runs(&self, to_fetch: &[ShardId]) -> Vec<RunPlan> {
        if to_fetch.is_empty() {
            return Vec::new();
        }
        // Choose one extent per shard, preferring a copy at or after a cursor
        // that advances with the chosen offsets, so a replicated layout can form
        // longer runs.
        let mut shards = to_fetch.to_vec();
        shards.sort_unstable();
        let mut cursor = 0_u64;
        let mut chosen: Vec<(u64, ShardId, u64)> = Vec::with_capacity(shards.len());
        for shard in shards {
            let bytes = self.table.get(&shard).map_or(0, |shard| shard.bytes);
            let positions = self.position.get(&shard);
            let offset = positions
                .into_iter()
                .flatten()
                .copied()
                .filter(|offset| *offset >= cursor)
                .min()
                .or_else(|| positions.and_then(|positions| positions.first().copied()))
                .unwrap_or(cursor);
            cursor = offset;
            chosen.push((offset, shard, bytes));
        }
        chosen.sort_unstable_by_key(|(offset, shard, _)| (*offset, *shard));
        let mut runs: Vec<RunPlan> = Vec::new();
        let mut current: Option<(RunPlan, u64)> = None;
        for (offset, shard, bytes) in chosen {
            match &mut current {
                None => {
                    current = Some((
                        RunPlan {
                            members: vec![shard],
                            total_bytes: bytes,
                            gap_bytes: 0,
                        },
                        offset.saturating_add(bytes),
                    ));
                }
                Some((run, end)) => {
                    let gap = offset.saturating_sub(*end);
                    let projected = run.total_bytes.saturating_add(gap).saturating_add(bytes);
                    if gap <= self.config.coalesce_gap_bytes
                        && projected <= self.config.max_run_bytes
                    {
                        run.members.push(shard);
                        run.total_bytes = projected;
                        run.gap_bytes = run.gap_bytes.saturating_add(gap);
                        *end = offset.saturating_add(bytes);
                    } else {
                        let fresh = RunPlan {
                            members: vec![shard],
                            total_bytes: bytes,
                            gap_bytes: 0,
                        };
                        let done = std::mem::replace(run, fresh);
                        runs.push(done);
                        *end = offset.saturating_add(bytes);
                    }
                }
            }
        }
        if let Some((run, _)) = current {
            runs.push(run);
        }
        runs
    }

    fn earliest_server(&self) -> (usize, u64) {
        let mut best = (0usize, u64::MAX);
        for (index, free_at) in self.servers.iter().enumerate() {
            if *free_at < best.1 {
                best = (index, *free_at);
            }
        }
        best
    }

    fn admit_time(&self, now: u64, bytes: u64) -> Result<u64> {
        if bytes > self.config.max_in_flight_bytes {
            return Err(SimError::invalid(
                "max_in_flight_bytes",
                format!(
                    "a {bytes} byte transfer can never fit the {} byte budget",
                    self.config.max_in_flight_bytes
                ),
            ));
        }
        let mut at = now;
        loop {
            let mut future: Vec<(u64, u64)> = self
                .transfers
                .iter()
                .filter(|transfer| transfer.finish_ns > at)
                .map(|transfer| (transfer.finish_ns, transfer.bytes))
                .collect();
            future.sort_unstable();
            let in_flight: u64 = future
                .iter()
                .map(|(_, bytes)| *bytes)
                .fold(0_u64, u64::saturating_add);
            if in_flight.saturating_add(bytes) <= self.config.max_in_flight_bytes {
                return Ok(at);
            }
            match future.first() {
                Some((finish, _)) => at = *finish,
                None => return Ok(at),
            }
        }
    }

    fn reserve_slot(&mut self, need_time: u64, bytes: u64) -> Result<(usize, u64)> {
        let admitted = self.admit_time(need_time, bytes)?;
        if admitted > need_time && self.config.backpressure == BackpressureMode::Fail {
            return Err(SimError::invalid(
                "backpressure",
                format!("a required transfer of {bytes} bytes would wait until {admitted} ns"),
            ));
        }
        if admitted > need_time {
            self.summary.backpressure_waits = self.summary.backpressure_waits.saturating_add(1);
        }
        if self.config.preempt_prefetch_for_required {
            loop {
                let (server, free_at) = self.earliest_server();
                if free_at <= admitted {
                    break;
                }
                let victim = self
                    .transfers
                    .iter()
                    .enumerate()
                    .filter(|(_, transfer)| {
                        transfer.server == server && transfer.prefetch && !transfer.consumed
                    })
                    .max_by_key(|(_, transfer)| (transfer.finish_ns, transfer.id))
                    .map(|(index, _)| index);
                match victim {
                    Some(index) => {
                        let cancelled = self.transfers[index].bytes;
                        self.summary.bytes_prefetch_cancelled = self
                            .summary
                            .bytes_prefetch_cancelled
                            .saturating_add(cancelled);
                        self.servers[server] = admitted;
                        self.transfers.remove(index);
                    }
                    None => break,
                }
            }
        }
        let (server, free_at) = self.earliest_server();
        Ok((server, free_at.max(admitted).max(need_time)))
    }

    fn commit_required(&mut self, server: usize, bytes: u64, finish: u64, run: &RunPlan) {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.servers[server] = finish;
        self.transfers.push(Transfer {
            id,
            shard: run.members.first().copied(),
            bytes,
            finish_ns: finish,
            server,
            prefetch: false,
            consumed: true,
        });
    }

    fn record_rows(
        &mut self,
        record: &TraceRecord,
        stall: u64,
        bytes_needed: u64,
        device: u64,
        phase_cold: u64,
    ) {
        let phase = self.phase_rows.entry(record.phase_id.clone()).or_default();
        phase.records = phase.records.saturating_add(1);
        phase.bytes_needed = phase.bytes_needed.saturating_add(bytes_needed);
        phase.bytes_device_read = phase.bytes_device_read.saturating_add(device);
        phase.bytes_phase_cold = phase.bytes_phase_cold.saturating_add(phase_cold);
        phase.samples.push(stall);
        let key = record
            .task
            .clone()
            .unwrap_or_else(|| "unlabelled".to_owned());
        let task = self.task_rows.entry(key).or_default();
        task.records = task.records.saturating_add(1);
        task.bytes_needed = task.bytes_needed.saturating_add(bytes_needed);
        task.bytes_device_read = task.bytes_device_read.saturating_add(device);
        task.bytes_phase_cold = task.bytes_phase_cold.saturating_add(phase_cold);
        task.samples.push(stall);
    }

    fn finish(mut self) -> Result<Simulation> {
        let mut cancelled = 0_u64;
        for transfer in &self.transfers {
            if transfer.prefetch && !transfer.consumed && transfer.finish_ns > self.now {
                cancelled = cancelled.saturating_add(transfer.bytes);
            }
        }
        self.summary.bytes_prefetch_cancelled = self
            .summary
            .bytes_prefetch_cancelled
            .saturating_add(cancelled);
        self.summary.bytes_prefetch_wasted = self
            .summary
            .bytes_prefetch
            .saturating_sub(self.summary.bytes_prefetch_used);
        self.summary.bytes_wasted = self
            .summary
            .bytes_overread
            .saturating_add(self.summary.bytes_prefetch_wasted);
        self.summary.stall = LatencySummary::of(&self.stall_samples);
        self.summary.end_mono_ns = self.now;
        self.summary.envelope_invalidations = self.invalidations;
        self.summary.check()?;
        let phases = self
            .phase_rows
            .into_iter()
            .map(|(phase_id, row)| PhaseSummary {
                phase_id,
                records: row.records,
                bytes_needed: row.bytes_needed,
                bytes_device_read: row.bytes_device_read,
                bytes_phase_cold: row.bytes_phase_cold,
                stall: LatencySummary::of(&row.samples),
            })
            .collect();
        let tasks = self
            .task_rows
            .into_iter()
            .map(|(task, row)| TaskSummary {
                task,
                records: row.records,
                bytes_needed: row.bytes_needed,
                bytes_device_read: row.bytes_device_read,
                bytes_phase_cold: row.bytes_phase_cold,
                stall_p95_ns: LatencySummary::of(&row.samples).p95_ns,
            })
            .collect();
        Ok(Simulation {
            summary: self.summary,
            phases,
            tasks,
            outcomes: self.outcomes,
            invalidations: self.invalidations,
        })
    }
}
