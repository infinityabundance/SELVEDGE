//! Placement: physical ordering of shards, and the objective used to rank it.
//!
//! The mandate's placement problem is to minimise expected stall plus bytes plus
//! eviction cost over the physical ordering. The simulator measures the stall and
//! bytes for a candidate order; this module supplies (a) the candidate orders and
//! (b) a fast surrogate used only to *propose* them.
//!
//! The surrogate's edge weight is **not** raw co-occurrence. For an unordered
//! pair `(a, b)` it is
//!
//! ```text
//! w(a,b) = lift(a,b) * marginal_ns(a,b)
//! lift   = P(a,b) / (P(a) P(b))        measured over trace records
//! margin = L(ba) + L(bb) - L(ba + bb)  from the calibrated size law
//! ```
//!
//! `lift` says how much more often the pair co-occurs than independence would
//! predict, and `marginal_ns` says how much time a *coalesced* transfer would
//! save. Their product is a local estimate of saved exposed stall, so a pair
//! that co-occurs often but is trivial to fetch separately is not promoted.
//! Uncertainty is reported: a seeded record-level bootstrap gives a confidence
//! interval for the lift of the strongest edges, and the report states how many
//! of the sampled edges are stable (interval excludes 1.0).

use std::collections::{BTreeMap, BTreeSet};

use selvedge_core::ShardId;
use serde::{Deserialize, Serialize};

use crate::calibration::SizeLaw;
use crate::error::{Result, SimError};
use crate::prng::SplitMix64;
use crate::trace::{Trace, TraceShard};

/// Schema of a placement report.
pub const PLACEMENT_SCHEMA: &str = "selvedge/sim-placement/v1";

/// Candidate physical layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementKind {
    /// The order the shard table already has (physical offset ascending).
    Sequential,
    /// Descending access frequency.
    FrequencySorted,
    /// Greedy blocks of shards that are needed together.
    CoAccessBlocked,
    /// Co-access blocks plus replicated hot fragments at phase boundaries.
    ReplicatedHot,
}

impl PlacementKind {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::FrequencySorted => "frequency-sorted",
            Self::CoAccessBlocked => "co-access-blocked",
            Self::ReplicatedHot => "replicated-hot",
        }
    }

    /// Parse a stable name.
    ///
    /// # Errors
    ///
    /// Rejects an unknown name.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "sequential" => Ok(Self::Sequential),
            "frequency-sorted" => Ok(Self::FrequencySorted),
            "co-access-blocked" => Ok(Self::CoAccessBlocked),
            "replicated-hot" => Ok(Self::ReplicatedHot),
            other => Err(SimError::invalid(
                "placement",
                format!("unknown placement {other}"),
            )),
        }
    }
}

/// Knobs shared by the builders.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementParams {
    /// Shards per co-access block.
    pub block_shards: usize,
    /// Hot shards eligible for replication.
    pub hot_shards: usize,
    /// Record-level bootstrap resamples for the lift interval.
    pub bootstrap_resamples: usize,
    /// Edges whose lift interval is reported.
    pub reported_edges: usize,
}

impl Default for PlacementParams {
    fn default() -> Self {
        Self {
            block_shards: 16,
            hot_shards: 64,
            bootstrap_resamples: 200,
            reported_edges: 32,
        }
    }
}

impl PlacementParams {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// Rejects a zero block size or resample count.
    pub fn validate(&self) -> Result<()> {
        if self.block_shards == 0 {
            return Err(SimError::invalid("block_shards", "must be non-zero"));
        }
        if self.bootstrap_resamples == 0 {
            return Err(SimError::invalid("bootstrap_resamples", "must be non-zero"));
        }
        Ok(())
    }
}

/// One placed extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacedExtent {
    /// Shard.
    pub shard: ShardId,
    /// Extent length.
    pub bytes: u64,
    /// Synthetic physical offset. Layouts pack extents back-to-back, so two
    /// extents are adjacent exactly when their offsets are.
    pub offset: u64,
    /// Replica index; zero for the first copy.
    pub replica: u32,
}

/// A physical layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    /// Layout family.
    pub kind: PlacementKind,
    /// Parameters used.
    pub params: PlacementParams,
    /// Extents in physical order.
    pub extents: Vec<PlacedExtent>,
    /// Unique shards placed.
    pub unique_shards: u64,
    /// Unique bytes.
    pub unique_bytes: u64,
    /// Placed bytes including replicas.
    pub placed_bytes: u64,
    /// Bytes attributable to replication.
    pub replicated_bytes: u64,
    /// Storage amplification `placed_bytes / unique_bytes`.
    pub amplification: f64,
}

/// A measured co-access edge and its weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedEdge {
    /// Lower shard id.
    pub a: ShardId,
    /// Higher shard id.
    pub b: ShardId,
    /// Records containing both.
    pub co_access: u64,
    /// `P(a,b) / (P(a) P(b))`.
    pub lift: f64,
    /// Lift interval from the record bootstrap.
    pub lift_low: f64,
    /// Upper end of the lift interval.
    pub lift_high: f64,
    /// Coalesced-transfer saving from the size law, in nanoseconds.
    pub marginal_ns: f64,
    /// `lift * marginal_ns`.
    pub weight: f64,
}

/// The co-access weight table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoAccessWeights {
    /// Records used.
    pub records: u64,
    /// Unique shards.
    pub shards: u64,
    /// Edges, descending by weight.
    pub edges: Vec<WeightedEdge>,
    /// Sampled edges whose lift interval excludes 1.0.
    pub stable_edges: u64,
    /// Sampled edges in the stability check.
    pub checked_edges: u64,
    /// Fraction of checked edges that are stable.
    pub stable_fraction: f64,
    /// Methodology statement.
    pub weighting: String,
    /// Caveats.
    pub notes: Vec<String>,
}

impl CoAccessWeights {
    /// Weight of an unordered pair.
    #[must_use]
    pub fn weight(&self, a: &ShardId, b: &ShardId) -> f64 {
        let (low, high) = canonical(a, b);
        self.edges
            .binary_search_by(|edge| {
                (edge.a, edge.b)
                    .cmp(&(low, high))
                    .then(std::cmp::Ordering::Equal)
            })
            .ok()
            .map_or(0.0, |index| self.edges[index].weight)
    }

    /// A weight lookup table keyed by canonical pairs.
    #[must_use]
    pub fn table(&self) -> BTreeMap<(ShardId, ShardId), f64> {
        self.edges
            .iter()
            .map(|edge| ((edge.a, edge.b), edge.weight))
            .collect()
    }
}

fn canonical(a: &ShardId, b: &ShardId) -> (ShardId, ShardId) {
    if a <= b { (*a, *b) } else { (*b, *a) }
}

/// Build the co-access weight table from a trace.
///
/// `size_law` supplies the coalescing saving; without it the marginal cost is
/// reported as zero and the weights degenerate to lift, which the returned
/// notes state.
///
/// # Errors
///
/// Rejects an empty trace, a zero resample count and size-law failures.
pub fn weights(
    trace: &Trace,
    size_law: Option<&SizeLaw>,
    params: PlacementParams,
    seed: u64,
) -> Result<CoAccessWeights> {
    params.validate()?;
    if trace.records.is_empty() {
        return Err(SimError::invalid("trace", "has no records"));
    }
    let table: BTreeMap<ShardId, TraceShard> = trace.table()?;
    let records = trace.records.len() as u64;
    let mut per_record: Vec<Vec<ShardId>> = trace
        .records
        .iter()
        .map(|record| record.selected.clone())
        .collect();
    for selected in &mut per_record {
        selected.sort_unstable();
        selected.dedup();
    }
    let mut unary: BTreeMap<ShardId, u64> = BTreeMap::new();
    let mut pair_counts: BTreeMap<(ShardId, ShardId), u64> = BTreeMap::new();
    for selected in &per_record {
        for shard in selected {
            *unary.entry(*shard).or_insert(0) += 1;
        }
        for left in 0..selected.len() {
            for right in (left + 1)..selected.len() {
                *pair_counts
                    .entry((selected[left], selected[right]))
                    .or_insert(0) += 1;
            }
        }
    }
    let n = records as f64;
    let mut edges = Vec::with_capacity(pair_counts.len());
    for ((a, b), count) in &pair_counts {
        let count = *count;
        if count < 2 {
            continue;
        }
        let ca = *unary.get(a).unwrap_or(&0) as f64;
        let cb = *unary.get(b).unwrap_or(&0) as f64;
        let lift = (count as f64 * n) / (ca * cb).max(1.0);
        let (bytes_a, bytes_b) = (
            table.get(a).map_or(1, |shard| shard.bytes.max(1)),
            table.get(b).map_or(1, |shard| shard.bytes.max(1)),
        );
        let marginal_ns = match size_law {
            Some(law) => {
                let separate = law.latency_ns(bytes_a)? + law.latency_ns(bytes_b)?;
                let together = law.latency_ns(bytes_a + bytes_b)?;
                (separate - together).max(0.0)
            }
            None => 0.0,
        };
        edges.push(WeightedEdge {
            a: *a,
            b: *b,
            co_access: count,
            lift,
            lift_low: lift,
            lift_high: lift,
            marginal_ns,
            weight: lift * marginal_ns,
        });
    }
    // Order deterministically before the bootstrap selects the reported subset.
    edges.sort_unstable_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then(left.co_access.cmp(&right.co_access))
            .then((left.a, left.b).cmp(&(right.a, right.b)))
    });
    // Record-level bootstrap for the lift of the reported edges.
    let reported = params.reported_edges.min(edges.len());
    if reported > 0 {
        let mut rng = SplitMix64::new(seed);
        let mut lows = vec![f64::MAX; reported];
        let mut highs = vec![0.0_f64; reported];
        for _ in 0..params.bootstrap_resamples {
            let mut resample_unary: BTreeMap<ShardId, u64> = BTreeMap::new();
            let mut resample_pairs: BTreeMap<(ShardId, ShardId), u64> = BTreeMap::new();
            for _ in 0..per_record.len() {
                let index = rng.below(per_record.len() as u64)? as usize;
                let selected = &per_record[index];
                for shard in selected {
                    *resample_unary.entry(*shard).or_insert(0) += 1;
                }
                for left in 0..selected.len() {
                    for right in (left + 1)..selected.len() {
                        *resample_pairs
                            .entry((selected[left], selected[right]))
                            .or_insert(0) += 1;
                    }
                }
            }
            for (slot, edge) in edges[..reported].iter().enumerate() {
                let count = *resample_pairs.get(&(edge.a, edge.b)).unwrap_or(&0) as f64;
                let ca = *resample_unary.get(&edge.a).unwrap_or(&0) as f64;
                let cb = *resample_unary.get(&edge.b).unwrap_or(&0) as f64;
                let lift = (count * n) / (ca * cb).max(1.0);
                lows[slot] = lows[slot].min(lift);
                highs[slot] = highs[slot].max(lift);
            }
        }
        for (slot, edge) in edges[..reported].iter_mut().enumerate() {
            edge.lift_low = lows[slot];
            edge.lift_high = highs[slot];
        }
    }
    let stable = edges[..reported]
        .iter()
        .filter(|edge| edge.lift_low > 1.0)
        .count() as u64;
    let stable_fraction = if reported == 0 {
        0.0
    } else {
        stable as f64 / reported as f64
    };
    let mut notes = vec![
        "lift is measured against independence over the trace's records; a lift \
         interval is a seeded record-level bootstrap percentile range, not a \
         significance test"
            .to_owned(),
        "the coalescing saving comes from the calibrated size law, not from a \
         direct measurement of every pair"
            .to_owned(),
    ];
    if size_law.is_none() {
        notes.push(
            "no size law was supplied, so every edge weight is zero and the \
             layout is driven by frequency only"
                .to_owned(),
        );
    }
    Ok(CoAccessWeights {
        records,
        shards: unary.len() as u64,
        edges,
        stable_edges: stable,
        checked_edges: reported as u64,
        stable_fraction,
        weighting: "w(a,b) = lift(a,b) * (L(ba) + L(bb) - L(ba + bb))".to_owned(),
        notes,
    })
}

/// Build a placement.
///
/// # Errors
///
/// Rejects invalid parameters and an empty shard table.
pub fn build(
    trace: &Trace,
    weights: &CoAccessWeights,
    kind: PlacementKind,
    params: PlacementParams,
) -> Result<Placement> {
    params.validate()?;
    let table: BTreeMap<ShardId, TraceShard> = trace.table()?;
    if table.is_empty() {
        return Err(SimError::invalid("shards", "table is empty"));
    }
    let mut frequency: BTreeMap<ShardId, u64> = table.keys().map(|id| (*id, 0)).collect();
    let mut phase_sets: Vec<(String, BTreeSet<ShardId>)> = Vec::new();
    let mut phase_index: BTreeMap<String, usize> = BTreeMap::new();
    for record in &trace.records {
        for shard in &record.selected {
            *frequency.entry(*shard).or_insert(0) += 1;
        }
        match phase_index.get(&record.phase_id) {
            Some(index) => {
                phase_sets[*index].1.extend(record.selected.iter().copied());
            }
            None => {
                phase_index.insert(record.phase_id.clone(), phase_sets.len());
                phase_sets.push((
                    record.phase_id.clone(),
                    record.selected.iter().copied().collect(),
                ));
            }
        }
    }
    let order: Vec<ShardId> = match kind {
        PlacementKind::Sequential => {
            let mut ordered: Vec<ShardId> = table.keys().copied().collect();
            ordered.sort_unstable_by_key(|id| (table[id].offset, *id));
            ordered
        }
        PlacementKind::FrequencySorted => frequency_order(&frequency, &table),
        PlacementKind::CoAccessBlocked => {
            coalesced_order(&frequency, &table, &weights.table(), params.block_shards)
        }
        PlacementKind::ReplicatedHot => {
            replicated_order(&frequency, &table, &weights.table(), params, &phase_sets)
        }
    };
    let mut extents = Vec::new();
    let mut offset = 0_u64;
    let mut seen: BTreeSet<ShardId> = BTreeSet::new();
    let mut placed_bytes = 0_u64;
    for shard in &order {
        let bytes = table[shard].bytes;
        let replica = u32::from(seen.contains(shard));
        seen.insert(*shard);
        extents.push(PlacedExtent {
            shard: *shard,
            bytes,
            offset,
            replica,
        });
        offset = offset.saturating_add(bytes);
        placed_bytes = placed_bytes.saturating_add(bytes);
    }
    let unique_bytes: u64 = table
        .values()
        .map(|shard| shard.bytes)
        .fold(0, u64::saturating_add);
    let replicated_bytes = placed_bytes.saturating_sub(unique_bytes);
    let amplification = if unique_bytes == 0 {
        0.0
    } else {
        placed_bytes as f64 / unique_bytes as f64
    };
    let placement = Placement {
        kind,
        params,
        extents,
        unique_shards: table.len() as u64,
        unique_bytes,
        placed_bytes,
        replicated_bytes,
        amplification,
    };
    placement.validate()?;
    Ok(placement)
}

fn frequency_order(
    frequency: &BTreeMap<ShardId, u64>,
    table: &BTreeMap<ShardId, TraceShard>,
) -> Vec<ShardId> {
    let mut ordered: Vec<ShardId> = table.keys().copied().collect();
    ordered.sort_unstable_by(|left, right| {
        frequency
            .get(right)
            .copied()
            .unwrap_or(0)
            .cmp(&frequency.get(left).copied().unwrap_or(0))
            .then(table[left].offset.cmp(&table[right].offset))
            .then(left.cmp(right))
    });
    ordered
}

fn coalesced_order(
    frequency: &BTreeMap<ShardId, u64>,
    table: &BTreeMap<ShardId, TraceShard>,
    weights: &BTreeMap<(ShardId, ShardId), f64>,
    block_shards: usize,
) -> Vec<ShardId> {
    let mut unplaced: BTreeSet<ShardId> = table.keys().copied().collect();
    let mut order = Vec::with_capacity(unplaced.len());
    while !unplaced.is_empty() {
        // Seed: the most frequent unplaced shard, deterministically.
        let seed = match unplaced.iter().max_by(|left, right| {
            frequency
                .get(*left)
                .copied()
                .unwrap_or(0)
                .cmp(&frequency.get(*right).copied().unwrap_or(0))
                .then(right.cmp(left))
        }) {
            Some(seed) => *seed,
            None => break,
        };
        unplaced.remove(&seed);
        let mut block = vec![seed];
        while block.len() < block_shards && !unplaced.is_empty() {
            let mut best: Option<(f64, u64, u64, ShardId)> = None;
            for candidate in &unplaced {
                let mut score = 0.0_f64;
                for member in &block {
                    let (low, high) = canonical(candidate, member);
                    score += weights.get(&(low, high)).copied().unwrap_or(0.0);
                }
                let key = (
                    score,
                    frequency.get(candidate).copied().unwrap_or(0),
                    table[candidate].offset,
                    *candidate,
                );
                let replace = match &best {
                    None => true,
                    Some(current) => {
                        key.0 > current.0
                            || (key.0 == current.0 && key.1 > current.1)
                            || (key.0 == current.0 && key.1 == current.1 && key.2 < current.2)
                            || (key.0 == current.0
                                && key.1 == current.1
                                && key.2 == current.2
                                && key.3 < current.3)
                    }
                };
                if replace {
                    best = Some(key);
                }
            }
            match best {
                Some((_, _, _, chosen)) => {
                    unplaced.remove(&chosen);
                    block.push(chosen);
                }
                None => break,
            }
        }
        order.extend(block);
    }
    order
}

fn replicated_order(
    frequency: &BTreeMap<ShardId, u64>,
    table: &BTreeMap<ShardId, TraceShard>,
    weights: &BTreeMap<(ShardId, ShardId), f64>,
    params: PlacementParams,
    phase_sets: &[(String, BTreeSet<ShardId>)],
) -> Vec<ShardId> {
    let mut hot: Vec<ShardId> = table.keys().copied().collect();
    hot.sort_unstable_by(|left, right| {
        frequency
            .get(right)
            .copied()
            .unwrap_or(0)
            .cmp(&frequency.get(left).copied().unwrap_or(0))
            .then(left.cmp(right))
    });
    hot.truncate(params.hot_shards);
    let hot_set: BTreeSet<ShardId> = hot.iter().copied().collect();
    // Contiguous hot fragment per phase, most frequent phase first.
    let mut phases: Vec<(&String, &BTreeSet<ShardId>)> =
        phase_sets.iter().map(|p| (&p.0, &p.1)).collect();
    phases.sort_unstable_by(|left, right| {
        let score = |set: &BTreeSet<ShardId>| -> u64 {
            set.iter()
                .filter(|shard| hot_set.contains(shard))
                .map(|shard| frequency.get(shard).copied().unwrap_or(0))
                .sum()
        };
        score(right.1).cmp(&score(left.1)).then(left.0.cmp(right.0))
    });
    let mut order: Vec<ShardId> = Vec::new();
    let mut emitted: BTreeSet<ShardId> = BTreeSet::new();
    for (_, set) in phases {
        let mut fragment: Vec<ShardId> = set
            .iter()
            .filter(|shard| hot_set.contains(*shard))
            .copied()
            .collect();
        if fragment.is_empty() {
            continue;
        }
        fragment.sort_unstable_by(|left, right| {
            frequency
                .get(right)
                .copied()
                .unwrap_or(0)
                .cmp(&frequency.get(left).copied().unwrap_or(0))
                .then(left.cmp(right))
        });
        fragment.truncate(params.block_shards);
        for shard in fragment {
            order.push(shard);
            emitted.insert(shard);
        }
    }
    // Every remaining shard once, in co-access order over the rest.
    let remaining = coalesced_order(frequency, table, weights, params.block_shards);
    for shard in remaining {
        if emitted.insert(shard) {
            order.push(shard);
        }
    }
    // Any shard not touched yet (cannot happen, but checked by validate()).
    for shard in table.keys() {
        if emitted.insert(*shard) {
            order.push(*shard);
        }
    }
    order
}

impl Placement {
    /// Validate the placement invariants.
    ///
    /// # Errors
    ///
    /// Rejects a missing shard, a duplicate copy of a shard when the layout is
    /// not replicated, a zero-byte extent, and non-monotone offsets.
    pub fn validate(&self) -> Result<()> {
        let mut seen: BTreeMap<ShardId, u32> = BTreeMap::new();
        let mut expected_offset = 0_u64;
        for extent in &self.extents {
            if extent.bytes == 0 {
                return Err(SimError::invalid("placement", "extent has zero bytes"));
            }
            if extent.offset != expected_offset {
                return Err(SimError::invalid(
                    "placement",
                    "extents are not packed back-to-back",
                ));
            }
            expected_offset = expected_offset.saturating_add(extent.bytes);
            let count = seen.entry(extent.shard).or_insert(0);
            *count += 1;
            if *count > 1 && self.kind != PlacementKind::ReplicatedHot {
                return Err(SimError::invalid(
                    "placement",
                    "a non-replicating layout repeats a shard",
                ));
            }
        }
        if seen.len() as u64 != self.unique_shards {
            return Err(SimError::invalid(
                "placement",
                format!(
                    "layout places {} distinct shards, table has {}",
                    seen.len(),
                    self.unique_shards
                ),
            ));
        }
        if expected_offset != self.placed_bytes {
            return Err(SimError::invalid(
                "placement",
                "placed byte accounting does not match the extents",
            ));
        }
        Ok(())
    }

    /// Physical position of each shard, first copy first.
    #[must_use]
    pub fn positions(&self) -> BTreeMap<ShardId, Vec<usize>> {
        let mut map: BTreeMap<ShardId, Vec<usize>> = BTreeMap::new();
        for (index, extent) in self.extents.iter().enumerate() {
            map.entry(extent.shard).or_default().push(index);
        }
        map
    }

    /// Number of physically adjacent pairs in the layout.
    #[must_use]
    pub fn adjacent_pairs(&self) -> usize {
        self.extents.len().saturating_sub(1)
    }

    /// Extent for a physical position.
    #[must_use]
    pub fn extent(&self, index: usize) -> Option<&PlacedExtent> {
        self.extents.get(index)
    }
}

/// A surrogate-objective report for one placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveReport {
    /// Layout family.
    pub kind: PlacementKind,
    /// Adjacent co-accessed pairs.
    pub adjacent_coaccessed_pairs: u64,
    /// Sum of `marginal_ns * co_access` over adjacent co-accessed pairs.
    pub estimated_stall_saved_ns: f64,
    /// Unique bytes.
    pub unique_bytes: u64,
    /// Placed bytes.
    pub placed_bytes: u64,
    /// Replicated bytes, the eviction-cost proxy.
    pub replicated_bytes: u64,
    /// `-saved + lambda_bytes * placed_bytes + lambda_eviction * replicated`.
    pub objective: f64,
    /// Weights for the two penalties.
    pub lambda_bytes: f64,
    /// Eviction penalty.
    pub lambda_eviction: f64,
    /// Statement of what this number is.
    pub interpretation: String,
}

/// Score a placement with the surrogate objective.
#[must_use]
pub fn objective(
    placement: &Placement,
    weights: &CoAccessWeights,
    lambda_bytes: f64,
    lambda_eviction: f64,
) -> ObjectiveReport {
    let table = weights.table();
    let mut saved = 0.0_f64;
    let mut pairs = 0_u64;
    for window in placement.extents.windows(2) {
        let [left, right] = window else {
            continue;
        };
        let (low, high) = canonical(&left.shard, &right.shard);
        if let Some(edge) = weights
            .edges
            .iter()
            .find(|edge| edge.a == low && edge.b == high)
        {
            pairs += 1;
            saved += edge.marginal_ns * edge.co_access as f64;
        } else if table.contains_key(&(low, high)) {
            pairs += 1;
        }
    }
    let objective = -saved
        + lambda_bytes * placement.placed_bytes as f64
        + lambda_eviction * placement.replicated_bytes as f64;
    ObjectiveReport {
        kind: placement.kind,
        adjacent_coaccessed_pairs: pairs,
        estimated_stall_saved_ns: saved,
        unique_bytes: placement.unique_bytes,
        placed_bytes: placement.placed_bytes,
        replicated_bytes: placement.replicated_bytes,
        objective,
        lambda_bytes,
        lambda_eviction,
        interpretation: "surrogate ranking score: -estimated stall saved + \
                        byte penalty + eviction proxy. It proposes layouts; the \
                        simulator decides between them."
            .to_owned(),
    }
}

/// Schema tag for a placement report written next to a simulation.
pub const PLACEMENT_REPORT_SCHEMA: &str = PLACEMENT_SCHEMA;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::calibration::SizePoint;
    use crate::trace::{
        TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, TraceBundle, TraceManifest, TraceProvenance,
        TraceRecord, TraceSource,
    };
    use selvedge_core::{SelectionMode, Sha256Digest};

    fn shard(tag: u8, offset: u64) -> TraceShard {
        TraceShard {
            id: ShardId::from_digest(Sha256Digest::of(&[tag])),
            bytes: 1000,
            offset,
        }
    }

    fn id(tag: u8) -> ShardId {
        ShardId::from_digest(Sha256Digest::of(&[tag]))
    }

    fn trace(records: Vec<Vec<u8>>) -> Trace {
        let mut used: Vec<u8> = records.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        let shards: Vec<TraceShard> = used
            .iter()
            .enumerate()
            .map(|(index, tag)| shard(*tag, index as u64 * 1000))
            .collect();
        let built: Vec<TraceRecord> = records
            .iter()
            .enumerate()
            .map(|(index, tags)| TraceRecord {
                schema: TRACE_SCHEMA.to_owned(),
                request_id: "r".to_owned(),
                phase_id: if index < 3 {
                    "p0".to_owned()
                } else {
                    "p1".to_owned()
                },
                token_index: index as u64,
                layer: 0,
                selected: {
                    let mut ids: Vec<ShardId> = tags.iter().map(|tag| id(*tag)).collect();
                    ids.sort_unstable();
                    ids
                },
                selector_id: "unit".to_owned(),
                mode: SelectionMode::Approximate,
                task: None,
                timestamps: None,
                observed: None,
            })
            .collect();
        let manifest = TraceManifest {
            schema: TRACE_MANIFEST_SCHEMA.to_owned(),
            trace_id: "unit".to_owned(),
            source: TraceSource::Synthetic,
            provenance: TraceProvenance {
                origin: "unit".to_owned(),
                receipt: None,
                receipt_sha256: None,
                revision: "rev".to_owned(),
                stamp: "stamp".to_owned(),
                notes: Vec::new(),
            },
            bundle: TraceBundle {
                bundle_id: "b".to_owned(),
                manifest_sha256: Sha256Digest::of(b"m"),
            },
            mode: SelectionMode::Approximate,
            exact_basis: None,
            shards,
            records: 0,
            records_sha256: Sha256Digest::of(b""),
            generator: None,
            created_utc: "stamp".to_owned(),
        };
        Trace::new(manifest, built).unwrap()
    }

    fn law() -> SizeLaw {
        SizeLaw::fit(
            vec![
                SizePoint {
                    bytes: 1000,
                    p50_ns: 200_000,
                },
                SizePoint {
                    bytes: 8000,
                    p50_ns: 800_000,
                },
            ],
            "test",
        )
        .unwrap()
    }

    #[test]
    fn sequential_is_a_permutation_of_extents() {
        let t = trace(vec![vec![1, 2], vec![2, 3], vec![1, 3]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 1).unwrap();
        let placement = build(
            &t,
            &w,
            PlacementKind::Sequential,
            PlacementParams::default(),
        )
        .unwrap();
        assert_eq!(placement.extents.len(), 3);
        assert_eq!(placement.replicated_bytes, 0);
        let mut tags: Vec<String> = placement
            .extents
            .iter()
            .map(|extent| extent.shard.short())
            .collect();
        let unique = tags.len();
        tags.sort();
        tags.dedup();
        assert_eq!(tags.len(), unique);
        assert!((placement.amplification - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn frequency_and_coaccess_layouts_are_permutations() {
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![3]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 2).unwrap();
        for kind in [
            PlacementKind::FrequencySorted,
            PlacementKind::CoAccessBlocked,
        ] {
            let placement = build(&t, &w, kind, PlacementParams::default()).unwrap();
            assert_eq!(placement.extents.len(), 3);
            assert_eq!(placement.replicated_bytes, 0);
            placement.validate().unwrap();
        }
    }

    #[test]
    fn coaccess_blocks_place_coaccessed_pairs_adjacent() {
        // 1 and 2 are always needed together; the block layout must put them
        // next to each other.
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![1, 2], vec![3]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 3).unwrap();
        let placement = build(
            &t,
            &w,
            PlacementKind::CoAccessBlocked,
            PlacementParams {
                block_shards: 2,
                ..PlacementParams::default()
            },
        )
        .unwrap();
        let positions = placement.positions();
        let one = positions[&id(1)][0];
        let two = positions[&id(2)][0];
        assert_eq!(one.abs_diff(two), 1);
    }

    #[test]
    fn replication_amplification_is_accounted() {
        // Two phases with the same hot pair: the second phase gets its own
        // contiguous copy.
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![1, 2], vec![1, 2]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 4).unwrap();
        let placement = build(
            &t,
            &w,
            PlacementKind::ReplicatedHot,
            PlacementParams {
                hot_shards: 2,
                block_shards: 2,
                ..PlacementParams::default()
            },
        )
        .unwrap();
        assert_eq!(placement.extents.len(), 4);
        assert_eq!(placement.unique_shards, 2);
        assert_eq!(placement.placed_bytes, 4000);
        assert_eq!(placement.replicated_bytes, 2000);
        assert!((placement.amplification - 2.0).abs() < 1e-12);
        placement.validate().unwrap();
    }

    #[test]
    fn weights_use_lift_and_marginal_cost() {
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![1], vec![3]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 5).unwrap();
        let edge = w
            .edges
            .iter()
            .find(|edge| edge.a == id(1) && edge.b == id(2))
            .unwrap();
        // P(1,2) = 0.5, P(1) = 0.75, P(2) = 0.5 -> lift = 0.5 / 0.375 = 4/3.
        assert!((edge.lift - 4.0 / 3.0).abs() < 1e-9);
        assert!(edge.marginal_ns > 0.0);
        assert!((edge.weight - edge.lift * edge.marginal_ns).abs() < 1e-9);
        assert!(w.checked_edges > 0);
        assert!(w.weight(&id(1), &id(2)) > 0.0);
        assert_eq!(w.weight(&id(1), &id(3)), 0.0);
    }

    #[test]
    fn objective_prices_replication() {
        // On a tiny fixture the stall saving of a second copy dominates any
        // small byte penalty, so the eviction price has to be explicit. The
        // surrogate is a ranking aid; the simulator decides.
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![1, 2], vec![1, 2]]);
        let w = weights(&t, Some(&law()), PlacementParams::default(), 6).unwrap();
        let plain = build(
            &t,
            &w,
            PlacementKind::CoAccessBlocked,
            PlacementParams::default(),
        )
        .unwrap();
        let replicated = build(
            &t,
            &w,
            PlacementKind::ReplicatedHot,
            PlacementParams {
                hot_shards: 2,
                block_shards: 2,
                ..PlacementParams::default()
            },
        )
        .unwrap();
        let plain_report = objective(&plain, &w, 0.0, 1000.0);
        let replicated_report = objective(&replicated, &w, 0.0, 1000.0);
        assert!(replicated_report.objective > plain_report.objective);
        assert_eq!(replicated_report.replicated_bytes, 2000);
    }

    #[test]
    fn unknown_placement_name_is_rejected() {
        assert!(PlacementKind::parse("nope").is_err());
        assert_eq!(
            PlacementKind::parse("co-access-blocked").unwrap(),
            PlacementKind::CoAccessBlocked
        );
    }

    #[test]
    fn weights_are_deterministic() {
        let t = trace(vec![vec![1, 2], vec![1, 2], vec![3]]);
        let a = weights(&t, Some(&law()), PlacementParams::default(), 9).unwrap();
        let b = weights(&t, Some(&law()), PlacementParams::default(), 9).unwrap();
        assert_eq!(a, b);
    }
}
