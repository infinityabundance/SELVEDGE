//! The bundle's execution plan: which shards are the model's native MLP
//! computation, and how a group is streamed out of the store.
//!
//! The plan is derived from the **validated manifest** and nothing else. That
//! derivation is the declared native computation set
//! ([`selvedge_core::NativeComputationSet`]) that exact mode is proven against,
//! and the declaration is a documented trust input: the runtime can prove that
//! it executed the set the manifest declares, not that the manifest describes
//! the true network. See `docs/RUNTIME.md`.
//!
//! The plan refuses to build unless the manifest declares a *complete* SwiGLU
//! partition: every layer present, three roles per group, and the group channel
//! lists partitioning `0..intermediate_size` exactly. A partial bundle cannot be
//! executed in exact mode by construction.

use std::collections::{BTreeMap, BTreeSet};

use selvedge_core::{ChannelSet, DType, NativeComputationSet, ShardId, StorageOrder, TensorRole};
use selvedge_format::{Manifest, ShardRecord};
use selvedge_store::{ExecutionScope, ReadReport, ShardStore};

use crate::config::{Qwen3Config, StreamConfig};
use crate::error::{Result, RuntimeError};

/// The transform identifier a complete f32 SwiGLU partition carries.
pub const COMPLETE_COMPATIBILITY_ID: &str = "swiglu-f32-complete-v1";

/// One shard of a group, with the numbers accounting needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardFragment {
    /// Content-derived shard id.
    pub id: ShardId,
    /// Learned elements the shard carries.
    pub elements: u64,
    /// Bytes the shard occupies.
    pub bytes: u64,
}

impl ShardFragment {
    /// Matrix-vector FLOPs for this shard: `2 * N * K` with `M = 1`.
    #[must_use]
    pub const fn flops(&self) -> u64 {
        2 * self.elements
    }
}

/// One contiguous group of intermediate channels for one layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    /// Zero-based group index, ascending within the layer.
    pub index: usize,
    /// Intermediate channels, ascending.
    pub channels: Vec<u64>,
    /// `gate_proj` rows for those channels.
    pub gate: ShardFragment,
    /// `up_proj` rows for those channels.
    pub up: ShardFragment,
    /// `down_proj` transposed block: physical `[H, |G|]`.
    pub down: ShardFragment,
}

impl GroupPlan {
    /// Number of channels in the group.
    #[must_use]
    pub fn width(&self) -> u64 {
        self.channels.len() as u64
    }

    /// Shard ids of the group in `(gate, up, down)` order.
    #[must_use]
    pub fn shards(&self) -> [ShardId; 3] {
        [self.gate.id, self.up.id, self.down.id]
    }

    /// Learned elements the group contributes to a token's active set.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.gate.elements + self.up.elements + self.down.elements
    }

    /// FLOPs of executing the group for one token: three matrix-vector
    /// products, `2 * N * K` each with `M = 1`.
    #[must_use]
    pub fn flops(&self) -> u64 {
        self.gate.flops() + self.up.flops() + self.down.flops()
    }

    /// Bytes the group moves when it is fetched.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.gate.bytes + self.up.bytes + self.down.bytes
    }
}

/// One layer's groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    /// Layer index.
    pub layer: u32,
    /// Groups, ascending by first channel.
    pub groups: Vec<GroupPlan>,
}

impl LayerPlan {
    /// Total channels covered, which equals `intermediate_size` for a complete
    /// partition.
    #[must_use]
    pub fn channels(&self) -> u64 {
        self.groups.iter().map(GroupPlan::width).sum()
    }
}

/// The declared native computation and its executable layout.
#[derive(Clone, Debug)]
pub struct BundlePlan {
    /// Bundle id from the manifest.
    pub bundle_id: String,
    /// Digest of the manifest the plan was derived from.
    pub manifest_sha256: selvedge_core::Sha256Digest,
    /// Layers, ascending.
    pub layers: Vec<LayerPlan>,
    /// The declaration exact mode is proven against.
    pub native: NativeComputationSet,
    /// Hidden width the shards declare.
    pub hidden: u64,
    /// Intermediate width the groups partition.
    pub intermediate: u64,
    /// Shards in the plan.
    pub shard_count: usize,
    /// Payload bytes the plan's shards cover.
    pub payload_bytes: u64,
}

/// Alias kept for readability in receipts.
pub type NativeDeclaration = NativeComputationSet;

impl BundlePlan {
    /// Derive the plan from a validated manifest and the architecture.
    ///
    /// # Errors
    ///
    /// Rejects a manifest that is not a complete unquantized f32 SwiGLU
    /// partition of exactly `intermediate_size` channels for exactly
    /// `num_hidden_layers` layers, a shape that disagrees with the channel list,
    /// a non-transposed down block, a duplicate group, and a bundle id that is
    /// empty.
    pub fn from_manifest(
        manifest: &Manifest,
        manifest_sha256: selvedge_core::Sha256Digest,
        config: &Qwen3Config,
    ) -> Result<Self> {
        config.validate()?;
        if manifest.bundle_id.is_empty() {
            return Err(RuntimeError::bundle("manifest has an empty bundle id"));
        }
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let layer_count = config.num_hidden_layers;

        let mut buckets: BTreeMap<(u32, String), Bucket> = BTreeMap::new();
        let mut all_ids: Vec<ShardId> = Vec::with_capacity(manifest.shards.len());
        for record in &manifest.shards {
            let role = match record.role {
                TensorRole::MlpGate => Role::Gate,
                TensorRole::MlpUp => Role::Up,
                TensorRole::MlpDown => Role::Down,
                ref other => {
                    return Err(RuntimeError::bundle(format!(
                        "shard {} has role {other}, which this runtime does not execute; a bundle must contain only SwiGLU MLP shards so the execution set can equal the declared set",
                        record.shard_id.short()
                    )));
                }
            };
            let layer = record.layer.ok_or_else(|| {
                RuntimeError::bundle(format!(
                    "shard {} has no layer index",
                    record.shard_id.short()
                ))
            })?;
            if layer >= layer_count as u32 {
                return Err(RuntimeError::bundle(format!(
                    "shard {} names layer {layer}, but the model has {layer_count}",
                    record.shard_id.short()
                )));
            }
            check_record(record, hidden)?;
            let channels = channel_list(&record.channels)?;
            let key = (layer, channels_key(&channels));
            let bucket = buckets.entry(key).or_insert_with(|| Bucket {
                layer,
                channels,
                gate: None,
                up: None,
                down: None,
            });
            let slot = match role {
                Role::Gate => &mut bucket.gate,
                Role::Up => &mut bucket.up,
                Role::Down => &mut bucket.down,
            };
            if slot.is_some() {
                return Err(RuntimeError::bundle(format!(
                    "layer {layer} declares two {role:?} shards for the same channels"
                )));
            }
            *slot = Some(ShardFragment {
                id: record.shard_id,
                elements: record.shape[0]
                    .checked_mul(record.shape[1])
                    .ok_or_else(|| RuntimeError::bundle("shard element count overflows"))?,
                bytes: record.length,
            });
            all_ids.push(record.shard_id);
        }

        let mut layers: BTreeMap<u32, Vec<GroupPlan>> = BTreeMap::new();
        for (_key, bucket) in buckets {
            let gate = bucket.gate.ok_or_else(|| {
                RuntimeError::bundle(format!(
                    "layer {} group {:?} is missing a gate shard",
                    bucket.layer, bucket.channels
                ))
            })?;
            let up = bucket.up.ok_or_else(|| {
                RuntimeError::bundle(format!(
                    "layer {} group {:?} is missing an up shard",
                    bucket.layer, bucket.channels
                ))
            })?;
            let down = bucket.down.ok_or_else(|| {
                RuntimeError::bundle(format!(
                    "layer {} group {:?} is missing a down shard",
                    bucket.layer, bucket.channels
                ))
            })?;
            let entry = layers.entry(bucket.layer).or_default();
            entry.push(GroupPlan {
                index: entry.len(),
                channels: bucket.channels,
                gate,
                up,
                down,
            });
        }

        if layers.len() != layer_count as usize {
            let missing: Vec<u32> = (0..layer_count as u32)
                .filter(|layer| !layers.contains_key(layer))
                .collect();
            return Err(RuntimeError::bundle(format!(
                "bundle declares {} layer(s), the model has {layer_count}; missing {missing:?}",
                layers.len()
            )));
        }

        let mut planned = Vec::with_capacity(layers.len());
        let mut payload_bytes = 0_u64;
        for (layer, mut groups) in layers {
            groups.sort_by(|left, right| {
                left.channels
                    .first()
                    .cmp(&right.channels.first())
                    .then_with(|| left.channels.cmp(&right.channels))
            });
            for (index, group) in groups.iter_mut().enumerate() {
                group.index = index;
                payload_bytes = payload_bytes
                    .checked_add(group.bytes())
                    .ok_or_else(|| RuntimeError::bundle("payload bytes overflow"))?;
            }
            check_partition(layer, &groups, intermediate)?;
            planned.push(LayerPlan { layer, groups });
        }

        all_ids.sort_unstable();
        let native = NativeComputationSet::declare(manifest.bundle_id.clone(), all_ids.clone())?;
        Ok(Self {
            bundle_id: manifest.bundle_id.clone(),
            manifest_sha256,
            layers: planned,
            native,
            hidden,
            intermediate,
            shard_count: all_ids.len(),
            payload_bytes,
        })
    }

    /// Number of groups in a layer.
    ///
    /// # Errors
    ///
    /// Rejects a layer index the plan does not contain.
    pub fn groups_for(&self, layer: usize) -> Result<&[GroupPlan]> {
        self.layers
            .get(layer)
            .map(|plan| plan.groups.as_slice())
            .ok_or_else(|| RuntimeError::bundle(format!("layer {layer} is not in the plan")))
    }

    /// Every shard id in the plan, ascending.
    #[must_use]
    pub fn shard_ids(&self) -> Vec<ShardId> {
        self.native.shards().to_vec()
    }

    /// The execution scope for the whole native set.
    ///
    /// # Errors
    ///
    /// Propagates a duplicate or empty set.
    pub fn native_scope(&self) -> Result<ExecutionScope> {
        Ok(ExecutionScope::exact(self.shard_ids())?)
    }

    /// Element and FLOP totals for the complete set.
    #[must_use]
    pub fn total_elements(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|layer| layer.groups.iter())
            .map(GroupPlan::elements)
            .sum()
    }
}

/// Why a shard could not be part of an executable plan.
fn check_record(record: &ShardRecord, hidden: u64) -> Result<()> {
    if record.dtype != DType::F32 || record.quant.is_some() {
        return Err(RuntimeError::bundle(format!(
            "shard {} is {} with quantization {:?}; only unquantized f32 is executable",
            record.shard_id.short(),
            record.dtype,
            record.quant
        )));
    }
    if record.channel_axis != 0 {
        return Err(RuntimeError::bundle(format!(
            "shard {} declares channel axis {}, expected 0",
            record.shard_id.short(),
            record.channel_axis
        )));
    }
    if record.compatibility_id != COMPLETE_COMPATIBILITY_ID {
        return Err(RuntimeError::bundle(format!(
            "shard {} has compatibility id {:?}, expected {:?}",
            record.shard_id.short(),
            record.compatibility_id,
            COMPLETE_COMPATIBILITY_ID
        )));
    }
    if record.shape.len() != 2 {
        return Err(RuntimeError::bundle(format!(
            "shard {} has shape {:?}, expected rank 2",
            record.shard_id.short(),
            record.shape
        )));
    }
    if record.shape[1] != hidden {
        return Err(RuntimeError::bundle(format!(
            "shard {} has hidden width {}, expected {hidden}",
            record.shard_id.short(),
            record.shape[1]
        )));
    }
    if record.shape[0] != record.channels.count() {
        return Err(RuntimeError::bundle(format!(
            "shard {} has {} rows but {} channels",
            record.shard_id.short(),
            record.shape[0],
            record.channels.count()
        )));
    }
    match record.role {
        TensorRole::MlpDown => {
            if record.storage_order != (StorageOrder::Transposed { axes: vec![1, 0] }) {
                return Err(RuntimeError::bundle(format!(
                    "down shard {} is not stored transposed; physical [H, |G|] is what the runtime executes",
                    record.shard_id.short()
                )));
            }
        }
        _ => {
            if record.storage_order != StorageOrder::RowMajor {
                return Err(RuntimeError::bundle(format!(
                    "shard {} is not row-major",
                    record.shard_id.short()
                )));
            }
        }
    }
    Ok(())
}

fn check_partition(layer: u32, groups: &[GroupPlan], intermediate: u64) -> Result<()> {
    if groups.is_empty() {
        return Err(RuntimeError::bundle(format!("layer {layer} has no groups")));
    }
    let mut covered = 0_u64;
    let mut expected = 0_u64;
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for group in groups {
        for channel in &group.channels {
            if !seen.insert(*channel) {
                return Err(RuntimeError::bundle(format!(
                    "layer {layer} assigns channel {channel} to two groups"
                )));
            }
            if *channel != expected {
                return Err(RuntimeError::bundle(format!(
                    "layer {layer} channels are not the partition 0..{intermediate}; found {channel} where {expected} was due"
                )));
            }
            expected += 1;
        }
        covered = covered.saturating_add(group.width());
    }
    if covered != intermediate {
        return Err(RuntimeError::bundle(format!(
            "layer {layer} covers {covered} intermediate channels, the model has {intermediate}"
        )));
    }
    Ok(())
}

fn channel_list(channels: &ChannelSet) -> Result<Vec<u64>> {
    match channels {
        ChannelSet::Range { start, end } => Ok((*start..*end).collect()),
        ChannelSet::List { indices } => Ok(indices.clone()),
    }
}

fn channels_key(channels: &[u64]) -> String {
    let mut key = String::with_capacity(channels.len() * 4);
    for channel in channels {
        key.push_str(&channel.to_string());
        key.push(',');
    }
    key
}

struct Bucket {
    layer: u32,
    channels: Vec<u64>,
    gate: Option<ShardFragment>,
    up: Option<ShardFragment>,
    down: Option<ShardFragment>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Gate,
    Up,
    Down,
}

/// The three matrices of one group, decoded to f32 in execution layout.
///
/// `gate` and `up` are `[|G|, H]` row-major. `down` is the bundle's physical
/// `[H, |G|]` block, which is exactly the layout a `hidden x width` matrix-vector
/// product needs, so no transpose is ever materialised.
#[derive(Clone, Debug, Default)]
pub struct GroupTensors {
    /// Gate rows.
    pub gate: Vec<f32>,
    /// Up rows.
    pub up: Vec<f32>,
    /// Down block, physical `[H, |G|]`.
    pub down: Vec<f32>,
    /// `|G|`.
    pub width: usize,
    /// `H`.
    pub hidden: usize,
}

impl GroupTensors {
    fn record_shape(&mut self, width: usize, hidden: usize) -> Result<()> {
        width
            .checked_mul(hidden)
            .ok_or_else(|| RuntimeError::memory("group element count overflows"))?;
        self.width = width;
        self.hidden = hidden;
        Ok(())
    }
}

/// Streams group shards out of a [`ShardStore`] inside a fixed scope.
///
/// The reader owns one reusable byte buffer and three reusable f32 buffers, so
/// per-group memory is bounded by the largest shard in the plan. Every read goes
/// through [`ShardStore::read_required`], whose counters are the measured
/// evidence for the accounting.
#[derive(Debug)]
pub struct GroupReader {
    scope: ExecutionScope,
    buffer: Vec<u8>,
    tensors: GroupTensors,
    executed: BTreeSet<ShardId>,
    read_calls: u64,
    bytes_used: u64,
    bytes_device: u64,
    hits: u64,
    injected_delay_ns: u64,
    last_sources: [bool; 3],
    source_slot: usize,
}

impl GroupReader {
    /// Build a reader for one scope.
    ///
    /// # Errors
    ///
    /// Rejects a stream configuration the store refuses, and a scope larger than
    /// `usize`.
    pub fn new(scope: ExecutionScope, stream: &StreamConfig) -> Result<Self> {
        let buffer = Vec::new();
        Ok(Self {
            scope,
            buffer,
            tensors: GroupTensors::default(),
            executed: BTreeSet::new(),
            read_calls: 0,
            bytes_used: 0,
            bytes_device: 0,
            hits: 0,
            injected_delay_ns: stream.injected_read_delay_ns,
            last_sources: [false; 3],
            source_slot: 0,
        })
    }

    /// Replace the scope. Used by approximate mode, whose selection is only
    /// known once the layer's hidden state exists.
    ///
    /// The replacement is a narrowing: every read is still refused outside it.
    pub fn replace_scope(&mut self, scope: ExecutionScope) {
        self.scope = scope;
    }

    /// Which of the last group's three shards came from the backing tier.
    ///
    /// `[gate, up, down]`; `true` means the bytes were read from the device
    /// rather than served by the cache. Used to attribute bytes to a tier in the
    /// ledger.
    #[must_use]
    pub const fn last_group_sources(&self) -> [bool; 3] {
        self.last_sources
    }

    /// The scope every read is confined to.
    #[must_use]
    pub fn scope(&self) -> &ExecutionScope {
        &self.scope
    }

    /// Shards actually read so far, ascending and deduplicated.
    #[must_use]
    pub fn executed(&self) -> &BTreeSet<ShardId> {
        &self.executed
    }

    /// Required read calls issued.
    #[must_use]
    pub const fn read_calls(&self) -> u64 {
        self.read_calls
    }

    /// Bytes delivered for use.
    #[must_use]
    pub const fn bytes_used(&self) -> u64 {
        self.bytes_used
    }

    /// Bytes that came from the backing tier.
    #[must_use]
    pub const fn bytes_device(&self) -> u64 {
        self.bytes_device
    }

    /// Reads served from the cache.
    #[must_use]
    pub const fn cache_hits(&self) -> u64 {
        self.hits
    }

    /// Reset the per-token counters, keeping the scope and buffers.
    pub fn reset_token(&mut self) {
        self.executed.clear();
        self.read_calls = 0;
        self.bytes_used = 0;
        self.bytes_device = 0;
        self.hits = 0;
    }

    /// Read one shard into the internal buffer and return its report.
    fn read_shard(&mut self, store: &ShardStore, fragment: &ShardFragment) -> Result<ReadReport> {
        let length = usize::try_from(fragment.bytes)
            .map_err(|_| RuntimeError::memory("shard byte length does not fit usize"))?;
        self.buffer.clear();
        self.buffer
            .try_reserve_exact(length)
            .map_err(|_| RuntimeError::memory(format!("cannot reserve {length} bytes")))?;
        self.buffer.resize(length, 0);
        let report = store.read_required(&self.scope, &fragment.id, &mut self.buffer)?;
        if report.bytes_used != fragment.bytes {
            return Err(RuntimeError::model(format!(
                "store delivered {} bytes for shard {}, expected {}",
                report.bytes_used,
                fragment.id.short(),
                fragment.bytes
            )));
        }
        self.read_calls = self.read_calls.saturating_add(1);
        self.bytes_used = self.bytes_used.saturating_add(report.bytes_used);
        self.bytes_device = self.bytes_device.saturating_add(report.device_bytes);
        if report.hit {
            self.hits = self.hits.saturating_add(1);
        }
        if self.source_slot < self.last_sources.len() {
            self.last_sources[self.source_slot] = !report.hit;
            self.source_slot += 1;
        }
        if self.injected_delay_ns > 0 && !report.hit {
            std::thread::sleep(core::time::Duration::from_nanos(self.injected_delay_ns));
        }
        Ok(report)
    }

    /// Read one shard for execution, marking it executed.
    fn read_executed_shard(&mut self, store: &ShardStore, fragment: &ShardFragment) -> Result<()> {
        self.read_shard(store, fragment)?;
        self.executed.insert(fragment.id);
        Ok(())
    }

    /// Read every shard of `shards` into the cache, in the given order.
    ///
    /// Used as an explicit prefetch. The bytes are verified and admitted to the
    /// cache exactly as a required read; the next required read of the same
    /// shard is then a hit. Prefetch never widens the scope. The return value
    /// names, for every shard, whether its bytes came from the backing tier, so
    /// the caller attributes them to a tier exactly once.
    ///
    /// # Errors
    ///
    /// Propagates any read failure. A prefetch that fails is a failed run: an
    /// advisory prefetch is allowed to be refused, but a prefetch this runtime
    /// performs is a read it has accounted for.
    pub fn prefetch(
        &mut self,
        store: &ShardStore,
        shards: &[ShardFragment],
    ) -> Result<Vec<(ShardFragment, bool)>> {
        let mut out = Vec::with_capacity(shards.len());
        for fragment in shards {
            let report = self.read_shard(store, fragment)?;
            out.push((*fragment, !report.hit));
        }
        Ok(out)
    }

    /// Read one group and decode it into execution layout.
    ///
    /// # Errors
    ///
    /// Propagates read failures and rejects a shard whose byte length disagrees
    /// with the plan.
    pub fn read_group(&mut self, store: &ShardStore, group: &GroupPlan, hidden: u64) -> Result<()> {
        let width = usize::try_from(group.width())
            .map_err(|_| RuntimeError::memory("group width does not fit usize"))?;
        let hidden_usize = usize::try_from(hidden)
            .map_err(|_| RuntimeError::memory("hidden width does not fit usize"))?;
        self.tensors.record_shape(width, hidden_usize)?;

        self.last_sources = [false; 3];
        self.source_slot = 0;
        let gate = group.gate;
        let up = group.up;
        let down = group.down;
        self.read_executed_shard(store, &gate)?;
        decode_f32_in_place(&self.buffer, &mut self.tensors.gate)?;
        self.read_executed_shard(store, &up)?;
        decode_f32_in_place(&self.buffer, &mut self.tensors.up)?;
        self.read_executed_shard(store, &down)?;
        decode_f32_in_place(&self.buffer, &mut self.tensors.down)?;

        if self.tensors.gate.len() != width * hidden_usize
            || self.tensors.up.len() != width * hidden_usize
            || self.tensors.down.len() != width * hidden_usize
        {
            return Err(RuntimeError::memory(format!(
                "group {} decoded buffers do not match |G|={width} and H={hidden}",
                group.index
            )));
        }
        Ok(())
    }

    /// Decoded buffers of the last group read.
    #[must_use]
    pub fn tensors(&self) -> &GroupTensors {
        &self.tensors
    }
}

/// Decode little-endian f32 bytes into `target`, appending nothing.
fn decode_f32_in_place(bytes: &[u8], target: &mut Vec<f32>) -> Result<()> {
    if !bytes.len().is_multiple_of(4) {
        return Err(RuntimeError::model(format!(
            "shard byte length {} is not a multiple of four",
            bytes.len()
        )));
    }
    let count = bytes.len() / 4;
    target.clear();
    target
        .try_reserve_exact(count)
        .map_err(|_| RuntimeError::memory(format!("cannot reserve {count} f32")))?;
    for chunk in bytes.as_chunks::<4>().0 {
        target.push(f32::from_le_bytes(*chunk));
    }
    Ok(())
}

/// Total FLOPs of executing `groups` for one token.
///
/// Each group is three matrix-vector products of `2 * |G| * H` FLOPs, so
/// `6 * |G| * H` per group. The formula is documented in `docs/RUNTIME.md`.
#[must_use]
pub fn group_flops(groups: &[GroupPlan]) -> u64 {
    groups.iter().map(GroupPlan::flops).sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn channel_keys_are_unambiguous() {
        assert_eq!(channels_key(&[1, 2, 3]), "1,2,3,");
        assert_eq!(channels_key(&[12, 3]), "12,3,");
        assert_ne!(channels_key(&[1, 23]), channels_key(&[12, 3]));
    }
}
