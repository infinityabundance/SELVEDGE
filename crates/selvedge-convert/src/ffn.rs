//! The exact SwiGLU FFN map, grouping rules and reference arithmetic.
//!
//! For an FFN with `H` hidden channels and `I` intermediate channels the
//! decomposition is
//!
//! ```text
//! FFN(x) = sum_g down[:, G_g] ( silu(gate[G_g, :] x) * (up[G_g, :] x) )
//! ```
//!
//! where `G_1..G_m` is a partition of `0..I`. The math below never contains a
//! tensor name: names and orientation live in [`FfnTensorMap`].
//!
//! Reduction order is fixed and documented: the dense path iterates
//! intermediate channels in ascending order; the grouped path iterates groups in
//! the order supplied and channels in ascending order inside each group. No
//! Kahan compensation is used, so grouped and dense sums may differ in the last
//! f32 bits even when the identity is exact.

use std::collections::BTreeMap;

use selvedge_core::{ChannelSet, DType};

use crate::error::{ConvertError, Result};
use crate::tensor_source::TensorSource;

/// Which matrix axis holds the intermediate-channel dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnOrientation {
    /// `gate`/`up` are `[intermediate, hidden]` and `down` is
    /// `[hidden, intermediate]`. This is the Qwen3/OLMoE `nn.Linear` layout.
    IntermediateFirst,
    /// `gate`/`up` are `[hidden, intermediate]` and `down` is
    /// `[intermediate, hidden]`.
    HiddenFirst,
}

impl FfnOrientation {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::IntermediateFirst => "intermediate_first",
            Self::HiddenFirst => "hidden_first",
        }
    }

    /// Parse the stable name.
    ///
    /// # Errors
    ///
    /// Rejects unknown names.
    pub fn parse(input: &str) -> Result<Self> {
        match input {
            "intermediate_first" | "intermediate-first" => Ok(Self::IntermediateFirst),
            "hidden_first" | "hidden-first" => Ok(Self::HiddenFirst),
            other => Err(ConvertError::inconsistent(format!(
                "unknown FFN orientation {other:?}"
            ))),
        }
    }

    /// Logical shape of the gate or up matrix.
    #[must_use]
    pub const fn gate_up_shape(self, intermediate: u64, hidden: u64) -> [u64; 2] {
        match self {
            Self::IntermediateFirst => [intermediate, hidden],
            Self::HiddenFirst => [hidden, intermediate],
        }
    }

    /// Logical shape of the down matrix.
    #[must_use]
    pub const fn down_shape(self, intermediate: u64, hidden: u64) -> [u64; 2] {
        match self {
            Self::IntermediateFirst => [hidden, intermediate],
            Self::HiddenFirst => [intermediate, hidden],
        }
    }

    /// Logical axis of the intermediate channels in the gate/up matrices.
    #[must_use]
    pub const fn gate_channel_axis(self) -> u32 {
        match self {
            Self::IntermediateFirst => 0,
            Self::HiddenFirst => 1,
        }
    }

    /// Logical axis of the intermediate channels in the down matrix.
    #[must_use]
    pub const fn down_channel_axis(self) -> u32 {
        match self {
            Self::IntermediateFirst => 1,
            Self::HiddenFirst => 0,
        }
    }
}

/// The typed mapping from an operation's roles to checkpoint tensor names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FfnTensorMap {
    /// Gate projection tensor key.
    pub gate: String,
    /// Up projection tensor key.
    pub up: String,
    /// Down projection tensor key.
    pub down: String,
    /// Actual checkpoint orientation.
    pub orientation: FfnOrientation,
}

impl FfnTensorMap {
    /// Construct a map.
    #[must_use]
    pub fn new(
        gate: impl Into<String>,
        up: impl Into<String>,
        down: impl Into<String>,
        orientation: FfnOrientation,
    ) -> Self {
        Self {
            gate: gate.into(),
            up: up.into(),
            down: down.into(),
            orientation,
        }
    }
}

/// A validated FFN's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfnDims {
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
}

/// Validate a map against a tensor source and return its dimensions.
///
/// # Errors
///
/// Rejects missing tensors, non-rank-2 tensors, dtypes other than f32/f16/bf16,
/// and shapes that do not match the declared orientation.
pub fn validate_ffn_map(source: &impl TensorSource, map: &FfnTensorMap) -> Result<FfnDims> {
    let gate = source.descriptor(&map.gate)?;
    let up = source.descriptor(&map.up)?;
    let down = source.descriptor(&map.down)?;
    for descriptor in [&gate, &up, &down] {
        if !matches!(descriptor.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(ConvertError::SourceDTypeUnsupported {
                tensor: descriptor.key.clone(),
                dtype: descriptor.dtype.stable_name(),
            });
        }
        if descriptor.rank() != 2 {
            return Err(ConvertError::RankMismatch {
                tensor: descriptor.key.clone(),
                rank: descriptor.rank(),
            });
        }
    }
    let dims = match map.orientation {
        FfnOrientation::IntermediateFirst => {
            let intermediate = gate.shape[0];
            let hidden = gate.shape[1];
            FfnDims {
                hidden,
                intermediate,
            }
        }
        FfnOrientation::HiddenFirst => {
            let hidden = gate.shape[0];
            let intermediate = gate.shape[1];
            FfnDims {
                hidden,
                intermediate,
            }
        }
    };
    if dims.hidden == 0 || dims.intermediate == 0 {
        return Err(ConvertError::inconsistent(format!(
            "FFN dimensions must be non-zero, got hidden={} intermediate={}",
            dims.hidden, dims.intermediate
        )));
    }
    let gate_up = map
        .orientation
        .gate_up_shape(dims.intermediate, dims.hidden);
    let down_shape = map.orientation.down_shape(dims.intermediate, dims.hidden);
    for (descriptor, expected, role) in [
        (&gate, gate_up, "gate"),
        (&up, gate_up, "up"),
        (&down, down_shape, "down"),
    ] {
        if descriptor.shape != expected {
            return Err(ConvertError::inconsistent(format!(
                "{role} tensor {:?} has shape {:?}, expected {:?} for hidden={} intermediate={}",
                descriptor.key, descriptor.shape, expected, dims.hidden, dims.intermediate
            )));
        }
    }
    Ok(dims)
}

/// One discovered FFN set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredFfn {
    /// Shared key prefix (everything before the projection suffix).
    pub prefix: String,
    /// Layer index parsed from `layers.<n>.` when present.
    pub layer: Option<u32>,
    /// Typed tensor map.
    pub map: FfnTensorMap,
}

const GATE_SUFFIX: &str = ".gate_proj.weight";
const UP_SUFFIX: &str = ".up_proj.weight";
const DOWN_SUFFIX: &str = ".down_proj.weight";

/// Discover every `gate_proj`/`up_proj`/`down_proj` FFN set in a key list.
///
/// A set is complete only when all three suffixes share a prefix. Keys that do
/// not end in one of those suffixes are ignored; a partial set is an error.
/// Results are sorted by prefix, so discovery is deterministic.
///
/// # Errors
///
/// Rejects an incomplete set, duplicate membership and an empty result.
pub fn discover_ffn_maps(keys: &[String]) -> Result<Vec<DiscoveredFfn>> {
    let mut found: BTreeMap<String, [Option<String>; 3]> = BTreeMap::new();
    for key in keys {
        if let Some(prefix) = key.strip_suffix(GATE_SUFFIX) {
            found.entry(prefix.to_owned()).or_default()[0] = Some(key.clone());
        } else if let Some(prefix) = key.strip_suffix(UP_SUFFIX) {
            found.entry(prefix.to_owned()).or_default()[1] = Some(key.clone());
        } else if let Some(prefix) = key.strip_suffix(DOWN_SUFFIX) {
            found.entry(prefix.to_owned()).or_default()[2] = Some(key.clone());
        }
    }
    if found.is_empty() {
        return Err(ConvertError::NoFfnTensors);
    }
    let mut maps = Vec::with_capacity(found.len());
    for (prefix, [gate, up, down]) in found {
        let gate = gate.ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "FFN prefix {prefix:?} has up/down tensors but no gate_proj.weight"
            ))
        })?;
        let up = up.ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "FFN prefix {prefix:?} has gate/down tensors but no up_proj.weight"
            ))
        })?;
        let down = down.ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "FFN prefix {prefix:?} has gate/up tensors but no down_proj.weight"
            ))
        })?;
        let layer = parse_layer(&prefix);
        maps.push(DiscoveredFfn {
            prefix,
            layer,
            map: FfnTensorMap::new(gate, up, down, FfnOrientation::IntermediateFirst),
        });
    }
    Ok(maps)
}

fn parse_layer(prefix: &str) -> Option<u32> {
    let marker = "layers.";
    let start = prefix.find(marker)? + marker.len();
    let rest = &prefix[start..];
    let end = rest.find('.').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// A validated channel group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// Zero-based group index in the order supplied.
    pub index: usize,
    /// Sorted intermediate channels.
    pub indices: Vec<u64>,
    /// Manifest channel set.
    pub channels: ChannelSet,
}

/// How intermediate channels are partitioned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupSpec {
    /// Contiguous groups. `size = ceil(intermediate / count)`; the final group
    /// is smaller when the split is uneven.
    ByCount(u32),
    /// Explicit channel lists in the supplied order.
    Explicit(Vec<Vec<u64>>),
}

impl GroupSpec {
    /// Parse the CLI form: a bare count, or semicolon-separated groups whose
    /// members are comma-separated indices or inclusive `a-b` ranges.
    ///
    /// # Errors
    ///
    /// Rejects empty input, zero counts, malformed ranges and empty groups.
    pub fn parse(input: &str) -> Result<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(ConvertError::grouping("group specification is empty"));
        }
        if trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
            let count: u32 = trimmed
                .parse()
                .map_err(|_| ConvertError::grouping(format!("bad group count {trimmed:?}")))?;
            if count == 0 {
                return Err(ConvertError::grouping("group count must be non-zero"));
            }
            return Ok(Self::ByCount(count));
        }
        let mut groups = Vec::new();
        for (index, part) in trimmed.split(';').enumerate() {
            let part = part.trim();
            if part.is_empty() {
                return Err(ConvertError::grouping(format!("group {index} is empty")));
            }
            let mut channels = Vec::new();
            for token in part.split(',') {
                let token = token.trim();
                if let Some((start, end)) = token.split_once('-') {
                    let start: u64 = start.trim().parse().map_err(|_| {
                        ConvertError::grouping(format!("bad range start in {token:?}"))
                    })?;
                    let end: u64 = end.trim().parse().map_err(|_| {
                        ConvertError::grouping(format!("bad range end in {token:?}"))
                    })?;
                    if start > end {
                        return Err(ConvertError::grouping(format!(
                            "range {token:?} is reversed"
                        )));
                    }
                    channels.extend(start..=end);
                } else {
                    let value: u64 = token.parse().map_err(|_| {
                        ConvertError::grouping(format!("bad channel index {token:?}"))
                    })?;
                    channels.push(value);
                }
            }
            groups.push(channels);
        }
        Ok(Self::Explicit(groups))
    }
}

/// Resolve a grouping specification against the intermediate width.
///
/// Groups are always checked for non-empty membership, sorted unique indices,
/// in-range channels and pairwise disjointness. When `require_complete` is set
/// the union must be exactly `0..intermediate`.
///
/// # Errors
///
/// Rejects every violation listed above.
pub fn resolve_groups(
    spec: &GroupSpec,
    intermediate: u64,
    require_complete: bool,
) -> Result<Vec<Group>> {
    if intermediate == 0 {
        return Err(ConvertError::grouping("intermediate width is zero"));
    }
    let raw: Vec<Vec<u64>> = match spec {
        GroupSpec::ByCount(count) => {
            let count = u64::from(*count);
            if count == 0 {
                return Err(ConvertError::grouping("group count must be non-zero"));
            }
            if count > intermediate {
                return Err(ConvertError::grouping(format!(
                    "cannot make {count} non-empty groups from {intermediate} channels"
                )));
            }
            let size = intermediate.div_ceil(count);
            let mut groups = Vec::new();
            let mut start = 0_u64;
            while start < intermediate {
                let end = (start + size).min(intermediate);
                groups.push((start..end).collect());
                start = end;
            }
            groups
        }
        GroupSpec::Explicit(groups) => {
            let mut out = Vec::with_capacity(groups.len());
            for group in groups {
                if group.is_empty() {
                    return Err(ConvertError::grouping("explicit group is empty"));
                }
                out.push(group.clone());
            }
            out
        }
    };
    if raw.is_empty() {
        return Err(ConvertError::grouping("grouping produced no groups"));
    }
    let mut seen = vec![
        false;
        usize::try_from(intermediate).map_err(|_| {
            ConvertError::grouping("intermediate width does not fit in memory")
        })?
    ];
    let mut groups = Vec::with_capacity(raw.len());
    for (index, mut channels) in raw.into_iter().enumerate() {
        let original = channels.clone();
        channels.sort_unstable();
        channels.dedup();
        if channels.len() != original.len() {
            return Err(ConvertError::grouping(format!(
                "group {index} contains duplicate channels"
            )));
        }
        for channel in &channels {
            if *channel >= intermediate {
                return Err(ConvertError::grouping(format!(
                    "group {index} contains channel {channel} outside 0..{intermediate}"
                )));
            }
            let slot = usize::try_from(*channel)
                .map_err(|_| ConvertError::grouping("channel index does not fit in memory"))?;
            if seen[slot] {
                return Err(ConvertError::grouping(format!(
                    "channel {channel} appears in more than one group"
                )));
            }
            seen[slot] = true;
        }
        let channel_set = contiguous_set(&channels)?;
        groups.push(Group {
            index,
            indices: channels,
            channels: channel_set,
        });
    }
    if require_complete && seen.iter().any(|covered| !covered) {
        return Err(ConvertError::grouping(format!(
            "groups do not cover all {intermediate} intermediate channels"
        )));
    }
    Ok(groups)
}

fn contiguous_set(channels: &[u64]) -> Result<ChannelSet> {
    let first = channels[0];
    let mut contiguous = true;
    for (offset, channel) in channels.iter().enumerate() {
        if *channel != first + offset as u64 {
            contiguous = false;
            break;
        }
    }
    if contiguous {
        ChannelSet::range(first, first + channels.len() as u64).map_err(ConvertError::from)
    } else {
        ChannelSet::list(channels.iter().copied()).map_err(ConvertError::from)
    }
}

/// A group's tensors, laid out for the grouped FFN.
///
/// `gate` and `up` are `[|G|, H]` row-major; `down_t` is the logical transpose
/// of the down slice, also `[|G|, H]`, so `down_t[j*H + h] = down[h, G[j]]`.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupMatrices {
    /// Group index.
    pub index: usize,
    /// Channels in the group, ascending.
    pub channels: Vec<u64>,
    /// Gate values `[|G|, H]`.
    pub gate: Vec<f32>,
    /// Up values `[|G|, H]`.
    pub up: Vec<f32>,
    /// Down transpose values `[|G|, H]`.
    pub down_t: Vec<f32>,
}

/// Compute the dense FFN in f32 with an independent reference path.
///
/// `gate` and `up` are `[I, H]`, `down` is `[H, I]`, and `x` is `[batch, H]`.
/// Channels are iterated in ascending order.
///
/// # Errors
///
/// Rejects inconsistent buffer lengths or shapes.
pub fn ffn_dense_f32(
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden: u64,
    intermediate: u64,
    batch: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, i, n) = lengths(hidden, intermediate, batch, x.len())?;
    if gate.len() != i * h || up.len() != i * h || down.len() != h * i {
        return Err(ConvertError::inconsistent(
            "dense FFN buffers disagree with hidden/intermediate",
        ));
    }
    let mut out = vec![0.0_f32; n * h];
    let mut act = vec![0.0_f32; i];
    for row in 0..n {
        for channel in 0..i {
            let mut gate_value = 0.0_f32;
            let mut up_value = 0.0_f32;
            for h_index in 0..h {
                let input = x[row * h + h_index];
                gate_value += gate[channel * h + h_index] * input;
                up_value += up[channel * h + h_index] * input;
            }
            act[channel] = silu_f32(gate_value) * up_value;
        }
        for h_index in 0..h {
            let mut value = 0.0_f32;
            for channel in 0..i {
                value += down[h_index * i + channel] * act[channel];
            }
            out[row * h + h_index] = value;
        }
    }
    Ok(out)
}

/// Compute the dense FFN in f64 with an independent reference path.
///
/// # Errors
///
/// Rejects inconsistent buffer lengths or shapes.
pub fn ffn_dense_f64(
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden: u64,
    intermediate: u64,
    batch: u64,
    x: &[f32],
) -> Result<Vec<f64>> {
    let (h, i, n) = lengths(hidden, intermediate, batch, x.len())?;
    if gate.len() != i * h || up.len() != i * h || down.len() != h * i {
        return Err(ConvertError::inconsistent(
            "dense FFN buffers disagree with hidden/intermediate",
        ));
    }
    let mut out = vec![0.0_f64; n * h];
    let mut act = vec![0.0_f64; i];
    for row in 0..n {
        for channel in 0..i {
            let mut gate_value = 0.0_f64;
            let mut up_value = 0.0_f64;
            for h_index in 0..h {
                let input = f64::from(x[row * h + h_index]);
                gate_value += f64::from(gate[channel * h + h_index]) * input;
                up_value += f64::from(up[channel * h + h_index]) * input;
            }
            act[channel] = silu_f64(gate_value) * up_value;
        }
        for h_index in 0..h {
            let mut value = 0.0_f64;
            for channel in 0..i {
                value += f64::from(down[h_index * i + channel]) * act[channel];
            }
            out[row * h + h_index] = value;
        }
    }
    Ok(out)
}

/// Compute the grouped FFN in f32.
///
/// # Errors
///
/// Rejects inconsistent group buffers.
pub fn ffn_grouped_f32(
    hidden: u64,
    groups: &[GroupMatrices],
    batch: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, n) = hidden_lengths(hidden, batch, x.len())?;
    let mut out = vec![0.0_f32; n * h];
    for group in groups {
        accumulate_group_f32(group, h, n, x, &mut out)?;
    }
    Ok(out)
}

/// Compute the grouped FFN in f64.
///
/// # Errors
///
/// Rejects inconsistent group buffers.
pub fn ffn_grouped_f64(
    hidden: u64,
    groups: &[GroupMatrices],
    batch: u64,
    x: &[f32],
) -> Result<Vec<f64>> {
    let (h, n) = hidden_lengths(hidden, batch, x.len())?;
    let mut out = vec![0.0_f64; n * h];
    for group in groups {
        accumulate_group_f64(group, h, n, x, &mut out)?;
    }
    Ok(out)
}

/// One group's contribution in f32.
///
/// # Errors
///
/// Rejects inconsistent group buffers.
pub fn group_contribution_f32(
    group: &GroupMatrices,
    hidden: u64,
    batch: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, n) = hidden_lengths(hidden, batch, x.len())?;
    let mut out = vec![0.0_f32; n * h];
    accumulate_group_f32(group, h, n, x, &mut out)?;
    Ok(out)
}

/// One group's contribution in f64.
///
/// # Errors
///
/// Rejects inconsistent group buffers.
pub fn group_contribution_f64(
    group: &GroupMatrices,
    hidden: u64,
    batch: u64,
    x: &[f32],
) -> Result<Vec<f64>> {
    let (h, n) = hidden_lengths(hidden, batch, x.len())?;
    let mut out = vec![0.0_f64; n * h];
    accumulate_group_f64(group, h, n, x, &mut out)?;
    Ok(out)
}

fn accumulate_group_f32(
    group: &GroupMatrices,
    hidden: usize,
    batch: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let width = group.channels.len();
    if group.gate.len() != width * hidden
        || group.up.len() != width * hidden
        || group.down_t.len() != width * hidden
    {
        return Err(ConvertError::inconsistent(format!(
            "group {} buffers do not match |G|={width} and hidden={hidden}",
            group.index
        )));
    }
    let mut act = vec![0.0_f32; width];
    for row in 0..batch {
        for (j, slot) in act.iter_mut().enumerate() {
            let mut gate_value = 0.0_f32;
            let mut up_value = 0.0_f32;
            for h_index in 0..hidden {
                let input = x[row * hidden + h_index];
                gate_value += group.gate[j * hidden + h_index] * input;
                up_value += group.up[j * hidden + h_index] * input;
            }
            *slot = silu_f32(gate_value) * up_value;
        }
        for h_index in 0..hidden {
            let mut value = 0.0_f32;
            for (j, activation) in act.iter().enumerate() {
                value += group.down_t[j * hidden + h_index] * activation;
            }
            out[row * hidden + h_index] += value;
        }
    }
    Ok(())
}

fn accumulate_group_f64(
    group: &GroupMatrices,
    hidden: usize,
    batch: usize,
    x: &[f32],
    out: &mut [f64],
) -> Result<()> {
    let width = group.channels.len();
    if group.gate.len() != width * hidden
        || group.up.len() != width * hidden
        || group.down_t.len() != width * hidden
    {
        return Err(ConvertError::inconsistent(format!(
            "group {} buffers do not match |G|={width} and hidden={hidden}",
            group.index
        )));
    }
    let mut act = vec![0.0_f64; width];
    for row in 0..batch {
        for (j, slot) in act.iter_mut().enumerate() {
            let mut gate_value = 0.0_f64;
            let mut up_value = 0.0_f64;
            for h_index in 0..hidden {
                let input = f64::from(x[row * hidden + h_index]);
                gate_value += f64::from(group.gate[j * hidden + h_index]) * input;
                up_value += f64::from(group.up[j * hidden + h_index]) * input;
            }
            *slot = silu_f64(gate_value) * up_value;
        }
        for h_index in 0..hidden {
            let mut value = 0.0_f64;
            for (j, activation) in act.iter().enumerate() {
                value += f64::from(group.down_t[j * hidden + h_index]) * activation;
            }
            out[row * hidden + h_index] += value;
        }
    }
    Ok(())
}

/// SiLU in f32.
#[must_use]
pub fn silu_f32(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// SiLU in f64.
#[must_use]
pub fn silu_f64(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn lengths(
    hidden: u64,
    intermediate: u64,
    batch: u64,
    input: usize,
) -> Result<(usize, usize, usize)> {
    let h = usize::try_from(hidden)
        .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
    let i = usize::try_from(intermediate)
        .map_err(|_| ConvertError::inconsistent("intermediate width does not fit usize"))?;
    let n = usize::try_from(batch)
        .map_err(|_| ConvertError::inconsistent("batch size does not fit usize"))?;
    let expected = n
        .checked_mul(h)
        .ok_or_else(|| ConvertError::inconsistent("input length overflows"))?;
    if input != expected {
        return Err(ConvertError::inconsistent(format!(
            "input has {input} values, expected {expected} for batch={n} hidden={h}"
        )));
    }
    Ok((h, i, n))
}

fn hidden_lengths(hidden: u64, batch: u64, input: usize) -> Result<(usize, usize)> {
    let h = usize::try_from(hidden)
        .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
    let n = usize::try_from(batch)
        .map_err(|_| ConvertError::inconsistent("batch size does not fit usize"))?;
    let expected = n
        .checked_mul(h)
        .ok_or_else(|| ConvertError::inconsistent("input length overflows"))?;
    if input != expected {
        return Err(ConvertError::inconsistent(format!(
            "input has {input} values, expected {expected} for batch={n} hidden={h}"
        )));
    }
    Ok((h, n))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn grouping_by_count_is_uneven_at_the_end() {
        let groups = resolve_groups(&GroupSpec::ByCount(3), 10, true).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].indices, vec![0, 1, 2, 3]);
        assert_eq!(groups[1].indices, vec![4, 5, 6, 7]);
        assert_eq!(groups[2].indices, vec![8, 9]);
        assert!(matches!(
            groups[2].channels,
            ChannelSet::Range { start: 8, end: 10 }
        ));
    }

    #[test]
    fn explicit_groups_reject_overlap_and_incomplete_coverage() {
        let spec = GroupSpec::Explicit(vec![vec![0, 1], vec![1, 2]]);
        assert!(resolve_groups(&spec, 3, false).is_err());
        let spec = GroupSpec::Explicit(vec![vec![0, 1]]);
        assert!(resolve_groups(&spec, 3, false).is_ok());
        assert!(resolve_groups(&spec, 3, true).is_err());
    }

    #[test]
    fn group_spec_parses_ranges_and_lists() {
        let spec = GroupSpec::parse("0-2;4,6;7-8").unwrap();
        assert_eq!(
            spec,
            GroupSpec::Explicit(vec![vec![0, 1, 2], vec![4, 6], vec![7, 8]])
        );
        assert_eq!(GroupSpec::parse("4").unwrap(), GroupSpec::ByCount(4));
    }

    #[test]
    fn dense_and_grouped_paths_agree_on_a_tiny_case() {
        let hidden = 3_usize;
        let intermediate = 5_usize;
        let gate: Vec<f32> = (0..intermediate * hidden)
            .map(|index| (index as f32) * 0.01 - 0.1)
            .collect();
        let up: Vec<f32> = (0..intermediate * hidden)
            .map(|index| 0.2 - (index as f32) * 0.003)
            .collect();
        let down: Vec<f32> = (0..hidden * intermediate)
            .map(|index| (index as f32) * 0.005 - 0.05)
            .collect();
        let x = [0.5_f32, -0.25, 0.75, 0.1, 0.2, -0.3];
        let dense =
            ffn_dense_f32(&gate, &up, &down, hidden as u64, intermediate as u64, 2, &x).unwrap();
        let groups = vec![
            GroupMatrices {
                index: 0,
                channels: vec![0, 1, 2],
                gate: gate[..3 * hidden].to_vec(),
                up: up[..3 * hidden].to_vec(),
                down_t: (0..3 * hidden)
                    .map(|index| down[(index % hidden) * intermediate + index / hidden])
                    .collect(),
            },
            GroupMatrices {
                index: 1,
                channels: vec![3, 4],
                gate: gate[3 * hidden..].to_vec(),
                up: up[3 * hidden..].to_vec(),
                down_t: (0..2 * hidden)
                    .map(|index| down[(index % hidden) * intermediate + 3 + index / hidden])
                    .collect(),
            },
        ];
        let grouped = ffn_grouped_f32(hidden as u64, &groups, 2, &x).unwrap();
        for (left, right) in dense.iter().zip(grouped.iter()) {
            assert!((left - right).abs() < 1e-6);
        }
    }
}
