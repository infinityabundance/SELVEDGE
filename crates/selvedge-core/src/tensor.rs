//! Tensor descriptions: roles, dtypes, quantization schemes, storage order,
//! slices and learnable-parameter identities.
//!
//! Nothing here guesses an on-disk orientation. [`StorageOrder`] records the
//! actual axis order and [`TensorSlice::channel_axis`] names the axis the
//! channel range applies to.

use core::fmt;
use core::ops::Range;
use core::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{CoreError, Result};

/// Stable tensor roles used by the manifest, selector and accounting layers.
///
/// Known roles serialise as fixed lowercase names (`attn.q`, `mlp.gate`, ...).
/// An unknown audited role is carried as `Other(String)` and serialises as
/// `other:<name>`; the name must not collide with a known role so that the
/// round-trip is unambiguous.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorRole {
    /// Token embedding table.
    EmbedTokens,
    /// Output projection (may alias the embedding table).
    LmHead,
    /// Attention query projection.
    AttnQ,
    /// Attention key projection.
    AttnK,
    /// Attention value projection.
    AttnV,
    /// Attention output projection.
    AttnO,
    /// Fused query/key/value projection.
    AttnQkvFused,
    /// SwiGLU gate projection.
    MlpGate,
    /// SwiGLU up projection.
    MlpUp,
    /// SwiGLU down projection.
    MlpDown,
    /// Pre-attention input normalization weights.
    InputNorm,
    /// Post-attention normalization weights.
    PostAttnNorm,
    /// Learned router or controller parameters.
    Router,
    /// Learned memory key vectors.
    MemoryKey,
    /// Learned memory value vectors.
    MemoryValue,
    /// Adapter A matrix.
    AdapterA,
    /// Adapter B matrix.
    AdapterB,
    /// Weights of a recurrent block read more than once.
    RecurrentBlock,
    /// Verifier/controller weights.
    Verifier,
    /// Any other audited role, carried verbatim.
    Other(String),
}

const ROLE_NAMES: &[(&str, TensorRole)] = &[
    ("embed_tokens", TensorRole::EmbedTokens),
    ("lm_head", TensorRole::LmHead),
    ("attn.q", TensorRole::AttnQ),
    ("attn.k", TensorRole::AttnK),
    ("attn.v", TensorRole::AttnV),
    ("attn.o", TensorRole::AttnO),
    ("attn.qkv_fused", TensorRole::AttnQkvFused),
    ("mlp.gate", TensorRole::MlpGate),
    ("mlp.up", TensorRole::MlpUp),
    ("mlp.down", TensorRole::MlpDown),
    ("input_norm", TensorRole::InputNorm),
    ("post_attn_norm", TensorRole::PostAttnNorm),
    ("router", TensorRole::Router),
    ("memory.key", TensorRole::MemoryKey),
    ("memory.value", TensorRole::MemoryValue),
    ("adapter.a", TensorRole::AdapterA),
    ("adapter.b", TensorRole::AdapterB),
    ("recurrent_block", TensorRole::RecurrentBlock),
    ("verifier", TensorRole::Verifier),
];

impl TensorRole {
    /// Canonical name without the `other:` prefix for known roles.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Other(raw) => raw,
            known => known.known_name(),
        }
    }

    fn known_name(&self) -> &'static str {
        ROLE_NAMES
            .iter()
            .find_map(|(name, role)| (role == self).then_some(*name))
            .unwrap_or("other")
    }

    /// Construct an audited custom role.
    ///
    /// # Errors
    ///
    /// Rejects empty names, names longer than 128 bytes, characters outside
    /// `A-Za-z0-9._:/-`, and names that would parse back as a known role.
    pub fn other(name: &str) -> Result<Self> {
        validate_role_name(name)?;
        if ROLE_NAMES.iter().any(|(known, _)| *known == name) {
            return Err(CoreError::invalid(
                "tensor_role",
                format!("{name:?} is a known role and cannot be used as a custom name"),
            ));
        }
        Ok(Self::Other(name.to_owned()))
    }

    /// Parse a stable role name.
    ///
    /// # Errors
    ///
    /// Rejects malformed names; unknown valid names become `Other`.
    pub fn parse(input: &str) -> Result<Self> {
        if let Some((_, role)) = ROLE_NAMES.iter().find(|(name, _)| *name == input) {
            return Ok(role.clone());
        }
        if let Some(raw) = input.strip_prefix("other:") {
            return Self::other(raw);
        }
        Self::other(input)
    }

    /// Whether the role carries learnable parameters. Every role in this
    /// enumeration does; the method exists so future non-learned roles must
    /// state it explicitly rather than being counted by default.
    #[must_use]
    pub const fn is_learned(&self) -> bool {
        true
    }
}

fn validate_role_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(CoreError::invalid("tensor_role", "role name is empty"));
    }
    if name.len() > 128 {
        return Err(CoreError::invalid(
            "tensor_role",
            "role name longer than 128 bytes",
        ));
    }
    if !name.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
    }) {
        return Err(CoreError::invalid(
            "tensor_role",
            "role name contains characters outside A-Za-z0-9._:/-",
        ));
    }
    Ok(())
}

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Other(raw) => write!(f, "other:{raw}"),
            known => f.write_str(known.known_name()),
        }
    }
}

impl FromStr for TensorRole {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        Self::parse(input)
    }
}

impl Serialize for TensorRole {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TensorRole {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct RoleVisitor;

        impl Visitor<'_> for RoleVisitor {
            type Value = TensorRole;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a stable SELVEDGE tensor role name")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> core::result::Result<Self::Value, E> {
                TensorRole::parse(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(RoleVisitor)
    }
}

/// A storage or numeric element type.
///
/// Unknown names parse into `Unsupported` so a manifest can be read and then
/// rejected with a clear typed error during validation. Constructing a
/// `DType` never silently substitutes a supported type for an unknown one.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// Signed 8-bit integer, block-quantized.
    I8,
    /// Unsigned 8-bit integer, block-quantized.
    U8,
    /// 4-bit integers packed two per byte with a block size.
    I4Packed {
        /// Number of logical elements per packed block; must be even.
        block_size: u32,
    },
    /// Ternary values packed four per byte with a block size.
    TernaryPacked {
        /// Number of logical elements per packed block; must be a multiple of 4.
        block_size: u32,
    },
    /// A named type this build does not support; validation must reject it.
    Unsupported(String),
}

impl DType {
    /// Stable wire name.
    #[must_use]
    pub fn stable_name(&self) -> String {
        match self {
            Self::F32 => "f32".to_owned(),
            Self::F16 => "f16".to_owned(),
            Self::BF16 => "bf16".to_owned(),
            Self::I8 => "i8".to_owned(),
            Self::U8 => "u8".to_owned(),
            Self::I4Packed { block_size } => format!("i4_packed:{block_size}"),
            Self::TernaryPacked { block_size } => format!("ternary_packed:{block_size}"),
            Self::Unsupported(raw) => format!("unsupported:{raw}"),
        }
    }

    /// Parse a stable wire name.
    ///
    /// # Errors
    ///
    /// Rejects malformed packed block sizes and empty `unsupported:` names.
    pub fn parse(input: &str) -> Result<Self> {
        match input {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "i8" => Ok(Self::I8),
            "u8" => Ok(Self::U8),
            _ => {
                if let Some(block) = input.strip_prefix("i4_packed:") {
                    let block_size = block.parse::<u32>().map_err(|_| {
                        CoreError::invalid("dtype", format!("bad i4 block size in {input:?}"))
                    })?;
                    if block_size == 0 || !block_size.is_multiple_of(2) {
                        return Err(CoreError::invalid(
                            "dtype",
                            "i4_packed block size must be even and non-zero",
                        ));
                    }
                    return Ok(Self::I4Packed { block_size });
                }
                if let Some(block) = input.strip_prefix("ternary_packed:") {
                    let block_size = block.parse::<u32>().map_err(|_| {
                        CoreError::invalid("dtype", format!("bad ternary block size in {input:?}"))
                    })?;
                    if block_size == 0 || !block_size.is_multiple_of(4) {
                        return Err(CoreError::invalid(
                            "dtype",
                            "ternary_packed block size must be a non-zero multiple of 4",
                        ));
                    }
                    return Ok(Self::TernaryPacked { block_size });
                }
                let raw = input.strip_prefix("unsupported:").unwrap_or(input);
                if raw.is_empty() || raw.len() > 64 {
                    return Err(CoreError::invalid(
                        "dtype",
                        "unsupported dtype name must be 1..=64 bytes",
                    ));
                }
                Ok(Self::Unsupported(raw.to_owned()))
            }
        }
    }

    /// Whether this build can decode the type exactly.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported(_))
    }

    /// Bits occupied by one logical element, if the type is supported.
    #[must_use]
    pub fn bits_per_element(&self) -> Option<u32> {
        match self {
            Self::F32 => Some(32),
            Self::F16 | Self::BF16 => Some(16),
            Self::I8 | Self::U8 => Some(8),
            Self::I4Packed { .. } => Some(4),
            Self::TernaryPacked { .. } => Some(2),
            Self::Unsupported(_) => None,
        }
    }

    /// Whether the type stores packed low-bit values.
    #[must_use]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::I4Packed { .. } | Self::TernaryPacked { .. })
    }

    /// Whether the type requires a quantization scheme.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::I8 | Self::U8 | Self::I4Packed { .. } | Self::TernaryPacked { .. }
        )
    }

    /// Reject types this build cannot decode.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::UnsupportedDType`] for `Unsupported`.
    pub fn require_supported(&self) -> Result<()> {
        if let Self::Unsupported(raw) = self {
            return Err(CoreError::UnsupportedDType { dtype: raw.clone() });
        }
        Ok(())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.stable_name())
    }
}

impl FromStr for DType {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        Self::parse(input)
    }
}

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.stable_name())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct DTypeVisitor;

        impl Visitor<'_> for DTypeVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a stable SELVEDGE dtype name")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> core::result::Result<Self::Value, E> {
                DType::parse(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(DTypeVisitor)
    }
}

/// Block quantization scheme (affine or symmetric).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum QuantScheme {
    /// Symmetric scale-only quantization.
    Symmetric {
        /// Logical elements per block; must be non-zero.
        block_size: u32,
    },
    /// Affine quantization with scales and optionally zero points.
    Affine {
        /// Logical elements per block; must be non-zero.
        block_size: u32,
        /// Whether zero-point values are stored alongside scales.
        zero_points: bool,
    },
}

impl QuantScheme {
    /// Logical elements per block.
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        match self {
            Self::Symmetric { block_size } | Self::Affine { block_size, .. } => *block_size,
        }
    }

    /// Whether zero points are present.
    #[must_use]
    pub const fn has_zero_points(&self) -> bool {
        matches!(
            self,
            Self::Affine {
                zero_points: true,
                ..
            }
        )
    }

    /// Stable wire name including the block size.
    #[must_use]
    pub fn stable_name(&self) -> String {
        match self {
            Self::Symmetric { block_size } => format!("symmetric:{block_size}"),
            Self::Affine {
                block_size,
                zero_points,
            } => format!("affine:{block_size}:{}", u8::from(*zero_points)),
        }
    }

    /// Validate the scheme against an element count and dtype.
    ///
    /// # Errors
    ///
    /// Rejects a zero block size, an element count that is not a whole number
    /// of blocks, a block size larger than the element count, and a mismatch
    /// between a packed dtype and the scheme block size.
    pub fn validate(&self, elements: u64, dtype: &DType) -> Result<()> {
        let block = u64::from(self.block_size());
        if block == 0 {
            return Err(CoreError::invalid("quant", "block size must be non-zero"));
        }
        if elements == 0 {
            return Err(CoreError::invalid(
                "quant",
                "element count must be non-zero",
            ));
        }
        if block > elements {
            return Err(CoreError::invalid(
                "quant",
                format!("block size {block} exceeds element count {elements}"),
            ));
        }
        if !elements.is_multiple_of(block) {
            return Err(CoreError::invalid(
                "quant",
                format!("element count {elements} is not a multiple of block size {block}"),
            ));
        }
        match dtype {
            DType::I4Packed { block_size } | DType::TernaryPacked { block_size }
                if u64::from(*block_size) != block =>
            {
                Err(CoreError::invalid(
                    "quant",
                    format!(
                        "packed dtype block {block_size} does not match scheme block {}",
                        self.block_size()
                    ),
                ))
            }
            _ => Ok(()),
        }
    }
}

/// On-disk orientation of a tensor.
///
/// `RowMajor` and `ColumnMajor` name the conventional orderings.
/// `Transposed { axes }` states the physical axis order explicitly:
/// `axes[physical_position] = logical_axis`. It is never inferred from a shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StorageOrder {
    /// Rows contiguous.
    RowMajor,
    /// Columns contiguous.
    ColumnMajor,
    /// Explicit physical axis order.
    Transposed {
        /// Permutation of `0..rank`.
        axes: Vec<u32>,
    },
}

impl StorageOrder {
    /// Validate against a tensor rank.
    ///
    /// # Errors
    ///
    /// Rejects wrong-length or non-permutation axis lists.
    pub fn validate(&self, rank: usize) -> Result<()> {
        if let Self::Transposed { axes } = self {
            if axes.len() != rank {
                return Err(CoreError::invalid(
                    "storage_order",
                    format!(
                        "transposed axis list has {} entries for rank {rank}",
                        axes.len()
                    ),
                ));
            }
            let mut seen = vec![false; rank];
            for axis in axes {
                let index = usize::try_from(*axis)
                    .map_err(|_| CoreError::invalid("storage_order", "axis does not fit usize"))?;
                if index >= rank {
                    return Err(CoreError::invalid(
                        "storage_order",
                        format!("axis {axis} is outside rank {rank}"),
                    ));
                }
                if seen[index] {
                    return Err(CoreError::invalid(
                        "storage_order",
                        format!("axis {axis} appears twice"),
                    ));
                }
                seen[index] = true;
            }
        }
        Ok(())
    }

    /// Physical axis order as logical axis indices.
    ///
    /// # Errors
    ///
    /// Propagates validation failures.
    pub fn physical_axes(&self, rank: usize) -> Result<Vec<u32>> {
        self.validate(rank)?;
        match self {
            Self::RowMajor => Ok((0..rank).map(|axis| axis as u32).collect()),
            Self::ColumnMajor => Ok((0..rank).rev().map(|axis| axis as u32).collect()),
            Self::Transposed { axes } => Ok(axes.clone()),
        }
    }
}

/// A validated, non-empty element range under one underlying parameter tensor.
///
/// Two [`ParamIdentity`] values with equal tensor keys alias the same learned
/// parameter elements; their ranges are unioned by the ledger. A tied
/// embedding and LM head must use the same tensor key to be counted once.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "ParamIdentityWire")]
pub struct ParamIdentity {
    tensor: ParamTensorId,
    elements: Range<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParamIdentityWire {
    tensor: ParamTensorId,
    elements: Range<u64>,
}

impl TryFrom<ParamIdentityWire> for ParamIdentity {
    type Error = CoreError;

    fn try_from(wire: ParamIdentityWire) -> Result<Self> {
        Self::new(wire.tensor, wire.elements)
    }
}

impl ParamIdentity {
    /// Construct from a tensor key and a half-open element range.
    ///
    /// # Errors
    ///
    /// Rejects empty or reversed ranges.
    pub fn new(tensor: ParamTensorId, elements: Range<u64>) -> Result<Self> {
        if elements.start >= elements.end {
            return Err(CoreError::invalid(
                "param_identity",
                format!(
                    "range {}..{} is empty or reversed",
                    elements.start, elements.end
                ),
            ));
        }
        Ok(Self { tensor, elements })
    }

    /// Tensor key.
    #[must_use]
    pub fn tensor(&self) -> &ParamTensorId {
        &self.tensor
    }

    /// Half-open element range.
    #[must_use]
    pub fn elements(&self) -> &Range<u64> {
        &self.elements
    }

    /// Number of elements covered by the range.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.elements.end - self.elements.start
    }

    /// Always `false`: a parameter identity is never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Whether two identities share at least one element of the same tensor.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.tensor == other.tensor
            && self.elements.start < other.elements.end
            && other.elements.start < self.elements.end
    }
}

/// Canonical key of an underlying parameter tensor.
///
/// The key is an explicit identifier, not a content hash: aliasing is expressed
/// by reusing the same key. It accepts `A-Za-z0-9._:/-` and is limited to 256
/// bytes so that a canonical encoding cannot be made unbounded by a key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ParamTensorId(String);

impl ParamTensorId {
    /// Construct a validated tensor key.
    ///
    /// # Errors
    ///
    /// Rejects empty, over-long or character-unsafe keys.
    pub fn new(key: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if key.is_empty() {
            return Err(CoreError::invalid("param_tensor_id", "key is empty"));
        }
        if key.len() > 256 {
            return Err(CoreError::invalid(
                "param_tensor_id",
                "key longer than 256 bytes",
            ));
        }
        if !key.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        }) {
            return Err(CoreError::invalid(
                "param_tensor_id",
                "key contains characters outside A-Za-z0-9._:/-",
            ));
        }
        Ok(Self(key))
    }

    /// Borrow the key.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ParamTensorId {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl fmt::Display for ParamTensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ParamTensorId {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        Self::new(input)
    }
}

/// A contiguous or enumerated set of channels along one tensor axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChannelSet {
    /// Half-open contiguous range.
    Range {
        /// First channel.
        start: u64,
        /// One past the last channel.
        end: u64,
    },
    /// Sorted, strictly increasing channel indices.
    List {
        /// Explicit channel indices.
        indices: Vec<u64>,
    },
}

impl ChannelSet {
    /// Construct a validated range.
    ///
    /// # Errors
    ///
    /// Rejects empty or reversed ranges.
    pub fn range(start: u64, end: u64) -> Result<Self> {
        if start >= end {
            return Err(CoreError::invalid(
                "channel_set",
                format!("range {start}..{end} is empty or reversed"),
            ));
        }
        Ok(Self::Range { start, end })
    }

    /// Construct a validated list.
    ///
    /// # Errors
    ///
    /// Rejects empty lists and unsorted or duplicate indices.
    pub fn list(indices: impl IntoIterator<Item = u64>) -> Result<Self> {
        let indices: Vec<u64> = indices.into_iter().collect();
        if indices.is_empty() {
            return Err(CoreError::invalid("channel_set", "list is empty"));
        }
        for pair in indices.windows(2) {
            if pair[0] >= pair[1] {
                return Err(CoreError::invalid(
                    "channel_set",
                    "list indices must be strictly increasing",
                ));
            }
        }
        Ok(Self::List { indices })
    }

    /// Number of channels.
    #[must_use]
    pub fn count(&self) -> u64 {
        match self {
            Self::Range { start, end } => end - start,
            Self::List { indices } => indices.len() as u64,
        }
    }

    /// Contiguous span as `(first, one-past-last)`.
    #[must_use]
    pub fn span(&self) -> (u64, u64) {
        match self {
            Self::Range { start, end } => (*start, *end),
            Self::List { indices } => {
                let first = indices.first().copied().unwrap_or(0);
                let last = indices.last().copied().unwrap_or(0);
                (first, last.saturating_add(1))
            }
        }
    }

    /// Validate the set.
    ///
    /// # Errors
    ///
    /// Rejects empty sets, reversed ranges and malformed lists (deserialized
    /// values bypass [`ChannelSet::range`]/[`ChannelSet::list`]).
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Range { start, end } => {
                if start >= end {
                    return Err(CoreError::invalid(
                        "channel_set",
                        format!("range {start}..{end} is empty or reversed"),
                    ));
                }
            }
            Self::List { indices } => {
                if indices.is_empty() {
                    return Err(CoreError::invalid("channel_set", "list is empty"));
                }
                for pair in indices.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(CoreError::invalid(
                            "channel_set",
                            "list indices must be strictly increasing",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A validated tensor slice.
///
/// `shape` is the logical shape of the stored slice (not the full tensor) and
/// `channel_range` applies along `channel_axis`; therefore
/// `shape[channel_axis] == channel_range.len()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TensorSliceWire")]
pub struct TensorSlice {
    /// Tensor role.
    pub role: TensorRole,
    /// Optional transformer layer index.
    pub layer: Option<u32>,
    /// Half-open channel range along [`TensorSlice::channel_axis`].
    pub channel_range: Range<u64>,
    /// Logical axis the channel range applies to.
    pub channel_axis: u32,
    /// Logical shape of the stored slice.
    pub shape: Vec<u64>,
    /// Actual on-disk orientation.
    pub storage_order: StorageOrder,
    /// Element type.
    pub dtype: DType,
    /// Quantization scheme when the dtype requires one.
    pub quant: Option<QuantScheme>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorSliceWire {
    role: TensorRole,
    layer: Option<u32>,
    channel_range: Range<u64>,
    channel_axis: u32,
    shape: Vec<u64>,
    storage_order: StorageOrder,
    dtype: DType,
    quant: Option<QuantScheme>,
}

impl TryFrom<TensorSliceWire> for TensorSlice {
    type Error = CoreError;

    fn try_from(wire: TensorSliceWire) -> Result<Self> {
        let slice = Self {
            role: wire.role,
            layer: wire.layer,
            channel_range: wire.channel_range,
            channel_axis: wire.channel_axis,
            shape: wire.shape,
            storage_order: wire.storage_order,
            dtype: wire.dtype,
            quant: wire.quant,
        };
        slice.validate()?;
        Ok(slice)
    }
}

impl TensorSlice {
    /// Construct and validate a slice.
    ///
    /// # Errors
    ///
    /// Rejects every invariant listed on [`TensorSlice::validate`].
    // Eight independent description fields are the data contract; grouping
    // them into another struct would only move the argument list.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: TensorRole,
        layer: Option<u32>,
        channel_range: Range<u64>,
        channel_axis: u32,
        shape: Vec<u64>,
        storage_order: StorageOrder,
        dtype: DType,
        quant: Option<QuantScheme>,
    ) -> Result<Self> {
        let slice = Self {
            role,
            layer,
            channel_range,
            channel_axis,
            shape,
            storage_order,
            dtype,
            quant,
        };
        slice.validate()?;
        Ok(slice)
    }

    /// Number of logical elements in the slice.
    ///
    /// # Errors
    ///
    /// Rejects shape products that overflow `u64`.
    pub fn element_count(&self) -> Result<u64> {
        crate::units::element_count(&self.shape)
    }

    /// Number of channels covered.
    #[must_use]
    pub fn channel_count(&self) -> u64 {
        self.channel_range
            .end
            .saturating_sub(self.channel_range.start)
    }

    /// Validate every invariant of the slice.
    ///
    /// # Errors
    ///
    /// Rejects: empty/reversed channel ranges; empty shapes or zero dimensions;
    /// a rank above 16; a channel axis outside the rank; a channel count not
    /// equal to `shape[channel_axis]`; a malformed storage order; unsupported
    /// dtypes; quantization schemes on unquantized dtypes, missing schemes on
    /// quantized dtypes, and block sizes that do not divide the element count.
    pub fn validate(&self) -> Result<()> {
        if self.channel_range.start >= self.channel_range.end {
            return Err(CoreError::invalid(
                "channel_range",
                format!(
                    "range {}..{} is empty or reversed",
                    self.channel_range.start, self.channel_range.end
                ),
            ));
        }
        if self.shape.is_empty() {
            return Err(CoreError::invalid("shape", "shape is empty"));
        }
        if self.shape.len() > 16 {
            return Err(CoreError::invalid("shape", "rank above 16"));
        }
        if self.shape.contains(&0) {
            return Err(CoreError::invalid(
                "shape",
                "shape contains a zero dimension",
            ));
        }
        let axis = usize::try_from(self.channel_axis)
            .map_err(|_| CoreError::invalid("channel_axis", "axis does not fit usize"))?;
        if axis >= self.shape.len() {
            return Err(CoreError::invalid(
                "channel_axis",
                format!(
                    "axis {} outside rank {}",
                    self.channel_axis,
                    self.shape.len()
                ),
            ));
        }
        if self.shape[axis] != self.channel_count() {
            return Err(CoreError::invalid(
                "shape",
                format!(
                    "shape[{}]={} does not equal channel count {}",
                    self.channel_axis,
                    self.shape[axis],
                    self.channel_count()
                ),
            ));
        }
        self.storage_order.validate(self.shape.len())?;
        self.dtype.require_supported()?;
        let elements = self.element_count()?;
        match (&self.dtype, &self.quant) {
            (DType::Unsupported(raw), _) => {
                return Err(CoreError::UnsupportedDType { dtype: raw.clone() });
            }
            (dtype, Some(scheme)) if dtype.is_quantized() => {
                scheme.validate(elements, dtype)?;
            }
            (dtype, None) if dtype.is_quantized() => {
                return Err(CoreError::invalid(
                    "quant",
                    format!("dtype {dtype} requires a quantization scheme"),
                ));
            }
            (dtype, Some(_)) => {
                return Err(CoreError::invalid(
                    "quant",
                    format!("dtype {dtype} must not carry a quantization scheme"),
                ));
            }
            (_, None) => {}
        }
        Ok(())
    }
}

/// Checked element count for shapes and ranges.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ElementCount(u64);

impl ElementCount {
    /// Wrap a raw count.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Raw count.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Checked addition.
    ///
    /// # Errors
    ///
    /// Rejects `u64` overflow.
    pub fn checked_add(self, other: Self) -> Result<Self> {
        self.0
            .checked_add(other.0)
            .map(Self)
            .ok_or_else(|| CoreError::overflow("element count addition"))
    }

    /// Checked multiplication.
    ///
    /// # Errors
    ///
    /// Rejects `u64` overflow.
    pub fn checked_mul(self, other: Self) -> Result<Self> {
        self.0
            .checked_mul(other.0)
            .map(Self)
            .ok_or_else(|| CoreError::overflow("element count multiplication"))
    }
}

impl From<u64> for ElementCount {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn f32_slice() -> TensorSlice {
        TensorSlice::new(
            TensorRole::MlpGate,
            Some(0),
            8..16,
            0,
            vec![8, 16],
            StorageOrder::RowMajor,
            DType::F32,
            None,
        )
        .unwrap()
    }

    #[test]
    fn role_round_trips_known_and_other() {
        for role in [
            TensorRole::EmbedTokens,
            TensorRole::AttnQkvFused,
            TensorRole::Other("routed.expert.7".to_owned()),
        ] {
            let text = role.to_string();
            assert_eq!(TensorRole::parse(&text).unwrap(), role);
        }
        assert!(TensorRole::other("mlp.gate").is_err());
        assert!(TensorRole::other("bad name").is_err());
    }

    #[test]
    fn dtype_names_round_trip() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::U8,
            DType::I4Packed { block_size: 32 },
            DType::TernaryPacked { block_size: 64 },
            DType::Unsupported("q4_k".to_owned()),
        ] {
            let text = dtype.stable_name();
            assert_eq!(DType::parse(&text).unwrap(), dtype);
        }
        assert!(!DType::Unsupported("q4_k".to_owned()).is_supported());
        assert!(DType::parse("i4_packed:7").is_err());
    }

    #[test]
    fn storage_order_permutations_are_validated() {
        assert!(StorageOrder::RowMajor.validate(3).is_ok());
        assert_eq!(
            StorageOrder::ColumnMajor.physical_axes(3).unwrap(),
            vec![2, 1, 0]
        );
        assert!(
            StorageOrder::Transposed { axes: vec![1, 0] }
                .validate(2)
                .is_ok()
        );
        assert!(
            StorageOrder::Transposed { axes: vec![0, 0] }
                .validate(2)
                .is_err()
        );
        assert!(
            StorageOrder::Transposed { axes: vec![0] }
                .validate(2)
                .is_err()
        );
    }

    #[test]
    fn slice_rejects_invariant_violations() {
        assert!(f32_slice().validate().is_ok());
        assert!(
            TensorSlice::new(
                TensorRole::MlpGate,
                None,
                8..8,
                0,
                vec![0, 16],
                StorageOrder::RowMajor,
                DType::F32,
                None,
            )
            .is_err()
        );
        assert!(
            TensorSlice::new(
                TensorRole::MlpGate,
                None,
                8..16,
                0,
                vec![16, 8],
                StorageOrder::RowMajor,
                DType::F32,
                None,
            )
            .is_err()
        );
        assert!(
            TensorSlice::new(
                TensorRole::MlpGate,
                None,
                0..8,
                0,
                vec![8, 16],
                StorageOrder::RowMajor,
                DType::I8,
                None,
            )
            .is_err()
        );
        assert!(
            TensorSlice::new(
                TensorRole::MlpGate,
                None,
                0..8,
                0,
                vec![8, 16],
                StorageOrder::RowMajor,
                DType::F32,
                Some(QuantScheme::Symmetric { block_size: 32 }),
            )
            .is_err()
        );
    }

    #[test]
    fn channel_set_validation_and_span() {
        let range = ChannelSet::range(4, 12).unwrap();
        assert_eq!(range.count(), 8);
        assert_eq!(range.span(), (4, 12));
        let list = ChannelSet::list([1_u64, 5, 9]).unwrap();
        assert_eq!(list.count(), 3);
        assert_eq!(list.span(), (1, 10));
        assert!(ChannelSet::list([5, 5]).is_err());
        assert!(ChannelSet::list([9, 5]).is_err());
    }

    #[test]
    fn param_identity_alias_detection() {
        let key = ParamTensorId::new("model.layers.0.mlp.gate.weight").unwrap();
        let left = ParamIdentity::new(key.clone(), 0..100).unwrap();
        let right = ParamIdentity::new(key, 50..150).unwrap();
        let other_key = ParamTensorId::new("model.layers.0.mlp.up.weight").unwrap();
        let other = ParamIdentity::new(other_key, 0..100).unwrap();
        assert!(left.overlaps(&right));
        assert!(!left.overlaps(&other));
        assert_eq!(left.len(), 100);
    }
}
