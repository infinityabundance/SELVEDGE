//! The tensor-source contract, dtype decoding and the in-flight byte budget.
//!
//! Two implementations exist: [`crate::safetensors::Safetensors`] streams from a
//! checkpoint file with positional reads, and [`MemoryTensorSource`] backs the
//! tests and the synthetic fixture. Both expose the same rank-2 region read so
//! the conversion math never sees a storage order or a source dtype directly.

use std::collections::BTreeMap;
use std::ops::Range;

use selvedge_core::{DType, StorageOrder};

use crate::error::{ConvertError, Result};

/// A checked, non-zero byte budget for tensors held in flight.
///
/// The converter reserves bytes before allocating an output buffer and releases
/// them after the extent has been written. Peak usage is therefore a property of
/// the algorithm, not of a hidden allocator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    limit: u64,
    in_flight: u64,
    peak: u64,
}

impl MemoryBudget {
    /// Construct a budget.
    ///
    /// # Errors
    ///
    /// Rejects a zero limit.
    pub fn new(limit: u64) -> Result<Self> {
        if limit == 0 {
            return Err(ConvertError::MemoryBudgetExceeded {
                requested: 0,
                used: 0,
                limit,
            });
        }
        Ok(Self {
            limit,
            in_flight: 0,
            peak: 0,
        })
    }

    /// Configured limit.
    #[must_use]
    pub const fn limit(&self) -> u64 {
        self.limit
    }

    /// Bytes currently reserved.
    #[must_use]
    pub const fn in_flight(&self) -> u64 {
        self.in_flight
    }

    /// Largest number of bytes ever simultaneously reserved.
    #[must_use]
    pub const fn peak(&self) -> u64 {
        self.peak
    }

    /// Reserve bytes.
    ///
    /// # Errors
    ///
    /// Rejects a reservation that would exceed the limit or overflow.
    pub fn reserve(&mut self, bytes: u64) -> Result<()> {
        let next = self
            .in_flight
            .checked_add(bytes)
            .ok_or(ConvertError::MemoryBudgetExceeded {
                requested: bytes,
                used: self.in_flight,
                limit: self.limit,
            })?;
        if next > self.limit {
            return Err(ConvertError::MemoryBudgetExceeded {
                requested: bytes,
                used: self.in_flight,
                limit: self.limit,
            });
        }
        self.in_flight = next;
        self.peak = self.peak.max(next);
        Ok(())
    }

    /// Release bytes previously reserved.
    ///
    /// # Errors
    ///
    /// Rejects releasing more bytes than are reserved, which would hide a
    /// bookkeeping bug.
    pub fn release(&mut self, bytes: u64) -> Result<()> {
        self.in_flight =
            self.in_flight
                .checked_sub(bytes)
                .ok_or(ConvertError::MemoryBudgetExceeded {
                    requested: bytes,
                    used: self.in_flight,
                    limit: self.limit,
                })?;
        Ok(())
    }
}

/// The description of one source tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Tensor key.
    pub key: String,
    /// Element type as decoded by this build.
    pub dtype: DType,
    /// Logical shape in the checkpoint.
    pub shape: Vec<u64>,
    /// Physical storage order of the source bytes.
    pub storage_order: StorageOrder,
    /// Number of logical elements (zero for a zero-size tensor).
    pub elements: u64,
    /// Encoded weight bytes.
    pub byte_len: u64,
}

impl TensorDescriptor {
    /// Rank of the tensor.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Rows of a rank-2 tensor.
    ///
    /// # Errors
    ///
    /// Rejects any other rank.
    pub fn rows(&self) -> Result<u64> {
        self.dim(0)
    }

    /// Columns of a rank-2 tensor.
    ///
    /// # Errors
    ///
    /// Rejects any other rank.
    pub fn cols(&self) -> Result<u64> {
        self.dim(1)
    }

    fn dim(&self, axis: usize) -> Result<u64> {
        if self.rank() != 2 {
            return Err(ConvertError::RankMismatch {
                tensor: self.key.clone(),
                rank: self.rank(),
            });
        }
        self.shape
            .get(axis)
            .copied()
            .ok_or_else(|| ConvertError::RankMismatch {
                tensor: self.key.clone(),
                rank: self.rank(),
            })
    }
}

/// A source of rank-2 f32 regions with per-tensor positional reads.
///
/// `read_region_f32` returns `rows.len() * cols.len()` values in row-major
/// logical order and never returns a factor other than 4 for the output.
pub trait TensorSource {
    /// Sorted tensor keys.
    fn tensor_keys(&self) -> Vec<String>;

    /// Descriptor for one tensor.
    ///
    /// # Errors
    ///
    /// Returns [`ConvertError::MissingTensor`] for an unknown key.
    fn descriptor(&self, key: &str) -> Result<TensorDescriptor>;

    /// Read a rectangular region of a rank-2 tensor as f32.
    ///
    /// # Errors
    ///
    /// Rejects unknown keys, non-rank-2 tensors, empty or out-of-bounds regions,
    /// unsupported dtypes, allocation refusal and short reads. The caller is
    /// responsible for accounting the returned buffer against a
    /// [`MemoryBudget`].
    fn read_region_f32(
        &mut self,
        key: &str,
        rows: Range<u64>,
        cols: Range<u64>,
    ) -> Result<Vec<f32>>;
}

/// A tensor held entirely in memory, used by tests and the synthetic fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryTensor {
    /// Descriptor.
    pub descriptor: TensorDescriptor,
    /// Bytes in physical order.
    pub data: Vec<u8>,
}

impl MemoryTensor {
    /// Build from logical row-major f32 values.
    ///
    /// # Errors
    ///
    /// Rejects a shape whose product disagrees with the value count.
    pub fn from_f32(
        key: impl Into<String>,
        shape: Vec<u64>,
        storage_order: StorageOrder,
        values: &[f32],
    ) -> Result<Self> {
        let key = key.into();
        let elements = shape_product(&key, &shape)?;
        if elements != values.len() as u64 {
            return Err(ConvertError::ByteLengthMismatch {
                tensor: key,
                extent: (values.len() as u64) * 4,
                expected: elements * 4,
                elements,
                dtype: "f32".to_owned(),
            });
        }
        let data = encode_values(&shape, &storage_order, values)?;
        Self::from_raw(key, DType::F32, shape, storage_order, data)
    }

    /// Build from logical row-major f16 values, encoded with round-to-nearest
    /// even.
    ///
    /// # Errors
    ///
    /// Rejects a shape/value-count mismatch.
    pub fn from_f16(
        key: impl Into<String>,
        shape: Vec<u64>,
        storage_order: StorageOrder,
        values: &[f32],
    ) -> Result<Self> {
        let key = key.into();
        let elements = shape_product(&key, &shape)?;
        if elements != values.len() as u64 {
            return Err(ConvertError::ByteLengthMismatch {
                tensor: key,
                extent: (values.len() as u64) * 2,
                expected: elements * 2,
                elements,
                dtype: "f16".to_owned(),
            });
        }
        let ordered = reorder_logical(&shape, &storage_order, values)?;
        let mut data = Vec::new();
        reserve_bytes(&mut data, ordered.len() as u64 * 2)?;
        for value in ordered {
            data.extend_from_slice(&encode_f16(value).to_le_bytes());
        }
        Self::from_raw(key, DType::F16, shape, storage_order, data)
    }

    /// Build from logical row-major bf16 values, encoded with round-to-nearest
    /// even.
    ///
    /// # Errors
    ///
    /// Rejects a shape/value-count mismatch.
    pub fn from_bf16(
        key: impl Into<String>,
        shape: Vec<u64>,
        storage_order: StorageOrder,
        values: &[f32],
    ) -> Result<Self> {
        let key = key.into();
        let elements = shape_product(&key, &shape)?;
        if elements != values.len() as u64 {
            return Err(ConvertError::ByteLengthMismatch {
                tensor: key,
                extent: (values.len() as u64) * 2,
                expected: elements * 2,
                elements,
                dtype: "bf16".to_owned(),
            });
        }
        let ordered = reorder_logical(&shape, &storage_order, values)?;
        let mut data = Vec::new();
        reserve_bytes(&mut data, ordered.len() as u64 * 2)?;
        for value in ordered {
            data.extend_from_slice(&encode_bf16(value).to_le_bytes());
        }
        Self::from_raw(key, DType::BF16, shape, storage_order, data)
    }

    /// Build from raw physical bytes.
    ///
    /// # Errors
    ///
    /// Rejects an unsupported dtype, a malformed storage order or a byte-length
    /// mismatch.
    pub fn from_raw(
        key: impl Into<String>,
        dtype: DType,
        shape: Vec<u64>,
        storage_order: StorageOrder,
        data: Vec<u8>,
    ) -> Result<Self> {
        let key = key.into();
        dtype.require_supported()?;
        storage_order.validate(shape.len())?;
        let elements = shape_product(&key, &shape)?;
        let expected = elements
            .checked_mul(u64::from(dtype_bytes(&dtype)?))
            .ok_or(ConvertError::Core(selvedge_core::CoreError::overflow(
                "memory tensor byte count",
            )))?;
        if expected != data.len() as u64 {
            return Err(ConvertError::ByteLengthMismatch {
                tensor: key,
                extent: data.len() as u64,
                expected,
                elements,
                dtype: dtype.stable_name(),
            });
        }
        Ok(Self {
            descriptor: TensorDescriptor {
                key,
                dtype,
                shape,
                storage_order,
                elements,
                byte_len: expected,
            },
            data,
        })
    }
}

/// A [`TensorSource`] over named in-memory tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryTensorSource {
    tensors: BTreeMap<String, MemoryTensor>,
}

impl MemoryTensorSource {
    /// An empty source.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a tensor.
    ///
    /// # Errors
    ///
    /// Rejects a duplicate key.
    pub fn insert(&mut self, tensor: MemoryTensor) -> Result<()> {
        let key = tensor.descriptor.key.clone();
        if self.tensors.insert(key.clone(), tensor).is_some() {
            return Err(ConvertError::DuplicateKey { key });
        }
        Ok(())
    }

    /// Number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the source is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Borrow a tensor.
    #[must_use]
    pub fn tensor(&self, key: &str) -> Option<&MemoryTensor> {
        self.tensors.get(key)
    }
}

impl TensorSource for MemoryTensorSource {
    fn tensor_keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn descriptor(&self, key: &str) -> Result<TensorDescriptor> {
        self.tensors
            .get(key)
            .map(|tensor| tensor.descriptor.clone())
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))
    }

    fn read_region_f32(
        &mut self,
        key: &str,
        rows: Range<u64>,
        cols: Range<u64>,
    ) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(key)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        read_region_from_bytes(&tensor.descriptor, &tensor.data, rows, cols)
    }
}

/// Read a rank-2 region from raw physical bytes.
pub(crate) fn read_region_from_bytes(
    descriptor: &TensorDescriptor,
    data: &[u8],
    rows: Range<u64>,
    cols: Range<u64>,
) -> Result<Vec<f32>> {
    if descriptor.rank() != 2 {
        return Err(ConvertError::RankMismatch {
            tensor: descriptor.key.clone(),
            rank: descriptor.rank(),
        });
    }
    let (height, width) = (descriptor.shape[0], descriptor.shape[1]);
    if rows.start >= rows.end || cols.start >= cols.end || rows.end > height || cols.end > width {
        return Err(ConvertError::RegionOutOfBounds {
            tensor: descriptor.key.clone(),
            rows_start: rows.start,
            rows_end: rows.end,
            cols_start: cols.start,
            cols_end: cols.end,
        });
    }
    let bytes_per = u64::from(dtype_bytes(&descriptor.dtype)?);
    let out_elements = (rows.end - rows.start)
        .checked_mul(cols.end - cols.start)
        .ok_or(ConvertError::Core(selvedge_core::CoreError::overflow(
            "region element count",
        )))?;
    let out_bytes = out_elements.checked_mul(4).ok_or(ConvertError::Core(
        selvedge_core::CoreError::overflow("region byte count"),
    ))?;
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, out_elements)?;
    let mut scratch = Vec::new();
    reserve_bytes(&mut scratch, bytes_per)?;
    scratch.resize(
        usize::try_from(bytes_per).map_err(|_| ConvertError::Allocation {
            bytes: bytes_per,
            reason: "element size does not fit usize".to_owned(),
        })?,
        0,
    );
    for row in rows.clone() {
        for col in cols.clone() {
            let index = physical_index(&descriptor.shape, &descriptor.storage_order, row, col)?;
            let offset = index.checked_mul(bytes_per).ok_or(ConvertError::Core(
                selvedge_core::CoreError::overflow("physical byte offset"),
            ))?;
            let start = usize::try_from(offset).map_err(|_| ConvertError::Allocation {
                bytes: offset,
                reason: "element offset does not fit usize".to_owned(),
            })?;
            let end = start
                .checked_add(scratch.len())
                .ok_or(ConvertError::Allocation {
                    bytes: offset,
                    reason: "element end overflow".to_owned(),
                })?;
            let slice = data
                .get(start..end)
                .ok_or_else(|| ConvertError::ByteLengthMismatch {
                    tensor: descriptor.key.clone(),
                    extent: data.len() as u64,
                    expected: end as u64,
                    elements: descriptor.elements,
                    dtype: descriptor.dtype.stable_name(),
                })?;
            out.push(decode_one(&descriptor.dtype, slice)?);
        }
    }
    if out_bytes != (out.len() as u64) * 4 {
        return Err(ConvertError::Allocation {
            bytes: out_bytes,
            reason: "output length disagrees with the requested region".to_owned(),
        });
    }
    Ok(out)
}

/// Read a whole tensor as f32 in logical row-major order, for any rank of one
/// or two.
///
/// [`read_region_from_bytes`] handles rank-2 regions; this is its whole-tensor
/// counterpart, used by the inference runtime for rank-1 normalization gains.
/// The physical-to-logical mapping uses [`physical_index`], so a transposed
/// rank-2 source is decoded correctly rather than reinterpreted as row-major.
///
/// # Errors
///
/// Rejects a rank above two, a byte length that disagrees with the descriptor,
/// an unsupported dtype and arithmetic overflow.
pub fn decode_tensor_f32(descriptor: &TensorDescriptor, data: &[u8]) -> Result<Vec<f32>> {
    if descriptor.byte_len != data.len() as u64 {
        return Err(ConvertError::ByteLengthMismatch {
            tensor: descriptor.key.clone(),
            extent: data.len() as u64,
            expected: descriptor.byte_len,
            elements: descriptor.elements,
            dtype: descriptor.dtype.stable_name(),
        });
    }
    let bytes_per = u64::from(dtype_bytes(&descriptor.dtype)?);
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, descriptor.elements)?;
    match descriptor.rank() {
        1 => {
            if descriptor.storage_order != StorageOrder::RowMajor {
                return Err(ConvertError::RankMismatch {
                    tensor: descriptor.key.clone(),
                    rank: descriptor.rank(),
                });
            }
            for index in 0..descriptor.elements {
                let start =
                    usize::try_from(index * bytes_per).map_err(|_| ConvertError::Allocation {
                        bytes: index * bytes_per,
                        reason: "element offset does not fit usize".to_owned(),
                    })?;
                let end = start + usize::try_from(bytes_per).unwrap_or(0);
                let slice =
                    data.get(start..end)
                        .ok_or_else(|| ConvertError::ByteLengthMismatch {
                            tensor: descriptor.key.clone(),
                            extent: data.len() as u64,
                            expected: descriptor.byte_len,
                            elements: descriptor.elements,
                            dtype: descriptor.dtype.stable_name(),
                        })?;
                out.push(decode_one(&descriptor.dtype, slice)?);
            }
        }
        2 => {
            let (rows, cols) = (descriptor.shape[0], descriptor.shape[1]);
            for row in 0..rows {
                for col in 0..cols {
                    let index =
                        physical_index(&descriptor.shape, &descriptor.storage_order, row, col)?;
                    let start = usize::try_from(index * bytes_per).map_err(|_| {
                        ConvertError::Allocation {
                            bytes: index * bytes_per,
                            reason: "element offset does not fit usize".to_owned(),
                        }
                    })?;
                    let end = start + usize::try_from(bytes_per).unwrap_or(0);
                    let slice =
                        data.get(start..end)
                            .ok_or_else(|| ConvertError::ByteLengthMismatch {
                                tensor: descriptor.key.clone(),
                                extent: data.len() as u64,
                                expected: descriptor.byte_len,
                                elements: descriptor.elements,
                                dtype: descriptor.dtype.stable_name(),
                            })?;
                    out.push(decode_one(&descriptor.dtype, slice)?);
                }
            }
        }
        rank => {
            return Err(ConvertError::RankMismatch {
                tensor: descriptor.key.clone(),
                rank,
            });
        }
    }
    if out.len() as u64 != descriptor.elements {
        return Err(ConvertError::Allocation {
            bytes: descriptor.byte_len,
            reason: "decoded length disagrees with the descriptor".to_owned(),
        });
    }
    Ok(out)
}

/// Compute the physical element index of a logical coordinate.
///
/// # Errors
///
/// Rejects out-of-range coordinates, non-rank-2 tensors, malformed orders and
/// overflow.
pub fn physical_index(shape: &[u64], order: &StorageOrder, row: u64, col: u64) -> Result<u64> {
    if shape.len() != 2 {
        return Err(ConvertError::inconsistent(
            "physical_index requires a rank-2 shape",
        ));
    }
    if row >= shape[0] || col >= shape[1] {
        return Err(ConvertError::inconsistent(format!(
            "coordinate ({row},{col}) is outside {:?}",
            shape
        )));
    }
    let axes = order.physical_axes(2)?;
    // Physical position 0 is the slowest-varying axis, position 1 the fastest.
    let coordinate = [row, col];
    let mut index = 0_u64;
    for (position, logical_axis) in axes.iter().enumerate() {
        let axis = usize::try_from(*logical_axis)
            .map_err(|_| ConvertError::inconsistent("physical axis does not fit usize"))?;
        let dimension = shape[axis];
        let value = coordinate[axis];
        // Later positions are faster, so multiply the accumulated index.
        if position > 0 {
            index = index.checked_mul(dimension).ok_or(ConvertError::Core(
                selvedge_core::CoreError::overflow("physical index"),
            ))?;
        }
        index = index.checked_add(value).ok_or(ConvertError::Core(
            selvedge_core::CoreError::overflow("physical index"),
        ))?;
    }
    Ok(index)
}

fn reorder_logical(shape: &[u64], order: &StorageOrder, values: &[f32]) -> Result<Vec<f32>> {
    if shape.len() != 2 {
        return Err(ConvertError::inconsistent(
            "reorder_logical requires a rank-2 shape",
        ));
    }
    let elements = shape_product("reorder", shape)?;
    if elements != values.len() as u64 {
        return Err(ConvertError::inconsistent(
            "logical value count disagrees with the shape",
        ));
    }
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, elements)?;
    let rows = shape[0];
    let cols = shape[1];
    let mut physical: Vec<f32> = Vec::new();
    reserve_elements(&mut physical, elements)?;
    physical.resize(values.len(), 0.0);
    for row in 0..rows {
        for col in 0..cols {
            let logical =
                usize::try_from(row * cols + col).map_err(|_| ConvertError::Allocation {
                    bytes: elements * 4,
                    reason: "logical index does not fit usize".to_owned(),
                })?;
            let index = physical_index(shape, order, row, col)?;
            let target = usize::try_from(index).map_err(|_| ConvertError::Allocation {
                bytes: elements * 4,
                reason: "physical index does not fit usize".to_owned(),
            })?;
            physical[target] = values[logical];
        }
    }
    out.extend_from_slice(&physical);
    Ok(out)
}

fn encode_values(shape: &[u64], order: &StorageOrder, values: &[f32]) -> Result<Vec<u8>> {
    let ordered = reorder_logical(shape, order, values)?;
    let mut data = Vec::new();
    reserve_bytes(&mut data, ordered.len() as u64 * 4)?;
    for value in ordered {
        data.extend_from_slice(&value.to_le_bytes());
    }
    Ok(data)
}

/// Number of bytes one logical element occupies.
///
/// # Errors
///
/// Rejects unsupported and packed dtypes.
pub fn dtype_bytes(dtype: &DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(4),
        DType::F16 | DType::BF16 => Ok(2),
        DType::I8 | DType::U8 => Ok(1),
        DType::I4Packed { .. } | DType::TernaryPacked { .. } | DType::Unsupported(_) => {
            Err(ConvertError::SourceDTypeUnsupported {
                tensor: "<unknown>".to_owned(),
                dtype: dtype.stable_name(),
            })
        }
    }
}

fn decode_one(dtype: &DType, bytes: &[u8]) -> Result<f32> {
    match (dtype, bytes) {
        (DType::F32, [a, b, c, d]) => Ok(f32::from_le_bytes([*a, *b, *c, *d])),
        (DType::F16, [a, b]) => Ok(decode_f16(u16::from_le_bytes([*a, *b]))),
        (DType::BF16, [a, b]) => Ok(decode_bf16(u16::from_le_bytes([*a, *b]))),
        (DType::I8, [a]) => Ok(f32::from(i8::from_le_bytes([*a]))),
        (DType::U8, [a]) => Ok(f32::from(*a)),
        (dtype, _) => Err(ConvertError::SourceDTypeUnsupported {
            tensor: "<region>".to_owned(),
            dtype: dtype.stable_name(),
        }),
    }
}

/// Decode IEEE-754 binary16 to f32 exactly.
///
/// Every finite binary16 value is exactly representable in binary32, so this is
/// a value-preserving widening. NaN payload bits are not preserved; the result
/// is NaN. Subnormals use `frac * 2^-24`, which is exact in f32.
#[must_use]
pub fn decode_f16(bits: u16) -> f32 {
    let sign = bits & 0x8000;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = u32::from(bits & 0x03ff);
    let magnitude = match exponent {
        0 => (fraction as f32) * pow2_f32(-24),
        31 => {
            if fraction == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => ((1024 + fraction) as f32) * pow2_f32(exponent - 25),
    };
    if sign == 0 { magnitude } else { -magnitude }
}

/// Decode bfloat16 to f32 exactly.
#[must_use]
pub fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encode f32 to binary16 with round-to-nearest, ties-to-even.
///
/// This encoder is used by the synthetic fixture and tests; the conversion
/// pipeline never downsamples weights. Values outside the binary16 range become
/// infinities; NaN stays NaN.
#[must_use]
pub fn encode_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7c00 | payload;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let half_exp = (unbiased + 15) as u32;
        let mut half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) == 1) {
            half_mantissa += 1;
            if half_mantissa == 0x0400 {
                let bumped = half_exp + 1;
                if bumped >= 31 {
                    return sign | 0x7c00;
                }
                return sign | ((bumped as u16) << 10);
            }
        }
        return sign | ((half_exp as u16) << 10) | (half_mantissa as u16);
    }
    if unbiased < -24 {
        return sign;
    }
    // Subnormal: shift the implicit one into the fraction field.
    let shift = -unbiased - 14;
    let full = mantissa | 0x0080_0000;
    let mut half_mantissa = full >> (shift + 13);
    let remainder = full & ((1_u32 << (shift + 13)) - 1);
    if remainder > (1_u32 << (shift + 12))
        || (remainder == (1_u32 << (shift + 12)) && (half_mantissa & 1) == 1)
    {
        half_mantissa += 1;
    }
    sign | (half_mantissa as u16)
}

/// Encode f32 to bfloat16 with round-to-nearest, ties-to-even.
#[must_use]
pub fn encode_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        // Preserve NaN as a quiet NaN.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff_u32 + lsb;
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn pow2_f32(exponent: i32) -> f32 {
    // |exponent| is at most 25 here, so a loop is exact and needs no libm.
    if exponent >= 0 {
        let mut value = 1.0_f32;
        for _ in 0..exponent {
            value *= 2.0;
        }
        value
    } else {
        let mut value = 1.0_f32;
        for _ in 0..(-exponent) {
            value *= 0.5;
        }
        value
    }
}

fn shape_product(key: &str, shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut total = 1_u64;
    for dimension in shape {
        total = total
            .checked_mul(*dimension)
            .ok_or_else(|| ConvertError::InvalidShape {
                tensor: key.to_owned(),
                reason: "dimension product overflows u64".to_owned(),
            })?;
    }
    Ok(total)
}

/// Reserve exactly `elements` f32 slots without aborting on OOM.
///
/// # Errors
///
/// Rejects values that do not fit `usize` and allocation refusal.
pub fn reserve_elements(buffer: &mut Vec<f32>, elements: u64) -> Result<()> {
    let count = usize::try_from(elements).map_err(|_| ConvertError::Allocation {
        bytes: elements * 4,
        reason: "element count does not fit usize".to_owned(),
    })?;
    buffer
        .try_reserve_exact(count)
        .map_err(|error| ConvertError::Allocation {
            bytes: elements * 4,
            reason: error.to_string(),
        })
}

/// Reserve exactly `bytes` bytes without aborting on OOM.
///
/// # Errors
///
/// Rejects values that do not fit `usize` and allocation refusal.
pub fn reserve_bytes(buffer: &mut Vec<u8>, bytes: u64) -> Result<()> {
    let count = usize::try_from(bytes).map_err(|_| ConvertError::Allocation {
        bytes,
        reason: "byte count does not fit usize".to_owned(),
    })?;
    buffer
        .try_reserve_exact(count)
        .map_err(|error| ConvertError::Allocation {
            bytes,
            reason: error.to_string(),
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening_is_exact_for_representable_values() {
        for bits in [
            0x0000_u16, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0x7c00,
        ] {
            let value = decode_f16(bits);
            if bits & 0x7fff == 0 {
                assert_eq!(value, 0.0);
            } else if bits == 0x7c00 {
                assert!(value.is_infinite());
            } else {
                let round_trip = encode_f16(value);
                assert_eq!(round_trip, bits, "f16 bits {bits:#06x}");
            }
        }
        assert!(decode_f16(0x7e00).is_nan());
    }

    #[test]
    fn bf16_round_trip_preserves_exact_values() {
        for value in [0.0_f32, -0.0, 1.0, -2.0, 0.5, 1.5, 0.0625] {
            assert_eq!(decode_bf16(encode_bf16(value)), value);
        }
        assert!(decode_bf16(encode_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_encoder_rounds_ties_to_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 and the next f16 value.
        let halfway = 1.0_f32 + 2.0_f32.powi(-11);
        assert_eq!(halfway.to_bits(), 0x3F80_1000);
        assert_eq!(encode_f16(halfway), 0x3c00);
        let just_over = f32::from_bits(halfway.to_bits() + 1);
        assert_eq!(encode_f16(just_over), 0x3c01);
    }

    #[test]
    fn physical_index_handles_all_rank_two_orders() {
        let shape = [2_u64, 3];
        assert_eq!(
            physical_index(&shape, &StorageOrder::RowMajor, 1, 2).unwrap(),
            5
        );
        assert_eq!(
            physical_index(&shape, &StorageOrder::ColumnMajor, 1, 2).unwrap(),
            5
        );
        assert_eq!(
            physical_index(&shape, &StorageOrder::Transposed { axes: vec![1, 0] }, 1, 2).unwrap(),
            5
        );
    }

    #[test]
    fn memory_budget_tracks_peak_and_refuses_overcommit() {
        let mut budget = MemoryBudget::new(10).unwrap();
        budget.reserve(6).unwrap();
        budget.reserve(4).unwrap();
        assert!(budget.reserve(1).is_err());
        assert_eq!(budget.peak(), 10);
        budget.release(6).unwrap();
        budget.reserve(3).unwrap();
        assert_eq!(budget.peak(), 10);
        assert!(budget.release(100).is_err());
    }
}
