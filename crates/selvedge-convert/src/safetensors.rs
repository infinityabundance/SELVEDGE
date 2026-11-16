//! A safe, streaming reader for the safetensors checkpoint format.
//!
//! The reader never maps the file and never performs an unchecked cast. It reads
//! the 8-byte header length, bounds it against a hard cap and the physical file,
//! parses the JSON header with duplicate-key detection, validates every
//! `{dtype, shape, data_offsets}` triple against the data section, and then
//! serves per-tensor positional reads through [`std::os::unix::fs::FileExt::read_at`].
//!
//! Supported dtypes are `F32`, `F16`, `BF16`, `I8` and `U8`. Every other dtype,
//! including `F64` and packed four-bit types, is rejected with a typed error and
//! no fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use selvedge_core::{DType, StorageOrder};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::error::{ConvertError, Result};
use crate::tensor_source::{
    TensorDescriptor, TensorSource, decode_bf16, decode_f16, dtype_bytes, read_region_from_bytes,
    reserve_bytes, reserve_elements,
};

/// Hard cap on the JSON header length, independent of configuration.
pub const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
/// Hard cap on the number of tensors in one header.
pub const MAX_TENSORS: u64 = 1_000_000;
/// Hard cap on a tensor name length.
pub const MAX_TENSOR_NAME_BYTES: usize = 4096;
/// Hard cap on a tensor rank.
pub const MAX_RANK: usize = 16;
/// Hard cap on files in one checkpoint directory.
pub const MAX_CHECKPOINT_FILES: usize = 100_000;
/// Hard cap on tensor keys across a multi-file checkpoint.
pub const MAX_TOTAL_TENSORS: u64 = 10_000_000;

/// Positional reading without `unsafe` and without mapping the file.
pub trait PositionalRead {
    /// Source length in bytes.
    ///
    /// # Errors
    ///
    /// Propagates metadata failures.
    fn byte_len(&self) -> Result<u64>;

    /// Read exactly `buf.len()` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// Rejects offsets outside the source and short reads.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()>;
}

impl PositionalRead for std::fs::File {
    fn byte_len(&self) -> Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let mut read = 0_usize;
        while read < buf.len() {
            let absolute = offset
                .checked_add(read as u64)
                .ok_or(ConvertError::ShortRead {
                    offset,
                    wanted: buf.len() as u64,
                    got: read as u64,
                })?;
            let count = self.read_at(&mut buf[read..], absolute)?;
            if count == 0 {
                return Err(ConvertError::ShortRead {
                    offset,
                    wanted: buf.len() as u64,
                    got: read as u64,
                });
            }
            read += count;
        }
        Ok(())
    }
}

impl PositionalRead for &[u8] {
    fn byte_len(&self) -> Result<u64> {
        Ok(self.len() as u64)
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        let start = usize::try_from(offset).map_err(|_| ConvertError::ShortRead {
            offset,
            wanted: buf.len() as u64,
            got: 0,
        })?;
        let end = start
            .checked_add(buf.len())
            .ok_or(ConvertError::ShortRead {
                offset,
                wanted: buf.len() as u64,
                got: 0,
            })?;
        let slice = self.get(start..end).ok_or(ConvertError::ShortRead {
            offset,
            wanted: buf.len() as u64,
            got: 0,
        })?;
        buf.copy_from_slice(slice);
        Ok(())
    }
}

/// One tensor declared by the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor key.
    pub key: String,
    /// Dtype as decoded by this build.
    pub dtype: DType,
    /// Original safetensors dtype string.
    pub dtype_name: String,
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Number of logical elements.
    pub elements: u64,
    /// Declared `data_offsets[0]`, relative to the data section.
    pub begin: u64,
    /// Declared `data_offsets[1]`, relative to the data section.
    pub end: u64,
}

impl TensorEntry {
    /// Descriptor used by the conversion layer. Safetensors is row-major.
    #[must_use]
    pub fn descriptor(&self) -> TensorDescriptor {
        TensorDescriptor {
            key: self.key.clone(),
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            storage_order: StorageOrder::RowMajor,
            elements: self.elements,
            byte_len: self.end - self.begin,
        }
    }
}

/// Parsed header plus the derived section length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderIndex {
    /// `__metadata__` string map, if present.
    pub metadata: BTreeMap<String, String>,
    /// Tensors sorted by key.
    pub tensors: BTreeMap<String, TensorEntry>,
    /// Header JSON length in bytes.
    pub header_len: u64,
    /// Data section length in bytes.
    pub data_len: u64,
}

impl HeaderIndex {
    /// Data-section offset of a tensor.
    ///
    /// # Errors
    ///
    /// Rejects unknown keys.
    pub fn tensor_range(&self, key: &str) -> Result<Range<u64>> {
        self.tensors
            .get(key)
            .map(|entry| entry.begin..entry.end)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorInfoWire {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct TopLevelVisitor;

impl<'de> Visitor<'de> for TopLevelVisitor {
    type Value = Vec<(String, serde_json::Value)>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors JSON header object")
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> core::result::Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate header key {key:?}"
                )));
            }
            let value: serde_json::Value = map.next_value()?;
            entries.push((key, value));
        }
        Ok(entries)
    }
}

fn parse_header_json(header: &[u8], data_len: u64) -> Result<HeaderIndex> {
    let text = std::str::from_utf8(header).map_err(|_| ConvertError::HeaderNotUtf8)?;
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let entries = deserializer
        .deserialize_map(TopLevelVisitor)
        .map_err(|error| ConvertError::malformed(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| ConvertError::malformed(error.to_string()))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    for (key, value) in entries {
        if key == "__metadata__" {
            let object = value.as_object().ok_or_else(|| {
                ConvertError::malformed("__metadata__ must be a JSON object of strings")
            })?;
            for (name, item) in object {
                let text = item.as_str().ok_or_else(|| {
                    ConvertError::malformed(format!(
                        "__metadata__ value for {name:?} is not a string"
                    ))
                })?;
                metadata.insert(name.clone(), text.to_owned());
            }
            continue;
        }
        if key.is_empty() || key.len() > MAX_TENSOR_NAME_BYTES {
            return Err(ConvertError::InvalidTensorName {
                cap: MAX_TENSOR_NAME_BYTES,
            });
        }
        if tensors.len() as u64 >= MAX_TENSORS {
            return Err(ConvertError::TooManyTensors {
                count: MAX_TENSORS + 1,
                cap: MAX_TENSORS,
            });
        }
        let wire: TensorInfoWire = serde_json::from_value(value)
            .map_err(|error| ConvertError::malformed(format!("tensor {key:?}: {error}")))?;
        let dtype = parse_dtype(&key, &wire.dtype)?;
        let elements = element_count(&key, &wire.shape)?;
        let per_element =
            u64::from(
                dtype_bytes(&dtype).map_err(|_| ConvertError::UnsupportedDType {
                    tensor: key.clone(),
                    dtype: wire.dtype.clone(),
                })?,
            );
        let expected =
            elements
                .checked_mul(per_element)
                .ok_or_else(|| ConvertError::InvalidShape {
                    tensor: key.clone(),
                    reason: "byte length overflows u64".to_owned(),
                })?;
        let [begin, end] = wire.data_offsets;
        if begin > end {
            return Err(ConvertError::OffsetOutOfBounds {
                tensor: key.clone(),
                begin,
                end,
                data_len,
            });
        }
        let extent = end - begin;
        if extent != expected {
            return Err(ConvertError::ByteLengthMismatch {
                tensor: key.clone(),
                extent,
                expected,
                elements,
                dtype: dtype.stable_name(),
            });
        }
        if end > data_len {
            return Err(ConvertError::OffsetOutOfBounds {
                tensor: key.clone(),
                begin,
                end,
                data_len,
            });
        }
        tensors.insert(
            key.clone(),
            TensorEntry {
                key,
                dtype,
                dtype_name: wire.dtype,
                shape: wire.shape,
                elements,
                begin,
                end,
            },
        );
    }
    validate_no_overlap(&tensors)?;
    Ok(HeaderIndex {
        metadata,
        tensors,
        header_len: header.len() as u64,
        data_len,
    })
}

fn parse_dtype(tensor: &str, name: &str) -> Result<DType> {
    match name {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I8" => Ok(DType::I8),
        "U8" => Ok(DType::U8),
        other => Err(ConvertError::UnsupportedDType {
            tensor: tensor.to_owned(),
            dtype: other.to_owned(),
        }),
    }
}

fn element_count(tensor: &str, shape: &[u64]) -> Result<u64> {
    if shape.len() > MAX_RANK {
        return Err(ConvertError::InvalidShape {
            tensor: tensor.to_owned(),
            reason: format!("rank {} exceeds the cap of {MAX_RANK}", shape.len()),
        });
    }
    let mut elements = 1_u64;
    for dimension in shape {
        elements = elements
            .checked_mul(*dimension)
            .ok_or_else(|| ConvertError::InvalidShape {
                tensor: tensor.to_owned(),
                reason: "dimension product overflows u64".to_owned(),
            })?;
    }
    Ok(elements)
}

fn validate_no_overlap(tensors: &BTreeMap<String, TensorEntry>) -> Result<()> {
    let mut extents: Vec<(&str, u64, u64)> = tensors
        .values()
        .filter(|entry| entry.begin != entry.end)
        .map(|entry| (entry.key.as_str(), entry.begin, entry.end))
        .collect();
    extents.sort_unstable_by_key(|(_, begin, _)| *begin);
    for pair in extents.windows(2) {
        if pair[0].2 > pair[1].1 {
            return Err(ConvertError::OverlappingExtents {
                first: pair[0].0.to_owned(),
                second: pair[1].0.to_owned(),
            });
        }
    }
    Ok(())
}

/// Parse a complete safetensors byte image held in memory.
///
/// This is the same parser the file reader uses, exposed so property tests can
/// exercise hostile inputs without touching the filesystem.
///
/// # Errors
///
/// Rejects truncated images, oversized or misaligned headers, non-UTF-8 or
/// malformed JSON, duplicate keys, unsupported dtypes, absurd shapes, reversed or
/// out-of-bounds offsets, overlapping extents and shape/byte mismatches.
pub fn parse_bytes(bytes: &[u8]) -> Result<HeaderIndex> {
    let file_len = bytes.len() as u64;
    if file_len < 8 {
        return Err(ConvertError::HeaderOutOfBounds { len: 8, file_len });
    }
    let mut length_bytes = [0_u8; 8];
    length_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(length_bytes);
    let data_len = validate_header_length(header_len, file_len)?;
    let start = usize::try_from(8_u64).map_err(|_| ConvertError::HeaderTooLarge {
        len: header_len,
        cap: MAX_HEADER_BYTES,
    })?;
    let end = usize::try_from(8_u64 + header_len).map_err(|_| ConvertError::HeaderTooLarge {
        len: header_len,
        cap: MAX_HEADER_BYTES,
    })?;
    let header = bytes
        .get(start..end)
        .ok_or(ConvertError::HeaderOutOfBounds {
            len: header_len,
            file_len,
        })?;
    parse_header_json(header, data_len)
}

fn validate_header_length(header_len: u64, file_len: u64) -> Result<u64> {
    if header_len > MAX_HEADER_BYTES {
        return Err(ConvertError::HeaderTooLarge {
            len: header_len,
            cap: MAX_HEADER_BYTES,
        });
    }
    if !header_len.is_multiple_of(8) {
        return Err(ConvertError::HeaderMisaligned { len: header_len });
    }
    let data_len = file_len
        .checked_sub(8)
        .and_then(|rest| rest.checked_sub(header_len))
        .ok_or(ConvertError::HeaderOutOfBounds {
            len: header_len,
            file_len,
        })?;
    Ok(data_len)
}

/// A safetensors image over any [`PositionalRead`] source.
#[derive(Debug)]
pub struct Safetensors<R> {
    source: R,
    index: HeaderIndex,
    data_start: u64,
}

impl Safetensors<std::fs::File> {
    /// Open and validate a safetensors file.
    ///
    /// # Errors
    ///
    /// Rejects unreadable files and every structural failure described on
    /// [`parse_bytes`].
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_source(file)
    }
}

impl<R: PositionalRead> Safetensors<R> {
    /// Validate an arbitrary positional source.
    ///
    /// # Errors
    ///
    /// Same failures as [`parse_bytes`].
    pub fn from_source(source: R) -> Result<Self> {
        let file_len = source.byte_len()?;
        if file_len < 8 {
            return Err(ConvertError::HeaderOutOfBounds { len: 8, file_len });
        }
        let mut length_bytes = [0_u8; 8];
        source.read_exact_at(&mut length_bytes, 0)?;
        let header_len = u64::from_le_bytes(length_bytes);
        let data_len = validate_header_length(header_len, file_len)?;
        let mut header = Vec::new();
        reserve_bytes(&mut header, header_len)?;
        let header_usize =
            usize::try_from(header_len).map_err(|_| ConvertError::HeaderTooLarge {
                len: header_len,
                cap: MAX_HEADER_BYTES,
            })?;
        header.resize(header_usize, 0);
        source.read_exact_at(&mut header, 8)?;
        let index = parse_header_json(&header, data_len)?;
        let data_start = header_len
            .checked_add(8)
            .ok_or(ConvertError::HeaderTooLarge {
                len: header_len,
                cap: MAX_HEADER_BYTES,
            })?;
        Ok(Self {
            source,
            index,
            data_start,
        })
    }

    /// Parsed header.
    #[must_use]
    pub const fn header(&self) -> &HeaderIndex {
        &self.index
    }

    /// Data-section start offset in the source.
    #[must_use]
    pub const fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Read one tensor's full bytes after validating its extent.
    ///
    /// # Errors
    ///
    /// Rejects unknown keys, extents above `max_bytes`, allocation refusal and
    /// short reads.
    pub fn read_tensor_bytes(&self, key: &str, max_bytes: u64) -> Result<Vec<u8>> {
        let entry = self
            .index
            .tensors
            .get(key)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        let length = entry.end - entry.begin;
        if length > max_bytes {
            return Err(ConvertError::Allocation {
                bytes: length,
                reason: format!("tensor {key:?} exceeds the {max_bytes}-byte read cap"),
            });
        }
        let mut bytes = Vec::new();
        reserve_bytes(&mut bytes, length)?;
        let length = usize::try_from(length).map_err(|_| ConvertError::Allocation {
            bytes: length,
            reason: "tensor length does not fit usize".to_owned(),
        })?;
        bytes.resize(length, 0);
        let offset = self
            .data_start
            .checked_add(entry.begin)
            .ok_or(ConvertError::ShortRead {
                offset: self.data_start,
                wanted: length as u64,
                got: 0,
            })?;
        self.source.read_exact_at(&mut bytes, offset)?;
        Ok(bytes)
    }

    /// Read a whole tensor as f32 in logical row-major order.
    ///
    /// Unlike [`TensorSource::read_region_f32`] this accepts rank-1 tensors,
    /// which is what a normalization gain is, and it decodes a transposed
    /// rank-2 tensor through its storage order instead of reinterpreting bytes.
    ///
    /// # Errors
    ///
    /// Rejects an unknown key, a rank above two, an unsupported dtype, an
    /// extent above `max_bytes` and short reads.
    pub fn read_tensor_f32(&self, key: &str, max_bytes: u64) -> Result<Vec<f32>> {
        let descriptor = self
            .index
            .tensors
            .get(key)
            .map(TensorEntry::descriptor)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        let bytes = self.read_tensor_bytes(key, max_bytes)?;
        crate::tensor_source::decode_tensor_f32(&descriptor, &bytes)
    }
}

impl<R: PositionalRead> TensorSource for Safetensors<R> {
    fn tensor_keys(&self) -> Vec<String> {
        self.index.tensors.keys().cloned().collect()
    }

    fn descriptor(&self, key: &str) -> Result<TensorDescriptor> {
        self.index
            .tensors
            .get(key)
            .map(TensorEntry::descriptor)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))
    }

    fn read_region_f32(
        &mut self,
        key: &str,
        rows: Range<u64>,
        cols: Range<u64>,
    ) -> Result<Vec<f32>> {
        let entry = self
            .index
            .tensors
            .get(key)
            .cloned()
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        let descriptor = entry.descriptor();
        if descriptor.rank() != 2 {
            return Err(ConvertError::RankMismatch {
                tensor: key.to_owned(),
                rank: descriptor.rank(),
            });
        }
        let (height, width) = (descriptor.shape[0], descriptor.shape[1]);
        if rows.start >= rows.end || cols.start >= cols.end || rows.end > height || cols.end > width
        {
            return Err(ConvertError::RegionOutOfBounds {
                tensor: key.to_owned(),
                rows_start: rows.start,
                rows_end: rows.end,
                cols_start: cols.start,
                cols_end: cols.end,
            });
        }
        let per_element = u64::from(dtype_bytes(&descriptor.dtype).map_err(|_| {
            ConvertError::UnsupportedDType {
                tensor: key.to_owned(),
                dtype: entry.dtype_name.clone(),
            }
        })?);
        let row_values = cols.end - cols.start;
        let out_elements = (rows.end - rows.start)
            .checked_mul(row_values)
            .ok_or_else(|| ConvertError::InvalidShape {
                tensor: key.to_owned(),
                reason: "region element count overflows".to_owned(),
            })?;
        let mut out: Vec<f32> = Vec::new();
        reserve_elements(&mut out, out_elements)?;
        let row_bytes =
            row_values
                .checked_mul(per_element)
                .ok_or_else(|| ConvertError::InvalidShape {
                    tensor: key.to_owned(),
                    reason: "row byte length overflows".to_owned(),
                })?;
        let mut scratch = Vec::new();
        reserve_bytes(&mut scratch, row_bytes)?;
        let scratch_len = usize::try_from(row_bytes).map_err(|_| ConvertError::Allocation {
            bytes: row_bytes,
            reason: "row length does not fit usize".to_owned(),
        })?;
        scratch.resize(scratch_len, 0);
        for row in rows {
            let element_offset = row
                .checked_mul(width)
                .and_then(|value| value.checked_add(cols.start))
                .ok_or_else(|| ConvertError::InvalidShape {
                    tensor: key.to_owned(),
                    reason: "row offset overflows".to_owned(),
                })?;
            let byte_offset = element_offset.checked_mul(per_element).ok_or_else(|| {
                ConvertError::InvalidShape {
                    tensor: key.to_owned(),
                    reason: "byte offset overflows".to_owned(),
                }
            })?;
            let absolute = self
                .data_start
                .checked_add(entry.begin)
                .and_then(|value| value.checked_add(byte_offset))
                .ok_or(ConvertError::ShortRead {
                    offset: self.data_start,
                    wanted: row_bytes,
                    got: 0,
                })?;
            self.source.read_exact_at(&mut scratch, absolute)?;
            decode_row(&descriptor.dtype, &scratch, &mut out)?;
        }
        Ok(out)
    }
}

fn decode_row(dtype: &DType, bytes: &[u8], out: &mut Vec<f32>) -> Result<()> {
    match dtype {
        DType::F32 => {
            for chunk in bytes.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
        }
        DType::F16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                out.push(decode_f16(u16::from_le_bytes(*chunk)));
            }
        }
        DType::BF16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                out.push(decode_bf16(u16::from_le_bytes(*chunk)));
            }
        }
        DType::I8 => {
            for byte in bytes {
                out.push(f32::from(i8::from_le_bytes([*byte])));
            }
        }
        DType::U8 => {
            for byte in bytes {
                out.push(f32::from(*byte));
            }
        }
        DType::I4Packed { .. } | DType::TernaryPacked { .. } | DType::Unsupported(_) => {
            return Err(ConvertError::SourceDTypeUnsupported {
                tensor: "<row>".to_owned(),
                dtype: dtype.stable_name(),
            });
        }
    }
    Ok(())
}

/// A checkpoint split over several safetensors files.
#[derive(Debug)]
pub struct MultiSafetensors {
    files: Vec<(String, Safetensors<std::fs::File>)>,
    index: BTreeMap<String, usize>,
}

impl MultiSafetensors {
    /// Open an ordered list of `(normalized relative path, filesystem path)`.
    ///
    /// # Errors
    ///
    /// Rejects duplicate tensor keys across files and every per-file failure.
    pub fn open(files: Vec<(String, PathBuf)>) -> Result<Self> {
        if files.len() > MAX_CHECKPOINT_FILES {
            return Err(ConvertError::TooManyFiles {
                count: files.len(),
                cap: MAX_CHECKPOINT_FILES,
            });
        }
        let mut opened = Vec::with_capacity(files.len());
        let mut index = BTreeMap::new();
        let mut total_tensors = 0_u64;
        for (relative, path) in files {
            let store = Safetensors::open(&path)?;
            total_tensors = total_tensors.saturating_add(store.index.tensors.len() as u64);
            if total_tensors > MAX_TOTAL_TENSORS {
                return Err(ConvertError::TooManyTensors {
                    count: total_tensors,
                    cap: MAX_TOTAL_TENSORS,
                });
            }
            for key in store.index.tensors.keys() {
                if index.insert(key.clone(), opened.len()).is_some() {
                    return Err(ConvertError::DuplicateKey { key: key.clone() });
                }
            }
            opened.push((relative, store));
        }
        Ok(Self {
            files: opened,
            index,
        })
    }

    /// Normalized file paths in the order they were opened.
    #[must_use]
    pub fn relative_paths(&self) -> Vec<&str> {
        self.files.iter().map(|(path, _)| path.as_str()).collect()
    }

    /// Number of files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether no files were opened.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Read a whole tensor as f32 in logical row-major order.
    ///
    /// Unlike [`TensorSource::read_region_f32`] this accepts rank-1 tensors,
    /// which is what a normalization gain is.
    ///
    /// # Errors
    ///
    /// Rejects an unknown key, a rank above two, an unsupported dtype and short
    /// reads.
    pub fn read_tensor_f32(&self, key: &str, max_bytes: u64) -> Result<Vec<f32>> {
        let file = self
            .index
            .get(key)
            .copied()
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        let (_, opened) = self
            .files
            .get(file)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        opened.read_tensor_f32(key, max_bytes)
    }
}

impl TensorSource for MultiSafetensors {
    fn tensor_keys(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn descriptor(&self, key: &str) -> Result<TensorDescriptor> {
        let file = self
            .index
            .get(key)
            .copied()
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        self.files
            .get(file)
            .and_then(|(_, opened)| opened.index.tensors.get(key))
            .map(TensorEntry::descriptor)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))
    }

    fn read_region_f32(
        &mut self,
        key: &str,
        rows: Range<u64>,
        cols: Range<u64>,
    ) -> Result<Vec<f32>> {
        let file = self
            .index
            .get(key)
            .copied()
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
        self.files
            .get_mut(file)
            .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?
            .1
            .read_region_f32(key, rows, cols)
    }
}

/// A source that owns precomputed descriptors and dispatches reads.
///
/// The conversion layer needs `descriptor(&self) -> &TensorDescriptor`, so this
/// wrapper caches every descriptor once instead of rebuilding it per call.
#[derive(Debug)]
pub enum CheckpointSource {
    /// One safetensors file.
    Single(Safetensors<std::fs::File>),
    /// Several safetensors files.
    Multi(MultiSafetensors),
}

impl CheckpointSource {
    /// Open a file or a directory of `.safetensors` files.
    ///
    /// # Errors
    ///
    /// Rejects unknown paths, empty directories, duplicate keys and every file
    /// parse failure.
    pub fn open(path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(path)?;
        if metadata.is_file() {
            return Ok(Self::Single(Safetensors::open(path)?));
        }
        if !metadata.is_dir() {
            return Err(ConvertError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is neither a file nor a directory", path.display()),
            )));
        }
        let files = collect_safetensors(path)?;
        if files.is_empty() {
            return Err(ConvertError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no .safetensors files under {}", path.display()),
            )));
        }
        Ok(Self::Multi(MultiSafetensors::open(files)?))
    }

    /// Sorted keys.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        match self {
            Self::Single(store) => store.index.tensors.keys().cloned().collect(),
            Self::Multi(store) => store.index.keys().cloned().collect(),
        }
    }

    /// Read a whole tensor as f32 in logical row-major order.
    ///
    /// # Errors
    ///
    /// Rejects an unknown key, a rank above two, an unsupported dtype and short
    /// reads.
    pub fn read_tensor_f32(&self, key: &str, max_bytes: u64) -> Result<Vec<f32>> {
        match self {
            Self::Single(store) => store.read_tensor_f32(key, max_bytes),
            Self::Multi(store) => store.read_tensor_f32(key, max_bytes),
        }
    }

    /// Normalized relative paths of the source files.
    #[must_use]
    pub fn relative_paths(&self) -> Vec<String> {
        match self {
            Self::Single(_) => Vec::new(),
            Self::Multi(store) => store
                .relative_paths()
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }
    }
}

impl TensorSource for CheckpointSource {
    fn tensor_keys(&self) -> Vec<String> {
        self.keys()
    }

    fn descriptor(&self, key: &str) -> Result<TensorDescriptor> {
        match self {
            Self::Single(store) => store
                .index
                .tensors
                .get(key)
                .map(TensorEntry::descriptor)
                .ok_or_else(|| ConvertError::MissingTensor(key.to_owned())),
            Self::Multi(store) => {
                let file = store
                    .index
                    .get(key)
                    .copied()
                    .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
                store
                    .files
                    .get(file)
                    .and_then(|(_, opened)| opened.index.tensors.get(key))
                    .map(TensorEntry::descriptor)
                    .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))
            }
        }
    }

    fn read_region_f32(
        &mut self,
        key: &str,
        rows: Range<u64>,
        cols: Range<u64>,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Single(store) => store.read_region_f32(key, rows, cols),
            Self::Multi(store) => store.read_region_f32(key, rows, cols),
        }
    }
}

fn collect_safetensors(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    collect_into(root, root, 0, &mut out)?;
    out.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(out)
}

fn collect_into(
    root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    if depth > 64 {
        return Err(ConvertError::plan(
            "checkpoint directory nesting exceeds 64 levels",
        ));
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<core::result::Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_into(root, &entry.path(), depth + 1, out)?;
        } else if file_type.is_file() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let relative = path.strip_prefix(root).map_err(|_| {
                    ConvertError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("{} is outside {}", path.display(), root.display()),
                    ))
                })?;
                let relative = relative
                    .components()
                    .map(|component| component.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((relative, path));
                if out.len() > MAX_CHECKPOINT_FILES {
                    return Err(ConvertError::TooManyFiles {
                        count: out.len(),
                        cap: MAX_CHECKPOINT_FILES,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Read a rank-2 region from an in-memory safetensors image.
///
/// This is a convenience shared by the proptest harness; the file reader uses
/// the same decoding path.
///
/// # Errors
///
/// Propagates parse and region failures.
pub fn read_region_in_memory(
    bytes: &[u8],
    key: &str,
    rows: Range<u64>,
    cols: Range<u64>,
) -> Result<Vec<f32>> {
    let index = parse_bytes(bytes)?;
    let entry = index
        .tensors
        .get(key)
        .ok_or_else(|| ConvertError::MissingTensor(key.to_owned()))?;
    let descriptor = entry.descriptor();
    let start = usize::try_from(entry.begin).map_err(|_| ConvertError::Allocation {
        bytes: entry.begin,
        reason: "data offset does not fit usize".to_owned(),
    })?;
    let end = usize::try_from(entry.end).map_err(|_| ConvertError::Allocation {
        bytes: entry.end,
        reason: "data end does not fit usize".to_owned(),
    })?;
    let header_end =
        8 + usize::try_from(index.header_len).map_err(|_| ConvertError::Allocation {
            bytes: index.header_len,
            reason: "header length does not fit usize".to_owned(),
        })?;
    let data = bytes
        .get(header_end + start..header_end + end)
        .ok_or_else(|| ConvertError::OffsetOutOfBounds {
            tensor: key.to_owned(),
            begin: entry.begin,
            end: entry.end,
            data_len: index.data_len,
        })?;
    read_region_from_bytes(&descriptor, data, rows, cols)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn header_json(body: &str) -> Vec<u8> {
        let bytes = body.as_bytes();
        let padded = bytes.len().div_ceil(8) * 8;
        let mut out = Vec::new();
        out.extend_from_slice(&(padded as u64).to_le_bytes());
        out.extend_from_slice(bytes);
        out.resize(8 + padded, b' ');
        out
    }

    #[test]
    fn parses_a_minimal_header_and_rejects_duplicates() {
        let image = {
            let mut bytes =
                header_json(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            bytes
        };
        let index = parse_bytes(&image).unwrap();
        assert_eq!(index.tensors.len(), 1);
        assert_eq!(index.data_len, 4);

        let duplicate = header_json(
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );
        assert!(matches!(
            parse_bytes(&duplicate),
            Err(ConvertError::HeaderMalformed { .. })
        ));
    }

    #[test]
    fn unsupported_dtypes_are_rejected() {
        for dtype in ["F64", "F4", "I4", "BOOL", "F8_E4M3"] {
            let body = format!(r#"{{"a":{{"dtype":"{dtype}","shape":[1],"data_offsets":[0,0]}}}}"#);
            let image = header_json(&body);
            assert!(matches!(
                parse_bytes(&image),
                Err(ConvertError::UnsupportedDType { .. })
            ));
        }
    }

    #[test]
    fn zero_size_tensors_are_accepted_and_overlaps_rejected() {
        let image = header_json(
            r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
        );
        let index = parse_bytes(&image).unwrap();
        assert_eq!(index.tensors.len(), 2);

        let overlapping = header_json(
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        );
        assert!(matches!(
            parse_bytes(&overlapping),
            Err(ConvertError::OffsetOutOfBounds { .. } | ConvertError::OverlappingExtents { .. })
        ));
    }

    #[test]
    fn byte_length_mismatch_is_rejected() {
        let image = header_json(r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#);
        assert!(matches!(
            parse_bytes(&image),
            Err(ConvertError::ByteLengthMismatch { .. })
        ));
    }

    #[test]
    fn non_utf8_and_misaligned_headers_are_rejected() {
        let mut image = 8_u64.to_le_bytes().to_vec();
        image.extend_from_slice(&[0xff_u8; 8]);
        assert!(matches!(
            parse_bytes(&image),
            Err(ConvertError::HeaderNotUtf8 | ConvertError::HeaderMalformed { .. })
        ));

        let mut image = 9_u64.to_le_bytes().to_vec();
        image.extend_from_slice(&[b' '; 9]);
        assert!(matches!(
            parse_bytes(&image),
            Err(ConvertError::HeaderMisaligned { .. })
        ));
    }
}
