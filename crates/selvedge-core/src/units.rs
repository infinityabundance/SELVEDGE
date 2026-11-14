//! Overflow-safe size arithmetic shared by the core and format crates.

use crate::error::{CoreError, Result};
use crate::tensor::{DType, ElementCount};

/// Checked `u64` addition with a named operation.
///
/// # Errors
///
/// Rejects overflow.
pub fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| CoreError::overflow(operation))
}

/// Checked `u64` multiplication with a named operation.
///
/// # Errors
///
/// Rejects overflow.
pub fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64> {
    left.checked_mul(right)
        .ok_or_else(|| CoreError::overflow(operation))
}

/// Ceiling division.
///
/// # Errors
///
/// Rejects a zero divisor and the `u64::MAX` addition overflow.
pub fn div_ceil(value: u64, divisor: u64) -> Result<u64> {
    if divisor == 0 {
        return Err(CoreError::invalid("div_ceil", "divisor is zero"));
    }
    let adjusted = value
        .checked_add(divisor - 1)
        .ok_or_else(|| CoreError::overflow("ceiling division"))?;
    Ok(adjusted / divisor)
}

/// Round `value` up to the next multiple of `alignment`.
///
/// # Errors
///
/// Rejects a zero alignment and rounding that would overflow.
pub fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(CoreError::invalid("alignment", "alignment is zero"));
    }
    checked_mul(div_ceil(value, alignment)?, alignment, "alignment rounding")
}

/// Product of a shape's dimensions, rejecting empty and zero shapes.
///
/// # Errors
///
/// Rejects an empty shape, a zero dimension, and overflow.
pub fn element_count(shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Err(CoreError::invalid("shape", "shape is empty"));
    }
    let mut total = ElementCount::new(1);
    for dimension in shape {
        if *dimension == 0 {
            return Err(CoreError::invalid(
                "shape",
                "shape contains a zero dimension",
            ));
        }
        total = total.checked_mul(ElementCount::new(*dimension))?;
    }
    Ok(total.get())
}

/// Bytes required for `elements` logical values of `dtype`.
///
/// Packed types are block-aligned: the element count must be a whole number of
/// `block_size` blocks. No padding or metadata is included; scale/zero-point
/// bytes are accounted separately.
///
/// # Errors
///
/// Rejects unsupported dtypes, non-block-aligned packed counts, malformed block
/// sizes and overflow.
pub fn bytes_for(elements: u64, dtype: &DType) -> Result<u64> {
    dtype.require_supported()?;
    if elements == 0 {
        return Err(CoreError::invalid(
            "bytes_for",
            "element count must be non-zero",
        ));
    }
    match dtype {
        DType::F32 => checked_mul(elements, 4, "f32 byte count"),
        DType::F16 | DType::BF16 => checked_mul(elements, 2, "f16 byte count"),
        DType::I8 | DType::U8 => Ok(elements),
        DType::I4Packed { block_size } => packed_bytes(elements, *block_size, 2, "i4"),
        DType::TernaryPacked { block_size } => packed_bytes(elements, *block_size, 4, "ternary"),
        DType::Unsupported(raw) => Err(CoreError::UnsupportedDType { dtype: raw.clone() }),
    }
}

fn packed_bytes(elements: u64, block_size: u32, elements_per_byte: u64, name: &str) -> Result<u64> {
    if block_size == 0 {
        return Err(CoreError::invalid(
            "bytes_for",
            format!("{name} block size is zero"),
        ));
    }
    let block = u64::from(block_size);
    if !elements.is_multiple_of(block) {
        return Err(CoreError::invalid(
            "bytes_for",
            format!("{elements} elements are not a whole number of {name} blocks of {block}"),
        ));
    }
    let bytes_per_block = block / elements_per_byte;
    checked_mul(elements / block, bytes_per_block, "packed byte count")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rounds_up() {
        assert_eq!(align_up(0, 4096).unwrap(), 0);
        assert_eq!(align_up(1, 4096).unwrap(), 4096);
        assert_eq!(align_up(4096, 4096).unwrap(), 4096);
        assert_eq!(align_up(4097, 4096).unwrap(), 8192);
        assert!(align_up(1, 0).is_err());
    }

    #[test]
    fn element_counts_are_checked() {
        assert_eq!(element_count(&[8, 16]).unwrap(), 128);
        assert!(element_count(&[]).is_err());
        assert!(element_count(&[0, 4]).is_err());
        assert!(element_count(&[u64::MAX, 2]).is_err());
    }

    #[test]
    fn dtype_byte_counts() {
        assert_eq!(bytes_for(100, &DType::F32).unwrap(), 400);
        assert_eq!(bytes_for(100, &DType::F16).unwrap(), 200);
        assert_eq!(bytes_for(100, &DType::BF16).unwrap(), 200);
        assert_eq!(bytes_for(100, &DType::I8).unwrap(), 100);
        assert_eq!(
            bytes_for(128, &DType::I4Packed { block_size: 32 }).unwrap(),
            64
        );
        assert_eq!(
            bytes_for(128, &DType::TernaryPacked { block_size: 64 }).unwrap(),
            32
        );
        assert!(bytes_for(100, &DType::I4Packed { block_size: 32 }).is_err());
        assert!(bytes_for(0, &DType::F32).is_err());
        assert!(bytes_for(1, &DType::Unsupported("q4_k".to_owned())).is_err());
    }

    #[test]
    fn addition_and_multiplication_report_overflow() {
        assert_eq!(checked_add(1, 2, "test").unwrap(), 3);
        assert!(checked_add(u64::MAX, 1, "test").is_err());
        assert!(checked_mul(u64::MAX, 2, "test").is_err());
        assert!(div_ceil(10, 3).unwrap() == 4);
        assert!(div_ceil(10, 0).is_err());
    }
}
