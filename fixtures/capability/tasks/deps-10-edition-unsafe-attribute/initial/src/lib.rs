//! An exported checksum. This does not build under edition 2024.

#[no_mangle]
pub extern "C" fn checksum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |total, byte| total.wrapping_add(*byte))
}
