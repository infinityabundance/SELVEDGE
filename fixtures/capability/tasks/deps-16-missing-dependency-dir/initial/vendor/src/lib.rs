//! Codec, version 2.

pub fn fold(text: &str) -> u32 {
    text.bytes()
        .fold(0u32, |total, byte| total.wrapping_mul(31).wrapping_add(byte as u32))
}
