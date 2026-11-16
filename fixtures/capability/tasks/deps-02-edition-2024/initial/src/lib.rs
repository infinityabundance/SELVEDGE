//! Written for Rust 2021; this crate declares edition 2024 and no
//! longer compiles.

pub fn describe(value: &Option<u32>) -> String {
    match value {
        Some(ref inner) => format!("some {inner}"),
        None => "none".to_string(),
    }
}

pub fn next_value(seed: u64) -> u64 {
    let gen = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    gen % 100
}
