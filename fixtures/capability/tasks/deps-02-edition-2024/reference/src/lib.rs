//! Edition 2024.

pub fn describe(value: &Option<u32>) -> String {
    match value {
        Some(inner) => format!("some {inner}"),
        None => "none".to_string(),
    }
}

pub fn next_value(seed: u64) -> u64 {
    let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    state % 100
}
