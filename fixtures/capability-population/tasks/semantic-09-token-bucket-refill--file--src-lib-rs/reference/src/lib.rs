//! Proportional, capped token-bucket refill.

pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
    let gained = (elapsed_ms.saturating_mul(rate_per_sec as u64) / 1000) as u32;
    tokens.saturating_add(gained).min(capacity)
}
