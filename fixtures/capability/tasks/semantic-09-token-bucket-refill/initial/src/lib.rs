//! Token-bucket refill, currently whole-second and uncapped.

pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
    let gained = (elapsed_ms / 1000) * rate_per_sec as u64;
    tokens.saturating_add(gained as u32)
}

#[cfg(test)]
mod tests {
    use super::refill;

    #[test]
    fn one_second_refills_the_rate() {
        assert_eq!(refill(0, 1000, 5, 10), 5);
    }
}
