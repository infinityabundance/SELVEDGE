//! Proportional, capped token-bucket refill.

pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn one_second_refills_the_whole_rate() {
        assert_eq!(refill(0, 1000, 5, 10), 5);
    }
}
