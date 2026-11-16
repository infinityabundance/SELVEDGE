//! Proportional, capped token-bucket refill.

pub fn refill(tokens: u32, elapsed_ms: u64, rate_per_sec: u32, capacity: u32) -> u32 {
    let gained = (elapsed_ms.saturating_mul(rate_per_sec as u64) / 1000) as u32;
    tokens.saturating_add(gained).min(capacity)
}

#[cfg(test)]
mod visible {
    use crate::refill;

    #[test]
    fn one_second_refills_the_whole_rate() {
        assert_eq!(refill(0, 1000, 5, 10), 5);
    }

    #[test]
    fn sub_second_time_refills_proportionally() {
        assert_eq!(refill(0, 500, 10, 100), 5);
    }

    #[test]
    fn the_capacity_bound_is_respected() {
        assert_eq!(refill(90, 1000, 20, 100), 100);
    }

    #[test]
    fn no_time_gains_nothing() {
        assert_eq!(refill(7, 0, 10, 100), 7);
    }

    #[test]
    fn quarter_second_refills_a_quarter() {
        assert_eq!(refill(0, 250, 8, 100), 2);
    }
}
