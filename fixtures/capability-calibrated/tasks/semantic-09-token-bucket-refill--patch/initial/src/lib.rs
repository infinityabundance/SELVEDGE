//! Proportional, capped token-bucket refill.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn one_second_refills_the_whole_rate() {
        assert_eq!(refill(0, 1000, 5, 10), 5);
    }
}
