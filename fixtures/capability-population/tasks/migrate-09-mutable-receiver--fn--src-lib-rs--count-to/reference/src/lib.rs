//! Count through the counter crate.

pub fn count_to(limit: u32) -> u32 {
    let mut counter = counter::Counter::new();
    for _ in 0..limit {
        counter.tick();
    }
    counter.value()
}

#[cfg(test)]
mod visible {
    use crate::count_to;

    #[test]
    fn ticks_the_exact_number_of_times() {
        assert_eq!(count_to(3), 3);
    }

    #[test]
    fn zero_ticks_is_zero() {
        assert_eq!(count_to(0), 0);
    }

    #[test]
    fn larger_counts_still_work() {
        assert_eq!(count_to(10), 10);
    }
}
