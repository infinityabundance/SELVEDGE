//! A sequence with an associated item type.

pub trait Sequence {
    type Item;
    fn next_item(&mut self) -> Option<Self::Item>;
}

pub struct Counter {
    pub current: u32,
}

impl Sequence for Counter {
    type Item = u64;

    fn next_item(&mut self) -> Option<Self::Item> {
        self.current += 1;
        Some(self.current as u64)
    }
}

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{drain, Counter};

    #[test]
    fn drains_the_requested_number_of_items() {
        let mut counter = Counter { current: 0 };
        assert_eq!(drain(&mut counter, 3), vec![1, 2, 3]);
    }

    #[test]
    fn a_zero_limit_drains_nothing() {
        let mut counter = Counter { current: 0 };
        assert!(drain(&mut counter, 0).is_empty());
    }

    #[test]
    fn the_counter_advances() {
        let mut counter = Counter { current: 10 };
        assert_eq!(drain(&mut counter, 1), vec![11]);
        assert_eq!(counter.current, 11);
    }
}
