//! A counter with sequential mutable borrows.

pub struct Counter {
    value: i32,
}

impl Counter {
    pub fn new(value: i32) -> Self {
        Counter { value }
    }

// TODO(FUNCTION)

    pub fn value_mut(&mut self) -> &mut i32 {
        &mut self.value
    }

    pub fn bump_twice(&mut self) {
        let first = self.value_mut();
        *first += 1;
        let second = self.value_mut();
        *second += 1;
    }
}

#[cfg(test)]
mod visible {
    use crate::Counter;

    #[test]
    fn bumps_a_fresh_counter_twice() {
        let mut counter = Counter::new(0);
        counter.bump_twice();
        assert_eq!(counter.value(), 2);
    }

    #[test]
    fn bumps_from_the_current_value() {
        let mut counter = Counter::new(5);
        counter.bump_twice();
        assert_eq!(counter.value(), 7);
    }

    #[test]
    fn repeated_calls_accumulate() {
        let mut counter = Counter::new(0);
        counter.bump_twice();
        counter.bump_twice();
        counter.bump_twice();
        assert_eq!(counter.value(), 6);
    }

    #[test]
    fn the_mutable_accessor_still_works() {
        let mut counter = Counter::new(1);
        *counter.value_mut() = 9;
        assert_eq!(counter.value(), 9);
    }
}
