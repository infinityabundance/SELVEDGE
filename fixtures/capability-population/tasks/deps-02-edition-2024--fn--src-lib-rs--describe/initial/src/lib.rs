//! Edition 2024.

// TODO(FUNCTION)

pub fn next_value(seed: u64) -> u64 {
    let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    state % 100
}

#[cfg(test)]
mod visible {
    use crate::{describe, next_value};

    #[test]
    fn describes_some() {
        assert_eq!(describe(&Some(3)), "some 3");
    }

    #[test]
    fn describes_none() {
        assert_eq!(describe(&None), "none");
    }

    #[test]
    fn values_are_bounded_and_distinct() {
        assert_eq!(next_value(0), 1);
        assert!(next_value(0) < 100 && next_value(1) < 100);
        assert_ne!(next_value(0), next_value(1));
    }
}
