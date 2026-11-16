//! Edition 2024.

pub fn describe(value: &Option<u32>) -> String {
    match value {
        Some(inner) => format!("some {inner}"),
        None => "none".to_string(),
    }
}

// TODO(FUNCTION)

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
