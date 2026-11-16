//! Service reads.

use crate::store::load;

pub fn fetch(key: &str, data: &[(&str, i32)]) -> Option<i32> {
    load(key, data).filter(|value| *value > 0)
}

#[cfg(test)]
mod visible {
    use crate::{fetch, report::describe};

    fn data() -> Vec<(&'static str, i32)> {
        vec![("a", 2), ("b", -1), ("zero", 0)]
    }

    #[test]
    fn fetches_a_positive_value() {
        assert_eq!(fetch("a", &data()), Some(2));
    }

    #[test]
    fn filters_a_non_positive_value() {
        assert_eq!(fetch("b", &data()), None);
        assert_eq!(fetch("zero", &data()), None);
        assert_eq!(fetch("missing", &data()), None);
    }

    #[test]
    fn describes_a_found_value() {
        assert_eq!(describe("a", &data()), "a=2");
    }

    #[test]
    fn describes_a_missing_value() {
        assert_eq!(describe("z", &data()), "z=missing");
    }
}
