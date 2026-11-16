//! Seed a table and read it back.

use std::collections::BTreeMap;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::seed;

    #[test]
    fn returns_the_seeded_value() {
        assert_eq!(seed("k"), "value-k".to_string());
    }

    #[test]
    fn successive_calls_are_independent() {
        let first = seed("one");
        let second = seed("two");
        assert_eq!(first, "value-one");
        assert_eq!(second, "value-two");
    }

    #[test]
    fn a_long_key_is_included_verbatim() {
        assert_eq!(seed("alpha-beta"), "value-alpha-beta".to_string());
    }
}
