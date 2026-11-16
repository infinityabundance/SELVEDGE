//! The registration flow.

use crate::registry::Registry;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn two_names_are_counted() {
        assert_eq!(register_and_count(Registry::new(), &["a", "b"]), 2);
    }
}
