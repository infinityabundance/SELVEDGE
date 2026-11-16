//! Cache reads.

use crate::api;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn caches_a_found_value() {
        assert_eq!(cached("ok"), 1);
    }
}
