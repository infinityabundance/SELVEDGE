//! Cache reads.

use crate::api;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{cache::cached, report::describe, run};

    #[test]
    fn caches_a_found_value() {
        assert_eq!(cached("ok"), 1);
    }

    #[test]
    fn a_miss_falls_back_to_zero() {
        assert_eq!(cached("missing"), 0);
        assert_eq!(cached(""), 0);
    }

    #[test]
    fn describes_a_found_value() {
        assert_eq!(describe("ok"), "ok=1");
        assert_eq!(describe("neg"), "neg=-2");
    }

    #[test]
    fn describes_an_error_as_missing() {
        assert_eq!(describe("missing"), "missing=missing");
        assert_eq!(run("missing"), "0:missing=missing");
    }
}
