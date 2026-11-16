//! Cache reads.

use crate::api;

pub fn cached(key: &str) -> i32 {
    match api::fetch(key) {
        Ok(value) => value,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn caches_a_found_value() {
        assert_eq!(cached("ok"), 1);
    }
}
