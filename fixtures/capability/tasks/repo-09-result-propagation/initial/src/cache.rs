//! Cache reads, still matching the old Option API.

use crate::api;

pub fn cached(key: &str) -> i32 {
    match api::fetch(key) {
        Some(value) => value,
        None => 0,
    }
}
