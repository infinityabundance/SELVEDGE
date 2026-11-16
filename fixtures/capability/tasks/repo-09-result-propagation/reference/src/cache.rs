//! Cache reads.

use crate::api;

pub fn cached(key: &str) -> i32 {
    match api::fetch(key) {
        Ok(value) => value,
        Err(_) => 0,
    }
}
