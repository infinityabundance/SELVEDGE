//! Report formatting, still matching the old Option API.

use crate::api;

pub fn describe(key: &str) -> String {
    match api::fetch(key) {
        Some(value) => format!("{key}={value}"),
        None => format!("{key}=missing"),
    }
}
