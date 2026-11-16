//! Report formatting.

use crate::api;

pub fn describe(key: &str) -> String {
    match api::fetch(key) {
        Ok(value) => format!("{key}={value}"),
        Err(_) => format!("{key}=missing"),
    }
}
