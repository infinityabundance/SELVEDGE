//! Report formatting.

use crate::store::load;

pub fn describe(key: &str, data: &[(&str, i32)]) -> String {
    match load(key, data) {
        Some(value) => format!("{key}={value}"),
        None => format!("{key}=missing"),
    }
}
