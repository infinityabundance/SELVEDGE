//! Service reads.

use crate::storage::load;

pub fn fetch(key: &str, data: &[(&str, i32)]) -> Option<i32> {
    load(key, data).filter(|value| *value > 0)
}
