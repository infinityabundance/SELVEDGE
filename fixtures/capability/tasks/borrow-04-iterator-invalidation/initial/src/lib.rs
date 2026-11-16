//! Remove every key longer than a bound.

use std::collections::BTreeMap;

pub fn drop_long_keys(map: &mut BTreeMap<String, i32>, max_len: usize) -> usize {
    let mut removed = 0;
    for (key, _value) in map.iter() {
        if key.len() > max_len {
            map.remove(key);
            removed += 1;
        }
    }
    removed
}
