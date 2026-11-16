//! Remove every key longer than a bound.

use std::collections::BTreeMap;

pub fn drop_long_keys(map: &mut BTreeMap<String, i32>, max_len: usize) -> usize {
    let doomed: Vec<String> = map
        .keys()
        .filter(|key| key.len() > max_len)
        .cloned()
        .collect();
    for key in &doomed {
        map.remove(key);
    }
    doomed.len()
}
