//! Store reads.

pub fn load(key: &str, data: &[(&str, i32)]) -> Option<i32> {
    data.iter()
        .find(|(name, _value)| *name == key)
        .map(|(_name, value)| *value)
}
