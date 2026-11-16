//! Increment every entry in place.

pub fn stamp(entries: &mut Vec<u32>) -> u32 {
    for index in 0..entries.len() {
        entries[index] += 1;
    }
    entries.iter().sum()
}
