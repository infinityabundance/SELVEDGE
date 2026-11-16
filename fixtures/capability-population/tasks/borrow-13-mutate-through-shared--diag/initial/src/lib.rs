//! Increment every entry in place. The crate does not build: the
//! vector is borrowed immutably but mutated.

pub fn stamp(entries: &Vec<u32>) -> u32 {
    for index in 0..entries.len() {
        entries[index] += 1;
    }
    entries.iter().sum()
}
