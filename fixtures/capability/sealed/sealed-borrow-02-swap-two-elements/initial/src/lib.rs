//! Swap the ends of a vector. The crate does not build: both ends are
//! borrowed mutably at once.

pub fn swap_ends(values: &mut Vec<i64>) {
    let first = &mut values[0];
    let last = &mut values[values.len() - 1];
    std::mem::swap(first, last);
}
