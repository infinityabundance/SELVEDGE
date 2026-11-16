//! Swap the ends of a vector.

pub fn swap_ends(values: &mut Vec<i64>) {
    let last = values.len().saturating_sub(1);
    if last > 0 {
        values.swap(0, last);
    }
}
