//! Squares of a prefix.

pub fn squares(count: usize) -> Vec<usize> {
    let generation = (0..count).collect::<Vec<_>>();
    generation.iter().map(|n| n * n).collect()
}
