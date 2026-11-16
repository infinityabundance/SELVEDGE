//! Squares of a prefix. This does not build under edition 2024:
//! `gen` is a reserved keyword.

pub fn squares(count: usize) -> Vec<usize> {
    let gen = (0..count).collect::<Vec<_>>();
    gen.iter().map(|n| n * n).collect()
}
