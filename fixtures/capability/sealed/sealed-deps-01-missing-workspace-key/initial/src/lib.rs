//! A nested package. This does not build: it declares no workspace.

pub fn max_gap(values: &[u32]) -> u32 {
    values
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .max()
        .unwrap_or(0)
}
