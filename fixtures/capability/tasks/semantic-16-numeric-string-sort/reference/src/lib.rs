//! Numeric sorting.

pub fn sorted(values: &[u32]) -> Vec<u32> {
    let mut out = values.to_vec();
    out.sort_unstable();
    out
}
