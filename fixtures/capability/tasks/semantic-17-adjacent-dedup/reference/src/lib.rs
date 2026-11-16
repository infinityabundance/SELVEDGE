//! Collapsing adjacent duplicates.

pub fn collapse(values: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for value in values {
        if out.last() != Some(value) {
            out.push(*value);
        }
    }
    out
}
