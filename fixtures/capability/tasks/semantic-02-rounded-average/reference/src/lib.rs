//! Integer averages with half-up rounding.

pub fn average(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let len = values.len() as u32;
    (values.iter().sum::<u32>() + len / 2) / len
}
