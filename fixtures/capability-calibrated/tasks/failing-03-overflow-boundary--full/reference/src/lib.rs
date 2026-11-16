//! Running total over bytes.

pub fn running_total(values: &[u8]) -> u8 {
    values
        .iter()
        .fold(0u8, |total, value| total.saturating_add(*value))
}
