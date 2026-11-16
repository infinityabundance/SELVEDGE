//! Running total over bytes.

pub fn running_total(values: &[u8]) -> u8 {
    
    values
        .iter()
        .fold(0u8, |total, value| total.saturating_add(*value))

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn saturates_at_the_boundary() {
        assert_eq!(running_total(&[250, 10]), 255);
    }
}
