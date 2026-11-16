//! Running total over bytes.

pub fn running_total(values: &[u8]) -> u8 {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn saturates_at_the_boundary() {
        assert_eq!(running_total(&[250, 10]), 255);
    }
}
