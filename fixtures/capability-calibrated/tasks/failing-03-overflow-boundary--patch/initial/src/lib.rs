//! Running total over bytes.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn saturates_at_the_boundary() {
        assert_eq!(running_total(&[250, 10]), 255);
    }
}
