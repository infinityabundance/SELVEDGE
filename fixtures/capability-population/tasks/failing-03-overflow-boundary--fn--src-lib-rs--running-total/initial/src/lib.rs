//! Running total over bytes.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::running_total;

    #[test]
    fn saturates_at_the_boundary() {
        assert_eq!(running_total(&[250, 10]), 255);
    }

    #[test]
    fn saturates_after_the_boundary() {
        assert_eq!(running_total(&[255, 255]), 255);
    }

    #[test]
    fn sums_below_the_boundary() {
        assert_eq!(running_total(&[1, 2, 3]), 6);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(running_total(&[]), 0);
    }
}
