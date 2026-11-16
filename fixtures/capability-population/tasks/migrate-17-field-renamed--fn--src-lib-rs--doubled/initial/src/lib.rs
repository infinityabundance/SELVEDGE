//! Doubling a measurement.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::doubled;

    #[test]
    fn doubles_the_amount() {
        assert_eq!(doubled("2.5"), Some(5.0));
    }

    #[test]
    fn zero_doubles_to_zero() {
        assert_eq!(doubled("0"), Some(0.0));
    }

    #[test]
    fn text_is_rejected() {
        assert_eq!(doubled("nope"), None);
    }
}
