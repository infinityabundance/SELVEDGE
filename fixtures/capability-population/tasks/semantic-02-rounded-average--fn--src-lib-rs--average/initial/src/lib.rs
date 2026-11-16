//! Integer averages with half-up rounding.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::average;

    #[test]
    fn rounds_half_up() {
        assert_eq!(average(&[1, 2]), 2);
    }

    #[test]
    fn exact_means_are_unchanged() {
        assert_eq!(average(&[2, 4]), 3);
    }

    #[test]
    fn rounds_down_below_half() {
        assert_eq!(average(&[1, 2, 3]), 2);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(average(&[]), 0);
    }

    #[test]
    fn single_value_is_itself() {
        assert_eq!(average(&[7]), 7);
    }
}
