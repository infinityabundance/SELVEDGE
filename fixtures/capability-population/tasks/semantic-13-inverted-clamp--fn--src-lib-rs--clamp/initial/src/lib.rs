//! Clamping with an inverted-bound rule.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::clamp;

    #[test]
    fn inverted_bounds_return_the_low_bound() {
        assert_eq!(clamp(5, 10, 0), 10);
        assert_eq!(clamp(-1, 3, 1), 3);
    }

    #[test]
    fn a_value_below_the_bounds_is_lifted() {
        assert_eq!(clamp(-5, 0, 10), 0);
    }

    #[test]
    fn a_value_above_the_bounds_is_capped() {
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn a_value_inside_is_unchanged() {
        assert_eq!(clamp(5, 0, 10), 5);
    }
}
