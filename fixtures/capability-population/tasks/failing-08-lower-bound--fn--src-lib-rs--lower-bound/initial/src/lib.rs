//! Lower-bound search.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::lower_bound;

    #[test]
    fn finds_a_present_value_at_its_index() {
        assert_eq!(lower_bound(&[1, 3, 5], 3), 1);
    }

    #[test]
    fn finds_the_first_duplicate() {
        assert_eq!(lower_bound(&[1, 2, 2, 2, 3], 2), 1);
    }

    #[test]
    fn reports_the_insertion_point_for_an_absent_value() {
        assert_eq!(lower_bound(&[1, 3, 5], 4), 2);
    }

    #[test]
    fn an_empty_slice_has_index_zero() {
        assert_eq!(lower_bound(&[], 9), 0);
    }

    #[test]
    fn a_value_before_everything_has_index_zero() {
        assert_eq!(lower_bound(&[4, 5, 6], 1), 0);
    }

    #[test]
    fn a_value_after_everything_has_length_index() {
        assert_eq!(lower_bound(&[1, 2], 9), 2);
    }
}
