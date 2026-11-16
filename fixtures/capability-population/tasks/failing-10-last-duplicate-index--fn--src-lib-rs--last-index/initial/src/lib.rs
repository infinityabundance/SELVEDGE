//! Last index of a value.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::last_index;

    #[test]
    fn finds_the_last_of_several_duplicates() {
        assert_eq!(last_index(&[1, 2, 2, 2, 3], 2), Some(3));
    }

    #[test]
    fn finds_a_unique_value() {
        assert_eq!(last_index(&[1, 3, 5], 5), Some(2));
    }

    #[test]
    fn a_missing_value_is_none() {
        assert_eq!(last_index(&[1, 2, 3], 4), None);
    }

    #[test]
    fn an_empty_slice_is_none() {
        assert_eq!(last_index(&[], 0), None);
    }
}
