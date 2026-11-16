//! Numeric sorting.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::sorted;

    #[test]
    fn wider_numbers_sort_by_value() {
        assert_eq!(sorted(&[10, 9, 100]), vec![9, 10, 100]);
    }

    #[test]
    fn single_digits_sort_by_value() {
        assert_eq!(sorted(&[3, 1, 2]), vec![1, 2, 3]);
    }

    #[test]
    fn duplicates_are_kept() {
        assert_eq!(sorted(&[2, 2, 10]), vec![2, 2, 10]);
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(sorted(&[]).is_empty());
    }
}
