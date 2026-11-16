//! Order-preserving de-duplication.

use std::collections::HashSet;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::unique;

    #[test]
    fn keeps_first_occurrence_order() {
        assert_eq!(unique(&[3, 1, 3, 2, 1]), vec![3, 1, 2]);
    }

    #[test]
    fn sorted_input_is_unchanged() {
        assert_eq!(unique(&[1, 1, 2]), vec![1, 2]);
    }

    #[test]
    fn all_duplicates_collapse_to_one() {
        assert_eq!(unique(&[5, 5, 5]), vec![5]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(unique(&[]).is_empty());
    }
}
