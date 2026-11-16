//! Push a value and report the value that was last before it.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::push_and_report;

    #[test]
    fn reports_zero_for_an_empty_vector() {
        let mut values = vec![];
        assert_eq!(push_and_report(&mut values, 7), 0);
        assert_eq!(values, vec![7]);
    }

    #[test]
    fn reports_the_previous_last() {
        let mut values = vec![1, 2];
        assert_eq!(push_and_report(&mut values, 9), 2);
        assert_eq!(values, vec![1, 2, 9]);
    }

    #[test]
    fn repeated_pushes_chain() {
        let mut values = vec![5];
        assert_eq!(push_and_report(&mut values, 6), 5);
        assert_eq!(push_and_report(&mut values, 8), 6);
        assert_eq!(values, vec![5, 6, 8]);
    }
}
