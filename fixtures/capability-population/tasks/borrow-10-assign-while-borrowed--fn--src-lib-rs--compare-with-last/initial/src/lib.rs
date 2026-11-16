//! Compare the previous value with the last reading.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::compare_with_last;

    #[test]
    fn adds_the_previous_and_the_last_reading() {
        assert_eq!(compare_with_last(5, &[7, 9]), 14);
    }

    #[test]
    fn an_empty_slice_compares_with_itself() {
        assert_eq!(compare_with_last(5, &[]), 10);
    }

    #[test]
    fn negative_values_are_preserved() {
        assert_eq!(compare_with_last(-3, &[-8]), -11);
    }
}
