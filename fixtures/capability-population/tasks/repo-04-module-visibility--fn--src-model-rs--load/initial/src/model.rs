//! Record parsing.

pub struct Record {
    pub value: i32,
}

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::run;

    #[test]
    fn totals_numbers() {
        assert_eq!(run(&["1", "2", "3"]), 6);
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(run(&[" 4 ", "5"]), 9);
    }

    #[test]
    fn invalid_lines_count_as_zero() {
        assert_eq!(run(&["x", "2"]), 2);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(run(&[]), 0);
    }
}
