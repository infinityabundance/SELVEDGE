//! CSV field quoting.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn plain_fields_are_unquoted() {
        assert_eq!(field("plain"), "plain");
    }
}
