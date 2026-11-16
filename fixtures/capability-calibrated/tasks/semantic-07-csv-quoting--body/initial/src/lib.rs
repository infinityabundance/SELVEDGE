//! CSV field quoting.

pub fn field(value: &str) -> String {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn plain_fields_are_unquoted() {
        assert_eq!(field("plain"), "plain");
    }
}
