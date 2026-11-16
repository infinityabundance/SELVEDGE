//! CSV field quoting.

pub fn field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod visible {
    use crate::field;

    #[test]
    fn plain_fields_are_unquoted() {
        assert_eq!(field("plain"), "plain");
    }

    #[test]
    fn comma_fields_are_quoted() {
        assert_eq!(field("a,b"), "\"a,b\"");
    }

    #[test]
    fn embedded_quotes_are_doubled() {
        assert_eq!(field("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn newline_fields_are_quoted() {
        assert_eq!(field("a\nb"), "\"a\nb\"");
    }

    #[test]
    fn empty_is_unquoted() {
        assert_eq!(field(""), "");
    }
}
