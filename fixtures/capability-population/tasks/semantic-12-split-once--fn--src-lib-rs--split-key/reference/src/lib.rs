//! Key/value splitting at the first separator.

pub fn split_key(text: &str) -> Option<(String, String)> {
    text.split_once(':')
        .map(|(key, value)| (key.to_string(), value.to_string()))
}

#[cfg(test)]
mod visible {
    use crate::split_key;

    #[test]
    fn the_value_keeps_later_separators() {
        assert_eq!(split_key("a:b:c"),
                   Some(("a".to_string(), "b:c".to_string())));
    }

    #[test]
    fn empty_parts_are_allowed() {
        assert_eq!(split_key("a:"), Some(("a".to_string(), String::new())));
        assert_eq!(split_key(":b"), Some((String::new(), "b".to_string())));
    }

    #[test]
    fn no_separator_is_none() {
        assert_eq!(split_key("abc"), None);
    }
}
