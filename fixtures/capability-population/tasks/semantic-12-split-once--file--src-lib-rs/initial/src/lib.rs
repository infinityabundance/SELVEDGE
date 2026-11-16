//! Key/value splitting at the first two components.

pub fn split_key(text: &str) -> Option<(String, String)> {
    let mut parts = text.split(':');
    let key = parts.next()?;
    let value = parts.next()?;
    Some((key.to_string(), value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::split_key;

    #[test]
    fn visible_single_separator() {
        assert_eq!(split_key("a:b"), Some(("a".to_string(), "b".to_string())));
    }
}
