//! Key/value splitting at the first separator.

pub fn split_key(text: &str) -> Option<(String, String)> {
    text.split_once(':')
        .map(|(key, value)| (key.to_string(), value.to_string()))
}
