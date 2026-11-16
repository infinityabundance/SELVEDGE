//! The new fallible fetch API.

pub fn fetch(key: &str) -> Result<i32, String> {
    if key.is_empty() {
        return Err("empty key".to_string());
    }
    match key {
        "ok" => Ok(1),
        "neg" => Ok(-2),
        _ => Err("missing".to_string()),
    }
}
