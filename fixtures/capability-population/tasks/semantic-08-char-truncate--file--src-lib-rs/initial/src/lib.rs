//! Truncation by byte index.

pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        text[..max_chars].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::truncate;

    #[test]
    fn truncates_ascii() {
        assert_eq!(truncate("abcdef", 3), "abc");
    }
}
