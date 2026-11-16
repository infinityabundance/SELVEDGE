//! Field cleaning with a leading trim only.

pub fn cleaned(text: &str) -> String {
    text.trim_start().to_string()
}

#[cfg(test)]
mod tests {
    use super::cleaned;

    #[test]
    fn visible_leading_whitespace() {
        assert_eq!(cleaned(" x"), "x");
    }
}
