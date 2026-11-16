//! Address normalisation with whole-string lowercasing.

pub fn normalise(text: &str) -> String {
    text.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::normalise;

    #[test]
    fn visible_already_lowercase() {
        assert_eq!(normalise("user@example.com"), "user@example.com");
    }
}
