//! Identifier parsing with the standard integer parser.

pub fn parse_id(text: &str) -> Option<u32> {
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::parse_id;

    #[test]
    fn visible_plain_number() {
        assert_eq!(parse_id("12"), Some(12));
    }
}
