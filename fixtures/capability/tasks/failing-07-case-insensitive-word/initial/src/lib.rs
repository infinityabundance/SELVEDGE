//! Command detection. The visible test fails: matching is
//! case-sensitive and an empty name matches an empty text.

pub fn is_command(text: &str, name: &str) -> bool {
    text.split_whitespace().next().map(|word| word == name).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::is_command;

    #[test]
    fn matches_regardless_of_case() {
        assert!(is_command("RUN now", "run"));
    }
}
