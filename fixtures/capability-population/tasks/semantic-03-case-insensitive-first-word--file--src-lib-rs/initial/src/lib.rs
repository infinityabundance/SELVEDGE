//! Command detection, currently case-sensitive.

pub fn is_command(text: &str, name: &str) -> bool {
    text.split_whitespace().next().map(|word| word == name).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::is_command;

    #[test]
    fn matches_a_lowercase_word() {
        assert!(is_command("run now", "run"));
    }
}
