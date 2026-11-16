//! A common-prefix helper that only truncates the first string.

pub fn common_prefix(values: &[&str]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let shortest = values.iter().map(|value| value.len()).min().unwrap_or(0);
    values[0][..shortest].to_string()
}

#[cfg(test)]
mod tests {
    use super::common_prefix;

    #[test]
    fn prefix_of_three_words() {
        assert_eq!(common_prefix(&["flower", "flow", "flight"]), "fl");
    }
}
