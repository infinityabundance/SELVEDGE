//! Initials.

pub fn initials(text: &str) -> String {
    text.split_whitespace()
        .filter_map(|word| word.chars().next())
        .flat_map(|first| first.to_uppercase())
        .collect()
}
