//! A workspace resolver. This does not build: the resolver is an integer.

pub fn reverse_words(text: &str) -> String {
    text.split_whitespace()
        .rev()
        .collect::<Vec<_>>()
        .join(" ")
}
