//! Split the first token length from an owned string.

pub fn split_first(text: String) -> (String, usize) {
    let first = text.split_whitespace().next().unwrap_or("");
    (text, first.len())
}
