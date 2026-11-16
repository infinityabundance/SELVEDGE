//! Split the first token length from an owned string.

pub fn split_first(text: String) -> (String, usize) {
    let len = text.split_whitespace().next().unwrap_or("").len();
    (text, len)
}
