//! A label built from a borrowed temporary.

pub fn first_word(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}

pub fn label(number: u32) -> String {
    let word = first_word(&format!("item {number}"));
    word.to_string()
}
