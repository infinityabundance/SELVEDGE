//! A label built from a borrowed temporary.

pub fn first_word(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}

pub fn label(number: u32) -> String {
    let formatted = format!("item {number}");
    first_word(&formatted).to_string()
}
