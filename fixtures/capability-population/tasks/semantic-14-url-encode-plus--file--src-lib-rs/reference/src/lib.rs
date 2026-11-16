//! Query-value encoding.

pub fn encode(text: &str) -> String {
    text.replace('+', "%2B").replace(' ', "+")
}
