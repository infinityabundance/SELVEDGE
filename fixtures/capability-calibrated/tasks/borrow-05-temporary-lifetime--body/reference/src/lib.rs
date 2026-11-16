//! A label built from a borrowed temporary.

pub fn first_word(text: &str) -> &str {
    text.split_whitespace().next().unwrap_or("")
}

pub fn label(number: u32) -> String {
    
    let formatted = format!("item {number}");
    first_word(&formatted).to_string()

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn labels_the_first_word() {
        assert_eq!(label(3), "item");
    }
}
