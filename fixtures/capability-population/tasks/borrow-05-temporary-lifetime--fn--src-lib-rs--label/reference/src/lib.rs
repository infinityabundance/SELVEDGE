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
    use crate::label;

    #[test]
    fn labels_the_first_word() {
        assert_eq!(label(3), "item");
    }

    #[test]
    fn zero_is_still_labelled() {
        assert_eq!(label(0), "item");
    }

    #[test]
    fn the_number_only_changes_the_temporary() {
        assert_eq!(label(42), "item");
    }

    #[test]
    fn successive_labels_are_independent() {
        assert_eq!(label(1), "item");
        assert_eq!(label(2), "item");
    }
}
