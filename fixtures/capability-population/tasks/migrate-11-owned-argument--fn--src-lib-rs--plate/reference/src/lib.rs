//! Render a plate.

pub fn plate(text: &str) -> String {
    nameplate::render(text.to_string())
}

#[cfg(test)]
mod visible {
    use crate::plate;

    #[test]
    fn wraps_the_text_in_brackets() {
        assert_eq!(plate("x"), "[x]");
    }

    #[test]
    fn empty_text_is_still_wrapped() {
        assert_eq!(plate(""), "[]");
    }

    #[test]
    fn spaces_are_preserved() {
        assert_eq!(plate("a b"), "[a b]");
    }
}
