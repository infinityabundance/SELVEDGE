//! Render a label.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::render;

    #[test]
    fn renders_a_value() {
        assert_eq!(render("alpha"), "alpha");
    }

    #[test]
    fn preserves_spacing() {
        assert_eq!(render("a b"), "a b");
    }

    #[test]
    fn renders_unicode() {
        assert_eq!(render("hé"), "hé");
    }

    #[test]
    fn empty_renders_empty() {
        assert_eq!(render(""), "");
    }
}
