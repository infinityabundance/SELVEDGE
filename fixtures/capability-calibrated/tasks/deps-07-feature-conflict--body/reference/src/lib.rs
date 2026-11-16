//! Select the vendored mode.

pub fn selected() -> &'static str {
    
    codec::mode()

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn selects_the_safe_mode() {
        assert_eq!(selected(), "safe");
    }
}
