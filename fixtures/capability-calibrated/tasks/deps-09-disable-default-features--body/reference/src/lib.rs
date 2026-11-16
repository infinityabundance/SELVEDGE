//! Render through the vendored crate.

pub fn render(value: &str) -> String {
    
    codec::render(value)

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn trims_and_upper_cases() {
        assert_eq!(render("  alpha  "), "ALPHA");
    }
}
