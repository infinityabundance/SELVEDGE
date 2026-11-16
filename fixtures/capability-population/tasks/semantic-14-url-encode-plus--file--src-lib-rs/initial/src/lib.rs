//! Query-value encoding with the standard space replacement.

pub fn encode(text: &str) -> String {
    text.replace(' ', "+")
}

#[cfg(test)]
mod tests {
    use super::encode;

    #[test]
    fn visible_space_becomes_plus() {
        assert_eq!(encode("a b"), "a+b");
    }
}
