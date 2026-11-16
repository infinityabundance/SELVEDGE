//! Read a value through the table crate.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    fn table() -> Vec<(&'static str, i32)> {
        vec![("alpha", 1), ("beta", 2)]
    }
    
    #[test]
    fn returns_a_present_value() {
        assert_eq!(read(&table(), "beta"), Ok(2));
    }
}
