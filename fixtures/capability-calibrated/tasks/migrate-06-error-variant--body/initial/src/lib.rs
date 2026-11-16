//! Read a value through the table crate.

pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
    todo!()
}

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
