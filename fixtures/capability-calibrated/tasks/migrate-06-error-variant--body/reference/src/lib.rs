//! Read a value through the table crate.

pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
    
    match table::lookup(table, key) {
        Ok(value) => Ok(value),
        Err(table::Error::Missing { .. }) => Err(format!("missing {key}")),
        Err(table::Error::Empty) => Err("empty key".to_string()),
    }

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
