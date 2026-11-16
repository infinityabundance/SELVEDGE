//! Read a value through the table crate.

pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
    match table::lookup(table, key) {
        Ok(value) => Ok(value),
        Err(table::Error::Missing { .. }) => Err(format!("missing {key}")),
        Err(table::Error::Empty) => Err("empty key".to_string()),
    }
}
