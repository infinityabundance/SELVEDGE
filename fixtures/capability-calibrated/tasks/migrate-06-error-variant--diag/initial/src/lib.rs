//! Read a value through the table crate. This does not build: the
//! `NotFound` variant was replaced by `Missing { key }` and `Empty`.

pub fn read(table: &[(&'static str, i32)], key: &str) -> Result<i32, String> {
    match table::lookup(table, key) {
        Ok(value) => Ok(value),
        Err(table::Error::Missing { .. }) => Err(format!("missing {key}")),
        Err(table::Error::NotFound) => Err(format!("missing {key}")),
    }
}
