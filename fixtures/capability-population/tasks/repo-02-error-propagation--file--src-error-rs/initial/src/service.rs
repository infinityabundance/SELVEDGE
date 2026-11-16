//! Import a batch of rows.

use crate::error::AppError;
use crate::parse::parse_row;
use crate::store::write_all;

pub fn import(path: &str, lines: &[String]) -> Result<usize, AppError> {
    let mut rows = Vec::new();
    for line in lines {
        rows.push(parse_row(line)?);
    }
    write_all(path, &rows)?;
    Ok(rows.len())
}
