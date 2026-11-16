//! Write rows to a file.

use std::io;

pub fn write_all(path: &str, rows: &[(String, i64)]) -> io::Result<()> {
    let mut text = String::new();
    for (name, value) in rows {
        text.push_str(&format!("{name},{value}\n"));
    }
    std::fs::write(path, text)
}
