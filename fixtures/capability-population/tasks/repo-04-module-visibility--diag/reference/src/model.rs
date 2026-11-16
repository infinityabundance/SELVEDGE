//! Record parsing.

pub struct Record {
    pub value: i32,
}

pub(crate) fn load(lines: &[&str]) -> Vec<Record> {
    lines
        .iter()
        .map(|line| Record { value: line.trim().parse().unwrap_or(0) })
        .collect()
}
