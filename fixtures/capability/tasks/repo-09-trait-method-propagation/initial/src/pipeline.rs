//! Sink pipeline.

pub trait Sink {
    fn write(&self, value: i32) -> String;
    fn flush(&self) -> String;
}

pub fn run(value: i32) -> String {
    let json = crate::json::Json;
    let csv = crate::csv::Csv;
    format!(
        "{};{};{}",
        Sink::write(&json, value),
        Sink::write(&csv, value),
        Sink::flush(&json)
    )
}
