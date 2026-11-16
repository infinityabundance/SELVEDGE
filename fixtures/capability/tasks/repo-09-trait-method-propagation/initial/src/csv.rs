//! CSV sink.

use crate::pipeline::Sink;

pub struct Csv;

impl Sink for Csv {
    fn write(&self, value: i32) -> String {
        format!("{value}")
    }
}
