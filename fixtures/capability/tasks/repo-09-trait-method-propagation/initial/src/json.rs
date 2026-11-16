//! JSON sink.

use crate::pipeline::Sink;

pub struct Json;

impl Sink for Json {
    fn write(&self, value: i32) -> String {
        format!("{{\"v\":{value}}}")
    }
}
