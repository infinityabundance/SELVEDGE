//! A fetch pipeline.

pub mod api;
pub mod cache;
pub mod report;

pub fn run(key: &str) -> String {
    format!("{}:{}", cache::cached(key), report::describe(key))
}
