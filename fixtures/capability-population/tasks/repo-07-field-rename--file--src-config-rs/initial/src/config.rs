//! Shared configuration.

pub struct Config {
    pub source: String,
    pub retries: u32,
}

impl Config {
    pub fn new(source: &str, retries: u32) -> Self {
        Config { source: source.to_string(), retries }
    }
}
