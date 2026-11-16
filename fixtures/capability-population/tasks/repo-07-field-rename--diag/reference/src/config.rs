//! Shared configuration.

pub struct Config {
    pub path: String,
    pub retries: u32,
}

impl Config {
    pub fn new(path: &str, retries: u32) -> Self {
        Config { path: path.to_string(), retries }
    }
}
