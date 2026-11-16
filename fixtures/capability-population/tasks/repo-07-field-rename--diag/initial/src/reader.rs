//! Configuration reader.

use crate::config::Config;

pub fn read(config: &Config) -> String {
    format!("{}:{}", config.path, config.retries)
}
