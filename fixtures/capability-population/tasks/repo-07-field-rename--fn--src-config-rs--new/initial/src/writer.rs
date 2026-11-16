//! Configuration writer.

use crate::config::Config;

pub fn write(config: &Config) -> String {
    format!("write {}", config.path)
}
