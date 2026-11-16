//! Build a configuration.

use crate::model::Config;

pub fn make() -> Config {
    Config { retries: 3, label: "default".to_string() }
}
