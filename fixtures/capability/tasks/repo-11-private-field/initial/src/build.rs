//! Build a configuration. This does not compile: it writes a private
//! field from another module.

use crate::model::Config;

pub fn make() -> Config {
    Config { value: 2 }
}
