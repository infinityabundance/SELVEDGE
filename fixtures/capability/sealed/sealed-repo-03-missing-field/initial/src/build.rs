//! Build a configuration. This does not compile: the model gained a field.

use crate::model::Config;

pub fn make() -> Config {
    Config { retries: 3 }
}
