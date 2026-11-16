//! A report over the configuration. This does not compile: the method it
//! calls is private to `model`.

use crate::model::Config;

pub fn describe(config: &Config) -> u8 {
    config.secret()
}
