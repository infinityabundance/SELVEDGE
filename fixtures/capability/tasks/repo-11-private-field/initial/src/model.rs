//! The configuration type.

pub struct Config {
    value: u8,
}

impl Config {
    pub fn new(value: u8) -> Self {
        Self { value }
    }

    pub fn value(&self) -> u8 {
        self.value
    }
}
