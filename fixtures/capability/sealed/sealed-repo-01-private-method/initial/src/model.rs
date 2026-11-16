//! The configuration type.

pub struct Config {
    pub value: u8,
}

impl Config {
    pub fn new(value: u8) -> Self {
        Self { value }
    }

    fn secret(&self) -> u8 {
        self.value
    }
}
