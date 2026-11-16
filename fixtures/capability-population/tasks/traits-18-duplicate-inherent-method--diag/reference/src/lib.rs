//! A meter with one inherent impl block.

pub struct Meter {
    pub value: u64,
}

impl Meter {
    pub fn ratio(&self) -> u64 {
        self.value
    }
}
