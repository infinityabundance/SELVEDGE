//! A meter with two inherent impl blocks. The crate does not build: both
//! blocks define `ratio`.

pub struct Meter {
    pub value: u64,
}

impl Meter {
    pub fn ratio(&self) -> u64 {
        self.value
    }
}

impl Meter {
    pub fn ratio(&self) -> u64 {
        self.value * 2
    }
}
