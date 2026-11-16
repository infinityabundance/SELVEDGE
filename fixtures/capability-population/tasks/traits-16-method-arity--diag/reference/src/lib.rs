//! A scaling trait.

pub trait Scale {
    fn scale(&self, factor: u32) -> u64;
}

pub struct Meter {
    pub value: u64,
}

impl Scale for Meter {
    fn scale(&self, factor: u32) -> u64 {
        self.value * factor as u64
    }
}

pub fn scaled(meter: &Meter, factor: u32) -> u64 {
    use Scale;
    meter.scale(factor)
}
