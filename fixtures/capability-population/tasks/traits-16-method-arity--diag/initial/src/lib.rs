//! A scaling trait. The crate does not build: the impl's parameter list
//! is shorter than the trait's.

pub trait Scale {
    fn scale(&self, factor: u32) -> u64;
}

pub struct Meter {
    pub value: u64,
}

impl Scale for Meter {
    fn scale(&self) -> u64 {
        self.value
    }
}

pub fn scaled(meter: &Meter, factor: u32) -> u64 {
    use Scale;
    meter.scale(factor)
}
