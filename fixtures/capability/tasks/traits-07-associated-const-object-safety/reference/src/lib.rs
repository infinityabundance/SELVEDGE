//! Units.

pub trait Unit {
    fn scale(&self) -> u32;

    fn convert(&self, value: u32) -> u32 {
        value * self.scale()
    }
}

pub struct Meters;

impl Unit for Meters {
    fn scale(&self) -> u32 {
        1000
    }
}

pub struct Centimeters;

impl Unit for Centimeters {
    fn scale(&self) -> u32 {
        10
    }
}

pub fn convert_all(units: &[Box<dyn Unit>], value: u32) -> Vec<u32> {
    units.iter().map(|unit| unit.convert(value)).collect()
}
