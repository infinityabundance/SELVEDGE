//! Units. The associated constant prevents dynamic dispatch.

pub trait Unit {
    const SCALE: u32;

    fn convert(&self, value: u32) -> u32;
}

pub struct Meters;

impl Unit for Meters {
    const SCALE: u32 = 1000;

    fn convert(&self, value: u32) -> u32 {
        value * Self::SCALE
    }
}

pub struct Centimeters;

impl Unit for Centimeters {
    const SCALE: u32 = 10;

    fn convert(&self, value: u32) -> u32 {
        value * Self::SCALE
    }
}

pub fn convert_all(units: &[Box<dyn Unit>], value: u32) -> Vec<u32> {
    units.iter().map(|unit| unit.convert(value)).collect()
}
