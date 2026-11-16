//! Summing measured values.

pub trait Measure {
    fn value(&self) -> u32;
}

pub struct Reading(pub u32);

impl Measure for Reading {
    fn value(&self) -> u32 {
        self.0
    }
}

pub fn total<T: Measure>(items: &[T]) -> u32 {
    items.iter().map(|item| item.value()).sum()
}
