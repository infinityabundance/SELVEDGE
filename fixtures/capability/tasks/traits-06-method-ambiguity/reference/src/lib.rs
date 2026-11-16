//! Two traits with the same method.

pub trait Left {
    fn value(&self) -> u32;
}

pub trait Right {
    fn value(&self) -> u32;
}

pub struct Item(pub u32);

impl Left for Item {
    fn value(&self) -> u32 {
        self.0
    }
}

impl Right for Item {
    fn value(&self) -> u32 {
        self.0 + 100
    }
}

pub fn left(item: &Item) -> u32 {
    Left::value(item)
}

pub fn right(item: &Item) -> u32 {
    Right::value(item)
}
