//! An element trait.

pub trait Element {
    type Value;
    fn value(&self) -> Self::Value;
}

pub struct Cell {
    pub item: u8,
}

impl Element for Cell {
    type Value = u8;

    fn value(&self) -> u8 {
        self.item
    }
}

pub fn clone_value<E: Element>(element: &E) -> E::Value {
    element.value()
}
