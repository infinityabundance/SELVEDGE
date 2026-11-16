//! The right item.

pub struct Item {
    pub value: u8,
}

impl Item {
    pub fn right_value(&self) -> u8 {
        self.value * 2
    }
}
