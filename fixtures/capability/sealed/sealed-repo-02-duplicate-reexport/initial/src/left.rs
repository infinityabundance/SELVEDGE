//! The left item.

pub struct Item {
    pub value: u8,
}

impl Item {
    pub fn left_value(&self) -> u8 {
        self.value
    }
}
