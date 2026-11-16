//! The keyed store.

pub trait Store {
    fn get(&self, key: u8) -> Option<u8>;
}

pub struct Map {
    pub value: u8,
}

impl Store for Map {
    fn get(&self, key: u8) -> Option<u8> {
        if key == 0 {
            Some(self.value)
        } else {
            None
        }
    }
}
