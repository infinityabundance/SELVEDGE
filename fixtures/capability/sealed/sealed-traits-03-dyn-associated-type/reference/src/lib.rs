//! A trait object with its associated type specified.

pub trait Store {
    type Output;
    fn get(&self, key: u8) -> Self::Output;
}

pub struct Map;

impl Store for Map {
    type Output = u8;

    fn get(&self, key: u8) -> u8 {
        key
    }
}

pub fn describe(store: &dyn Store<Output = u8>) -> u8 {
    store.get(0)
}
