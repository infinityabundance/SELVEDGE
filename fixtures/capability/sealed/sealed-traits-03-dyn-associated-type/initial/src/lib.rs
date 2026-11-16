//! A trait object with an unspecified associated type. The crate does
//! not build.

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

pub fn describe(store: &dyn Store) -> u8 {
    store.get(0)
}
