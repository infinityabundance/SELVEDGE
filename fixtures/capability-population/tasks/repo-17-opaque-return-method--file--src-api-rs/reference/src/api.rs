//! The public store API.

use crate::store::{Map, Store};

pub fn open() -> impl Store {
    Map { value: 7 }
}

pub fn peek() -> Option<u8> {
    open().get(0)
}
