//! The public store API. This does not compile: `first` is not a trait
//! method, so it is not visible on the opaque return type.

use crate::store::{Map, Store};

pub fn open() -> impl Store {
    Map { value: 7 }
}

pub fn peek() -> Option<u8> {
    open().first()
}
