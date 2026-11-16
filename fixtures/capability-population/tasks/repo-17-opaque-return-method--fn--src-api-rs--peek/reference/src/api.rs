//! The public store API.

use crate::store::{Map, Store};

pub fn open() -> impl Store {
    Map { value: 7 }
}

pub fn peek() -> Option<u8> {
    open().get(0)
}

#[cfg(test)]
mod visible {
    use crate::{open, peek, Store};

    #[test]
    fn peek_returns_the_value_at_key_zero() {
        assert_eq!(peek(), Some(7));
    }

    #[test]
    fn the_opaque_value_exposes_the_trait_method() {
        assert_eq!(open().get(0), Some(7));
    }

    #[test]
    fn other_keys_are_absent() {
        assert_eq!(open().get(1), None);
    }
}
