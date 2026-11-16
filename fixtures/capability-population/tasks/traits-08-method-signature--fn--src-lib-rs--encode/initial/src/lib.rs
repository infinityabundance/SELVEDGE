//! A hex codec.

pub trait Codec {
    fn encode(&self, value: u32) -> String;
}

pub struct Hex;

impl Codec for Hex {
// TODO(FUNCTION)
}

pub fn encode_all<T: Codec>(codec: &T, values: &[u32]) -> Vec<String> {
    values.iter().map(|value| codec.encode(*value)).collect()
}

#[cfg(test)]
mod visible {
    use crate::{encode_all, Codec, Hex};

    #[test]
    fn encodes_the_maximum_byte() {
        assert_eq!(Hex.encode(255), "ff");
    }

    #[test]
    fn encodes_through_the_generic_helper() {
        assert_eq!(encode_all(&Hex, &[1, 15, 255]), vec!["1", "f", "ff"]);
    }

    #[test]
    fn zero_encodes_to_one_zero() {
        assert_eq!(Hex.encode(0), "0");
    }

    #[test]
    fn an_empty_slice_is_empty() {
        assert!(encode_all(&Hex, &[]).is_empty());
    }
}
