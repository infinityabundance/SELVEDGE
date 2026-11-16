//! A generic encoder. The crate does not build: the impl omits the
//! trait's type argument.

pub trait Encode<T> {
    fn encode(&self, value: T) -> String;
}

pub struct Hex;

impl Encode for Hex {
    fn encode(&self, value: u8) -> String {
        format!("{value:02x}")
    }
}

pub fn encode_byte(value: u8) -> String {
    use Encode;
    Hex.encode(value)
}
