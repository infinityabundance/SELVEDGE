//! A generic encoder.

pub trait Encode<T> {
    fn encode(&self, value: T) -> String;
}

pub struct Hex;

impl Encode<u8> for Hex {
    fn encode(&self, value: u8) -> String {
        format!("{value:02x}")
    }
}

pub fn encode_byte(value: u8) -> String {
    use Encode;
    Hex.encode(value)
}
