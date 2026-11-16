//! A hex codec whose impl signature does not match the trait.

pub trait Codec {
    fn encode(&self, value: u32) -> String;
}

pub struct Hex;

impl Codec for Hex {
    fn encode(&self, value: u64) -> String {
        format!("{value:x}")
    }
}

pub fn encode_all<T: Codec>(codec: &T, values: &[u32]) -> Vec<String> {
    values.iter().map(|value| codec.encode(*value)).collect()
}
