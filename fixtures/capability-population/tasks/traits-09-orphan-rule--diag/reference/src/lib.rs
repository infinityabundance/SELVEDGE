//! Render bytes as decimal text through a local newtype.

pub struct Bytes(pub Vec<u8>);

impl core::fmt::Display for Bytes {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = self
            .0
            .iter()
            .map(|byte| byte.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(formatter, "{text}")
    }
}

pub fn render(values: &[u8]) -> String {
    Bytes(values.to_vec()).to_string()
}
