//! Labels, version 2. Rendering is through `Display`; the inherent
//! `text` accessor was removed.

pub struct Label(String);

impl Label {
    pub fn new(value: &str) -> Self {
        Label(value.to_string())
    }
}

impl core::fmt::Display for Label {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}
