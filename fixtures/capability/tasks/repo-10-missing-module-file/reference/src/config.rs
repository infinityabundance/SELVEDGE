//! Settings.

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub retries: u32,
    pub label: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self { retries: 3, label: String::from("default") }
    }
}
