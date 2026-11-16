//! A window over a text, caching its first word as a slice.

pub struct Window {
    text: String,
    first: &'static str,
}

impl Window {
    pub fn new(text: String) -> Self {
        let first = text.split_whitespace().next().unwrap_or("");
        Self { text, first }
    }

    pub fn first(&self) -> &str {
        self.first
    }

    pub fn replace(&mut self, text: String) {
        self.text = text;
    }
}
