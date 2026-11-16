//! A window over a text, computing its first word on demand.

pub struct Window {
    text: String,
}

impl Window {
    pub fn new(text: String) -> Self {
        Self { text }
    }

    pub fn first(&self) -> &str {
        self.text.split_whitespace().next().unwrap_or("")
    }

    pub fn replace(&mut self, text: String) {
        self.text = text;
    }
}
