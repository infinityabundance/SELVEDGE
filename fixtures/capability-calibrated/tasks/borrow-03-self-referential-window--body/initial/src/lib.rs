//! A window over a text, computing its first word on demand.

pub struct Window {
    text: String,
}

impl Window {
    pub fn new(text: String) -> Self {
        Self { text }
    }

    pub fn first(&self) -> &str {
    todo!()
}

    pub fn replace(&mut self, text: String) {
        self.text = text;
    }
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn first_word() {
        assert_eq!(Window::new("alpha beta".to_string()).first(), "alpha");
    }
}
