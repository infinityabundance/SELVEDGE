//! A view that borrows text but has no lifetime parameter.

pub struct View {
    pub text: &str,
}

impl View {
    pub fn new(text: &str) -> View {
        View { text }
    }

    pub fn text(&self) -> &str {
        self.text
    }
}
