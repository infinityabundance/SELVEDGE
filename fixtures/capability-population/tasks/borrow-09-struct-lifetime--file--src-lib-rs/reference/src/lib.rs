//! A view with an explicit lifetime.

pub struct View<'a> {
    pub text: &'a str,
}

impl<'a> View<'a> {
    pub fn new(text: &'a str) -> View<'a> {
        View { text }
    }

    pub fn text(&self) -> &str {
        self.text
    }
}
