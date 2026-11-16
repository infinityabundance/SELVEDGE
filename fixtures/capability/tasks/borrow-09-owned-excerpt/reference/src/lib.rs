//! An excerpt that owns its text.

pub struct Excerpt {
    pub text: String,
}

impl Excerpt {
    pub fn text(&self) -> &str {
        &self.text
    }
}

pub fn excerpt(source: &str) -> Excerpt {
    Excerpt { text: source.trim().to_uppercase() }
}
