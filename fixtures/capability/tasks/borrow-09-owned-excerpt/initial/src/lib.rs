//! An excerpt that must own its text.

pub struct Excerpt {
    pub text: &'static str,
}

impl Excerpt {
    pub fn text(&self) -> &str {
        self.text
    }
}

pub fn excerpt(source: &str) -> Excerpt {
    let owned = source.trim().to_uppercase();
    Excerpt { text: &owned }
}
