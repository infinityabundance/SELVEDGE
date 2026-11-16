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

#[cfg(test)]
mod visible {
    use crate::Window;

    #[test]
    fn first_word() {
        assert_eq!(Window::new("alpha beta".to_string()).first(), "alpha");
    }

    #[test]
    fn empty_text_has_no_first_word() {
        assert_eq!(Window::new(String::new()).first(), "");
    }

    #[test]
    fn first_word_survives_replace() {
        let mut window = Window::new("alpha beta".to_string());
        window.replace("gamma delta".to_string());
        assert_eq!(window.first(), "gamma");
    }
}
